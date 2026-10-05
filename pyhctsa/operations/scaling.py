from numpy.typing import ArrayLike
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import qr, solve_triangular
import logging
logger = logging.getLogger('pyhctsa')

from ..toolboxes.Max_Little import fastdfa
from ..robust import bf_theil_sen
from ..toolboxes.matlab.matlab_fit import robustfit
from ..utils import _linspace, make_mat_buffer
from ..operations.correlation import autocorr

def fast_dfa(y: ArrayLike) -> float:
    """
    Measures the scaling exponent of the time series using a fast implementation
    of detrended fluctuation analysis (DFA).

    This is a Python wrapper for Max Little's fastdfa code.
    The original `fastdfa` code is by [1].

    References
    ----------
    .. [1] Max A. Little, http://www.maxlittle.net/software/index.php

    Parameters
    ----------
    y : array-like
        Input time series (1D array), fed straight into the `fastdfa` script.

    Returns
    -------
    float
        Estimated scaling exponent from log-log linear fit of fluctuation vs interval.
    """
    y = np.asarray(y)
    intervals, flucts = fastdfa.fastdfa(y)
    idx = np.argsort(intervals)
    intervals_sorted = intervals[idx]
    flucts_sorted = flucts[idx]

    # Log-log linear fit
    coeffs = np.polyfit(np.log10(intervals_sorted), np.log10(flucts_sorted), 1)
    alpha = coeffs[0]
    
    return alpha

def fluctuation_analysis(x: np.ndarray, q: float | int = 2,
                         wtf: str = 'rsrange', tau_step: int = 1, k: int = 1,
                         lag: int | None = None, log_inc: bool = True) -> dict:
    """
    Implements fluctuation analysis by a variety of methods.
 
    Much of our implementation is based on the well-explained discussion of scaling
    methods [1].
 
    The main difference between algorithms for estimating scaling exponents amount
    to differences in how fluctuations, F, are quantified in time-series segments.
    Many alternatives are implemented in this function.
 
    References
    ----------
    .. [1] "Power spectrum and detrended fluctuation analysis: Application to daily
        temperatures" P. Talkner and R. O. Weber, Phys. Rev. E 62(1) 150 (2000)
    .. [2] D. C. Caccia et al., "Analyzing exact fractal time series: evaluating dispersional
        analysis and rescaled range methods", Physica A 246(3-4) 609 (1997)
    .. [3] J. Alvarez-Ramirez et al., "Using detrended fluctuation analysis for lagged
        correlation analysis of nonstationary signals", Phys. Rev. E 79(5) 057202 (2009)
 
    Parameters
    ----------
    x : array-like
        The input time series.
    q : Union[float, int], optional
        The parameter in the fluctuation function. The default is q = 2 which gives RMS
        fluctuations.
    wtf : str, optional
        What to fluctuate. Options are:
 
        - 'endptdiff': Calculates the differences in end points in each segment
        - 'range': Calculates the range in each segment
        - 'std': Takes the standard deviation in each segment [1]
        - 'iqr': Takes the interquartile range in each segment
        - 'dfa': Removes a polynomial trend of order k in each segment
        - 'rsrange': Returns the range after removing a straight line fit [2]
        - 'rsrangefit': Fits a polynomial of order k and returns the range [2]
 
        Default is ``'rsrange'``.
 
        For 'rsrangefit', an optional timelag can be applied for computing the
        cumulative sum (integrated profile) [3].
    tau_step : int, optional
        Number of tau (locInc true), or increments in tau for linear range. Default is 1.
    k : int, optional
        Polynomial order of detrending (for 'dfa' & 'rsrangefit'). Default is 1.
    lag : int or None, optional
        Optional time-lag, as in Alvarez-Ramirez [3]. Default is `None`.
    log_inc : bool, optional
        Whether to use logarithmic increments in tau (it should be logarithmic). Default is `True`.
 
    Returns
    -------
    dict
        Statistics of fitting a linear function to a plot of log(F) as
        a function of log(tau), and for fitting two straight lines to the same data,
        choosing the split point at tau = tau_{split} as that which minimizes the
        combined fitting errors. The lines are fitted by the Theil-Sen method (the
        median of the slopes between all pairs of points, a robust estimator with a
        closed form: :func:`pyhctsa.robust.bf_theil_sen`).

        - ``linfitint``, ``alpha``, ``se1``, ``se2``, ``ssr``, ``resac1``: the intercept,
          slope (the scaling exponent alpha), standard errors of the intercept and the
          slope (the usual least-squares formulas applied to the residuals of the robust
          fit), mean squared residual, and lag-1 autocorrelation of the residuals, of the
          single line fitted over all timescales; ``r1_*`` and ``r2_*``: the same for the
          first (shorter-timescale) and the second line of the two-line fit.
        - ``logtausplit``, ``prop_r1``: the value of log(tau) at the split between the two
          lines, and the proportion of the timescales covered by the first line.
        - ``splitgain``: the proportional reduction in mean squared error from fitting two
          least-squares lines rather than one over the whole range, one minus the ratio of the
          minimum two-line error to the one-line error (between 0 and 1; NaN if the one-line
          fit is exact).
        - ``meanssr``, ``stdssr``: the mean and the standard deviation of the two-line
          fitting error across the candidate split points.
        - ``alphadiff``: the difference ``r1_alpha - r2_alpha`` between the scaling
          exponents of the two lines.

        NaN if there are too few timescales; the two-line outputs are NaN if the
        timescales are too few to support two lines.
    """
    # Compute integrated sequence
    if (lag is None) | (lag == 1):
        # normal cumsum
        y = np.cumsum(x)
    else:
        # if a lag is specified, do a decimation...
        y = np.cumsum(x[::lag])
    N = len(y)  # length of the integrated series (shorter than x if a lag is used)
 
    # perform scaling over a range of tau, up to a fifth of the time-series length
    if log_inc:
        taur = np.unique(np.floor(np.exp(np.linspace(np.log(5), np.log(np.floor(N / 2)), int(tau_step))) + 0.5))
    else:
        taur = np.arange(5, np.floor(N / 2) + 1, int(tau_step))  # maybe increased??
    ntau = len(taur)  # analyze the time series across this many timescales
    if ntau < 8:  # fewer than 8 points
        logger.warning(f'This time series (N = {N}) is too short to analyze using this fluctuation analysis.')
        out = np.nan
        return out
 
    F = np.zeros(ntau)
    # % 2) Compute the fluctuation function, F
    for i in range(ntau):
        tau = int(taur[i])  # time scale on which to compute fluctuations
        y_buff = make_mat_buffer(y, tau)
        if y_buff.shape[1] > (N // tau):  # zero-padded, remove trailing set of points...
            y_buff = y_buff[:, :-1]
        nn = y_buff.shape[1] * tau
 
        if wtf == 'nothing':
            y_dt = y_buff.reshape(nn, 1, order='F')  # FIX [5]: column-major to match MATLAB
        elif wtf == 'endptdiff':
            y_dt = y_buff[-1, :] - y_buff[0, :]
        elif wtf == "range":
            y_dt = np.max(y_buff, axis=0) - np.min(y_buff, axis=0)
        elif wtf == 'std':
            # standard deviation (N-1 normalization, as MATLAB's std) in each segment
            y_dt = np.std(y_buff, axis=0, ddof=1)
        elif wtf == 'iqr':
            # interquartile range in each segment, using MATLAB's quantile definition
            # (piecewise-linear through the (i-0.5)/n points, i.e. Hazen's method)
            q75, q25 = np.percentile(y_buff, [75, 25], axis=0, method='hazen')
            y_dt = q75 - q25
        elif wtf == 'dfa':
            tt = np.arange(1, tau + 1).reshape(-1, 1)  # faux time range (column vector)
            for j in range(y_buff.shape[1]):
                # fit a polynomial of order k in each subsegment
                p = np.polyfit(tt.flatten(), y_buff[:, j], k)
                # remove the trend, store back in y_buff
                y_buff[:, j] = y_buff[:, j] - np.polyval(p, tt.flatten())
 
            # reshape to a column vector, y_dt (detrended)
            y_dt = y_buff.reshape(nn, 1, order='F')  # FIX [5]: column-major to match MATLAB
        elif wtf == 'rsrange':
            b = y_buff[0, :]
            m = y_buff[-1, :] - b
            y_buff = y_buff - (np.linspace(0, 1, tau).reshape(-1, 1) * m + np.ones((tau, 1)) * b)
            y_dt = np.ptp(y_buff, axis=0)
        elif wtf == 'rsrangefit':
            tt = np.arange(1, tau + 1).reshape(-1, 1)  # faux time range (column vector)
            for j in range(y_buff.shape[1]):
                # fit a polynomial of order k in each subsegment
                p = np.polyfit(tt.flatten(), y_buff[:, j], k)
                # remove the trend, store back in y_buff
                y_buff[:, j] = y_buff[:, j] - np.polyval(p, tt.flatten())
 
            y_dt = np.ptp(y_buff, axis=0)
        else:
            raise ValueError(f"Unknown fluctuation analysis method: {wtf}")
        F[i] = np.mean(y_dt ** q) ** (1 / q)
    # % Smooth unevenly-distributed points in log space:
    if log_inc:
        logtt = np.log(taur)
        logFF = np.log(F)
        num_timescales = ntau
    else:  # need to smooth the unevenly-distributed points (using a spline)
        logtaur = np.log(taur)
        logF = np.log(F)
        num_timescales = 50
        logtt = np.linspace(np.min(logtaur), np.max(logtaur), num_timescales)
        logFF = interp1d(logtaur, logF, kind='cubic')(logtt)
    # % Linear fit the log-log plot: full range
    out = _robust_linear_fit(logtt, logFF, np.arange(0, num_timescales), '')
 
    # minPoints scales with the number of timescales rather than being a fixed constant: a
    # small fixed minPoints lets the search reach breakpoints right at the edge of the
    # domain, where a segment of a handful of points trivially achieves near-zero fit
    # error. The floor of 8 matches _robust_linear_fit's minimum length.
    sserr = np.full(num_timescales, np.nan)  # don't choose the end points
    min_points = max(8, int(_round(0.25 * num_timescales)))
    if num_timescales >= 2 * min_points:
        # Single straight line over the whole range (least squares), for comparison
        p0 = np.polyfit(logtt, logFF, 1)
        ssr1 = np.sum((np.polyval(p0, logtt) - logFF) ** 2) / num_timescales
        for i in range(min_points - 1, num_timescales - min_points):
            r1 = slice(0, i + 1)  # first segment: points 0..i  (i+1 points)
            p1 = np.polyfit(logtt[r1], logFF[r1], 1)

            r2 = slice(i, num_timescales)  # second segment: points i..end
            p2 = np.polyfit(logtt[r2], logFF[r2], 1)

            # Mean squared error pooled across both segments, normalized by the total
            # number of points sampled (num_timescales), so that it is comparable to ssr1:
            e1 = np.polyval(p1, logtt[r1]) - logFF[r1]
            e2 = np.polyval(p2, logtt[r2]) - logFF[r2]
            sserr[i] = (np.sum(e1 ** 2) + np.sum(e2 ** 2)) / num_timescales

    if np.all(np.isnan(sserr)):
        # Too few timescales to fit two distinct linear regimes meaningfully
        r1 = r2 = np.array([], dtype=int)
        out['prop_r1'] = np.nan
        out['logtausplit'] = np.nan
        out['splitgain'] = np.nan
        out['meanssr'] = np.nan
        out['stdssr'] = np.nan
    else:
        # The error curve can be flat, so take the first split whose error is within a tiny
        # relative tolerance of the minimum, rather than testing for exact equality (which
        # rounding errors can decide)
        min_err = np.nanmin(sserr)
        break_pt = np.where(sserr <= min_err * (1 + 1e-9))[0][0]
        r1 = np.arange(0, break_pt + 1)
        r2 = np.arange(break_pt, num_timescales)

        out['prop_r1'] = len(r1) / num_timescales
        out['logtausplit'] = logtt[break_pt]
        # The proportional reduction in squared error from using two lines rather than one
        # (NaN if the single line is exact to rounding error, when the ratio is meaningless)
        if ssr1 > 1e-24:
            out['splitgain'] = max(0.0, 1 - min_err / ssr1)
        else:
            out['splitgain'] = np.nan

        out['meanssr'] = np.nanmean(sserr)
        valid_sserr = sserr[~np.isnan(sserr)]
        # std of the valid errors, normalised by N-1 (as MATLAB), and 0 for a single value
        out['stdssr'] = np.std(valid_sserr, ddof=1) if valid_sserr.size > 1 else 0.0

    out2 = _robust_linear_fit(logtt, logFF, r1, 'r1_')
    out3 = _robust_linear_fit(logtt, logFF, r2, 'r2_')
 
    out_final = out | out2 | out3
 
    # Change in scaling exponent between the two regimes:
    out_final['alphadiff'] = out_final['r1_alpha'] - out_final['r2_alpha']
 
    return out_final
 
 
def _robust_linear_fit(log_tt: np.ndarray, log_ff: np.ndarray, the_range, field_name):
    """
    Robust (Theil-Sen) linear fit statistics on a scaling range.

    The line is the Theil-Sen fit (:func:`pyhctsa.robust.bf_theil_sen`); the standard errors
    are the usual least-squares formulas applied to its residuals. All outputs are NaN for
    fewer than 8 points or an all-NaN segment.
    """
    seg = log_ff[the_range]
    if np.size(the_range) < 8 or np.all(np.isnan(seg)):
        return {
            f'{field_name}linfitint': np.nan,
            f'{field_name}alpha': np.nan,
            f'{field_name}se1': np.nan,
            f'{field_name}se2': np.nan,
            f'{field_name}ssr': np.nan,
            f'{field_name}resac1': np.nan,
        }

    xx = log_tt[the_range]
    slope, intercept = bf_theil_sen(xx, seg)
    resid = seg - (slope * xx + intercept)
    n = len(xx)
    sxx = np.sum((xx - np.mean(xx)) ** 2)
    s2 = np.sum(resid ** 2) / (n - 2)
    out = {}
    out[f'{field_name}linfitint'] = intercept  # linear fit intercept
    out[f'{field_name}alpha'] = slope  # linear fit gradient
    out[f'{field_name}se1'] = np.sqrt(s2 * (1 / n + np.mean(xx) ** 2 / sxx))  # standard error in intercept
    out[f'{field_name}se2'] = np.sqrt(s2 / sxx)  # standard error in gradient
    out[f'{field_name}ssr'] = np.mean(resid ** 2)  # mean squares residual
    out[f'{field_name}resac1'] = autocorr(resid, 1, 'Fourier')[0]  # autocorr at lag 1
    return out


def _colon(base, step, limit):
    base = float(base)
    step = float(step)
    limit = float(limit)
    if step == 0 or (step > 0 and base > limit) or (step < 0 and base < limit):
        return np.zeros(0)

    # Number of steps, floored with a few-ulp tolerance so that a limit reached only up to
    # rounding error is included (as in MATLAB), but a limit that is not on the grid is not:
    ndelta = (limit - base) / step
    n = int(np.floor(ndelta + 4 * np.finfo(float).eps * max(1.0, abs(ndelta))))
    # The last element is the limit itself if the grid reaches it (to within rounding),
    # otherwise the last grid point:
    last = base + n * step
    if abs(last - limit) <= 4 * np.finfo(float).eps * max(abs(base), abs(limit), abs(step)):
        last = limit

    i = np.arange(n + 1, dtype=float)
    out = np.empty(n + 1)
    lower = i <= n / 2.0
    out[lower] = base + i[lower] * step
    out[~lower] = last - (n - i[~lower]) * step
    out[0] = base
    if n > 0:
        out[n] = last
    return out

def _round(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def _polyfit(x, y, deg):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = deg + 1
    V = np.ones((x.size, n))
    for j in range(deg - 1, -1, -1):
        V[:, j] = x * V[:, j + 1]

    Q, R, perm = qr(V, mode="economic", pivoting=True)

    # Rank from the pivoted diagonal, using Matlab's mldivide tolerance:
    diagR = np.abs(np.diag(R))
    if diagR.size == 0 or diagR[0] == 0:
        return np.zeros(n)
    tol = max(V.shape) * np.spacing(diagR[0])
    rank = int(np.sum(diagR > tol))

    p = np.zeros(n)
    p[perm[:rank]] = solve_triangular(
        R[:rank, :rank], (Q.T @ y)[:rank], lower=False
    )
    return p

def _std(x, axis=None):
    x = np.asarray(x, dtype=float)
    n = x.size if axis is None else x.shape[axis]
    if n < 2:
        return np.zeros(()) if axis is None else np.zeros(
            tuple(d for i, d in enumerate(x.shape) if i != axis % x.ndim)
        )
    return np.std(x, axis=axis, ddof=1)

def mma(y: np.ndarray, do_overlap: bool = False, scale_range: None | list = None, 
        q_range: None | list = None) -> dict:
    """Scale-dependent estimates of multifractal scaling in a time series.

    Physionet implementation of multiscale multifractal analysis (MMA). Method was first proposed in [1].
    Original author is Jan Gieraltowski (Warsaw University of Technology, Faculty of Physics). 

    References
    ----------
    .. [1] J. Gieraltowski, J. J. Zebrowski, and R. Baranowski,
        Multiscale multifractal analysis of heart rate variability recordings
        with a large number of occurrences of arrhythmia,
        Phys. Rev. E 85, 021915 (2012).
        http://dx.doi.org/10.1103/PhysRevE.85.021915
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
        Mietus JE, Moody GB, Peng C-K, Stanley HE (2000)
        PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource
        for Complex Physiologic Signals. Circulation 101(23):e215-e220

    Parameters
    ----------
    y : array_like
        Input time series (a 1-D vector).
    do_overlap : bool, optional
        False (default): partition into non-overlapping windows of analysis.
        True: overlapping windows with a step of 1 (much longer calculations).
    scale_range : sequence of 2 numbers, optional
        [min_scale, max_scale]. Defaults to [10, max(100, round(N/40))] (the floor of 100
        keeps short series, N below ~4000, computable). max_scale must be a multiple of 5
        and is rounded to one if it is not. Returns NaN if max_scale exceeds N.
    q_range : sequence of 2 numbers, optional
        [q_min, q_max] multifractal parameter range. Defaults to [-5, 5].

    Returns
    -------
    dict
        Summary statistics.
    """
    y = np.asarray(y, dtype=float).ravel()

    # Time-series length:
    n = y.size

    # --------------------------------------------------------------------------
    # Check inputs:
    # --------------------------------------------------------------------------
    if scale_range is None:
        scale_range = [10, float(max(100, _round(n / 40)))]
    min_scale = scale_range[0]
    max_scale = scale_range[1]

    if max_scale > n:
        logger.warning(
            "Time-series (N=%u) too short for multiscale multifractal analysis "
            "(max_scale=%u exceeds N)" % (n, max_scale)
        )
        return float("nan")
    elif (max_scale / 5) < min_scale:
        logger.warning(
            "Time-series (N=%u) too short for multiscale multifractal analysis" % n
        )
        return float("nan")
    elif max_scale % 5 != 0:
        max_scale = float(_round(max_scale / 5)) * 5
        logger.warning("adjusted max_scale to %u" % max_scale)

    if q_range is None:
        q_range = [-5, 5]
    q_min = q_range[0]
    q_max = q_range[1]

    q_list = _colon(q_min, 0.1, q_max)
    q_list[q_list == 0] = 0.0001

    # --------------------------------------------------------------------------

    prof = np.cumsum(y)
    slength = prof.size

    num_increments = 20
    s_list_full = np.unique(_round(_linspace(min_scale, max_scale, num_increments)))

    # Preallocate fqs (one row per scale x q combination):
    fqs = np.zeros((s_list_full.size * q_list.size, 3))
    row_idx = 0

    for s in s_list_full:
        s = int(s)

        if do_overlap:
            # Sliding windows of length s, step 1:
            num_segments = slength - s + 1
            segments = np.lib.stride_tricks.sliding_window_view(prof, s)
        else:
            num_segments = slength // s
            segments = prof[: num_segments * s].reshape(num_segments, s)

        x_base = _colon(1, 1, s)
        f2_nis = np.zeros(num_segments)

        for ni in range(num_segments):
            seg = segments[ni, :]
            fit = _polyfit(x_base, seg, 2)
            f2_nis[ni] = np.mean((seg - np.polyval(fit, x_base)) ** 2)

        for q in q_list:
            fqs[row_idx, :] = [q, s, np.mean(f2_nis ** (q / 2)) ** (1 / q)]
            row_idx += 1

    fqs_ll = np.column_stack(
        (fqs[:, 0], fqs[:, 1], np.log(fqs[:, 1]), np.log(fqs[:, 2]))
    )

    # --------------------------------------------------------------------------
    # Now compute Hurst exponents as the gradients of F(q) curves
    # --------------------------------------------------------------------------
    if np.sum(s_list_full <= max_scale / 5) >= 10:
        s_list = s_list_full[s_list_full <= max_scale / 5]
    elif min_scale == max_scale / 5:
        # Single-point range: min_scale:0:(max_scale/5) would otherwise silently
        # return empty (a zero-step colon range is always empty in Matlab, even
        # when start == stop), so just take the one point directly:
        s_list = np.array([float(min_scale)])
    else:
        # Sample higher in the scale dimension:
        s_spacing = ((max_scale / 5) - min_scale) / 10
        s_list = _colon(min_scale, s_spacing, max_scale / 5)

    # Coarser sampling of q space:
    q_list = _colon(q_min, 0.5, q_max)
    q_list[q_list == 0] = 0.0001

    hqs = np.zeros((q_list.size, s_list.size))
    for si, s_val in enumerate(s_list):
        for qi, q_val in enumerate(q_list):
            mask = (
                (np.abs(fqs_ll[:, 0] - q_val) < 1e-8)
                & (fqs_ll[:, 1] >= s_val)
                & (fqs_ll[:, 1] <= 5 * s_val)
            )
            fit_temp = fqs_ll[mask, :]
            hqs[qi, si] = _polyfit(fit_temp[:, 2], fit_temp[:, 3], 1)[0]

    # Not completely on top of the algorithm, but for some reason this was
    # recorded as a multiple of 3 in the original algorithm:
    s_list_scaled = s_list * 3

    # --------------------------------------------------------------------------
    # Output statistics:
    # --------------------------------------------------------------------------

    give_me_grad = lambda x_data, y_data : _polyfit(x_data, y_data, 1)[0]

    out = {}

    # Global properties (hqs(:) is column-major in Matlab):
    all_exponents = hqs.ravel(order="F")
    out["meanHurstExponent"] = np.mean(all_exponents)
    out["stdHurstExponent"] = _std(all_exponents)
    out["minHurstExponent"] = np.min(all_exponents)
    out["maxHurstExponent"] = np.max(all_exponents)

    # Changes with scale (mean(hqs,1) is the mean down each column):
    mean_over_q = np.mean(hqs, axis=0)
    out["scaleHurstStd"] = _std(mean_over_q)
    out["scaleHurstTrend"] = give_me_grad(s_list_scaled, mean_over_q)

    # Changes with q:
    mean_over_scale = np.mean(hqs, axis=1)
    out["qHurstStd"] = _std(mean_over_scale)
    out["qHurstTrend"] = give_me_grad(q_list, mean_over_scale)

    # max/min points are where in scale/q space?
    # `find(...,1)` takes the first hit in column-major order:
    qi, si = np.unravel_index(
        np.argmax(hqs.ravel(order="F") == np.max(all_exponents)),
        hqs.shape,
        order="F",
    )
    out["maxHurstQ"] = q_list[qi]
    out["maxHurstScale"] = s_list_scaled[si]
    qi, si = np.unravel_index(
        np.argmax(hqs.ravel(order="F") == np.min(all_exponents)),
        hqs.shape,
        order="F",
    )
    out["minHurstQ"] = q_list[qi]
    out["minHurstScale"] = s_list_scaled[si]

    # Phase transitions: there is some peak or trough somewhere, so the standard
    # deviation across scales or q is inconsistent
    std_s = _std(hqs, axis=0)
    std_q = _std(hqs, axis=1)
    out["stdStdHurstQ"] = _std(std_q)  # large if variance changes a lot with q
    out["stdStdHurstScale"] = _std(std_s)  # large if variance changes a lot with scale

    return out

def higuchi_fd(y: ArrayLike, kmax: int | None = None) -> dict:
    """
    Higuchi's fractal dimension of a time series.

    Estimates the fractal dimension of the time series' waveform (its graph in
    the (index,value) plane) via Higuchi's curve-length method: at each scale
    k, the series is split into k interleaved subsequences, each subsequence's
    normalized curve length is measured, and the k lengths are averaged to
    give L(k). The fractal dimension is (minus) the slope of log(L(k)) against
    log(1/k).

    References
    ----------
    .. [1] T. Higuchi, "Approach to an irregular time series on the basis of the
        fractal theory", Physica D 31(2) 277-283 (1988).

    Parameters
    ----------
    y : array-like
        The input time series.
    kmax : int, optional
        The maximum interleaving scale to include in the fit (default: a
        small, fixed value -- see Notes above for why a large kmax is
        actively harmful, not just unnecessary).

    Returns
    -------
    dict
        The fitted (Higuchi) log-log slope and diagnostics of the linear fit's
        quality.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size

    if kmax is None:
        kmax = min(8, max(5, N // 10))
    kmax = int(kmax)

    # --------------------------------------------------------------------------
    # Compute the curve length L(k) at each scale k = 1,...,kmax
    # --------------------------------------------------------------------------
    log_l = np.full(kmax, np.nan)
    log_invk = np.full(kmax, np.nan)
    for k in range(1, kmax + 1):
        lk = np.full(k, np.nan)
        for m in range(1, k + 1):
            n_max = (N - m) // k
            if n_max < 1:
                continue
            idx = np.arange(m - 1, m + n_max * k, k)
            lk[m - 1] = np.sum(np.abs(np.diff(y[idx]))) * (N - 1) / (n_max * k) / k
        if np.all(np.isnan(lk)):
            continue
        l_bar = np.nanmean(lk)
        if not np.isnan(l_bar) and l_bar > 0:
            log_l[k - 1] = np.log(l_bar)
            log_invk[k - 1] = np.log(1 / k)

    good_k = ~np.isnan(log_l)
    if np.sum(good_k) < 5:
        return float("nan")

    # --------------------------------------------------------------------------
    # Robust linear fit of log(L(k)) against log(1/k)
    # --------------------------------------------------------------------------
    linfit, stats = robustfit(log_invk[good_k], log_l[good_k])
    resid = log_l[good_k] - (linfit[0] + linfit[1] * log_invk[good_k])

    out = {}
    out["HFD"] = linfit[1]  # the Higuchi fractal dimension estimate (log-log slope)
    out["intercept"] = linfit[0]
    out["se_HFD"] = stats["se"][1]  # standard error on the dimension estimate
    out["ssr"] = np.mean(resid ** 2)  # mean squared residual of the linear fit
    out["resac1"] = autocorr(resid, 1, 'Fourier')[0]  # residual autocorrelation

    return out

def mfdfa(y: ArrayLike, scale_range: list | None = None, q_range: list | None = None,
          order: int = 1) -> dict | float:
    """
    Multifractal detrended fluctuation analysis (MFDFA): the multifractal spectrum of a time series.

    Estimates the multifractal singularity spectrum f(alpha) of a time series by the classical
    MFDFA algorithm of Kantelhardt et al. (2002) [1]. The mean-subtracted series is integrated
    (cumulative sum) to a profile, which is divided into non-overlapping segments of length s
    (taken from both the start and the end of the series, to use the whole series when N is
    not a multiple of s). Each segment is detrended by a polynomial of a given order, and the
    q-th order fluctuation function F_q(s) is formed by averaging the segment variances raised
    to the power q/2 (with a log-averaging limit at q = 0). h(q), the slope of log F_q(s)
    against log s, is Legendre-transformed via the mass exponent tau(q) = q h(q) - 1 into the
    singularity spectrum f(alpha), where alpha = d tau / dq and f = q alpha - tau.

    Unlike `fast_dfa` and `fluctuation_analysis` (monofractal, q = 2) and `mma` (which reports
    how the raw h(q) surface varies with scale), this fixes the scaling range and focuses on
    the q axis.

    References
    ----------
    .. [1] J. W. Kantelhardt et al., "Multifractal detrended fluctuation analysis of
        nonstationary time series", Physica A 316(1-4), 87-114 (2002).

    Parameters
    ----------
    y : array-like
        The input time series.
    scale_range : list, optional
        [min_scale, max_scale], the range of segment lengths s used for the fluctuation-function
        fit, as 20 log-spaced values. Default is [16, floor(N/4)].
    q_range : list, optional
        [q_min, q_max], the range of the multifractal order q, sampled in steps of 0.5, with
        q = 0 handled by its log-averaging limit. Default is [-5, 5].
    order : int, optional
        The order of the polynomial used to detrend each segment (1 = linear detrending, MFDFA1).
        Default is 1.

    Returns
    -------
    dict or float
        Dictionary containing:

        - `meanR2`: the mean (across q) of the R^2 of the log-log fits of F_q(s) against s.
        - `h2`: the generalized Hurst exponent h(q) at q = 2 (NaN if q = 2 is not in `q_range`).
        - `alphaMin`, `alphaMax`: the smallest and largest singularity exponents alpha.
        - `alphaWidth`: alphaMax - alphaMin, the degree of multifractality.
        - `fAlphaMax`: the maximum of f(alpha), the height of the spectrum's peak.
        - `alpha0`: the alpha at the peak of the spectrum (the dominant exponent).
        - `spectrumAsymmetry`: (alpha0 - alphaMin) / (alphaMax - alpha0).

        A scalar NaN is returned if the series or scale range cannot support the analysis.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    if scale_range is None or len(scale_range) == 0:
        scale_range = [16, N // 4]
    min_scale, max_scale = scale_range[0], scale_range[1]
    if q_range is None or len(q_range) == 0:
        q_range = [-5, 5]
    q_min, q_max = q_range[0], q_range[1]

    # At least order+3 points per segment, and at least 8 distinct scales spanning the range
    if min_scale < order + 3:
        min_scale = order + 3
    num_scales = 20
    if max_scale > N // 4 or (max_scale / min_scale) < 2 or N // (2 * min_scale) < 4:
        return np.nan
    scales = np.unique(_round(np.exp(_linspace(np.log(min_scale), np.log(max_scale), num_scales)))).astype(int)
    if scales.size < 8:
        return np.nan

    q_true = _colon(q_min, 0.5, q_max)
    q_list = q_true.copy()
    q_zero = np.flatnonzero(q_true == 0)
    q_list[q_list == 0] = 1e-4  # q = 0 handled separately via log-averaging below
    q_zero_idx = q_zero[0] if q_zero.size else None

    # Profile (integrated, mean-subtracted series)
    profile = np.cumsum(y - np.mean(y))

    n_q = q_list.size
    log_fq = np.full((scales.size, n_q), np.nan)
    eps = np.finfo(float).eps
    for si, s in enumerate(scales):
        Ns = N // s
        # Detrending is a fixed linear projection for all segments of length s
        tt = np.arange(1, s + 1, dtype=float)
        V = np.vander(tt / s, order + 1)  # (scaled for conditioning; the projection is unchanged)
        resid_op = np.eye(s) - V @ np.linalg.pinv(V)
        starts = np.concatenate([np.arange(Ns) * s, N - (np.arange(Ns) + 1) * s])
        segs = profile[starts[:, None] + np.arange(s)[None, :]]  # segments from the start, then from the end
        F2 = np.mean((segs @ resid_op.T) ** 2, axis=1)
        F2[F2 < eps] = eps  # floor to avoid log(0) / 0^(negative q)
        for qi, q in enumerate(q_list):
            log_fq[si, qi] = (1 / q) * np.log(np.mean(F2 ** (q / 2)))
        if q_zero_idx is not None:
            # q = 0 limit: F_0(s) = exp{ (1/(4Ns)) * sum( log F^2(s,v) ) }
            log_fq[si, q_zero_idx] = np.mean(np.log(F2)) / 2

    # h(q): slope of log(F_q(s)) vs log(s), for each q
    log_s = np.log(scales.astype(float))
    hq = np.full(n_q, np.nan)
    r2q = np.full(n_q, np.nan)
    for qi in range(n_q):
        good = np.isfinite(log_fq[:, qi])
        if good.sum() < 8:
            continue
        p = np.polyfit(log_s[good], log_fq[good, qi], 1)
        hq[qi] = p[0]
        fitted = np.polyval(p, log_s[good])
        ss_res = np.sum((log_fq[good, qi] - fitted) ** 2)
        ss_tot = np.sum((log_fq[good, qi] - np.mean(log_fq[good, qi])) ** 2)
        if ss_tot > 0:
            r2q[qi] = 1 - ss_res / ss_tot
    if np.any(np.isnan(hq)):
        return np.nan

    # Legendre transform: mass exponent tau(q), singularity spectrum alpha/f(alpha)
    tauq = q_true * hq - 1
    alpha = np.gradient(tauq, q_true)  # d(tau)/dq
    falpha = q_true * alpha - tauq

    out = {'meanR2': np.nanmean(r2q) if not np.all(np.isnan(r2q)) else np.nan}
    idx2 = np.flatnonzero(np.abs(q_true - 2) < 1e-8)
    out['h2'] = hq[idx2[0]] if idx2.size else np.nan
    out['alphaMin'] = np.min(alpha)
    out['alphaMax'] = np.max(alpha)
    out['alphaWidth'] = out['alphaMax'] - out['alphaMin']
    i_peak = int(np.argmax(falpha))
    out['fAlphaMax'] = falpha[i_peak]
    out['alpha0'] = alpha[i_peak]
    left_width = out['alpha0'] - out['alphaMin']
    right_width = out['alphaMax'] - out['alpha0']
    out['spectrumAsymmetry'] = left_width / right_width if right_width > 0 else np.nan
    return out
