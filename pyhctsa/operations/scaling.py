from numpy.typing import ArrayLike
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import qr, solve_triangular
import statsmodels.api as sm
import logging
logger = logging.getLogger('pyhctsa')

from ..toolboxes.Max_Little import fastdfa
from ..utils import _matlab_linspace, make_mat_buffer
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
                         lag: int | None = None, log_inc: bool = True,
                         guard_ratsplit: bool = False, ssr_tol: float = 1e-12) -> dict:
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
    guard_ratsplit : bool, optional
        If True, return NaN for ``ratsplitminerr`` when the full-range fit residual
        (``ssr``) underflows ``ssr_tol``, where the statistic is numerically
        meaningless. Default is False (exact MATLAB parity). Default is `False`.
    ssr_tol : float, optional
        Tolerance for the ``guard_ratsplit`` underflow check. Default is 1e-12.
 
    Returns
    -------
    dict
        Statistics of fitting a linear function to a plot of log(F) as
        a function of log(tau), and for fitting two straight lines to the same data,
        choosing the split point at tau = tau_{split} as that which minimizes the
        combined fitting errors.
 
    """
    N = len(x)
 
    # Compute integrated sequence
    if (lag is None) | (lag == 1):
        # normal cumsum
        y = np.cumsum(x)
    else:
        # if a lag is specified, do a decimation...
        y = np.cumsum(x[::lag])
 
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
            raise NotImplementedError(f"{wtf} not yet implemented.")
        elif wtf == 'iqr':
            raise NotImplementedError(f"{wtf} not yet implemented.")
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
 
    sserr = np.full(num_timescales, np.nan)  # don't choose the end points
    min_points = 6
 
    for i in range(min_points - 1, num_timescales - min_points):
        r1 = slice(0, i + 1)  # first segment: points 0..i  (i+1 points)
        p1 = np.polyfit(logtt[r1], logFF[r1], 1)
 
        r2 = slice(i, num_timescales)  # second segment: points i..end
        p2 = np.polyfit(logtt[r2], logFF[r2], 1)
 
        # Sum of errors from fitting lines to both segments:
        sserr[i] = (np.linalg.norm(np.polyval(p1, logtt[r1]) - logFF[r1]) +
                    np.linalg.norm(np.polyval(p2, logtt[r2]) - logFF[r2]))
 
    break_pt = np.where(sserr == np.nanmin(sserr))[0][0]  # find first occurrence of minimum
    r1 = np.arange(0, break_pt + 1)
    r2 = np.arange(break_pt, num_timescales)
 
    out['prop_r1'] = len(r1) / num_timescales
    out['logtausplit'] = logtt[break_pt]
 
    if guard_ratsplit and (not np.isfinite(out['ssr']) or out['ssr'] < ssr_tol):
        out['ratsplitminerr'] = np.nan
    else:
        out['ratsplitminerr'] = np.nanmin(sserr) / out['ssr']
 
    out['meanssr'] = np.nanmean(sserr)
    out['stdssr'] = np.nanstd(sserr, ddof=1)  # FIX [4]: MATLAB nanstd normalises by N-1
 
    out2 = _robust_linear_fit(logtt, logFF, r1, 'r1_')
    out3 = _robust_linear_fit(logtt, logFF, r2, 'r2_')
 
    out_final = out | out2 | out3
 
    if np.isnan(out_final['r1_alpha']) or np.isnan(out_final['r2_alpha']):
        out_final['alpha_rat'] = np.nan
    else:
        out_final['alpha_rat'] = out_final['r1_alpha'] / out_final['r2_alpha']
 
    return out_final
 
 
def _robust_linear_fit(log_tt: np.ndarray, log_ff: np.ndarray, the_range, field_name):
    """
    Robust linear fit using Tukey's biweight function for M-estimation.
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
 
    x = sm.add_constant(log_tt[the_range])
    rlm = sm.RLM(seg, x, M=sm.robust.norms.TukeyBiweight())
    results = rlm.fit()
    linfit = results.params  # [intercept, slope]
    out = {}
    # Store results in dictionary (Python equivalent of MATLAB struct)
    out[f'{field_name}linfitint'] = linfit[0]  # linear fit intercept
    out[f'{field_name}alpha'] = linfit[1]  # linear fit gradient
    out[f'{field_name}se1'] = results.bse[0]  # standard error in intercept
    out[f'{field_name}se2'] = results.bse[1]  # standard error in slope
    out[f'{field_name}ssr'] = np.mean(results.resid ** 2)  # mean squares residual
    out[f'{field_name}resac1'] = autocorr(results.resid, 1, 'Fourier')[0]  # autocorr at lag 1
    return out


def _colon(base, step, limit):
    base = float(base)
    step = float(step)
    limit = float(limit)
    if step == 0 or (step > 0 and base > limit) or (step < 0 and base < limit):
        return np.zeros(0)

    n = int(np.floor((limit - base) / step + 0.5))
    # Guard against the rounding above overshooting the limit:
    while n > 0 and abs(base + n * step) > abs(limit) + abs(step) * 0.5:
        n -= 1

    i = np.arange(n + 1, dtype=float)
    out = np.empty(n + 1)
    lower = i <= n / 2.0
    out[lower] = base + i[lower] * step
    out[~lower] = limit - (n - i[~lower]) * step
    out[0] = base
    if n > 0:
        out[n] = limit
    return out

def _round(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def _polyfit(x, y, deg):
    """
    Least-squares polynomial fit that tolerates underdetermined systems.

    Deliberately *not* :func:`pyhctsa.toolboxes.matlab.matlab_fit.polyfit`, despite
    computing the same thing for well-determined fits. That one applies the
    Householder reflectors via LAPACK's ``dormqr`` for bit-parity with MATLAB, and
    consequently raises when there are fewer points than coefficients
    (``len(x) < deg + 1``). ``mma`` reaches exactly that case -- a single-point
    gradient at the edge of the scale range -- and needs the rank-deficient basic
    solution (zeros in the pivoted-out slots) that this returns instead.

    The two also disagree in the last bit or two (~1e-13 relative) on
    well-determined fits, because forming ``Q`` explicitly and multiplying by it
    rounds differently from applying the reflectors. Do not merge them without
    re-checking both properties.
    """
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
        [min_scale, max_scale]. Defaults to [10, round(N/40)]. max_scale must be
        a multiple of 5 and is rounded to one if it is not.
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
        scale_range = [10, float(_round(n / 40))]
    min_scale = scale_range[0]
    max_scale = scale_range[1]

    if (max_scale / 5) < min_scale:
        logging.warning(
            "Time-series (N=%u) too short for multiscale multifractal analysis" % n
        )
        return float("nan")
    elif max_scale % 5 != 0:
        max_scale = float(_round(max_scale / 5)) * 5
        logging.warning("adjusted max_scale to %u" % max_scale)

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
    s_list_full = np.unique(_round(_matlab_linspace(min_scale, max_scale, num_increments)))

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
                (fqs_ll[:, 0] == q_val)
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
