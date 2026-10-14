from numpy.typing import ArrayLike
import numpy as np
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger('pyhctsa')

from ..toolboxes.physionet.mma import mma_hurst
from ..robust import bf_theil_sen
from ..fitting import polyfit, robust_linear_fit
from ..utils import _arange_inclusive, dict_output, frame_signal, _round_half_away_array, _sample_std
from ..operations.correlation import autocorr

def _detrend_windows(y_buff: np.ndarray, k: int) -> np.ndarray:
    """
    Remove a least-squares polynomial of order ``k`` from every column (window) of ``y_buff``.

    ``y_buff`` has shape (window length, number of windows). The fit in every window uses the
    same time axis, so the detrending is the projection of each window onto the orthogonal
    complement of the span of the Vandermonde basis, ``y - Q Q' y``, with ``Q`` an orthonormal
    basis of that span (from a QR factorization of the Vandermonde matrix on a time axis
    rescaled to [-1, 1] for conditioning). One matrix product handles all windows at once;
    the result equals window-by-window ``numpy.polyfit`` / ``numpy.polyval`` detrending.
    """
    tau = y_buff.shape[0]
    t = np.linspace(-1.0, 1.0, tau) if tau > 1 else np.zeros(1)
    basis = np.vander(t, int(k) + 1, increasing=True)
    qmat, _ = np.linalg.qr(basis)
    return y_buff - qmat @ (qmat.T @ y_buff)


def _dfa_fluctuation(resid: np.ndarray, q: float) -> float:
    """
    The q-th order fluctuation function of MF-DFA [Kantelhardt et al. 2002] from the
    detrended windows ``resid`` (window length x number of windows).

    With F^2(v) the variance (mean square of the residuals) of window v,
    ``F_q = ( mean_v F^2(v)^(q/2) )^(1/q)``, and for q = 0 the logarithmic average
    ``F_0 = exp( mean_v ln F^2(v) / 2 )``. Any real q is allowed. For q <= 0 the result is NaN
    if a window has exactly zero variance (the average is then undefined).
    """
    var = np.mean(resid ** 2, axis=0)
    if q == 2:
        return float(np.sqrt(np.mean(var)))
    if q <= 0 and np.any(var <= 0):
        return np.nan
    if q == 0:
        return float(np.exp(0.5 * np.mean(np.log(var))))
    return float(np.mean(var ** (q / 2)) ** (1 / q))


@dict_output
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
    .. [4] J. W. Kantelhardt et al., "Multifractal detrended fluctuation analysis of
        nonstationary time series", Physica A 316, 87 (2002)
    .. [5] C.-K. Peng et al., "Mosaic organization of DNA nucleotides", Phys. Rev. E 49,
        1685 (1994)
    .. [6] J. W. Kantelhardt et al., "Detecting long-range correlations with detrended
        fluctuation analysis", Physica A 295, 441 (2001)
 
    Parameters
    ----------
    x : array-like
        The input time series.
    q : Union[float, int], optional
        The order of the fluctuation function. The default is q = 2, which gives RMS
        fluctuations (standard DFA for ``wtf='dfa'``). For ``wtf='dfa'`` any real q (including
        negative and non-integer values, and q = 0) gives the q-th order fluctuation function of
        multifractal DFA (MF-DFA) [4]: with F^2(v) the variance of the detrended profile in
        window v, ``F_q(s) = ( mean_v F^2(v)^(q/2) )^(1/q)``, and for q = 0 the logarithmic
        average ``F_0(s) = exp( mean_v ln F^2(v) / 2 )``. The scaling exponent returned is then
        the generalized Hurst exponent h(q); for q < 0 it is NaN if some window has zero
        variance. For the other ``wtf`` options, ``F = mean(f ** q) ** (1/q)`` is taken
        over the per-window quantity f (q must then be nonzero).
    wtf : str, optional
        What to fluctuate. Options are:
 
        - 'endptdiff': Calculates the differences in end points in each segment
        - 'range': Calculates the range in each segment
        - 'std': Takes the standard deviation in each segment [1]
        - 'iqr': Takes the interquartile range in each segment
        - 'dfa': Detrended fluctuation analysis [5, 6]: the mean of the series is removed, the
          profile (cumulative sum) is formed, and a polynomial trend of order k is removed
          in each segment
        - 'rsrange': Returns the range after removing a straight line fit [2]
        - 'rsrangefit': Fits a polynomial of order k and returns the range [2]
 
        Default is ``'rsrange'``.
 
        For 'rsrangefit', an optional timelag can be applied for computing the
        cumulative sum (integrated profile) [3].
    tau_step : int, optional
        Number of tau (locInc true), or increments in tau for linear range. Default is 1.
    k : int, optional
        Polynomial order of detrending (for 'dfa' & 'rsrangefit'). Default is 1 (DFA1).
        Note that k = 0 (only the segment mean of the profile is removed, no trend) is not
        DFA proper: its exponent saturates near alpha = 1 for strongly persistent series and
        is biased low (by about 0.05-0.1 at N of a few hundred) for weakly persistent ones.
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
    # Compute integrated sequence (the profile)
    x = np.asarray(x, dtype=float)
    if (lag is None) | (lag == 1):
        x_prof = x
    else:
        # if a lag is specified, do a decimation...
        x_prof = x[::lag]
    if wtf == 'dfa':
        # DFA integrates the series after removing its mean, so that the profile does not
        # depend on the input having been z-scored (or otherwise centered) beforehand
        x_prof = x_prof - np.mean(x_prof)
    y = np.cumsum(x_prof)
    N = len(y)  # length of the integrated series (shorter than x if a lag is used)
 
    # perform scaling over a range of tau, up to a fifth of the time-series length
    if log_inc:
        taur = np.unique(_round_half_away_array(np.exp(np.linspace(np.log(5), np.log(np.floor(N / 2)), int(tau_step)))))
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
        y_buff = frame_signal(y, tau)
        if y_buff.shape[1] > (N // tau):  # zero-padded, remove trailing set of points...
            y_buff = y_buff[:, :-1]
        nn = y_buff.shape[1] * tau
 
        if wtf == 'nothing':
            y_dt = y_buff.reshape(nn, 1, order='F')  # column-major: segment by segment
        elif wtf == 'endptdiff':
            y_dt = y_buff[-1, :] - y_buff[0, :]
        elif wtf == "range":
            y_dt = np.max(y_buff, axis=0) - np.min(y_buff, axis=0)
        elif wtf == 'std':
            # sample standard deviation (N-1 normalization) in each segment
            y_dt = np.std(y_buff, axis=0, ddof=1)
        elif wtf == 'iqr':
            # interquartile range in each segment, with Hazen's quantile definition
            # (piecewise-linear through the (i-0.5)/n points)
            q75, q25 = np.percentile(y_buff, [75, 25], axis=0, method='hazen')
            y_dt = q75 - q25
        elif wtf == 'dfa':
            # remove a polynomial of order k in every window, then the (MF-)DFA fluctuation function
            F[i] = _dfa_fluctuation(_detrend_windows(y_buff, k), q)
            continue
        elif wtf == 'rsrange':
            b = y_buff[0, :]
            m = y_buff[-1, :] - b
            y_buff = y_buff - (np.linspace(0, 1, tau).reshape(-1, 1) * m + np.ones((tau, 1)) * b)
            y_dt = np.ptp(y_buff, axis=0)
        elif wtf == 'rsrangefit':
            # fit and remove a polynomial of order k in each subsegment
            y_dt = np.ptp(_detrend_windows(y_buff, k), axis=0)
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
    min_points = max(8, int(_round_half_away_array(0.25 * num_timescales)))
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
        # std of the valid errors, normalised by N-1, and 0 for a single value (pyhctsa convention)
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
    out[f'{field_name}resac1'] = autocorr(resid, 1, 'Fourier')  # autocorr at lag 1
    return out


@dict_output
def mma(y: np.ndarray, do_overlap: bool = False, scale_range: None | list = None, 
        q_range: None | list = None) -> dict:
    """Scale-dependent estimates of multifractal scaling in a time series.

    Physionet implementation of multiscale multifractal analysis (MMA). Method was first proposed in [1].
    Original author is Jan Gieraltowski (Warsaw University of Technology, Faculty of Physics). 

    Derived from PhysioNet's ``MMA.m``, Copyright (C) 2014 Jan Gierałtowski (GNU GPL, version 2 or
    later), via hctsa's ``SC_MMA`` (modified by B. D. Fulcher, 2015). The Hurst-exponent
    computation is in :func:`pyhctsa.toolboxes.physionet.mma.mma_hurst`; this function summarizes it.

    A segment that the quadratic detrending fits exactly (to within rounding: residual variance at
    most 1e-20 of its mean square, e.g. a run of constant values) has no fluctuation: its
    q-th moment is infinite for q < 0 (and the q = 0 geometric mean, for which q = 1e-4 stands
    in, is zero), so it is left out of the average there; it contributes 0 for q > 0. The output
    is NaN if no segment fluctuates at some scale.

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
    res = mma_hurst(y, do_overlap, scale_range, q_range)
    if res is None:
        return float("nan")
    hqs, s_list_scaled, q_list = res
    if np.any(np.isnan(hqs)):  # a scale at which no segment fluctuates (q < 0 undefined)
        return float("nan")

    # --------------------------------------------------------------------------
    # Output statistics:
    # --------------------------------------------------------------------------

    give_me_grad = lambda x_data, y_data : polyfit(x_data, y_data, 1)[0]

    out = {}

    # Global properties (flattened in column-major order):
    all_exponents = hqs.ravel(order="F")
    out["meanHurstExponent"] = np.mean(all_exponents)
    out["stdHurstExponent"] = _sample_std(all_exponents)
    out["minHurstExponent"] = np.min(all_exponents)
    out["maxHurstExponent"] = np.max(all_exponents)

    # Changes with scale (mean(hqs,1) is the mean down each column):
    mean_over_q = np.mean(hqs, axis=0)
    out["scaleHurstStd"] = _sample_std(mean_over_q)
    out["scaleHurstTrend"] = give_me_grad(s_list_scaled, mean_over_q)

    # Changes with q:
    mean_over_scale = np.mean(hqs, axis=1)
    out["qHurstStd"] = _sample_std(mean_over_scale)
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
    std_s = _sample_std(hqs, axis=0)
    std_q = _sample_std(hqs, axis=1)
    out["stdStdHurstQ"] = _sample_std(std_q)  # large if variance changes a lot with q
    out["stdStdHurstScale"] = _sample_std(std_s)  # large if variance changes a lot with scale

    return out

@dict_output
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
    linfit, stats = robust_linear_fit(log_invk[good_k], log_l[good_k])
    resid = log_l[good_k] - (linfit[0] + linfit[1] * log_invk[good_k])

    out = {}
    out["HFD"] = linfit[1]  # the Higuchi fractal dimension estimate (log-log slope)
    out["intercept"] = linfit[0]
    out["se_HFD"] = stats["se"][1]  # standard error on the dimension estimate
    out["ssr"] = np.mean(resid ** 2)  # mean squared residual of the linear fit
    out["resac1"] = autocorr(resid, 1, 'Fourier')  # residual autocorrelation

    return out

@dict_output
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

    Unlike `fluctuation_analysis` (monofractal, q = 2) and `mma` (which reports
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
    scales = np.unique(_round_half_away_array(np.exp(np.linspace(np.log(min_scale), np.log(max_scale), num_scales)))).astype(int)
    if scales.size < 8:
        return np.nan

    q_true = _arange_inclusive(q_min, 0.5, q_max)
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
        tt = np.arange(s, dtype=float)
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
