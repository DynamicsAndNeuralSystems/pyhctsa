from numpy.typing import ArrayLike
import numpy as np
import logging
logger = logging.getLogger('pyhctsa')

from ..toolboxes.physionet.mma import mma_hurst
from ..robust import bf_theil_sen
from ..fitting import polyfit, robust_linear_fit
from ..utils import _arange_inclusive, dict_output, get_tau, _round_half_away_array, _sample_std
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


def _fluctuations(y: np.ndarray, taus: np.ndarray, wtf: str, q: float, k: int) -> np.ndarray:
    """
    The fluctuation function F(tau) of the profile ``y`` at each window length in ``taus``.

    At each tau the profile is cut into ``N // tau`` non-overlapping windows from the start and
    again from the end (one set when tau divides N), and F is the q-th order average over all
    windows of the per-window fluctuation set by ``wtf``.
    """
    N = len(y)
    F = np.empty(len(taus))
    for i, tau in enumerate(taus):
        tau = int(tau)
        n_win = N // tau
        windows = y[:n_win * tau].reshape(n_win, tau).T
        if N % tau:
            windows = np.hstack([windows, y[N - n_win * tau:].reshape(n_win, tau).T])
        if wtf == 'dfa':
            # remove a polynomial of order k in every window, then the (MF-)DFA fluctuation function
            F[i] = _dfa_fluctuation(_detrend_windows(windows, k), q)
            continue
        if wtf == 'endptdiff':
            f = windows[-1, :] - windows[0, :]
        elif wtf == 'range':
            f = np.ptp(windows, axis=0)
        elif wtf == 'std':
            # sample standard deviation (N-1 normalization) in each window
            f = np.std(windows, axis=0, ddof=1)
        elif wtf == 'iqr':
            # interquartile range in each window, with Hazen's quantile definition
            # (piecewise-linear through the (i-0.5)/n points)
            q75, q25 = np.percentile(windows, [75, 25], axis=0, method='hazen')
            f = q75 - q25
        elif wtf == 'rsrange':
            # range after removing the straight line through the window's end points
            line = np.linspace(0, 1, tau)[:, None] * (windows[-1, :] - windows[0, :]) + windows[0, :]
            f = np.ptp(windows - line, axis=0)
        elif wtf == 'rsrangefit':
            # range after removing a least-squares polynomial of order k in each window
            f = np.ptp(_detrend_windows(windows, k), axis=0)
        else:
            raise ValueError(f"Unknown fluctuation analysis method: {wtf}")
        F[i] = np.mean(np.abs(f) ** q) ** (1 / q)
    return F


def _log_tau_grid(tau_min: float, tau_max: float, n_tau: int) -> np.ndarray:
    """``n_tau`` log-spaced window lengths from tau_min to tau_max, rounded, duplicates dropped."""
    return np.unique(_round_half_away_array(
        np.exp(np.linspace(np.log(tau_min), np.log(tau_max), int(n_tau))))).astype(int)


def _broken_line_fit(log_tau: np.ndarray, log_f: np.ndarray, min_decades: float = 0.3,
                     min_points: int = 6) -> dict:
    """
    Least-squares continuous broken line (two slopes meeting at a breakpoint) through
    (log_tau, log_f).

    Candidate breakpoints are the grid points that leave at least ``min_decades`` decades of tau and
    ``min_points`` points on each side (the breakpoint counts on both sides). At each candidate s the
    model ``log_f = c + a1 * min(u - s, 0) + a2 * max(u - s, 0)`` (u = log tau) is fitted by least
    squares, and the candidate with the smallest residual sum of squares is kept. If that is the
    first or last candidate, no change of slope is resolved within the range and all outputs are NaN.
    """
    out = dict.fromkeys(['r1_alpha', 'r2_alpha', 'alphadiff', 'r1_ssr', 'r2_ssr'], np.nan)
    n = len(log_tau)
    decades = (log_tau - log_tau[0]) / np.log(10)
    span = decades[-1]
    cand = [i for i in range(n) if decades[i] >= min_decades and span - decades[i] >= min_decades
            and i + 1 >= min_points and n - i >= min_points]
    if len(cand) < 3:
        return out
    best = None
    for j, i in enumerate(cand):
        s = log_tau[i]
        design = np.column_stack([np.minimum(log_tau - s, 0), np.maximum(log_tau - s, 0), np.ones(n)])
        coef, *_ = np.linalg.lstsq(design, log_f, rcond=None)
        resid = log_f - design @ coef
        sse = np.sum(resid ** 2)
        if best is None or sse < best[0]:
            best = (sse, j, i, coef, resid)
    _, j, i, coef, resid = best
    if j == 0 or j == len(cand) - 1:
        return out
    out['r1_alpha'], out['r2_alpha'] = coef[0], coef[1]
    out['alphadiff'] = coef[0] - coef[1]
    out['r1_ssr'] = np.mean(resid[:i + 1] ** 2)
    out['r2_ssr'] = np.mean(resid[i:] ** 2)
    return out


@dict_output
def fluctuation_analysis(x: np.ndarray, q: float | int = 2, wtf: str = 'rsrange', n_tau: int = 50,
                         k: int = 1, lag: int | None = None) -> dict:
    """
    Scaling exponents of a fluctuation function, from a single line and from a broken line.

    Fluctuation analysis measures how the fluctuations F(tau) of the profile (the cumulative sum
    of the series) grow with the window length tau. For a scaling process ``F(tau) ~ tau**alpha``,
    with alpha = H for fractional Gaussian noise with Hurst exponent H and alpha = 0.5 for any
    short-range correlated stationary process. The methods differ in how fluctuations are
    quantified in each window [1]_. Much of the implementation follows the discussion of
    scaling methods in [1]_; rescaled-range analysis goes back to Hurst [2]_, and DFA to
    Peng et al. [3]_ [4]_.

    The profile is cut into non-overlapping windows of length tau, from the start and again from
    the end of the series, so that every sample is used [5]_. With f the fluctuation in a window
    (set by ``wtf``), ``F(tau) = mean(|f|**q)**(1/q)`` over the windows; for DFA, F is the q-th
    order fluctuation function of multifractal DFA [5]_. Timescales are ``n_tau`` log-spaced
    values of tau, rounded to integers, with duplicates dropped. No window is shorter than
    ``tau_0 = max(5, 3p)``, where p is the number of parameters removed in each window (k + 1
    for 'dfa' and 'rsrangefit', 2 for 'rsrange', 0 otherwise): a window only a few times longer
    than the detrending fit leaves little fluctuation, which biases F low at small tau [6]_.

    Two fits are made to log F against log tau, each over its own range of timescales:

    - A single line from ``max(tau_0, 4 max(2, p) tau_c, N/50)`` to N/2, where tau_c is the 1/e
      decorrelation time of the analysed series (:func:`pyhctsa.utils.get_tau`, rule 'ac1e').
      Short-range correlation bends F(tau) over timescales up to several correlation times, so a
      slope fitted from small tau mixes the short-range correlation into the long-range exponent
      and can suggest long memory where there is none [7]_ [8]_. Starting the fit beyond the
      correlation time reduces this; the remaining bias falls as the start of the range moves
      further beyond the correlation time, at the cost of a shorter, noisier fit [6]_. The start
      grows with p (8 tau_c for DFA1, 12 tau_c for DFA2, 16 tau_c for DFA3) because the
      deviations from scaling at small tau are stronger for higher-order detrending [6]_. The
      1/e time does not register correlation spread thinly over many lags, as in
      anti-persistent series (for which tau_c = 1), so the start is also at least N/50, which
      keeps the fit to the largest timescales, where the long-range exponent dominates [7]_.
      The slope is the Theil-Sen slope (:func:`pyhctsa.robust.bf_theil_sen`).
    - A continuous broken line (two slopes that meet at a breakpoint) from tau_0 to N/2, which
      describes series whose scaling changes across timescales, such as the short- and long-range
      exponents of heart-rate variability [9]_, or crossovers produced by trends [10]_ [11]_. It is
      fitted by least squares at each candidate breakpoint, keeping the one with the smallest
      residual sum of squares. Candidates leave at least 0.3 decades of tau and 6 timescales on
      each side. If the best breakpoint is the first or last candidate, the range shows no change
      of slope that it can resolve, and the broken-line outputs are NaN.

    References
    ----------
    .. [1] P. Talkner and R. O. Weber, "Power spectrum and detrended fluctuation analysis:
        Application to daily temperatures", Phys. Rev. E 62, 150 (2000).
    .. [2] H. E. Hurst, "Long-term storage capacity of reservoirs", Trans. Am. Soc. Civ. Eng.
        116, 770 (1951); D. C. Caccia et al., "Analyzing exact fractal time series: evaluating
        dispersional analysis and rescaled range methods", Physica A 246, 609 (1997).
    .. [3] C.-K. Peng et al., "Long-range correlations in nucleotide sequences", Nature 356,
        168 (1992).
    .. [4] C.-K. Peng et al., "Mosaic organization of DNA nucleotides", Phys. Rev. E 49, 1685
        (1994).
    .. [5] J. W. Kantelhardt et al., "Multifractal detrended fluctuation analysis of
        nonstationary time series", Physica A 316, 87 (2002).
    .. [6] J. W. Kantelhardt et al., "Detecting long-range correlations with detrended
        fluctuation analysis", Physica A 295, 441 (2001).
    .. [7] D. Maraun, H. W. Rust and J. Timmer, "Tempting long-memory - on the interpretation of
        DFA results", Nonlin. Processes Geophys. 11, 495 (2004).
    .. [8] M. Höll and H. Kantz, "The relationship between the detrendend fluctuation analysis
        and the autocorrelation function of a signal", Eur. Phys. J. B 88 (2015),
        doi:10.1140/epjb/e2015-60721-1.
    .. [9] C.-K. Peng, S. Havlin, H. E. Stanley and A. L. Goldberger, "Quantification of scaling
        exponents and crossover phenomena in nonstationary heartbeat time series", Chaos 5, 82
        (1995).
    .. [10] K. Hu, P. Ch. Ivanov, Z. Chen, P. Carpena and H. E. Stanley, "Effect of trends on
        detrended fluctuation analysis", Phys. Rev. E 64, 011114 (2001).
    .. [11] R. M. Bryce and K. B. Sprague, "Revisiting detrended fluctuation analysis", Sci. Rep.
        2, 315 (2012).
    .. [12] J. Alvarez-Ramirez et al., "Using detrended fluctuation analysis for lagged
        correlation analysis of nonstationary signals", Phys. Rev. E 79, 057202 (2009).
    .. [13] A. A. Anis and E. H. Lloyd, "The expected value of the adjusted rescaled Hurst range
        of independent normal summands", Biometrika 63, 111 (1976).
    .. [14] R. Weron, "Estimating long-range dependence: finite sample properties and confidence
        intervals", Physica A 312, 285 (2002).

    Parameters
    ----------
    x : array-like
        The input time series.
    q : float or int, optional
        The order of the fluctuation function. The default is q = 2, which gives RMS
        fluctuations (standard DFA for ``wtf='dfa'``). For ``wtf='dfa'`` any real q (including
        negative and non-integer values, and q = 0) gives the q-th order fluctuation function of
        MF-DFA [5]_: with F^2(v) the variance of the detrended profile in window v,
        ``F_q = ( mean_v F^2(v)^(q/2) )^(1/q)``, and for q = 0 the logarithmic average
        ``F_0 = exp( mean_v ln F^2(v) / 2 )``; the exponent is then the generalized Hurst
        exponent h(q), NaN for q < 0 if some window has zero variance. For the other methods q
        must be nonzero.
    wtf : str, optional
        What to fluctuate. Options are:

        - 'dfa': detrended fluctuation analysis [3]_ [4]_ [6]_: the mean of the series is
          removed before the profile is formed, and F is the RMS residual after removing a
          least-squares polynomial of order k in each window
        - 'rsrange': the range after removing the straight line through the window's end
          points [2]_
        - 'rsrangefit': the range after removing a least-squares polynomial of order k [2]_
        - 'std': the standard deviation in each window [1]_
        - 'iqr': the interquartile range in each window
        - 'range': the range in each window
        - 'endptdiff': the difference between the end points of each window

        Default is ``'rsrange'``.
    n_tau : int, optional
        Number of log-spaced timescales in each range, before rounding. Default is 50.
    k : int, optional
        Order of the polynomial removed in each window, for 'dfa' and 'rsrangefit'. Default is 1
        (DFA1). k = 0 (only the window mean of the profile is removed) is not DFA proper.
    lag : int or None, optional
        If given, the series is decimated by this factor before the profile is formed, which
        analyses correlation at that lag [12]_. Default is ``None``.

    Returns
    -------
    dict
        - ``alpha``, ``ssr``: the slope (the scaling exponent) and the mean squared residual
          of the single line.
        - ``r1_alpha``, ``r2_alpha``, ``r1_ssr``, ``r2_ssr``: the slopes of the broken line
          over shorter (r1) and longer (r2) timescales, and the mean squared residual on each
          side of the breakpoint.
        - ``alphadiff``: the change of slope, ``r1_alpha - r2_alpha``.

        ``alpha`` and ``ssr`` are NaN if tau_c is undefined, if the single-line range spans less
        than a factor of 1.5 (N/2 less than 1.5 times its start), or if it holds fewer than 8
        distinct timescales. The broken-line outputs are NaN for fewer than 12 distinct
        timescales, or when no interior breakpoint is resolved. Either set is NaN if F is zero at
        some timescale of its range.

    Notes
    -----
    ``alpha`` is NaN for series that decorrelate slowly relative to their length, including
    strongly persistent ones, for which a long-range exponent cannot be told apart from
    short-range correlation. Short-range correlation still raises ``alpha`` a little, more for
    short series. The range-based exponents ('rsrange', 'rsrangefit') are biased upward, because
    in short windows the expected range grows faster than tau**0.5; the bias falls slowly with
    the window length, so these exponents decrease with series length [13]_ [14]_. The methods
    that remove no trend in each window ('std', 'iqr', 'range', 'endptdiff', and 'dfa' with
    k = 0) mostly reflect the short-range correlation. Trends and changes in variance also change F(tau) [10]_
    [11]_. The location of the breakpoint is not returned: at typical series lengths the curve
    bends gradually and the breakpoint is poorly determined.
    """
    out = dict.fromkeys(['alpha', 'ssr', 'r1_alpha', 'r2_alpha', 'alphadiff', 'r1_ssr', 'r2_ssr'], np.nan)
    x = np.asarray(x, dtype=float)
    x_prof = x if lag is None or lag == 1 else x[::int(lag)]
    if len(x_prof) < 4 or not np.all(np.isfinite(x_prof)):
        return np.nan
    tau_c = get_tau(x_prof, 'ac1e')  # decorrelation time of the analysed series
    if wtf == 'dfa':
        # DFA integrates the series after removing its mean, so that the profile does not
        # depend on the input having been z-scored (or otherwise centered) beforehand
        x_prof = x_prof - np.mean(x_prof)
    y = np.cumsum(x_prof)
    N = len(y)
    n_par = {'dfa': int(k) + 1, 'rsrangefit': int(k) + 1, 'rsrange': 2}.get(wtf, 0)
    tau_0 = max(5, 3 * n_par)
    tau_max = np.floor(N / 2)

    # Single line, from beyond the correlation time (further for higher-order detrending) and
    # over no more than the largest 1.4 decades of timescales
    tau_start = max(tau_0, 4 * max(2, n_par) * tau_c, N / 50)
    if np.isfinite(tau_c) and tau_max >= 1.5 * tau_start:
        taus = _log_tau_grid(tau_start, tau_max, n_tau)
        if len(taus) >= 8:
            F = _fluctuations(y, taus, wtf, q, int(k))
            if np.all(F > 0):
                log_tau, log_f = np.log(taus), np.log(F)
                slope, intercept = bf_theil_sen(log_tau, log_f)
                out['alpha'] = slope
                out['ssr'] = np.mean((log_f - (slope * log_tau + intercept)) ** 2)

    # Broken line, over all timescales from tau_0
    if tau_max > tau_0:
        taus = _log_tau_grid(tau_0, tau_max, n_tau)
        if len(taus) >= 12:
            F = _fluctuations(y, taus, wtf, q, int(k))
            if np.all(F > 0):
                out |= _broken_line_fit(np.log(taus), np.log(F))
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
