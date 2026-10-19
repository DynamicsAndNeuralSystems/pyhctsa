import numpy as np

from ..toolboxes.Max_Little.ML_l1pwc import l1pwc, l1pwc_lmax

def stepdetect(y: np.ndarray, method: str = 'l1pwc', params: float | int = 10) -> dict:
    """
    Analysis of discrete steps in a time series.

    Gives information about discrete steps in the signal, using the function
    `l1pwc` from Max A. Little's step detection toolkit.

    References
    ----------
    .. [1] S.J. Kim, K. Koh, S. Boyd and D. Gorinevsky (2009) "ℓ1 Trend Filtering", SIAM Review,
        vol. 51, no. 2, pp. 339-360, doi: 10.1137/070690274.
    .. [2] M.A. Little, Nick S. Jones (2010) "Sparse Bayesian Step-Filtering for
        High-Throughput Analysis of Molecular Machine Dynamics", in 2010 IEEE
        International Conference on Acoustics, Speech and Signal Processing,
        ICASSP 2010 Proceedings.

    Parameters
    ----------
    y : array-like
        The input time series.
    method : str
        The step-detection method. Currently only `l1pwc` is supported.
    params : float or str
        The parameters for the given method used. For `l1pwc` this corresponds to lambda. Default is 10.

    Returns
    -------
    dict
        Statistics on the output of the step-detection method, including the intervals
        between change points, the proportion of constant segments, the reduction in
        variance from removing the piece-wise constants, and stationarity in the
        occurrence of change points. The keys are `E` (objective per sample), `s`,
        `lambdamax` (scaled by 1/sqrt(N)), `nsegments`, `rmsoff`, `rmsoffpstep`,
        `ratn12`, `diffn12` (the absolute difference between the numbers of change points in
        the two halves, as a proportion of the number of segments), `pshort_3`, `meanstepintgt3` (in samples), `cvstepint`
        (std/mean of the step intervals) and `medianstepint` (in samples).
    """

    y = np.asarray(y).ravel()
    N = len(y)

    out = {}
    if method == 'l1pwc':
        lamb = params
        if lamb < 1:  # specify as a proportion of lambdamax
            lamb = l1pwc_lmax(y) * lamb
        steppedy, E, s, lambda_max = l1pwc(y, lamb, False)  # defaults for stoptol/maxiter

        # l1pwc returns x as N x L; collapse to 1-D so diff/subtraction behave
        steppedy = np.asarray(steppedy).ravel()
        E = np.atleast_1d(E)
        s = np.atleast_1d(s)

        # Round to remove numerical fluctuations of order less than 1e-4
        steppedy = np.round(steppedy * 1e4) / 1e4
        out = {
            "E": E[0] / N,        # energy per sample (E sums over the series)
            "s": s[0],          # collapses to 1 for some parameter values
            "lambdamax": lambda_max / np.sqrt(N),  # lambdamax itself grows as ~sqrt(N)
        }
        # Start (0-based index) of each run of constant level; the first run starts at 0.
        chpts = np.concatenate(([0], np.flatnonzero(np.diff(steppedy) != 0) + 1))
    elif method == 'kv':
        raise NotImplementedError('step detection method not yet implemented.')
    else:
        raise ValueError(f'unknown step detection method: {method}')

    num_change_points = len(chpts)

    # Intervals -- of change
    chints = np.diff(np.concatenate([chpts, [N]]))  # lengths of the runs

    # Number of constant segments per sample
    out["nsegments"] = num_change_points / N  # will be 1 if there are no changes

    # How much reduces variance
    out["rmsoff"] = np.std(y, ddof=1) - np.std(y - steppedy, ddof=1)

    # Reduces variance per step
    out["rmsoffpstep"] = out["rmsoff"] / num_change_points

    # Ratio of number of steps in first half of time series to second half
    sum1 = np.sum(chpts < N / 2) - 1  # (exclude the run start that's always sitting at 0)
    sum2 = np.sum(chpts >= N / 2)
    if (sum2 > 0) and (sum1 > 0):
        if sum2 > sum1:
            out["ratn12"] = sum1 / sum2
        else:
            out["ratn12"] = sum2 / sum1
    else:
        out["ratn12"] = 0

    # Difference between number of change points in first and second half of data
    # as a proportion of the number of segments (k change points give k + 1 segments)
    # Maximal, k/(k + 1), when all k change points are in one half of data
    # Minimal (0) when same number of change points in both halves
    out["diffn12"] = abs(sum1 - sum2) / num_change_points

    # Proportion of really short steps:
    out["pshort_3"] = np.sum(chints <= 3) / N

    # Mean interval greater than 3 samples (in samples):
    long_ints = chints[chints > 3]
    out["meanstepintgt3"] = np.mean(long_ints) if long_ints.size else np.nan

    # Coefficient of variation of the step intervals (a single interval has spread 0, by the
    # pyhctsa convention for the standard deviation of one value):
    sd_ints = np.std(chints, ddof=1) if chints.size > 1 else 0.0
    out["cvstepint"] = sd_ints / np.mean(chints)

    # Median step interval:
    out["medianstepint"] = np.median(chints)

    return out

def l1pwc_sweep_lambda(y: np.ndarray, lambdar: np.ndarray) -> dict:
    """
    Dependence of step detection on regularization parameter.
 
    Gives information about discrete steps in the signal across a range of
    regularization parameters lambda, using the function l1pwc from Max Little's
    step detection toolkit.
 
    cf.,
    "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
 
    Parameters
    ----------
    y : array_like
        The input time series.
    lambdar : array_like
        A vector specifying the lambda parameters to use.
 
    Returns
    -------
    dict
        At each iteration, the CP_ML_StepDetect code was run with a given
        lambda, and the number of segments, and reduction in root mean square
        error from removing the piecewise constants was recorded. Outputs
        summarize how these quantities vary with lambda.
    """
    y = np.asarray(y, dtype=float)
    lambdar = np.atleast_1d(np.asarray(lambdar, dtype=float))
 
    Llambdar = len(lambdar)
    nsegs = np.zeros(Llambdar)
    rmserrs = np.zeros(Llambdar)
    rmserrpsegs = np.zeros(Llambdar)
 
    for i in range(Llambdar):
        lam = lambdar[i]
        # Run the (deterministic) step detection algorithm:
        outi = stepdetect(y, method='l1pwc', params=lam)
        nsegs[i] = outi['nsegments']
        rmserrs[i] = outi['rmsoff']
        rmserrpsegs[i] = outi['rmsoffpstep']
 
    # ------------------------------------------------------------------------------
    # Define the FirstUnder function:
    # ------------------------------------------------------------------------------
    # Often finding the first time a certain vector (x) drops under a given
    # threshold (y). This function does it:
 
    def FirstUnder(x, thresh):
        idx = np.flatnonzero(x < thresh)
        return idx[0] if idx.size > 0 else None
 
    def NaNIfEmpty(idx):
        # Returns a NaN if idx is empty, otherwise returns lambdar at idx.
        return np.nan if idx is None else lambdar[idx]
 
    out = {}
 
    # ------------------------------------------------------------------------------
    # (*) Use rmsunderx subfunction to analyze when RMS errors drop under a set of
    # thresholds, *x*, for the first time:
    # ------------------------------------------------------------------------------
    out['rmserrsu05'] = NaNIfEmpty(FirstUnder(rmserrs, 0.5))
    out['rmserrsu02'] = NaNIfEmpty(FirstUnder(rmserrs, 0.2))
    out['rmserrsu01'] = NaNIfEmpty(FirstUnder(rmserrs, 0.1))
 
    # ------------------------------------------------------------------------------
    # (*) Use the nsegsunderx subfunction to analyze when nseg drops under a set of
    # thresholds, *x*, for the first time:
    # nsegunderx = @(x) find(nsegs < x, 1, 'first');
    # ------------------------------------------------------------------------------
    out['nsegsu005'] = NaNIfEmpty(FirstUnder(nsegs, 0.05))
    out['nsegsu001'] = NaNIfEmpty(FirstUnder(nsegs, 0.01))
 
    # Calculate the correlation between the number of segments and rmserrs
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(nsegs, rmserrs)
    out['corrsegerr'] = R[1, 0]
 
    # Maximum rmserrpsegment
    indbest = int(np.argmax(rmserrpsegs))  # where the maximum occurs
    out['bestrmserrpseg'] = rmserrpsegs[indbest]
    out['bestlambda'] = lambdar[indbest]
 
    return out

def _variance_segmentations(y: np.ndarray, K: int, d: int) -> tuple:
    """
    Optimal segmentations of ``y`` into 1..K+1 segments of constant variance.

    Parameters
    ----------
    y : np.ndarray
        1-D signal, length N (centered internally).
    K : int
        Maximum number of change points (>= 1).
    d : int
        Minimum segment length (>= 1).

    Returns
    -------
    J : np.ndarray
        Shape (K + 1,): ``J[k]`` is the minimal total contrast with k change points
        (``+inf`` if infeasible).
    t_est : np.ndarray
        Shape (K + 1, K + 1), int: row k holds the k change points (1-based instants =
        0-based exclusive segment ends) followed by N; zero padded; all zero for
        infeasible k.

    Notes
    -----
    Contrast of a segment of length n: ``n ln(sum(x^2)/n)`` (Gaussian negative
    log-likelihood of a zero-mean segment with its own variance, up to constants),
    Lavielle, M. (1999) "Detection of multiple changes in a sequence of dependent
    variables", Stoch. Proc. Appl. 83(1):79-102. Minimized exactly by dynamic programming
    (Bellman; O(K N^2)). The signal is centered first (a zero-mean contrast), and every
    segment has at least ``d`` samples.
    """
    x = np.asarray(y, dtype=float).ravel()
    N = x.size
    K = int(K)
    d = max(int(d), 1)
    J = np.full(K + 1, np.inf)
    t_est = np.zeros((K + 1, K + 1), dtype=int)
    x = x - np.mean(x)
    cs = np.concatenate([[0.0], np.cumsum(x ** 2)])
    # Fm[m, e]: minimal contrast of the first e samples split into m segments (m = 0..K+1)
    Fm = np.full((K + 2, N + 1), np.inf)
    Fm[0, 0] = 0.0
    back = np.zeros((K + 2, N + 1), dtype=np.int64)
    for e in range(d, N + 1):
        smax = e - d  # the last segment has at least d samples
        if smax < 0:
            continue
        s = np.arange(smax + 1)
        n = (e - s).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = n * np.log(np.maximum(cs[e] - cs[s], 0.0) / n)
            cand = Fm[:K + 1, :smax + 1] + c[None, :]
        cand[np.isnan(cand)] = np.inf
        idx = np.argmin(cand, axis=1)
        Fm[1:, e] = cand[np.arange(K + 1), idx]
        back[1:, e] = idx
    for k in range(K + 1):
        m = k + 1
        J[k] = Fm[m, N]
        if not np.isfinite(J[k]):
            continue
        e = N
        ends = []
        for mm in range(m, 0, -1):
            ends.append(e)
            e = int(back[mm, e])
        ends = ends[::-1]  # segment ends, the last being N
        t_est[k, :k + 1] = ends
    return J, t_est


def _variance_inflation(x: np.ndarray, max_lag: int = 200) -> float:
    """
    Variance inflation of segment-wise mean squares caused by serial dependence and heavy tails.

    For a stationary Gaussian signal with autocorrelation rho(h), the mean square of a long
    segment of n samples has variance ``2 sigma^4 / n * (1 + 2 sum_h rho(h)^2)``. The factor
    ``tau = 1 + 2 sum_{h=1}^{L} (1 - h/(L+1)) rho_hat(h)^2`` (Bartlett taper, L = min(max_lag, N // 4))
    is therefore the factor by which the number of independent observations is smaller than
    N. It is multiplied by ``max(1, (kurtosis - 1) / 2)`` to account for non-Gaussian tails.
    Returns a value >= 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    xc = x - np.mean(x)
    v = np.mean(xc ** 2)
    if N < 8 or not v > 0:
        return 1.0
    L = int(min(max_lag, N // 4))
    f = np.fft.rfft(xc, 2 * N)
    a = np.fft.irfft(f * np.conj(f))[:L + 1]
    rho = a[1:] / a[0]
    h = np.arange(1, L + 1)
    tau = 1.0 + 2.0 * np.sum((1.0 - h / (L + 1.0)) * rho ** 2)
    kurt = np.mean(xc ** 4) / v ** 2
    return float(max(tau, 1.0) * max(1.0, (kurt - 1.0) / 2.0))


def _num_changes_penalized(J: np.ndarray, t_est: np.ndarray, N: int, tau: float = 1.0) -> int:
    """
    Number of change points by a dependence-adjusted modified-BIC rule.

    Chooses ``k`` minimizing ``J[k] / tau + sum_j ln(n_j) + k ln(N)`` over the feasible
    k, where ``J[k]`` is the Gaussian variance contrast (minus twice the maximized
    log-likelihood, up to a constant) of the best segmentation with k change points,
    ``n_j`` are its segment lengths, and ``tau`` >= 1 is the variance inflation of
    :func:`_variance_inflation`. The penalty is the Zhang and Siegmund (2007) mBIC form
    (a ln(n_j) term for each segment's own parameter and ln N for each change-point
    location); dividing the contrast by ``tau`` counts N / tau effective observations.

    Notes
    -----
    Zhang, N. R. and Siegmund, D. O. (2007) "A modified Bayes information criterion with
    applications to the analysis of comparative genomic hybridization data", Biometrics
    63(1):22-32. The use of the penalty with a variance-inflation adjustment, and its
    unit multiplier, were calibrated on simulations.
    """
    best, kbest = np.inf, 0
    for k in range(len(J)):
        if not np.isfinite(J[k]):
            break
        ends = np.concatenate([[0], t_est[k, :k + 1]]).astype(float)
        n = np.diff(ends)
        if np.any(n <= 0):
            continue
        obj = J[k] / tau + np.sum(np.log(n)) + k * np.log(N)
        if obj < best:
            best, kbest = obj, k
    return int(kbest)


def _wvarchg(y: np.ndarray, K: int, d: int) -> int:
    """
    Number of variance change points of a signal.

    For j = 0..K the best segmentation of the (centered) signal into j + 1 segments of
    constant variance, each at least ``d`` samples long, is found by dynamic programming
    on a Gaussian variance contrast (Lavielle 1999); the number of change points is then
    chosen by a modified BIC whose likelihood is divided by an estimate of the variance
    inflation caused by serial dependence and heavy tails (see
    :func:`_num_changes_penalized`).

    Parameters
    ----------
    y : np.ndarray
        1-D signal.
    K : int
        Maximum number of change points.
    d : int
        Minimum delay between change points (samples).

    Returns
    -------
    int
        Number of change points, 0..K (0 if no segmentation with a change point is
        feasible or the input is not finite).

    Notes
    -----
    Segmentation: Lavielle, M. (1999) Stoch. Proc. Appl. 83(1):79-102.
    Selection of the number of change points: modified BIC of Zhang, N. R. and Siegmund,
    D. O. (2007) Biometrics 63(1):22-32, applied to the contrast divided by the
    variance-inflation factor of the signal. Calibrated by simulation so that, for
    stationary noise (white, AR(1), fGn, heavy-tailed) passed through the same wavelet
    pipeline, the probability of reporting at least one change point is about 5%. The
    penalized-contrast rule of Lavielle (2005) is not used for this: it reports spurious
    change points in most stationary series.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 2 or not np.all(np.isfinite(y)) or K < 1:
        return 0
    J, t_est = _variance_segmentations(y, K, d)
    return _num_changes_penalized(J, t_est, y.size, _variance_inflation(y))


def wavelet_var_chg(y: np.ndarray, w_name: str = 'db3', level: int | str = 3,
                    max_n_chpts: int = 5, min_delay: float | int = 0.01) -> float:
    """
    Number of variance change points found from a wavelet detail signal.

    The series is decomposed with a mother wavelet to a given level, and the detail signal at
    that level is reconstructed. The 2% of its values with the largest magnitude are replaced
    by the mean of the detail signal, so that single large events do not dominate. A variance
    change-point estimator (dynamic programming on a Gaussian variance contrast, Lavielle 1999)
    then finds up to `max_n_chpts` points at least `min_delay` samples apart. The number to keep
    is chosen with a modified BIC (Zhang and Siegmund 2007) on the contrast divided by an estimate
    of the variance inflation due to the (band-limited) detail signal's serial dependence, so that
    stationary series return 0 about 95% of the time. The output is that number.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with, e.g., 'db3', 'sym2'.
        Default is 'db3'.
    level : int or 'max', optional
        The level of wavelet decomposition; 'max' uses the maximum level for the series
        length. Default is 3.
    max_n_chpts : int, optional
        The maximum number of change points. Default is 5.
    min_delay : float or int, optional
        The minimum delay between consecutive change points, in samples; a value between 0
        and 1 is taken as a proportion of the time-series length, e.g., 0.02 ensures that
        change points are separated by at least 2% of the length. Default is 0.01.

    Returns
    -------
    float
        The optimal number of change points, from 0 to `max_n_chpts`, or NaN if `level` is
        too large for the wavelet on this series.
    """
    import pywt
    from .wavelet import _wavedec, _wrcoef

    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    max_level = pywt.dwt_max_level(N, w_name)
    if level == 'max':
        level = max_level
    if min_delay is None:
        min_delay = 0.01
    if 0 < min_delay < 1:
        min_delay = int(np.ceil(min_delay * N))
    min_delay = int(min_delay)
    if max_level < level or level < 1:
        return np.nan

    # Detail signal at the chosen level
    det = _wrcoef(_wavedec(y, w_name, level=level), w_name, level, N)

    # Replace the 2% of the greatest (absolute) values by the mean
    x = np.sort(np.abs(det))
    idx = int(np.fix(x.size * 0.98))
    if idx < 1:
        return np.nan
    v2p100 = x[idx - 1]
    det = det.copy()
    det[np.abs(det) > v2p100] = np.mean(det)

    return float(_wvarchg(det, max_n_chpts, min_delay))
