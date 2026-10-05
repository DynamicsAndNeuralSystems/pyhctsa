from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.special import gammaln
from scipy.stats import spearmanr
from scipy.signal import correlate

from ..operations.model_fit import residual_analysis
from ..operations.correlation import first_crossing, first_min, autocorr
from ..toolboxes.Tisean_3_0_1 import tisean as _tisean
from ..utils import _ml_rng, get_tau, matlab_quantile, theiler_window, time_delay_embed

def zero_one_test(y, num_c=20, max_n=10000):
    """Modified 0-1 test for chaos.

    Parameters
    ----------
    y : array_like
        Input scalar time series.
    num_c : int, default=20
        Number of frequencies c over which the statistics are computed.
    max_n : int or "full", default=10000
        Maximum number of samples to analyze. If "full", use the complete
        time series.

    Returns
    -------
    out : dict
        K : float
            Median correlation statistic across frequencies.
        Kstd : float
            Standard deviation of K across frequencies.
        D : float
            Median diffusion rate across frequencies.
        Dstd : float
            Standard deviation of D across frequencies.

    Notes
    -----
    K ~ 0 indicates bounded/regular dynamics, whereas K ~ 1 indicates
    unbounded dynamics. Unbounded dynamics may be deterministic chaos
    or stochastic noise; the 0-1 test alone does not distinguish them.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    if isinstance(max_n, str):
        if max_n != "full":
            raise ValueError("maxN must be an integer or 'full'.")

        if n > 50_000:
            logger.warning(
                f"Time series ({n} samples) exceeds 50000 with "
                "maxN='full'; computation may be slow."
            )

    else:
        max_n = int(max_n)

        if n > max_n:
            logger.warning(
                f"Time series ({n} samples) exceeds maxN={max_n}; "
                f"analyzing the first {max_n} samples."
            )
            y = y[:max_n]
            n = max_n

    if n < 200:
        logger.warning(
            f"Time series (N={n}) too short for a meaningful "
            "0-1 test (need >= 200)."
        )

        return {
            "K": np.nan,
            "Kstd": np.nan,
            "D": np.nan,
            "Dstd": np.nan,
        }

    # ------------------------------------------------------------------
    # Set up test
    # ------------------------------------------------------------------
    tcut = n // 10

    # Number of displacement pairs used for every lag.
    n_pairs = n - tcut

    # j is a mathematical index in cos(j*c), not a Python array index.
    j = np.arange(1, n + 1, dtype=float)

    lags_idx = np.arange(1, tcut + 1)
    lags = lags_idx.astype(float)

    mean_y = np.mean(y)

    cs = np.linspace(
        np.pi / 5,
        4 * np.pi / 5,
        num_c,
    )

    k_c = np.empty(num_c)
    d_c = np.empty(num_c)

    # Quantities required for both Pearson correlation and OLS slope.
    lags_centered = lags - np.mean(lags)
    ss_lags = np.dot(lags_centered, lags_centered)

    # ------------------------------------------------------------------
    # 0-1 test
    # ------------------------------------------------------------------
    for k, c in enumerate(cs):

        phase = j * c

        # Translation variables.
        p = np.cumsum(y * np.cos(phase))
        q = np.cumsum(y * np.sin(phase))

        energy_cumsum = np.empty(n + 1)
        energy_cumsum[0] = 0.0

        np.cumsum(
            p * p + q * q,
            out=energy_cumsum[1:],
        )

        # Energy of the common starting interval i=0,...,n_pairs-1.
        base_energy = energy_cumsum[n_pairs]

        # Energy of the lagged intervals
        # i=n,...,n+n_pairs-1 for all n simultaneously.
        shifted_energy = (
            energy_cumsum[lags_idx + n_pairs]
            - energy_cumsum[lags_idx]
        )

        p_cross = correlate(
            p,
            p[:n_pairs],
            mode="valid",
            method="fft",
        )

        q_cross = correlate(
            q,
            q[:n_pairs],
            mode="valid",
            method="fft",
        )

        # Element 0 corresponds to zero lag; we need lags 1:tcut.
        cross = p_cross[1:] + q_cross[1:]

        m_c = (
            base_energy
            + shifted_energy
            - 2.0 * cross
        ) / n_pairs

        v_osc = (
            mean_y**2
            * (1.0 - np.cos(lags * c))
            / (1.0 - np.cos(c))
        )

        m_c_mod = m_c - v_osc

        m_c_centered = m_c_mod - np.mean(m_c_mod)

        covariance = np.dot(
            lags_centered,
            m_c_centered,
        )

        ss_m_c = np.dot(
            m_c_centered,
            m_c_centered,
        )

        if ss_m_c == 0:
            k_c[k] = np.nan
        else:
            k_c[k] = covariance / np.sqrt(
                ss_lags * ss_m_c
            )

        d_c[k] = covariance / ss_lags

    return {
        "K": np.median(k_c),
        "Kstd": np.std(k_c, ddof=1),
        "D": np.median(d_c),
        "Dstd": np.std(d_c, ddof=1),
    }

def _first_fn(p, threshold, over_or_under='under'):
    """Position (counting from one) of the first element of ``p`` on the given
    side of ``threshold``, or ``len(p) + 1`` if there is none."""
    if over_or_under == 'under':
        indices = np.where(p < threshold)[0]
    elif over_or_under == 'over':
        indices = np.where(p > threshold)[0]
    else:
        raise ValueError(f'Unknown setting: {over_or_under}')

    return indices[0] + 1 if len(indices) > 0 else len(p) + 1

def _normed_single_curve_length(x: np.ndarray, lag: int, nrmdegree: int) -> np.ndarray:
    # helper function for _normed_single_curve_length_windowed and nsamdf
    # crossed curve length of x at each delay 0, 1, ..., lag
    rx1 = np.zeros(lag+1)
    for delay in range(1, lag+1):  # rx1[0] is the norm of a zero vector, i.e., zero
        rx1[delay] = np.linalg.norm(x[:-delay] - x[delay:], ord=nrmdegree)
    return rx1

def _normed_single_curve_length_windowed(x: ArrayLike, win_len: int,
                                         shift_len: int, lag: int,
                                         nrmdegree: int) -> np.ndarray:
    # helper function for nsamdf
    # mean curve length over sliding windows; shiftlen is winlen - overlaplen
    x = np.asarray(x, dtype=float)
    m = int(np.floor((len(x) - win_len)/shift_len) + 1)

    r_sum = np.zeros(lag+1)
    for start in range(0, m*shift_len, shift_len):
        r_sum += _normed_single_curve_length(x[start:start+win_len], lag, nrmdegree)

    return r_sum / m

def _ms_embed(z, v, w):
    # helper function for nlpe
    z = np.asarray(z, dtype=float).squeeze()
    if z.ndim != 1:
        raise ValueError("MS_embed requires a 1-D time series as first argument.")

    n = z.size
    if v is None:
        lags = np.array([0, 1, 2])
    elif w is not None:
        lags = np.arange(0, w * int(v), w) # length v
    else:
        lags = np.asarray(v, dtype=int).ravel()

    lags = np.sort(lags)
    dim  = len(lags)
    if n <= lags[-1]:
        logger.warning("Vector is too small to be embedded with the given lags.")
        return np.full((dim, 1), np.nan), None

    w_win = lags[-1] - lags[0] # window width  (renamed to avoid shadowing arg)
    m = n - w_win # number of embeddable points
    t = np.arange(m) + lags[-1]    # embed times (0-indexed: t[i] = i + lags[-1])

    x = z[t[np.newaxis, :] - lags[:, np.newaxis]] # (dim, m)

    # Split into past (x) and future (y) components
    neg_mask = lags < 0
    if np.any(neg_mask):
        y = x[neg_mask, :]
        x = x[~neg_mask, :]
    else:
        y = None

    return x, y

def _ms_nlpe(y: ArrayLike, de: int, tau: int, theiler_win: int = 0) -> float:
    # helper function for nlpe (hctsa's MS_nlpe, with its Theiler-window argument)
    y = np.asarray(y, dtype=float)

    # Case 1: y is already a matrix (pre-embedded)
    if y.ndim == 2 and min(y.shape) > 1:
        x = y[:, :-1]
        y = y[0, 1:]

    # Case 2: de is a vector of embedding indices
    elif de is not None and np.asarray(de).size > 1:
        de = np.asarray(de)
        v = de[de > 0]
        x, y = _ms_embed(z=y, v=(v - 1), w=None)
        y = y.squeeze()  # (1, m) -> (m,)

    # Case 3: scalar de and tau
    else:
        lags = np.concatenate(([-1], np.arange(0, de * tau, tau)))
        x, y = _ms_embed(z=y, v=lags, w=None)
        y = y.squeeze()  # (1, m) -> (m,)

    if x is None or x.size == 0:
        logger.warning("Error embedding the time series.")
        return np.nan

    de_dim, n = x.shape

    # Nearest neighbour of each point under the squared Euclidean distance.
    # The full n-by-n distance matrix is never held in memory at once: rows are
    # processed in blocks of at most ~32MB, which is both kinder on memory for
    # long time series and friendlier to cache.
    block = max(1, 4_000_000 // n)
    near = np.empty(n, dtype=np.intp)  # nearest neighbour index per point
    dd = np.empty((min(block, n), n))
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = dd[:stop - start]
        rows.fill(0.0)
        for i in range(de_dim):
            diff = x[i, np.newaxis, :] - x[i, start:stop, np.newaxis]
            rows += diff ** 2
        # MS_nlpe adds 1 to every off-diagonal squared distance (its way of excluding the
        # point itself): it only matters for near-ties, which it rounds to exact ties,
        # so the nearest neighbor (first index) of quantized series matches the original.
        rows += 1.0
        rows[np.arange(stop - start), np.arange(start, stop)] = np.inf # exclude self
        if theiler_win > 0:  # also exclude neighbors within a Theiler window in time
            lo = np.maximum(np.arange(start, stop) - theiler_win, 0)
            hi = np.minimum(np.arange(start, stop) + theiler_win + 1, n)
            cols = np.arange(n)[np.newaxis, :]
            rows[(cols >= lo[:, np.newaxis]) & (cols < hi[:, np.newaxis])] = np.inf
        near[start:stop] = np.argmin(rows, axis=1)

    e = y[near] - y  # now y is (m,) so y[near] works correctly

    return e

def nsamdf(x: ArrayLike, tau_mult: Union[int, float] = 2, win_len_rel: Union[int, float] = 10,
           shift_len_rel: Union[float, int] = 0.5, degree: int = 7) -> dict:
    """
    Computes the nonlinearity measure L through nsAMDF
    (nonlinear average magnitude difference function), developed by Ozkurt et al. [1].

    The lag range and window of the nsAMDF are set from the time series' own
    correlation time: with ``tau`` the first zero-crossing of the autocorrelation
    function, the maximum lag is ``ceil(tau_mult*tau)`` and the window length
    ``win_len_rel`` times that. The normalized curves of the nsAMDF for
    ``p = 2`` and ``p = degree`` are compared by their root-mean-square difference.

    This function was authored by Tolga Esat Ozkurt, 2020. (tolgaozkurt@gmail.com).
    Edits by Ben Fulcher for incorporating into hctsa and Joshua Moore for incorporating into pyhctsa.

    References
    ----------
    .. [1] Ozkurt et al. (2020), "Identification of nonlinear features in cortical and
        subcortical signals of Parkinson's Disease patients via a novel efficient measure", NeuroImage.
    
    Parameters
    ----------
    x : array-like
        Input time series.
    tau_mult : float or int
        The maximum lag, as a multiple of the first zero-crossing of the
        autocorrelation function. Default is 2.
    win_len_rel : float or int
        The window length, as a multiple of the maximum lag (a long enough segment is
        important to estimate the nonlinearity). Default is 10.
    shift_len_rel : float or int
        The shift between successive windows, as a proportion of the window length
        (window length minus overlap). Default is 0.5.
    degree : int
        The chosen degree p should ideally be large enough to capture the
        highest order of nonlinearity within the data. Default is 7.
    
    Returns
    -------
    dict
        ``L``, the nsAMDF nonlinearity measure: the root-mean-square difference
        between the nsAMDF curves for p = 2 and p = ``degree``, each normalized by
        its maximum (so it is invariant to the scale of the series). Returns NaN if the
        autocorrelation function has no zero-crossing, or the window is longer than
        the time series.
    """
    x = np.asarray(x, dtype=float).ravel()
    tau = first_crossing(x, 'ac', 0, 'discrete')
    if np.isnan(tau):
        logger.warning('No autocorrelation zero-crossing to set the nsAMDF lag range')
        return np.nan
    max_lag = int(np.ceil(tau_mult * tau))
    window_length = win_len_rel * max_lag
    if window_length > len(x):
        logger.warning('Time series too short relative to its correlation time')
        return np.nan
    window_length = int(window_length)
    shift_length = max(1, int(np.floor(shift_len_rel * window_length)))

    s2 = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length,
                                              lag=max_lag, nrmdegree=2)
    sd = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length,
                                              lag=max_lag, nrmdegree=degree)
    with np.errstate(divide='ignore', invalid='ignore'):
        s2n = s2 / np.max(s2)
        sdn = sd / np.max(sd)

    return {'L': np.sqrt(np.mean((s2n - sdn)**2))}

def nlpe(y: ArrayLike, de: Union[int, str, list] = 3, tau: Union[int, str] = 1,
         max_n: Union[int, str] = 5000,
         theiler_win: Union[int, float, list, tuple] = ('ac', 1)) -> dict:
    """
    Normalized drop-one-out constant interpolation nonlinear prediction error.

    Computes the nlpe for a time-delay embedded time series using Michael Small's
    code, nlpe [1]. Neighbors within a Theiler window in time are excluded from
    the search for the nearest neighbor of each embedded point.

    Modifications by Joshua B. Moore for incorporating into pyhctsa.

    References
    ----------
    .. [1] M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
        Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
        Vol. 52 (2005)
    
    Parameters
    ----------
    y : array-like
        Input time series (should be z-scored).
    de : int, optional
        The embedding dimension. Default is 3. (hctsa's ``'fnn'`` option, which
        sets it by TISEAN's false nearest neighbors, is not yet available.)
    tau : int or str, optional
        The time-delay: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of the
        autocorrelation function, ``'ac1e'`` the (floored) first 1/e crossing of
        the autocorrelation function, and ``'mi'`` the smaller of the first
        minimum of the (Kraskov) automutual information and the 1/e time.
        Default is 1.
    max_n : int or 'full', optional
        The maximum length of the time series on which to compute the nlpe (the
        first ``max_n`` samples are used), or ``'full'`` to use the whole series
        (memory use grows quadratically with length). Default is 5000.
    theiler_win : int, float, or ``['ac', k]``, optional
        The Theiler window (see :func:`pyhctsa.utils.theiler_window`), computed
        on the (cropped) series: a number of samples, or ``['ac', k]`` for ``k``
        times the first zero-crossing of the autocorrelation function. Default
        is ``['ac', 1]``.
    
    Returns
    -------
    dict
        Measures of the mean error of the nonlinear predictor (``msqerr``), and the
        ``'full'`` residual analysis (:func:`pyhctsa.operations.model_fit.residual_analysis`:
        correlation, Gaussianity, etc. of the residuals, and ``taurat`` relative to the
        series used, i.e. after any ``max_n`` crop).
        Returns NaN if the delay or Theiler window cannot be set, or the series
        is too short.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Time series cannot be embedded (could not get the time delay)')
        return np.nan
    tau = int(tau)

    #% nlpe can cause memory pains for long time series
    #% Let's do this dirty cheat
    if isinstance(max_n, str):
        if max_n != 'full':
            raise ValueError(f"max_n must be an integer or 'full', got '{max_n}'")
    elif n > max_n:
        # crop the time series to the first max_n samples
        y = y[:int(max_n)]
        logger.info(f"Michael Small's nlpe code is only being evaluated on the first {max_n} (/{n}) samples.")
        n = int(max_n)

    if n < 20: # short time series cause problems
        logger.warning(f'Time series (N = {len(y)}) is too short.')
        return np.nan

    if isinstance(de, str):
        if de == 'fnn':
            raise NotImplementedError(
                "nlpe(de='fnn') needs a port of TISEAN's false_nearest (hctsa's NL_FNN), "
                "which is not yet available in pyhctsa; pass an integer embedding dimension.")
        raise ValueError(f"Invalid embedding dimension '{de}'")

    theiler_win = theiler_window(y, theiler_win, n)
    if np.isnan(theiler_win):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan

    # run the nonlinear prediction error code
    res = _ms_nlpe(y, de, tau, int(theiler_win))
    if np.isscalar(res) and np.isnan(res):
        # a scalar nan has been returned instead of expected array
        return np.nan

    # compute outputs
    out = {}
    out['msqerr'] = np.mean(res**2)
    res = residual_analysis(res, y, 'full')
    # combine with residual analysis results
    out = out | res

    return out

def delay_time(y: ArrayLike, max_delay: Union[int, float, list, tuple] = ('ac', 10),
               past: Union[int, float, list, tuple] = ('ac', 1),
               random_seed: Union[int, None] = 0) -> dict:
    """
    Optimal delay time using the method of Parlitz and Wichard.

    For a set of randomly chosen reference points, the nearest neighbors in value
    (one just below and one just above the reference value) are found, and the
    mean absolute difference between the following values of the neighbor and
    reference is tracked as a function of the delay.

    Parameters
    ----------
    y : array-like
        Input time series.
    max_delay : int, float, or ``['ac', k]``, optional
        Maximum value of the delay to consider. ``['ac', k]`` sets it to ``k`` times
        the first zero-crossing of the autocorrelation function (see
        :func:`pyhctsa.utils.theiler_window`). Values in (0, 1) are interpreted as
        a proportion of the time-series length (legacy). Delays below 10 are raised
        to 10, and a delay too long for the series (at least ``N/2``) is shortened to
        ``ceil(N/2) - 1`` (NaN only if that is below 10). Default is ``['ac', 10]``.
    past : int, float, or ``['ac', k]``, optional
        Number of time-correlated points to discard (samples) when searching
        for value-neighbors, i.e., the Theiler window: a number of samples, or
        ``['ac', k]`` for ``k`` times the autocorrelation time (``['ac1e', k]`` is
        also accepted). Default is ``['ac', 1]``.
    random_seed : int or None, optional
        Seed for the Mersenne Twister used to draw the reference points.

    Returns
    -------
    dict
        The first three values of ``tau``, the differences between them, and
        the mean, standard deviation, minimum and maximum of ``tau``. Returns
        NaN if an autocorrelation-based ``max_delay`` or ``past`` cannot be set (the ACF
        never crosses zero), if the series is too short for a delay of 10, or if no
        reference point clears the Theiler window.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    if isinstance(max_delay, (list, tuple)):  # a multiple of the autocorrelation time
        max_delay = theiler_window(y, max_delay, N)
        if np.isnan(max_delay):
            logger.warning('No autocorrelation zero-crossing to set the maximum delay')
            return np.nan
    elif 0 < max_delay < 1:
        max_delay = int(theiler_window(None, max_delay, N))  # a proportion of the time-series length
    max_delay = int(max_delay)

    if max_delay < 10:
        max_delay = 10
        logger.warning('Max delay set to its minimum: delaytime = 10')
    if max_delay >= N/2:
        # Too long for the series: shorten to fit (keeping the minimum of 10)
        max_delay = int(np.ceil(N/2)) - 1
        if max_delay < 10:
            logger.warning(f'Time series of length {N} too short for a maximum delay of 10')
            return np.nan

    past = theiler_window(y, past, N)
    if np.isnan(past):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan

    iterations = 64
    max_attempts = 1000
    length = N - max_delay
    # index[r] is the position in the time series of the (r+1)th smallest value
    index = np.argsort(y[:length], kind='stable')

    rng = np.random.RandomState() if random_seed is None else _ml_rng(random_seed)

    err = np.zeros(max_delay + 1)
    for _ in range(iterations):
        # Redraw until the reference point has a value-neighbor on both sides
        # that clears the Theiler window (as hctsa, give up with NaN after
        # max_attempts draws, which is data dependent).
        for _ in range(max_attempts):
            ref = int(np.ceil(rng.random_sample()*length))  # a random value-rank (from one)
            actual = index[ref-1]
            below, above = index[:ref-1], index[ref:]
            pre_candidates = below[np.abs(below - actual) > past]
            post_candidates = above[np.abs(above - actual) > past]
            if pre_candidates.size > 0 and post_candidates.size > 0:
                pre = pre_candidates[-1]   # nearest-in-value candidate below ref
                post = post_candidates[0]  # nearest-in-value candidate above ref
                break
        else:
            logger.warning('No reference point with value-neighbors outside a '
                           f'Theiler window of {past} samples was found in '
                           f'{max_attempts} draws')
            return np.nan
        y_ref = y[actual:actual + max_delay + 1]
        err += np.abs(y[pre:pre + max_delay + 1] - y_ref)
        err += np.abs(y[post:post + max_delay + 1] - y_ref)

    tau = err/iterations

    out = {}
    out['tau1'] = tau[0]
    out['tau2'] = tau[1]
    out['tau3'] = tau[2]
    out['difftau12'] = tau[1] - tau[0]
    out['difftau13'] = tau[2] - tau[0]
    out['meantau'] = np.mean(tau)
    out['stdtau'] = np.std(tau, ddof=1)
    out['mintau'] = np.min(tau)
    out['maxtau'] = np.max(tau)

    return out

def embed_pca(y: ArrayLike, tau: Union[str, int] = 'ac', m: int = 3) -> dict:
    """
    Reconstructs the time series as a time-delay embedding, and performs Principal
    Components Analysis on the result.
    This technique is known as singular spectrum analysis [1].

    References
    ----------
    .. [1] "Extracting qualitative dynamics from experimental data"
        D. S. Broomhead and G. P. King, Physica D 20(2-3) 217 (1986)
    
    Parameters
    ----------
    y : array-like
        Input time series.
    tau: str or int
        The time-delay: an integer, or a rule understood by :func:`pyhctsa.utils.get_tau`.
        ``'ac'`` is the first zero-crossing of the autocorrelation function,
        ``'ac1e'`` the (floored) first 1/e crossing of the autocorrelation
        function, and ``'mi'`` the smaller of the first minimum of the (Kraskov)
        automutual information and the 1/e time. Default is ``'ac'``.
    m : int
        The embedding dimension. Default is 3.
    
    Returns 
    -------
    dict 
        Various statistics summarizing the obtained eigenvalue distribution.

    """
    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Could not get time delay (time series too short?)')
        return np.nan
    try:
        y_embed = time_delay_embed(y, m, int(tau))
    except ValueError as e:  # embedding failed (time series too short)
        logger.warning(str(e))
        return np.nan
    if y_embed.shape[0] - 1 < m or m < 2:
        logger.warning(f'Not enough embedding vectors ({y_embed.shape[0]}) for a rank-{m} PCA')
        return np.nan
    # do the PCA
    pca = PCA().fit(y_embed)
    #proportion of variance explained
    perc = pca.explained_variance_/np.sum(pca.explained_variance_)
    out = {}
    for i in range(m):
        out[f'perc_{i+1}'] = perc[i]
    #%% Get statistics of the eigenvalue distribution
    out['std'] = np.std(perc, ddof=1)
    out['range'] = np.ptp(perc)
    out['min'] = np.min(perc)
    out['max'] = np.max(perc)
    out['top2'] = np.sum(perc[:2]) # variance expl. in top two eigendirections

    #% Number of eigenvalues you need to reconstruct X%
    csperc = np.cumsum(perc)
    out['nto50'] = _first_fn(csperc, 0.5, 'over')
    out['nto60'] = _first_fn(csperc, 0.6, 'over')
    out['nto70'] = _first_fn(csperc, 0.7, 'over')
    out['nto80'] = _first_fn(csperc, 0.8, 'over')
    out['nto90'] = _first_fn(csperc, 0.9, 'over')

    #% When individual % variance explained goes below X for the first time:
    out['fb05'] = _first_fn(perc, 0.5, 'under')
    out['fb02'] = _first_fn(perc, 0.2, 'under')
    out['fb01'] = _first_fn(perc, 0.1, 'under')
    out['fb001'] = _first_fn(perc, 0.01, 'under')

    return out

def local_density(y: ArrayLike, nnr: int = 3,
                  past: Union[int, float, list, tuple] = ('ac', 1),
                  tau: Union[str, int] = 'ac', m: Union[str, int] = 2) -> dict:
    """
    How densely the delay-embedded trajectory is sampled around each of its
    points, and how that density changes along the orbit.

    Computes a k-nearest-neighbor estimate of the local probability density at
    each point of the time-delay embedding: ``density(i) = (k/Neff) / (V_m *
    r_k(i)^m)``, where ``r_k(i)`` is the distance from point i to its k-th
    (``k = nnr``) nearest neighbor (excluding temporally-close points within a
    Theiler window of ``past`` samples), ``m`` is the embedding dimension,
    ``V_m = pi^(m/2) / Gamma(m/2 + 1)`` is the volume of the unit m-ball, and
    ``Neff = N_embed - 2*past - 1`` is the number of points that can be
    neighbors. The estimate is computed in units of the series' standard
    deviation, and its logarithm is analyzed::

        log density(i) = log(k/Neff) - log(V_m) - m*log(r_k(i)/std(y)),

    which makes the outputs independent of the units of ``y`` and, for a
    stationary process, of the number of points. Working with the log density
    keeps the statistics well behaved (the density itself is heavy-tailed). To
    avoid infinite densities when there are repeated values (zero neighbor
    distance, as in quantized or held series), distances are smoothed as
    ``sqrt(r^2 + (0.01*median(r[r > 0]))^2)``. (hctsa previously used TSTOOL's
    ``localdensity`` and then ``1/r^m``; neither is used any more.)

    Parameters
    ----------
    y : array-like
        Input time series.
    nnr : int, optional
        Number of nearest neighbors to compute. Default is 3.
    past : int, float, or ``['ac', k]``, optional
        The Theiler window of time-correlated points to discard (see
        :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k`` times
        the first zero-crossing of the autocorrelation function (also
        ``['ac1e', k]``), or a number of samples. Default is ``['ac', 1]``.
    tau : str or int, optional
        The time-delay of the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of
        the autocorrelation function, ``'ac1e'`` the (floored) first 1/e
        crossing of the autocorrelation function, and ``'mi'`` the smaller of
        the first minimum of the (Kraskov) automutual information and the 1/e
        time. Default is ``'ac'``.
    m : int, optional
        The embedding dimension. Default is 2. (hctsa's ``'fnn'`` option, which
        sets it by TISEAN's false nearest neighbors, is not yet available, and
        raises ``NotImplementedError``.)

    Returns
    -------
    dict
        Statistics on the log local density series (output names retain 'den'),
        in the time order of the embedded points: the minimum, maximum,
        interquartile range, range, standard deviation, mean and median
        (``minden`` ... ``medianden``), the autocorrelation at lags 1 to 5
        (``ac1den`` ... ``ac5den``), and the correlation lengths of the density
        sequence, ``tauacden`` (first zero-crossing of the autocorrelation
        function) and ``taumigaussden`` (first minimum of the Gaussian automutual
        information function, a monotonic function of the autocorrelation).
        Returns NaN if the Theiler window or delay cannot be set, the series is
        too short, or all neighbor distances are zero.
    """
    y = np.asarray(y, dtype=float).ravel()

    past = theiler_window(y, past, len(y))
    if np.isnan(past):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    past = int(past)

    if isinstance(m, str):
        if m == 'fnn':
            raise NotImplementedError(
                "local_density(m='fnn') needs a port of TISEAN's false_nearest (hctsa's "
                "NL_FNN), which is not yet available in pyhctsa; pass an integer embedding "
                "dimension.")
        raise ValueError(f"Invalid embedding dimension '{m}'")
    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Could not get the time delay (time series too short?)')
        return np.nan
    try:
        y_embed = time_delay_embed(y, m, int(tau))
    except ValueError as e:  # embedding failed (time series too short)
        logger.warning(str(e))
        return np.nan
    n_embed, m = y_embed.shape

    if n_embed <= nnr + 2*past:
        logger.warning('Time series too short to do a local density estimate with these parameters.')
        return np.nan

    k_fetch = min(n_embed - 1, nnr + 2*past + 5)
    nbrs = NearestNeighbors(n_neighbors=k_fetch+1).fit(y_embed)
    dist, idx = nbrs.kneighbors(y_embed)

    valid = np.abs(idx - np.arange(n_embed)[:, None]) > past
    # only the nnr-th smallest valid distance is needed, so partition rather than sort
    valid_dists = np.partition(np.where(valid, dist, np.inf), nnr-1, axis=1)
    dk = valid_dists[:, nnr-1]  # distance to the nnr-th neighbor outside the Theiler window

    # Fall back to a full pairwise search wherever the over-fetch wasn't enough:
    for i in np.flatnonzero(valid.sum(axis=1) < nnr):
        all_dists = np.linalg.norm(y_embed - y_embed[i], axis=1)
        all_dists[np.abs(np.arange(n_embed) - i) <= past] = np.inf
        dk[i] = np.sort(all_dists)[nnr-1]

    if not np.any(dk > 0):  # all neighbor distances are zero (e.g., a constant series)
        return np.nan

    # Smooth the distances so that repeated values (zero distances) give a finite density:
    d = np.sqrt(dk**2 + (0.01*np.median(dk[dk > 0]))**2)

    # Log of the k-NN density estimate, with distances in units of the series' SD:
    neff = n_embed - 2*past - 1  # number of points that can be neighbors of a given point
    locden = (np.log(nnr/neff) - ((m/2)*np.log(np.pi) - gammaln(m/2 + 1))
              - m*np.log(d/np.std(y, ddof=1)))

    out = {}
    out['minden'] = np.min(locden)
    out['maxden'] = np.max(locden)
    out['iqrden'] = (np.percentile(locden, 75, method='hazen')
                     - np.percentile(locden, 25, method='hazen'))
    out['rangeden'] = np.ptp(locden)
    out['stdden'] = np.std(locden, ddof=1)
    out['meanden'] = np.mean(locden)
    out['medianden'] = np.median(locden)

    for i in range(1, 6):
        out[f'ac{i}den'] = autocorr(locden, i, 'Fourier')[0]

    # Estimates of correlation length:
    # first zero-crossing of the autocorrelation function:
    out['tauacden'] = first_crossing(locden, 'ac', 0, 'continuous')
    # first minimum of the (Gaussian) automutual information function:
    out['taumigaussden'] = first_min(locden, 'mi-gaussian')

    return out


class _D2DataError(ValueError):
    """A data-dependent failure of :func:`tisean_d2`, for which hctsa returns NaN."""


def _c2g_overflows(c2: list) -> bool:
    """
    Whether TISEAN's ``c2g`` binary would write ``Infinity``/``NaN`` for these correlation sums.

    ``c2g.f`` keeps the log-lengths and log-correlation sums in single precision, and the
    prefactor ``f = exp((e(k+1)c(k) - e(k)c(k+1))/(e(k+1) - e(k)))`` of its piecewise
    power-law interpolation overflows (above about 88.7 in the exponent) for some series,
    which contaminates the whole block with ``Infinity``/``NaN``. hctsa cannot read that
    output and returns NaN (``NL_d2``: 'Inf/NaN-contaminated output'). The vendored
    :func:`pyhctsa.toolboxes.Tisean_3_0_1.tisean.c2g` works in double precision and does
    not overflow, so this reproduces the single-precision condition from the same
    buffers (including the leftover slot of a block cut short by a zero).
    """
    meps = 1000
    e_buf = np.zeros(meps, dtype=np.float32)
    c_buf = np.zeros(meps, dtype=np.float32)
    for block in c2:
        me = 0
        for ee, cc in block:
            me += 1
            if cc <= 0.0:
                break
            e_buf[me - 1] = np.log(np.float32(ee))
            c_buf[me - 1] = np.log(np.float32(cc))
        if me == 0:
            continue
        order = _tisean._tisean_argsort(e_buf[:me])
        e_buf[:me] = e_buf[:me][order]
        c_buf[:me] = c_buf[:me][order]
        e, c = e_buf[:me], c_buf[:me]
        with np.errstate(all='ignore'):
            de = e[1:] - e[:-1]
            f = np.exp((e[1:] * c[:-1] - e[:-1] * c[1:]) / de)
        # intervals of zero width are skipped by c2g.f
        if np.any(~np.isfinite(f[de != 0])):
            return True
    return False


def _argmin_first_colmajor(m: np.ndarray):
    """First index of the minimum in column-major order; NaNs ignored."""
    flat = m.ravel(order='F')
    if flat.size == 0 or np.all(np.isnan(flat)):
        return None, None, np.nan
    k = int(np.nanargmin(flat))
    i, j = np.unravel_index(k, m.shape, order='F')
    return int(i), int(j), float(flat[k])


def _scaling_range_endpoints(l: int) -> tuple:
    """Candidate 1-based start/end points: start in the first half, end in the second."""
    stptr = np.arange(1, int(np.floor(l / 2)))
    endptr = np.arange(int(np.ceil(l / 2)) + 1, l + 1)
    return stptr, endptr


def _best_flat_range(v: np.ndarray, gamma: float, stptr: np.ndarray,
                     endptr: np.ndarray) -> tuple:
    """Scaling range over which ``v`` is most nearly constant.

    Rescales ``v`` to [0,1] so the comparison is independent of its range, then
    scores each candidate range by the spread of ``v`` across it, less a bonus
    (``gamma``) per additional point spanned. Returns the winning
    ``(start index, end index, score)`` into ``stptr``/``endptr``.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        vnorm = (v - v.min()) / (v.max() - v.min())
    mybad = np.empty((stptr.size, endptr.size))
    for i, s in enumerate(stptr):
        for j, e in enumerate(endptr):
            mybad[i, j] = np.std(vnorm[s - 1:e], ddof=1)
    mybad -= gamma * (endptr[np.newaxis, :] - stptr[:, np.newaxis] + 1)
    return _argmin_first_colmajor(mybad)


def _sub_takens(dat: list, eup: float) -> np.ndarray:
    # Takens' estimator at the cutoff length scale eup, one value per embedding
    # dimension (NaN where the scan never reached eup).
    out = np.full(len(dat), np.nan)
    for i, d in enumerate(dat):
        if d.size == 0:
            continue
        idx = np.flatnonzero(d[:, 0] > eup)
        if idx.size > 0:
            out[i] = d[idx[0], 1]
    return out


def _sub_findmmin(ds: ArrayLike) -> dict:
    # Estimated dimensions for m = 1, ..., maxm: find where they stabilise, by
    # dropping points from the start to minimise variance over what remains.
    ds = np.asarray(ds, dtype=float).ravel()
    l = ds.size
    gamma = 0.1  # regularizer, chosen ad hoc; rewards a longer constant region
    dsraw = ds
    dsmin = np.min(ds)
    with np.errstate(invalid='ignore', divide='ignore'):
        dsn = (ds - dsmin) / (np.max(ds) - dsmin)  # rescale to [0,1] so weights are consistent

    out = {'ri1': None, 'goodness': np.nan, 'stabled': np.nan, 'linrmserr': np.nan}
    if l < 2:
        return out

    mybad = np.array([np.std(dsn[i - 1:], ddof=1) - gamma * (l - i + 1)
                      for i in range(1, l)])
    if not np.all(np.isnan(mybad)):
        a = int(np.nanargmin(mybad))  # 0-based
        out['ri1'] = a + 1
        out['goodness'] = float(mybad[a])
        out['stabled'] = float(np.mean(dsraw[a:]))

    # How linear is it?
    if np.all(np.isfinite(dsn)):
        x = np.arange(1, l + 1)
        pfit = np.polyval(np.polyfit(x, dsn, 1), x)
        out['linrmserr'] = float(np.sqrt(np.mean((dsn - pfit) ** 2)))
    return out


def _findscalingr(x: np.ndarray) -> dict:
    # Find a constant region shared by every row of x (i.e. all embedding
    # dimensions must exhibit scaling over the same range of length scales).
    x = np.atleast_2d(x)
    l = x.shape[1]
    gamma = 0.002  # regularization parameter selected empirically
    stptr, endptr = _scaling_range_endpoints(l)

    out = {'ri1': None, 'ri2': None, 'goodness': np.nan,
           'dimest': np.nan, 'dimstd': np.nan}
    if stptr.size == 0 or endptr.size == 0:
        return out

    # mean squared deviation from the middle value (the exponent estimate) over
    # each candidate range, less a bonus for a longer range
    mybad = np.empty((stptr.size, endptr.size))
    for i, s in enumerate(stptr):
        for j, e in enumerate(endptr):
            mybad[i, j] = x[:, s - 1:e].var()
    mybad -= gamma * (endptr[np.newaxis, :] - stptr[:, np.newaxis] + 1)

    a, b, best = _argmin_first_colmajor(mybad)
    if a is None:
        return out
    ri1, ri2 = int(stptr[a]), int(endptr[b])
    sub = x[:, ri1 - 1:ri2]
    out['ri1'] = ri1
    out['ri2'] = ri2
    out['goodness'] = best
    out['dimest'] = float(sub.mean())
    out['dimstd'] = (0.0 if 1 in sub.shape
                     else float(np.std(sub.mean(axis=0), ddof=1)))
    return out


def _findscalingr_ind(x: np.ndarray) -> np.ndarray:
    # As _findscalingr, but each embedding dimension gets its own scaling range.
    # Returns rows of [start, end, goodness, dimension].
    x = np.atleast_2d(x)
    ndim, l = x.shape
    gamma = 1E-3  # regularization parameter selected 'empirically'
    stptr, endptr = _scaling_range_endpoints(l)
    if stptr.size == 0 or endptr.size == 0:
        raise _D2DataError('time series is too short to contain a scaling range')

    results = np.full((ndim, 4), np.nan)
    for c in range(ndim):
        v = x[c, :]
        a, b, best = _best_flat_range(v, gamma, stptr, endptr)
        if a is None:
            # every candidate range scored NaN (e.g. a constant curve): no scaling range
            # (MATLAB's assignment from an empty index fails here)
            raise _D2DataError('no scaling range in the TISEAN d2 output')
        results[c] = [stptr[a], endptr[b], best, np.mean(v[stptr[a] - 1:endptr[b]])]
    return results


def _sub_celltomat(blocks: list, column: int) -> tuple:
    # Stack one column of each per-dimension block into a matrix. Higher
    # embedding dimensions may not reach as far down in length scale, so first
    # restrict every block to the span they all share.
    blocks = [np.asarray(b, dtype=float) for b in blocks]
    if any(b.size == 0 for b in blocks):
        raise _D2DataError('no data returned by TISEAN for at least one dimension')

    mini = max(b[:, 0].min() for b in blocks)
    maxi = min(b[:, 0].max() for b in blocks)
    blocks = [b[(b[:, 0] >= mini) & (b[:, 0] <= maxi), :] for b in blocks]

    thevector = blocks[0][:, 0]
    ee = thevector.size

    if any(b.shape[0] != ee for b in blocks):
        # TISEAN sometimes repeats an 'x' value -- drop the duplicates
        blocks = [b if b.shape[0] == ee
                  else b[np.unique(b[:, 0], return_index=True)[1], :]
                  for b in blocks]

    thematrix = np.zeros((len(blocks), ee))
    for i, b in enumerate(blocks):
        if b.shape[0] != ee:
            break
        thematrix[i, :] = b[:, column - 1]
    return thevector, thematrix


def _sub_getslopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Best-fitting local gradient of each row of Y, over the scaling range that
    # minimises the (regularised) spread of those gradients.
    dx = np.log10(x[1]) - np.log10(x[0])
    ndim = Y.shape[0]
    gamma = 2E-3  # regularizer, chosen 'empirically' (i.e. ad hoc)
    l = Y.shape[1] - 1
    stptr, endptr = _scaling_range_endpoints(l)
    if stptr.size == 0 or endptr.size == 0:
        return None

    results = np.full((ndim, 4), np.nan)
    for c in range(ndim):
        v = np.diff(Y[c, :]) / dx  # vector of local gradients
        a, b, best = _best_flat_range(v, gamma, stptr, endptr)
        if a is None:
            return None  # no scaling range (MATLAB's assignment from an empty index fails)
        results[c] = [stptr[a], endptr[b], best, np.mean(v[stptr[a] - 1:endptr[b]])]
    return results


def _sub_doesflatten(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Look for a region of zero gradient flanked by regions of negative
    # gradient -- the signature of deterministic chaos in h2. Returns, per
    # embedding dimension, how flat the best intermediate region is and the
    # mean of Y across it.
    dx = np.log10(x[1]) - np.log10(x[0])
    ndim = Y.shape[0]
    l = Y.shape[1] - 1
    stptr = np.arange(5, int(np.floor(l / 2)))
    endptr = np.arange(int(np.ceil(l / 2)) + 1, l - 5 + 1)
    if stptr.size == 0 or endptr.size == 0:
        return None

    results = np.full((ndim, 2), np.nan)
    for c in range(ndim):
        v = np.diff(Y[c, :]) / dx
        with np.errstate(invalid='ignore', divide='ignore'):
            vnorm = np.abs(v) / np.abs(v).max()
        # the two outside regions each depend on a single endpoint, so they only
        # need computing once per candidate rather than once per (start, end) pair
        left = np.array([abs(np.mean(vnorm[0:s])) for s in stptr])
        right = np.array([abs(np.mean(vnorm[e - 1:])) for e in endptr])
        mybad = np.empty((stptr.size, endptr.size))
        for i, s in enumerate(stptr):
            for j, e in enumerate(endptr):
                mybad[i, j] = abs(np.mean(vnorm[s - 1:e]))  # deviation from zero inside
        mybad -= left[:, np.newaxis]  # minus that of the outside regions
        mybad -= right[np.newaxis, :]
        a, b, best = _argmin_first_colmajor(mybad)
        if a is None:
            continue
        results[c] = [best, np.mean(Y[c, stptr[a] - 1:endptr[b]])]
    return results


def _summarise_d2_scaling(dat_v: np.ndarray, dat_M: np.ndarray, p: str,
                          out: dict) -> None:
    # Summarise local slopes of the correlation integral for one variant of the
    # D2 estimate (raw ``d2`` or Gaussian-smoothed ``d2g``). ``p`` prefixes the
    # output keys; results are written into ``out`` in place.
    try:
        benfind = _findscalingr_ind(dat_M)
    except Exception as exc:
        raise _D2DataError('Could not find a scaling range in the TISEAN d2 output '
                           'for this series') from exc

    # rows: increasing embedding m; columns: stpt, endpt, goodness, dim
    out[f'ben{p}_mindim'] = np.min(benfind[:, 3])
    out[f'ben{p}_maxdim'] = np.max(benfind[:, 3])
    out[f'ben{p}_meandim'] = np.mean(benfind[:, 3])
    out[f'ben{p}_meangoodness'] = np.mean(benfind[:, 2])

    mmin = _sub_findmmin(benfind[:, 3])
    # minimum scale at which a scaling range is observed: the start of the
    # scaling range found for embedding dimension m_min (column 0 of benfind
    # holds the 1-based start index into dat_v)
    start = np.nan if mmin['ri1'] is None else benfind[mmin['ri1'] - 1, 0]
    out[f'benmmin{p}_logminl'] = (np.nan if np.isnan(start)
                                  else np.log(dat_v[int(start) - 1]))
    out[f'benmmin{p}_goodness'] = mmin['goodness']
    out[f'benmmin{p}_stabledim'] = mmin['stabled']
    out[f'benmmin{p}_linrmserr'] = mmin['linrmserr']

    # Reshaped: only for large enough m (as determined by the criteria above),
    # then find a scaling region across m for a saturated range of m. Without a
    # starting dimension there are no rows to search (an empty selection in
    # MATLAB), and all of the outputs below are NaN.
    if mmin['ri1'] is None:
        sc = {'ri1': None, 'ri2': None, 'goodness': np.nan,
              'dimest': np.nan, 'dimstd': np.nan}
    else:
        sc = _findscalingr(dat_M[mmin['ri1'] - 1:, :])
    out[f'{p}_logminscr'] = (np.nan if sc['ri1'] is None
                             else np.log(dat_v[sc['ri1'] - 1]))
    out[f'{p}_logmaxscr'] = (np.nan if sc['ri2'] is None
                             else np.log(dat_v[sc['ri2'] - 1]))
    out[f'{p}_logscr'] = out[f'{p}_logmaxscr'] - out[f'{p}_logminscr']
    out[f'{p}_goodness'] = sc['goodness']
    out[f'{p}_dimest'] = sc['dimest']
    out[f'{p}_dimstd'] = sc['dimstd']


def tisean_d2(y: ArrayLike, tau: Union[int, str] = 1, maxm: int = 10,
              theiler_win: Union[int, float, list, tuple] = ('ac', 1)) -> Union[dict, float]:
    """
    Correlation dimension and entropy from the TISEAN package's ``d2`` routine.

    Estimates the correlation sum, the correlation dimension and the correlation
    entropy of the time series [1]_, then summarises the results.

    Takens' estimator [2]_ is computed for the correlation dimension, along with
    related statistics: other dimension estimates obtained by finding suitable
    scaling ranges, and a search for a flat region in the output of TISEAN's
    ``h2`` algorithm, which indicates determinism/deterministic chaos [3]_.

    To find a suitable scaling range, a penalized regression procedure is used to
    determine an optimal scaling range that simultaneously spans the greatest
    range of scales and shows the best fit to the data, and return the range, a
    goodness of fit statistic, and a dimension estimate.

    Unlike hctsa, which shells out to installed TISEAN binaries, this runs the
    vendored TISEAN sources in-process (see
    :mod:`pyhctsa.toolboxes.Tisean_3_0_1.tisean`).

    References
    ----------
    .. [1] R. Hegger, H. Kantz and T. Schreiber, "Practical implementation of
        nonlinear time series methods: The TISEAN package", Chaos 9(2) 413 (1999)
    .. [2] J. Theiler, "Spurious dimension from correlation algorithms applied to
        limited time-series data", Phys. Rev. A 34(3) 2427 (1986)
    .. [3] H. Kantz and T. Schreiber, "Nonlinear Time Series Analysis",
        Cambridge University Press (2004)

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        The time-delay: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of the
        autocorrelation function, ``'ac1e'`` the (floored) first 1/e crossing of
        the autocorrelation function, and ``'mi'`` the smaller of the first
        minimum of the (Kraskov) automutual information and the 1/e time.
        Default is 1.
    maxm : int, optional
        The maximum embedding dimension. Default is 10.
    theiler_win : int, float, or ``['ac', k]``, optional
        The Theiler window (see :func:`pyhctsa.utils.theiler_window`): a number of
        samples, ``['ac', k]`` for ``k`` times the first zero-crossing of the
        autocorrelation function (``['ac1e', k]`` is also accepted), or a value in
        ``(0, 1)`` taken as a proportion of the time-series length (legacy).
        Default is ``['ac', 1]``.

    Returns
    -------
    dict or float
        Statistics summarising Takens' estimator, the local slopes of the
        correlation sum (raw and Gaussian-kernel smoothed), and the correlation
        entropy. Returns NaN, as hctsa does, if the time series is too short, the time
        delay cannot be determined, or the TISEAN output is unusable for this series
        (no valid output for a long delay, Inf/NaN-contaminated correlation sums, no
        correlation-dimension data, or no scaling range).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size  # data length (number of samples)
    if n < 50:
        logger.warning(f'N = {n} too short for nonlinear dimension analysis')
        return np.nan

    # Time delay, tau
    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Time series cannot be embedded (could not get the time delay)')
        return np.nan
    tau = int(tau)

    # Theiler window
    theiler_win = theiler_window(y, theiler_win, n)
    if np.isnan(theiler_win):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    theiler_win = int(theiler_win)

    # Data-dependent failures (no usable TISEAN output, no scaling range, ...) return
    # NaN, as in hctsa, rather than raising
    try:
        return _tisean_d2_summary(y, tau, maxm, theiler_win)
    except _D2DataError as exc:
        logger.warning(str(exc))
        return np.nan


def _tisean_d2_summary(y: np.ndarray, tau: int, maxm: int, theiler_win: int) -> dict:
    # Run TISEAN's d2 and summarise it; raises _D2DataError where hctsa returns NaN.
    try:
        tables = _tisean.d2(y, delay=tau, embed=maxm, theiler=theiler_win)
    except ValueError as exc:  # e.g. a delay vector longer than the series
        raise _D2DataError(f'TISEAN d2 produced invalid output (perhaps due to long '
                           f'tau = {tau}, N = {y.size}): {exc}') from exc
    if _c2g_overflows(tables['c2']):
        raise _D2DataError('TISEAN d2 produced Inf/NaN-contaminated output for this data')
    c2gdat = _tisean.c2g(tables['c2'])
    c2tdat = _tisean.c2t(tables['c2'])
    d2dat, h2dat = tables['d2'], tables['h2']

    out = {}

    # --------------------------------------------------------------------------
    # (1) Takens estimator
    # --------------------------------------------------------------------------
    # Correlation dimension at an upper length scale of 0.5: for a z-scored time
    # series that is half the standard deviation, as Kantz & Schreiber recommend.
    takens05 = _sub_takens(c2tdat, 0.5)
    out['takens05_mean'] = np.mean(takens05)
    out['takens05_median'] = np.median(takens05)
    out['takens05_max'] = np.nanmax(takens05)
    out['takens05_min'] = np.nanmin(takens05)
    out['takens05_std'] = np.std(takens05, ddof=1)
    if np.all(np.isnan(takens05)):
        out['takens05_iqr'] = np.nan
    else:
        q75, q25 = np.percentile(takens05[~np.isnan(takens05)], [75, 25], method='hazen')
        out['takens05_iqr'] = q75 - q25

    # Find outliers as a means of inferring m_min: look for the estimate
    # approaching a constant for m > m_min
    mmintakens05 = _sub_findmmin(takens05)
    # minimum dimension at which a scaling range is observed:
    out['takens05mmin_ri'] = (np.nan if mmintakens05['ri1'] is None
                              else mmintakens05['ri1'])
    out['takens05mmin_goodness'] = mmintakens05['goodness']
    out['takens05mmin_stabled'] = mmintakens05['stabled']
    out['takens05mmin_linrmserr'] = mmintakens05['linrmserr']

    # --------------------------------------------------------------------------
    # (2) D2: local slopes of the correlation integral
    # --------------------------------------------------------------------------
    if all(b.size == 0 for b in d2dat):
        raise _D2DataError('TISEAN d2 returned no usable correlation-dimension data '
                           'for this series')
    d2dat_v, d2dat_M = _sub_celltomat(d2dat, 2)
    _summarise_d2_scaling(d2dat_v, d2dat_M, 'd2', out)

    # --------------------------------------------------------------------------
    # (3) Gaussian-smoothed estimates: as for D2, on c2g's third column
    # --------------------------------------------------------------------------
    d2gdat_v, d2gdat_M = _sub_celltomat(c2gdat, 3)
    _summarise_d2_scaling(d2gdat_v, d2gdat_M, 'd2g', out)

    # --------------------------------------------------------------------------
    # (4) H2: a flat region indicates determinism/deterministic chaos
    # --------------------------------------------------------------------------
    h2dat_v, h2dat_M = _sub_celltomat(h2dat, 2)
    h2results = _sub_getslopes(h2dat_v, h2dat_M)
    if h2results is None:
        return np.nan
    slopesh2 = h2results[:, 3]  # slopes for each dimension

    # What are the (robust, mid-range) slopes like?
    findch_h2 = _sub_findmmin(slopesh2)
    out['slopesh2_ri1'] = np.nan if findch_h2['ri1'] is None else findch_h2['ri1']
    out['slopesh2_goodness'] = findch_h2['goodness']
    out['slopesh2_stabled'] = findch_h2['stabled']
    out['slopesh2_linrmserr'] = findch_h2['linrmserr']

    # Are there any intermediate flat regions (signature of deterministic chaos)?
    flattens = _sub_doesflatten(h2dat_v, h2dat_M)
    if flattens is None:
        return np.nan
    out['h2meangoodness'] = np.mean(flattens[:, 0])  # how close to having flat regions
    out['h2bestgoodness'] = np.min(flattens[:, 0])   # best you can do
    out['h2besth2'] = flattens[int(np.argmin(flattens[:, 0])), 1]
    out['meanh2'] = np.mean(flattens[:, 1])
    out['medianh2'] = np.median(flattens[:, 1])

    flatsh2min = _sub_findmmin(flattens[:, 1])
    out['flatsh2min_ri1'] = np.nan if flatsh2min['ri1'] is None else flatsh2min['ri1']
    out['flatsh2min_goodness'] = flatsh2min['goodness']
    out['flatsh2min_stabled'] = flatsh2min['stabled']
    out['flatsh2min_linrmserr'] = flatsh2min['linrmserr']

    return out

from ..toolboxes.matlab.matlab_fit import robustfit
from ..utils import _round_half_away


def _embed_tau_m(y: np.ndarray, embed_params) -> tuple:
    """
    The delay and dimension of ``[tau, m]`` embedding parameters (hctsa's
    ``BF_Embed(y, tau, m, true)``), without doing the embedding.

    ``tau`` is an integer or a rule understood by :func:`pyhctsa.utils.get_tau`;
    ``m`` is an integer, or ``'fnn'`` (or ``['fnn', threshold]``) for the
    embedding dimension from TISEAN's false nearest neighbors. Returns
    ``(nan, nan)`` if the delay cannot be set.
    """
    if not isinstance(embed_params, (list, tuple)) or len(embed_params) != 2:
        raise ValueError('Embedding parameters are formatted incorrectly -- need [tau, m]')
    tau = get_tau(y, embed_params[0])
    if np.isnan(tau):
        return np.nan, np.nan
    m = embed_params[1]
    if isinstance(m, (list, tuple)):
        m = m[0] if len(m) == 1 and not isinstance(m[0], str) else m
    if isinstance(m, str) or isinstance(m, (list, tuple)):
        if (m if isinstance(m, str) else m[0]) == 'fnn':
            raise NotImplementedError(
                "m='fnn' needs a port of TISEAN's false_nearest (hctsa's NL_FNN), which is "
                "not yet available in pyhctsa; pass an integer embedding dimension.")
        raise ValueError('Embedding dimension, m, incorrectly specified.')
    return int(tau), int(m)


def gp_corr_sum(y: ArrayLike, nref: Union[int, float] = 500, r: float = 0.05,
                thwin: Union[int, float, list, tuple] = ('ac', 1), nbins: int = 20,
                embed_params: Union[list, tuple] = ('ac', 'fnn'), do_two: int = 1) -> Union[dict, float]:
    """
    How the number of close pairs of points in the delay embedding grows with
    distance (the correlation sum and its scaling).

    Computes the correlation sum, :math:`C(\\epsilon)`, the fraction of pairs of
    time-delay-embedded points closer than :math:`\\epsilon`, by the
    Grassberger-Procaccia algorithm [1]_, using TISEAN's ``d2`` (hctsa no longer
    uses TSTOOL's ``corrsum``/``corrsum2``). For a low-dimensional attractor,
    :math:`\\ln C` rises linearly with :math:`\\ln \\epsilon`, with a slope equal to the
    correlation dimension. The outputs summarize the range of :math:`\\ln \\epsilon`
    and :math:`\\ln C(\\epsilon)`, and an iteratively re-weighted least squares (robust)
    linear fit to the log-log plot.

    References
    ----------
    .. [1] P. Grassberger and I. Procaccia, "Characterization of Strange Attractors",
        Phys. Rev. Lett. 50(5), 346 (1983).

    Parameters
    ----------
    y : array-like
        Input time series.
    nref : int or float, optional
        Number of (randomly chosen) reference points: ``-1`` uses all points, a
        value in (0, 1) is a fraction of the time-series length. Default is 500.
    r : float, optional
        Maximum search radius, in units of ``std(y) * sqrt(m)`` where ``m`` is the
        embedding dimension. Default is 0.05.
    thwin : int, float, or ``['ac', k]``, optional
        The Theiler window of samples to exclude before and after each reference
        index (see :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k``
        times the first zero-crossing of the autocorrelation function, or a number
        of samples. Default is ``['ac', 1]``.
    nbins : int, optional
        Number of (log-spaced) radii at which the correlation sum is found.
        Default is 20.
    embed_params : [tau, m], optional
        Embedding parameters: ``tau`` is an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), ``m`` an
        integer, or ``'fnn'`` (TISEAN's false nearest neighbors, not yet available
        in pyhctsa and raises ``NotImplementedError``). Default is ``['ac', 'fnn']``.
    do_two : int, optional
        Only 1 (corrsum-style log-spaced radii, the default) is supported; 2 has no
        TISEAN equivalent and raises ``ValueError``, as in hctsa.

    Returns
    -------
    dict or float
        Only radii with a finite :math:`\\ln C(\\epsilon)` are used. ``minlnr``,
        ``maxlnr``: the smallest and largest :math:`\\ln \\epsilon`; ``minlnCr``,
        ``maxlnCr``, ``rangelnCr``, ``meanlnCr``: the minimum, maximum, range and
        mean of :math:`\\ln C`; ``robfit_a1``, ``robfit_a2``: intercept and slope of a
        robust linear fit of :math:`\\ln C` against :math:`\\ln \\epsilon`;
        ``robfit_sigrat``: ratio of the ordinary least-squares to the robust estimate
        of the residual standard deviation; ``robfit_s``: the robust estimate of the
        residual standard deviation; ``robfit_sea1``, ``robfit_sea2``: standard
        errors of the intercept and slope; ``robfitresmeanabs``, ``robfitresmeansq``,
        ``robfitresac1``: mean absolute and mean squared residual, and lag-1
        autocorrelation of the residuals. The fit outputs are NaN when too few radii
        have a finite :math:`\\ln C`. Returns NaN if the delay or Theiler window cannot
        be set, the embedding is too short, or no correlation sum is obtained.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    # Number of reference points
    if 0 < nref < 1:
        nref = int(_round_half_away(n * nref))  # a proportion of the series length
    if nref >= n:
        nref = -1  # capped at the time-series length

    # Remove spurious correlations of adjacent points
    thwin = theiler_window(y, thwin, n)
    if np.isnan(thwin):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    thwin = int(thwin)

    if do_two == 2:
        raise ValueError("gp_corr_sum: do_two = 2 (corrsum2's fixed-pairs-per-bin binning) "
                         "has no TISEAN equivalent and is not supported.")

    tau, m = _embed_tau_m(y, embed_params)
    if np.isnan(tau):
        logger.warning('Could not determine embedding parameters for this time series')
        return np.nan

    if (n - (m - 1) * tau) < thwin:
        logger.warning(f'Embedded time series (N = {n}, m = {m}, tau = {tau}) too short '
                       'to do a correlation sum')
        return np.nan

    # TISEAN's d2: -N0 uses all pairs; the radius is in standard deviations of y, scaled by
    # sqrt(m) since a pairwise distance in an m-dimensional embedding scales as std(y)*sqrt(m)
    max_eps = float('%g' % (r * np.std(y, ddof=1) * np.sqrt(m)))  # -R%g: six significant digits
    try:
        tables = _tisean.d2(y, delay=tau, embed=m, theiler=thwin, howoften=nbins,
                            maxfound=0 if nref == -1 else int(nref), epsmax=max_eps)
    except ValueError as exc:  # e.g. a delay vector longer than the series
        logger.warning(f'TISEAN d2 produced invalid output: {exc}')
        return np.nan

    # [r, C(r)] at the m-th embedding dimension (the radii are log-spaced)
    rc = tables['c2'][m - 1]
    if rc.shape[0] == 0:
        logger.warning("No output obtained from d2's correlation sum.")
        return np.nan
    with np.errstate(divide='ignore'):
        lnr = np.log(rc[:, 0])
        lncr = np.log(rc[:, 1])

    # Only keep finite values
    good = np.isfinite(lncr)
    if not good.any():
        logger.warning('No good outputs obtained from the correlation sum.')
        return np.nan
    lnr, lncr = lnr[good], lncr[good]

    out = {}
    out['minlnr'] = np.min(lnr)
    out['maxlnr'] = np.max(lnr)
    out['minlnCr'] = np.min(lncr)
    out['maxlnCr'] = np.max(lncr)
    out['rangelnCr'] = np.ptp(lncr)
    out['meanlnCr'] = np.mean(lncr)

    # Robust linear fit to the log-log plot (full range)
    try:
        a, stats = robustfit(lnr, lncr)
    except (ValueError, np.linalg.LinAlgError):  # too few finite points to fit
        a = None
    if a is not None:
        res = lncr - (a[1] * lnr + a[0])
        out['robfit_a1'] = a[0]
        out['robfit_a2'] = a[1]
        out['robfit_sigrat'] = stats['ols_s'] / stats['robust_s']
        out['robfit_s'] = stats['s']
        out['robfit_sea1'] = stats['se'][0]
        out['robfit_sea2'] = stats['se'][1]
        out['robfitresmeanabs'] = np.mean(np.abs(res))
        out['robfitresmeansq'] = np.mean(res ** 2)
        out['robfitresac1'] = autocorr(res, 1, 'Fourier')[0]
    else:
        for k in ('robfit_a1', 'robfit_a2', 'robfit_sigrat', 'robfit_s', 'robfit_sea1',
                  'robfit_sea2', 'robfitresmeanabs', 'robfitresmeansq', 'robfitresac1'):
            out[k] = np.nan

    return out


from scipy.spatial import cKDTree


def takens_estimator(y: ArrayLike, nref: int = -1, rad: float = 0.05,
                     past: Union[int, float, list, tuple] = ('ac', 1),
                     embed_params: Union[list, tuple] = ('ac', 'fnn')) -> float:
    """
    Takens' estimator for the correlation dimension.

    Takens' maximum-likelihood estimator [1]_ of the correlation dimension at an
    upper length scale ``eup = rad * std(y)``:

    .. math::
        D_T = 1 / \\langle \\ln(\\epsilon_{up} / r_{ij}) \\rangle,

    the mean taken over all pairs ``(i, j)`` of delay vectors with max-norm distance
    :math:`r_{ij} < \\epsilon_{up}`, excluding pairs closer in time than the Theiler
    window and exact duplicate vectors. It is computed natively (a KD-tree range search
    at the one radius needed), exactly at ``eup`` from the pair distances themselves,
    rather than from the logarithmically binned correlation sum of TISEAN's ``d2``
    followed by ``c2t`` as in earlier versions of hctsa. Kantz and Schreiber's
    recommendation of half a standard deviation for the length scale is used the same
    way in :func:`tisean_d2`.

    References
    ----------
    .. [1] F. Takens, "On the numerical determination of the dimension of an attractor",
        in B.L.J. Braaksma, H.W. Broer and F. Takens (eds.), Dynamical Systems and
        Bifurcations (Groningen, 1984), Lecture Notes in Mathematics 1125, 99-106,
        Springer, Berlin (1985). DOI: 10.1007/BFb0075637

    Parameters
    ----------
    y : array-like
        Input time series.
    nref : int, optional
        The number of reference points (the first ``nref`` delay vectors); ``-1`` uses
        all points. Default is -1.
    rad : float, optional
        The upper length scale at which to read off the dimension estimate, in standard
        deviations of ``y``. Default is 0.05.
    past : int, float, or ``['ac', k]``, optional
        The Theiler window (see :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for
        ``k`` times the first zero-crossing of the autocorrelation function, or a number
        of samples. Default is ``['ac', 1]``.
    embed_params : [tau, m], optional
        Embedding parameters: ``tau`` is an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), ``m`` an
        integer, or ``'fnn'`` (TISEAN's false nearest neighbors, not yet available in
        pyhctsa and raises ``NotImplementedError``). Default is ``['ac', 'fnn']``.

    Returns
    -------
    float
        Takens' estimator of the correlation dimension. NaN if the delay or Theiler
        window cannot be set, the series cannot be embedded, it is constant, or no pair
        of delay vectors lies within the length scale (or all such pairs are exact
        duplicates, e.g. heavily quantized data). For high embedding dimensions of
        noise-like series no pair may fall within the length scale at all.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    past = theiler_window(y, past, n)
    if np.isnan(past):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan

    tau, m = _embed_tau_m(y, embed_params)
    if np.isnan(tau):
        logger.warning('Could not embed this time series with these embedding parameters')
        return np.nan
    try:
        emb = time_delay_embed(y, m, tau)
    except ValueError as exc:  # too short to embed
        logger.warning(str(exc))
        return np.nan
    n_emb = emb.shape[0]

    # Reference points: the first nref delay vectors, or all
    n_ref = n_emb if (nref == -1 or nref >= n_emb) else int(nref)

    eup = rad * np.std(y, ddof=1)  # upper length scale, in data units
    if not eup > 0:
        return np.nan  # constant series

    # Sum ln(eup / r_ij) over pairs with r_ij <= eup (max norm) outside the Theiler window,
    # over reference points in chunks (a low-dimensional attractor can have O(N^2) pairs
    # within eup, so they are never all held at once)
    tree = cKDTree(emb)
    sum_log, num_pairs = 0.0, 0
    chunk = 500
    for c in range(0, n_ref, chunk):
        refs = np.arange(c, min(c + chunk, n_ref))
        pairs = cKDTree(emb[refs]).sparse_distance_matrix(
            tree, eup, p=np.inf, output_type='ndarray')
        keep = np.abs(pairs['j'] - refs[pairs['i']]) > past  # outside the Theiler window
        d = pairs['v'][keep]
        d = d[d > 0]  # exact duplicates carry no length-scale information (ln -> Inf)
        sum_log += np.sum(np.log(eup / d))
        num_pairs += d.size

    if num_pairs == 0:
        logger.warning(f'No pairs within {rad:g} standard deviations of each other to '
                       'estimate a correlation dimension from')
        return np.nan

    return num_pairs / sum_log  # Takens' estimator: 1 / mean(ln(eup/r))


from scipy.optimize import minimize_scalar
from ..utils import _ml_randperm
from scipy.special import digamma


def _fractal_dim_error(d: float, g: float, kmin: int, kmax: int, mom: np.ndarray) -> float:
    # Robust (log(1 + e^2/2)) error between the measured k-th-neighbor-distance moments,
    # mom[kmin:kmax], and the curve expected for dimension d, after fitting the curve's free
    # overall scale factor (gendimest.cpp's Error_Function, van de Water & Schram 1988).
    ks = np.arange(kmin, kmax + 1)
    if g == 0:
        z = np.exp(digamma(ks) / d)
    else:
        z = np.ones(ks.size)  # anchored at k = kmin
        running = np.cumprod((ks[:-1] + g / d) / ks[:-1])
        with np.errstate(invalid='ignore'):
            z[1:] = running ** (1 / g)
    mk = mom[kmin - 1:kmax]
    scale_err = lambda a: np.sum(np.log(1 + 0.5 * (mk - a * z) ** 2))
    a = minimize_scalar(scale_err, bounds=(0, 1e6), method='bounded',
                        options={'xatol': 1e-5}).x
    return scale_err(a)


def fractal_dimensions(y: ArrayLike, kmin: int = 3, kmax: int = 10,
                       nref: Union[int, float] = 0.2, gstart: float = 1, gend: float = 10,
                       past: Union[int, float, list, tuple] = ('ac', 1), steps: int = 32,
                       embed_params: Union[list, tuple] = ('ac', 'fnn'),
                       random_seed: Union[int, None] = 0) -> Union[dict, float]:
    """
    The spectrum of generalized (fractal) dimensions of the delay embedding, estimated
    from nearest-neighbor distances.

    Estimates :math:`D(q)`, the generalized dimension of the time-delay embedding as a
    function of the order of the moment, from the distances of reference points to their
    nearest neighbors, by the method of van de Water and Schram [1]_ (that of TSTOOL's
    ``fracdims``). For each of ``nref`` reference points, the distances to its 1st to
    ``kmax``-th nearest neighbors are found (excluding a Theiler window of ``past``
    samples). For each moment order :math:`\\gamma` swept linearly from ``gstart`` to
    ``gend`` (``steps`` values), the :math:`\\gamma`-th moment of the k-th-neighbor
    distance across all reference points is

    .. math::
        M(k) = \\langle r_k^\\gamma \\rangle^{1/\\gamma}

    (or :math:`\\exp\\langle \\ln r_k \\rangle` as :math:`\\gamma \\to 0`), for
    :math:`k = 1..k_{max}`. Under an assumed dimension :math:`D` the expected relation is
    :math:`M(k) \\propto (\\Gamma(k + \\gamma/D)/\\Gamma(k))^{1/\\gamma}` (or
    :math:`\\exp(\\psi(k)/D)` as :math:`\\gamma \\to 0`), up to an overall scale factor that is
    fitted separately. :math:`D(\\gamma)` is the value that best matches the measured moments
    :math:`M(k_{min}..k_{max})` under a robust :math:`\\log(1 + e^2/2)` error, found by nested
    bounded one-dimensional minimizations. Finally :math:`q(\\gamma) = 1 - \\gamma / D(\\gamma)`.
    The outputs summarize :math:`D` and :math:`q` across the moments, and a straight-line
    fit of :math:`D` against :math:`q`.

    References
    ----------
    .. [1] W. van de Water and P. Schram, "Generalized dimensions from near-neighbor
        information", Phys. Rev. A 37(8), 3118-3125 (1988). DOI: 10.1103/PhysRevA.37.3118

    Parameters
    ----------
    y : array-like
        Input time series.
    kmin : int, optional
        Minimum number of neighbors for each reference point. Default is 3.
    kmax : int, optional
        Maximum number of neighbors for each reference point. Default is 10.
    nref : int or float, optional
        Number of randomly chosen reference points: ``-1`` uses all points, a value in
        (0, 1) is a proportion of the embedded points. Default is 0.2.
    gstart, gend : float, optional
        Starting and ending values of the moment order. Defaults are 1 and 10.
    past : int, float, or ``['ac', k]``, optional
        The Theiler window of samples to exclude before and after each reference index (see
        :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k`` times the first
        zero-crossing of the autocorrelation function, or a number of samples. Default is
        ``['ac', 1]``.
    steps : int, optional
        Number of moments to calculate. Default is 32.
    embed_params : [tau, m], optional
        Embedding parameters: ``tau`` is an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), ``m`` an integer,
        or ``'fnn'`` (TISEAN's false nearest neighbors, not yet available in pyhctsa and
        raises ``NotImplementedError``). Default is ``['ac', 'fnn']``.
    random_seed : int, optional
        Seed for choosing the random subsample of reference points (relevant when
        ``nref != -1``; the subsample differs from MATLAB's). Default is 0.

    Returns
    -------
    dict or float
        ``rangeDq``, ``maxDq``, ``meanDq``: range, maximum and mean of :math:`D` across the
        moments; ``maxq``, ``rangeq``, ``meanq``: maximum, range and mean of :math:`q`;
        ``linfit_a``, ``linfit_b``: slope and intercept of a linear fit of :math:`D` against
        :math:`q`; ``linfit_rmsqres``: root-mean-square residual of that fit. Returns NaN
        if the Theiler window or delay cannot be set, the embedding fails, ``kmax`` is not
        smaller than the number of embedded points, or too few neighbors lie outside the
        Theiler window.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    # Number of reference points
    if 0 < nref < 1:
        nref = int(_round_half_away(n * nref))  # a proportion of time-series length

    past = theiler_window(y, past)
    if np.isnan(past):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    past = int(past)

    # Embed the signal
    tau, m = _embed_tau_m(y, embed_params)
    if np.isnan(tau):
        logger.warning(f'Embedding of the {n}-sample time series failed')
        return np.nan
    try:
        emb = time_delay_embed(y, m, tau)
    except ValueError as exc:  # too short to embed
        logger.warning(f'Embedding of the {n}-sample time series failed: {exc}')
        return np.nan
    n_emb = emb.shape[0]

    if kmax >= n_emb:  # too many neighbors requested
        return np.nan

    # Reference points
    if nref == -1 or nref >= n_emb:
        ref_idx = np.arange(n_emb)
    else:
        ref_idx = _ml_randperm(n_emb, _ml_rng(0 if random_seed is None else int(random_seed)))[:int(nref)] - 1
    n_ref = ref_idx.size

    # For each reference point, the distances to its 1st..kmax-th nearest neighbors outside
    # the Theiler window (a KD-tree, over-fetching neighbors to cover those excluded)
    k_fetch = min(n_emb - 1, kmax + 2 * past + 5)
    dist, idx = cKDTree(emb).query(emb[ref_idx], k=k_fetch + 1)
    valid = np.abs(idx - ref_idx[:, None]) > past
    dist = np.sort(np.where(valid, dist, np.inf), axis=1)[:, :kmax]
    for ii in np.flatnonzero(valid.sum(axis=1) < kmax):  # fall back to a full search
        all_dists = np.linalg.norm(emb - emb[ref_idx[ii]], axis=1)
        all_dists[np.abs(np.arange(n_emb) - ref_idx[ii]) <= past] = np.inf
        all_dists = np.sort(all_dists)
        if np.sum(np.isfinite(all_dists)) < kmax:
            return np.nan  # not enough valid neighbors exist at all
        dist[ii] = all_dists[:kmax]
    # dist[i, k]: the i-th reference point's distance to its (k+1)-th nearest neighbor

    # Sweep the moment order and fit a dimension D(gamma) to each
    gammas = np.linspace(gstart, gend, int(steps)) if (gend - gstart > 0 and steps > 1) \
        else np.array([gstart], dtype=float)
    dq = np.zeros(gammas.size)
    q = np.zeros(gammas.size)
    with np.errstate(divide='ignore', invalid='ignore'):
        for gi, g in enumerate(gammas):
            # gamma-th moment of the k-th-neighbor distance, across reference points
            if g == 0:
                mom = np.exp(np.mean(np.log(dist), axis=0))
            else:
                mom = np.mean(dist ** g, axis=0) ** (1 / g)
            dq[gi] = minimize_scalar(
                lambda d: _fractal_dim_error(d, g, kmin, kmax, mom),
                bounds=(max(0.05, -g / kmin), 128), method='bounded',
                options={'xatol': 1e-4}).x
            q[gi] = 1 - g / dq[gi]

    out = {}
    out['rangeDq'] = np.ptp(dq)
    out['maxDq'] = np.max(dq)
    out['meanDq'] = np.mean(dq)
    out['maxq'] = np.max(q)
    out['rangeq'] = np.ptp(q)
    out['meanq'] = np.mean(q)

    # Linear fit of D against q
    p = np.polyfit(q, dq, 1)
    res = np.polyval(p, q) - dq
    out['linfit_a'] = p[0]
    out['linfit_b'] = p[1]
    out['linfit_rmsqres'] = np.sqrt(np.mean(res ** 2))
    return out


from ..toolboxes.Tisean_3_0_1.tisean import _e, _round_significant


def _tisean_boxcount(y: np.ndarray, delay: int, maxembed: int, epscount: int) -> tuple:
    # TISEAN's ``boxcount -M1,<maxembed> -d<delay> -Q0.0 -#<epscount>`` (source_c/boxcount.c),
    # in process: ln N(eps), the log of the number of occupied cells of a partition of the
    # delay embedding into cubes of side eps, for embedding dimensions 1..maxembed.
    # Returns (eps, logN), with eps of shape (epscount,) and logN (epscount, maxembed), both
    # rounded through C's %e as in the .box file hctsa reads back.
    y = _round_significant(np.asarray(y, dtype=float).ravel(), 7)  # BF_WriteTempFile
    ymin = np.min(y)
    maxinterval = np.max(y) - ymin
    if maxinterval == 0:
        raise _D2DataError('boxcount: the data are constant')
    epsmin, epsmax = 1e-3, 1.0  # relative to the data interval
    x = (y - ymin) / maxinterval
    x[x >= 1.0] -= epsmin / 2.0
    length = y.size - (maxembed - 1) * delay
    if length < 1 or epscount < 2:
        raise _D2DataError('boxcount: time series too short for this embedding')
    epsfaktor = (epsmax / epsmin) ** (1.0 / (epscount - 1))

    eps = np.empty(epscount)
    log_n = np.empty((epscount, maxembed))
    heps = epsmax * epsfaktor
    epsi_old = 0
    for k in range(epscount):
        while True:  # the number of boxes per axis is an integer that must increase
            heps /= epsfaktor
            epsi = int(1.0 / heps)
            if epsi > epsi_old:
                break
        epsi_old = epsi
        eps[k] = heps * maxinterval
        labels = np.zeros(length, dtype=np.int64)
        for d in range(maxembed):  # nested partition: cells are distinguished by coordinates 1..d+1
            box = (x[d * delay:d * delay + length] * epsi).astype(np.int64)
            labels = np.unique(labels * epsi + box, return_inverse=True)[1].ravel()
            log_n[k, d] = np.log(labels.max() + 1)
    return np.array([_e(v) for v in eps]), np.vectorize(_e)(log_n)


def _dimensions_scaling_range(logr: np.ndarray, log_n: np.ndarray, gamma: float = 0.02) -> tuple:
    # The scaling range of ln N(eps) against ln eps (start in the first half, end in the second
    # half) that minimizes the mean absolute residual of a straight-line fit less gamma per point
    # spanned. Returns (first index, last index, badness matrix, polyfit coefficients, residuals).
    stptr, endptr = _scaling_range_endpoints(logr.size)
    if stptr.size == 0 or endptr.size == 0:
        raise _D2DataError('too few length scales to find a scaling range')
    mybad = np.empty((stptr.size, endptr.size))
    for i, s in enumerate(stptr):
        for j, e in enumerate(endptr):
            xs, ys = logr[s - 1:e], log_n[s - 1:e]
            p = np.polyfit(xs, ys, 1)
            mybad[i, j] = np.mean(np.abs(p[0] * xs + p[1] - ys)) - gamma * xs.size
    a, b, _ = _argmin_first_colmajor(mybad)
    s, e = int(stptr[a]), int(endptr[b])  # 1-based, inclusive
    xs, ys = logr[s - 1:e], log_n[s - 1:e]
    p = np.polyfit(xs, ys, 1)
    return s, e, mybad, p, p[0] * xs + p[1] - ys


def _dimensions_by_m(logr: np.ndarray, log_n: np.ndarray, prefix: str, out: dict) -> None:
    # How ln N(eps) (or ln C(eps)) changes with m; at least m = 3 is always computed
    cols = ((0, '1'), (1, '2'), (2, '3'), (-1, 'max'))
    for j, lab in cols:
        out[f'{prefix}_meanm{lab}' if j >= 0 else f'{prefix}_meanmmax'] = np.mean(log_n[:, j])
    for j, lab in cols:
        out[f'{prefix}_minm{lab}' if j >= 0 else f'{prefix}_minmmax'] = np.min(log_n[:, j])
    for j, lab in cols:
        out[f'{prefix}_range{lab}' if j >= 0 else f'{prefix}_rangemmax'] = np.ptp(log_n[:, j])
    # increments with m
    out[f'{prefix}_mindiff'] = np.mean([np.min(log_n[:, 1]) - np.min(log_n[:, 0]),
                                        np.min(log_n[:, 2]) - np.min(log_n[:, 1])])
    out[f'{prefix}_meandiff'] = np.mean([np.mean(log_n[:, 1]) - np.mean(log_n[:, 0]),
                                         np.mean(log_n[:, 2]) - np.mean(log_n[:, 1])])
    # slopes and goodness of a straight-line fit across the whole range of length scales
    for j, lab in cols:
        p = np.polyfit(logr, log_n[:, j], 1)
        out[f'{prefix}_lfitm{lab}'] = p[0]
        out[f'{prefix}_lfitb{lab}'] = p[1]
        out[f'{prefix}_lfitmeansqdev{lab}'] = np.mean((log_n[:, j] - (p[0] * logr + p[1])) ** 2)


def _dimensions_scaling(logr: np.ndarray, log_n: np.ndarray, prefix: str, out: dict) -> None:
    # The scaling range for one embedding dimension, and the fit within it
    s, e, mybad, p, res = _dimensions_scaling_range(logr, log_n)
    out[f'{prefix}_logrmin'] = logr[s - 1]  # minimum of the scaling range
    out[f'{prefix}_logrmax'] = logr[e - 1]  # maximum of the scaling range
    out[f'{prefix}_logrrange'] = logr[e - 1] - logr[s - 1]
    out[f'{prefix}_pgone'] = (s - 1 + logr.size - e) / logr.size  # proportion of points removed
    out[f'{prefix}_meanabsres'] = np.mean(np.abs(res))
    out[f'{prefix}_meansqres'] = np.mean(res ** 2)
    out[f'{prefix}_scaling_exp'] = p[0]
    out[f'{prefix}_scaling_int'] = p[1]
    out[f'{prefix}_minbad'] = np.min(mybad)


def _dimensions_best_m(logr: np.ndarray, log_nn: np.ndarray, prefix: str, out: dict) -> None:
    # The scaling exponent in each embedding dimension, and which dimension is fitted best
    exps = np.empty(log_nn.shape[1])
    msq = np.empty(log_nn.shape[1])
    for k in range(log_nn.shape[1]):
        _, _, _, p, res = _dimensions_scaling_range(logr, log_nn[:, k])
        exps[k], msq[k] = p[0], np.mean(res ** 2)
    out[f'{prefix}_minscalingexp'] = np.min(exps)
    out[f'{prefix}_meanscalingexp'] = np.mean(exps)
    out[f'{prefix}_maxscalingexp'] = np.max(exps)
    out[f'{prefix}_mbestfit'] = int(np.argmin(msq)) + 1


def dimensions(y: ArrayLike, num_bins: int = 50,
               embed_params: Union[list, tuple] = ('ac', 'fnn')) -> Union[dict, float]:
    """
    Box-counting and correlation-sum estimates of the dimension of the delay embedding, and
    how they change with the embedding dimension.

    Uses TISEAN's ``boxcount`` (the Renyi entropy of order 0, :math:`\\ln N(\\epsilon)`, the log of
    the number of occupied boxes of a partition of the delay embedding) and ``d2`` (the
    correlation sum :math:`\\ln C(\\epsilon)`) over ``num_bins`` geometrically spaced length scales
    and embedding dimensions 1 to ``max(m, 3)`` (``m`` the embedding dimension of
    ``embed_params``), to summarize the curves' means, minima, ranges and straight-line fits at
    ``m`` = 1, 2, 3 and the largest ``m``, their changes with ``m``, the scaling range in
    :math:`\\ln \\epsilon` (the range of scales, in the first and second halves of the scales,
    minimizing the mean absolute error of a linear fit less 0.02 per point spanned) for
    ``m`` = 1, 2, 3 and the embedding dimension ``m``, and the embedding dimension with the best
    scaling fit. Unlike hctsa, which shells out to installed TISEAN binaries, this runs the
    vendored ``d2`` and an in-process port of ``boxcount`` (hctsa's TSTOOL-based version of
    this operation is no longer used).

    Parameters
    ----------
    y : array-like
        Input time series.
    num_bins : int, optional
        Number of length scales (bins per axis) at which to evaluate the box counts and
        correlation sums. Default is 50.
    embed_params : [tau, m], optional
        Embedding parameters: ``tau`` is an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), ``m`` an integer, or
        ``'fnn'`` (TISEAN's false nearest neighbors, not yet available in pyhctsa and raises
        ``NotImplementedError``). Default is ``['ac', 'fnn']``.

    Returns
    -------
    dict or float
        With prefix ``bc`` (box counting, :math:`\\ln N`) and ``co`` (correlation sum,
        :math:`\\ln C`): ``<p>_meanm1/2/3/max``, ``<p>_minm1/2/3/max``, ``<p>_range1/2/3/max``,
        ``<p>_mindiff``, ``<p>_meandiff``, ``<p>_lfitm1/2/3/max`` (slope), ``<p>_lfitb...``
        (intercept), ``<p>_lfitmeansqdev...``; ``scr_<p>_m1/m2/m3/mopt_*``: scaling range
        (``logrmin``, ``logrmax``, ``logrrange``, ``pgone``) and fit (``meanabsres``,
        ``meansqres``, ``scaling_exp``, ``scaling_int``, ``minbad``); ``<p>_minscalingexp``,
        ``<p>_meanscalingexp``, ``<p>_maxscalingexp``, ``<p>_mbestfit``. Returns NaN if the delay
        cannot be set, any ln N or ln C is not finite (e.g. some correlation sum is zero), or the
        series is constant or too short.
    """
    y = np.asarray(y, dtype=float).ravel()

    tau, mopt = _embed_tau_m(y, embed_params)
    if np.isnan(tau):
        logger.warning('Could not determine embedding parameters for this time series')
        return np.nan
    big_m = max(mopt, 3)  # at least three dimensions, for the statistics below

    try:
        # Box counting
        bc_r, bc_logn = _tisean_boxcount(y, tau, big_m, num_bins)
        bc_logr = np.log(bc_r)

        # Correlation sum, over the same number of scales: epsilon from max_eps/10 to max_eps
        max_eps = float('%g' % (np.std(y, ddof=1) * np.sqrt(big_m)))
        min_eps = float('%g' % (max_eps / 10))
        tables = _tisean.d2(y, delay=tau, embed=big_m, theiler=0, howoften=num_bins,
                            maxfound=0, epsmax=max_eps, epsmin=min_eps)
        if any(b.shape[0] != num_bins for b in tables['c2']):
            raise _D2DataError("TISEAN d2 returned an unexpected number of length scales")
        co_logr = np.log(tables['c2'][0][:, 0])
        with np.errstate(divide='ignore'):
            co_logc = np.column_stack([np.log(b[:, 1]) for b in tables['c2']])

        if not (np.all(np.isfinite(bc_logn)) and np.all(np.isfinite(co_logc))):
            logger.warning('No good outputs obtained from the box-counting/correlation dimension curves.')
            return np.nan

        out = {}
        _dimensions_by_m(bc_logr, bc_logn, 'bc', out)
        _dimensions_by_m(co_logr, co_logc, 'co', out)
        for prefix, logr, logn in (('bc', bc_logr, bc_logn), ('co', co_logr, co_logc)):
            for col, lab in ((0, 'm1'), (1, 'm2'), (2, 'm3'), (mopt - 1, 'mopt')):
                _dimensions_scaling(logr, logn[:, col], f'scr_{prefix}_{lab}', out)
        _dimensions_best_m(bc_logr, bc_logn, 'bc', out)
        _dimensions_best_m(co_logr, co_logc, 'co', out)
    except (_D2DataError, ValueError) as exc:  # data-dependent failures give NaN
        logger.warning(str(exc))
        return np.nan
    return out


def _count_boxes(x: np.ndarray, y: np.ndarray, nbox: int) -> np.ndarray:
    """Counts of points per box, where the boxes are quantiles along each axis."""
    props = np.arange(nbox + 1) / nbox
    xbox = matlab_quantile(x, props)
    ybox = matlab_quantile(y, props)
    # Nudge the top edge so the largest point falls inside the last box.
    xbox[-1] += 1
    ybox[-1] += 1

    boxcounts = np.zeros((nbox, nbox))
    for ii in range(nbox):  # x
        rx = (x >= xbox[ii]) & (x < xbox[ii + 1])  # these x are in range
        # only need to look at those ys for which the xs are in range
        yr = y[rx]
        for jj in range(nbox):  # y
            boxcounts[ii, jj] = np.sum((yr >= ybox[jj]) & (yr < ybox[jj + 1]))
    return boxcounts


def poincare_section(y: ArrayLike, ref: str = 'max',
                     tau: Union[int, str] = 'mi') -> Union[dict, float]:
    """
    Poincare section analysis of a time series.

    Time-delay embeds the time series and computes a Poincare section using
    TISEAN's ``poincare``, which cuts the trajectory on a fixed embedding
    coordinate (the last, by convention) held at its own mean, in a single
    crossing direction. The embedding dimension is fixed at 3, so that the
    section is two-dimensional.

    Parameters
    ----------
    y : array-like
        Input time series.
    ref : {'max', 'min'}, optional
        Which of the two crossing directions to use: ``'max'`` takes crossings
        heading toward a local maximum (ascending through the mean, TISEAN's
        "from below", ``-C0``) and ``'min'`` those heading toward a local
        minimum (descending, ``-C1``). Default is ``'max'``.

        hctsa's operation previously used TSTOOL's ``poincare``, which cut a
        hyperplane orthogonal to the local tangent vector at a chosen reference
        point -- a construction TISEAN has no equivalent for -- and ``ref`` was
        repurposed to pick the crossing direction when it moved to TISEAN.
    tau : int or str, optional
        The time-delay of the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of the
        autocorrelation function, ``'ac1e'`` the (floored) first 1/e crossing of
        the autocorrelation function, and ``'mi'`` the smaller of the first
        minimum of the (Kraskov) automutual information and the 1/e time.
        Default is ``'mi'``.

    Returns
    -------
    dict or float
        Statistics on the x- and y-components of the vectors on the Poincare
        surface, on distances between adjacent points and from the mean
        position, and on the (Miller-Madow-corrected) entropy of the boxed vector
        cloud. Returns NaN if fewer than two section points were found.
    """
    if ref == 'max':
        direction = 0  # crossing from below (heading toward a local maximum)
    elif ref == 'min':
        direction = 1  # crossing from above (heading toward a local minimum)
    else:
        raise ValueError(f"ref must be 'max' or 'min', got '{ref}'. TISEAN's "
                         'poincare has no reference-point concept, only a '
                         'choice of crossing direction.')

    y = np.asarray(y, dtype=float).ravel()
    n = y.size  # length of the time series

    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Could not get time delay (time series too short?)')
        return np.nan
    tau = int(tau)

    # Embed in three dimensions, and cut on the last coordinate at TISEAN's own
    # default threshold (that coordinate's mean). hctsa reads the .poin file
    # back, so the section points are the ones TISEAN printed.
    try:
        v = _tisean.poincare(y, dim=3, delay=tau, comp=3, direction=direction,
                             as_written=True)
    except ValueError as exc:  # e.g. a constant series: no section can be cut
        logger.warning(f'TISEAN poincare failed: {exc}')
        return np.nan

    # Columns are the two uncut embedding coordinates, followed by the
    # (interpolated) crossing time -- only the first two are point coordinates:
    v = v[:, :2]
    nn = v.shape[0]
    if nn < 2:
        logger.warning('No section points found to run poincare_section')
        return np.nan

    # Labeling poincare surface plane x-y
    x, yy = v[:, 0], v[:, 1]

    out = {}

    # Basic statistics:
    out['pcross'] = nn / n  # proportion of time series that crosses poincare surface

    for lab, u in (('x', x), ('y', yy)):
        q25, q75 = matlab_quantile(u, [0.25, 0.75])
        out[f'max{lab}'] = np.max(u)
        out[f'min{lab}'] = np.min(u)
        out[f'std{lab}'] = np.std(u, ddof=1)
        out[f'iqr{lab}'] = q75 - q25
        out[f'mean{lab}'] = np.mean(u)
        out[f'ac1{lab}'] = autocorr(u, 1, 'Fourier')[0]
        out[f'ac2{lab}'] = autocorr(u, 2, 'Fourier')[0]
        out[f'tauac{lab}'] = first_crossing(u, 'ac', 0, 'continuous')

    out['boxarea'] = np.ptp(x) * np.ptp(yy)

    # Statistics on distance between adjacent points, ds
    vdiff = np.diff(v, axis=0)
    ds = np.sqrt(vdiff[:, 0]**2 + vdiff[:, 1]**2)

    # Probability that next point in series is within radius r of current point
    # in the poincare section:
    out['pwithinr01'] = np.sum(ds < 0.1) / (nn - 1)
    out['pwithin02'] = np.sum(ds < 0.2) / (nn - 1)
    out['pwithin03'] = np.sum(ds < 0.3) / (nn - 1)
    out['pwithin05'] = np.sum(ds < 0.5) / (nn - 1)
    out['pwithin1'] = np.sum(ds < 1) / (nn - 1)
    out['pwithin2'] = np.sum(ds < 2) / (nn - 1)
    out['meands'] = np.mean(ds)
    out['maxds'] = np.max(ds)
    out['minds'] = np.min(ds)
    q25, q75 = matlab_quantile(ds, [0.25, 0.75])
    out['iqrds'] = q75 - q25

    # Now normalize both axes and look for structure in the cloud of points.
    # Don't normalize for standard deviation -- this probably reveals some
    # structure...? But location is already noted.
    x = x - np.mean(x)
    yy = yy - np.mean(yy)

    # Statistics on distance on Poincare surface from (mean,mean)
    d = np.sqrt(x**2 + yy**2)
    q25, q75 = matlab_quantile(d, [0.25, 0.75])
    out['maxD'] = np.max(d)
    out['minD'] = np.min(d)
    out['stdD'] = np.std(d, ddof=1)
    out['iqrD'] = q75 - q25
    out['meanD'] = np.mean(d)
    out['ac1D'] = autocorr(d, 1, 'Fourier')[0]
    out['ac2D'] = autocorr(d, 2, 'Fourier')[0]
    out['tauacD'] = first_crossing(d, 'ac', 0, 'continuous')

    # Statistics of the boxed distribution, with 5 and then 10 partitions per axis:
    for num_partitions in (5, 10):
        pbox = _count_boxes(x, yy, num_partitions) / nn
        pos = pbox[pbox > 0]

        out[f'maxpbox{num_partitions}'] = np.max(pbox)
        out[f'minpbox{num_partitions}'] = np.min(pbox)
        out[f'zerospbox{num_partitions}'] = np.sum(pbox == 0)
        out[f'meanpbox{num_partitions}'] = np.mean(pbox)
        out[f'rangepbox{num_partitions}'] = np.ptp(pbox)
        # Box-occupancy entropy, Miller-Madow corrected: the plug-in estimator
        # -sum(p log p) is biased low by (M-1)/(2n) for M occupied boxes and n
        # points on the section, so the raw value tracks the series length.
        out[f'hboxcounts{num_partitions}'] = (
            -np.sum(pos * np.log(pos)) + (pos.size - 1) / (2 * nn))
        out[f'tracepbox{num_partitions}'] = np.sum(np.diag(pbox))  # trace

    return out

def ssa(y: ArrayLike, L: Union[int, None] = None) -> dict:
    """
    Singular Spectrum Analysis of a time series.

    Constructs the trajectory (Hankel) matrix of the time series using a
    window length L (i.e., a time-delay embedding with delay tau = 1), and
    performs an uncentered singular value decomposition of the result.

    Unlike ``embed_pca`` (which centers the embedded data before decomposing it,
    and allows a general embedding delay), this implements classic "Basic SSA":
    a fixed delay of 1, no centering (so that a genuine trend is not removed
    before decomposition), and diagonal averaging ("Hankelization") of the
    leading elementary matrices back into component time series. Statistics
    are computed on the singular-value pairing structure, and on the
    reconstructed leading trend/oscillatory components themselves, rather than
    on the raw eigenvalue spectrum (which ``embed_pca`` already covers).

    References
    ----------
    .. [1] "Extracting qualitative dynamics from experimental data"
        D. S. Broomhead and G. P. King, Physica D 20(2-3) 217 (1986)
    .. [2] "Analysis of Time Series Structure: SSA and Related Techniques"
        N. Golyandina, V. Nekrutkin, A. Zhigljavsky, Chapman & Hall/CRC (2001)

    Parameters
    ----------
    y : array-like
        The input time series.
    L : int, optional
        The window length (default: floor(N/4)). Must satisfy
        ``4 <= L <= floor(N/2)``.

    Returns
    -------
    dict
        Statistics on the singular-value pairing/decay structure (gap1-gap5,
        sepIdx), and on the reconstructed leading block of components as a
        whole (trend strength trend_r2/trend_rho, dominant period of its most
        tightly-paired internal mode pairperiod, and w-correlation-based
        separability from the residual wcorr_leadresid). Individual
        eigentriples within a near-degenerate pair are not uniquely determined
        by the SVD, so all outputs are computed from basis-independent
        quantities: singular-value gaps, or sums of whole blocks of components
        rather than single components.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    if L is None:
        max_default_L = 200
        L = min(N//4, max_default_L)
    L = int(L)

    if L < 4 or L > N//2:
        logger.warning(f'Window length L = {L} is not suitable for a series '
                       f'of length {N}')
        return np.nan

    # Build the trajectory matrix (an embedding with delay tau = 1):
    try:
        X = time_delay_embed(y, L, 1)  # K x L, K = N - L + 1
    except ValueError:
        logger.warning('Could not construct a trajectory matrix for this time series')
        return np.nan

    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    d = len(sigma)

    if d < 4:
        logger.warning(f'Not enough singular values ({d}) obtained for this '
                       f'window length')
        return np.nan

    max_check = min(6, d-1)
    all_gaps = (sigma[:max_check] - sigma[1:max_check+1])/sigma[:max_check]
    out = {}
    for i in range(1, 6):
        out[f'gap{i}'] = all_gaps[i-1] if i <= max_check else np.nan

    #%% Leading structured block vs. residual
    # sepIdx locates the largest relative drop in the spectrum: components
    # 1:sepIdx are taken as the "structured" leading block, the rest as residual.
    sep_idx = int(np.argmax(all_gaps)) + 1
    out['sepIdx'] = sep_idx

    t = np.arange(1, N+1, dtype=float)
    # The number of matrix entries on each anti-diagonal (which is also the
    # w-correlation weight vector below):
    w = np.minimum(np.minimum(t, L), N - t + 1)

    def diagonal_average(components):
        """
        Reconstructs a length-N component series from an elementary matrix
        by averaging over its anti-diagonals ("Hankelization").
        """
        # The anti-diagonal sums of the rank-one matrix sigma_i*u_i*v_i' are the
        # (linear) convolution of u_i with v_i, so the elementary matrices never
        # have to be formed:
        diag_sums = np.zeros(len(w))
        for i in components:
            diag_sums += sigma[i]*np.convolve(U[:, i], V[:, i])
        return diag_sums/w

    c_lead = diagonal_average(range(sep_idx))
    resid = y - c_lead  # exact, since diagonal-averaging the full X recovers y

    #%% Trend diagnostics on the leading block
    out['trend_rho'] = spearmanr(c_lead, t).statistic

    A = np.column_stack((t, np.ones(N)))
    lin_fit = np.linalg.lstsq(A, c_lead, rcond=None)[0]
    c_leadhat = A @ lin_fit
    ss_res = np.sum((c_lead - c_leadhat)**2)
    ss_tot = np.sum((c_lead - np.mean(c_lead))**2)
    if ss_tot > 0:
        out['trend_r2'] = 1 - ss_res/ss_tot
    else:
        out['trend_r2'] = np.nan

    #%% Period of the most tightly-paired mode within the leading block
    if sep_idx >= 2:
        i_star = int(np.argmin(all_gaps[:sep_idx-1]))
        c_pair = diagonal_average((i_star, i_star+1))
        c_pair = c_pair - np.mean(c_pair)
        num_sign_changes = np.count_nonzero(np.diff(np.sign(c_pair)) != 0)
        if num_sign_changes > 0:
            out['pairperiod'] = 2*(N-1)/num_sign_changes
        else:
            out['pairperiod'] = np.nan
    else:
        out['pairperiod'] = np.nan

    out['wcorr_leadresid'] = (np.sum(w*c_lead*resid)
                              / np.sqrt(np.sum(w*c_lead**2)*np.sum(w*resid**2)))

    return out