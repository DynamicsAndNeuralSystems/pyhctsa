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

# ------------------------------------------------------------------------------
# Recurrence- and embedding-based operations (NL_RecurrenceTimes, NL_RQA, ...)
# ------------------------------------------------------------------------------
# (the imports for this block sit here, rather than at the top of the module, only to
# keep the block self-contained)
from numba import njit
from scipy.spatial.distance import pdist
from sklearn.neighbors import KDTree

from ..utils import _ml_randperm


@njit(cache=True)
def _fnn_nearest(s, delay, max_emb, theiler):
    """
    For each query point and each embedding dimension 1..max_emb, the nearest neighbor
    (max-norm, non-zero distance, outside the Theiler window) among the candidate points,
    as TISEAN's ``false_nearest`` finds it, and whether the minimum distance is shared by
    several candidates (a tie).
    """
    n_len = s.size
    n_query = n_len - max_emb * delay
    n_cand = n_len - (max_emb + 1) * delay
    best = np.full((n_query, max_emb), 1.1)
    which = np.full((n_query, max_emb), -1, dtype=np.int64)
    tied = np.zeros((n_query, max_emb), dtype=np.bool_)
    for n in range(n_query):
        for e in range(n_cand):
            if abs(e - n) <= theiler:
                continue
            mx = 0.0
            for d in range(max_emb):
                dx = abs(s[n + d] - s[e + d])
                if dx > mx:
                    mx = dx
                if mx > 0.0:
                    if mx < best[n, d]:
                        best[n, d] = mx
                        which[n, d] = e
                        tied[n, d] = False
                    elif mx == best[n, d]:
                        tied[n, d] = True
    return best, which, tied


def _fnn_break_ties(s, best, which, tied, delay, max_emb, theiler, eps_grid):
    """
    Among tied nearest neighbors, pick the one TISEAN's box search meets first: it scans the
    3x3 boxes (of side epsilon, in the first and the last coordinate) around the point in
    order, and each box's points from the latest to the earliest, keeping the first minimum.
    """
    n_cand = s.size - (max_emb + 1) * delay
    cand = np.arange(n_cand)
    for n, d in zip(*np.nonzero(tied)):
        mx = np.zeros(n_cand)
        for k in range(d + 1):
            mx = np.maximum(mx, np.abs(s[n + k] - s[cand + k]))
        ties = cand[(mx == best[n, d]) & (np.abs(cand - n) > theiler)]
        eps = eps_grid[min(np.searchsorted(eps_grid, best[n, d]), len(eps_grid) - 1)]
        cx, cy = (s[ties] / eps).astype(np.int64) & 1023, (s[ties + d] / eps).astype(np.int64) & 1023
        x, y = int(s[n] / eps) & 1023, int(s[n + d] / eps) & 1023
        da, db = (cx - x + 1) & 1023, (cy - y + 1) & 1023  # offset + 1, if within the 3x3 boxes
        visible = (da <= 2) & (db <= 2)
        if visible.any():
            ties, da, db = ties[visible], da[visible], db[visible]
            which[n, d] = ties[np.lexsort((-ties, db, da))[0]]


def _false_nearest(y: ArrayLike, delay: int = 1, max_dim: int = 10, theiler: int = 0,
                   escape_factor: float = 2.0) -> Union[dict, None]:
    """
    Fraction of false nearest neighbors by embedding dimension (TISEAN's ``false_nearest``,
    as called by hctsa's NL_FNN with ``-m1 -M1,max_dim``).

    A nearest neighbor (max-norm, outside the Theiler window) of an embedded point is false
    when, after adding the next coordinate, the distance to it grows by more than a factor
    ``escape_factor``. The series is rescaled to [0, 1] and, as in hctsa, written to TISEAN
    to 7 significant digits.

    Returns a dictionary of arrays over the embedding dimensions TISEAN reports (``dim``,
    ``pfnn``, ``nhood`` (mean size of the neighborhoods) and ``nhood_std``), or ``None`` when
    TISEAN gives no output (constant or too-short series, or no neighbor within range at
    the first dimension). TISEAN stops at the first dimension for which no neighbor is
    found, keeping the dimensions before it.
    """
    y = _tisean._round_significant(np.asarray(y, dtype=float).ravel(), 7)
    n_len = y.size
    delay, max_dim, theiler = int(delay), int(max_dim), int(theiler)
    if (max_dim + 1) * delay >= n_len:
        return None
    lo, hi = y.min(), y.max()
    interval = hi - lo
    if interval == 0:
        return None
    s = (y - lo) / interval
    varianz = np.sqrt(np.abs(np.mean(s * s) - np.mean(s) ** 2))

    best, which, tied = _fnn_nearest(s, delay, max_dim, theiler)
    # TISEAN's grid of neighborhood sizes: 1e-5, increased by sqrt(2) up to 2*varianz/escape_factor
    eps_grid = [1e-5]
    while eps_grid[-1] < 2 * varianz / escape_factor:
        eps_grid.append(eps_grid[-1] * np.sqrt(2.0))
    _fnn_break_ties(s, best, which, tied, delay, max_dim, theiler, np.array(eps_grid))
    rows = {'dim': [], 'pfnn': [], 'nhood': [], 'nhood_std': []}
    for emb in range(1, max_dim + 1):
        mindx, nbr = best[:, emb - 1], which[:, emb - 1]
        found = (nbr >= 0) & (mindx <= varianz / escape_factor)
        n_found = int(found.sum())
        if n_found == 0:
            break  # TISEAN: "Not enough points found!"
        q = np.flatnonzero(found)
        factor = np.abs(s[q + emb] - s[nbr[q] + emb]) / mindx[q]
        rows['dim'].append(emb)
        rows['pfnn'].append(_tisean._e(np.count_nonzero(factor > escape_factor) / n_found))
        rows['nhood'].append(_tisean._e(np.mean(mindx[q]) * interval))
        rows['nhood_std'].append(_tisean._e(np.sqrt(np.mean(mindx[q] ** 2)) * interval))
    if not rows['dim']:
        return None
    return {k: np.array(v) for k, v in rows.items()}


def _fnn_embedding_dim(y: np.ndarray, tau: int, threshold: float = 0.4) -> Union[int, float]:
    """
    Embedding dimension by false nearest neighbors, as hctsa's ``BF_Embed(y, tau, 'fnn')``:
    the first dimension (of 1 to 10) at which the fraction of false nearest neighbors falls
    below `threshold` (TISEAN's ``false_nearest`` with a Theiler window of one
    autocorrelation time and an escape factor of 5), or one more than the largest
    dimension TISEAN reports if it never does. NaN when it cannot be determined.
    """
    if y.size < 10:
        logger.warning(f'Time series (N={y.size}) too short for fnn')
        return np.nan
    theiler = theiler_window(y, ('ac', 1), y.size)
    if np.isnan(theiler):
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    res = _false_nearest(y, tau, 10, int(theiler), 5.0)
    if res is None:
        logger.warning('TISEAN false_nearest produced no usable output for this data')
        return np.nan
    below = np.flatnonzero(res['pfnn'] < threshold)
    return int(res['dim'][below[0]]) if below.size else int(res['dim'][-1]) + 1


def _embedding_params(y: np.ndarray, tau: Union[int, str], m: Union[int, str, list, tuple]
                      ) -> Union[tuple, None]:
    """
    The time delay and embedding dimension, as hctsa's ``BF_Embed(y, tau, m, true)``.

    `tau` is an integer or a rule understood by :func:`pyhctsa.utils.get_tau`. `m` is an
    integer, ``'fnn'`` (false nearest neighbors, threshold 0.4), or ``('fnn', threshold)``.
    Returns ``(tau, m)``, or None if either cannot be determined.
    """
    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Could not determine the time delay for the embedding')
        return None
    tau = int(tau)
    if isinstance(m, (list, tuple)):
        m = m[0] if len(m) == 1 else m
    if isinstance(m, (list, tuple)) or isinstance(m, str):
        if (m if isinstance(m, str) else m[0]) != 'fnn':
            raise ValueError(f"Embedding dimension, m, incorrectly specified: {m!r}")
        m = _fnn_embedding_dim(y, tau, 0.4 if isinstance(m, str) else m[1])
        if np.isnan(m):
            return None
    return tau, int(m)


def _bf_embed(y: np.ndarray, tau: Union[int, str], m: Union[int, str, list, tuple]
              ) -> Union[np.ndarray, None]:
    """
    Time-delay embedding with hctsa's ``BF_Embed(y, tau, m, false)``: the embedded points as
    rows (see :func:`_embedding_params` for `tau` and `m`), or None when it fails
    (undetermined parameters, or a time series too short).
    """
    params = _embedding_params(y, tau, m)
    if params is None:
        return None
    try:
        return time_delay_embed(y, params[1], params[0])
    except ValueError as e:
        logger.warning(str(e))
        return None


def _random_subset(n: int, k: int, random_seed: Union[int, str, None]) -> np.ndarray:
    """
    ``k`` of ``n`` indices (from zero) in random order, from the Mersenne Twister seeded as
    hctsa's ``BF_ResetSeed`` (an integer seed, ``'default'`` for seed 0, or ``None``/``'none'``
    for an unseeded stream).
    """
    if random_seed is None or random_seed == 'none':
        rng = np.random.RandomState()
    else:
        rng = _ml_rng(0 if random_seed == 'default' else int(random_seed))
    return _ml_randperm(n, rng)[:k] - 1


def _recurrence_radius(Y: np.ndarray, rr: float, random_seed: Union[int, str, None]) -> float:
    """
    Neighborhood radius giving the target recurrence rate `rr`: its quantile of the pairwise
    distances between (at most) 500 randomly chosen embedded points.
    """
    n_emb = Y.shape[0]
    sub = _random_subset(n_emb, min(500, n_emb), random_seed)
    return float(matlab_quantile(pdist(Y[sub]), rr)[0])


def _recurrent_pairs(Y: np.ndarray, radius: float) -> tuple:
    """All ordered pairs (src, dst) of embedded points within `radius` (Euclidean) of each other."""
    # (the tree search is padded, then squared distances compared with the squared radius, as
    # MATLAB's rangesearch does, so that pairs lying exactly at the radius -- common for
    # quantized data -- are counted consistently)
    nbrs = KDTree(Y).query_radius(Y, radius * (1 + 1e-9))
    src = np.repeat(np.arange(Y.shape[0]), [nb.size for nb in nbrs])
    dst = np.concatenate(nbrs)
    keep = np.sum((Y[src] - Y[dst]) ** 2, axis=1) <= radius ** 2
    return src[keep], dst[keep]


def _check_max_n(y: np.ndarray, max_n: Union[int, str], what: str) -> np.ndarray:
    """Crop the series to its first `max_n` samples (``'full'`` for no cropping)."""
    if isinstance(max_n, str):
        if max_n != 'full':
            raise ValueError(f"max_n must be an integer or 'full', got '{max_n}'")
    elif y.size > max_n:
        logger.warning(f'Time series ({y.size} > {max_n}) is too long for {what}. '
                       f'Analyzing the first {int(max_n)} samples')
        y = y[:int(max_n)]
    return y


def _recurrence_time_stats(Y: np.ndarray, radius: float, theiler: int) -> tuple:
    """
    Mean recurrence time and modal probability mass of the white vertical line lengths of
    the recurrence plot of `Y` (hctsa's ``SUB_recurrenceTimeStats``).

    For each point, the number of non-recurrent points between successive recurrent points
    (neighbors within `radius`, outside the Theiler window; the edges of the Theiler band
    count as recurrent, and the neighbors before and after the point are differenced
    separately) are pooled over all points.
    """
    n = Y.shape[0]
    src, dst = _recurrent_pairs(Y, radius)
    keep = np.abs(dst - src) > theiler  # excludes the Theiler window (and the point itself)
    src, dst = src[keep], dst[keep]
    j = np.arange(n)
    # the band edges act as recurrent points, so that lines start at the edge of the band
    left, right = j[j - theiler >= 0], j[j + theiler <= n - 1]
    before = np.concatenate([np.column_stack((src[dst < src], dst[dst < src])),
                             np.column_stack((left, left - theiler))])
    after = np.concatenate([np.column_stack((src[dst > src], dst[dst > src])),
                            np.column_stack((right, right + theiler))])
    w = []
    for pairs in (before, after):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        same = pairs[1:, 0] == pairs[:-1, 0]
        w.append(np.diff(pairs[:, 1])[same] - 1)
    w = np.concatenate(w)
    w = w[w >= 1]  # (drops the zero-length "lines" between consecutive recurrent points)
    if w.size == 0:
        return np.nan, np.nan
    return float(np.mean(w)), float(np.bincount(w).max() / w.size)


def recurrence_times(y: ArrayLike, tau: Union[int, str] = 1, m: Union[int, str, list, tuple] = 3,
                     theiler_win: Union[int, float, list, tuple] = ('ac', 1), rr: float = 0.1,
                     num_segments: int = 4, max_n: Union[int, str] = 10000,
                     random_seed: Union[int, str, None] = 'default') -> dict:
    """
    Recurrence-time statistics from a recurrence plot.

    Embeds the series in a time-delay space and finds, for each embedded point, the times
    at which the trajectory returns to its neighborhood: the lengths of the white vertical
    lines of the recurrence plot (the numbers of non-recurrent points between successive
    recurrent points), cf. [1]. This is the distribution of the times between recurrences
    to a given neighborhood, rather than the black line-length statistics of
    :func:`rqa`. Quasi-periodic dynamics on a torus return at a few distinct times, so the
    distribution of recurrence times has a few sharp peaks, whereas strange nonchaotic
    attractors have a more broadly distributed (and segment-to-segment more variable)
    set of recurrence times.

    The neighborhood radius is set once, from the full embedded series, to the `rr`-quantile
    of a random subsample of pairwise distances; the same radius is then reused for every
    segment, so that segment-to-segment differences reflect the dynamics rather than a
    re-calibrated threshold. The Theiler band around each point counts as recurrent, so
    that no white line spans the excluded band.

    References
    ----------
    .. [1] Ngamga, E.J., Nandi, A., Ramaswamy, R., Romano, M.C., Thiel, M. and Kurths, J.
        "Recurrence-time distributions in strange nonchaotic systems", Phys. Rev. E
        75, 036222 (2007).

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time delay for the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'`` or ``'mi'``). Default is 1.
    m : int, str or tuple, optional
        The embedding dimension: an integer, or ``'fnn'`` to choose it by false nearest
        neighbors (TISEAN's ``false_nearest``, as hctsa's ``BF_Embed``). Default is 3.
    theiler_win : int, float or ``['ac', k]``, optional
        The Theiler window excluding temporally-correlated neighbors (see
        :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k`` times the first
        zero-crossing of the autocorrelation function, or a number of samples. Narrowed
        to ``Nemb // 5`` for short series. Default is ``['ac', 1]``.
    rr : float, optional
        The target recurrence rate used to set the neighborhood radius. Default is 0.1.
    num_segments : int, optional
        The embedded trajectory is divided into this many contiguous, non-overlapping
        segments, and the mean recurrence time and modal probability are recomputed
        independently within each; their variance across segments is the paper's diagnostic
        for the torus-to-SNA transition. Each segment needs at least 50 embedded points,
        otherwise the variance outputs (but not the full-series ``T_MRT``/``N_MPRT``) are
        NaN. Default is 4.
    max_n : int or 'full', optional
        The maximum number of samples to consider (the first ``max_n``); ``'full'`` to
        disable cropping. Default is 10000.
    random_seed : int, str or None, optional
        The seed of the Mersenne Twister for the random subsample used to set the radius, as
        hctsa's ``BF_ResetSeed``: an integer, ``'default'`` (seed 0), or ``None``/``'none'``
        (unseeded). The radius is the same as hctsa's only when there are at most 500 embedded
        points (the subsample is then the whole series): MATLAB's ``randperm(n, k)`` draws a
        different random subset from the same seed. Default is ``'default'``.

    Returns
    -------
    dict or float
        NaN if the embedding or Theiler window cannot be determined, or the embedded series is
        too short (under 50 points) or degenerate (zero radius). Otherwise:

        - ``T_MRT``: the mean recurrence time of the full series (the mean white-line length)
        - ``N_MPRT``: the modal recurrence-time probability mass of the full series: the
          fraction of all recurrence-time samples taking the single most common value (the
          paper's raw count, normalized so that it does not scale with series length)
        - ``T_MRT_var``, ``N_MPRT_var``: the variance of ``T_MRT`` and of ``N_MPRT`` across
          the ``num_segments`` segments
    """
    y = _check_max_n(np.asarray(y, dtype=float).ravel(), max_n, 'recurrence-time analysis')

    Y = _bf_embed(y, tau, m)
    if Y is None:
        logger.warning('Embedding failed')
        return np.nan
    n_emb = Y.shape[0]

    theiler = theiler_window(y, theiler_win, n_emb)
    if np.isnan(theiler):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    theiler = min(int(theiler), n_emb // 5)  # narrowed for short series
    if n_emb < 50:
        logger.warning(f'Time series too short for meaningful recurrence-time statistics '
                       f'(Nemb = {n_emb}, theilerWin = {theiler})')
        return np.nan

    radius = _recurrence_radius(Y, rr, random_seed)
    if not radius > 0:
        logger.warning('Degenerate neighborhood radius (data may be too degenerate/short)')
        return np.nan

    out = {}
    out['T_MRT'], out['N_MPRT'] = _recurrence_time_stats(Y, radius, theiler)
    out['T_MRT_var'] = out['N_MPRT_var'] = np.nan
    if np.isnan(out['T_MRT']):
        logger.warning('No recurrence-time samples found outside the Theiler window -- radius too small?')
        out['N_MPRT'] = np.nan
        return out

    # Variance across independent, contiguous segments (with the same global radius)
    seg_len = n_emb // num_segments
    if seg_len < 50:  # too short for a meaningful within-segment estimate
        return out
    theiler_seg = min(theiler, seg_len // 5)
    stats = np.array([_recurrence_time_stats(Y[s * seg_len:(s + 1) * seg_len], radius, theiler_seg)
                      for s in range(num_segments)])
    if not np.any(np.isnan(stats[:, 0])):
        out['T_MRT_var'] = float(np.var(stats[:, 0], ddof=1))
        out['N_MPRT_var'] = float(np.var(stats[:, 1], ddof=1))
    return out
