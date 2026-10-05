import logging
import math
import warnings
from typing import Union

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, find_peaks, welch as _welch
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma, gammaln
from scipy.stats import norm as _norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, NearestNeighbors

from ..operations.model_fit import residual_analysis
from ..operations.correlation import first_crossing, first_min, autocorr
from ..toolboxes.matlab.matlab_fit import goodness_of_fit, lsqcurvefit_trr, robustfit
from ..toolboxes.Tisean_3_0_1 import tisean as _tisean
from ..toolboxes.Tisean_3_0_1.tisean import _e, _round_significant
from ..robust import bf_hist_edges, bf_random
from ..utils import (_linspace, _ml_randperm, _ml_rng, _round_half_away, bin_picker, get_tau,
                     matlab_quantile, theiler_window, time_delay_embed)

logger = logging.getLogger('pyhctsa')


def _bf_random_seed(random_seed: Union[int, float, str, None]) -> int:
    """The integer seed for :func:`~pyhctsa.robust.bf_random` that a ``randomSeed`` input stands
    for (hctsa's ``BF_RandomSeed``): ``'default'`` is 0, a number is itself (rounded, made
    non-negative, below 4e9), and ``None`` or ``'none'`` a seed drawn from NumPy's global stream."""
    if random_seed is None or (isinstance(random_seed, str) and random_seed == 'none'):
        return int(np.random.randint(0, 4_000_000_000))
    if isinstance(random_seed, str):
        if random_seed != 'default':
            raise ValueError(f"Not sure how to interpret the random seed '{random_seed}'")
        return 0
    return int(np.mod(_round_half_away(abs(float(random_seed))), 4e9))

# ------------------------------------------------------------------------------
# Embedding parameters (hctsa's NL_FNN and BF_Embed), shared by the operations below
# ------------------------------------------------------------------------------

def fnn(y: ArrayLike, tau: Union[int, str] = 1, maxm: int = 10,
        theiler_win: Union[int, float, list, tuple] = ('ac', 1), just_best: bool = True,
        bestp: float = 0.4, escape_factor: Union[float, None] = None) -> Union[dict, float]:
    """
    How the fraction of false nearest neighbors falls as the embedding dimension increases.

    Uses TISEAN's ``false_nearest`` (run in process; see
    :func:`pyhctsa.toolboxes.Tisean_3_0_1.tisean.false_nearest`). For each embedding
    dimension ``m = 1, ..., maxm``, every point's nearest neighbor (maximum norm, outside
    the Theiler window in time) is found; the neighbor is false if, when one more
    coordinate is added, the two points move apart by more than ``escape_factor`` times
    their distance. For a deterministic system the fraction of false neighbors falls
    toward zero once the embedding dimension is large enough; for noise it stays high.
    Neighbors farther apart than the standard deviation of the data divided by the
    escape factor are skipped.

    Note: as in TISEAN 3.0.1 for a scalar series, the delay only limits the number of
    points used; the embedding coordinates are consecutive samples.

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        The time delay: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.
    maxm : int, optional
        The maximum embedding dimension. Default is 10.
    theiler_win : int, float, or ``['ac', k]``, optional
        The Theiler window (see :func:`pyhctsa.utils.theiler_window`). Default is
        ``['ac', 1]``.
    just_best : bool, optional
        If True, return only the first dimension at which the fraction of false nearest
        neighbors is below ``bestp``. Default is True.
    bestp : float, optional
        The fraction of false neighbors defining the embedding dimension when
        ``just_best``. Default is 0.4.
    escape_factor : float, optional
        TISEAN's ``-f``: a neighbor is false if its distance grows by more than this
        factor when a dimension is added. ``None`` (the default) is TISEAN's own default,
        2. (``BF_Embed``'s false-nearest-neighbors choice of dimension uses 5.)

    Returns
    -------
    float or dict
        If ``just_best``, the embedding dimension (``maxm + 1`` if the fraction never
        falls below ``bestp``). Otherwise a dict with, for ``i = 1, ..., maxm``, ``pfnn_i``
        (the fraction of false nearest neighbors) and ``nHood2_i`` (the typical,
        root-mean-square, distance from a point to its nearest neighbor; NaN at a
        dimension for which TISEAN found too few neighbors); ``minpfnn``, ``meanpfnn``,
        ``stdpfnn``, ``maxnHood2``, ``meannHood2``; ``firstunder09`` ... ``firstunder005``
        (the first dimension at which the fraction is below 90%, 80%, ..., 10%, 5%; ``maxm
        + 1`` if never); ``max1stepchange`` (the largest absolute change in the fraction
        between consecutive dimensions), ``mdrop`` (the mean change per added dimension)
        and ``pdrop`` (the fraction of decreasing steps minus the fraction of increasing
        steps). NaN if the series is too short (N < 10), the time delay or Theiler window
        cannot be set, or TISEAN gives no output (constant series, or a delay too long for
        the series).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 10:
        logger.warning(f'Time series (N={n}) too short for fnn')
        return np.nan
    tau = get_tau(y, tau)
    if np.isnan(tau):
        return np.nan
    theiler = theiler_window(y, theiler_win, n)
    if np.isnan(theiler):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    maxm = int(maxm)
    # (-f%g: six significant digits)
    escape = 2.0 if escape_factor is None else float('%g' % escape_factor)
    try:
        res = _tisean.false_nearest(y, int(tau), 1, maxm, int(theiler), escape)
    except ValueError as exc:  # no output, e.g. delay times dimension too large for the series
        logger.warning(f'No output from TISEAN routine false_nearest on the data: {exc}')
        return np.nan
    # (TISEAN prints these with %e, and hctsa reads that text)
    m_dim, p_nn, n_hood2 = res['dim'], np.vectorize(_tisean._e)(res['pfnn']), np.vectorize(_tisean._e)(res['sdeps'])

    if just_best:
        return float(_tisean.fnn_first_under(m_dim, p_nn, bestp))

    out = {}
    for i in range(1, maxm + 1):
        found = i <= m_dim.size
        out[f'pfnn_{i}'] = p_nn[i - 1] if found else np.nan
        out[f'nHood2_{i}'] = n_hood2[i - 1] if found else np.nan
    out['minpfnn'] = np.min(p_nn)
    out['meanpfnn'] = np.mean(p_nn)
    out['stdpfnn'] = np.std(p_nn, ddof=1) if p_nn.size > 1 else 0.0
    out['maxnHood2'] = np.max(n_hood2)
    out['meannHood2'] = np.mean(n_hood2)
    for name, th in (('09', 0.9), ('08', 0.8), ('07', 0.7), ('06', 0.6), ('05', 0.5),
                     ('04', 0.4), ('03', 0.3), ('02', 0.2), ('01', 0.1), ('005', 0.05)):
        out[f'firstunder{name}'] = float(_tisean.fnn_first_under(m_dim, p_nn, th))
    steps = np.diff(p_nn)
    out['max1stepchange'] = np.max(np.abs(steps)) if steps.size else np.nan
    out['mdrop'] = np.mean(steps) if steps.size else np.nan
    out['pdrop'] = -np.mean(np.sign(steps)) if steps.size else np.nan
    return out


def _fnn_embedding_dim(y: np.ndarray, tau: int, threshold: float = 0.4) -> float:
    """
    Embedding dimension by false nearest neighbors, as hctsa's ``BF_Embed(y, tau, {'fnn',
    threshold})``: :func:`fnn` with ten dimensions, a Theiler window of one autocorrelation
    time and an escape factor of 5 (TISEAN's own, 2, makes the test look as if it never
    converges), returning the first dimension at which the fraction of false nearest
    neighbors falls below `threshold`. NaN when it cannot be determined.
    """
    return fnn(y, tau, 10, ('ac', 1), True, threshold, 5)


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


def _embed_tau_m(y: np.ndarray, embed_params) -> tuple:
    """
    The delay and dimension of ``[tau, m]`` embedding parameters (hctsa's
    ``BF_Embed(y, tau, m, true)``), without doing the embedding: see
    :func:`_embedding_params`. Returns ``(nan, nan)`` if they cannot be determined.
    """
    if not isinstance(embed_params, (list, tuple)) or len(embed_params) != 2:
        raise ValueError('Embedding parameters are formatted incorrectly -- need [tau, m]')
    params = _embedding_params(y, embed_params[0], embed_params[1])
    return (np.nan, np.nan) if params is None else params


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
        The embedding dimension, or ``'fnn'`` to set it by TISEAN's false nearest
        neighbors (:func:`fnn`: first dimension at which the fraction of false
        neighbors is below 0.05, with a Theiler window of one autocorrelation time
        and an escape factor of 5). Default is 3.
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
        if de != 'fnn':
            raise ValueError(f"Invalid embedding dimension '{de}'")
        de = _fnn_embedding_dim(y, tau, 0.05)
        if np.isnan(de):  # the dimension could not be determined (e.g. a delay too long)
            return np.nan
    de = int(de)

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
    random_seed : int, str or None, optional
        Seed of the random reference points (see :func:`_bf_random_seed`; ``'default'`` is 0).
        They come from the portable generator :func:`~pyhctsa.robust.bf_random`, so the
        results are the same as hctsa's for the same seed. Default is 0.

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

    # Random numbers for the reference points: the next unused number of one reproducible stream
    # (more of the same stream is generated if the numbers run out)
    seed = _bf_random_seed(random_seed)
    rand_stream = bf_random(512, seed)
    num_used = 0

    err = np.zeros(max_delay + 1)
    for _ in range(iterations):
        # Redraw until the reference point has a value-neighbor on both sides
        # that clears the Theiler window (as hctsa, give up with NaN after
        # max_attempts draws, which is data dependent).
        for _ in range(max_attempts):
            if num_used >= len(rand_stream):
                rand_stream = bf_random(4 * len(rand_stream), seed)  # same stream, longer
            ref = int(np.ceil(rand_stream[num_used]*length))  # a random value-rank (from one)
            num_used += 1
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
    m : int or 'fnn', optional
        The embedding dimension: an integer, or ``'fnn'`` (or ``['fnn', threshold]``)
        to set it by TISEAN's false nearest neighbors (:func:`fnn`, threshold 0.4 by
        default; as hctsa's ``BF_Embed``). Default is 2.

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

    y_embed = _bf_embed(y, tau, m)
    if y_embed is None:  # the delay or dimension could not be set, or the series is too short
        logger.warning('Embedding failed')
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
                     endptr: np.ndarray, normalize: bool = True) -> tuple:
    """Scaling range over which ``v`` is most nearly constant.

    Rescales ``v`` to [0,1] (unless ``normalize`` is False) so the comparison is independent of
    its range, then scores each candidate range by the spread of ``v`` across it, less a bonus
    (``gamma``) per additional point spanned. Returns the winning
    ``(start index, end index, score)`` into ``stptr``/``endptr``.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        vnorm = (v - v.min()) / (v.max() - v.min()) if normalize else v
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
        (no valid output for a long delay, no correlation-dimension data, or no scaling
        range).
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
        integer, or ``'fnn'`` (or ``['fnn', threshold]``) for the dimension from
        TISEAN's false nearest neighbors (:func:`fnn`; threshold 0.4 by default). Default is ``['ac', 'fnn']``.
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
        integer, or ``'fnn'`` (or ``['fnn', threshold]``) for the dimension from
        TISEAN's false nearest neighbors (:func:`fnn`; threshold 0.4 by default). Default is ``['ac', 'fnn']``.

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
        or ``'fnn'`` (or ``['fnn', threshold]``) for the dimension from
        TISEAN's false nearest neighbors (:func:`fnn`; threshold 0.4 by default). Default is ``['ac', 'fnn']``.
    random_seed : int, str or None, optional
        Seed for choosing the random subsample of reference points (relevant when
        ``nref != -1``; see :func:`_bf_random_seed`). The numbers come from the portable generator
        :func:`~pyhctsa.robust.bf_random`, so the subsample is the same as hctsa's. Default is 0.

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
        ref_idx = bf_random(n_emb, _bf_random_seed(random_seed), 'perm')[:int(nref)] - 1  # random subsample

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




def _tisean_boxcount(y: np.ndarray, delay: int, maxembed: int, epscount: int, q: float = 0.0) -> tuple:
    # TISEAN's ``boxcount -M1,<maxembed> -d<delay> -Q<q> -#<epscount>`` (source_c/boxcount.c), in
    # process: the Renyi entropy of order q, H_q = log(sum_i p_i^q) / (1 - q) (Shannon, -sum p log p,
    # for q = 1), of the partition of the delay embedding into cubes of side eps -- p_i being the
    # fraction of embedded points in cube i -- for embedding dimensions 1..maxembed. For q = 0
    # that is ln N(eps), the log of the number of occupied cubes.
    # Returns (eps, H, dH): eps of shape (epscount,), and H and dH, of shape (epscount, maxembed),
    # the entropy and its increment with the embedding dimension (H itself for dimension 1), all
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
    h = np.empty((epscount, maxembed))
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
            _, labels, counts = np.unique(labels * epsi + box, return_inverse=True, return_counts=True)
            p = counts / length
            h[k, d] = -np.sum(p * np.log(p)) if q == 1.0 else np.log(np.sum(p ** q)) / (1.0 - q)
    dh = np.diff(h, axis=1, prepend=0.0)
    return np.array([_e(v) for v in eps]), np.vectorize(_e)(h), np.vectorize(_e)(dh)


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
        ``'fnn'`` (or ``['fnn', threshold]``) for the dimension from
        TISEAN's false nearest neighbors (:func:`fnn`; threshold 0.4 by default). Default is ``['ac', 'fnn']``.

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
        bc_r, bc_logn, _ = _tisean_boxcount(y, tau, big_m, num_bins)
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


def _tisean_lyap_r(y: np.ndarray, delay: int, dim: int, mindist: int, steps: int) -> Union[tuple, None]:
    # TISEAN's ``lyap_r -d<delay> -m<dim> -t<mindist> -s<steps>`` (Rosenstein et al.; source_c/lyap_r.c
    # as modified in hctsa), in process. For every reference vector, the nearest neighbor (outside
    # the Theiler window ``mindist``) is found at the first length scale of the geometric sequence
    # 1e-3 * 1.1^k (of the data interval) at which any neighbor lies within it, and the logarithm
    # of the distance between the two trajectories is averaged over all references at each of
    # ``steps`` + 1 times. Returns the (time, mean ln distance) rows as in the .ros file, or None
    # where TISEAN gives no output (a reference point has no neighbor within the search radius:
    # exit status 54).
    y = _round_significant(np.asarray(y, dtype=float).ravel(), 7)  # BF_WriteTempFile
    length = y.size
    ymin = np.min(y)
    interval = np.max(y) - ymin
    if interval == 0:
        raise _D2DataError('lyap_r: the data are constant')
    x = (y - ymin) / interval  # rescaled to [0, 1]
    delay_span = delay * (dim - 1)
    maxlength = length - delay_span - steps - 1 - mindist
    if maxlength < 0:
        raise _D2DataError('lyap_r: time series too short')
    n_cand = length - delay_span - steps  # reference vectors that can be neighbors
    emb = x[np.arange(length - delay_span)[:, None] + delay * np.arange(dim)[None, :]]
    refs = np.arange(maxlength + 1)

    # Length-scale sequence over which neighbors are looked for
    eps_seq = []
    eps = 1e-3
    while eps <= 2.0:
        eps_seq.append(eps)
        eps *= 1.1
    eps_sq = np.square(eps_seq)

    # Nearest neighbors, in chunks of reference points
    neighbor = np.full(refs.size, -1)
    cand = np.arange(n_cand)
    chunk = max(1, int(2e6 // (n_cand * dim)))
    for c in range(0, refs.size, chunk):
        r = refs[c:c + chunk]
        d2 = np.zeros((r.size, n_cand))
        for k in range(dim):
            d2 += np.square(emb[r, k][:, None] - emb[:n_cand, k][None, :])
        d2[np.abs(r[:, None] - cand[None, :]) <= mindist] = np.inf
        d2[d2 >= 1.0] = np.inf  # only neighbors within squared distance 1 are accepted
        dmin = d2.min(axis=1)
        if np.any(dmin > eps_sq[-1]):
            return None  # some reference point never gets a neighbor
        # the neighbor is the nearest one with a positive distance, if within the first scale
        # at which anything (even an exact duplicate) was found
        k_found = np.searchsorted(eps_sq, dmin, side='left')
        d2[d2 == 0] = np.inf
        jmin = d2.argmin(axis=1)
        dnear = d2[np.arange(r.size), jmin]
        ok = dnear <= eps_sq[k_found]
        # Exactly tied neighbors (quantized data): lyap_r takes the first met scanning its
        # 256 x 256 grid of boxes (cells of side eps in the first and last coordinates, rows
        # then columns, newest point first within a cell)
        for ri in np.flatnonzero(ok & (np.sum(d2 == dnear[:, None], axis=1) > 1)):
            tied = np.flatnonzero(d2[ri] == dnear[ri])
            epsinv = 1.0 / eps_seq[k_found[ri]]
            cell = [(emb[:, k] * epsinv).astype(int) & 255 for k in (0, dim - 1)]
            rank = [(cell[k][tied] - cell[k][r[ri]] + 128) % 256 - 128 for k in (0, 1)]
            jmin[ri] = tied[np.lexsort((-tied, rank[1], rank[0]))[0]]
        neighbor[c:c + chunk] = np.where(ok, jmin, -1)

    # Divergence of the neighbor pairs over time
    found = np.zeros(steps + 1)
    lyap = np.zeros(steps + 1)
    use = refs[neighbor >= 0]
    nb = neighbor[neighbor >= 0]
    chunk = max(1, int(2e6 // ((steps + 1) * dim)))
    steps_idx = np.arange(steps + 1)
    for c in range(0, use.size, chunk):
        a = use[c:c + chunk, None] + steps_idx[None, :]
        b = nb[c:c + chunk, None] + steps_idx[None, :]
        dx = np.sum(np.square(emb[a] - emb[b]), axis=2)
        pos = dx > 0
        found += pos.sum(axis=0)
        lyap += np.sum(np.log(np.where(pos, dx, 1.0)), axis=0)
    have = found > 0
    if not have.any():
        return np.empty((0, 2))
    return np.column_stack([steps_idx[have], np.vectorize(_e)(lyap[have] / found[have] / 2.0)])


def largest_lyap(y: ArrayLike, nref: Union[int, float] = -1,
                 maxtstep: Union[int, float, list, tuple] = ('ac1e', 30),
                 past: Union[int, float, list, tuple] = ('ac1e', 1), nnr: int = 3,
                 embed_params: Union[list, tuple] = ('ac', 'fnn')) -> Union[dict, float]:
    """
    Largest Lyapunov exponent: how nearest-neighbor trajectories diverge, and the scaling
    of that divergence.

    Estimates the divergence of nearby trajectories of the delay embedding as a function of
    time by the algorithm of Rosenstein et al. [1]_ (TISEAN's ``lyap_r``): the logarithm of
    the distance between each reference point's trajectory and that of its nearest neighbor
    (outside a Theiler window) averaged over the reference points, :math:`p(t)`, for ``t`` up
    to ``maxtstep``. For a chaotic system, :math:`p` rises linearly before saturating, with
    slope the largest Lyapunov exponent. Outputs summarize :math:`p` (its first values,
    maximum, crossings of fractions of the maximum, times to reach them), straight-line fits to
    its scaling region (varying both start and end times, ``vse``, or just the end time,
    ``ve``, chosen to minimize the mean absolute residual less 0.006 per point), those slopes
    per autocorrelation time, and a saturating exponential fit
    :math:`a (1 - e^{bt})`.

    Unlike hctsa, which shells out to installed TISEAN binaries, this runs an in-process port
    of ``lyap_r``.

    References
    ----------
    .. [1] M. T. Rosenstein, J. J. Collins, and C. J. De Luca, "A practical method for
        calculating largest Lyapunov exponents from small data sets", Physica D 65(1-2),
        117-134 (1993).

    Parameters
    ----------
    y : array-like
        Input time series.
    nref : int or float, optional
        Unused (kept for compatibility with hctsa's signature). Default is -1.
    maxtstep : int, float, or ``['ac1e', k]``, optional
        The maximum number of time steps to follow neighbors for: a number of steps, a
        value in (0, 1) as a proportion of the time-series length, or ``['ac1e', k]`` (or
        ``['ac', k]``) for ``k`` times the autocorrelation time (see
        :func:`pyhctsa.utils.theiler_window`). At least 10 and at most half the series length;
        an ``'ac1e'`` horizon too long for the series is shortened to fit (NaN if this leaves fewer
        than ``max(10, 3 * ac1e time)`` steps). Default is ``['ac1e', 30]``.
    past : int, float, or ``['ac1e', k]``, optional
        The Theiler window of time-correlated points to exclude as neighbors (see
        :func:`pyhctsa.utils.theiler_window`). Default is ``['ac1e', 1]``.
    nnr : int, optional
        Unused (kept for compatibility with hctsa's signature). Default is 3.
    embed_params : [tau, m], optional
        Embedding parameters: ``tau`` is an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), ``m`` an integer, or
        ``'fnn'`` (or ``['fnn', threshold]``) for the dimension from
        TISEAN's false nearest neighbors (:func:`fnn`; threshold 0.4 by default). Default is ``['ac', 'fnn']``.

    Returns
    -------
    dict or float
        ``p1`` to ``p5``, ``maxp``: the first five values and the maximum of :math:`p(t)`
        (relative to :math:`p(0)`); ``ncross08max``, ``pcross08max``, ``ncross09max``,
        ``pcross09max``: the number (and proportion) of times :math:`p` crosses 80% and 90% of its
        maximum; ``to095max`` ... ``to05max``: the time to first exceed 95%, 90%, 80%, 70%, 50%
        of the maximum; ``vse_*`` and ``ve_*`` (``meanabsres``, ``rmsres``, ``gradient``,
        ``intercept``, ``minbad``, and ``gradient_pertau``, the gradient per autocorrelation
        time): the straight-line fits (NaN if the rise to 95% of the maximum takes at most three
        steps); ``expfit_a``, ``expfit_b``, ``expfit_r2``, ``expfit_adjr2``, ``expfit_rmse``: the
        saturating exponential fit. Returns NaN if the delay, horizon, or Theiler window cannot be
        set, the series is too short for the horizon, TISEAN finds no neighbor for some reference
        point (data too sparse or quantized) or too little output, or the curve is flat.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    # Maximum time step
    cap_horizon = (isinstance(maxtstep, (list, tuple)) and len(maxtstep) == 2
                   and maxtstep[0] == 'ac1e')  # cap (rather than NaN) a horizon too long for the series
    if isinstance(maxtstep, (list, tuple)):  # a multiple of the autocorrelation time
        maxtstep = theiler_window(y, maxtstep)
        if np.isnan(maxtstep):
            logger.warning('No autocorrelation time to set the prediction length')
            return np.nan
    if 0 < maxtstep < 1:
        maxtstep = int(_round_half_away(n * maxtstep))  # a proportion of the time-series length
    maxtstep = int(max(maxtstep, 10))  # minimum prediction length
    maxtstep = min(maxtstep, n // 2)  # can't look further than half the time series length

    # Theiler window
    past = theiler_window(y, past, n)
    if np.isnan(past):  # no autocorrelation time (e.g. the ACF never decays to 1/e)
        logger.warning('No autocorrelation time to set the Theiler window')
        return np.nan
    past = int(past)
    if cap_horizon and maxtstep + 2 * past > n // 2:
        # cap an 'ac1e' horizon to fit, keeping at least a few correlation times
        maxtstep = n // 2 - 2 * past
        if maxtstep < max(10, 3 * get_tau(y, 'ac1e')):
            logger.warning(f'Time series too short (N = {n}) for a prediction horizon with '
                           f'Theiler window {past}')
            return np.nan
    if maxtstep + 2 * past > n / 2:
        # too few correlation times in the series to follow neighbor divergence
        logger.warning(f'Time series too short (N = {n}) for maxtstep = {maxtstep} with '
                       f'Theiler window {past}')
        return np.nan

    tau, m = _embed_tau_m(y, embed_params)
    if np.isnan(tau):
        logger.warning('Could not determine embedding parameters for this time series')
        return np.nan

    try:
        rows = _tisean_lyap_r(y, tau, m, past, maxtstep)
    except _D2DataError as exc:
        logger.warning(str(exc))
        return np.nan
    if rows is None or rows.shape[0] == 0:
        logger.warning('No output obtained from lyap_r')
        return np.nan
    t, p = rows[:, 0], rows[:, 1]
    if p.size < 6:
        logger.warning('Not enough output from lyap_r to compute statistics')
        return np.nan
    p = p - p[0]
    if np.all(p == 0):
        return np.nan

    out = {}
    for i in range(5):
        out[f'p{i + 1}'] = p[i]
    pmax = np.max(p)
    out['maxp'] = pmax

    def ncross(x):  # number of crossings of the fraction x of the maximum
        return np.sum((p[:-1] - x * pmax) * (p[1:] - x * pmax) < 0)

    out['ncross08max'] = ncross(0.8)
    out['pcross08max'] = ncross(0.8) / (p.size - 1)
    out['ncross09max'] = ncross(0.9)
    out['pcross09max'] = ncross(0.9) / (p.size - 1)

    def time_to(x):  # (1-based) index of the first value above the fraction x of the maximum
        idx = np.flatnonzero(p > x * pmax)
        return idx[0] + 1 if idx.size else None

    for lab, x in (('095', 0.95), ('09', 0.9), ('08', 0.8), ('07', 0.7), ('05', 0.5)):
        i = time_to(x)
        out[f'to{lab}max'] = np.nan if i is None else i - 1

    imax = time_to(0.95)
    if imax is None:
        return np.nan  # cannot be (a flat curve at its maximum of zero)
    if imax <= 3:
        # not a suitable range for finding scaling
        for pre in ('vse', 've'):
            for lab in ('meanabsres', 'rmsres', 'gradient', 'intercept', 'minbad'):
                out[f'{pre}_{lab}'] = np.nan
    else:
        t_scal, p_scal = t[:imax], p[:imax]

        # Vary both the start and end times for the best scaling
        s, e, mybad, pp, res = _dimensions_scaling_range(t_scal, p_scal, gamma=0.006)
        out['vse_meanabsres'] = np.mean(np.abs(res))
        out['vse_rmsres'] = np.sqrt(np.mean(res ** 2))
        out['vse_gradient'] = pp[0]
        out['vse_intercept'] = pp[1]
        out['vse_minbad'] = np.min(mybad)

        # Vary just the end time, which is at least at the 50% mark of the maximum
        imin = time_to(0.5)
        endptr = np.arange(imin, imax + 1)
        bad = np.empty(endptr.size)
        for i, e in enumerate(endptr):
            pf = np.polyfit(t_scal[:e], p_scal[:e], 1)
            bad[i] = np.mean(np.abs(pf[0] * t_scal[:e] + pf[1] - p_scal[:e])) - 0.006 * e
        e = int(endptr[int(np.argmin(bad))])
        pp = np.polyfit(t_scal[:e], p_scal[:e], 1)
        res = pp[0] * t_scal[:e] + pp[1] - p_scal[:e]
        out['ve_meanabsres'] = np.mean(np.abs(res))
        out['ve_rmsres'] = np.sqrt(np.mean(res ** 2))
        out['ve_gradient'] = pp[0]
        out['ve_intercept'] = pp[1]
        out['ve_minbad'] = np.min(bad)

    # The gradients per autocorrelation time
    tau1e = first_crossing(y, 'ac', 1 / np.e, 'continuous')
    if np.isnan(get_tau(y, 'ac1e')):
        tau1e = np.nan  # the ACF never falls to 1/e
    out['vse_gradient_pertau'] = out['vse_gradient'] * tau1e
    out['ve_gradient_pertau'] = out['ve_gradient'] * tau1e

    # Saturating exponential fit
    try:
        a, b = lsqcurvefit_trr(lambda c, x: c[0] * (1 - np.exp(c[1] * x)), [pmax, -0.5], t, p)
        gof = goodness_of_fit(p, a * (1 - np.exp(b * t)), 2)
        out['expfit_a'] = a
        out['expfit_b'] = b
        out['expfit_r2'] = gof['rsquare']
        out['expfit_adjr2'] = gof['adjrsquare']
        out['expfit_rmse'] = gof['rmse']
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        for lab in ('a', 'b', 'r2', 'adjr2', 'rmse'):
            out[f'expfit_{lab}'] = np.nan

    return out




def _ml_randsample(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    # MATLAB's randsample(n, k) (k values out of 1:n without replacement), from a stream
    # emulating MATLAB's Mersenne twister: the first k of randperm(n) if the sample is more than
    # a quarter of the population, otherwise repeated draws with replacement (randi) until k
    # unique values, in a random order.
    if 4 * k > n:
        return _ml_randperm(n, rng)[:k]
    seen = np.zeros(n, dtype=bool)
    while np.count_nonzero(seen) < k:
        seen[np.floor(n * rng.random_sample(k - np.count_nonzero(seen))).astype(int)] = True
    return np.flatnonzero(seen)[_ml_randperm(k, rng) - 1] + 1


def _dvv_draws(rng: np.random.RandomState, n: int, n_dv: int, nsub: int, num_surr: int) -> tuple:
    # The random draws of NL_DVV, in the order hctsa makes them (and from MATLAB's stream, for
    # the same seed): the reference delay vectors (randsample without replacement) for the data,
    # one random permutation per surrogate, then the reference delay vectors for each surrogate.
    # Returns 0-based (refs, perms).
    refs = [_ml_randsample(n_dv, nsub, rng) - 1]
    perms = [_ml_randperm(n, rng) - 1 for _ in range(num_surr)]
    refs += [_ml_randsample(n_dv, nsub, rng) - 1 for _ in range(num_surr)]
    return refs, perms


def _dvv_curve(x: np.ndarray, m: int, nsub: int, nd: float, ntv: int, tau: int,
               ref: np.ndarray) -> np.ndarray:
    # Delay vector variance (Gautama, Mandic & Van Hulle 2004; the DVV toolbox's dvv.m as modified
    # in hctsa): the target variance (of the value following each delay vector) of the sets of
    # delay vectors within distance rd of each of the nsub reference vectors ``ref`` (indices of
    # delay vectors), averaged over references and normalized by the variance of x, at ntv
    # values of rd spanning the mean distance between delay vectors +/- nd standard deviations.
    # Returns the ntv x 2 array of the standardized distance and the target variance.
    n = x.size
    n_dv = n - m * tau  # number of delay vectors (each with a target)
    dvs = x[np.arange(n_dv)[:, None] + tau * np.arange(m)[None, :]]
    d = np.sqrt(np.sum((dvs[:, None, :] - dvs[ref][None, :, :]) ** 2, axis=2))  # n_dv x nsub

    # Mean and standard deviation of the distances over all (reference, vector) pairs except
    # each reference with itself
    count = nsub * n_dv - nsub
    avg = np.sum(d) / count
    sd = np.sqrt((np.sum((d - avg) ** 2) - nsub * avg ** 2) / (count - 1))
    rd = avg - nd * sd + (2 * nd * sd * np.arange(ntv)) / (ntv - 1)

    # For each reference, the variance of the targets of the delay vectors within each rd (from
    # cumulative sums of the targets sorted by distance); only sets of at least 30 count
    tot = np.zeros(ntv)
    num = np.zeros(ntv)
    for k in range(nsub):
        order = np.argsort(d[:, k], kind='stable')
        ds = d[order, k]
        xs = x[order + m * tau]
        s1 = np.cumsum(xs)
        s2 = np.cumsum(xs ** 2)
        c = np.searchsorted(ds, rd, side='right')  # number of vectors within each rd
        t1 = np.where(c > 0, s1[np.maximum(c, 1) - 1], 0.0)
        t2 = np.where(c > 0, s2[np.maximum(c, 1) - 1], 0.0)
        # hctsa excludes from each set the target with (1-based) index k, the loop counter
        # rather than the reference's own index (a quirk of the original toolbox, kept)
        if k + 1 > m * tau:
            inside = d[k - m * tau, k] <= rd
            c = c - inside
            t1 = t1 - inside * x[k]
            t2 = t2 - inside * x[k] ** 2
        ok = (c >= 30) & (rd > 0)
        tot[ok] += (t2[ok] - t1[ok] ** 2 / c[ok]) / (c[ok] - 1)
        num[ok] += 1
    y = np.full(ntv, np.nan)
    has = num > 0
    y[has] = tot[has] / (num[has] * np.var(x, ddof=1))
    return np.column_stack([(rd - avg) / sd, y])


def _dvv_iaaft(x: np.ndarray, perm: np.ndarray, max_it: int = 100, tol: float = 1e-5) -> np.ndarray:
    # Iterated amplitude-adjusted Fourier transform surrogate (the DVV toolbox's surrogate.m,
    # real-valued branch as modified in hctsa): from the permutation ``perm`` of x, alternately
    # impose the amplitude spectrum and rank-order back onto the sorted values of x, returning the
    # iterate (of at most max_it) that matches the amplitude spectrum best.
    x_amp = np.abs(np.fft.fft(x))
    x_sorted = np.sort(x)
    r_prev = x[perm]
    mse, mse_prev = np.inf, 0.0
    best_mse, best_r = np.inf, r_prev
    it = 0
    while abs(mse - mse_prev) > tol and it < max_it:
        mse_prev = mse
        s = np.fft.ifft(x_amp * np.exp(1j * np.angle(np.fft.fft(r_prev)))).real
        r = np.empty_like(x)
        r[np.argsort(s, kind='stable')] = x_sorted
        mse = np.mean(np.abs(x_amp - np.abs(np.fft.fft(r))))
        if mse < best_mse:
            best_mse, best_r = mse, r
        r_prev = r
        it += 1
    return best_r


def dvv(y: ArrayLike, m: int = 3, num_dvs: int = 100, nd: float = 2.0,
        ntv: Union[int, None] = None, num_surr: int = 10, random_seed: Union[int, None] = 0,
        tau: Union[int, str] = 1) -> Union[dict, float]:
    """
    How well nearby delay vectors predict the next value, compared with surrogates.

    The delay vector variance (DVV) method [1]_ for detecting determinism and nonlinearity. A
    delay vector is a run of ``m`` values (spaced ``tau`` apart) and its target is the value
    that follows. For each of ``num_dvs`` randomly chosen reference vectors, the variance of the
    targets of all delay vectors within a distance ``rd`` of the reference is found (only if at
    least 30 vectors lie within ``rd``). The mean of these variances over the references, divided
    by the variance of the series, is the target variance; it is computed at ``ntv`` values of
    ``rd`` that span the mean distance between delay vectors plus or minus ``nd`` standard
    deviations. The same curve is found for ``num_surr`` iterated amplitude-adjusted Fourier
    transform (IAAFT) surrogates, and the outputs summarize the data curve and its difference
    from the mean surrogate curve.

    References
    ----------
    .. [1] T. Gautama, D.P. Mandic and M.M. Van Hulle, "The delay vector variance method for
        detecting determinism and nonlinearity in time series", Physica D 190(3-4), 167-176
        (2004). DOI: 10.1016/j.physd.2003.11.001

    Notes
    -----
    The surrogates keep the best-matching of (at most) 100 IAAFT iterates, as hctsa's patched
    DVV toolbox does, and the rank-ordering step ranks the real part of the spectrum-matched
    series by value (hctsa's ``DVV_surrogate.m`` takes ``real(ifft(...))``: ``ifft`` of the
    conjugate-symmetric spectrum has imaginary parts of about 1e-16, by which MATLAB's
    ``sort`` of a complex vector, ranking by magnitude, used to order the surrogate by
    ``|s|`` instead of ``s``). The reference vectors and surrogate permutations are drawn from
    an emulation of MATLAB's random stream (``rng(random_seed, 'twister')`` with ``randsample``
    and ``randperm``), so the outputs agree with hctsa's for the same seed.

    Parameters
    ----------
    y : array-like
        Input time series (real-valued).
    m : int, optional
        The delay embedding dimension. Default is 3.
    num_dvs : int, optional
        Number of reference delay vectors to consider. Default is 100.
    nd : float, optional
        Span over which to perform DVV, in standard deviations of the distances between delay
        vectors. Default is 2.
    ntv : int, optional
        Number of points on the horizontal (distance) axis. Default is ``25 * nd``.
    num_surr : int, optional
        Number of surrogates to compare to. Default is 10.
    random_seed : int, optional
        Seed for the random choices of reference vectors and surrogates. Default is 0.
    tau : int or str, optional
        The time delay between delay-vector elements: an integer number of samples, or a rule
        understood by :func:`pyhctsa.utils.get_tau` (``'ac1e'``, ``'mi'``, ...). The same delay
        is used for the data and the surrogates. An adaptive delay makes the statistics much less
        dependent on the sampling rate. Default is 1.

    Returns
    -------
    dict or float
        Statistics of the curve of target variance against standardized distance (the data curve,
        at the distances where it is defined): ``trend`` (slope of a linear fit), ``max``,
        ``min``, ``mean``, ``meanDiff`` and ``stdDiff`` (mean and standard deviation of its
        successive differences), ``trendDiff`` (slope of a linear fit to them); and against the
        mean surrogate curve: ``rmsDiffSurr``, ``meanDiffSurr``, ``dataSurrCorr``,
        ``trendDataSurr`` (slope of the surrogate curve against the data curve),
        ``numZeroCrossings`` (sign changes of the difference), ``trendSurr``,
        ``meanDiffTrendSurr`` (``trendSurr`` minus ``trend``) and ``meanNormCDF`` (the mean over
        distances of the Gaussian cumulative probability of the data value given the mean and
        standard deviation across surrogates). Returns NaN if no delay can be set, the series is
        too short to draw ``num_dvs`` reference vectors, or the data curve is defined at fewer
        than two distances.
    """
    x = np.asarray(y, dtype=float).ravel()
    n = x.size
    nd = float(nd)
    ntv = int(25 * nd) if ntv is None else int(ntv)

    tau = get_tau(x, tau)
    if np.isnan(tau):
        return np.nan  # no delay can be set (e.g. a constant series)
    tau = int(tau)

    if n - m * tau < num_dvs:
        logger.warning(f'Time series (N = {n}) too short to draw {num_dvs} reference delay '
                       f'vectors at m = {m}, tau = {tau}')
        return np.nan
    n_dv = n - m * tau

    refs, perms = _dvv_draws(_ml_rng(0 if random_seed is None else int(random_seed)),
                             n, n_dv, num_dvs, num_surr)
    dvv_data = _dvv_curve(x, m, num_dvs, nd, ntv, tau, refs[0])
    dvv_surr = np.stack([_dvv_curve(_dvv_iaaft(x, perms[i]), m, num_dvs, nd, ntv, tau, refs[i + 1])[:, 1]
                         for i in range(num_surr)], axis=1)  # ntv x num_surr
    mean_surr = np.mean(dvv_surr, axis=1)

    # Data curve, using only the distances where it is defined
    good = ~np.isnan(dvv_data[:, 1])
    if good.sum() < 2:
        logger.warning('The data curve is defined at fewer than two distances')
        return np.nan
    t_d, v_d = dvv_data[good, 0], dvv_data[good, 1]
    d_v = np.diff(v_d)

    out = {}
    out['trend'] = np.polyfit(t_d, v_d, 1)[0]
    out['max'] = np.max(v_d)
    out['min'] = np.min(v_d)
    out['mean'] = np.mean(v_d)
    out['meanDiff'] = np.mean(d_v)
    out['stdDiff'] = np.std(d_v, ddof=1)
    out['trendDiff'] = np.polyfit(t_d[:-1], d_v, 1)[0]

    # Comparison to the surrogates, where both are defined
    both = good & ~np.isnan(mean_surr)
    if both.sum() < 2:
        logger.warning('Too few distances at which both the data and surrogate curves are defined')
        return np.nan
    t_b, v_b, s_b = dvv_data[both, 0], dvv_data[both, 1], mean_surr[both]
    out['rmsDiffSurr'] = np.sqrt(np.mean((v_b - s_b) ** 2))
    out['meanDiffSurr'] = np.mean(v_b - s_b)
    out['dataSurrCorr'] = np.corrcoef(v_b, s_b)[0, 1]
    out['trendDataSurr'] = np.polyfit(v_b, s_b, 1)[0]
    out['numZeroCrossings'] = np.sum((v_b - s_b)[1:] * (v_b - s_b)[:-1] < 0)
    out['trendSurr'] = np.polyfit(t_b, s_b, 1)[0]
    out['meanDiffTrendSurr'] = out['trendSurr'] - out['trend']
    with np.errstate(invalid='ignore', divide='ignore'):
        probs = _norm.cdf(v_b, loc=np.mean(dvv_surr[both], axis=1),
                          scale=np.std(dvv_surr[both], axis=1, ddof=1))
    out['meanNormCDF'] = np.mean(probs)
    return out




def _period_normalized_tau(y: np.ndarray) -> Union[int, str]:
    # An embedding delay of one fifth of the series' dominant period (from the most prominent
    # peak of the log Welch spectrum), so it scales with the sampling rate; 'mi' if the series is
    # too short or has no sufficiently prominent peak. (hctsa: SUB_periodNormalizedTau.)
    if y.size < 16:
        return 'mi'
    win_length = int(np.floor(y.size / 4.5))  # MATLAB's pwelch default: 8 segments, 50% overlap
    f, pxx = _welch(y - np.mean(y), fs=1, window=np.hamming(win_length), nperseg=win_length,
                    noverlap=win_length // 2, nfft=max(256, 2 ** int(np.ceil(np.log2(win_length)))),
                    detrend=False, return_onesided=True, scaling='density')
    f, pxx = f[1:], pxx[1:]  # exclude the DC bin
    if pxx.size < 3:
        return 'mi'
    log_p = np.log(pxx + np.spacing(np.max(pxx)))
    locs, props = find_peaks(log_p, prominence=2.0)
    if locs.size == 0:
        return 'mi'
    best = int(np.argmax(props['prominences']))  # the most prominent peak, not necessarily the tallest
    return max(1, int(_round_half_away(1 / f[locs[best]] / 5)))


def persistent_homology(y: ArrayLike, tau: Union[int, str] = 'mi', m: int = 3, max_dim: int = 1,
                        max_n: Union[int, str] = 1000) -> Union[dict, float]:
    """
    Persistent homology of the delay embedding: the lifetimes of its loops and components.

    Embeds the time series in ``m`` dimensions, builds the Vietoris-Rips filtration of the point
    cloud, and summarizes the lifetimes (death minus birth, as a proportion of the filtration
    threshold, the maximum distance between 300 evenly spaced embedded points) of its
    one-dimensional holes (loops, H1; a periodic series gives a long-lived loop) and its
    connected components (H0). Uses ripser [1]_ (the ``ripser`` Python package, an optional
    dependency; hctsa runs the ripser executable).

    References
    ----------
    .. [1] U. Bauer, "Ripser: efficient computation of Vietoris-Rips persistence barcodes",
        J. Appl. Comput. Topol. 5, 391-423 (2021). DOI: 10.1007/s41468-021-00071-5

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        The embedding delay: an integer number of samples, a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``), or ``'periodWelch'`` for
        one fifth of the dominant period (the most prominent peak of the log Welch power
        spectrum, falling back to ``'mi'`` if there is none). A fixed ``tau = 1`` embeds a
        smooth periodic series nearly collinearly, which makes its loop short-lived. Default is
        ``'mi'``.
    m : int, optional
        The embedding dimension. Default is 3.
    max_dim : int, optional
        The maximum homology dimension computed (at least 1 for the H1 outputs; with 0 they are
        zero). Default is 1.
    max_n : int or ``'full'``, optional
        The maximum number of embedded points: a longer point cloud is subsampled at evenly spaced
        points (the cost of ripser grows steeply with the size of the cloud), or ``'full'`` for no
        subsampling. Default is 1000.

    Returns
    -------
    dict or float
        ``maxPersistenceH1``, ``totalPersistenceH1``, ``persistenceEntropyH1``: the maximum and
        sum of the (normalized) lifetimes of the H1 classes and the entropy of their distribution
        (all 0 if there are none); ``totalPersistenceH0``: the sum of the lifetimes of the H0
        classes. Returns NaN if the delay cannot be set, the embedding fails, there are fewer than 20
        embedded points, or the point cloud is degenerate (e.g. a constant series).

    Raises
    ------
    ImportError
        If the ``ripser`` package is not installed.
    """
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError("persistent_homology needs the 'ripser' package "
                          "(pip install ripser)") from exc

    y = np.asarray(y, dtype=float).ravel()

    if isinstance(tau, str) and tau.lower() == 'periodwelch':
        tau = _period_normalized_tau(y)
    tau = get_tau(y, tau)
    if np.isnan(tau):
        return np.nan
    try:
        x = time_delay_embed(y, m, int(tau))
    except ValueError as exc:  # embedding failed
        logger.warning(str(exc))
        return np.nan
    n_emb = x.shape[0]
    if n_emb < 20:  # too few embedded points for a meaningful Rips filtration
        return np.nan

    if isinstance(max_n, str):
        if max_n.lower() != 'full':
            raise ValueError(f"max_n must be an integer or 'full', got '{max_n}'")
    elif n_emb > max_n:
        # an evenly spaced subsample over the whole embedded point cloud
        x = x[np.floor(np.linspace(1, n_emb, int(max_n)) + 0.5).astype(int) - 1]
        n_emb = x.shape[0]

    # The filtration threshold: the largest distance among (at most) 300 evenly spaced points
    n_sub = min(300, n_emb)
    sub = np.floor(np.linspace(1, n_emb, n_sub) + 0.5).astype(int) - 1
    threshold = np.max(pdist(x[sub]))
    if not threshold > 0:
        return np.nan  # degenerate point cloud (e.g. a constant series)
    threshold = float('%.8g' % threshold)  # as passed to ripser

    # hctsa hands ripser the point cloud as a text file of 7 significant digits
    dgms = ripser(squareform(pdist(_round_significant(x.ravel(), 7).reshape(x.shape))),
                  maxdim=int(max_dim), thresh=threshold, distance_matrix=True)['dgms']

    def lifetimes(d):  # finite lifetimes as a proportion of the threshold (essential classes dropped)
        p = (d[:, 1] - d[:, 0]) if d.size else np.empty(0)
        return p[np.isfinite(p)] / threshold

    out = {}
    pers1 = lifetimes(dgms[1]) if len(dgms) > 1 else np.empty(0)
    if pers1.size == 0:
        out['maxPersistenceH1'] = 0.0
        out['totalPersistenceH1'] = 0.0
        out['persistenceEntropyH1'] = 0.0
    else:
        out['maxPersistenceH1'] = np.max(pers1)
        out['totalPersistenceH1'] = np.sum(pers1)
        # zero-persistence pairs contribute nothing to the entropy (0 log 0 = 0)
        pos = pers1[pers1 > 0]
        p = pos / np.sum(pos)
        out['persistenceEntropyH1'] = -np.sum(p * np.log(p)) if p.size else 0.0
    out['totalPersistenceH0'] = np.sum(lifetimes(dgms[0]))
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
def _random_subset(n: int, k: int, random_seed: Union[int, str, None]) -> np.ndarray:
    """
    The first ``k`` of a random permutation of ``n`` indices (from zero), from the portable
    generator :func:`~pyhctsa.robust.bf_random` (as hctsa's
    ``BF_Random(n, BF_RandomSeed(randomSeed), 'perm')``): an integer seed, ``'default'`` for
    seed 0, or ``None``/``'none'`` for a seed from NumPy's global stream.
    """
    return bf_random(n, _bf_random_seed(random_seed), 'perm')[:k] - 1


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


def _line_lengths(group: np.ndarray, pos: np.ndarray, min_len: int) -> np.ndarray:
    """
    Lengths (at least `min_len`) of the runs of consecutive `pos` values within each `group`
    (e.g. diagonal offset or column of a recurrence plot).
    """
    order = np.lexsort((pos, group))
    group, pos = group[order], pos[order]
    new_run = np.ones(group.size, dtype=bool)
    new_run[1:] = (group[1:] != group[:-1]) | (pos[1:] - pos[:-1] != 1)
    run_len = np.diff(np.append(np.flatnonzero(new_run), group.size))
    return run_len[run_len >= min_len]


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
        The seed of the random subsample used to set the radius, as hctsa's ``BF_RandomSeed``:
        an integer, ``'default'`` (seed 0), or ``None``/``'none'`` (a seed from NumPy's global
        stream). The subsample is the first 500 of a :func:`~pyhctsa.robust.bf_random`
        permutation, so the radius is the same as hctsa's. Default is ``'default'``.

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


def rqa(y: ArrayLike, tau: Union[int, str] = 1, m: Union[int, str, list, tuple] = 3,
        theiler_win: Union[int, float, list, tuple] = ('ac', 1), rr: float = 0.1,
        lmin: int = 2, vmin: int = 2, max_n: Union[int, str] = 10000,
        random_seed: Union[int, str, None] = 'default') -> dict:
    """
    Recurrence quantification analysis (RQA) of the delay-embedded series.

    Embeds the time series in an `m`-dimensional delay space and computes standard recurrence
    quantification measures from the resulting recurrence plot [1]: recurrence rate,
    determinism, laminarity, trapping time, and related diagonal and vertical line-length
    statistics. Two embedded states are recurrent if they lie within a radius of each other;
    the radius is set to give a target recurrence rate ``rr``. Pairs closer in time than the
    Theiler window are excluded.

    Neighbors are found with a KD-tree rather than by forming the full N x N distance
    matrix, and the line-length statistics are computed directly from the list of recurrent
    pairs. The number of recurrent pairs is itself about ``rr * N**2``, so run time grows
    roughly quadratically with N at fixed ``rr`` (hence the ``max_n`` cap).

    References
    ----------
    .. [1] N. Marwan, M. C. Romano, M. Thiel and J. Kurths, "Recurrence plots for the analysis
        of complex systems", Phys. Rep. 438, 237 (2007).

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
        The Theiler window excluding temporally-correlated neighbors from the main diagonal
        (see :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k`` times the first
        zero-crossing of the autocorrelation function, or a number of samples. Default is
        ``['ac', 1]``.
    rr : float, optional
        The target recurrence rate used to set the neighborhood radius: the radius is the
        ``rr``-quantile of a random subsample of pairwise distances in the embedded space
        (standard RQA practice, to fix the recurrence rate for comparability across
        series). Default is 0.1.
    lmin : int, optional
        The minimum diagonal line length counted toward determinism and the line-length
        entropy. Default is 2.
    vmin : int, optional
        The minimum vertical line length counted toward laminarity and trapping time.
        Default is 2.
    max_n : int or 'full', optional
        The maximum number of samples to consider: longer series are reduced to their first
        ``max_n`` points, since the number of recurrent pairs grows as ``rr * N**2``.
        ``'full'`` disables cropping (a warning is logged above N = 20000). Default is 10000.
    random_seed : int, str or None, optional
        The seed of the random subsample used to set the radius, as hctsa's ``BF_RandomSeed``:
        an integer, ``'default'`` (seed 0), or ``None``/``'none'`` (a seed from NumPy's global
        stream). The subsample is the first 500 of a :func:`~pyhctsa.robust.bf_random`
        permutation, so the radius is the same as hctsa's. Default is ``'default'``.

    Returns
    -------
    dict or float
        NaN if the embedding or Theiler window cannot be determined, the series is too short
        (fewer than 50 embedded points, or no more than four Theiler windows), the radius is
        degenerate, or no points recur outside the Theiler window. Otherwise:

        - ``RR``: recurrence rate, the proportion of pairs outside the Theiler window that
          are recurrent
        - ``DET``: determinism, the proportion of recurrent points on diagonal lines of
          length at least ``lmin``
        - ``L_mean``, ``L_max``: mean and maximum diagonal line length (lines of length at
          least ``lmin``)
        - ``L_entr``: Shannon entropy (nats) of the distribution of diagonal line lengths
        - ``DIV``: divergence, ``1 / L_max``
        - ``LAM``: laminarity, the proportion of recurrent points on vertical lines of length
          at least ``vmin``
        - ``TT``: trapping time, the mean vertical line length (lines of length at least
          ``vmin``)
        - ``V_max``: the maximum vertical line length

        If there are no diagonal lines, ``DET = 0``, ``L_mean = NaN``, ``L_max = 0``,
        ``L_entr = 0`` and ``DIV = inf``; if there are no vertical lines, ``LAM = 0``,
        ``TT = NaN`` and ``V_max = 0``.
    """
    y = np.asarray(y, dtype=float).ravel()
    if isinstance(max_n, str) and max_n == 'full' and y.size > 20000:
        logger.warning(f"Time series ({y.size} samples) exceeds 20000 with max_n='full'; RQA "
                       "computation may be slow (recurrent pairs grow as rr*N^2)")
    y = _check_max_n(y, max_n, 'RQA at this recurrence rate')

    Y = _bf_embed(y, tau, m)
    if Y is None:
        logger.warning('Embedding failed')
        return np.nan
    n_emb = Y.shape[0]

    theiler = theiler_window(y, theiler_win, n_emb)
    if np.isnan(theiler):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    theiler = int(theiler)
    if n_emb < 50 or n_emb <= 4 * theiler:
        logger.warning(f'Time series too short for a meaningful RQA (Nemb = {n_emb}, theilerWin = {theiler})')
        return np.nan

    radius = _recurrence_radius(Y, rr, random_seed)
    if not radius > 0:
        logger.warning('Degenerate neighborhood radius (data may be too degenerate/short)')
        return np.nan

    # All recurrent pairs outside the Theiler window (which includes the trivial diagonal)
    src, dst = _recurrent_pairs(Y, radius)
    keep = np.abs(src - dst) > theiler
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        logger.warning('No recurrent points found outside the Theiler window -- radius too small?')
        return np.nan

    out = {}
    # Recurrence rate: the fraction of the recurrence matrix outside the Theiler band
    # that is recurrent
    excluded_band = (2 * theiler + 1) * n_emb - theiler * (theiler + 1)
    out['RR'] = src.size / (n_emb ** 2 - excluded_band)

    # Diagonal line lengths, from the upper triangle only (the matrix is symmetric)
    upper = dst > src
    diag_lengths = _line_lengths(dst[upper] - src[upper], src[upper], lmin)
    if diag_lengths.size == 0:
        out['DET'], out['L_mean'], out['L_max'], out['L_entr'], out['DIV'] = 0.0, np.nan, 0, 0.0, np.inf
    else:
        out['DET'] = diag_lengths.sum() / upper.sum()
        out['L_mean'] = diag_lengths.mean()
        out['L_max'] = diag_lengths.max()
        out['DIV'] = 1 / out['L_max']
        counts = np.bincount(diag_lengths)
        p = counts[counts > 0] / diag_lengths.size
        out['L_entr'] = -np.sum(p * np.log(p))

    # Vertical line lengths, from the full band-excluded matrix, grouping recurrent points by column
    vert_lengths = _line_lengths(dst, src, vmin)
    if vert_lengths.size == 0:
        out['LAM'], out['TT'], out['V_max'] = 0.0, np.nan, 0
    else:
        out['LAM'] = vert_lengths.sum() / src.size
        out['TT'] = vert_lengths.mean()
        out['V_max'] = vert_lengths.max()

    return out


def return_time(y: ArrayLike, nnr: Union[int, float] = 0.01, num_lags: int = 100,
                past: Union[int, float, list, tuple] = ('ac', 1), nref: int = -1,
                embed_params: Union[list, tuple] = ('ac', 'fnn')) -> dict:
    """
    Analysis of the histogram of return times.

    Return times are the times taken for the time series to return to a similar location in
    phase space from a given reference point. Strong peaks in the histogram indicate
    periodicities in the data.

    For each reference point in the embedding space, its ``nnr`` nearest neighbors are found
    (excluding a Theiler window of ``past`` samples either side), and the time offset ``T`` of
    each neighbor from the reference point is recorded. The histogram of these offsets over
    the ``num_lags`` lags beyond the Theiler window, ``T = past + 1, ..., past + num_lags``
    (the "return-time profile"), is analyzed. This follows TSTOOL's ``return_time``
    (which hctsa previously called), with one change: each lag's count is divided by its
    expected count if neighbors were placed at random among the valid (Theiler-excluded)
    candidates, rather than by TSTOOL's ``2 * nnr * (N - T)``, so that the profile is about 1
    at every lag for an uncorrelated process at any series length (TSTOOL's normalization
    scaled as 1/N). Values above 1 mark lags at which the trajectory preferentially returns
    to its neighborhood. The profile is closely related to the tau-recurrence rate of
    recurrence quantification analysis, with neighborhoods holding a fixed proportion of
    points rather than having a fixed radius. (For the distribution of *first* return times to
    a neighborhood, see :func:`recurrence_times`.)

    References
    ----------
    .. [1] N. Marwan, M. C. Romano, M. Thiel and J. Kurths, "Recurrence plots for the analysis
        of complex systems", Phys. Rep. 438, 237 (2007) (recurrence quantification analysis).

    Parameters
    ----------
    y : array-like
        The input time series.
    nnr : int or float, optional
        The number of nearest neighbors, or, if in (0, 1), a proportion of the number of
        embedded points (keeping neighborhoods the same size in probability as the series
        length changes). Default is 0.01.
    num_lags : int, optional
        The number of lags beyond the Theiler window to analyze, in samples (at least 2).
        Default is 100.
    past : int, float or ``['ac', k]``, optional
        The Theiler window, excluding neighbors that are close only because they are close in
        time (see :func:`pyhctsa.utils.theiler_window`): ``['ac', k]`` for ``k`` times the first
        zero-crossing of the autocorrelation function, or a number of samples.
        Default is ``['ac', 1]``.
    nref : int, optional
        The number of reference points, spaced evenly through the series (-1 uses all
        points). A fixed number keeps the number of neighbors counted at each lag, and so the
        sampling noise of the histogram, independent of the series length (neighbors are still
        sought among all points). Default is -1.
    embed_params : list or tuple, optional
        The embedding, as ``(tau, m)``: the time delay (an integer or a rule understood by
        :func:`pyhctsa.utils.get_tau`) and the embedding dimension (an integer, or ``'fnn'``
        for false nearest neighbors). Default is ``('ac', 'fnn')``.

    Returns
    -------
    dict or float
        NaN if the Theiler window or embedding cannot be determined, or the series is too short
        (fewer embedded points than twice the largest lag, or than ``nnr`` plus two Theiler
        windows). Otherwise measures of the return-time profile (the neighbor count at each lag
        relative to chance), and of the histogram of its values:

        - ``max``, ``std``, ``iqr``: the maximum, standard deviation and interquartile range of
          the profile
        - ``pzeros``: the proportion of lags with no neighbors
        - ``pg05``: the proportion of lags at which the profile exceeds half its maximum
        - ``meanpeaksep``, ``maxpeaksep``, ``minpeaksep``, ``rangepeaksep``, ``stdpeaksep``:
          statistics of the spacings between successive crossings of half the maximum, as a
          proportion of the number of lags (``stdpeaksep`` is divided by the square root of the
          number of lags instead); all are NaN with fewer than 3 crossings
        - ``statrtys``, ``statrtym``: the ratio of the standard deviation (``statrtys``) or mean
          (``statrtym``) of the profile over the first half of the lags to that over the second
        - ``hhist``: the entropy of the profile as a distribution over lags
        - ``hcgdist``, ``rangecgdist``, ``pzeroscgdist``: the entropy, range and proportion of
          zeros of the profile after summing it into 20 equal bins of lags (as a distribution
          over bins)
        - ``maxhisthist``, ``phisthistmin``, ``hhisthist``: the maximum, the first (lowest-value)
          bin probability, and the entropy of the histogram of profile values (square-root bins)
    """
    y = np.asarray(y, dtype=float).ravel()
    if num_lags < 2:
        raise ValueError(f'num_lags ({num_lags}) must be at least 2')

    past = theiler_window(y, past)
    if np.isnan(past):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    past = int(past)
    max_t = past + int(num_lags)  # the maximum return time (lag) to consider

    Y = _bf_embed(y, embed_params[0], embed_params[1])
    if Y is None:
        logger.warning('Embedding failed')
        return np.nan
    n_emb = Y.shape[0]
    if 0 < nnr < 1:  # a proportion of the number of embedded points
        nnr = max(1, int(_round_half_away(nnr * n_emb)))
    nnr = int(nnr)
    if n_emb < 2 * max_t or n_emb <= nnr + 2 * past + 1:
        # every lag in the histogram needs to be sampled by at least half the points
        logger.warning('Time series too short to do a return-time analysis with these parameters')
        return np.nan

    # The neighborhood radius of each reference point: the distance to its nnr-th nearest
    # neighbor outside the Theiler window (as a squared radius; NaN for non-reference points)
    if nref == -1 or nref >= n_emb:
        refs = np.arange(n_emb)
    else:
        refs = np.unique(np.floor(_linspace(1, n_emb, int(nref)) + 0.5).astype(int)) - 1  # (MATLAB's round)
    # at most 2*past + 1 points (the reference point itself included) fall within the Theiler
    # window, so nnr + 2*past + 1 neighbors always hold nnr valid ones
    k = min(n_emb, nnr + 2 * past + 1)
    tree = KDTree(Y)
    r2 = np.full(n_emb, np.nan)
    chunk = max(1, int(2e6 // k))
    for c in range(0, refs.size, chunk):
        the_refs = refs[c:c + chunk]
        dist, idx = tree.query(Y[the_refs], k=k)
        is_valid = np.abs(idx - the_refs[:, None]) > past
        which_col = np.argmax(np.cumsum(is_valid, axis=1) >= nnr, axis=1)
        r2[the_refs] = dist[np.arange(the_refs.size), which_col] ** 2 * (1 + 1e-9)

    # Count the neighbors at each lag, relative to the count expected by chance
    lags = np.arange(past + 1, max_t + 1)
    counts = np.zeros(lags.size)
    for i, lag in enumerate(lags):
        fwd = refs[refs + lag <= n_emb - 1]  # references with a partner `lag` ahead
        bwd = refs[refs - lag >= 0]  # references with a partner `lag` behind
        counts[i] = (np.sum(np.sum((Y[fwd + lag] - Y[fwd]) ** 2, axis=1) <= r2[fwd])
                     + np.sum(np.sum((Y[bwd - lag] - Y[bwd]) ** 2, axis=1) <= r2[bwd]))
    # By chance, a given valid candidate is one of reference i's nnr neighbors with probability
    # nnr/V_i, where V_i is the number of points outside i's Theiler window
    i = np.arange(n_emb)
    v = n_emb - (np.minimum(i, past) + np.minimum(n_emb - 1 - i, past) + 1)
    w = np.zeros(n_emb)
    w[refs] = nnr / v[refs]
    cw = np.cumsum(w)
    expected = cw[n_emb - lags - 1] + (cw[-1] - cw[lags - 1])  # forward + backward partners
    trett = counts / expected

    out = {}
    nn = lags.size
    max_trett = np.max(trett)
    out['max'] = max_trett
    out['std'] = np.std(trett, ddof=1)
    out['pzeros'] = np.sum(trett == 0) / nn
    out['pg05'] = np.sum(trett > max_trett * 0.5) / nn
    q25, q75 = matlab_quantile(trett, [0.25, 0.75])
    out['iqr'] = q75 - q25

    # Recurrent peaks
    icross05 = np.flatnonzero((trett[:-1] - 0.5 * max_trett) * (trett[1:] - 0.5 * max_trett) < 0)
    if icross05.size > 2:
        d = np.diff(icross05)
        d = d[d > 0.4 * d.max()]  # remove small entries, crossing peaks
        out['meanpeaksep'] = np.mean(d) / nn
        out['maxpeaksep'] = np.max(d) / nn
        out['minpeaksep'] = np.min(d) / nn
        out['rangepeaksep'] = np.ptp(d) / nn
        out['stdpeaksep'] = (np.std(d, ddof=1) if d.size > 1 else 0.0) / np.sqrt(nn)
    else:
        for name in ('meanpeaksep', 'maxpeaksep', 'minpeaksep', 'rangepeaksep', 'stdpeaksep'):
            out[name] = np.nan

    # Short lags compared to long lags
    half = nn // 2
    out['statrtys'] = np.std(trett[:half], ddof=1) / np.std(trett[half:], ddof=1)
    out['statrtym'] = np.mean(trett[:half]) / np.mean(trett[half:])

    # Entropy of the histogram, as a distribution over lags
    p_trett = trett / np.sum(trett)
    out['hhist'] = -np.sum(p_trett[p_trett > 0] * np.log(p_trett[p_trett > 0]))

    # Coarse-grain to 20 bins of lags
    num_bins = 20
    inds = np.floor(_linspace(0, nn, num_bins + 1) + 0.5).astype(int)  # (MATLAB's round)
    cglav = np.array([np.sum(p_trett[inds[b]:inds[b + 1]]) for b in range(num_bins)])
    out['hcgdist'] = -np.sum(cglav[cglav > 0] * np.log(cglav[cglav > 0]))
    out['rangecgdist'] = np.ptp(cglav)
    out['pzeroscgdist'] = np.sum(cglav == 0) / num_bins

    # Distribution of the profile values (MATLAB's 'sqrt' bin rule, as histcounts)
    n_bins = max(int(np.ceil(np.sqrt(nn))), 1)
    lo, hi = np.min(trett), np.max(trett)
    edges = bin_picker(np.float64(lo), np.float64(hi), None, (hi - lo) / n_bins)
    nhist = np.histogram(trett, bins=edges)[0] / nn
    out['maxhisthist'] = np.max(nhist)
    out['phisthistmin'] = nhist[0]  # probability in the first (lowest-value) bin
    out['hhisthist'] = -np.sum(nhist[nhist > 0] * np.log(nhist[nhist > 0]))

    return out


def embed_cluster(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 2, k_max: int = 4,
                  max_n: Union[int, str] = 'full') -> dict:
    """
    Whether the time-delay embedding of the series forms separate clusters of points.

    Reconstructs the time series as a time-delay embedding (as in :func:`embed_pca`) and
    fits Gaussian mixture models with a small grid of component counts (1, ..., ``k_max``) to
    the resulting point cloud. A dynamical process whose trajectory visits distinct regions of
    phase space (e.g., alternating between two attractor states, or a system with
    intermittent bursts) leaves a multi-modal point cloud in the embedding; a process with a
    single smooth (e.g., unimodal-stochastic or single-loop periodic) attractor does not. This
    is a distinct signal from marginal-distribution multi-modality, since two states can
    overlap entirely in amplitude yet still separate cleanly once lagged coordinates are
    added, and from regime-switching detected by hidden Markov models, which cluster points
    in raw-amplitude (not lagged/embedded) space.

    Rather than reporting only the BIC-optimal number of components (a discrete,
    model-selection-driven output that can be noisy across similar time series), the main
    outputs are continuous separation statistics from a *fixed* 2-component fit, alongside the
    (secondary) BIC-optimal number of components for reference.

    The ``sep_*`` outputs are always computed from the fixed 2-component fit, so they are not
    gated by whether that fit is favored by BIC over a single Gaussian: even a genuinely
    unimodal-but-elongated point cloud (e.g., AR(1) noise) gets split into two "confident"
    halves. Likewise, a curved-but-unimodal manifold (e.g., the ring traced out by a periodic
    signal in a 2-d embedding) is poorly fit by any single elliptical Gaussian and so also
    drives ``bestK`` and ``dBIC`` up, despite having no distinct dynamical states.
    ``dBIC == 0`` (``bestK == 1``) is a clean "no mixture structure at all" signal, but
    ``dBIC > 0`` does not by itself distinguish true multi-modality from curvature.

    The mixtures are fitted with scikit-learn's ``GaussianMixture`` (full covariances,
    k-means++ initialization, 3 initializations, at most 500 iterations, covariance
    regularization of ``1e-6`` times the mean variance of the embedded coordinates), seeded
    with 0. hctsa's ``fitgmdist`` runs the same fits, with its own initialization draws,
    so a fit that does not have a clear optimum can differ.

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time delay of the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of the
        autocorrelation function, ``'ac1e'`` the (floored) first 1/e crossing of the
        autocorrelation function, and ``'mi'`` the smaller of the first minimum of the
        (Kraskov) automutual information and the 1/e time. Default is ``'ac'``.
    m : int, optional
        The embedding dimension. Default is 2.
    k_max : int, optional
        The maximum number of Gaussian mixture components to consider when searching for the
        BIC-optimal component count. Default is 4.
    max_n : int or 'full', optional
        The maximum number of embedded points used to fit the mixture models: longer
        embeddings are reduced to their first ``max_n`` points (a memory/time cap; the
        separation estimates keep sharpening with more points), or ``'full'`` for no cropping.
        Default is ``'full'``.

    Returns
    -------
    dict or float
        NaN if the embedding fails, there are fewer than ``20 * m * k_max`` embedded points,
        the embedded point cloud is constant, or the single-component fit fails. Otherwise:

        - ``bestK``: the BIC-optimal number of mixture components over ``1:k_max``
        - ``dBIC``: the relative BIC improvement of the best fit over a single (unimodal)
          Gaussian fit, ``(BIC_1 - BIC_best) / |BIC_1|``; 0 when ``bestK == 1``
        - ``sep_mahal``: ``log1p`` of the Mahalanobis separation between the two component
          means of the 2-component fit, using their pooled covariance (log-compressed to tame
          the heavy tail from near-singular covariance on near-deterministic embeddings)
        - ``sep_conf``: mean posterior cluster-assignment confidence (mean of the larger of
          each point's two posterior probabilities) under the 2-component fit; between 0.5 (fully
          ambiguous assignment) and 1
        - ``sep_silh``: mean silhouette value (squared Euclidean distance, as MATLAB's
          ``silhouette``) of the hard (posterior-argmax) 2-cluster assignment
        - ``sep_weightbalance``: ratio of the smaller to the larger mixture weight under the
          2-component fit; 1 for balanced clusters, tending to 0 as one component comes to
          dominate (degenerating toward a unimodal fit)

        The ``sep_*`` outputs are NaN if ``k_max < 2`` or the 2-component fit fails.
    """

    y = np.asarray(y, dtype=float).ravel()
    y_embed = _bf_embed(y, tau, m)
    if y_embed is None:
        logger.warning('Embedding parameters are not suitable for this time series')
        return np.nan

    # Enough points, relative to m and k_max, for a well-posed full-covariance fit at the
    # largest component count considered
    n_embed = y_embed.shape[0]
    if n_embed < 20 * m * k_max:
        logger.warning(f'Not enough embedded points ({n_embed}) for a stable {m}-dimensional, '
                       f'up-to-{k_max}-component mixture fit')
        return np.nan

    # A constant (or near-constant) embedded point cloud cannot be usefully clustered
    if np.all(np.ptp(y_embed, axis=0) < 1e-10):
        return np.nan

    if isinstance(max_n, str):
        if max_n != 'full':
            raise ValueError(f"max_n must be an integer or 'full', got '{max_n}'")
    elif n_embed > max_n:
        logger.warning(f'Cropping to the first {int(max_n)} of {n_embed} embedded points for mixture '
                       'fitting (memory/time cap, not a convergence point)')
        y_embed = y_embed[:int(max_n)]

    # Regularize covariance estimates proportionally to the data's own scale
    reg_val = 1e-6 * np.mean(np.var(y_embed, axis=0, ddof=1))

    def fit(k):
        gm = GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg_val, n_init=3,
                             init_params='k-means++', max_iter=500, tol=1e-6, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # (replicates that do not converge are expected)
            return gm.fit(y_embed)

    # Fit k = 1, ..., k_max, and record their BIC
    bic = np.full(k_max, np.nan)
    models = [None] * k_max
    for k in range(1, k_max + 1):
        try:
            models[k - 1] = fit(k)
            bic[k - 1] = models[k - 1].bic(y_embed)
        except Exception:  # (e.g., an empty or near-singular component)
            if k == 1:  # no useful mixture structure can be assessed either
                return np.nan
    best_k = int(np.nanargmin(bic)) + 1

    out = {'bestK': best_k}
    out['dBIC'] = 0 if best_k == 1 else (bic[0] - bic[best_k - 1]) / abs(bic[0])

    # Continuous separation statistics from a fixed 2-component fit
    if k_max < 2 or models[1] is None:
        for name in ('sep_mahal', 'sep_conf', 'sep_silh', 'sep_weightbalance'):
            out[name] = np.nan
        return out

    gm2 = models[1]
    post = gm2.predict_proba(y_embed)
    clust = np.argmax(post, axis=1)
    out['sep_conf'] = np.mean(np.max(post, axis=1))
    out['sep_weightbalance'] = np.min(gm2.weights_) / np.max(gm2.weights_)
    pooled_cov = (gm2.covariances_[0] + gm2.covariances_[1]) / 2
    d_mu = gm2.means_[0] - gm2.means_[1]
    # (log1p-compressed: the raw value explodes for near-singular pooled covariance)
    out['sep_mahal'] = np.log1p(np.sqrt(d_mu @ np.linalg.solve(pooled_cov, d_mu)))
    if np.unique(clust).size < 2:
        # all points collapsed onto one component under hard assignment
        out['sep_silh'] = 0.0
    else:
        s = silhouette_samples(y_embed, clust, metric='sqeuclidean')
        s[np.bincount(clust)[clust] == 1] = 1.0  # (MATLAB gives a singleton cluster 1)
        out['sep_silh'] = np.mean(s)
    return out


def _spectrum_stats(perc: np.ndarray, m: int) -> dict:
    """
    Statistics of a normalized (summing to 1), descending eigenvalue spectrum, as
    :func:`embed_pca` (hctsa's ``SUB_spectrumstats`` in NL_EmbedKernelPCA).
    """
    stats = {f'perc_{i + 1}': perc[i] for i in range(m)}
    # The spread statistics are taken over the leading m components only: the linear spectrum
    # has exactly m entries, but the kernel spectrum has one per embedded point, and taken over
    # all of them its spread falls with the number of points
    top = perc[:m]
    stats['std'] = np.std(top, ddof=1)
    stats['range'] = np.ptp(top)
    stats['min'] = np.min(top)
    stats['max'] = np.max(top)
    stats['top2'] = np.sum(perc[:2])
    csperc = np.cumsum(perc)
    for pct in (50, 60, 70, 80, 90):
        stats[f'nto{pct}'] = _first_fn(csperc, pct / 100, 'over')
    for name, thresh in (('fb05', 0.5), ('fb02', 0.2), ('fb01', 0.1), ('fb001', 0.01)):
        stats[name] = _first_fn(perc, thresh, 'under')
    return stats


def embed_kernel_pca(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 3,
                     max_n: Union[int, str] = 2000) -> dict:
    """
    Kernel PCA of a time-delay embedding of the series, compared with linear PCA.

    Reconstructs the time series as a time-delay embedding (as in :func:`embed_pca`) and
    performs kernel principal components analysis on the result using an RBF kernel
    ``exp(-d^2 / median(d^2))``, with ``d`` the distance between embedded points, then
    compares the resulting eigenvalue spectrum to that of ordinary (linear) PCA on the same
    embedded points [1, 2].

    At any finite kernel bandwidth, kernel PCA's spectrum is less compact than linear PCA's in
    absolute terms (its RBF feature space is far higher-dimensional than the embedding
    itself), so the kernel-to-linear ratios (``top2_ratio``, ``nto80_ratio``,
    ``nto50_ratio``) are below 1 (``top2_ratio``) or at least 1 (``nto*_ratio``) for every
    series. In simulations (N = 1000), the ratios are nearer 1 for series on a curved
    low-dimensional manifold than for the linear (Gaussian) process with the same power
    spectrum: for the logistic and Henon maps, ``top2_ratio`` is 0.77 and 0.73 against 0.52
    and 0.50 for their phase-randomized surrogates (``tau = 'ac'``, ``m = 3``). The signal is
    weaker for the Lorenz system and absent for a Roessler oscillator, which is close to
    linear at this sampling. The ratios are not a stand-alone nonlinearity test, however: they
    also rise with linear autocorrelation (``top2_ratio`` is about 0.52 for white noise and AR(1)
    with phi = 0.5, but 0.61-0.71 for AR(1) with phi = 0.99), so a smooth linear process can look
    more 'nonlinear' than a chaotic map. Compare against surrogates to isolate nonlinearity.
    ``std_ratio`` did not separate nonlinear from linear series consistently.

    References
    ----------
    .. [1] B. Scholkopf, A. Smola and K.-R. Muller, "Nonlinear Component Analysis as a
        Kernel Eigenvalue Problem", Neural Comput. 10(5), 1299 (1998).
    .. [2] D. S. Broomhead and G. P. King, "Extracting qualitative dynamics from
        experimental data", Physica D 20(2-3), 217 (1986).

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time delay of the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau`. ``'ac'`` is the first zero-crossing of the
        autocorrelation function, ``'ac1e'`` the (floored) first 1/e crossing of the
        autocorrelation function, and ``'mi'`` the smaller of the first minimum of the
        (Kraskov) automutual information and the 1/e time. Default is ``'ac'``.
    m : int, optional
        The embedding dimension (at least 2). Default is 3.
    max_n : int or 'full', optional
        The maximum number of embedded points used to form the N x N kernel matrix, whose
        eigendecomposition costs O(N^3). Longer embeddings are reduced to their first ``max_n``
        points (a memory/time cap, not a convergence point: the spectrum estimate keeps
        sharpening with more points); ``'full'`` disables this, with a warning above 5000
        points, where the eigendecomposition takes several seconds. Default is 2000.

    Returns
    -------
    dict or float
        NaN if the embedding fails, there are too few embedded points for a rank-``m``
        decomposition, the embedded points coincide, or the kernel spectrum is degenerate.
        Otherwise statistics of the normalized kernel PCA spectrum (the proportion of
        variance in feature space explained by each kernel principal component, ordered from
        largest, one per embedded point), with the linear PCA spectrum of the same points
        (``m`` entries) for comparison:

        - ``perc_1``, ..., ``perc_m``: the proportion of variance explained by each of the top
          ``m`` kernel components
        - ``std``, ``range``, ``min``, ``max``: standard deviation, range, minimum and maximum
          of the top ``m`` proportions only (so they are comparable with linear PCA)
        - ``top2``: the proportion of variance explained by the top two kernel components
        - ``nto50``, ``nto60``, ``nto70``, ``nto80``, ``nto90``: the number of kernel
          components needed to explain more than 50%, 60%, 70%, 80% or 90% of the variance
        - ``fb05``, ``fb02``, ``fb01``, ``fb001``: the position of the first kernel component
          whose proportion of variance is below 0.5, 0.2, 0.1 or 0.01
        - ``top2_ratio``, ``top2_diff``: ``top2`` of the kernel PCA over (and minus) that of
          linear PCA
        - ``nto80_ratio``, ``nto80_diff``: ``nto80`` of the kernel PCA over (and minus) that of
          linear PCA
        - ``nto50_ratio``: ``nto50`` of the kernel PCA over that of linear PCA
        - ``std_ratio``: ``std`` of the kernel PCA over that of linear PCA
    """
    y = np.asarray(y, dtype=float).ravel()
    y_embed = _bf_embed(y, tau, m)
    if y_embed is None:
        logger.warning('Embedding parameters are not suitable for this time series')
        return np.nan

    # pca needs m components (and at least 2, for top2)
    if y_embed.shape[0] - 1 < m or m < 2:
        logger.warning(f'Not enough embedding vectors ({y_embed.shape[0]}) for a rank-{m} PCA')
        return np.nan

    # Crop to max_n embedded points for the kernel matrix (memory/time cap)
    n_emb = y_embed.shape[0]
    if isinstance(max_n, str):
        if max_n != 'full':
            raise ValueError(f"max_n must be an integer or 'full', got '{max_n}'")
        if n_emb > 5000:
            logger.warning(f"{n_emb} embedded points exceeds 5000 with max_n='full'; the kernel "
                           'eigendecomposition may take several seconds')
    elif n_emb > max_n:
        logger.warning(f'Cropping to the first {int(max_n)} of {n_emb} embedded points for kernel PCA '
                       '(memory/time cap, not a convergence point)')
        y_embed = y_embed[:int(max_n)]
    n = y_embed.shape[0]
    if n - 1 < m:
        logger.warning(f'Not enough embedded points ({n}) after cropping for a rank-{m} kernel PCA')
        return np.nan

    # Linear PCA on the (possibly cropped) embedded points, for comparison
    latent_lin = PCA().fit(y_embed).explained_variance_
    stats_lin = _spectrum_stats(latent_lin / np.sum(latent_lin), m)

    # Kernel PCA with an RBF kernel, whose bandwidth is the median heuristic: the median of
    # the pairwise squared distances sets the kernel's length scale to the data's own typical
    # point-to-point spacing
    sq_dist = squareform(pdist(y_embed, 'sqeuclidean'))
    med_sq_dist = np.median(sq_dist[~np.eye(n, dtype=bool)])
    if med_sq_dist == 0:  # all embedded points coincide
        return np.nan
    kmat = np.exp(-sq_dist / med_sq_dist)

    # Center the kernel matrix in feature space
    one_n = np.full((n, n), 1 / n)
    kc = kmat - one_n @ kmat - kmat @ one_n + one_n @ kmat @ one_n
    kc = (kc + kc.T) / 2  # symmetrize away numerical asymmetry

    # The eigenvalues of the centered kernel matrix are N times those of the empirical
    # covariance operator in feature space, but the constant factor cancels in the normalized
    # spectrum. Small negative values are numerical noise (the matrix is positive
    # semi-definite in theory)
    eig_k = np.sort(np.linalg.eigvalsh(kc))[::-1]
    eig_k[eig_k < 0] = 0
    if np.sum(eig_k) == 0 or eig_k[m - 1] == 0:
        logger.warning('Kernel PCA produced a degenerate (near-zero-rank) spectrum')
        return np.nan
    stats_kern = _spectrum_stats(eig_k / np.sum(eig_k), m)

    out = dict(stats_kern)
    # Ratios and differences of matched linear and kernel spectrum statistics, the
    # nonlinearity signal. The kernel spectrum is *always* less compact than the linear one in
    # absolute terms (not itself the signal); what differs by system is *how much* less
    # compact: on a genuinely low-dimensional nonlinear manifold, kernel PCA still finds much
    # more compact structure than it does for a linear/stochastic process, so the ratios sit
    # closer to 1.
    out['top2_ratio'] = stats_kern['top2'] / stats_lin['top2']
    out['top2_diff'] = stats_kern['top2'] - stats_lin['top2']
    out['nto80_ratio'] = stats_kern['nto80'] / stats_lin['nto80']
    out['nto80_diff'] = stats_kern['nto80'] - stats_lin['nto80']
    out['nto50_ratio'] = stats_kern['nto50'] / stats_lin['nto50']
    out['std_ratio'] = stats_kern['std'] / stats_lin['std']
    return out


def box_count_entropy_rate(y: ArrayLike, num_bins: int = 100,
                           embed_params: Union[list, tuple] = ('ac', 'fnn')) -> dict:
    """
    How the box-counting (order-2 Renyi) entropy of a delay embedding grows with embedding
    dimension.

    Time-delay embeds the series in ``d = 1, ..., m`` dimensions and partitions the space into
    boxes of side ``epsilon``, using TISEAN's ``boxcount`` (this operation previously used
    TSTOOL's ``corrdim``). With ``p_i`` the fraction of embedded points in box ``i``,
    ``boxcount`` gives the order-2 Renyi (collision) entropy
    ``H(epsilon, d) = -log(sum_i p_i^2)`` for a sweep of ``num_bins`` box sizes, from the full
    range of the series downward, and the increment over the ``(d-1)``-dimensional embedding,
    ``I(epsilon, d) = H(epsilon, d) - H(epsilon, d-1)`` (defined for ``d = 2, ..., m``; at
    ``d = 1``, ``boxcount`` reports ``H`` itself, which is not an increment, so ``d = 1`` is
    excluded from all summaries). The matrix ``I`` (length scales by embedding dimensions
    ``2, ..., m``) is summarized across length scales for each dimension, across dimensions for
    each length scale, and overall.

    The increment ``I`` approaches the entropy rate of the process (the K2 entropy, per delay
    step) rather than a slope against ``log(epsilon)``, so these features are entropy-rate-like,
    not correlation dimensions. (This function was previously named NL_BoxCorrDim in hctsa,
    after the TSTOOL correlation-dimension code it replaced.) hctsa registers ``meanr``,
    ``medianr``, ``minr`` and ``meanchr`` at ``r`` = 2, 3, 4, 6, 8, 11, 14, 17, 20, 24, 28, 32,
    36 of ``num_bins = 50`` (2 to 429 boxes per axis): coarse scales change quickly with ``r``
    and are sampled densely; neighboring finer scales are nearly redundant; for flows (long
    delays) the informative scales lie beyond ``r = 18``; and beyond ``r = 36`` the
    5-dimensional embedding saturates (``I = 0``) for series of a few thousand points.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_bins : int, optional
        The number of length-scale (``epsilon``) values in the box-counting sweep (at least 2).
        TSTOOL's "maximum number of partitions per axis" has no exact TISEAN equivalent; this is
        the closest analogue. Default is 100.
    embed_params : list or tuple, optional
        The embedding parameters as ``(tau, m)``: the time delay (an integer, or a rule
        understood by :func:`pyhctsa.utils.get_tau`: ``'ac'``, ``'ac1e'`` or ``'mi'``) and the
        embedding dimension (an integer, or ``'fnn'`` for false nearest neighbors). hctsa uses
        ``('ac1e', 5)``. Default is ``('ac', 'fnn')``.

    Returns
    -------
    dict or float
        NaN if the embedding parameters cannot be determined, the series is constant, or the
        embedding dimension is below 2. Otherwise summaries of ``I(epsilon, d)``, with ``d`` the
        embedding dimension and ``r`` the index (from 1) of the length scale (``r = 1`` is the
        full range of the series, larger ``r`` are finer scales):

        - ``meand<d>``, ``mediand<d>``: mean and median of ``I`` over length scales, at embedding
          dimension ``d = 2, ..., m``
        - ``meanr<r>``, ``medianr<r>``, ``minr<r>``: mean, median and minimum of ``I`` over
          embedding dimensions ``2, ..., m``, at length scale ``r = 2, ..., num_bins``
        - ``meanchr<r>``: mean change of ``I`` from one embedding dimension to the next
          (``d = 2, ..., m``), at length scale ``r = 2, ..., num_bins`` (NaN for ``m = 2``)
        - ``stdmean``, ``stdmedian``: standard deviation, across embedding dimensions
          ``2, ..., m``, of the mean (or median) of ``I`` over length scales
        - ``medianstretch``, ``iqrstretch``: median and interquartile range of ``I`` over all
          length scales and embedding dimensions ``2, ..., m``

        (The minima over all length scales, formerly ``mind<d>`` and ``minstretch``, were removed:
        the coarsest scale is a single box, where ``I = 0``, so they were always 0.)
    """
    y = np.asarray(y, dtype=float).ravel()
    if num_bins < 2:
        raise ValueError('num_bins must be at least 2')
    params = _embedding_params(y, embed_params[0], embed_params[1])
    if params is None:
        logger.warning('Could not determine embedding parameters for this time series')
        return np.nan
    tau, m_max = params

    # (the order-2 Renyi entropy of the partition into boxes, as a function of the box size and the
    # embedding dimension d, and its increment with d; the increments are analyzed)
    try:
        rs = _tisean_boxcount(y, tau, m_max, int(num_bins), 2.0)[2]
    except _D2DataError as exc:
        logger.warning(f'boxcount failed: {exc}')
        return np.nan
    if m_max < 2:
        # the increment I is only defined from d = 2 (d = 1 holds H itself)
        logger.warning(f'Embedding dimension m = {m_max} is too low for a box-counting entropy increment')
        return np.nan

    out = {}
    for d in range(2, m_max + 1):
        out[f'meand{d}'] = np.mean(rs[:, d - 1])
        out[f'mediand{d}'] = np.median(rs[:, d - 1])
    for r in range(2, rs.shape[0] + 1):
        row = rs[r - 1, 1:]
        out[f'meanr{r}'] = np.mean(row)
        out[f'medianr{r}'] = np.median(row)
        out[f'minr{r}'] = np.min(row)
        out[f'meanchr{r}'] = np.mean(np.diff(row)) if row.size > 1 else np.nan
    out['stdmean'] = np.std(np.mean(rs[:, 1:], axis=0), ddof=1) if m_max > 2 else 0.0
    out['stdmedian'] = np.std(np.median(rs[:, 1:], axis=0), ddof=1) if m_max > 2 else 0.0
    stretch = rs[:, 1:].ravel()
    out['medianstretch'] = np.median(stretch)
    q25, q75 = matlab_quantile(stretch, [0.25, 0.75])
    out['iqrstretch'] = q75 - q25
    return out


def evt_local_dim(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 3, q: float = 0.98,
                  theiler_win: Union[int, float, list, tuple] = ('ac', 1), n_poles: int = 200,
                  m_order: int = 5, max_n: Union[int, str] = 'full',
                  random_seed: Union[int, str, None] = 'default') -> dict:
    """
    The local dimension and persistence of the reconstructed attractor, from extreme-value
    statistics of close returns.

    Time-delay embeds the series and, for a sample of reference points ("poles") on the
    reconstructed orbit, treats close returns of the orbit to each pole as extreme events:
    ``g_i = -log(||Y_i - pole||)`` is large exactly when the orbit passes close to the pole.
    Extreme value theory applied to this observable gives two local quantities per pole:

    - a local dimension ``d(pole)``: under the Freitas-Freitas-Todd theorem [1], the Gumbel-law
      scale parameter of the extreme value law for ``g_i`` equals the local dimension of the
      attractor at that pole exactly. In practice this is estimated by a peaks-over-threshold
      fit [2, 3, 4]: take the exceedances of ``g_i`` above a high quantile ``q``, and set
      ``d(pole) = 1 / mean(exceedances)`` (the reciprocal of the exponential maximum-likelihood
      scale, i.e. the generalized Pareto fit with shape fixed at its ansatz-implied value of 0,
      appropriate here because ``g_i = -log(distance)`` is unbounded above, putting it in the
      Gumbel/exponential-tail domain).
    - a persistence ``theta(pole)`` (the "extremal index" of ``g_i`` at that pole): whether
      close returns to the pole arrive as isolated events (``theta`` near 1) or cluster into
      runs where the orbit lingers nearby (``theta`` well below 1, i.e. long average residence
      time near that point of phase space; ``1 / theta`` is the average cluster/sojourn size).
      Estimated with the O'Brien order-``m_order`` estimator, which Caby et al. [5] found more
      reliable for this observable than the Suveges likelihood estimator [6], particularly near
      near-periodic (sticky) poles.

    This differs from the attractor-dimension operations that pool all pairwise distances or
    neighbor ranks into one global scaling exponent: it estimates a genuinely *local* dimension
    and persistence separately at each of several poles and reports how they are distributed
    (and covary) across the attractor, capturing multifractal-style local heterogeneity that a
    single global exponent cannot.

    References
    ----------
    .. [1] A.C.M. Freitas, J.M. Freitas and M. Todd, "Hitting time statistics and extreme value
        theory", Probab. Theory Relat. Fields 147(3-4), 675-710 (2010).
    .. [2] V. Lucarini, D. Faranda, A.C.G.M.M. de Freitas, J.M. de Freitas, M. Holland, T. Kuna,
        M. Nicol, M. Todd and S. Vaienti, "Extremes and Recurrence in Dynamical Systems",
        Wiley (2016).
    .. [3] D. Faranda, G. Messori and P. Yiou, "Dynamical proxies of North Atlantic
        predictability and extremes", Sci. Rep. 7, 41278 (2017).
    .. [4] D. Faranda, J.M. Freitas, P. Guiraud and S. Vaienti, "Sampling local properties of
        attractors via extreme value theory", Chaos Solitons Fractals 74, 55-66 (2015).
    .. [5] Th. Caby, D. Faranda, S. Vaienti and P. Yiou, "On the computation of the extremal
        index for time series", J. Stat. Phys. 179(5-6), 1666-1697 (2019) (Eqs 19/21).
    .. [6] M. Suveges, "Likelihood estimation of the extremal index", Extremes 10(1-2),
        41-55 (2007).

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored).
    tau : int or str, optional
        The embedding time delay: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``: the first zero-crossing of the autocorrelation
        function, ``'ac1e'``: the floor of its first 1/e crossing, ``'mi'``: the smaller of the
        first minimum of the Kraskov automutual information and the ``'ac1e'`` delay).
        Default is ``'ac'``.
    m : int or str, optional
        The embedding dimension (an integer, or ``'fnn'`` for false nearest neighbors).
        Default is 3.
    q : float, optional
        The quantile level defining "extreme" close returns: exceedances of ``g_i`` above its
        ``q``-quantile are treated as events (default 0.98, i.e. the closest 2% of returns to
        each pole).
    theiler_win : int, float or ``['ac', k]``, optional
        The Theiler window excluding temporally-correlated neighbors of each pole from being
        treated as (trivially close) returns (see :func:`pyhctsa.utils.theiler_window`):
        ``['ac', k]`` for ``k`` times the first zero-crossing of the autocorrelation function,
        or a number of samples. Default is ``['ac', 1]``.
    n_poles : int, optional
        The number of reference points (poles) to sample from the embedded orbit (the cost is
        ``O(n_poles * Nemb)``). Default is 200.
    m_order : int, optional
        The order of the O'Brien persistence estimator: how many steps ahead to check for a
        further exceedance before counting a given exceedance as "isolated" (default 5,
        following Caby et al.).
    max_n : int or 'full', optional
        The maximum number of samples to consider (the first ``max_n``); ``'full'`` for no
        cropping (a warning is logged above 50000 samples). Default is ``'full'``.
    random_seed : int, str or None, optional
        The seed for sampling the poles, as hctsa's ``BF_RandomSeed``: an integer, ``'default'``
        (seed 0), or ``None``/``'none'`` (a seed from NumPy's global stream). The poles are the
        first of a :func:`~pyhctsa.robust.bf_random` permutation, so they are the same as
        hctsa's. Default is ``'default'``.

    Returns
    -------
    dict or float
        NaN if the embedding or Theiler window cannot be determined, or the embedded series is
        too short for the exceedances required. Otherwise:

        - ``propValidPoles``: the proportion of poles that gave a valid local dimension (at least
          15 exceedances; such a pole also has a valid persistence, as long as ``m_order`` is
          smaller than the number of returns used): a diagnostic of whether ``q``, ``n_poles`` and
          the series length were adequate, not a property of the dynamics
        - ``meanLocalDim``, ``stdLocalDim``: mean and standard deviation of the local dimension
          across poles
        - ``meanTheta``, ``stdTheta``: mean and standard deviation of the persistence (extremal
          index) across poles
        - ``corrDimTheta``: correlation across poles between the local dimension and the
          persistence (NaN if fewer than 10 poles are valid)
    """
    y = np.asarray(y, dtype=float).ravel()
    if isinstance(max_n, str) and max_n == 'full' and y.size > 50000:
        logger.warning(f"Time series ({y.size} samples) exceeds 50000 with max_n='full'; "
                       'computation may be slow')
    y = _check_max_n(y, max_n, 'extreme-value analysis')

    Y = _bf_embed(y, tau, m)
    if Y is None:
        return np.nan
    n_emb = Y.shape[0]

    theiler = theiler_window(y, theiler_win, n_emb)
    if np.isnan(theiler):  # the autocorrelation function never crosses zero
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    theiler = int(theiler)

    min_exceed = 15  # the minimum number of exceedances for a pole's estimate to be trusted
    min_n_emb = int(np.ceil(min_exceed / (1 - q))) + 2 * theiler + m_order + 1
    if n_emb < max(min_n_emb, 100):
        return np.nan

    # Sample poles (reference points) from the embedded orbit
    n_poles = min(int(n_poles), n_emb)
    pole_idx = _random_subset(n_emb, n_poles, random_seed)

    local_dim = np.full(n_poles, np.nan)
    theta = np.full(n_poles, np.nan)
    idx = np.arange(n_emb)
    for p, j in enumerate(pole_idx):
        # Exclude the Theiler window around this pole, keeping the rest of the orbit in its
        # original chronological order
        keep = np.abs(idx - j) > theiler
        dist_j = np.sqrt(np.sum((Y[keep] - Y[j]) ** 2, axis=1))
        dist_j = dist_j[dist_j > 0]  # excludes exact duplicate embedded points
        if dist_j.size < min_exceed / (1 - q):
            continue
        g = -np.log(dist_j)  # (chronological order preserved)

        u = matlab_quantile(g, q)[0]
        exceed = g > u  # exceedance indicator, in chronological order
        n_u = int(exceed.sum())
        if n_u < min_exceed:
            continue

        # Local dimension: the reciprocal of the mean exceedance (the exponential, or
        # shape-0 generalized Pareto, scale maximum-likelihood estimate)
        local_dim[p] = 1 / np.mean(g[exceed] - u)

        # Persistence: the O'Brien order-m_order estimator (Caby et al. 2019, Eq 19),
        # vectorized via a cumulative sum of the exceedance indicator
        n_tot = exceed.size
        if n_tot > m_order + 1:
            cum_e = np.concatenate(([0], np.cumsum(exceed)))
            i = np.arange(n_tot - m_order)
            future_sum = cum_e[i + 1 + m_order] - cum_e[i + 1]  # exceedances in the next m_order steps
            is_isolated = exceed[i] & (future_sum == 0)
            theta[p] = min((is_isolated.sum() / (n_tot - m_order)) / (n_u / n_tot), 1)  # clip finite-sample overshoot

    def nanmean(x):
        return np.mean(x[~np.isnan(x)]) if np.any(~np.isnan(x)) else np.nan

    def nanstd(x):
        x = x[~np.isnan(x)]
        return (np.std(x, ddof=1) if x.size > 1 else 0.0) if x.size else np.nan

    valid = ~np.isnan(local_dim) & ~np.isnan(theta)
    out = {}
    out['propValidPoles'] = np.mean(~np.isnan(local_dim))
    out['meanLocalDim'] = nanmean(local_dim)
    out['stdLocalDim'] = nanstd(local_dim)
    out['meanTheta'] = nanmean(theta)
    out['stdTheta'] = nanstd(theta)
    if valid.sum() >= 10 and np.std(local_dim[valid]) > 0 and np.std(theta[valid]) > 0:
        out['corrDimTheta'] = np.corrcoef(local_dim[valid], theta[valid])[0, 1]
    else:
        out['corrDimTheta'] = np.nan
    return out


# ------------------------------------------------------------------------------
# TISEAN c1 (fixed-mass information dimension) and lyap_spec (Lyapunov spectrum)
# ------------------------------------------------------------------------------

F32 = np.float32

# digamma at the integers 0..20 (as tabulated in TISEAN's d1.f)
_C1_PSI = np.array([0., -0.57721566490, 0.42278433509, 0.92278433509, 1.25611766843,
                    1.50611766843, 1.70611766843, 1.87278433509, 2.01564147795, 2.14064147795,
                    2.25175258906, 2.35175258906, 2.44266167997, 2.52599501330, 2.60291809023,
                    2.67434666166, 2.74101332832, 2.80351332832, 2.86233685773, 2.91789241329,
                    2.97052399224], dtype=np.float32)


def _expf(x) -> np.float32:
    with np.errstate(over='ignore', under='ignore'):
        return F32(np.exp(np.float64(x)))


def _logf(x) -> np.float32:
    with np.errstate(divide='ignore', invalid='ignore'):
        return F32(np.log(np.float64(x)))


@njit(cache=True)
def _slatec_rand(state):
    """SLATEC's ``RAND(0.)`` (the generator TISEAN's Fortran programs use): one draw in (0, 1)."""
    iy0 = 1029 * state[1]
    iy1 = 1536 * state[0] + 507 * (state[1] - state[0]) + iy0
    iy0 = iy0 + 1731
    state[1] = iy0 % 2048
    iy1 = iy1 + (iy0 - state[1]) // 2048
    state[0] = iy1 % 2048
    return np.float32(state[0] * 2048 + state[1]) / np.float32(4194304.0)


@njit(cache=True)
def _c1_shuffle(nmax, m, delay, state, slots, ju):
    """
    The permutation of the reference points in TISEAN's d1.f. Its swap partner is
    ``int(rand * nmax - (m-1) * delay) + 1`` (rather than ``rand * (nmax - (m-1) * delay)``),
    which for ``m >= 3`` is sometimes zero or negative, i.e. outside the array. Those writes and
    reads are kept, on a persistent ``slots`` array that starts at zero, so that the points
    chosen are the same as in the TISEAN binary. The list ``ju`` is also kept between calls: it is
    refilled only up to ``nmax - (m-1) * delay``, so if more reference points are requested than
    that (``-n``) the rest are what earlier calls left there (zeros, at first).
    """
    n = nmax - (m - 1) * delay
    for i in range(n):
        ju[i] = i + (m - 1) * delay + 1
    for i in range(1, n + 1):
        r = _slatec_rand(state)
        x = np.float32(r * np.float32(nmax)) - np.float32((m - 1) * delay)
        iperm = min(int(x) + 1, n)
        ih = ju[i - 1]
        if iperm >= 1:
            ju[i - 1] = ju[iperm - 1]
            ju[iperm - 1] = ih
        else:
            ju[i - 1] = slots[-iperm]
            slots[-iperm] = ih


@njit(cache=True)
def _c1_kth_distance(ye, off, refs, delay, m, ncomp, nmin, k):
    """
    For each reference point, the k-th smallest max-norm distance (float32) between its
    m-dimensional delay vector and those of the points outside its (cyclic) Theiler window; NaN
    where there are fewer than k such points. (TISEAN finds these by growing a box search until
    k neighbors are inside it, then taking the k-th distance, which is this.)
    """
    lo = (m - 1) * delay + 1
    hi = ncomp + (m - 1) * delay
    out = np.empty(refs.size, dtype=np.float32)
    buf = np.empty(max(hi - lo + 1, 1), dtype=np.float32)
    for r in range(refs.size):
        n = refs[r]
        cnt = 0
        for q in range(lo, hi + 1):
            nmd = abs(q - n) % ncomp
            if nmd <= nmin or nmd >= ncomp - nmin:
                continue
            dis = np.float32(0.0)
            for i in range(m):
                dx = abs(ye[off + n - i * delay] - ye[off + q - i * delay])
                if dx > dis:
                    dis = dx
            buf[cnt] = dis
            cnt += 1
        if cnt < k:
            out[r] = np.nan
        else:
            out[r] = np.partition(buf[:cnt], k - 1)[k - 1]
    return out


@njit(cache=True)
def _f32_cumsum_last(v):
    s = np.float32(0.0)
    for i in range(v.size):
        s = np.float32(s + v[i])
    return s


def _c1_curves(y: np.ndarray, delay: int, m_from: int, m_to: int, nmin: int, n_ref: int,
               kmax: int = 100, res: float = 2.0) -> Union[list, None]:
    """
    TISEAN's ``c1 -d<delay> -m<m_from> -M<m_to> -t<nmin> -n<n_ref>``: for each embedding
    dimension, the table of [mean log distance to the k-th neighbor -> exp, mass] pairs (as
    float32, as TISEAN writes them) for masses doubling every ``1/res`` octaves. Everything
    that is single precision in the Fortran is single precision here. None if TISEAN could not
    finish (it would search for neighbors for ever).
    """
    y32 = np.array([float('%.7g' % v) for v in y], dtype=np.float32)  # BF_WriteTempFile
    nmax = y32.size
    # the data sit at 1..nmax of `ye`; what lies before them is not data: zero, except that the
    # variable `res` happens to sit at index -1 in the compiled c1 (read when a reference point
    # of the permutation above falls outside the array)
    off = (m_to - 1) * delay + 1
    ye = np.zeros(off + nmax + 1, dtype=np.float32)
    ye[off + 1:off + 1 + nmax] = y32
    ye[off - 1] = res
    sd = F32(math.sqrt(float(F32(_f32_cumsum_last(((y32 - F32(_f32_cumsum_last(y32) / F32(nmax)))
                                                   ** 2).astype(np.float32)) / F32(nmax)))))
    state = np.zeros(2, np.int64)
    _slatec_rand(state)  # the program draws once when it seeds the generator
    slots = np.zeros(max((m_to - 1) * delay, 1) + 2, np.int64)
    ju = np.zeros(max(nmax, n_ref) + 1, np.int64)
    resl = F32(_logf(F32(2.0)) / F32(res))
    sqrt2 = F32(math.sqrt(2.0))
    curves = []
    rln = F32(0.0)
    for m in range(m_from, m_to + 1):
        rows = []
        pr = F32(0.0)
        if nmax - (m - 1) * delay < 1:  # the delay vectors are longer than the series
            return None
        pl = _logf(F32(F32(1.0) / F32(nmax - (m - 1) * delay)))
        # (a Fortran DO loop over a real: the trip count is fixed first, the variable then accumulates)
        for _ in range(int(F32(F32(F32(0.0) - pl) + resl) / resl)):
            ncomp = nmax - (m - 1) * delay
            n_eff = ncomp - 2 * nmin - 1
            if n_eff < 1:  # every point lies in every other's Theiler window: TISEAN never finds neighbors
                return None
            kpr = int(F32(_expf(pr) * F32(n_eff))) + 1
            k = int(F32(_expf(pl) * F32(n_eff))) + 1
            if k > kmax:  # fixed mass: use fewer points rather than more neighbors
                ncomp = int(F32(F32(F32(n_eff) * F32(kmax)) / F32(k)) + F32(2 * nmin + 1))
                k = kmax
            psi = _C1_PSI[k] if k <= 20 else F32(_logf(F32(k)) - F32(F32(1.0) / F32(2.0 * k)))
            pln = F32(psi - _logf(F32(ncomp - 2 * nmin - 1)))
            if k != kpr:
                _c1_shuffle(nmax, m, delay, state, slots, ju)
                e = _c1_kth_distance(ye, off, ju[:n_ref], delay, m, ncomp, nmin, k)
                if np.isnan(e).any():
                    return None
                # TISEAN sweeps the reference points with a neighborhood size that grows by
                # sqrt(2) until each has k neighbors; the sum of logs is in that order
                eps = F32(_expf(F32(pln / F32(m))) * sd)
                eps_seq = [eps]
                while eps <= e.max() and len(eps_seq) < 500:
                    eps = F32(eps * sqrt2)
                    eps_seq.append(eps)
                sweep = np.searchsorted(np.array(eps_seq, dtype=np.float32), e, side='right')
                order = np.argsort(sweep, kind='stable')
                # (each sweep moves the points still waiting to the front of ``ju``)
                waiting, wait_sweep = ju[:n_ref].copy(), sweep
                for s_i in range(int(sweep.max()) + 1):
                    keep = wait_sweep > s_i
                    waiting, wait_sweep = waiting[keep], wait_sweep[keep]
                    ju[:waiting.size] = waiting
                log_e = np.log(np.maximum(e[order].astype(np.float64), 1e-20)).astype(np.float32)
                rln = F32(_f32_cumsum_last(log_e) / F32(n_ref - (m - 1) * delay))
            if pln != pr:
                pr = pln
                rows.append((_expf(rln), _expf(pln)))
            pl = F32(pl + resl)
        curves.append(np.array(rows, dtype=np.float32).reshape(-1, 2))
    return curves


def _c2d_slopes(curves: list, iav: int = 2) -> list:
    """TISEAN's ``c2d -a<iav>``: local slopes of log mass against log distance over windows of
    ``2 * iav + 1`` points (float32), keeping the positive ones."""
    n = 2 * iav + 1
    out = []
    for rows in curves:
        e = np.array([_logf(F32(r[0])) for r in rows], dtype=np.float32)
        c = np.array([_logf(F32(r[1])) for r in rows], dtype=np.float32)
        block = []
        for j in range(iav, len(rows) - iav):
            x, v = e[j - iav:j + iav + 1], c[j - iav:j + iav + 1]
            sx = F32(0.0)
            for xi in x:
                sx = F32(sx + xi)
            sa, a = F32(0.0), F32(0.0)
            for i in range(n):
                dev = F32(x[i] - F32(sx / F32(n)))
                sa = F32(sa + F32(dev * dev))
                a = F32(a + F32(v[i] * dev))
            slope = F32(a / sa)
            if slope > 0:
                block.append((_expf(F32(F32(0.5) * F32(e[j + iav] + e[j - iav]))), slope))
        # (written by TISEAN with nine significant digits, read back by MATLAB as doubles)
        out.append(np.array([(float('%.9g' % a), float('%.9g' % b)) for a, b in block]).reshape(-1, 2))
    return out


def _c1_scaling_range(slopes: np.ndarray) -> Union[tuple, None]:
    """
    hctsa's search (inside NL_c1) for the length scales over which the slopes are most nearly
    constant: a start in the first quarter and an end in the last three quarters, scoring the
    standard deviation across the range less 0.005 per point it spans. Returns the 1-based
    ``(start, end, score, mean, std)``, or None if there are too few points.
    """
    n = slopes.size
    stptr = np.arange(1, n // 4)
    endptr = np.arange(int(np.ceil(n / 4)) + 1, n + 1)
    if stptr.size == 0 or endptr.size == 0:
        return None
    a, b, best = _best_flat_range(slopes, 0.005, stptr, endptr, normalize=False)
    if a is None:
        return None
    rng = slopes[stptr[a] - 1:endptr[b]]
    return stptr[a], endptr[b], best, np.mean(rng), np.std(rng, ddof=1)


def tisean_c1(y: ArrayLike, tau: Union[int, str] = 1, mmm: Union[list, tuple] = (2, 10),
              tsep: Union[int, float] = 0.02, nref: Union[int, float] = 0.5) -> Union[dict, float]:
    """
    Information dimension by the fixed-mass method (TISEAN's ``c1``).

    For a range of embedding dimensions, ``c1`` finds the distance to the k-th nearest neighbor
    (k increasing with the fraction of the points, the "mass", included) of a set of reference
    points, and the information dimension is the slope of the log mass against the log of the
    mean log distance (``c2d`` averages the local slope over five points). hctsa looks for a
    scaling range in each dimension and summarizes the dimension estimates across them.

    The routine is a port of TISEAN 3.0.1's Fortran (``c1.f``, ``d1.f``, ``c2d.f``) in single
    precision, and reproduces the binary's output, including the permutation of the reference
    points (its random numbers are SLATEC's ``RAND``) and the out-of-range indices it contains for
    ``m >= 3``.

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        The time delay: an integer or a rule understood by :func:`pyhctsa.utils.get_tau`
        (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.
    mmm : [m_min, m_max], optional
        The minimum and maximum embedding dimension. Default is ``[2, 10]``.
    tsep : int or float, optional
        The minimum time separation of neighbors (the Theiler window), in samples, or a proportion
        of the series length if between 0 and 1 (exclusive). Default is 0.02.
    nref : int or float, optional
        The number of reference points: a number of samples, or a proportion of the series
        length if at most 1. At most 2500 and, if the series is longer than 100 points, at least
        100. Default is 0.5.

    Returns
    -------
    dict
        Properties of the dimension estimates of the scaling ranges: ``bestestd`` and
        ``bestestdstd`` (the mean and standard deviation of the dimension over the scaling range
        with the best score), ``bestgoodness`` (that score), ``mediand``, ``mind``, ``maxd``,
        ``ranged`` (across embedding dimensions), ``maxmd`` (at the largest embedding dimension),
        ``meanstd`` (the mean of the standard deviations), ``bestscrd`` and ``longestscr`` (the
        dimension at, and log-length of, the longest scaling range). NaN if the series has fewer
        than 100 points or is constant, the delay cannot be set, or no scaling range is found.
        (Series whose length leaves a remainder of at most 6 on division by 128 lose their last
        samples, as in hctsa, because TISEAN's ``c1`` stalls on them.)
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 100:
        logger.warning('Time series too short for c1')
        return np.nan
    freaky = n % 128
    if freaky <= 6:
        logger.warning(f"TISEAN's c1 stalls on a series of this length: ignoring the last {freaky + 1} points")
        y = y[:n - (freaky + 1)]
        n = y.size
    if np.unique(y).size == 1:
        return np.nan

    if isinstance(tau, str) and tau not in ('ac', 'ac1e', 'mi'):
        raise ValueError(f"Unknown time delay '{tau}'")
    tau = get_tau(y, tau)
    if np.isnan(tau):
        return np.nan
    tau = int(tau)

    if len(mmm) != 2:
        raise ValueError('Please set a minimum and maximum embedding dimension as a length-2 vector')
    if 0 < tsep < 1:  # a proportion of the data length
        tsep = _round_half_away(tsep * n)
    if 0 < nref <= 1:
        nref = int(np.ceil(nref * n))
    nref = min(int(nref), 2500)
    if nref < 100 and n > 100:
        nref = 100

    curves = _c1_curves(y, tau, int(mmm[0]), int(mmm[1]), int(tsep), nref)
    if curves is None:
        logger.warning("TISEAN's c1 could not find enough neighbors")
        return np.nan
    slopes = _c2d_slopes(curves)

    sc = np.full((len(slopes), 6), np.nan)
    for i, block in enumerate(slopes):
        found = _c1_scaling_range(block[:, 1]) if block.shape[0] else None
        if found is None:
            logger.warning('Could not find a scaling range in the TISEAN c1 output (time series too short?)')
            return np.nan
        start, end, goodness, mean, std = found
        sc[i] = [block[start - 1, 0], block[end - 1, 0], goodness, mean, std,
                 np.log(block[end - 1, 0]) - np.log(block[start - 1, 0])]

    best = int(np.argmin(sc[:, 2]))
    longest = int(np.argmax(sc[:, 5]))
    out = {}
    out['bestestd'] = sc[best, 3]
    out['bestestdstd'] = sc[best, 4]
    out['bestgoodness'] = np.min(sc[:, 2])
    out['mediand'] = np.median(sc[:, 3])
    out['mind'] = np.min(sc[:, 3])
    out['maxd'] = np.max(sc[:, 3])
    out['ranged'] = np.ptp(sc[:, 3])
    out['maxmd'] = sc[-1, 3]
    out['meanstd'] = np.mean(sc[:, 4])
    out['bestscrd'] = sc[longest, 3]
    out['longestscr'] = np.max(sc[:, 5])
    return out


@njit(cache=True)
def _tisean_rnd_deltas(n, seed):
    """TISEAN's ``rnd_init(seed)``, 10000 warm-up calls of ``rnd_long``, then ``n * n`` values
    ``rnd_long() / ULONG_MAX`` (the random starting vectors of ``lyap_spec``)."""
    factor = np.uint64(13) ** np.uint64(13)
    arr = np.empty(9689, dtype=np.uint64)
    arr[0] = np.uint64(seed)
    for i in range(1, 9689):
        arr[i] = factor * arr[i - 1] + np.uint64(1)
    t, t1, t2, t3 = 9688, 9688 - 157, 9688 - 314, 9688 - 471
    out = np.empty((n, n))
    for c in range(10000 + n * n):
        t = (t + 1) % 9689
        t1 = (t1 + 1) % 9689
        t2 = (t2 + 1) % 9689
        t3 = (t3 + 1) % 9689
        arr[t] = arr[t] ^ arr[t1] ^ arr[t2] ^ arr[t3]
        if c >= 10000:
            out[(c - 10000) // n, (c - 10000) % n] = np.float64(arr[t]) / 18446744073709551615.0
    return out


@njit(cache=True)
def _gram_schmidt(delta):
    """Orthonormalize the rows of ``delta`` in place (classical Gram-Schmidt, as TISEAN); returns
    their lengths before normalization."""
    n = delta.shape[0]
    dnew = np.zeros((n, n))
    stretch = np.zeros(n)
    for i in range(n):
        diff = np.zeros(n)
        for j in range(i):
            nrm = 0.0
            for k in range(n):
                nrm += delta[i, k] * dnew[j, k]
            for k in range(n):
                diff[k] -= nrm * dnew[j, k]
        nrm = 0.0
        for j in range(n):
            nrm += (delta[i, j] + diff[j]) ** 2
        nrm = np.sqrt(nrm)
        stretch[i] = nrm
        for j in range(n):
            dnew[i, j] = (delta[i, j] + diff[j]) / nrm
    delta[:, :] = dnew
    return stretch


@njit(cache=True)
def _solve_in_place(mat, vec):
    """TISEAN's ``solvele`` (Gaussian elimination with row pivoting, in place); False if singular."""
    n = vec.size
    for i in range(n - 1):
        mx = abs(mat[i, i])
        maxi = i
        for j in range(i + 1, n):
            if abs(mat[j, i]) > mx:
                mx = abs(mat[j, i])
                maxi = j
        if maxi != i:
            for c in range(n):
                tmp = mat[i, c]
                mat[i, c] = mat[maxi, c]
                mat[maxi, c] = tmp
            tmp = vec[i]
            vec[i] = vec[maxi]
            vec[maxi] = tmp
        pivot = mat[i, i]
        if pivot == 0.0:
            return False
        for j in range(i + 1, n):
            q = -mat[j, i] / pivot
            mat[j, i] = 0.0
            for k in range(i + 1, n):
                mat[j, k] += q * mat[i, k]
            vec[j] += q * vec[i]
    vec[n - 1] /= mat[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(n - 1, i, -1):
            vec[i] -= mat[i, j] * vec[j]
        vec[i] /= mat[i, i]
    return True


_LYAP_BOX = 512  # cells per axis of the neighbor search grid, as in lyap_spec.c


@njit(cache=True)
def _lyap_neighbors(S, act, theiler, k_nn, epsmin, box, link, found, dist):
    """
    The neighbors of point ``act`` as TISEAN's ``lyap_spec`` finds them, in its order: a grid of cells
    (on the first and last coordinates) of side epsilon, grown by 1.2 until more than ``k_nn`` points
    lie within epsilon in every coordinate (the point itself, and the points inside the Theiler
    window dropped); the ``k_nn`` nearest of them, by the selection sort in ``sort()`` (so ties are
    resolved as there). Returns (number found, average distance of the last neighbor).
    """
    L, m = S.shape
    nl = L - 1  # points that have a successor
    ib = _LYAP_BOX - 1
    epsilon = epsmin / 1.2
    nfound = 0
    foundeps = 0.0
    while True:
        epsilon *= 1.2
        if epsilon > 1.0:
            epsilon = 1.0
        for x in range(_LYAP_BOX):
            for y in range(_LYAP_BOX):
                box[x, y] = -1
        for i in range(nl):
            x = int(S[i, 0] / epsilon) & ib
            y = int(S[i, m - 1] / epsilon) & ib
            link[i] = box[x, y]
            box[x, y] = i
        nfound = 0
        i0 = int(S[act, 0] / epsilon) & ib
        j0 = int(S[act, m - 1] / epsilon) & ib
        for i1 in range(i0 - 1, i0 + 2):
            for j1 in range(j0 - 1, j0 + 2):
                element = box[i1 & ib, j1 & ib]
                while element != -1:
                    inside = True
                    for c in range(m):
                        if abs(S[act, c] - S[element, c]) > epsilon:
                            inside = False
                            break
                    if inside:
                        found[nfound] = element
                        nfound += 1
                    element = link[element]
        if theiler > 0:  # drop neighbors within the Theiler window (the point itself is kept)
            kept = 0
            for q in range(nfound):
                if found[q] == act or abs(found[q] - act) > theiler:
                    found[kept] = found[q]
                    kept += 1
            nfound = kept
        if nfound > k_nn:
            # sort(): distances to all but the point itself, which is swapped to the end
            self_at = 0
            for q in range(nfound):
                if found[q] != act:
                    d = 0.0
                    for c in range(m):
                        dx = abs(S[act, c] - S[found[q], c])
                        if dx > d:
                            d = dx
                    dist[q] = d
                else:
                    self_at = q
            if self_at != nfound - 1:
                dist[self_at] = dist[nfound - 1]
                found[self_at] = found[nfound - 1]
            for i in range(k_nn):
                for j in range(i + 1, nfound - 1):
                    if dist[j] < dist[i]:
                        tmp = dist[i]
                        dist[i] = dist[j]
                        dist[j] = tmp
                        tmpi = found[i]
                        found[i] = found[j]
                        found[j] = tmpi
            return k_nn, dist[k_nn - 1]
        if epsilon >= 1.0:
            break
    return nfound, foundeps


@njit(cache=True)
def _lyap_spec_sums(S, k_nn, theiler, seed):
    """
    Sano and Sawada's method as in TISEAN's ``lyap_spec`` for the m columns of ``S`` (each
    rescaled to [0, 1]) as the state, mapped forward one sample: at each point a local affine
    map is fitted to the next values of its ``k_nn`` nearest neighbors (max norm, outside the
    Theiler window), and a set of m tangent vectors is carried along it and re-orthonormalized.
    Returns the status (0 ok, 50 too few neighbors, 1 singular fit), the sums of the log
    stretching rates and the number of steps.
    """
    L, m = S.shape
    delta = _tisean_rnd_deltas(m, seed)
    _gram_schmidt(delta)
    factor = np.zeros(m)
    nl = L - 1
    dyn = np.zeros((m, m))
    mat = np.zeros((m + 1, m + 1))
    hmat = np.zeros((m + 1, m + 1))
    imat = np.zeros((m + 1, m + 1))
    vec = np.zeros(m + 1)
    dist = np.empty(nl)
    found = np.empty(nl, dtype=np.int64)
    link = np.empty(nl, dtype=np.int64)
    box = np.empty((_LYAP_BOX, _LYAP_BOX), dtype=np.int64)
    epsmin = 1e-3
    aveps = 0.0
    count = 0
    for act in range(nl):
        count += 1
        nf, foundeps = _lyap_neighbors(S, act, theiler, k_nn, epsmin, box, link, found, dist)
        aveps += foundeps
        epsmin = aveps / count  # the neighborhood size to start the next search with
        if nf < k_nn:
            return 50, factor, count
        # normal equations of the local affine fit
        mat[:, :] = 0.0
        for a in range(nf):
            p = found[a]
            mat[0, 0] += 1.0
            for j in range(m):
                mat[0, j + 1] += S[p, j]
            for j in range(m):
                for k in range(j, m):
                    mat[j + 1, k + 1] += S[p, k] * S[p, j]
        for i in range(m + 1):
            for j in range(i, m + 1):
                mat[i, j] = mat[i, j] / nf
                mat[j, i] = mat[i, j]
        for i in range(m + 1):  # invert column by column
            for j in range(m + 1):
                vec[j] = 1.0 if i == j else 0.0
                hmat[j, :] = mat[j, :]
            if not _solve_in_place(hmat, vec):
                return 1, factor, count
            imat[:, i] = vec
        for d in range(m):
            vec[:] = 0.0
            for a in range(nf):
                p = found[a]
                hv = S[p + 1, d]
                vec[0] += hv
                for j in range(m):
                    vec[j + 1] += hv * S[p, j]
            vec /= nf
            for i in range(1, m + 1):
                dyn[d, i - 1] = 0.0
                for j in range(m + 1):
                    dyn[d, i - 1] += imat[i, j] * vec[j]
        dnew = np.zeros((m, m))  # carry the tangent vectors along the local map
        for i in range(m):
            for j in range(m):
                dnew[i, j] = dyn[j, 0] * delta[i, 0]
                for k in range(1, m):
                    dnew[i, j] += dyn[j, k] * delta[i, k]
        delta[:, :] = dnew
        stretch = _gram_schmidt(delta)
        for j in range(m):
            factor[j] += np.log(stretch[j])
    return 0, factor, count


def lyap_spec(y: ArrayLike, tau_method: Union[int, str] = 1, m: int = 3, k_nn: int = 30,
              max_n: Union[int, str] = 10000, theiler_win: Union[int, float, list, tuple] = ('ac', 1),
              random_seed: int = 42) -> Union[dict, float]:
    """
    The spectrum of Lyapunov exponents of the delay-embedded trajectory.

    Estimates all ``m`` exponents (not just the largest, cf. :func:`largest_lyap`) with TISEAN's
    ``lyap_spec``, an implementation of the method of Sano and Sawada: local affine maps of the
    embedded trajectory are fitted to the neighbors of each point, and tangent vectors propagated
    along them with periodic orthonormalization. From the spectrum come the rate of
    volume contraction (the sum of the exponents) and the Kaplan-Yorke estimate of the
    attractor's dimension. The routine is run in process (a port of ``lyap_spec.c``, which
    reproduces the TISEAN binary, including its random initial vectors).

    A little noise (0.001 of the standard deviation) is added to the series first, so that the
    local fits are well posed for quantized series; ``random_seed`` seeds it (NumPy's generator,
    not MATLAB's ``rng(42)`` stream).

    Parameters
    ----------
    y : array-like
        Input time series.
    tau_method : int or str, optional
        The time delay of the embedding: an integer, or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.
    m : int, optional
        The embedding dimension, and the number of exponents estimated (at least 3). Default is 3.
    k_nn : int, optional
        The number of neighbors used for each local fit. Default is 30.
    max_n : int or 'full', optional
        The maximum number of samples (the series is cropped to its first ``max_n`` samples), or
        ``'full'``. Default is 10000.
    theiler_win : int, float, or ``['ac', k]``, optional
        The Theiler window (see :func:`pyhctsa.utils.theiler_window`): neighbors closer in time
        than this are not used. Default is ``['ac', 1]``.
    random_seed : int, optional
        Seed of the added noise. Default is 42.

    Returns
    -------
    dict
        ``LE1``, ``LE2``, ``LE3`` (the three largest exponents, per sample), ``sumPos`` and
        ``numPos`` (the sum and number of positive exponents), ``sumAll`` (their sum) and ``KYdim``
        (the Kaplan-Yorke dimension). NaN if the delay or Theiler window cannot be set, the
        series is too short for the fits (fewer than ``10 * k_nn + 2 * theiler_win`` embedded
        points), TISEAN finds too few neighbors, or an exponent is not finite.
    """
    y = _check_max_n(np.asarray(y, dtype=float).ravel(), max_n, 'lyap_spec')
    n = y.size
    if m < 3:
        raise ValueError('The embedding dimension, m, must be at least 3 (the outputs include LE3)')

    y = y + 0.001 * np.std(y, ddof=1) * np.random.RandomState(random_seed).randn(n)

    params = _embedding_params(y, tau_method, m)
    if params is None:
        logger.warning('Could not determine a suitable time delay for this time series')
        return np.nan
    tau, m = params
    theiler = theiler_window(y, theiler_win, n)
    if np.isnan(theiler):
        logger.warning('No autocorrelation zero-crossing to set the Theiler window')
        return np.nan
    theiler = int(theiler)

    n_emb = n - (m - 1) * tau
    if n_emb < 10 * k_nn + 2 * theiler:
        logger.warning(f'Time series too short to estimate the Lyapunov spectrum (N = {n}, tau = {tau}, m = {m})')
        return np.nan
    # the embedding as TISEAN reads it (BF writes seven significant digits), each column rescaled
    emb = np.array([[float('%.7g' % v) for v in y[j * tau:j * tau + n_emb]] for j in range(m)]).T
    span = emb.max(axis=0) - emb.min(axis=0)
    if np.any(span == 0):
        return np.nan
    status, factor, count = _lyap_spec_sums((emb - emb.min(axis=0)) / span, int(k_nn), theiler, 0x098342)
    if status != 0:
        return np.nan
    # (TISEAN prints the averages with %e)
    le = np.array([float('%e' % (f / count)) for f in factor])
    if not np.all(np.isfinite(le)):
        return np.nan

    out = {'LE1': le[0], 'LE2': le[1], 'LE3': le[2]}
    out['sumPos'] = np.sum(le[le > 0])
    out['numPos'] = int(np.sum(le > 0))
    out['sumAll'] = np.sum(le)
    cum = np.cumsum(le)
    below = np.flatnonzero(cum < 0)
    if below.size == 0:
        out['KYdim'] = float(m)
    else:
        pos_k = int(below[0])  # the number of leading exponents whose cumulative sum is not negative
        out['KYdim'] = 0.0 if pos_k == 0 else pos_k + cum[pos_k - 1] / abs(le[pos_k])
    return out
