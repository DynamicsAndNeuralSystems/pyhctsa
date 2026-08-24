from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.signal import correlate

from ..operations.model_fit import residual_analysis, _ml_rng, _ml_randperm_k
from ..operations.correlation import first_crossing, autocorr, _resolve_time_delay
from ..operations.information import first_min
from ..operations.scaling import _round
from ..toolboxes.Tisean_3_0_1 import tisean as _tisean
from ..utils import _first_index_past_threshold, matlab_quantile, time_delay_embed

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
            warnings.warn(
                f"Time series ({n} samples) exceeds 50000 with "
                "maxN='full'; computation may be slow.",
                RuntimeWarning,
            )

    else:
        max_n = int(max_n)

        if n > max_n:
            warnings.warn(
                f"Time series ({n} samples) exceeds maxN={max_n}; "
                f"analyzing the first {max_n} samples.",
                RuntimeWarning,
            )
            y = y[:max_n]
            n = max_n

    if n < 200:
        warnings.warn(
            f"Time series (N={n}) too short for a meaningful "
            "0-1 test (need >= 200).",
            RuntimeWarning,
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
    idx = _first_index_past_threshold(p, threshold, over_or_under)

    return len(p) + 1 if idx is None else idx + 1

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

def _ms_nlpe(y: ArrayLike, de: int, tau: int) -> float:
    # helper function for nlpe
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
        rows[np.arange(stop - start), np.arange(start, stop)] = np.inf # exclude self
        near[start:stop] = np.argmin(rows, axis=1)

    e = y[near] - y  # now y is (m,) so y[near] works correctly

    return e

def nsamdf(x: ArrayLike, fs: Union[float, int] = 1.0, win_len_rel: Union[int, float] = 14,
           shift_len_rel: Union[float, int] = 0.5, lag_rel: Union[int, float] = 1,
           degree: int = 7) -> dict:
    """
    Computes the nonlinearity measure L through nsAMDF
    (nonlinear average magnitude difference function), developed by Ozkurt et al. [1].

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
    fs : float or int
        Sampling frequency in Hz. Default is 1.0.
    win_len_rel : float or int
        Window length (a long enough segment is important to estimate the nonlinearity). Default is 14.
    shift_len_rel : float or int
        This amounts to window length - overlap length btw windows. Default is 0.5.
    lag_rel : float or int
        TMaximum lag for nsAMDF, we chose it as 1. Default is 1.
    degree : The chosen degree p should ideally be large enough to capture the
           highest order of nonlinearity within the data. Default is 7.
    
    Returns
    -------
    float
        The nsAMDF nonlinearity measure L. 
    """
    window_length = int(win_len_rel * fs)
    shift_length = int(shift_len_rel * window_length)
    lag = int(fs * lag_rel)

    out = {}
    # nsAMDF for p = 2
    s2 = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length, lag=lag, nrmdegree=2)
    #out['s2'] = s2 / np.max(s2) # normalized

    # nsAMDF for p = degree:
    sd = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length, lag=lag, nrmdegree=degree)
    #out['sd'] = sd / np.max(sd) # normalized
    
    #% If you like, you can bandpass filter s2 and sd for the specific frequency band
    #% of nonlinear effect both to compute L and plot them as such
    out['L'] = np.linalg.norm(s2 - sd)

    return out

def nlpe(y: ArrayLike, de: int = 3, tau: Union[int, str] = 1, max_n: int = 5000) -> dict:
    """
    Normalized drop-one-out constant interpolation nonlinear prediction error.

    Computes the nlpe for a time-delay embedded time series using Michael Small's
    code, nlpe [1].

    Modifications by Joshua B. Moore for incorporating into pyhctsa.

    References
    ----------
    .. [1] M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
        Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
        Vol. 52 (2005)
    
    Parameters
    ----------
    y : array-like
        Input time series.
    de : int
        The embedding dimension. Default is 3.
    tau : int or str, optional
        The time-delay. Can be either an integer or ``'ac'`` to use the first
        zero-crossing of the ACF, or ``'mi'`` to use the first minimum of the
        automutual information function. Default is 1.
    max_n : int, optional
        The maximum length of the time series on which to compute the nlpe.
        Default is 5000.
    
    Returns
    -------
    dict
        Measures of the mean error of the nonlinear predictor, and a
        set of measures on the correlation, Gaussianity, etc. of the residuals.
    """

    n = len(y)

    if isinstance(tau, str):
        tau = _resolve_time_delay(y, tau)
        # check the tau
        if np.isnan(tau):
            logger.warning('Time series cannot be embedded (too short?)')
            return np.nan
    #% nlpe can cause memory pains for long time series
    #% Let's do this dirty cheat
    if n > max_n:
        # crop the time series to the first max_n samples
        y = y[:max_n]
        logger.info(f"Michael Small's nlpe code is only being evaluated on the first {max_n} (/{n}) samples.")
        n = max_n
    
    if n < 20: # short time series cause problems
        logger.warning(f'Time series (N = {len(y)}) is too short.')
        return np.nan

    # run the nonlinear prediction error code
    res = _ms_nlpe(y, de, tau)
    if np.isscalar(res) and np.isnan(res):
        # a scalar nan has been returned instead of expected array
        return np.nan

    # compute outputs
    out = {}
    out['msqerr'] = np.mean(res**2)
    res = residual_analysis(res)
    # combine with residual analysis results
    out = out | res

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
        The time-delay, can be an integer or 'ac', or 'mi' for first
        zero-crossing of the autocorrelation function or first minimum
        of the automutual information, respectively. Default is ``'ac'``.
    m : int
        The embedding dimension. Default is 3.
    
    Returns 
    -------
    dict 
        Various statistics summarizing the obtained eigenvalue distribution.

    """
    if isinstance(tau, str):
        tau = _resolve_time_delay(y, tau)
        if np.isnan(tau):
            logger.warning('Could not get time delay (time series too short?)')
            return np.nan
    try:
        y_embed = time_delay_embed(y, m, int(tau))
    except ValueError as e:  # embedding failed (time series too short)
        logger.warning(str(e))
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

def _ssa_diagonal_average(U: np.ndarray, sigma: np.ndarray, V: np.ndarray,
                          components, counts: np.ndarray) -> np.ndarray:
    """
    Reconstructs a length-N component series from an elementary matrix
    by averaging over its anti-diagonals ("Hankelization").
    """
    # The anti-diagonal sums of the rank-one matrix sigma_i*u_i*v_i' are the
    # (linear) convolution of u_i with v_i, so the elementary matrices never
    # have to be formed:
    diag_sums = np.zeros(len(counts))
    for i in components:
        diag_sums += sigma[i]*np.convolve(U[:, i], V[:, i])

    return diag_sums/counts

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

    c_lead = _ssa_diagonal_average(U, sigma, V, range(sep_idx), w)
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
        c_pair = _ssa_diagonal_average(U, sigma, V, (i_star, i_star+1), w)
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

def local_density(y: ArrayLike, nnr: int = 3, past: int = 40,
                  tau: Union[str, int] = 'ac', m: int = 2) -> dict:
    """
    Local density estimates in the time-delay embedding space.
    
    Computes a standard k-nearest-neighbor local density estimate at each
    point of the time-delay embedding: density(i) is proportional to
    1/r_NNR(i)^m, where r_NNR(i) is the distance from point i to its NNR-th
    nearest neighbor (excluding temporally-close points within a Theiler
    window of "past" samples) and m is the embedding dimension. 

    Parameters
    ----------
    y : array-like
        Input time series.
    nnr : int, optional
        Number of nearest neighbours to compute. Default is 3.
    past : int, optional
        Number of time-correlated points to discard (samples), i.e., the
        Theiler window. Default is 40.
    tau : str or int, optional
        The time-delay of the embedding, either an integer or ``'ac'`` for the
        first zero-crossing of the autocorrelation function. Default is ``'ac'``.
    m : int, optional
        The embedding dimension. Default is 2.

    Returns
    -------
    dict
        Various statistics on the local density estimates at each point in the
        time-delay embedding, including the minimum and maximum values, the
        range, the standard deviation, mean, median, and autocorrelation.
    """
    if isinstance(tau, str) and tau != 'ac':
        raise ValueError(f"Invalid time-delay method: '{tau}'. Only 'ac' (or an integer) is supported.")
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        logger.warning('Could not get time delay by ACF (time series too short?)')
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
    r_nnr = valid_dists[:, nnr-1]

    # Fall back to a full pairwise search wherever the over-fetch wasn't enough:
    for i in np.flatnonzero(valid.sum(axis=1) < nnr):
        all_dists = np.linalg.norm(y_embed - y_embed[i], axis=1)
        all_dists[np.abs(np.arange(n_embed) - i) <= past] = np.inf
        r_nnr[i] = np.sort(all_dists)[nnr-1]

    with np.errstate(divide='ignore', over='ignore'):
        locden = 1 / (r_nnr**m)

    if np.all(locden == 0) or np.any(~np.isfinite(locden)):
        return np.nan

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
    # first minimum of the automutual information function:
    out['taumiden'] = first_min(locden, 'mi')

    return out

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
        raise ValueError('time series is too short to contain a scaling range')

    results = np.full((ndim, 4), np.nan)
    for c in range(ndim):
        v = x[c, :]
        a, b, best = _best_flat_range(v, gamma, stptr, endptr)
        if a is None:
            continue
        results[c] = [stptr[a], endptr[b], best, np.mean(v[stptr[a] - 1:endptr[b]])]
    return results


def _sub_celltomat(blocks: list, column: int) -> tuple:
    # Stack one column of each per-dimension block into a matrix. Higher
    # embedding dimensions may not reach as far down in length scale, so first
    # restrict every block to the span they all share.
    blocks = [np.asarray(b, dtype=float) for b in blocks]
    if any(b.size == 0 for b in blocks):
        raise ValueError('no data returned by TISEAN for at least one dimension')

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
        v = np.diff(Y[c, :]) * dx  # vector of local gradients
        a, b, best = _best_flat_range(v, gamma, stptr, endptr)
        if a is None:
            continue
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
        v = np.diff(Y[c, :]) * dx
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
        raise ValueError('Error finding scaling range') from exc

    # rows: increasing embedding m; columns: stpt, endpt, goodness, dim
    out[f'ben{p}_mindim'] = np.min(benfind[:, 3])
    out[f'ben{p}_maxdim'] = np.max(benfind[:, 3])
    out[f'ben{p}_meandim'] = np.mean(benfind[:, 3])
    out[f'ben{p}_meangoodness'] = np.mean(benfind[:, 2])

    mmin = _sub_findmmin(benfind[:, 3])
    # minimum scale at which a scaling range is observed:
    out[f'benmmin{p}_logminl'] = (np.nan if mmin['ri1'] is None
                                  else np.log(dat_v[mmin['ri1'] - 1]))
    out[f'benmmin{p}_goodness'] = mmin['goodness']
    out[f'benmmin{p}_stabledim'] = mmin['stabled']
    out[f'benmmin{p}_linrmserr'] = mmin['linrmserr']

    # Reshaped: only for large enough m (as determined by the criteria above),
    # then find a scaling region across m for a saturated range of m.
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
              theiler_win: Union[int, float] = 0.01) -> Union[dict, float]:
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
        The time-delay. Can be an integer, or ``'ac'`` for the first
        zero-crossing of the autocorrelation function, or ``'mi'`` for the first
        minimum of the automutual information. Default is 1.
    maxm : int, optional
        The maximum embedding dimension. Default is 10.
    theiler_win : int or float, optional
        The Theiler window. A value in ``(0, 1)`` is taken as a proportion of the
        time-series length. Default is 0.01, i.e. 1% of the data length.

    Returns
    -------
    dict or float
        Statistics summarising Takens' estimator, the local slopes of the
        correlation sum (raw and Gaussian-kernel smoothed), and the correlation
        entropy. Returns NaN if the time series is too short.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size  # data length (number of samples)
    if n < 50:
        logger.warning(f'N = {n} too short for nonlinear dimension analysis')
        return np.nan

    # Time delay, tau
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        raise ValueError('Time series cannot be embedded (too short?)')
    tau = int(tau)

    # Theiler window
    if 0 < theiler_win < 1:  # specify proportion of time-series length
        theiler_win = round(theiler_win * n)
    theiler_win = int(theiler_win)

    tables = _tisean.d2(y, delay=tau, embed=maxm, theiler=theiler_win)
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
        raise ValueError('No data...')
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

def _findscalingr_c1(x: np.ndarray) -> np.ndarray:
    # As _findscalingr_ind, but for the fixed-mass curves: the candidate ranges
    # start in the first quarter rather than the first half, the reward for a
    # longer range is larger, and the spread is measured on the raw values
    # rather than on a rescaled copy. Returns
    # [start, end, goodness, dimension, dimension std].
    v = np.asarray(x, dtype=float).ravel()
    l = v.size
    gamma = 0.005  # regularization parameter, chosen empirically
    stptr = np.arange(1, int(np.floor(l / 4)))         # must be in the first quarter
    endptr = np.arange(int(np.ceil(l / 4)) + 1, l + 1)  # and in the last three
    if stptr.size == 0 or endptr.size == 0:
        raise ValueError('Error finding scaling range')

    mybad = np.empty((stptr.size, endptr.size))
    for i, s in enumerate(stptr):
        for j, e in enumerate(endptr):
            mybad[i, j] = np.std(v[s - 1:e], ddof=1)
    mybad -= gamma * (endptr[np.newaxis, :] - stptr[:, np.newaxis] + 1)

    a, b, best = _argmin_first_colmajor(mybad)
    if a is None:
        raise ValueError('Error finding scaling range')
    ri1, ri2 = int(stptr[a]), int(endptr[b])
    sub = v[ri1 - 1:ri2]
    return np.array([ri1, ri2, best, np.mean(sub), np.std(sub, ddof=1)])


def tisean_c1(y: ArrayLike, tau: Union[int, str] = 1, mmm: tuple = (2, 10),
              tsep: Union[int, float] = 0.02,
              nref: Union[int, float] = 0.5) -> Union[dict, float]:
    """
    Information dimension from the TISEAN package's ``c1`` routine.

    Runs the fixed-mass computation of the information dimension [1]_: for a
    geometrically spaced sequence of neighbour masses, ``c1`` reports the mean
    radius that encloses that mass, and ``c2d`` turns each curve into local
    slopes. Every embedding dimension then gets its own scaling range, found by
    a penalized search that trades the flatness of the slopes against the span
    of scales they hold over, and the returned statistics summarise the
    dimension estimates and the scaling ranges across dimensions.

    References
    ----------
    .. [1] R. Hegger, H. Kantz and T. Schreiber, "Practical implementation of
        nonlinear time series methods: The TISEAN package", Chaos 9(2) 413 (1999)

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        The time-delay. Can be an integer, or ``'ac'`` for the first
        zero-crossing of the autocorrelation function, or ``'mi'`` for the first
        minimum of the automutual information. Default is 1.
    mmm : tuple of int, optional
        Minimum and maximum embedding dimension. Default is ``(2, 10)``.
    tsep : int or float, optional
        Time separation between a reference point and the neighbours counted
        around it. A value in ``(0, 1)`` is taken as a proportion of the
        time-series length. Default is 0.02, i.e. 2% of the data length.
    nref : int or float, optional
        Number of reference points. A value in ``(0, 1]`` is taken as a
        proportion of the time-series length, and the result is held between 100
        and 2500. Default is 0.5, i.e. half the data.

    Returns
    -------
    dict or float
        Dimension estimates and scaling-range statistics across embedding
        dimensions. Returns NaN if the time series is too short or constant.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size  # time-series length (number of samples)

    if n < 100:
        logger.warning(f'N = {n} too short for c1')
        return np.nan

    # TISEAN's c1 stalls on lengths just past a multiple of 128, so hctsa drops
    # the last few points in that case; kept here so the two agree on which
    # samples are analysed.
    freaky_stat = n % 128
    if freaky_stat <= 6:
        y = y[:n - (freaky_stat + 1)]
        n = y.size

    if np.unique(y).size == 1:  # c1 also freezes on constant data
        return np.nan

    # Time delay, tau
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        raise ValueError('Time series cannot be embedded (too short?)')
    tau = int(tau)

    if len(mmm) != 2:
        raise ValueError('Please set a minimum and maximum embedding dimension '
                         'as a length-2 tuple')

    # Time separation, tsep
    if 0 < tsep < 1:  # specify proportion of time-series length
        tsep = _round(tsep * n)
    tsep = int(tsep)

    # Number of reference points, Nref
    if 0 < nref <= 1:  # specify proportion of time-series length
        nref = np.ceil(nref * n)
    nref = int(nref)
    nref = min(nref, 2500)  # for time reasons, don't use more than 2500
    if nref < 100 and n > 100:
        nref = 100          # and can't have fewer than 100

    c1dat = _tisean.c2d(_tisean.c1(y, delay=tau, mmin=int(mmm[0]),
                                   mmax=int(mmm[1]), tsep=tsep, nref=nref),
                        average=2)

    # Rows: increasing embedding m. Columns: start and end of the scaling range
    # (as length scales), its goodness, and the dimension estimate over it with
    # its standard deviation.
    c1sc = np.zeros((len(c1dat), 6))
    for i, block in enumerate(c1dat):
        c1sc[i, :5] = _findscalingr_c1(block[:, 1])
        c1sc[i, 0] = block[int(c1sc[i, 0]) - 1, 0]
        c1sc[i, 1] = block[int(c1sc[i, 1]) - 1, 0]
    c1sc[:, 5] = c1sc[:, 1] - c1sc[:, 0]  # length of the scaling range

    out = {}

    # Best fit embedding dimension:
    where_best_est = int(np.argmin(c1sc[:, 2]))
    out['bestestd'] = c1sc[where_best_est, 3]
    out['bestestdstd'] = c1sc[where_best_est, 4]
    out['bestgoodness'] = np.min(c1sc[:, 2])

    out['mediand'] = np.median(c1sc[:, 3])
    out['mind'] = np.min(c1sc[:, 3])
    out['maxd'] = np.max(c1sc[:, 3])
    out['ranged'] = np.max(c1sc[:, 3]) - np.min(c1sc[:, 3])
    out['maxmd'] = c1sc[-1, 3]
    out['meanstd'] = np.mean(c1sc[:, 4])

    # Longest scaling range estimate of the embedding dimension:
    where_longest_scr = int(np.argmax(c1sc[:, 5]))
    out['bestscrd'] = c1sc[where_longest_scr, 3]
    out['longestscr'] = np.max(c1sc[:, 5])

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
        The time-delay of the embedding. Can be an integer, or ``'ac'`` for the
        first zero-crossing of the autocorrelation function, or ``'mi'`` for the
        first minimum of the automutual information. Default is ``'mi'``.

    Returns
    -------
    dict or float
        Statistics on the x- and y-components of the vectors on the Poincare
        surface, on distances between adjacent points and from the mean
        position, and on the entropy of the boxed vector cloud. Returns NaN if
        fewer than two section points were found.
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

    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        logger.warning('Could not get time delay (time series too short?)')
        return np.nan
    tau = int(tau)

    # Embed in three dimensions, and cut on the last coordinate at TISEAN's own
    # default threshold (that coordinate's mean). hctsa reads the .poin file
    # back, so the section points are the ones TISEAN printed.
    v = _tisean.poincare(y, dim=3, delay=tau, comp=3, direction=direction,
                         as_written=True)

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
        # This probably needs to be normalized:
        out[f'hboxcounts{num_partitions}'] = -np.sum(pos * np.log(pos))
        out[f'tracepbox{num_partitions}'] = np.sum(np.diag(pbox))  # trace

    return out


def delay_time(y: ArrayLike, max_delay: Union[int, float] = 0.2, past: int = 1,
               random_seed: Union[int, None] = 0) -> dict:
    """
    Optimal delay time using the method of Parlitz and Wichard.


    Parameters
    ----------
    y : array-like
        Input time series.
    max_delay : int or float, optional
        Maximum value of the delay to consider. Values in (0, 1) are
        interpreted as a proportion of the time-series length. Delays below 10
        are raised to 10. Default is 0.2.
    past : int, optional
        Number of time-correlated points to discard (samples) when searching
        for value-neighbours, i.e., the Theiler window. Default is 1.
    random_seed : int or None, optional
        Seed for the Mersenne Twister used to draw the reference points (as
        ``BF_ResetSeed`` does). ``None`` leaves the stream alone, matching
        ``BF_ResetSeed('none')``. Default is 0.

    Returns
    -------
    dict
        The first three values of ``tau``, the differences between them, and
        the mean, standard deviation, minimum and maximum of ``tau``. Returns
        NaN if ``max_delay`` is too long for the given time series, or if no
        reference point clears the Theiler window.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    if 0 < max_delay < 1:
        max_delay = _round(N * max_delay)  # a proportion of the time-series length
    max_delay = int(max_delay)

    if max_delay < 10:
        max_delay = 10
        logger.warning('Max delay set to its minimum: delaytime = 10')
    if max_delay >= N/2:
        # Heuristic for appropriate time delay
        logger.warning(f'Max delay, {max_delay}, too long for time series of length {N}')
        return np.nan

    iterations = 64
    max_attempts = 10000
    length = N - max_delay
    # index[r] is the position in the time series of the (r+1)th smallest value
    index = np.argsort(y[:length], kind='stable')

    rng = np.random.RandomState() if random_seed is None else _ml_rng(random_seed)

    err = np.zeros(max_delay + 1)
    for _ in range(iterations):
        # Redraw until the reference point has a value-neighbour on both sides
        # that clears the Theiler window (MATLAB retries forever; the attempt
        # cap here only bites when almost no rank qualifies, i.e. when the
        # original would spin rather than terminate).
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
            logger.warning('No reference point with value-neighbours outside a '
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


_EVT_POLE_BLOCK = 64  # poles per distance-matrix block

def _evt_quantile(x: np.ndarray, p: float) -> float:
    """The `p`-quantile of `x`, from the two order statistics it interpolates."""
    n = x.size
    r = (100.0*p/100.0)*n
    k = np.floor(r + 0.5)
    kp1 = k + 1
    r = r - k

    k = 1 if (k < 1 or np.isnan(k)) else int(k)
    kp1 = min(int(kp1), n)
    xs = np.partition(x, (k - 1, kp1 - 1))
    xk, xkp1 = xs[k - 1], xs[kp1 - 1]

    if xk == xkp1 or r == -0.5:
        return xk

    return (0.5 + r)*xkp1 + (0.5 - r)*xk

def _omitnan_mean(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]

    return float(np.mean(x)) if x.size > 0 else np.nan

def _omitnan_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0

def evt_local_dim(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 3,
                  q: float = 0.98, theiler_win: Union[int, float] = 0.01,
                  n_poles: int = 200, m_order: int = 5, max_n: int = 10000,
                  random_seed: Union[int, None] = 0) -> dict:
    """
    Extreme value theory local dimension and persistence of the attractor.

    Time-delay embeds the series and, for a sample of reference points
    ("poles") on the reconstructed orbit, treats close returns of the orbit
    to each pole as extreme events: g_i = -log(||Y_i - pole||) is large
    exactly when the orbit passes close to the pole. Extreme value theory
    applied to this observable gives two local quantities per pole
    (Freitas, Freitas & Todd 2010; Lucarini et al., "Extremes and
    Recurrence in Dynamical Systems", 2016; Faranda, Messori & Yiou 2017):

      - a local dimension d(pole): under the Freitas-Freitas-Todd theorem,
        the Gumbel-law scale parameter of the extreme value law for g_i
        equals the local dimension of the attractor at that pole exactly
        (Faranda, Freitas, Guiraud & Vaienti 2014, Props 1-2). In practice
        this is estimated by a peaks-over-threshold fit: take the
        exceedances of g_i above a high quantile q, and set
        d(pole) = 1/mean(exceedances) (the reciprocal of the exponential
        MLE scale, i.e. the GPD fit with shape fixed at its ansatz-implied
        value of 0 -- appropriate here because g_i = -log(distance) is
        unbounded above, putting it in the Gumbel/exponential-tail domain).
      - a persistence theta(pole) (the "extremal index" of g_i at that
        pole): whether close returns to the pole arrive as isolated events
        (theta near 1) or cluster into runs where the orbit lingers nearby
        (theta << 1, i.e. long average residence time near that point of
        phase space -- 1/theta is the average cluster/sojourn size).
        Estimated with the O'Brien order-m estimator (Caby, Faranda,
        Vaienti & Yiou, J. Stat. Phys. 2019, Eqs 19/21), which that paper's
        own comparison against the alternative Suveges (2007) likelihood
        estimator found more reliable for this observable, particularly
        near near-periodic (sticky) poles.

    This differs mechanistically from the other attractor-dimension
    operations (fractal_dimensions' nearest-neighbour-moment generalized
    dimensions): those pool all pairwise distances/neighbour ranks into one
    global scaling exponent, whereas this estimates a genuinely *local*
    dimension and persistence separately at each of several poles and
    reports how they're distributed (and covary) across the attractor --
    capturing multifractal-style local heterogeneity that a single global
    exponent cannot.

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored).
    tau : int or str, optional
        Embedding time delay ('ac', the first zero-crossing of the
        autocorrelation function, or 'mi', the first minimum of the
        automutual information, or an integer). Default: 'ac'.
    m : int, optional
        The embedding dimension. Default: 3.
    q : float, optional
        The quantile level defining "extreme" close returns: exceedances of
        g_i = -log(distance) above its q-quantile are treated as events.
        Default: 0.98, i.e. the closest 2% of returns to each pole.
    theiler_win : int or float, optional
        Theiler window excluding temporally-correlated neighbours of each
        pole from being treated as (trivially close) returns (a proportion
        of the embedded length if in (0,1)). Default: 0.01.
    n_poles : int, optional
        Number of reference points (poles) to sample from the embedded
        orbit (cost is O(n_poles*Nemb)). Default: 200.
    m_order : int, optional
        The order m of the O'Brien persistence estimator -- how many steps
        ahead to check for a further exceedance before counting a given
        exceedance as "isolated". Default: 5, following Caby et al.
    max_n : int, optional
        Maximum number of samples to consider; longer series are cropped to
        their first max_n points. Default: 10000.
    random_seed : int or None, optional
        Seed for the Mersenne Twister used to sample the poles. ``None``
        leaves the stream alone. Default: 0.

    Returns
    -------
    dict
        Mean and standard deviation of the local dimension and persistence
        across poles, their correlation across poles, and the proportion of
        poles that yielded a valid estimate (too few exceedances otherwise
        -- a diagnostic of whether q/n_poles/series length were adequate,
        not a property of the dynamics itself).
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    if N > max_n:
        y = y[:max_n]

    # Embed the signal
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        logger.warning('Embedding failed')
        return np.nan
    try:
        Y = time_delay_embed(y, m, int(tau))
    except ValueError:
        logger.warning('Embedding failed')
        return np.nan
    n_emb = Y.shape[0]

    if 0 < theiler_win < 1:  # specify a proportion
        theiler_win = _round(theiler_win * n_emb)
    theiler_win = int(theiler_win)

    min_exceed = 15  # minimum exceedances required for a pole's estimate to be trusted
    min_n_emb = np.ceil(min_exceed/(1 - q)) + 2*theiler_win + m_order + 1
    if n_emb < max(min_n_emb, 100):
        return np.nan

    # Sample poles (reference points) from the embedded orbit
    rng = np.random.RandomState() if random_seed is None else _ml_rng(random_seed)
    n_poles = min(n_poles, n_emb)
    pole_idx = _ml_randperm_k(n_emb, n_poles, rng) - 1

    local_dim = np.full(n_poles, np.nan)
    theta = np.full(n_poles, np.nan)

    min_valid = min_exceed/(1 - q)
    for block in range(0, n_poles, _EVT_POLE_BLOCK):
        idx = pole_idx[block:block + _EVT_POLE_BLOCK]
        dists = cdist(Y[idx, :], Y)
        for b, j in enumerate(idx):
            # Exclude the Theiler window around this pole, keeping the rest of
            # the orbit in its original chronological order:
            lo, hi = max(0, j - theiler_win), min(n_emb - 1, j + theiler_win)
            dist_j = np.concatenate((dists[b, :lo], dists[b, hi+1:]))

            is_valid = dist_j > 0  # excludes exact duplicate embedded points
            if np.count_nonzero(is_valid) < min_valid:
                continue
            g = -np.log(dist_j[is_valid])  # chronological order preserved

            u = _evt_quantile(g, q)
            E = g > u  # exceedance indicator, in chronological order
            n_u = np.count_nonzero(E)
            if n_u < min_exceed:
                continue

            p = block + b
            # Local dimension: reciprocal of the mean exceedance (exponential/GPD-
            # with-shape-0 scale MLE), per the Freitas-Freitas-Todd theorem:
            local_dim[p] = 1/np.mean(g[E] - u)

            # Persistence: O'Brien order-m_order estimator (Caby et al. 2019, Eq 19),
            # vectorized via a cumulative sum of the exceedance indicator.
            n_tot = len(E)
            if n_tot > m_order + 1:
                cum_e = np.concatenate(([0], np.cumsum(E)))
                n_valid = n_tot - m_order
                future_sum = cum_e[m_order+1:m_order+1+n_valid] - cum_e[1:1+n_valid]
                is_isolated = E[:n_valid] & (future_sum == 0)
                numer = np.count_nonzero(is_isolated)/n_valid
                denom = n_u/n_tot
                theta[p] = min(numer/denom, 1)  # clip finite-sample overshoot above 1

    # Outputs
    valid_poles = ~np.isnan(local_dim) & ~np.isnan(theta)
    out = {}
    out['propValidPoles'] = np.mean(~np.isnan(local_dim))

    out['meanLocalDim'] = _omitnan_mean(local_dim)
    out['stdLocalDim'] = _omitnan_std(local_dim)
    out['meanTheta'] = _omitnan_mean(theta)
    out['stdTheta'] = _omitnan_std(theta)

    if (np.count_nonzero(valid_poles) >= 10 and _omitnan_std(local_dim[valid_poles]) > 0
            and _omitnan_std(theta[valid_poles]) > 0):
        out['corrDimTheta'] = float(np.corrcoef(local_dim[valid_poles],
                                                theta[valid_poles])[0, 1])
    else:
        out['corrDimTheta'] = np.nan

    return out
