from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from sklearn.decomposition import PCA
from scipy.linalg import lapack, qr, solve_triangular
from sklearn.neighbors import NearestNeighbors

from ..operations.model_fit import residual_analysis
from ..operations.correlation import first_crossing, first_min, autocorr
from ..toolboxes.Tisean_3_0_1 import tisean as _tisean

def _first_fn(p, threshold, over_or_under='under'):
    if over_or_under == 'under':
        indices = np.where(p < threshold)[0]
    elif over_or_under == 'over':
        indices = np.where(p > threshold)[0]
    else:
        raise ValueError(f'Unknown setting: {over_or_under}')
    
    return indices[0] if len(indices) > 0 else len(p) + 1

def _normed_single_curve_length(x: ArrayLike, lag: int, fs: Union[float, int],
                                nrmdegree: int) -> tuple:
    # helper function for _normed_single_curve_length_windowed and nsamdf
    # compute crossed curve length for a single time series x
    x = np.asarray(x, dtype=float)
    rx1 = np.zeros(lag+1)
    for delay in range(0, lag+1):
        if delay == 0:
            rx1[delay] = np.linalg.norm(x - x, ord=nrmdegree)
        else:
            rx1[delay] = np.linalg.norm(x[:-delay] - x[delay:], ord=nrmdegree)
    # mirror to form full symmetric sequence [rx1_reversed, rx1[1:]]
    rx = np.concatenate([rx1[::-1], rx1[1:]])

    pxy = np.abs(np.fft.fft(rx))
    frq = np.linspace(0, 0.5 * fs, lag + 1)
    return rx1, pxy, frq

def _normed_single_curve_length_windowed(x: ArrayLike, win_len: int,
                                         shift_len: int, lag: int,
                                         fs: Union[float, int], nrmdegree: int) -> tuple:
    # helper function for nsamdf
    # shiftlen is winlen - overlaplen
    m = int(np.floor((len(x) - win_len)/shift_len) + 1)
    
    p_sum = np.zeros(2*lag+1)
    r_sum = np.zeros(lag+1)

    for ii in range(1, m+1):
        x_seg = x[(ii-1)*shift_len:(ii-1)*shift_len+win_len]
        rx, px, _ = _normed_single_curve_length(x_seg, lag=lag, fs=fs, nrmdegree=nrmdegree)
        p_sum += px
        r_sum += rx
    
    p_mean = p_sum / m
    r_mean = r_sum / m

    p_mean2 = np.abs(np.fft.fft(np.concatenate([r_mean[::-1], r_mean[1:]])))
    frq = np.linspace(0, 0.5 * fs, lag+1)

    return r_mean, p_mean, frq, p_mean2

def _ms_embed(z, v, w):
    # helper function for nlpe
    z = np.asarray(z, dtype=float).squeeze()
    if z.ndim != 1:
        raise ValueError("MS_embed requires a 1-D time series as first argument.")

    n = z.size
    if v is None:
        lags = np.array([0, 1, 2])
        auto_neg = False
    elif w is not None:
        # MS_embed(z, dim, lag)  →  lags = 0 : w : w*(v-1)
        v = int(v)
        lags = np.arange(0, w * v, w)          # length v
        auto_neg = False
    else:
        lags = np.asarray(v, dtype=int).ravel()
        auto_neg = False
    has_neg_input = np.any(lags < 0)

    lags = np.sort(lags)
    dim  = len(lags)
    if n <= lags[-1]:
        logger.warning("Vector is too small to be embedded with the given lags.")
        return np.full((dim, 1), np.nan), None

    w_win = lags[-1] - lags[0]          # window width  (renamed to avoid shadowing arg)
    m     = n - w_win                   # number of embeddable points
    t     = np.arange(m) + lags[-1]    # embed times (0-indexed: t[i] = i + lags[-1])

    x = np.zeros((dim, m), dtype=float)
    for i, lag in enumerate(lags):
        x[i, :] = z[t - lag]

    # Split into past (x) and future (y) components
    neg_mask = lags < 0
    pos_mask = lags >= 0

    if np.any(neg_mask):
        y = x[neg_mask, :]
        x = x[pos_mask, :]
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

    dd = np.zeros((n, n))
    for i in range(de_dim):
        diff = x[i, :][np.newaxis, :] - x[i, :][:, np.newaxis]
        dd += diff ** 2
    np.fill_diagonal(dd, np.inf)

    near = np.argmin(dd, axis=1)  # (n,) — nearest neighbour index per point
    e = y[near] - y               # now y is (m,) so y[near] works correctly

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
    s2, _, _, _ = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length, lag=lag, fs=fs, nrmdegree=2)
    #out['s2'] = s2 / np.max(s2) # normalized

    # nsAMDF for p = degree:
    sd, _, _, _ = _normed_single_curve_length_windowed(x, win_len=window_length, shift_len=shift_length, lag=lag, fs=fs, nrmdegree=degree)
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
        if tau == 'ac':
            tau = first_crossing(y, 'ac', 0, 'discrete')
        elif tau == 'mi':
            tau = first_min(y, 'mi')
        else:
            raise ValueError("tau can be either 'mi' or 'ac'")
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
    n = len(y)
    if isinstance(tau, str):
        if tau == 'ac':
            tau = first_crossing(y, 'ac', 0, 'discrete')
            if np.isnan(tau):
                logger.warning('Could not get time delay by ACF (time series too short?)')
                return np.nan
        elif tau == 'mi':
            tau = first_min(y, 'mi')
            if np.isnan(tau):
                logger.warning('Could not get time delay by mutual information (time series too short?)')
                return np.nan
        else:
            raise ValueError(f'Invalid time-delay method: {tau}. Choose either mi or ac.')
    n_embed = n - (m-1)*tau
    if n_embed <= 0:
        logger.warning(f'Time series (N = {n}) too short to embed with these embedding parameters.')
        return np.nan

    y_embed = np.zeros((n_embed, m))
    for i in range(m):
        y_embed[:, i] = y[i*tau : n_embed + i*tau]
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

def _time_delay_embed(y: ArrayLike, tau: Union[str, int], m: int) -> np.ndarray:
    """
    Time-delay embedding of a univariate time series into an m-dimensional space.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if isinstance(tau, str):
        if tau == 'ac':
            tau = first_crossing(y, 'ac', 0, 'discrete')
            if np.isnan(tau):
                raise ValueError('Could not get time delay by ACF (time series too short?)')
            tau = int(tau)
        else:
            raise ValueError(f"Invalid time-delay method: '{tau}'. Only 'ac' (or an integer) is supported.")
    tau = int(tau)
    m = int(m)

    n_embed = n - (m-1)*tau
    if n_embed <= 0:
        raise ValueError(f'Time series (N = {n}) too short to embed with these embedding parameters.')

    y_embed = np.zeros((n_embed, m))
    for i in range(m):
        y_embed[:, i] = y[i*tau : n_embed + i*tau]

    return y_embed

def local_density(y: ArrayLike, nnr: int = 3, past: int = 40,
                  tau: Union[str, int] = 'ac', m: int = 2) -> dict:
    """
    Local density estimates in the time-delay embedding space.
    
    Computes a standard k-nearest-neighbor local density estimate at each
    point of the time-delay embedding: density(i) is proportional to
    1/r_NNR(i)^m, where r_NNR(i) is the distance from point i to its NNR-th
    nearest neighbor (excluding temporally-close points within a Theiler
    window of "past" samples) and m is the embedding dimension. This
    operation previously used TSTOOL's 'localdensity', which the original
    author noted was "very poorly documented in the TSTOOL package" -- its
    exact algorithm was never confirmed, only assumed to be some form of
    local density estimate in the embedding space, which is what's computed
    here natively (no toolbox dependency at all). The missing normalizing
    constant (relating 1/r^m to a true probability density) is the same for
    every point in a given call, so it cancels out of all of the relative
    statistics below (min/max/std/mean/median/autocorrelation).

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
    try:
        y_embed = _time_delay_embed(y, tau, m)
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
    valid_dists = np.sort(np.where(valid, dist, np.inf), axis=1)
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
    """First index of the minimum in MATLAB's column-major order; NaNs ignored."""
    flat = m.ravel(order='F')
    if flat.size == 0 or np.all(np.isnan(flat)):
        return None, None, np.nan
    best = np.nanmin(flat)
    k = int(np.flatnonzero(flat == best)[0])
    i, j = np.unravel_index(k, m.shape, order='F')
    return int(i), int(j), float(best)


def _scaling_range_endpoints(l: int) -> tuple:
    """Candidate 1-based start/end points: start in the first half, end in the second."""
    stptr = np.arange(1, int(np.floor(l / 2)))
    endptr = np.arange(int(np.ceil(l / 2)) + 1, l + 1)
    return stptr, endptr


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
    spread = np.max(ds) - np.min(ds)
    with np.errstate(invalid='ignore', divide='ignore'):
        dsn = (ds - np.min(ds)) / spread  # rescale to [0,1] so weights are consistent

    out = {'ri1': None, 'goodness': np.nan, 'stabled': np.nan, 'linrmserr': np.nan}
    if l < 2:
        return out

    mybad = np.array([np.std(dsn[i - 1:], ddof=1) - gamma * (l - i + 1)
                      for i in range(1, l)])
    if not np.all(np.isnan(mybad)):
        best = np.nanmin(mybad)
        a = int(np.flatnonzero(mybad == best)[0])  # 0-based
        out['ri1'] = a + 1
        out['goodness'] = float(best)
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

    mybad = np.empty((stptr.size, endptr.size))
    for i, s in enumerate(stptr):
        for j, e in enumerate(endptr):
            sub = x[:, s - 1:e]
            mm = sub.mean()  # middle value for this range: exponent estimate
            # mean squared deviation from mm, minus a bonus for a longer range
            mybad[i, j] = ((sub - mm) ** 2).mean() - gamma * (e - s + 1)

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
        with np.errstate(invalid='ignore', divide='ignore'):
            vnorm = (v - v.min()) / (v.max() - v.min())  # normalize regardless of range
        mybad = np.empty((stptr.size, endptr.size))
        for i, s in enumerate(stptr):
            for j, e in enumerate(endptr):
                mybad[i, j] = np.std(vnorm[s - 1:e], ddof=1) - gamma * (e - s + 1)
        a, b, best = _argmin_first_colmajor(mybad)
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
            break  # MATLAB bails out here, leaving the remaining rows zero
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
        return None  # MATLAB hits a dimension mismatch here and returns NaN

    results = np.full((ndim, 4), np.nan)
    for c in range(ndim):
        v = np.diff(Y[c, :]) * dx  # vector of local gradients
        with np.errstate(invalid='ignore', divide='ignore'):
            vnorm = (v - v.min()) / (v.max() - v.min())
        mybad = np.empty((stptr.size, endptr.size))
        for i, s in enumerate(stptr):
            for j, e in enumerate(endptr):
                mybad[i, j] = np.std(vnorm[s - 1:e], ddof=1) - gamma * (e - s + 1)
        a, b, best = _argmin_first_colmajor(mybad)
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
        mybad = np.empty((stptr.size, endptr.size))
        for i, s in enumerate(stptr):
            for j, e in enumerate(endptr):
                mybad[i, j] = (abs(np.mean(vnorm[s - 1:e]))       # deviation from zero inside
                               - abs(np.mean(vnorm[0:s]))         # minus that of the outside regions
                               - abs(np.mean(vnorm[e - 1:])))
        a, b, best = _argmin_first_colmajor(mybad)
        if a is None:
            continue
        results[c] = [best, np.mean(Y[c, stptr[a] - 1:endptr[b]])]
    return results


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
    if isinstance(tau, str):
        if tau == 'ac':
            tau = first_crossing(y, 'ac', 0, 'discrete')
        elif tau == 'mi':
            tau = first_min(y, 'mi')
        else:
            raise ValueError(f'Invalid time-delay method: {tau}. Choose either mi or ac.')
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

    try:
        benfindd2 = _findscalingr_ind(d2dat_M)
    except Exception as exc:
        raise ValueError('Error finding scaling range') from exc

    # rows: increasing embedding m; columns: stpt, endpt, goodness, dim
    out['bend2_mindim'] = np.min(benfindd2[:, 3])
    out['bend2_maxdim'] = np.max(benfindd2[:, 3])
    out['bend2_meandim'] = np.mean(benfindd2[:, 3])
    out['bend2_meangoodness'] = np.mean(benfindd2[:, 2])

    mminfulcherd2 = _sub_findmmin(benfindd2[:, 3])
    # minimum scale at which a scaling range is observed:
    out['benmmind2_logminl'] = (np.nan if mminfulcherd2['ri1'] is None
                                else np.log(d2dat_v[mminfulcherd2['ri1'] - 1]))
    out['benmmind2_goodness'] = mminfulcherd2['goodness']
    out['benmmind2_stabledim'] = mminfulcherd2['stabled']
    out['benmmind2_linrmserr'] = mminfulcherd2['linrmserr']

    # Reshaped: only for large enough m (as determined by the criteria above),
    # then find a scaling region across m for a saturated range of m.
    scd2 = _findscalingr(d2dat_M[mminfulcherd2['ri1'] - 1:, :])

    out['d2_logminscr'] = (np.nan if scd2['ri1'] is None
                           else np.log(d2dat_v[scd2['ri1'] - 1]))
    out['d2_logmaxscr'] = (np.nan if scd2['ri2'] is None
                           else np.log(d2dat_v[scd2['ri2'] - 1]))
    out['d2_logscr'] = out['d2_logmaxscr'] - out['d2_logminscr']
    out['d2_goodness'] = scd2['goodness']
    out['d2_dimest'] = scd2['dimest']
    out['d2_dimstd'] = scd2['dimstd']

    # --------------------------------------------------------------------------
    # (3) Gaussian-smoothed estimates: as for D2, on c2g's third column
    # --------------------------------------------------------------------------
    d2gdat_v, d2gdat_M = _sub_celltomat(c2gdat, 3)

    try:
        benfindd2g = _findscalingr_ind(d2gdat_M)
    except Exception as exc:
        raise ValueError('Error finding scaling range') from exc

    out['bend2g_mindim'] = np.min(benfindd2g[:, 3])
    out['bend2g_maxdim'] = np.max(benfindd2g[:, 3])
    out['bend2g_meandim'] = np.mean(benfindd2g[:, 3])
    out['bend2g_meangoodness'] = np.mean(benfindd2g[:, 2])

    mminfulcherd2g = _sub_findmmin(benfindd2g[:, 3])
    out['benmmind2g_logminl'] = (np.nan if mminfulcherd2g['ri1'] is None
                                 else np.log(d2gdat_v[mminfulcherd2g['ri1'] - 1]))
    out['benmmind2g_goodness'] = mminfulcherd2g['goodness']
    out['benmmind2g_stabledim'] = mminfulcherd2g['stabled']
    out['benmmind2g_linrmserr'] = mminfulcherd2g['linrmserr']

    scd2g = _findscalingr(d2gdat_M[mminfulcherd2g['ri1'] - 1:, :])

    out['d2g_logminscr'] = (np.nan if scd2g['ri1'] is None
                            else np.log(d2gdat_v[scd2g['ri1'] - 1]))
    out['d2g_logmaxscr'] = (np.nan if scd2g['ri2'] is None
                            else np.log(d2gdat_v[scd2g['ri2'] - 1]))
    out['d2g_logscr'] = out['d2g_logmaxscr'] - out['d2g_logminscr']
    out['d2g_goodness'] = scd2g['goodness']
    out['d2g_dimest'] = scd2g['dimest']
    out['d2g_dimstd'] = scd2g['dimstd']

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


    
