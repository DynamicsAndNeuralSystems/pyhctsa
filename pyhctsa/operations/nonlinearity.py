from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from sklearn.decomposition import PCA

from ..operations.model_fit import residual_analysis
from ..operations.correlation import first_crossing, first_min

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
        print("Vector is too small to be embedded with the given lags.")
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
        raise ValueError("Error embedding the time series.")

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
            raise ValueError('Time series cannot be embedded (too short?)')
    #% nlpe can cause memory pains for long time series
    #% Let's do this dirty cheat
    if n > max_n:
        # crop the time series to the first max_n samples
        y = y[:max_n]
        print(f"Michael Small's nlpe code is only being evaluated on the first {max_n} (/{n}) samples.")
        n = max_n
    
    if n < 20: # short time series cause problems
        print(f'Time series (N = {len(y)}) is too short.')

    # run the nonlinear prediction error code
    res = _ms_nlpe(y, de, tau)

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
                print('Could not get time delay by ACF (time series too short?)')
                return np.nan
        elif tau == 'mi':
            tau = first_min(y, 'mi')
            if np.isnan(tau):
                print('Could not get time delay by mutual information (time series too short?)')
                return np.nan
        else:
            raise ValueError(f'Invalid time-delay method: {tau}. Choose either mi or ac.')
    n_embed = n - (m-1)*tau
    if n_embed <= 0:
        print(f'Time series (N = {n}) too short to embed with these embedding parameters.')
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
