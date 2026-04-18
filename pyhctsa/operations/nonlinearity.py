from typing import Union

import numpy as np
from numpy.typing import ArrayLike


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
        Sampling frequency in Hz.
    win_len_rel : float or int
        Window length (a long enough segment is important to estimate the nonlinearity).
    shift_len_rel : float or int
        This amounts to window length - overlap length btw windows.
    lag_rel : float or int
        TMaximum lag for nsAMDF, we chose it as 1.
    degree : The chosen degree p should ideally be large enough to capture the
           highest order of nonlinearity within the data.
           We chose p=7 for in our case of Parkinsonian data in the paper.
    
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
