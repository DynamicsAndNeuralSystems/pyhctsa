import numpy as np
import pywt
from numpy.typing import ArrayLike
from typing import Union

def wavedec(data, wavelet, mode='symmetric', level=1, axis=-1):
    """
    Multiple level 1-D discrete fast wavelet decomposition

    # Taken from https://github.com/izlotnik/wavelet-wrcoef/blob/master/wrcoef.py
    # Ilya Zlotnik 2017
    """
    data = np.asarray(data)

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet(wavelet)

    # Initialization
    coefs, lengths = [], []

    # Decomposition
    lengths.append(len(data))
    for i in range(level):
        data, d = pywt.dwt(data, wavelet, mode, axis)

        # Store detail and its length
        coefs.append(d)
        lengths.append(len(d))

    # Add the last approximation
    coefs.append(data)
    lengths.append(len(data))

    # Reverse (since we've appended to the end of list)
    coefs.reverse()
    lengths.reverse()

    return np.concatenate(coefs).ravel(), lengths


def detcoef(coefs, lengths, levels=None):
    """
    1-D detail coefficients extraction
    """
    if not levels:
        levels = range(len(lengths) - 2)

    if not isinstance(levels, list):
        levels = [levels]

    first = np.cumsum(lengths) + 1
    first = first[-3::-1]
    last = first + lengths[-2:0:-1] - 1

    x = []
    for level in levels:
        d = coefs[first[level - 1] - 1:last[level - 1]]
        x.append(d)

    if len(x) == 1:
        x = x[0]

    return x


def wrcoef(coefs, lengths, wavelet, level):
    """
    Restruction from single branch from multiple level decomposition
    """
    def upsconv(x, f, s):
        # returns an extended copy of vector x obtained by inserting zeros
        # as even-indexed elements of data: y(2k-1) = data(k), y(2k) = 0.
        y_len = 2 * len(x) + 1
        y = np.zeros(y_len)
        y[1:y_len:2] = x

        # performs the 1-D convolution of the vectors y and f
        y = np.convolve(y, f, 'full')

        # extracts the vector y from the input vector
        sy = len(y)
        d = (sy - s) / 2.0
        y = y[int(np.floor(d)):(sy - int(np.ceil(d)))]

        return y

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet(wavelet)

    data = detcoef(coefs, lengths, level)

    idx = len(lengths) - level
    data = upsconv(data, wavelet.rec_hi, lengths[idx])
    for k in range(level-1):
        data = upsconv(data, wavelet.rec_lo, lengths[idx + k + 1])

    return data

def findMyThreshold(x, det_s, N):
    indices = np.argwhere(det_s < x * np.max(det_s))
    if indices.size == 0:
        return np.nan
    else:
        pr = indices[0]/N
        return pr[0]
    
def WLCoeffs(y : ArrayLike, wname : str = 'db3', level : Union[int, str] = 3) -> dict:
    """
    Wavelet decomposition of the time series.

    Performs a wavelet decomposition of the time series using a given wavelet at a
    specified level and returns a set of statistics on the coefficients obtained.

    Parameters
    ----------
    y : list or array-like
        The input time series.
    wname : str, optional
        The wavelet name (e.g., 'db3'). See PyWavelets documentation for all options.
        Default is 'db3'.
    level : int or 'max', optional
        The level of wavelet decomposition. If 'max', uses the maximum allowed level for the data length and wavelet.
        Default is 3.

    Returns
    -------
    dict
        Dictionary containing statistics of the wavelet coefficients, including:
            - 'mean_coeff': Mean of sorted absolute detail coefficients.
            - 'max_coeff': Maximum of sorted absolute detail coefficients.
            - 'med_coeff': Median of sorted absolute detail coefficients.
            - 'wb75m', 'wb50m', 'wb25m', 'wb10m', 'wb1m': Decay rate statistics (fraction of coefficients below a threshold of the maximum).
    """
    y = np.asarray(y)
    N = len(y)
    if level == 'max':
        level = pywt.dwt_max_level(N, wname)
        if level == 0:
            raise ValueError("Cannot compute wavelet coefficients (short time series)")
    
    if pywt.dwt_max_level(N, wname) < level:
        raise ValueError(f"Chosen level, {level}, is too large for this wavelet on this signal.")
    
    C, L = wavedec(y, wavelet=wname, level=level)
    det = wrcoef(C, L, wname, level)
    det_s = np.sort(np.abs(det))[::-1]

    #%% Return statistics
    out = {}
    out['mean_coeff'] = np.mean(det_s)
    out['max_coeff'] = np.max(det_s)
    out['med_coeff'] = np.median(det_s)

    #% Decay rate stats ('where below _ maximum' = 'wb_m')
    #out['wb99m'] = findMyThreshold(0.99, det_s, N)
    #out['wb90m'] = findMyThreshold(0.90, det_s, N)
    out['wb75m'] = findMyThreshold(0.75, det_s, N)
    out['wb50m'] = findMyThreshold(0.50, det_s, N)
    out['wb25m'] = findMyThreshold(0.25, det_s, N)
    out['wb10m'] = findMyThreshold(0.10, det_s, N)
    out['wb1m'] = findMyThreshold(0.01, det_s, N)

    return out
