import numpy as np
import pywt
from pywt import cwt
from numpy.typing import ArrayLike
from typing import Union
from ..Utilities.utils import signChange

def CWT(y : ArrayLike, wname : str = 'db3', maxScale : int = 32) -> dict:
    """
    Continuous wavelet transform of a time series.

    Computes the continuous wavelet transform (CWT) of a time series using the specified wavelet and maximum scale,
    and returns a set of statistics on the coefficients, entropy, and results of coefficients summed across scales.

    Parameters
    ----------
    y : array-like
        The input time series.
    wname : str, optional
        The wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), etc. Default is 'db3'.
    maxScale : int, optional
        The maximum scale of wavelet analysis. Default is 32.

    Returns
    -------
    dict
        Dictionary of statistics on the CWT coefficients.
    """
    y = np.asarray(y)
    N = len(y)
    scales = np.arange(1, maxScale+1)
    coeffs, _ = cwt(data=y,scales=scales,wavelet=wname)
    S = np.abs(coeffs * coeffs)
    SC = 100*S/np.sum(S)

    # Get statistics from CWT
    numEntries = SC.shape[0] * SC.shape[1]
    # 1) Coefficients, coeffs
    allCoeffs = coeffs if pywt.Wavelet(wname).symmetry == 'asymmetric' else -coeffs
    out = {}
    out['meanC'] = np.mean(allCoeffs)

    out['meanabsC'] = np.mean(abs(allCoeffs))
    out['medianabsC'] = np.median(abs(allCoeffs))
    out['maxabsC'] = np.max(abs(allCoeffs))
    out['maxonmeanC'] = out['maxabsC']/out['meanabsC']

    out['maxonmeanSC'] = np.max(SC)/np.mean(SC)

    #% Proportion of coeffs matrix over ___ maximum (thresholded)
    poverfn = lambda x : np.sum(SC[SC > x * np.max(SC)])/numEntries
    out['pover99'] = poverfn(0.99)
    out['pover98'] = poverfn(0.88)
    out['pover95'] = poverfn(0.95)
    out['pover90'] = poverfn(0.90)
    out['pover80'] = poverfn(0.80)

    # Distribution of scaled power
    #shape, loc, scale = gamma.fit(SC, floc=0, method="MM")
    # out['gam1'] = shape
    # out['gam2'] = scale
    # 2D entropy
    SC_a = SC/np.sum(SC)
    out['SC_h'] = -np.sum(SC_a * np.log(SC_a))

    SSC = sum(SC)
    out['max_ssc'] = np.max(SSC)
    out['min_ssc'] = np.min(SSC)
    out['maxonmed_ssc'] = np.max(SSC) / np.median(SSC)
    out['pcross_maxssc50'] = np.sum(signChange(SSC - 0.5 * np.max(SSC))) / (N - 1)
    out['std_ssc'] = np.std(SSC)

    #Stationarity
    midpoint = N // 2  # Integer division is equivalent to floor
    SC_1 = SC[:, :midpoint]
    SC_2 = SC[:, midpoint:]

    mean2_1 = SC_1.mean()
    mean2_2 = SC_2.mean()

    std2_1 = SC_1.std(ddof=1)
    std2_2 = SC_2.std(ddof=1)

    out['stat_2_m_s'] = np.mean([std2_1, std2_2]) / SC.mean()
    out['stat_2_s_m'] = np.std([mean2_1, mean2_2], ddof=1) / SC.std(ddof=1)
    out['stat_2_s_s'] = np.std([std2_1, std2_2], ddof=1) / SC.std(ddof=1)
    SCs = np.array_split(SC, 5, axis=1)
    for i in range(1, 6):
        out[f'mean5_{i}'] = np.mean(SCs[i-1])
        out[f'std5_{i}'] = np.std(SCs[i-1], ddof=1)
    
    out['stat_5_m_s'] = np.mean([out['std5_1'], out['std5_2'], out['std5_3'], out['std5_4'], out['std5_5']])/np.mean(SC)
    out['stat_5_s_m'] = np.std([out['mean5_1'], out['mean5_2'], out['mean5_3'], out['mean5_4'], out['mean5_5']], ddof=1)/np.std(SC, ddof=1)
    out['stat_5_s_s'] = np.std([out['std5_1'], out['std5_2'], out['std5_3'], out['std5_4'], out['std5_5']], ddof=1)/np.std(SC, ddof=1)


    return out

def _slosr(xx) -> int:
    # helper function for DetailCoeffs
    theMaxLevel = len(xx)
    slosr = np.zeros(theMaxLevel-2)
    for i in range(2, theMaxLevel):
        slosr[i-2] = np.sum(xx[:i-1])/np.sum(xx[i:])
    absm1 = np.abs(slosr - 1)
    idx = np.argwhere(absm1 == np.min(absm1).flatten())[0][0] + 1
    return idx

def DetailCoeffs(y : ArrayLike, wname : str = 'db3', maxlevel : Union[int, str] = 20) -> dict:
    """
    Detail coefficients of a wavelet decomposition.

    Compares the detail coefficients obtained at each level of the wavelet decomposition from 1 to the maximum possible level for the wavelet,
    given the length of the input time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    wname : str, optional
        The name of the mother wavelet to analyze the data with (e.g., 'db3', 'sym2').
        See the Wavelet Toolbox or PyWavelets documentation for details. Default is 'db3'.
    maxlevel : int or 'max', optional
        The maximum wavelet decomposition level. If 'max', uses the maximum allowed level for the data length and wavelet.
        Default is 20.

    Returns
    -------
    dict
        Statistics on the detail coefficients at each level.
    """
    y = np.asarray(y)
    N = len(y)
    if maxlevel == 'max':
        maxlevel = pywt.dwt_max_level(N, wname)
    if pywt.dwt_max_level(N, wname) < maxlevel:
        print(f"Chosen wavelet level is too large for the {wname} wavelet for this signal of length N = {N}")
        maxlevel = pywt.dwt_max_level(N, wname)
        print(f"Using a wavelet level of {maxlevel} instead.")
    # Perform a single-level wavelet decomposition
    means = np.zeros(maxlevel) # mean detail coefficient magnitude at each level
    medians = np.zeros(maxlevel) # median detail coefficient magnitude at each level
    maxs = np.zeros(maxlevel) # max detail coefficient magnitude at each level
    
    for k in range(1, maxlevel+1):
        level = k
        c, l = wavedec(data=y, wavelet=wname, level=level)
        det = wrcoef(coefs=c, lengths=l, wavelet=wname, level=level)
        absdet = np.abs(det)
        means[k-1] = np.mean(absdet)
        medians[k-1] = np.median(absdet)
        maxs[k-1] = np.max(absdet)
    
    #Return statistics on detail coefficients
    means_s = np.sort(means)[::-1] # descending order
    medians_s = np.sort(medians)[::-1]
    maxs_s = np.sort(maxs)[::-1]

    # % What is the maximum across these levels
    out = {}
    out['max_mean'] = means_s[0]
    out['max_median'] = medians_s[0]
    out['max_max'] = maxs_s[0]

    #% stds
    out['std_mean'] = np.std(means, ddof=1)
    out['std_median'] = np.std(medians, ddof=1)
    out['std_max'] = np.std(maxs, ddof=1)

    #% At what level is the maximum
    out['wheremax_mean'] = np.argwhere(means == means_s[0]).flatten()[0]
    out['wheremax_median'] = np.argwhere(medians == medians_s[0]).flatten()[0]
    out['wheremax_max'] = np.argwhere(maxs == maxs_s[0]).flatten()[0]

    #% Size of maximum (relative to next maximum)
    out['max1on2_mean'] = means_s[0]/means_s[1]
    out['max1on2_median'] = medians_s[0]/medians_s[1]
    out['max1on2_max'] = maxs_s[0]/maxs_s[1]

    # % Where sum of values to left equals sum of values to right
    # % Measure of centrality
    out['wslesr_mean'] = _slosr(means)
    out['wslesr_median'] = _slosr(medians)
    out['wslesr_max'] = _slosr(maxs)
    
    #% What's the correlation between maximum and median
    r = np.corrcoef(maxs, medians)
    out['corrcoef_max_medians'] = r[0, 1]

    return out

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

