from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import pywt
from pywt import cwt

from ..utils import sign_change

def wfbm(x : ArrayLike) -> dict:
    """
    Parameters of fractional Gaussian noise/Brownian motion in a time series.

    Parameters
    ----------
    x : array-like
        The input time series.

    Returns
    -------
    dict
        Dictionary containing the three estimates of the fractal index H of the signal x: 
        - (i) using a second order discrete derivative, 
        - (ii) using a second order discrete derivative with wavelets, 
        - (iii) using wavelet variance versus wavelet level.
    """
    x = np.asarray(x)
    y = np.cumsum(np.diff(x))
    b1 = np.array([1.0, -2.0, 1.0])
    b2 = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
    y1 = np.convolve(y, b1, mode='valid')
    y2 = np.convolve(y, b2, mode='valid')
    s1 = np.mean(y1**2)
    s2 = np.mean(y2**2)
    H1 = 0.5 * np.log2(s2 / s1)

    w = pywt.Wavelet('sym5')
    c1 = np.array(w.dec_hi, dtype=float)
    c2 = np.zeros(2 * len(c1))
    c2[::2] = c1
    cy1 = np.convolve(y, c1,  mode='valid')
    cy2 = np.convolve(y, c2,  mode='valid')
    cs1 = np.mean(cy1**2)
    cs2 = np.mean(cy2**2)
    H2 = 0.5*np.log2(cs2 / cs1)

    level_decomp = min(pywt.dwt_max_level(len(x), 'haar'), 6)
    C, L = wavedec(x, wavelet='haar', level=level_decomp)
    all_levels = np.arange(1, level_decomp+1)
    stdc = np.zeros(len(all_levels))
    for i in range(len(all_levels)):
        d = detcoef(coefs=C, lengths=L, levels=all_levels[i])
        stdc_val = np.median(np.abs(d)) / 0.67448975
        stdc[i] = stdc_val
    po = np.polyfit(all_levels, np.log2(stdc**2), 1)
    H3 = (po[0] - 1)/2

    return {"p1": H1, "p2": H2, "p3": H3}

def scal_2_freq(y : ArrayLike, w_name : str = 'db3', a_max : int = 5, delta : int = 1) -> dict:
    """
    Frequency components in a periodic time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with: e.g., 'db3', 'sym2'.
    a_max : int, optional
        The maximum scale / level (can be 'max' to set according to wmaxlev)
    delta: int, optional
        TThe sampling period.
            
    Returns
    -------
    dict
        Dictionary containing the level with the highest energy coefficients, the dominant period, and the dominant pseudo-frequency.
    """
    y = np.asarray(y)
    N = len(y)
    max_level = pywt.dwt_max_level(N, w_name)
    if a_max == 'max':
        a_max = max_level
    if max_level < a_max:
        print(f'Chosen level {a_max} is too large for this wavelet on this signal...')
        a_max = max_level # set to max allowed level
        print(f'changed to maximum level computed with wmaxlev: {a_max}')

    # % Define scales.
    scales = np.arange(1, a_max+1)
    a = 2**scales
    # Compute associated pseudo-frequencies.
    f = pywt.scale2frequency(w_name, a, delta+1)
    # Compute associated pseudo-periods.
    per = 1/f
    #Decompose the time series at level specified as maximum
    C, L = wavedec(y, wavelet=w_name, level=a_max)
    #Estimate standard deviation of detail coefficients.
    stdc = []
    for k in range(1, a_max+1):
        d = detcoef(coefs=C, lengths=L, levels=k)
        stdc_val = np.median(np.abs(d)) / 0.67448975
        stdc.append(stdc_val)
    #% Compute identified period.
    j_max = np.argmax(stdc)
    out = {"lmax": j_max + 1, # level with highest energy coefficients
           "period": per[j_max], #% output dominant period
           "pf": f[j_max]} # output dominant pseudo-frequency

    return out

def dwt_coeff(y : ArrayLike, w_name : str = 'db3', level : int = 3) -> dict:
    """
    Discrete wavelet transform coefficients.

    Decomposes the time series using a given wavelet and outputs statistics on the
    coefficients obtained up to a maximum level, level.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The mother wavelet, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), etc. Default is 'db3'.
    level : int, optional
        The level of wavelet decomposition (can be set to 'max' for the maximum level determined by wmaxlev)
            
    Returns
    -------
    dict
        Dictionary of statistics on the DWT coefficients.
    """
    y = np.asarray(y)
    N = len(y)
    if level == 'max':
        level = pywt.dwt_max_level(N, w_name)
    max_level_allowed = pywt.dwt_max_level(N, w_name)
    if max_level_allowed < level:
        print("Chosen level is too large for this wavelet on this signal....\n")
    #%% Perform Wavelet Decomposition
    C, L = None, None
    if max_level_allowed < level: # if level exceeds max level, just use max level instead
        C, L = wavedec(y, wavelet=w_name, level=max_level_allowed)
    else:
        C, L = wavedec(y, wavelet=w_name, level=level)
    #%% Get statistics on coefficients
    out = {}
    for k in range(1, level+1):
        if k <= max_level_allowed:
            d = detcoef(coefs=C, lengths=L, levels=k) # detail coeffs at level k
            # max coeff at this level
            out[f'maxd_l{k}'] = np.max(d)
            #% minimum coefficient at this level:
            out[f'mind_l{k}'] = np.min(d)
            # std coefficients at this level:
            out[f'stdd_l{k}'] = np.std(d, ddof=1)
            #% 1-D noise coefficient estimate (estimate of the noise std):
            out[f'noisestd_l{k}'] = np.median(np.abs(d)) / 0.67448975
        else:
            # exceeds max level, return nans
            out[f'maxd_l{k}'] = np.nan 
            out[f'mind_l{k}'] = np.nan 
            out[f'stdd_l{k}'] = np.nan
            out[f'noisestd_l{k}'] = np.nan
    
    return out

def cwt(y : ArrayLike, w_name : str = 'db3', max_scale : int = 32) -> dict:
    """
    Continuous wavelet transform of a time series.

    Computes the continuous wavelet transform (CWT) of a time series using the specified wavelet and maximum scale,
    and returns a set of statistics on the coefficients, entropy, and results of coefficients summed across scales.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), etc. Default is 'db3'.
    max_scale : int, optional
        The maximum scale of wavelet analysis. Default is 32.

    Returns
    -------
    dict
        Dictionary of statistics on the CWT coefficients.
    """
    y = np.asarray(y)
    N = len(y)
    scales = np.arange(1, max_scale+1)
    coeffs, _ = cwt(data=y, scales=scales, wavelet=w_name)
    S = np.abs(coeffs * coeffs)
    SC = 100*S/np.sum(S)

    # Get statistics from CWT
    num_entries = SC.shape[0] * SC.shape[1]
    # 1) Coefficients, coeffs
    all_coeffs = coeffs if pywt.Wavelet(w_name).symmetry == 'asymmetric' else -coeffs
    out = {}
    out['meanC'] = np.mean(all_coeffs)

    out['meanabsC'] = np.mean(abs(all_coeffs))
    out['medianabsC'] = np.median(abs(all_coeffs))
    out['maxabsC'] = np.max(abs(all_coeffs))
    out['maxonmeanC'] = out['maxabsC']/out['meanabsC']

    out['maxonmeanSC'] = np.max(SC)/np.mean(SC)

    #% Proportion of coeffs matrix over ___ maximum (thresholded)
    poverfn = lambda x : np.sum(SC[SC > x * np.max(SC)])/num_entries
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
    out['pcross_maxssc50'] = np.sum(sign_change(SSC - 0.5 * np.max(SSC))) / (N - 1)
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

def _slosr(xx : ArrayLike) -> int:
    # helper function for DetailCoeffs
    the_max_level = len(xx)
    slosr = np.zeros(the_max_level-2)
    for i in range(2, the_max_level):
        slosr[i-2] = np.sum(xx[:i-1])/np.sum(xx[i:])
    absm1 = np.abs(slosr - 1)
    idx = np.argwhere(absm1 == np.min(absm1).flatten())[0][0] + 1

    return idx

def detail_coeffs(y : ArrayLike, w_name : str = 'db3', max_level : Union[int, str] = 20) -> dict:
    """
    Detail coefficients of a wavelet decomposition.

    Compares the detail coefficients obtained at each level of the wavelet decomposition from 1 to the maximum possible level for the wavelet,
    given the length of the input time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with (e.g., 'db3', 'sym2').
        See the Wavelet Toolbox or PyWavelets documentation for details. Default is 'db3'.
    max_level : int or 'max', optional
        The maximum wavelet decomposition level. If 'max', uses the maximum allowed level for the data length and wavelet.
        Default is 20.

    Returns
    -------
    dict
        Statistics on the detail coefficients at each level.
    """
    y = np.asarray(y)
    N = len(y)
    if max_level == 'max':
        max_level = pywt.dwt_max_level(N, w_name)
    if pywt.dwt_max_level(N, w_name) < max_level:
        print(f"Chosen wavelet level is too large for the {w_name} wavelet for this signal of length N = {N}")
        max_level = pywt.dwt_max_level(N, w_name)
        print(f"Using a wavelet level of {max_level} instead.")
    # Perform a single-level wavelet decomposition
    means = np.zeros(max_level) # mean detail coefficient magnitude at each level
    medians = np.zeros(max_level) # median detail coefficient magnitude at each level
    maxs = np.zeros(max_level) # max detail coefficient magnitude at each level
    
    for k in range(1, max_level+1):
        level = k
        c, l = wavedec(data=y, wavelet=w_name, level=level)
        det = wrcoef(coefs=c, lengths=l, wavelet=w_name, level=level)
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

def wl_coeffs(y : ArrayLike, w_name : str = 'db3', level : Union[int, str] = 3) -> dict:
    """
    Wavelet decomposition of the time series.

    Performs a wavelet decomposition of the time series using a given wavelet at a
    specified level and returns a set of statistics on the coefficients obtained.

    Parameters
    ----------
    y : list or array-like
        The input time series.
    w_name : str, optional
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
        level = pywt.dwt_max_level(N, w_name)
        if level == 0:
            raise ValueError("Cannot compute wavelet coefficients (short time series)")
    
    if pywt.dwt_max_level(N, w_name) < level:
        raise ValueError(f"Chosen level, {level}, is too large for this wavelet on this signal.")
    
    C, L = wavedec(y, wavelet=w_name, level=level)
    det = wrcoef(C, L, w_name, level)
    det_s = np.sort(np.abs(det))[::-1]

    #%% Return statistics
    out = {}
    out['mean_coeff'] = np.mean(det_s)
    out['max_coeff'] = np.max(det_s)
    out['med_coeff'] = np.median(det_s)

    #% Decay rate stats ('where below _ maximum' = 'wb_m')
    out['wb99m'] = find_my_threshold(0.99, det_s, N)
    out['wb90m'] = find_my_threshold(0.90, det_s, N)
    out['wb75m'] = find_my_threshold(0.75, det_s, N)
    out['wb50m'] = find_my_threshold(0.50, det_s, N)
    out['wb25m'] = find_my_threshold(0.25, det_s, N)
    out['wb10m'] = find_my_threshold(0.10, det_s, N)
    out['wb1m'] = find_my_threshold(0.01, det_s, N)

    return out

def wavedec(data : ArrayLike, wavelet : str, mode : str ='symmetric', level : int = 1, axis=-1):
    """
    Multiple level 1-D discrete fast wavelet decomposition.
    Taken from https://github.com/izlotnik/wavelet-wrcoef/blob/master/wrcoef.py
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

def find_my_threshold(x : ArrayLike, det_s : ArrayLike, N : int):
    indices = np.argwhere(det_s < x * np.max(det_s))
    if indices.size == 0:
        return np.nan
    else:
        pr = indices[0]/N
        return pr[0]
