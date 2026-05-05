# Part of this module is derived from PyWavelets (https://github.com/PyWavelets/pywt)
# Copyright (c) 2006-2024 PyWavelets contributors.
# Licensed under the MIT License.
from typing import Union
from math import ceil, floor

import numpy as np
from numpy.typing import ArrayLike
import pywt
import logging

from ..utils import sign_change
from pywt._extensions._pywt import (
    ContinuousWavelet,
    DiscreteContinuousWavelet,
    Wavelet,
    _check_dtype,
)
from pywt._functions import integrate_wavelet, scale2frequency
from pywt._utils import AxisError
from pywt._cwt import next_fast_len

def _custom_cwt(data, scales, wavelet, sampling_period=1., method='conv', axis=-1,
        *, precision=12):
    """
    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    scales : array_like
        The wavelet scales to use. One can use
        ``f = scale2frequency(wavelet, scale)/sampling_period`` to determine
        what physical frequency, ``f``. Here, ``f`` is in hertz when the
        ``sampling_period`` is given in seconds.
    wavelet : Wavelet object or name
        Wavelet to use
    sampling_period : float
        Sampling period for the frequencies output (optional).
        The values computed for ``coefs`` are independent of the choice of
        ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
        period).
    method : {'conv', 'fft'}, optional
        The method used to compute the CWT. Can be any of:
            - ``conv`` uses ``numpy.convolve``.
            - ``fft`` uses frequency domain convolution.
            - ``auto`` uses automatic selection based on an estimate of the
              computational complexity at each scale.

        The ``conv`` method complexity is ``O(len(scale) * len(data))``.
        The ``fft`` method is ``O(N * log2(N))`` with
        ``N = len(scale) + len(data) - 1``. It is well suited for large size
        signals but slightly slower than ``conv`` on small ones.
    axis: int, optional
        Axis over which to compute the CWT. If not given, the last axis is
        used.
    precision: int, optional
        Length of wavelet (``2 ** precision``) used to compute the CWT. Greater
        will increase resolution, especially for higher scales, but will
        compute a bit slower. Too low will distort coefficients and their
        norms, with a zipper-like effect. The default is 12, it's recommended
        to use >=12.

    Returns
    -------
    coefs : array_like
        Continuous wavelet transform of the input signal for the given scales
        and wavelet. The first axis of ``coefs`` corresponds to the scales.
        The remaining axes match the shape of ``data``.
    frequencies : array_like
        If the unit of sampling period are seconds and given, then frequencies
        are in hertz. Otherwise, a sampling period of 1 is assumed.

    Note
    ----
    This is a modified version of `pywt.cwt` from the PyWavelets 
    package, adapted to handle complex-type casting.
    """

    # accept array-like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.asarray(data, dtype=dt)
    dt_cplx = np.result_type(dt, np.complex64)
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)

    scales = np.atleast_1d(scales)
    if np.any(scales <= 0):
        raise ValueError("`scales` must only include positive values")

    if not np.isscalar(axis):
        raise AxisError("axis must be a scalar.")

    dt_out = dt_out = dt_cplx if (isinstance(wavelet, ContinuousWavelet) and wavelet.complex_cwt) else dt
    out = np.empty((np.size(scales),) + data.shape, dtype=dt_out)

    int_psi, x = integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi) if (isinstance(wavelet, ContinuousWavelet) and wavelet.complex_cwt) else int_psi

    # convert int_psi, x to the same precision as the data
    dt_psi = dt_cplx if int_psi.dtype.kind == 'c' else dt
    int_psi = np.asarray(int_psi, dtype=dt_psi)
    x = np.asarray(x, dtype=data.real.dtype)

    if method == 'fft':
        size_scale0 = -1
        fft_data = None
    elif method != "conv":
        raise ValueError("method must be 'conv' or 'fft'")

    if data.ndim > 1:
        # move axis to be transformed last (so it is contiguous)
        data = data.swapaxes(-1, axis)

        # reshape to (n_batch, data.shape[-1])
        data_shape_pre = data.shape
        data = data.reshape((-1, data.shape[-1]))

    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]

        if method == 'conv':
            if data.ndim == 1:
                conv = np.convolve(data, int_psi_scale)
            else:
                # batch convolution via loop
                conv_shape = list(data.shape)
                conv_shape[-1] += int_psi_scale.size - 1
                conv_shape = tuple(conv_shape)
                conv = np.empty(conv_shape, dtype=dt_out)
                for n in range(data.shape[0]):
                    conv[n, :] = np.convolve(data[n], int_psi_scale)
        else:
            # The padding is selected for:
            # - optimal FFT complexity
            # - to be larger than the two signals length to avoid circular
            #   convolution
            size_scale = next_fast_len(
                data.shape[-1] + int_psi_scale.size - 1
            )
            if size_scale != size_scale0:
                # Must recompute fft_data when the padding size changes.
                fft_data = np.fft.fft(data, size_scale, axis=-1)
            size_scale0 = size_scale
            fft_wav = np.fft.fft(int_psi_scale, size_scale, axis=-1)
            conv = np.fft.ifft(fft_wav * fft_data, axis=-1)
            conv = conv[..., :data.shape[-1] + int_psi_scale.size - 1]

        coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
        if out.dtype.kind != 'c':
            coef = coef.real
        # transform axis is always -1 due to the data reshape above
        d = (coef.shape[-1] - data.shape[-1]) / 2.
        if d > 0:
            coef = coef[..., floor(d):-ceil(d)]
        elif d < 0:
            raise ValueError(
                f"Selected scale of {scale} too small.")
        if data.ndim > 1:
            # restore original data shape and axis position
            coef = coef.reshape(data_shape_pre)
            coef = coef.swapaxes(axis, -1)
        out[i, ...] = coef

    frequencies = scale2frequency(wavelet, scales, precision)
    if np.isscalar(frequencies):
        frequencies = np.array([frequencies])
    frequencies /= sampling_period

    return out, frequencies

def wfbm(x: ArrayLike) -> dict:
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

def scal_2_freq(y: ArrayLike, w_name: str = 'db3', a_max: int = 5, delta: int = 1) -> dict:
    """
    Frequency components in a periodic time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with: e.g., 'db3', 'sym2'. Default is ``'db3'``.
    a_max : int, optional
        The maximum scale / level (can be 'max' to set according to wmaxlev). Default is 5.
    delta: int, optional
        The sampling period. Default is 1.
            
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
        logging.info(f'Chosen level {a_max} is too large for this wavelet on this signal...')
        a_max = max_level # set to max allowed level
        logging.info(f'changed to maximum level computed with wmaxlev: {a_max}')

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

def dwt_coeff(y: ArrayLike, w_name: str = 'db3', level: int = 3) -> dict:
    """
    Discrete wavelet transform coefficients.

    Decomposes the time series using a given wavelet and outputs statistics on the
    coefficients obtained up to a maximum level, level.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The mother wavelet, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), etc. Default is ``'db3'``.
    level : int, optional
        The level of wavelet decomposition (can be set to 'max' for the maximum level determined by wmaxlev). Default is 3.
            
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
        logging.warning("Chosen level is too large for this wavelet on this signal....\n")
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

def cwt(y: ArrayLike, w_name: str = 'db3', max_scale: int = 32) -> dict:
    """
    Continuous wavelet transform of a time series.

    Computes the continuous wavelet transform (CWT) of a time series using the specified wavelet and maximum scale,
    and returns a set of statistics on the coefficients, entropy, and results of coefficients summed across scales.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), etc. Default is ``'db3'``.
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
    coeffs, _ = _custom_cwt(data=y, scales=scales, wavelet=w_name)
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

    # Sum across scales
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
    # helper function for detail_coeffs
    the_max_level = len(xx)
    slosr = np.zeros(the_max_level-2)
    for i in range(2, the_max_level):
        slosr[i-2] = np.sum(xx[:i-1])/np.sum(xx[i:])
    absm1 = np.abs(slosr - 1)
    idx = np.argwhere(absm1 == np.min(absm1).flatten())[0][0] + 1

    return idx

def detail_coeffs(y: ArrayLike, w_name: str = 'db3', max_level: Union[int, str] = 20) -> dict:
    """
    Detail coefficients of a wavelet decomposition.

    Compares the detail coefficients obtained at each level of the wavelet decomposition from 1 to the maximum possible level for the wavelet,
    given the length of the input time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with (e.g., ``'db3'``, ``'sym2'``).
        See the Wavelet Toolbox or PyWavelets documentation for details. Default is ``'db3'``.
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
        logging.info(f"Chosen wavelet level is too large for the {w_name} wavelet for this signal of length N = {N}")
        max_level = pywt.dwt_max_level(N, w_name)
        logging.info(f"Using a wavelet level of {max_level} instead.")
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

def wl_coeffs(y: ArrayLike, w_name: str = 'db3', level: Union[int, str] = 3) -> dict:
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
        Default is ``'db3'``.
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

def wavedec(data: ArrayLike, wavelet: str, mode: str ='symmetric', level: int = 1, axis=-1):
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

def find_my_threshold(x: ArrayLike, det_s: ArrayLike, N: int):
    indices = np.argwhere(det_s < x * np.max(det_s))
    if indices.size == 0:
        return np.nan
    else:
        pr = indices[0]/N
        return pr[0]
