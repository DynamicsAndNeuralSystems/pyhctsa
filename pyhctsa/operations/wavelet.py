from typing import Union

import warnings
import numpy as np
from numpy.typing import ArrayLike
import pywt
from scipy import stats
import logging
logger = logging.getLogger('pyhctsa')

from ..utils import _even_split_points, dict_output, sign_change
from ..toolboxes.pywavelets.cwt import _custom_cwt

# Scale factor of the median absolute deviation of Gaussian noise, Phi^{-1}(0.75) = 0.67449
# (Donoho & Johnstone 1994, Biometrika 81(3):425-455): sigma = median(|d|) / _MAD_TO_SIGMA.
_MAD_TO_SIGMA = float(stats.norm.ppf(0.75))


def _valid_filter_mean_square(z: np.ndarray, f: np.ndarray) -> np.float64:
    """Mean of the squared 'valid' convolution of z with f (NaN if f is longer than z)."""
    if f.size > z.size:
        return np.float64(np.nan)
    return np.float64(np.mean(np.convolve(z, f, mode='valid') ** 2))


def _wfbmesti(path: ArrayLike) -> np.ndarray:
    """
    Three estimates of the Hurst index H of a fractional Brownian motion path.

    Parameters
    ----------
    path : array_like
        Sampled path (the cumulative sum of the series of interest).

    Returns
    -------
    np.ndarray
        ``[H1, H2, H3]``: (1) discrete second-order derivative estimator, (2) its
        wavelet (sym5) version, (3) slope of the log detail variance of a Haar DWT
        against level. An estimate whose inputs are empty or degenerate is NaN.

    Notes
    -----
    (1) Istas, J. & Lang, G. (1997) "Quadratic variations and estimation of the local
    Holder index of a Gaussian process", Ann. Inst. H. Poincare B 33(4):407-436;
    Kent, J.T. & Wood, A.T.A. (1997) J. R. Stat. Soc. B 59(3):679-699. The mean squared
    second-order difference ``x[t] - 2 x[t+a] + x[t+2a]`` of fBm grows as ``a**(2H)``, so
    ``H = log2(V_2 / V_1) / 2`` for the mean squares ``V_a`` at dilations ``a = 1, 2``
    (all samples are used).
    (2) The same ratio with the sym5 high-pass decomposition filter and its dilation by 2
    (zeros inserted between the taps).
    (3) Flandrin, P. (1992) IEEE Trans. Inf. Theory 38(2):910-917: the variance of the Haar
    detail coefficients at level j grows as ``2**(j (2H + 1))``, so ``H = (b - 1) / 2`` for the
    slope ``b`` of ``log2`` mean squared detail coefficient against level, regressed over all
    levels that have at least 8 coefficients (at least 2 such levels are needed).
    """
    p = np.asarray(path, dtype=float).ravel()
    est = np.full(3, np.nan)
    if p.size < 2 or not np.all(np.isfinite(p)):
        return est
    with np.errstate(divide='ignore', invalid='ignore'):
        # (1) discrete second-order derivative
        a = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
        # filtered energy at rounding-noise level (e.g. a straight-line path) is degenerate
        floor = (1e-10 * np.max(np.abs(p - np.mean(p)))) ** 2
        v1 = _valid_filter_mean_square(p, a)
        v2 = _valid_filter_mean_square(p, a2)
        if v1 > floor and v2 > floor:
            est[0] = 0.5 * np.log2(v2 / v1)

        # (2) wavelet version with the sym5 high-pass decomposition filter
        h = np.asarray(pywt.Wavelet('sym5').dec_hi, dtype=float)
        h2 = np.zeros(2 * h.size)
        h2[::2] = h
        w1 = _valid_filter_mean_square(p, h)
        w2 = _valid_filter_mean_square(p, h2)
        if w1 > floor and w2 > floor:
            est[1] = 0.5 * np.log2(w2 / w1)

        # (3) Haar detail-variance regression (Flandrin 1992)
        min_coeffs = 8
        n_levels = pywt.dwt_max_level(p.size, 'haar')
        if n_levels >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coeffs = pywt.wavedec(p, 'haar', mode='symmetric', level=n_levels)
            details = coeffs[1:][::-1]  # j = 1..n_levels (finest first)
            levels = [j for j, d in enumerate(details, start=1) if d.size >= min_coeffs]
            if len(levels) >= 2:
                log_var = np.array([np.log2(np.mean(details[j - 1] ** 2)) for j in levels])
                if np.all(np.isfinite(log_var)):
                    beta = np.polyfit(np.asarray(levels, dtype=float), log_var, 1)[0]
                    est[2] = (beta - 1.0) / 2.0
    est[~np.isfinite(est)] = np.nan
    return est


def wfbm(x: ArrayLike) -> dict:
    """
    Estimates of the fractal index (Hurst exponent) of a signal.

    The series is treated as the increments of a fractional Brownian motion, so its
    cumulative sum is the path whose Hurst index H is estimated.

    Parameters
    ----------
    x : array_like
        Time series.

    Returns
    -------
    dict
        ``H_deriv2``: estimate from the discrete second-order derivative;
        ``H_deriv2Wavelet``: its wavelet-domain version (sym5). NaN when the series is
        too short for an estimate.

    Notes
    -----
    See :func:`_wfbmesti` for the estimators and references (Istas & Lang 1997;
    Kent & Wood 1997; Flandrin 1992).
    """
    path = np.cumsum(np.asarray(x, dtype=float).ravel())
    est = _wfbmesti(path)
    return {'H_deriv2': float(est[0]), 'H_deriv2Wavelet': float(est[1])}

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
        The maximum scale / level (can be 'max' to set according to ``pywt.dwt_max_level``). Default is 5.
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
        logger.info(f'Chosen level {a_max} is too large for this wavelet on this signal...')
        a_max = max_level # set to max allowed level
        logger.info(f'changed to maximum level computed with pywt.dwt_max_level: {a_max}')

    # % Define scales.
    scales = np.arange(1, a_max+1)
    a = 2**scales
    # Compute associated pseudo-frequencies.
    f = pywt.scale2frequency(w_name, a) / delta  # center frequency / (scale * sampling period)
    # Compute associated pseudo-periods.
    per = 1/f
    #Decompose the time series at level specified as maximum
    coeffs = _wavedec(y, wavelet=w_name, level=a_max)
    #Estimate standard deviation of detail coefficients.
    stdc = []
    for k in range(1, a_max+1):
        d = _detcoef(coeffs, k)
        stdc.append(np.median(np.abs(d)) / _MAD_TO_SIGMA)
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
        The level of wavelet decomposition (can be set to 'max' for the maximum level determined by ``pywt.dwt_max_level``). Default is 3.
            
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
        logger.warning("Chosen level is too large for this wavelet on this signal....\n")
    #%% Perform Wavelet Decomposition
    # if level exceeds max level, just use max level instead
    coeffs = _wavedec(y, wavelet=w_name, level=min(level, max_level_allowed))
    #%% Get statistics on coefficients
    out = {}
    for k in range(1, level+1):
        if k <= max_level_allowed:
            d = _detcoef(coeffs, k) # detail coeffs at level k
            # max coeff at this level
            out[f'maxd_l{k}'] = np.max(d)
            #% minimum coefficient at this level:
            out[f'mind_l{k}'] = np.min(d)
            # std coefficients at this level:
            out[f'stdd_l{k}'] = np.std(d, ddof=1)
            #% 1-D noise coefficient estimate (estimate of the noise std):
            out[f'noisestd_l{k}'] = np.median(np.abs(d)) / _MAD_TO_SIGMA  # robust noise sd (Donoho & Johnstone)
        else:
            # exceeds max level, return nans
            out[f'maxd_l{k}'] = np.nan 
            out[f'mind_l{k}'] = np.nan 
            out[f'stdd_l{k}'] = np.nan
            out[f'noisestd_l{k}'] = np.nan
    
    return out

@dict_output
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
        The wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet), or the continuous
        Morlet wavelet 'morl' (the second wavelet registered in hctsa). For either kind, the
        wavelet function is integrated on a grid of 2**10 points over its support, and the
        integral, stretched to each scale, is convolved with the series and differenced (the
        continuous transform with that wavelet at integer scales).
        Default is ``'db3'``.
    max_scale : int, optional
        The maximum scale of wavelet analysis. Default is 32.

    The scaled power is relative to the mean power, ``SC = S / mean(S)`` with
    ``S = |coeffs|**2`` (so the mean of SC is 1, independent of the number of
    coefficients). ``pover99``, ``pover98``, ``pover95``, ``pover90`` and ``pover80`` are the
    energy shares of the strongest 1, 2, 5, 10 and 20% of coefficients (the sum of the
    largest ``max(1, floor((100 - p)/100*numEntries))`` values of SC, divided by
    ``numEntries``). ``SC_h`` is the entropy of SC relative to its maximum,
    ``-sum(SC_a*log(SC_a)) - log(numEntries)`` with ``SC_a = SC/sum(SC)``.

    Returns
    -------
    dict
        Dictionary of statistics on the CWT coefficients, including ``gam1`` and
        ``gam2`` (the shape and scale of a gamma distribution fitted to SC by maximum
        likelihood; NaN if any scaled power is exactly zero, which the gamma distribution
        cannot describe) and ``dd_SC_h`` (the entropy of the maximum of SC in each of 10 time
        boxes at each scale).
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    scales = np.arange(1, max_scale+1)
    # the wavelet is integrated on a grid of 2**10 points over its support, and the integral,
    # stretched to each scale, is convolved with the series and differenced
    coeffs = _custom_cwt(data=y, scales=scales, wavelet=w_name, precision=10)
    S = np.abs(coeffs * coeffs)
    SC = S/np.mean(S)  # scaled power, relative to the mean power

    # Get statistics from CWT
    num_entries = SC.size
    max_SC = np.max(SC)
    mean_SC = np.mean(SC)
    std_SC = np.std(SC, ddof=1)

    # 1) Coefficients, coeffs
    abs_coeffs = np.abs(coeffs)
    out = {}
    out['meanC'] = np.mean(coeffs)

    out['meanabsC'] = np.mean(abs_coeffs)
    out['medianabsC'] = np.median(abs_coeffs)
    out['maxabsC'] = np.max(abs_coeffs)
    out['maxonmeanC'] = out['maxabsC']/out['meanabsC']

    out['maxonmeanSC'] = max_SC/mean_SC

    #% Proportion of coeffs matrix over ___ maximum (thresholded)
    # Energy share of the strongest (100 - p)% of coefficients
    sc_sorted = np.sort(SC.ravel())[::-1]
    poverfn = lambda p: np.sum(sc_sorted[:max(1, int(np.floor((100 - p) / 100 * num_entries)))]) / num_entries
    out['pover99'] = poverfn(99)
    out['pover98'] = poverfn(98)
    out['pover95'] = poverfn(95)
    out['pover90'] = poverfn(90)
    out['pover80'] = poverfn(80)

    # Gamma distribution fitted to the scaled power (maximum likelihood, as gamfit)
    if np.any(SC <= 0):
        # A gamma distribution has no mass at exactly zero power (a coefficient that is exactly
        # zero, as for an exactly constant stretch of the series), so its maximum-likelihood fit
        # does not exist
        gam_shape = gam_scale = np.nan
    else:
        try:
            gam_shape, _, gam_scale = stats.gamma.fit(SC.ravel(), floc=0)
        except (stats.FitError, ValueError, RuntimeError):
            gam_shape = gam_scale = np.nan
    out['gam1'] = gam_shape
    out['gam2'] = gam_scale

    # 2D entropy (relative to its maximum)
    SC_a = SC/np.sum(SC)
    SC_pos = SC_a[SC_a > 0]  # (0*log(0) = 0 by continuity)
    out['SC_h'] = -np.sum(SC_pos * np.log(SC_pos)) - np.log(num_entries)

    # Entropy of the maximum of SC in each of 10 time boxes at each scale
    num_boxes = 10
    if N < num_boxes:
        logger.warning('Time series too short')
        return np.nan
    cutoffs = _even_split_points(0, N, num_boxes + 1)  # box boundaries, rounded
    dd_SC = np.zeros((max_scale, num_boxes))
    for j in range(num_boxes):
        dd_SC[:, j] = np.max(SC[:, cutoffs[j]:cutoffs[j + 1]], axis=1)
    dd_SC = dd_SC / np.sum(dd_SC)
    dd_SC_o = dd_SC.ravel()
    dd_SC_o = dd_SC_o[dd_SC_o > 0]  # (0*log(0) = 0 by continuity)
    out['dd_SC_h'] = -np.sum(dd_SC_o * np.log(dd_SC_o))

    # Sum across scales
    SSC = sum(SC)
    max_SSC = np.max(SSC)
    out['max_ssc'] = max_SSC
    out['min_ssc'] = np.min(SSC)
    out['maxonmed_ssc'] = max_SSC / np.median(SSC)
    out['pcross_maxssc50'] = np.sum(sign_change(SSC - 0.5 * max_SSC)) / (N - 1)
    out['std_ssc'] = np.std(SSC, ddof=1)

    #Stationarity
    midpoint = N // 2  # Integer division is equivalent to floor
    SC_1 = SC[:, :midpoint]
    SC_2 = SC[:, midpoint:]

    mean2_1 = SC_1.mean()
    mean2_2 = SC_2.mean()

    std2_1 = SC_1.std(ddof=1)
    std2_2 = SC_2.std(ddof=1)

    out['stat_2_m_s'] = np.mean([std2_1, std2_2]) / mean_SC
    out['stat_2_s_m'] = np.std([mean2_1, mean2_2], ddof=1) / std_SC
    out['stat_2_s_s'] = np.std([std2_1, std2_2], ddof=1) / std_SC

    return out

def _slosr(xx: ArrayLike) -> float:
    """
    Helper for :func:`detail_coeffs`: the level at which the sum of values to
    the left most nearly equals the sum of values to the right.
    """
    the_max_level = len(xx)
    if the_max_level < 3:
        # Too few wavelet levels (short series relative to the wavelet) to define a
        # meaningful split point between "left" and "right":
        return np.nan
    slosr = np.zeros(the_max_level-2)
    for i in range(2, the_max_level):
        slosr[i-2] = np.sum(xx[:i-1])/np.sum(xx[i:])

    return np.argmin(np.abs(slosr - 1)) + 2  # the (1-based) level i in 2..L-1 at the split

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
        See the PyWavelets documentation for details. Default is ``'db3'``.
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
        logger.info(f"Chosen wavelet level is too large for the {w_name} wavelet for this signal of length N = {N}")
        max_level = pywt.dwt_max_level(N, w_name)
        logger.info(f"Using a wavelet level of {max_level} instead.")
    # Perform a single-level wavelet decomposition
    means = np.zeros(max_level) # mean detail coefficient magnitude at each level
    medians = np.zeros(max_level) # median detail coefficient magnitude at each level
    maxs = np.zeros(max_level) # max detail coefficient magnitude at each level
    
    # Decompose once at the maximum level; the pyramid algorithm is prefix-stable, so the level-k
    # detail branch is the same whether decomposed to level k or to max_level.
    coeffs = _wavedec(data=y, wavelet=w_name, level=max_level)
    for k in range(1, max_level+1):
        det = _wrcoef(coeffs, w_name, k, N)
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
    out['wheremax_mean'] = np.argwhere(means == means_s[0]).flatten()[0] + 1  # 1-based level
    out['wheremax_median'] = np.argwhere(medians == medians_s[0]).flatten()[0] + 1  # 1-based level
    out['wheremax_max'] = np.argwhere(maxs == maxs_s[0]).flatten()[0] + 1  # 1-based level

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

@dict_output
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

        - 'wb99m', 'wb90m', 'wb75m', 'wb50m', 'wb25m', 'wb10m', 'wb1m': Decay rate statistics
          (the position, as a proportion of the series length, at which the sorted
          absolute detail coefficients first drop below the given proportion of the maximum).
    """
    y = np.asarray(y)
    N = len(y)
    if level == 'max':
        level = pywt.dwt_max_level(N, w_name)
        if level == 0:
            logger.warning("Cannot compute wavelet coefficients (short time series)")
            return np.nan
    
    if pywt.dwt_max_level(N, w_name) < level:
        logger.warning(f"Chosen level, {level}, is too large for this wavelet on this signal.")
        return np.nan
    
    coeffs = _wavedec(y, wavelet=w_name, level=level)
    det = _wrcoef(coeffs, w_name, level, N)
    det_s = np.sort(np.abs(det))[::-1]

    #%% Return statistics
    out = {}
    #% Decay rate stats ('where below _ maximum' = 'wb_m')
    out['wb99m'] = _find_my_threshold(0.99, det_s, N)
    out['wb90m'] = _find_my_threshold(0.90, det_s, N)
    out['wb75m'] = _find_my_threshold(0.75, det_s, N)
    out['wb50m'] = _find_my_threshold(0.50, det_s, N)
    out['wb25m'] = _find_my_threshold(0.25, det_s, N)
    out['wb10m'] = _find_my_threshold(0.10, det_s, N)
    out['wb1m'] = _find_my_threshold(0.01, det_s, N)

    return out

def _wavedec(data: ArrayLike, wavelet: str, mode: str = 'symmetric', level: int = 1) -> list:
    """
    Multilevel 1-D discrete wavelet decomposition.

    Parameters
    ----------
    data : array_like
        1-D real signal of length N.
    wavelet : str
        PyWavelets wavelet name.
    mode : str, optional
        Signal-extension mode (default 'symmetric', half-sample symmetric).
    level : int, optional
        Decomposition level (may exceed the maximal useful level).

    Returns
    -------
    list of np.ndarray
        ``[cA_J, cD_J, cD_{J-1}, ..., cD_1]``, as returned by :func:`pywt.wavedec`.

    Notes
    -----
    Mallat's pyramid algorithm (Mallat 1989, IEEE Trans. PAMI 11(7):674-693) via
    ``pywt.wavedec``.
    """
    x = np.asarray(data, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pywt.wavedec(x, wavelet, mode=mode, level=level)


def _detcoef(coeffs: list, level: int) -> np.ndarray:
    """
    Detail coefficients ``cD_level`` of a decomposition from :func:`_wavedec`
    (``level`` = 1 is the finest level, ``len(coeffs) - 1`` the coarsest).
    """
    J = len(coeffs) - 1
    if not (1 <= level <= J):
        raise ValueError(f"level must be between 1 and {J}")
    return np.asarray(coeffs[J - level + 1], dtype=float)


def _wrcoef(coeffs: list, wavelet: str, level: int, n: int, mode: str = 'symmetric') -> np.ndarray:
    """
    Signal component carried by one detail band: the level-``level`` detail of the
    multiresolution analysis, at the original length ``n``.

    Parameters
    ----------
    coeffs : list of np.ndarray
        Decomposition from :func:`_wavedec`.
    wavelet : str
        PyWavelets wavelet name used for the decomposition.
    level : int
        Detail level to reconstruct (1 = finest).
    n : int
        Length of the original signal.

    Returns
    -------
    np.ndarray
        Length ``n``; the inverse transform (:func:`pywt.waverec`) with every coefficient
        block zero except ``cD_level``. Summing these components over all levels and adding the
        analogous approximation component reconstructs the signal.
    """
    J = len(coeffs) - 1
    if not (1 <= level <= J):
        raise ValueError(f"level must be between 1 and {J}")
    blocks = [np.zeros_like(c) for c in coeffs]
    blocks[J - level + 1] = np.asarray(coeffs[J - level + 1], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rec = pywt.waverec(blocks, wavelet, mode=mode)
    return np.asarray(rec[:n], dtype=float)

def _find_my_threshold(x: float, det_s: ArrayLike, N: int):
    """
    Fraction of the way into ``det_s`` (sorted descending) at which the
    coefficients first drop below ``x`` times the maximum: the 1-based position of the
    first such coefficient divided by ``N``.
    """
    below = det_s < x * np.max(det_s)
    if not below.any():
        return np.nan

    return (np.argmax(below) + 1)/N

def _modwt(x: ArrayLike, w_name: str, level: int) -> np.ndarray:
    """
    Maximal-overlap discrete wavelet transform (periodic boundary).

    Parameters
    ----------
    x : array_like
        1-D real signal, length N >= 2.
    w_name : str
        Name of an orthogonal PyWavelets wavelet.
    level : int
        Number of levels, 1 <= level <= floor(log2 N).

    Returns
    -------
    np.ndarray
        Shape (level + 1, N): rows 0..level-1 are the wavelet coefficients W_1..W_level,
        row ``level`` the scaling coefficients V_level (no time-alignment shift).

    Notes
    -----
    Percival, D.B. & Walden, A.T. (2000) "Wavelet Methods for Time Series Analysis",
    CUP, ch. 5 (MODWT pyramid algorithm): with the filters ``h/sqrt(2)``, ``g/sqrt(2)``,
    ``W_{j,t} = sum_l h~_l V_{j-1,(t - 2^{j-1} l) mod N}`` and likewise for V_j. The
    filter pair is the reconstruction pair (``rec_hi``, ``rec_lo``) of the wavelet.
    """
    wav = pywt.Wavelet(w_name)
    if not wav.orthogonal:
        raise ValueError("MODWT requires an orthogonal wavelet")
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    ht = np.asarray(wav.rec_hi, dtype=float) / np.sqrt(2.0)
    gt = np.asarray(wav.rec_lo, dtype=float) / np.sqrt(2.0)
    W = np.empty((level + 1, N), dtype=float)
    V = x
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        Wj = np.zeros(N)
        Vj = np.zeros(N)
        for l in range(ht.size):
            shifted = np.roll(V, (step * l) % N)  # shifted[t] = V[(t - step*l) mod N]
            Wj += ht[l] * shifted
            Vj += gt[l] * shifted
        W[j - 1] = Wj
        V = Vj
    W[level] = V
    return W

def _modwt_var(w: np.ndarray, w_name: str) -> np.ndarray:
    """
    Unbiased wavelet variance by level from MODWT coefficients.

    Parameters
    ----------
    w : np.ndarray
        Output of :func:`_modwt`, shape (J + 1, N).
    w_name : str
        Wavelet name (its filter length L sets the boundary width).

    Returns
    -------
    np.ndarray
        For levels j = 1..J' (J' = largest j <= J with M_j = N - L_j + 1 >= 1, where
        L_j = (2^j - 1)(L - 1) + 1) the mean square ``sum(W_j**2) / M_j`` of the M_j
        boundary-unaffected wavelet coefficients (the last M_j columns). The scaling-level
        value is appended only if all J levels are retained: the variance (about the mean,
        divisor M_J) of the M_J boundary-unaffected scaling coefficients. Empty if M_1 < 1.

    Notes
    -----
    Percival & Walden (2000), "Wavelet Methods for Time Series Analysis", ch. 8: the unbiased
    estimator of the wavelet variance at scale ``2^(j-1)`` uses only the coefficients
    unaffected by the periodic boundary. Wavelet coefficients have zero mean, so they are not
    centred; the scaling coefficients carry the mean of the series, which is not variance, so
    for the scaling band the mean is removed. Levels with no boundary-unaffected coefficient
    (M_j < 1) are dropped.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    L = int(pywt.Wavelet(w_name).dec_len)
    J = w.shape[0] - 1
    N = w.shape[1]
    out = []
    kept = 0
    for j in range(1, J + 1):
        Lj = (2 ** j - 1) * (L - 1) + 1
        Mj = N - Lj + 1
        if Mj < 1:
            break
        seg = w[j - 1, N - Mj:]
        out.append(np.mean(seg ** 2))
        kept = j
    if kept == J and J >= 1:
        Lj = (2 ** J - 1) * (L - 1) + 1
        Mj = N - Lj + 1
        seg = w[J, N - Mj:]
        out.append(np.mean((seg - np.mean(seg)) ** 2))
    return np.asarray(out, dtype=float)

@dict_output
def modwt_var(y: ArrayLike, w_name: str = 'db3', level: Union[int, str] = 5) -> dict:
    """
    Multiscale variance decomposition via the maximal overlap DWT.

    Decomposes the time series into octave-scale bands using the maximal
    overlap discrete wavelet transform (MODWT) and computes summary statistics
    on how variance is distributed across scales. Unlike the standard
    (decimated) DWT used elsewhere in this codebase, the MODWT is shift-invariant
    and its associated variance estimator is unbiased and accounts
    for boundary-affected coefficients at each level -- the standard approach
    for a scale-wise variance decomposition (Percival & Walden).

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The mother wavelet, e.g., ``'db3'``, ``'sym2'``. Default is ``'db3'``.
    level : int or 'max', optional
        The level of wavelet decomposition (can be set to 'max' for the maximum
        level supported by the series length, ``floor(log2(N))``). Default is 5.

    Returns
    -------
    dict
        Dictionary containing:

        - 'scalingFrac': the fraction of variance in the lowest-frequency
          (scaling/trend) band -- the part of the signal not resolved by any detail level.
        - 'domlevel': the level (1 = highest frequency, ..., level+1 = scaling/trend
          band) that carries the most variance.
        - 'decaySlope': the slope of log2(variance) vs. level across the detail bands --
          a wavelet-based scaling exponent, analogous to a Hurst estimate but using the
          MODWT's unbiased, boundary-corrected variance rather than an ad hoc regression
          on raw coefficients.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y) # length of the time series

    maxLevelAllowed = int(np.floor(np.log2(N)))
    if level == 'max':
        level = maxLevelAllowed
    if maxLevelAllowed < level:
        logger.info(f"Chosen level ({level}) is too large for a series of length N = {N}")
        level = maxLevelAllowed
        logger.info(f"Using a level of {level} instead.")
    if level < 1:
        logger.warning("Time series too short for a MODWT decomposition")
        return np.nan

    #%% MODWT and its unbiased multiscale variance decomposition
    w = _modwt(y, w_name, level)
    wvar = _modwt_var(w, w_name) # nominally length level+1: levels 1..level (detail), then scaling/trend
    if wvar.size == 0:
        logger.warning("No boundary-unaffected coefficients for an unbiased MODWT variance estimate")
        return np.nan
    numLevelsReturned = wvar.size - 1 # number of detail levels actually returned

    #%% Return statistics
    out = {}
    totalVar = np.sum(wvar)
    out['scalingFrac'] = wvar[-1]/totalVar # variance fraction in the trend/scaling band

    out['domlevel'] = int(np.argmax(wvar)) + 1 # level (1..numLevelsReturned+1) carrying the most variance

    # Scaling exponent: log2(variance) vs. level, across detail bands only
    detailVar = wvar[:numLevelsReturned]
    validLevels = np.flatnonzero(detailVar > 0) + 1
    if validLevels.size >= 2:
        p = np.polyfit(validLevels, np.log2(detailVar[validLevels-1]), 1)
        out['decaySlope'] = p[0]
    else:
        out['decaySlope'] = np.nan

    return out


def _shannon_entropy(c: np.ndarray) -> float:
    """
    Additive (non-normalized) Shannon entropy cost of a coefficient vector.

    Parameters
    ----------
    c : np.ndarray
        1-D real coefficient vector (may be empty).

    Returns
    -------
    float
        ``-sum_{c_i != 0} c_i^2 ln(c_i^2)`` (natural log, ``0 ln 0 = 0``); 0.0 for empty
        input, NaN if any entry is NaN or infinite.

    Notes
    -----
    Cost function of Coifman, R.R. & Wickerhauser, M.V. (1992) "Entropy-based algorithms
    for best basis selection", IEEE Trans. Inf. Theory 38(2):713-718; also the
    non-normalized Shannon entropy cost.
    """
    c = np.asarray(c, dtype=float).ravel()
    if c.size == 0:
        return 0.0
    if not np.all(np.isfinite(c)):
        return float('nan')
    s = c ** 2
    nz = s > 0
    return float(-np.sum(s[nz] * np.log(s[nz])))


@dict_output
def wpd_best_tree(y: ArrayLike, w_name: str = 'db3', max_level: Union[int, str] = 5) -> Union[dict, float]:
    """
    Adaptive best-basis wavelet packet decomposition.

    Decomposes the time series with a full wavelet packet tree and then prunes it to the
    entropy-optimal ("best basis") subtree using the Coifman-Wickerhauser algorithm: working
    up from the deepest level, a node's two children are kept only if their (best-subtree)
    Shannon entropies sum to strictly less than the node's own. Unlike a standard DWT, which
    always splits the frequency axis into the same octave bands, the best basis is a
    non-uniform partition tailored to the signal. The statistics summarize the shape of that
    adaptively chosen partition and how energy is spread across it.

    Parameters
    ----------
    y : array-like
        The input time series.
    w_name : str, optional
        The mother wavelet, e.g., 'db3', 'sym2'. Default is 'db3'.
    max_level : int or 'max', optional
        The maximum depth of the initial (pre-pruning) wavelet packet tree; 'max' uses the
        maximum level allowed for the series length and wavelet. Default is 5 (reduced to the
        maximum allowed level for short series).

    Returns
    -------
    dict or float
        Dictionary containing:

        - `numLeaves`: the number of terminal nodes in the best-basis tree.
        - `meanDepth`: the mean depth of terminal nodes.
        - `stdDepth`: the spread of terminal-node depths (zero if the best basis is uniform
          in depth).
        - `entropy`: the Shannon entropy of the energy distribution across terminal nodes.

        NaN if the series is too short for a wavelet packet decomposition.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    w = pywt.Wavelet(w_name)
    max_level_allowed = pywt.dwt_max_level(N, w)
    if max_level == 'max':
        max_level = max_level_allowed
    if max_level_allowed < max_level:
        logger.info(f'Chosen level ({max_level}) is too large for the {w_name} wavelet on this signal (N = {N}); '
                    f'using {max_level_allowed} instead.')
        max_level = max_level_allowed
    if max_level < 1:
        return np.nan  # time series too short for a wavelet packet decomposition

    # Full wavelet packet tree (symmetric extension), keyed by path ('' = root; 'a'/'d' = low/high-pass child)
    data = {'': y}
    for depth in range(max_level):
        for path in [p for p in data if len(p) == depth]:
            a, d = pywt.dwt(data[path], w, mode='symmetric')
            data[path + 'a'], data[path + 'd'] = a, d
    ent = {p: _shannon_entropy(c) for p, c in data.items()}

    # Best basis, bottom-up: split a node only if its children's best entropies sum to less than its own
    best_ent = {}
    is_split = {}
    for path in sorted(data, key=len, reverse=True):
        if len(path) == max_level:
            best_ent[path] = ent[path]
            is_split[path] = False
            continue
        echild = best_ent[path + 'a'] + best_ent[path + 'd']
        if echild < ent[path]:
            best_ent[path] = echild
            is_split[path] = True
        else:
            best_ent[path] = ent[path]
            is_split[path] = False

    leaves = []
    stack = ['']
    while stack:
        p = stack.pop()
        if is_split[p]:
            stack.extend([p + 'a', p + 'd'])
        else:
            leaves.append(p)
    depths = np.array([len(p) for p in leaves], dtype=float)
    E = np.array([np.sum(data[p] ** 2) for p in leaves])
    E = E / np.sum(E)  # proportion of the energy in the terminal nodes

    out = {
        'numLeaves': len(leaves),
        'meanDepth': np.mean(depths),
        'stdDepth': np.std(depths, ddof=1) if depths.size > 1 else 0.0,
    }
    p = E[E > 0]
    out['entropy'] = -np.sum(p * np.log(p))
    return out
