import numpy as np
from numpy.typing import ArrayLike
import scipy.fft
import statsmodels.api as sm

from ..operations.correlation import autocorr, first_crossing
from ..operations.distribution import moments
from ..utils import make_mat_buffer, sign_change

def spectral_summaries(y: ArrayLike, psd_meth: str = 'fft', window_type: str = 'none') -> dict:
    """
    Statistics of the power spectrum of a time series.

    Computes a range of statistics summarizing the power spectrum of a time series.
    The spectrum can be estimated using a periodogram, fast Fourier transform (FFT), 
    or Welch's method.

    Parameters
    ----------
    y : array-like
        The input time series.
    psd_meth : {'periodogram', 'fft', 'welch'}, optional
        The method for obtaining the spectrum from the signal:

        - 'periodogram': periodogram
        - 'fft': fast Fourier transform (default)
        - 'welch': Welch's method

    window_type : {'boxcar', 'rect', 'bartlett', 'hann', 'hamming', 'none'}, optional
        The window to use for spectral estimation:
            
        - 'boxcar'
        - 'rect'
        - 'bartlett'
        - 'hann'
        - 'hamming'
        - 'none' (default)

    Returns
    -------
    dict
        Statistics summarizing various properties of the spectrum.
    """

    y = np.asarray(y)
    ny = len(y)

    window = None
    # Set window (for periodogram and welch):
    if window_type == 'none':
        window = []
    elif window_type == 'hamming':
        window = np.hamming(ny)
    elif window_type == 'hann':
        window = np.hanning(ny)
    elif window_type == 'bartlett':
        window = np.bartlett(ny)
    elif window_type == 'boxcar':
        window = scipy.signal.windows.boxcar(ny)
    elif window_type == 'rect':
        window = np.ones(ny)
    else:
        raise ValueError(f"Unknown window: {window_type}")

    # Compute the Fourier Transform
    if psd_meth == 'fft':
        fs = 1  # sampling freq
        nfft = 2 ** (int(np.ceil(np.log2(ny))))  # next power of 2
        f = (fs / 2) * np.linspace(0, 1, int(nfft / 2) + 1)  # freq
        w = 2 * np.pi * f  # angular freq
        s = scipy.fft.fft(y, nfft)  # do the fourier transform
        s = 2 * np.abs(s[:int(nfft / 2) + 1]) ** 2 / ny  # single-sided power spectral density
        s = s / (2 * np.pi)  # convert to angular freq space

    elif psd_meth == 'welch':
        # welch power spectral density estimate
        fs = 1
        n = 2 ** (int(np.ceil(np.log2(ny))))
        f, s = scipy.signal.welch(y, window=window, noverlap=0, nfft=n, fs=fs)
        w = 2 * np.pi * f  # angular frequency
        s = s / (2 * np.pi)  # adjust so that area remains normalized in angular frequency space
    elif psd_meth == 'periodogram':
        raise NotImplementedError("Periodogram is not yet available.")
    # elif psd_meth == 'periodogram':
    #     if nf:
    #         w = np.linspace(0, np.pi, nf)
    #         S, w = scipy.signal.periodogram(y, window=window, )
    #     else:
    #         w, S = scipy.signal.periodogram(y, window=window)
    else:
        raise ValueError(f"Unknown spectral estimation method: {psd_meth}.")

    if not np.any(np.isfinite(s)):
        return np.nan

    n = len(s)
    log_s = np.log(s)
    dw = w[1] - w[0]  # spacing increment in w

    # Simple measures of the power spectrum
    # Peaks
    out = {}
    i_max_s = np.argmax(s)
    out = {'maxS': s[i_max_s], 'maxw': w[i_max_s]}
    r, l = np.where(s[i_max_s + 1:] < s[i_max_s])[0], np.where(s[:i_max_s] < s[i_max_s])[0]
    out['maxWidth'] = w[i_max_s + 1 + r[0]] - w[l[-1]] if len(r) > 0 and len(l) > 0 else 0

    right_indices = np.where(s[i_max_s + 1:] < out['maxS'])[0]
    if len(right_indices) > 0:
        right_idx = i_max_s + 1 + right_indices[0]
    else:
        right_idx = None

    # Find last index before i_maxS where S < maxS
    left_indices = np.where(s[:i_max_s] < out['maxS'])[0]
    if len(left_indices) > 0:
        left_idx = left_indices[-1]
    else:
        left_idx = None

    # Calculate maxWidth
    if right_idx is not None and left_idx is not None:
        out['maxWidth'] = w[right_idx] - w[left_idx]
    else:
        out['maxWidth'] = 0

    min_dist_w = 0.02
    pts_per_w = len(s) / np.pi
    min_pk_dist = np.ceil(min_dist_w * pts_per_w)
    pk_height, pk_loc = _findpeaks(s, min_pk_dist, 'descend')
    pk_width = scipy.signal.peak_widths(s, pk_loc)[0]
    pk_prom = (scipy.signal.peak_prominences(s, pk_loc)[0])
    pk_width = pk_width / pts_per_w
    pk_loc = pk_loc / pts_per_w  # diff due to indexing difference

    # Characterize mean peak prominence
    out['numPeaks'] = len(pk_height)
    out['numPromPeaks_1'] = np.sum(pk_prom > 1)  # number of peaks with prominence of at least 1
    out['numPromPeaks_2'] = np.sum(pk_prom > 2)  # number of peaks with prominence of at least 2
    out['numPromPeaks_5'] = np.sum(pk_prom > 5)  # number of peaks with prominence of at least 5
    # number of peaks with prominence greater than the mean (low for skewed distn)
    out['numPeaks_overmean'] = np.sum(pk_prom > np.mean(pk_prom))
    out['maxProm'] = np.max(pk_prom)
    # mean peak prominence of those with prominence of at least 2
    out['meanProm_2'] = np.mean(pk_prom[pk_prom > 2])
    out['meanPeakWidth_prom2'] = np.mean(pk_width[pk_prom > 2])
    out['width_weighted_prom'] = np.sum(pk_width * pk_prom) / np.sum(pk_prom)

    # Power in top N peaks
    nn = lambda x: np.arange(0, np.minimum(x, out['numPeaks'] - 1))
    out['peakPower_2'] = np.sum(pk_height[nn(2)] * pk_width[nn(2)])
    out['peakPower_5'] = np.sum(pk_height[nn(5)] * pk_width[nn(5)])
    # power in peaks with prominence of at least 2
    out['peakPower_prom2'] = np.sum(pk_height[pk_prom > 2] * pk_width[pk_prom > 2])
    # note any features which depend on pKLoc will yield slightly diff answers due to one-indexing,
    # but should be perfectly correlated
    out['w_weighted_peak_prom'] = np.sum(pk_loc * pk_prom) / np.sum(pk_prom)
    #where are prominent peaks located on average (weighted by height)
    out['w_weighted_peak_height'] = np.sum(pk_loc * pk_height) / np.sum(pk_height)
    # Number of peaks required to get to 50% of power in peaks
    peak_power = pk_height * pk_width
    out['numPeaks_50power'] = np.where(np.cumsum(peak_power) > 0.5 * np.sum(peak_power))[0][0]
    out['peakpower_1'] = peak_power[0] / sum(peak_power)

    # Distribution
    # quantiles
    iqr75 = np.quantile(s, 0.75, method='hazen')
    iqr25 = np.quantile(s, 0.25, method='hazen')
    out['iqr'] = iqr75 - iqr25
    out['logiqr'] = np.quantile(log_s, 0.75, method='hazen') - np.quantile(log_s, 0.25, method='hazen')
    out['q25'] = iqr25
    out['median'] = np.median(s)
    out['q75'] = iqr75

    # Moments
    out['std'] = np.std(s, ddof=1)
    out['stdlog'] = np.log(out['std'])
    out['logstd'] = np.std(log_s, ddof=1)
    out['mean'] = np.mean(s)
    out['logmean'] = np.mean(log_s)
    for i in range(3, 6):
        out[f'mom{i}'] = moments(s, i)

    # Autocorrelation of amplitude spectrum:
    auto_corrs_s = autocorr(s, [1, 2, 3, 4], 'Fourier')
    out['ac1'] = auto_corrs_s[0]
    out['ac2'] = auto_corrs_s[1]
    out['tau'] = first_crossing(s, 'ac', 0, 'continuous')  # first zero crossing

    # Shape of cumulative sum curve
    cs_s = np.cumsum(s)
    f_frac_w_max = lambda frac: w[np.where(cs_s >= cs_s[-1] * frac)[0][0]]
    # @ what frequency is csS a fraction p of its maximum?
    out['wmax_5'] = f_frac_w_max(0.05)
    out['wmax_10'] = f_frac_w_max(0.1)
    out['wmax_25'] = f_frac_w_max(0.25)
    out['centroid'] = f_frac_w_max(0.5)
    out['wmax_75'] = f_frac_w_max(0.75)
    out['wmax_90'] = f_frac_w_max(0.9)
    out['wmax_95'] = f_frac_w_max(0.95)
    out['wmax_99'] = f_frac_w_max(0.99)

    #Width of saturation measures
    out['w10_90'] = out['wmax_90'] - out['wmax_10']  # % from 10% to 90%:
    out['w25_75'] = out['wmax_75'] - out['wmax_25']

    # Fit some functions to this cumulative sum:
    # Quadratic
    a, b, c = np.polyfit(w, cs_s, deg=2)
    out['fpoly2csS_p1'] = a
    out['fpoly2csS_p2'] = b
    out['fpoly2csS_p3'] = c
    quad = lambda x, a, b, c: a * x**2 + b * x + c
    residuals = quad(w, a, b, c) - cs_s
    sum_sq_err = np.sum(residuals**2)
    out['fpoly2_sse'] = sum_sq_err
    out['fpoly2_r2'] = 1 - (sum_sq_err / (np.sum((cs_s - np.mean(cs_s))**2)))

    # Fit polysat a*x^2/(b+x^2) (has zero derivative at zero, though)
    # polysat = lambda x, a, b : (a*(x**2))/(b + x**2)
    # popt, _ = curve_fit(polysat, w, csS, p0=[csS[-1], 100])
    # a, b = popt
    # out['fpolysat_a'] = a
    # out['fpolysat_b'] = b
    # residuals = polysat(w, a, b) - csS
    # sum_sq_err = np.sum(residuals**2)
    # out['fpolysat_r2'] = 1 - (sum_sq_err/(np.sum((csS - np.mean(csS))**2)))
    # out['fpolysat_rmse'] = np.sqrt(np.mean(residuals**2))

    # Shannon spectral entropy
    h_shann = -s * np.log(s)
    out['spect_shann_ent'] = np.sum(h_shann)
    out['spect_shann_ent_norm'] = np.mean(h_shann)

    #"Spectral Flatness Measure"
    #which is given in dB as 10 log_10(gm/am) where gm is the geometric mean and am
    # is the arithmetic mean of the power spectral density
    out['sfm'] = 10 * np.log10(np.exp(np.mean(np.log(s))) / np.mean(s))

    # Areas under power spectrum
    out['areatopeak'] = np.sum(s[0:np.argmax(s) + 1]) * dw
    out['ylogareatopeak'] = np.sum(log_s[0:np.argmax(s) + 1]) * dw  # % (semilogy)

    # Robust Fits
    # across full range
    r_all = w > 0
    across_full_range_res = give_me_robust_stats(np.log(w[r_all]), np.log(s[r_all]), 'linfitloglog_all')
    out = out | across_full_range_res
    # across first half (low frequency)
    r_lf = (w > 0)
    r_lf[int(np.floor(n/2)):] = 0 #% remove second half of angular frequenciesf
    first_half_res = give_me_robust_stats(np.log(w[r_lf]), np.log(s[r_lf]), 'linfitloglog_lf')
    out = out | first_half_res
    # across second half (high frequency)
    r_hf = np.arange(n // 2, n)
    second_half_res = give_me_robust_stats(np.log(w[r_hf]), np.log(s[r_hf]), 'linfitloglog_hf')
    out = out | second_half_res
    #Middle half (mid-frequencies)
    start = int(np.round(n / 4)) - 1
    stop = int(np.round(n * 3 / 4))
    r_mf = np.arange(start, stop)
    middle_half_res = give_me_robust_stats(np.log(w[r_mf]), np.log(s[r_mf]), 'linfitloglog_mf')
    out = out | middle_half_res
    #Fit linear to semilog plot (across full range)
    res_semilog = give_me_robust_stats(w, np.log(s), 'linfitsemilog_all')
    out = out | res_semilog

    # Power in specific frequency bands
    # % 2 bands
    split = make_mat_buffer(s, int(np.floor(n / 2)))
    if split.shape[1] > 2:
        split = split[:, :2]
    out['area_2_1'] = np.sum(split[:, 0]) * dw
    out['logarea_2_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_2_2'] = np.sum(split[:, 1]) * dw
    out['logarea_2_2'] = np.sum(np.log(split[:, 1])) * dw
    out['statav2_m'] = np.std(np.mean(split, axis=0), ddof=1) / np.std(s, ddof=1)
    out['statav2_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)

    # 3 bands
    split = make_mat_buffer(s, int(np.floor(n / 3)))
    if split.shape[1] > 3:
        split = split[:, :3]
    out['area_3_1'] = np.sum(split[:, 0]) * dw
    out['logarea_3_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_3_2'] = np.sum(split[:, 1]) * dw
    out['logarea_3_2'] = np.sum(np.log(split[:, 1])) * dw
    out['area_3_3'] = np.sum(split[:, 2]) * dw
    out['logarea_3_3'] = np.sum(np.log(split[:, 2])) * dw
    out['statav3_m'] = np.std(np.mean(split, axis=0), ddof=1) / np.std(s, ddof=1)
    out['statav3_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)

    # 4 bands
    split = make_mat_buffer(s, int(np.floor(n / 4)))
    if split.shape[1] > 4:
        split = split[:, :4]
    out['area_4_1'] = np.sum(split[:, 0]) * dw
    out['logarea_4_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_4_2'] = np.sum(split[:, 1]) * dw
    out['logarea_4_2'] = np.sum(np.log(split[:, 1])) * dw
    out['area_4_3'] = np.sum(split[:, 2]) * dw
    out['logarea_4_3'] = np.sum(np.log(split[:, 2])) * dw
    out['area_4_4'] = np.sum(split[:, 3]) * dw
    out['logarea_4_4'] = np.sum(np.log(split[:, 3])) * dw
    out['statav4_m'] = np.std(np.mean(split, axis=0), ddof=1) / np.std(s, ddof=1)
    out['statav4_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)

    # 5 bands
    split = make_mat_buffer(s, int(np.floor(n / 5)))
    if split.shape[1] > 5:
        split = split[:, :5]
    out['area_5_1'] = np.sum(split[:, 0]) * dw
    out['logarea_5_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_5_2'] = np.sum(split[:, 1]) * dw
    out['logarea_5_2'] = np.sum(np.log(split[:, 1])) * dw
    out['area_5_3'] = np.sum(split[:, 2]) * dw
    out['logarea_5_3'] = np.sum(np.log(split[:, 2])) * dw
    out['area_5_4'] = np.sum(split[:, 3]) * dw
    out['logarea_5_4'] = np.sum(np.log(split[:, 3])) * dw
    out['area_5_5'] = np.sum(split[:, 4]) * dw
    out['logarea_5_5'] = np.sum(np.log(split[:, 4])) * dw
    out['statav5_m'] = np.std(np.mean(split, axis=0), ddof=1) / np.std(s, ddof=1)
    out['statav5_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)

    # Count crossings:
    # Get a horizontal line and count the number of crossings with the power spectrum
    ncrossfn_rel = lambda frac: np.sum(sign_change(s - frac * np.max(s)))
    out['ncross_f05'] = ncrossfn_rel(0.05)
    out['ncross_f01'] = ncrossfn_rel(0.1)
    out['ncross_f02'] = ncrossfn_rel(0.2)
    out['ncross_f05'] = ncrossfn_rel(0.5)

    return out

def _findpeaks(s, min_pk_dist=0, sort_str='none'):
    """
    Parameters:
    S: input signal
    minPkDist: minimum peak distance
    sort_str: 'none', 'ascend', or 'descend'

    Returns:
    pkHeight, pkLoc
    """
    # find ALL local maxima
    # a peak is considered to be a point higher than both neighbors
    # Handle infinite values
    inf_peaks = np.where(np.isinf(s) & (s > 0))[0]

    # Find finite peaks by checking if each point is greater than both neighbors
    finite_peaks = []
    for i in range(1, len(s) - 1):
        if not np.isinf(s[i]) and not np.isnan(s[i]):
            if s[i] > s[i - 1] and s[i] > s[i + 1]:
                finite_peaks.append(i)

    finite_peaks = np.array(finite_peaks, dtype=int)

    # Combine finite and infinite peaks
    all_peaks = np.concatenate([finite_peaks, inf_peaks]) if len(inf_peaks) > 0 else finite_peaks
    all_peaks = np.sort(all_peaks)

    if len(all_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # apply minimum peak distance constraint
    if min_pk_dist > 0:
        # start with largest peaks and remove smaller ones in neighborhood
        peak_heights = s[all_peaks]

        # sort by height (descending)
        sort_idx = np.argsort(peak_heights)[::-1]
        sorted_peaks = all_peaks[sort_idx]

        # keep track of which peaks to delete
        to_delete = np.zeros(len(sorted_peaks), dtype=bool)

        for i in range(len(sorted_peaks)):
            if not to_delete[i]:
                current_peak = sorted_peaks[i]
                # mark all peaks within minPkDist of current peak for deletion
                for j in range(len(sorted_peaks)):
                    if not to_delete[j]:
                        distance = abs(sorted_peaks[j] - current_peak)
                        if distance <= min_pk_dist and distance > 0:
                            to_delete[j] = True

        # keep only non-deleted peaks
        final_peaks = sorted_peaks[~to_delete]

        # convert back to original indices for sorting
        back_to_original = np.zeros(len(final_peaks), dtype=int)
        for i, peak in enumerate(final_peaks):
            back_to_original[i] = np.where(all_peaks == peak)[0][0]

        final_peaks = all_peaks[np.sort(back_to_original)]
    else:
        final_peaks = all_peaks

    if len(final_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    pk_height = s[final_peaks]
    pk_loc = final_peaks.astype(int)

    if sort_str == 'descend':
        sort_idx = np.argsort(pk_height)[::-1]
        pk_height = pk_height[sort_idx]
        pk_loc = pk_loc[sort_idx]
    elif sort_str == 'ascend':
        sort_idx = np.argsort(pk_height)
        pk_height = pk_height[sort_idx]
        pk_loc = pk_loc[sort_idx]

    return pk_height, pk_loc

def give_me_robust_stats(x_data, y_data, field_name) -> dict:
    """
    Statistics based on a robust linear fit
    """
    x = sm.add_constant(x_data)
    rlm = sm.RLM(y_data, x, M=sm.robust.norms.TukeyBiweight())
    out = {}
    try:
        results = rlm.fit()
        linfit = results.params  # [intercept, slope]
        out[f'{field_name}_a1'] = linfit[0]  # linear fit intercept
        out[f'{field_name}_a2'] = linfit[1]  # linear fit gradient
        out[f'{field_name}_sea1'] = results.bse[0]  # standard error in intercept
        out[f'{field_name}_sea2'] = results.bse[1]  # standard error in slope
    except Exception:
        out[f'{field_name}_a1'] = np.nan
        out[f'{field_name}_a2'] = np.nan
        out[f'{field_name}_sea1'] = np.nan
        out[f'{field_name}_sea2'] = np.nan
    return out
