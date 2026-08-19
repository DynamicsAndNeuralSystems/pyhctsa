import numpy as np
from numpy.typing import ArrayLike
import scipy.fft

from ..toolboxes.matlab.matlab_fit import lsqcurvefit_trr, goodness_of_fit, robustfit

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
        - 'fft': fast Fourier transform
        - 'welch': Welch's method

        Default is ``'fft'``.

    window_type : {'boxcar', 'rect', 'bartlett', 'hann', 'hamming', 'none'}, optional
        The window to use for spectral estimation:

        - 'boxcar'
        - 'rect'
        - 'bartlett'
        - 'hann'
        - 'hamming'
        - 'none'

        Default is ``'none'``.

    Returns
    -------
    dict
        Statistics summarizing various properties of the spectrum.
    """

    y = np.asarray(y)
    ny = len(y)

    if psd_meth == 'welch':
        win_length = max(min(256, int(np.floor(ny / 4 + 0.5))), 16)
    else:
        win_length = ny

    if window_type == 'none':
        # MATLAB's pwelch falls back to a Hamming window when passed []
        window = np.hamming(win_length) if psd_meth == 'welch' else []
    elif window_type == 'hamming':
        window = np.hamming(win_length)
    elif window_type == 'hann':
        window = np.hanning(win_length)
    elif window_type == 'bartlett':
        window = np.bartlett(win_length)
    elif window_type == 'boxcar':
        window = scipy.signal.windows.boxcar(win_length)
    elif window_type == 'rect':
        window = np.ones(win_length)
    else:
        raise ValueError(f"Unknown window: {window_type}")

    if psd_meth == 'fft':
        fs = 1
        nfft = 2 ** (int(np.ceil(np.log2(ny))))
        f = (fs / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        w = 2 * np.pi * f
        s = scipy.fft.fft(y, nfft)
        s = 2 * np.abs(s[:int(nfft / 2) + 1]) ** 2 / ny
        s = s / (2 * np.pi)

    elif psd_meth == 'welch':
        fs = 1
        n = max(256, 2 ** int(np.ceil(np.log2(len(window)))))
        f, s = scipy.signal.welch(y, window=window, noverlap=None, nfft=n, fs=fs)
        w = 2 * np.pi * f
        s = s / (2 * np.pi)
    elif psd_meth == 'periodogram':
        win = np.ones(ny) if (window is None or len(window) == 0) else np.asarray(window)
        nfft = max(256, 2 ** int(np.ceil(np.log2(ny))))
        f, s = scipy.signal.periodogram(
            y, fs=1, window=win, nfft=nfft, detrend=False,
            return_onesided=True, scaling='density'
        )
        w = 2 * np.pi * f
        s = s / (2 * np.pi)
    else:
        raise ValueError(f"Unknown spectral estimation method: {psd_meth}.")

    if not np.any(np.isfinite(s)):
        return np.nan

    n = len(s)
    log_s = np.log(s)
    dw = w[1] - w[0]

    out = {}
    i_max_s = np.argmax(s)
    out = {'maxS': s[i_max_s], 'maxw': w[i_max_s]}

    half_power = out['maxS'] / 2
    right_indices = np.where(s[i_max_s + 1:] < half_power)[0]
    i_upper = i_max_s + 1 + right_indices[0] if len(right_indices) > 0 else len(s) - 1
    left_indices = np.where(s[:i_max_s] < half_power)[0]
    i_lower = left_indices[-1] if len(left_indices) > 0 else 0
    out['maxWidth'] = w[i_upper] - w[i_lower]

    min_dist_w = 0.02
    pts_per_w = len(s) / np.pi
    min_pk_dist = np.ceil(min_dist_w * pts_per_w)
    pk_height, pk_loc = _findpeaks(log_s, min_pk_dist, 'descend')
    pk_width = scipy.signal.peak_widths(log_s, pk_loc)[0]
    pk_prom = (scipy.signal.peak_prominences(log_s, pk_loc)[0])
    pk_width = pk_width / pts_per_w
    pk_loc = pk_loc / pts_per_w

    num_peaks = len(pk_height)
    out['numPeaks'] = num_peaks
    out['numPromPeaks_3'] = np.sum(pk_prom > 3)
    out['numPromPeaks_5'] = np.sum(pk_prom > 5)
    out['numPromPeaks_8'] = np.sum(pk_prom > 8)
    out['numPeaks_overmean'] = np.sum(pk_prom > np.mean(pk_prom)) if num_peaks > 0 else np.nan
    out['maxProm'] = np.max(pk_prom) if num_peaks > 0 else np.nan
    out['meanProm_5'] = np.mean(pk_prom[pk_prom > 5]) if np.any(pk_prom > 5) else np.nan
    out['meanPeakWidth_prom5'] = np.mean(pk_width[pk_prom > 5]) if np.any(pk_prom > 5) else np.nan
    out['width_weighted_prom'] = np.sum(pk_width * pk_prom) / np.sum(pk_prom) if num_peaks > 0 else np.nan

    nn = lambda x: np.arange(0, min(x, num_peaks))
    out['peakPower_2'] = np.sum(pk_height[nn(2)] * pk_width[nn(2)])
    out['peakPower_5'] = np.sum(pk_height[nn(5)] * pk_width[nn(5)])
    out['peakPower_prom5'] = np.sum(pk_height[pk_prom > 5] * pk_width[pk_prom > 5])
    out['w_weighted_peak_prom'] = np.sum(pk_loc * pk_prom) / np.sum(pk_prom) if num_peaks > 0 else np.nan
    out['w_weighted_peak_height'] = np.sum(pk_loc * pk_height) / np.sum(pk_height) if num_peaks > 0 else np.nan
    peak_power = pk_height * pk_width
    total_peak_power = np.sum(peak_power)
    if peak_power.size == 0:
        out['numPeaks_50power'] = np.nan
        out['peakpower_1'] = np.nan
    else:
        reached = np.where(np.cumsum(peak_power) > 0.5 * total_peak_power)[0]
        out['numPeaks_50power'] = reached[0] if reached.size > 0 else np.nan
        out['peakpower_1'] = peak_power[0] / total_peak_power if total_peak_power != 0 else np.nan

    iqr75 = np.quantile(s, 0.75, method='hazen')
    iqr25 = np.quantile(s, 0.25, method='hazen')
    out['iqr'] = iqr75 - iqr25
    out['logiqr'] = np.quantile(log_s, 0.75, method='hazen') - np.quantile(log_s, 0.25, method='hazen')
    out['q25'] = iqr25
    out['median'] = np.median(s)
    out['q75'] = iqr75

    out['std'] = np.std(s, ddof=1)
    out['stdlog'] = np.log(out['std'])
    out['logstd'] = np.std(log_s, ddof=1)
    out['mean'] = np.mean(s)
    out['logmean'] = np.mean(log_s)
    for i in range(3, 6):
        out[f'mom{i}'] = moments(s, i)

    auto_corrs_s = autocorr(s, [1, 2, 3, 4], 'Fourier')
    out['ac1'] = auto_corrs_s[0]
    out['ac2'] = auto_corrs_s[1]
    out['tau'] = first_crossing(s, 'ac', 0, 'continuous') * dw

    cs_s = np.cumsum(s)
    f_frac_w_max = lambda frac: w[np.where(cs_s >= cs_s[-1] * frac)[0][0]]
    out['wmax_5'] = f_frac_w_max(0.05)
    out['wmax_10'] = f_frac_w_max(0.1)
    out['wmax_25'] = f_frac_w_max(0.25)
    out['centroid'] = f_frac_w_max(0.5)
    out['wmax_75'] = f_frac_w_max(0.75)
    out['wmax_90'] = f_frac_w_max(0.9)
    out['wmax_95'] = f_frac_w_max(0.95)
    out['wmax_99'] = f_frac_w_max(0.99)

    out['w10_90'] = out['wmax_90'] - out['wmax_10']
    out['w25_75'] = out['wmax_75'] - out['wmax_25']

    a, b, c = np.polyfit(w, cs_s, deg=2)
    out['fpoly2csS_p1'] = a
    out['fpoly2csS_p2'] = b
    out['fpoly2csS_p3'] = c
    quad = lambda x, a, b, c: a * x**2 + b * x + c
    gof = goodness_of_fit(cs_s, quad(w, a, b, c), 3)
    out['fpoly2_sse'] = gof['sse']
    out['fpoly2_r2'] = gof['rsquare']
    out['fpoly2_rmse'] = gof['rmse']

    polysat = lambda p, x: (p[0] * (x**2)) / (p[1] + x**2)
    a, b = lsqcurvefit_trr(polysat, [cs_s[-1], 100], w, cs_s)
    out['fpolysat_a'] = a
    out['fpolysat_b'] = b
    gof = goodness_of_fit(cs_s, polysat([a, b], w), 2)
    out['fpolysat_r2'] = gof['rsquare']
    out['fpolysat_rmse'] = gof['rmse']

    h_shann = -s * np.log(s)
    out['spect_shann_ent'] = np.sum(h_shann)
    out['spect_shann_ent_norm'] = np.mean(h_shann)

    out['sfm'] = 10 * np.log10(np.exp(np.mean(np.log(s))) / np.mean(s))

    out['areatopeak'] = np.sum(s[0:np.argmax(s) + 1]) * dw
    out['ylogareatopeak'] = np.sum(log_s[0:np.argmax(s) + 1]) * dw

    r_all = w > 0
    across_full_range_res = give_me_robust_stats(np.log(w[r_all]), np.log(s[r_all]), 'linfitloglog_all')
    out = out | across_full_range_res
    r_lf = (w > 0)
    r_lf[int(np.floor(n/2)):] = 0
    first_half_res = give_me_robust_stats(np.log(w[r_lf]), np.log(s[r_lf]), 'linfitloglog_lf')
    out = out | first_half_res
    r_hf = np.arange(n // 2, n)
    second_half_res = give_me_robust_stats(np.log(w[r_hf]), np.log(s[r_hf]), 'linfitloglog_hf')
    out = out | second_half_res
    start = int(np.round(n / 4)) - 1
    stop = int(np.round(n * 3 / 4))
    r_mf = np.arange(start, stop)
    middle_half_res = give_me_robust_stats(np.log(w[r_mf]), np.log(s[r_mf]), 'linfitloglog_mf')
    out = out | middle_half_res
    res_semilog = give_me_robust_stats(w, np.log(s), 'linfitsemilog_all')
    out = out | res_semilog

    split = make_mat_buffer(s, int(np.floor(n / 2)))
    if split.shape[1] > 2:
        split = split[:, :2]
    out['area_2_1'] = np.sum(split[:, 0]) * dw
    out['logarea_2_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_2_2'] = np.sum(split[:, 1]) * dw
    out['logarea_2_2'] = np.sum(np.log(split[:, 1])) * dw
    out['statav2_m'] = np.std(np.mean(split, axis=0), ddof=1) / np.std(s, ddof=1)
    out['statav2_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)

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

    ncrossfn_rel = lambda frac: np.sum(sign_change(s - frac * np.max(s)))
    out['ncross_f05'] = ncrossfn_rel(0.05)
    out['ncross_f01'] = ncrossfn_rel(0.1)
    out['ncross_f02'] = ncrossfn_rel(0.2)
    out['ncross_f50'] = ncrossfn_rel(0.5)

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
    inf_peaks = np.where(np.isinf(s) & (s > 0))[0]

    if len(s) < 3:
        finite_peaks = np.array([], dtype=int)
    else:
        mid = s[1:-1]
        cond = np.isfinite(mid) & (mid > s[:-2]) & (mid > s[2:])
        finite_peaks = (np.flatnonzero(cond) + 1).astype(int)

    all_peaks = np.concatenate([finite_peaks, inf_peaks]) if len(inf_peaks) > 0 else finite_peaks
    all_peaks = np.sort(all_peaks)

    if len(all_peaks) == 0:
        return np.array([]), np.array([], dtype=int)

    if min_pk_dist > 0:
        peak_heights = s[all_peaks]
        sort_idx = np.argsort(peak_heights)[::-1]
        sorted_peaks = all_peaks[sort_idx]
        to_delete = np.zeros(len(sorted_peaks), dtype=bool)

        for i in range(len(sorted_peaks)):
            if not to_delete[i]:
                current_peak = sorted_peaks[i]
                for j in range(len(sorted_peaks)):
                    if not to_delete[j]:
                        distance = abs(sorted_peaks[j] - current_peak)
                        if distance <= min_pk_dist and distance > 0:
                            to_delete[j] = True

        final_peaks = sorted_peaks[~to_delete]
        back_to_original = np.searchsorted(all_peaks, final_peaks)
        final_peaks = all_peaks[np.sort(back_to_original)]
    else:
        final_peaks = all_peaks

    if len(final_peaks) == 0:
        return np.array([]), np.array([], dtype=int)

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

def give_me_robust_stats(x_data: ArrayLike, y_data: ArrayLike, field_name: str) -> dict:
    """
    Statistics based on a robust linear fit
    """
    out = {}
    try:
        a, stats = robustfit(x_data, y_data)
        out[f'{field_name}_a1'] = a[0]
        out[f'{field_name}_a2'] = a[1]
        out[f'{field_name}_sigrat'] = stats['ols_s'] / stats['robust_s']
        out[f'{field_name}_sigma'] = stats['s']
        out[f'{field_name}_sea1'] = stats['se'][0]
        out[f'{field_name}_sea2'] = stats['se'][1]
    except Exception:
        for key in ('a1', 'a2', 'sigrat', 'sigma', 'sea1', 'sea2'):
            out[f'{field_name}_{key}'] = np.nan
    return out
