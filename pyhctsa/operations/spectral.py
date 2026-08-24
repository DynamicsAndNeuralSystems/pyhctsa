import warnings
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import scipy.fft
import scipy.signal

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

def spectral_summaries_phase(y: ArrayLike) -> dict:
    """
    Statistics of the Fourier phase spectrum of a time series.

    cf. :func:`spectral_summaries`, which characterizes the *magnitude* spectrum in
    detail but discards phase entirely. For a linear, Gaussian stochastic process,
    Fourier phases are theoretically i.i.d. uniform on (-pi, pi] -- that's exactly why
    phase randomization works as a surrogate null model (cf. J. Theiler et al.,
    "Testing for nonlinearity in time series: the method of surrogate data",
    Physica D 58(1-4), 77 (1992)). This operation characterizes the phase spectrum
    directly: deviations from uniformity/independence across frequency are a direct
    signature of determinism, nonlinearity, or transient/localized structure that the
    magnitude spectrum alone cannot see.

    Phases are weighted by their bin's magnitude throughout (a standard approach in
    circular statistics for data of uneven reliability): a single pure tone
    concentrates essentially all energy in 1-2 bins, and every other bin's magnitude
    is set by numerical noise, so its "phase" is meaningless and must not be allowed
    to swamp an unweighted average. The DC and Nyquist bins (both purely real, phase
    undefined in the usual oscillatory sense) are excluded throughout.

    Parameters
    ----------
    y : array-like
        The input time series.

    Returns
    -------
    dict
        Statistics of the phase spectrum.
    """
    # Compute the FFT (same convention as spectral_summaries: Fs=1, NFFT a power of 2)
    y = np.asarray(y).ravel()
    ny = len(y)
    nfft = 2 ** int(np.ceil(np.log2(ny)))
    fs = 1
    f = fs / 2 * np.linspace(0, 1, nfft // 2 + 1)
    w = 2 * np.pi * f

    sc = scipy.fft.fft(y - np.mean(y), nfft)  # mean-subtracted, so the DC bin is (numerically) exactly zero
    sc = sc[:nfft // 2 + 1]  # single-sided
    mag = np.abs(sc)
    ph = np.angle(sc)

    # Exclude DC (bin 1) and Nyquist (last bin): both purely real, phase undefined
    # in the usual oscillatory sense.
    idx = slice(1, len(ph) - 1)
    ph = ph[idx]
    mag = mag[idx]
    ww = w[idx]

    if not np.any(mag > 0) or not np.all(np.isfinite(mag)):
        return np.nan

    wgt = mag / np.sum(mag)

    out = {}

    # Magnitude-weighted circular concentration
    r_vec = np.sum(wgt * np.exp(1j * ph))
    out['R'] = np.abs(r_vec)

    # Magnitude-weighted, normalized phase entropy (20-bin histogram)
    n_bins = 20
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(ph, edges)
    bin_idx = np.clip(bin_idx, 1, n_bins)  # guard the (rare) ph == pi edge case
    p_bin = np.bincount(bin_idx - 1, weights=wgt, minlength=n_bins)
    p_bin_nz = p_bin[p_bin > 0]
    out['phEnt'] = -np.sum(p_bin_nz * np.log(p_bin_nz)) / np.log(n_bins)

    # Group delay: magnitude-weighted linear fit of unwrapped phase vs frequency
    ph_unwrap = np.unwrap(ph)
    X = np.column_stack((np.ones(len(ww)), ww))
    XtW = X.T * wgt
    beta = np.linalg.solve(XtW @ X, XtW @ ph_unwrap)
    out['groupDelay'] = -beta[1]
    resid = ph_unwrap - X @ beta
    out['phaseLinearity'] = np.sqrt(np.sum(wgt * resid ** 2))

    # Magnitude-phase correlation
    out['magPhaseCorr'] = np.corrcoef(mag, ph)[0, 1]

    # Weighted lag-1 autocorrelation of unwrapped-phase increments across frequency
    d_phi = np.diff(ph_unwrap)
    d1 = d_phi[:-1]
    d2 = d_phi[1:]
    wgt3 = wgt[:-2]
    wgt3 = wgt3 / np.sum(wgt3)
    m1 = np.sum(wgt3 * d1)
    m2 = np.sum(wgt3 * d2)
    cov12 = np.sum(wgt3 * (d1 - m1) * (d2 - m2))
    v1 = np.sum(wgt3 * (d1 - m1) ** 2)
    v2 = np.sum(wgt3 * (d2 - m2) ** 2)
    out['phaseUnwrapAC1'] = cov12 / np.sqrt(v1 * v2)

    return out


def specparam(y: ArrayLike, aperiodic_mode: str = 'fixed', max_n_peaks: int = 4,
              peak_threshold: float = 1.0,
              peak_width_limits: ArrayLike = (0.02, 0.5),
              seg_length: Union[int, None] = None,
              max_segments: float = np.inf) -> Union[dict, float]:
    """
    Separates the power spectrum into aperiodic (1/f) and periodic (oscillatory)
    components.

    Parameterizes the power spectrum as a smooth aperiodic '1/f' background plus a
    small number of Gaussian peaks sitting on top of it, in the spirit of the
    FOOOF/specparam algorithm [1].

    References
    ----------
    .. [1] T. Donoghue et al., "Parameterizing neural power spectra into periodic and aperiodic components", Nat. Neurosci. 23: 1655 (2020)
    

    Parameters
    ----------
    y : array-like
        The input time series.
    aperiodic_mode : {'fixed', 'knee'}, optional
        The form of the aperiodic component:

        - 'fixed': ``b - chi*log10(f)``, a straight line in log-log, i.e. pure
          power-law.
        - 'knee': ``b - log10(k + f**chi)``, which additionally allows the spectrum to
          flatten off below a 'knee' frequency, as real spectra commonly do. Note the
          knee model is not identifiable when the data has no actual knee (k -> 0), so
          it falls back to the 'fixed' fit if the optimization fails or returns a
          degenerate knee.

        Default is ``'fixed'``.
    max_n_peaks : int, optional
        The maximum number of Gaussian peaks to extract. Default is 4.
    peak_threshold : float, optional
        How far above the noise a candidate peak must stand to be accepted (default 1).
        This is expressed as a multiple of the largest deviation that noise alone would
        be expected to produce -- specifically of ``sqrt(2*log(nBins))`` robust standard
        deviations of the flattened spectrum, ``nBins`` being the number of frequency
        bins searched. Expressing it that way (rather than as a plain multiple of sigma)
        is necessary because the test is applied to the maximum over all bins: a fixed
        small multiple of sigma fires on pure noise essentially always, and the
        correction adapts as ``nBins`` changes. So a value of 1 means 'must exceed what
        noise alone would give', and larger values are correspondingly more
        conservative.
    peak_width_limits : array-like, optional
        Two-element ``[min, max]`` on Gaussian peak standard deviation, in
        log10-frequency units (default ``(0.02, 0.5)``). Bounding the width both stops
        the optimizer fitting a single enormously wide 'peak' that is really leftover
        aperiodic background, and stops it fitting single-bin spectral noise.
    seg_length : int, optional
        Length of the Welch segments. ``None`` (default) adapts to the series length,
        as ``max(round(N/8), 32)``, so that longer series buy both finer frequency
        resolution and more segments to average over.
    max_segments : float, optional
        Maximum number of Welch segments to use. ``np.inf`` (default) uses all the
        available data.

    Returns
    -------
    dict
        The aperiodic parameters (``apExponent``, ``apOffset``, and for 'knee' mode
        ``apKnee``); the number of peaks found above threshold (``numPeaks``) and the
        centre frequency, height and bandwidth of the largest (``maxPeakFreq``,
        ``maxPeakPower``, ``maxPeakBW``); the total power in the periodic component
        (``totalPeakPower``) and the fraction of spectral power it accounts for
        (``periodicFraction``); and the quality of the combined fit (``modelR2`` and
        ``modelMAE``).
    """
    y = np.asarray(y, dtype=float).ravel()

    if aperiodic_mode not in ('fixed', 'knee'):
        raise ValueError(f"Unknown aperiodic_mode '{aperiodic_mode}' "
                         "(expected 'fixed' or 'knee')")
    peak_width_limits = np.asarray(peak_width_limits, dtype=float)

    N = len(y)
    if seg_length is None:
        # Scale the segment length with the series, so that longer series buy
        # both finer frequency resolution and more segments to average over.
        seg_length = max(int(np.floor(N / 8 + 0.5)), 32)
    min_segments = 4  # need several segments to average over for a usable estimate
    min_length = seg_length + (min_segments - 1) * (seg_length // 2)
    if N < min_length:
        warnings.warn(f"Time series (N = {N}) too short for a spectral parameterization "
                      f"with segLength = {seg_length} (need >= {min_length})")
        return np.nan
    if np.all(y == y[0]):
        warnings.warn("Constant time series has no spectral structure")
        return np.nan

    win_length = seg_length
    max_samples = win_length + (max_segments - 1) * (win_length // 2)
    if N > max_samples:
        y = y[:int(max_samples)]  # use a fixed amount of data so features stay comparable
    nfft = 2 ** int(np.ceil(np.log2(win_length)))
    f, s = scipy.signal.welch(y, fs=1, window=np.hamming(win_length),
                              noverlap=win_length // 2, nfft=nfft, detrend=False)

    # Restrict to the frequencies this estimate can actually resolve. A
    # frequency only completing one or two cycles within a Welch segment is
    # essentially unestimated, and on a log-frequency axis those lowest bins
    # sit isolated far to the left, where a straight-line fit is
    # unconstrained -- so any wiggle there reads as a huge 'peak'.
    min_cycles_per_segment = 5
    f_min = min_cycles_per_segment / win_length

    # Also exclude DC (log10(0) = -Inf) and any non-positive/non-finite power:
    valid = (f >= f_min) & (s > 0) & np.isfinite(s)
    if np.sum(valid) < 10:
        warnings.warn("Too few valid spectral points for a parameterization")
        return np.nan
    fv = f[valid]
    log_f = np.log10(fv)
    log_s = np.log10(s[valid])

    # Peaks bias this first fit; that is expected and is corrected by the
    # refit at the end, once the peaks have been identified and removed.
    ap0 = _fit_aperiodic(fv, log_f, log_s, aperiodic_mode)

    resid = log_s - ap0['pred']
    peak_list = []
    peak_sum = np.zeros(log_s.shape)

    n_bins = len(resid)
    # The acceptance threshold has to account for the fact that we are testing
    # the *maximum* over all nBins frequency bins, not one pre-specified bin:
    # the largest of nBins noise samples is expected to sit around
    # sqrt(2*log(nBins)) standard deviations up (~3.5 for a few hundred bins),
    # so any fixed small multiple of sigma fires on pure noise essentially
    # always.
    null_max_factor = np.sqrt(2 * np.log(n_bins))

    for _ in range(max_n_peaks):
        # Robust spread: the residual still contains the very peaks we are
        # looking for, and those positive outliers inflate a plain std --
        # which would make the test *less* sensitive exactly when there is
        # real structure. A MAD-based sigma is not pulled about by them.
        resid_sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if resid_sd <= 0:
            break
        i_pk = int(np.argmax(resid))
        pk_height = resid[i_pk]
        if pk_height < peak_threshold * null_max_factor * resid_sd:
            break  # nothing left standing above what noise alone would give

        g = _fit_gaussian(log_f, resid, i_pk, peak_width_limits)
        if g is None:
            break  # fit failed or returned a degenerate/out-of-bounds peak

        peak_list.append(g)
        peak_sum = peak_sum + g['pred']
        resid = resid - g['pred']  # peel this peak off and look for the next

    # This is the step that decouples the two components: with the oscillatory
    # peaks subtracted, the background fit is no longer dragged by them, so
    # the exponent estimates the true 1/f background rather than a blend of
    # background and oscillations.
    ap_final = _fit_aperiodic(fv, log_f, log_s - peak_sum, aperiodic_mode)

    out = {}
    out['apExponent'] = ap_final['exponent']
    # Report the background level at a reference frequency *inside* the fitted
    # band, rather than the raw intercept. The intercept is the fitted value at
    # log10(f) = 0, i.e. f = 1 -- above the Nyquist frequency of 0.5, so it is
    # a pure extrapolation whose value swings with both the fitted slope and
    # wherever the fitted range happens to start.
    ref_freq = 0.1
    out['apOffset'] = _eval_aperiodic(ap_final, ref_freq)
    if aperiodic_mode == 'knee':
        out['apKnee'] = ap_final['knee']

    out['numPeaks'] = len(peak_list)
    if len(peak_list) == 0:
        out['maxPeakFreq'] = np.nan
        out['maxPeakPower'] = np.nan
        out['maxPeakBW'] = np.nan
        out['totalPeakPower'] = 0
    else:
        heights = np.array([p['height'] for p in peak_list])
        i_max = int(np.argmax(heights))
        out['maxPeakFreq'] = 10 ** peak_list[i_max]['centre']  # back to linear frequency
        out['maxPeakPower'] = heights[i_max]  # height above the aperiodic background, in log10 power
        out['maxPeakBW'] = peak_list[i_max]['width']
        out['totalPeakPower'] = np.sum(heights)

    # Share of the (log-)spectrum's variation accounted for by the periodic
    # component, rather than by the aperiodic background:
    total_var = np.sum((log_s - np.mean(log_s)) ** 2)
    if total_var > 0:
        out['periodicFraction'] = np.sum(peak_sum ** 2) / total_var
    else:
        out['periodicFraction'] = np.nan

    model = ap_final['pred'] + peak_sum
    resid_final = log_s - model
    if total_var > 0:
        out['modelR2'] = 1 - np.sum(resid_final ** 2) / total_var
    else:
        out['modelR2'] = np.nan
    out['modelMAE'] = np.mean(np.abs(resid_final))

    return out


def _eval_aperiodic(ap: dict, fq: float) -> float:
    # Value of the fitted aperiodic curve at frequency fq.
    if 'knee' in ap and ap['knee'] > 0:
        return ap['offset'] - np.log10(ap['knee'] + fq ** ap['exponent'])
    return ap['offset'] - ap['exponent'] * np.log10(fq)


def _fit_aperiodic(fv: ArrayLike, log_f: ArrayLike, log_s: ArrayLike,
                   aperiodic_mode: str) -> dict:
    # Fit the smooth aperiodic background of the log10 spectrum.
    # 'fixed': logS = offset - exponent*log10(f)
    # 'knee' : logS = offset - log10(knee + f^exponent)

    # Robust straight-line fit in log-log, used directly for 'fixed' mode
    # and as the starting point for the nonlinear 'knee' fit:
    b, _ = robustfit(log_f, log_s)
    ap = {
        'offset': b[0],
        'exponent': -b[1],  # conventionally reported as a positive falling slope
        'pred': b[0] + b[1] * log_f,
    }

    if aperiodic_mode == 'fixed':
        return ap

    # 'knee' mode: a nonlinear fit, seeded from the linear one. The knee
    # is only identifiable when the spectrum actually flattens at low
    # frequency; when it does not, the optimizer drives knee -> 0 (or
    # fails outright), in which case the model degenerates to the 'fixed'
    # form and we keep the robust linear fit rather than a bogus knee.
    try:
        # fittype('a - log10(k + x^c)') names its coefficients in alphabetical
        # order, so the fitted vector is [a, c, k]:
        knee_model = lambda p, x: p[0] - np.log10(p[2] + x ** p[1])
        p = lsqcurvefit_trr(knee_model,
                            [ap['offset'], 1e-3, max(ap['exponent'], 0.1)],
                            fv, log_s,
                            lower=[-np.inf, 0, 0], upper=[np.inf, np.inf, 10],
                            max_iter=400)
        knee_val = p[2]
        pred_knee = p[0] - np.log10(knee_val + fv ** p[1])
        if np.isfinite(knee_val) and knee_val > 1e-10 and np.all(np.isfinite(pred_knee)):
            ap['offset'] = p[0]
            ap['knee'] = knee_val
            ap['exponent'] = p[1]
            ap['pred'] = pred_knee
        else:
            ap['knee'] = 0  # degenerate: no detectable knee, keep the linear fit
    except Exception:
        ap['knee'] = 0  # optimization failed: fall back to the linear fit

    return ap


def _fit_gaussian(log_f: ArrayLike, resid: ArrayLike, i_pk: int,
                  peak_width_limits: ArrayLike) -> Union[dict, None]:
    # Fit a single Gaussian to the flattened spectrum, centred near the
    # current maximum at index iPk. Returns None if the fit fails or lands
    # outside the permitted width range.
    x0 = log_f[i_pk]
    h0 = resid[i_pk]
    if not np.isfinite(h0) or h0 <= 0:
        return None
    w0 = np.mean(peak_width_limits)

    try:
        gauss_model = lambda p, x: p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
        p = lsqcurvefit_trr(gauss_model, [h0, x0, w0], log_f, resid,
                            lower=[0, np.min(log_f), peak_width_limits[0]],
                            upper=[np.inf, np.max(log_f), peak_width_limits[1]],
                            max_iter=400)
    except Exception:
        return None

    h, m, w = p[0], p[1], p[2]
    if not np.isfinite(h) or not np.isfinite(m) or not np.isfinite(w) or h <= 0:
        return None
    return {
        'height': h,
        'centre': m,
        'width': w,
        'pred': h * np.exp(-(log_f - m) ** 2 / (2 * w ** 2)),
    }
