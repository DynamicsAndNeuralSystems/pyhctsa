import warnings
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import scipy.fft

from ..toolboxes.matlab.matlab_fit import lsqcurvefit_trr, goodness_of_fit, robustfit, polyfit

from ..operations.correlation import autocorr, first_crossing
from ..operations.distribution import moments
from ..utils import make_mat_buffer, sign_change

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
        be expected to produce.
    peak_width_limits : array-like, optional
        Two-element ``[min, max]`` on Gaussian peak standard deviation, in
        log10-frequency units (default ``(0.02, 0.5)``).
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
        win = np.ones(ny) if (window is None or len(window) == 0) else np.asarray(window)
        nfft = max(256, 2 ** int(np.ceil(np.log2(ny))))
        f, s = scipy.signal.periodogram(
            y, fs=1, window=win, nfft=nfft, detrend=False,
            return_onesided=True, scaling='density'
        )
        w = 2 * np.pi * f  # angular frequency (rad/sample)
        s = s / (2 * np.pi)  # normalized in angular frequency space
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
    gof = goodness_of_fit(cs_s, quad(w, a, b, c), 3)
    out['fpoly2_sse'] = gof['sse']
    out['fpoly2_r2'] = gof['rsquare']
    out['fpoly2_rmse'] = gof['rmse']

    # Fit polysat a*x^2/(b+x^2) (has zero derivative at zero, though)
    polysat = lambda p, x: (p[0] * (x**2)) / (p[1] + x**2)
    a, b = lsqcurvefit_trr(polysat, [cs_s[-1], 100], w, cs_s)
    out['fpolysat_a'] = a
    out['fpolysat_b'] = b
    gof = goodness_of_fit(cs_s, polysat([a, b], w), 2)
    out['fpolysat_r2'] = gof['rsquare']
    out['fpolysat_rmse'] = gof['rmse']

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
    # Vectorised: a finite interior point strictly greater than both neighbours.
    # np.isfinite excludes +/-inf and nan, matching `not isinf and not isnan`.
    if len(s) < 3:
        finite_peaks = np.array([], dtype=int)
    else:
        mid = s[1:-1]
        cond = np.isfinite(mid) & (mid > s[:-2]) & (mid > s[2:])
        finite_peaks = (np.flatnonzero(cond) + 1).astype(int)

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

        # convert back to original indices for sorting. all_peaks is sorted-ascending
        # and unique, so positions come from one searchsorted instead of an O(p^2) scan.
        back_to_original = np.searchsorted(all_peaks, final_peaks)
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

def give_me_robust_stats(x_data: ArrayLike, y_data: ArrayLike, field_name: str) -> dict:
    """
    Statistics based on a robust linear fit
    """
    out = {}
    try:
        a, stats = robustfit(x_data, y_data)
        out[f'{field_name}_a1'] = a[0]  # robust intercept
        out[f'{field_name}_a2'] = a[1]  # robust gradient
        # ratio of sigma estimates between ordinary least squares and the robust fit:
        out[f'{field_name}_sigrat'] = stats['ols_s'] / stats['robust_s']
        # sigma as the larger of robust_s and a weighted average of ols_s and robust_s:
        out[f'{field_name}_sigma'] = stats['s']
        out[f'{field_name}_sea1'] = stats['se'][0]  # standard error in intercept
        out[f'{field_name}_sea2'] = stats['se'][1]  # standard error in slope
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

def cepstrum(y: ArrayLike, max_period: int = 100, min_period: int = 4) -> dict:
    """
    Cepstral statistics: harmonic (comb) structure of the power spectrum.

    Computes the real cepstrum, the inverse Fourier transform of the log magnitude
    spectrum, and summarizes the structure of its dominant peak.

    Parameters
    ----------
    y : array-like
        The input time series.
    max_period : int, optional
        The longest fundamental period (in samples) to search for. Default is
        100.
    min_period : int, optional
        The shortest fundamental period (in samples) to search for. Default is 4.

    Returns
    -------
    dict
        - ``period``: the estimated fundamental period (quefrency of the dominant
          cepstral peak), in samples.
        - ``peak``: the height of that peak.
        - ``meanCeps``, ``stdCeps``: the mean and standard deviation of the cepstrum
          over the search range.
        - ``peakRatio``: the peak height in units of the standard deviation of the
          cepstrum over the search range.
        - ``CPP``: the cepstral peak prominence, the standard robust measure, being
          the peak height above a linear regression fit through the cepstrum across
          the search range (this normalizes away the overall cepstral trend, so it
          does not simply track the spectrum's dynamic range).
        - ``rahmonicRatio``: comparing the cepstrum at twice the peak quefrency to
          the peak itself (a genuine harmonic comb repeats at multiples of the
          fundamental period, so a real periodicity shows a secondary 'rahmonic'
          peak, whereas an isolated fluke does not).

        Returns NaN if the time series is too short for the requested search range,
        or is constant.
    """
    y = np.asarray(y, dtype=float).ravel()

    max_period, min_period = int(max_period), int(min_period)
    if min_period < 2:
        raise ValueError(f"min_period = {min_period} is below the Nyquist limit "
                         f"(a period needs >= 2 samples)")
    if max_period <= min_period:
        raise ValueError(f"max_period ({max_period}) must exceed min_period ({min_period})")

    N = len(y)
    minCycles = 4 # need several cycles of the longest period searched for a meaningful estimate
    if N < minCycles * max_period:
        warnings.warn(f"Time series (N = {N}) too short to search for periods up to "
                      f"{max_period} samples (need >= {minCycles * max_period})")
        return np.nan

    if np.all(y == y[0]): # constant series has an all-zero spectrum -> log(0)
        warnings.warn("Constant time series has no spectral (or cepstral) structure")
        return np.nan

    # ------------------------------------------------------------------------------
    # Real cepstrum
    # ------------------------------------------------------------------------------
    NFFT = 2 ** int(np.ceil(np.log2(N)))
    X = scipy.fft.fft(y, NFFT)
    logMag = np.log(np.abs(X) + np.finfo(float).eps) # eps guards spectral nulls (|X| exactly 0)

    envOrder = 4
    nHalf = NFFT // 2 + 1
    halfLogMag = logMag[:nHalf]
    fIdx = np.arange(nHalf, dtype=float) / (nHalf - 1) # normalized frequency axis for conditioning
    pEnv = polyfit(fIdx, halfLogMag, envOrder)
    halfDetrended = halfLogMag - np.polyval(pEnv, fIdx)

    # Mirror back to a full Hermitian-symmetric spectrum so the cepstrum is real:
    logMagDetrended = np.concatenate((halfDetrended, halfDetrended[1:-1][::-1]))
    c = np.real(scipy.fft.ifft(logMagDetrended))

    # Quefrency index q corresponds to a period of q samples:
    periods = np.arange(min_period, max_period + 1)
    if periods[-1] + 1 > NFFT // 2:
        # Shouldn't be reachable given the length check above, but the cepstrum is
        # only meaningful over its first half (it is symmetric):
        periods = periods[periods + 1 <= NFFT // 2]
    cSearch = c[periods]

    iPeak = int(np.argmax(cSearch))
    peakVal = cSearch[iPeak]

    out = {}
    out['period'] = float(periods[iPeak]) # estimated fundamental period, in samples
    out['peak'] = peakVal

    # Basic distributional context over the search range:
    out['meanCeps'] = np.mean(cSearch)
    out['stdCeps'] = np.std(cSearch, ddof=1)

    # Peak height in units of the cepstrum's own spread over the search range
    # (scale-free, unlike `peak` itself):
    if out['stdCeps'] > 0:
        out['peakRatio'] = (peakVal - out['meanCeps']) / out['stdCeps']
    else:
        out['peakRatio'] = np.nan

    pFit = polyfit(periods.astype(float), cSearch, 1)
    baseline = np.polyval(pFit, periods)
    out['CPP'] = peakVal - baseline[iPeak]

    q2 = 2 * periods[iPeak] + 1
    peakAboveBase = peakVal - baseline[iPeak]
    if q2 <= NFFT // 2 and peakAboveBase > 0:
        rahmonicAboveBase = c[q2 - 1] - np.polyval(pFit, 2 * periods[iPeak])
        out['rahmonicRatio'] = rahmonicAboveBase / peakAboveBase
    else:
        out['rahmonicRatio'] = np.nan

    return out

