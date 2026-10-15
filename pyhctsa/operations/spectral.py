import warnings
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import scipy.fft
import scipy.signal
import scipy.optimize
import scipy.stats
import scipy.special

from ..fitting import goodness_of_fit, robust_linear_fit

from ..operations.correlation import autocorr, first_crossing
from ..operations.distribution import moments
from ..robust import bf_fit_sinusoids, bf_residual_stats
from ..utils import _even_split_points, dict_output, sign_change

# Smallest FFT length of the zero-padded spectral estimates ('periodogram' and 'welch' in
# spectral_summaries): the spectrum is evaluated on max(256, next power of two >= segment length)
# frequencies, an interpolated grid that is finer than the DFT grid of short segments. The
# peak-prominence thresholds of spectral_summaries are calibrated on this grid (a pyhctsa choice).
_MIN_NFFT = 256


def _padded_nfft(length: int) -> int:
    """FFT length of the zero-padded spectral estimates: ``max(256, 2**ceil(log2(length)))``."""
    return max(_MIN_NFFT, 2 ** int(np.ceil(np.log2(length))))


@dict_output
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

def _bounded_lsq(fun, p0, x, y, lower, upper) -> np.ndarray:
    # Bounded nonlinear least squares, min sum((fun(p, x) - y)^2), by the trust-region-reflective
    # algorithm of scipy.optimize.least_squares with its default tolerances. The start point is
    # clipped into the bounds.
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    p0 = np.clip(np.asarray(p0, dtype=float), lower, upper)
    sol = scipy.optimize.least_squares(lambda p: fun(p, x) - y, p0, bounds=(lower, upper),
                                       method='trf')
    return sol.x


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
    b, _ = robust_linear_fit(log_f, log_s)
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
        # Coefficients ordered [a, k, c] (hctsa names the order explicitly, since
        # fittype('a - log10(k + x^c)') would order them alphabetically as
        # [a, c, k]); the start point and bounds below are in this order:
        knee_model = lambda p, x: p[0] - np.log10(p[1] + x ** p[2])
        p = _bounded_lsq(knee_model,
                         [ap['offset'], 1e-3, max(ap['exponent'], 0.1)],
                         fv, log_s,
                         lower=[-np.inf, 0, 0], upper=[np.inf, np.inf, 10])
        knee_val = p[1]
        pred_knee = p[0] - np.log10(knee_val + fv ** p[2])
        if np.isfinite(knee_val) and knee_val > 1e-10 and np.all(np.isfinite(pred_knee)):
            ap['offset'] = p[0]
            ap['knee'] = knee_val
            ap['exponent'] = p[2]
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
        p = _bounded_lsq(gauss_model, [h0, x0, w0], log_f, resid,
                         lower=[0, np.min(log_f), peak_width_limits[0]],
                         upper=[np.inf, np.max(log_f), peak_width_limits[1]])
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

def _polysat_fit(w: np.ndarray, y: np.ndarray) -> dict:
    """Global least-squares fit of ``a*w**2/(b + w**2)`` with ``b`` in ``[1e-6, 100] * max(w)**2``.

    For a given ``b`` the best ``a`` is linear least squares in closed form, so only ``b`` is
    searched: on 200 log-spaced values, then refined by 40 golden-section steps in ``log(b)``
    around the best of these. ``a`` and ``b`` are only separately determined when the curve
    bends over within the range of ``w``; for a curve that keeps rising like ``w**2`` the best
    ``b`` is the upper limit (where only ``a/b`` matters). Returns ``a``, ``b``, ``r2``
    (``1 - SSE/SST``) and ``rmse`` (``sqrt(SSE/(n - 2))``); NaN if ``y`` is constant.
    """
    w = np.asarray(w, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 3 or not np.all(np.isfinite(y)) or not np.ptp(y) > 1e-12 * np.max(np.abs(y)):
        return dict(a=np.nan, b=np.nan, r2=np.nan, rmse=np.nan)
    w2 = w ** 2
    scale = np.max(w2)

    def sse(lb):
        g = w2[:, None] / (np.exp(np.atleast_1d(lb))[None, :] + w2[:, None])
        return np.sum(y ** 2) - (g.T @ y) ** 2 / np.sum(g ** 2, axis=0)

    u = np.linspace(np.log(1e-6 * scale), np.log(1e2 * scale), 200)
    k = int(np.argmin(sse(u)))
    lo, hi = u[max(k - 1, 0)], u[min(k + 1, u.size - 1)]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):  # golden-section search in log(b)
        u1 = hi - phi * (hi - lo)
        u2 = lo + phi * (hi - lo)
        if sse(u1)[0] <= sse(u2)[0]:
            hi = u2
        else:
            lo = u1
    b = np.exp((lo + hi) / 2)
    g = w2 / (b + w2)
    a = np.dot(g, y) / np.dot(g, g)
    s = np.sum((y - a * g) ** 2)
    return dict(a=a, b=b, r2=1 - s / np.sum((y - np.mean(y)) ** 2), rmse=np.sqrt(s / (n - 2)))


@dict_output
def spectral_summaries(y: ArrayLike, psd_meth: str = 'fft', window_type: str = 'hamming') -> dict:
    """
    Statistics of the power spectrum of a time series.

    Estimates the power spectrum (as a periodogram, a plain fast Fourier transform, or by
    Welch's method) and returns many summary statistics: the location and width of its main
    peak, the number and prominence of peaks, the distribution of power values, the
    autocorrelation of power across frequency, the frequencies below which given fractions of
    the power lie, power-weighted moments of frequency, fits to the cumulative power, a spectral
    entropy and flatness, robust power-law fits to the log-log spectrum, the power in 2 and 5
    equal frequency bands, and the number of crossings of the log spectrum at various levels.
    Many statistics have a log-domain version computed on log(S). The spectrum is floored at
    1e-12 of its maximum before any statistic is computed, so that bins at the rounding level
    of the estimator do not determine the log-domain statistics.

    Parameters
    ----------
    y : array-like
        The input time series.
    psd_meth : {'periodogram', 'fft', 'welch'}, optional
        The method for obtaining the spectrum from the signal:

        - 'periodogram': periodogram (window across the whole series)
        - 'fft': fast Fourier transform, single-sided power spectral density; no window is
          applied and the DC bin is dropped
        - 'welch': Welch's method, with windows of ``max(min(256, round(N/4)), 16)`` samples,
          50% overlap, no detrending, and each segment zero-padded to
          ``nfft = max(256, next power of 2 >= window length)`` (also used by 'periodogram',
          with the window length ``N``)

        Default is ``'fft'``.

    window_type : {'boxcar', 'rect', 'bartlett', 'hann', 'hamming', 'none'}, optional
        The window to use (for 'periodogram' and 'welch'; ignored by 'fft'). Default is
        ``'hamming'``.

    Returns
    -------
    dict
        The spectrum S is a power spectral density in angular frequency w (radians per sample,
        0 to pi), normalized for all three estimators so that its area (the sum of S times the
        bin spacing dw) equals the variance of the series (~1 for a z-scored series).
        Statistics that accumulate over bins (cumulative-area fits, entropy) are therefore
        integrals over w, independent of the number of bins. Returns NaN for a constant series.
        Output fields:

        Peaks
            - ``maxS``, ``maxw``: the maximum of S and the angular frequency at which it occurs.
            - ``maxWidth``: half-power bandwidth of the dominant peak.
            - ``numPromPeaks_3``, ``numPromPeaks_5``, ``numPromPeaks_8``: number of peaks of
              log(S) with prominence above 3, 5, 8 (natural-log units).
            - ``meanProm_5``, ``meanPeakWidth_prom5``: mean prominence and mean width of the
              peaks with prominence above 5.
            - ``width_weighted_prom``, ``w_weighted_peak_prom``: mean peak width and mean peak
              location, weighted by prominence.
            - ``peakPower_2``, ``peakPower_5``, ``peakPower_prom5``: power (height x width) in
              the 2 and 5 tallest peaks and in the peaks with prominence above 5.
            - ``numPeaks_50power``, ``peakpower_1``: number of tallest peaks needed to hold
              half the peak power, and the fraction of peak power in the tallest peak.

            Peaks are found on log(S), and the prominence thresholds are calibrated for (and
            hctsa registers these fields only with) ``psd_meth='welch'``.

        Distribution of power values (and of log power values, with prefix ``log``)
            ``iqr``, ``logiqr``, ``q25``, ``median``, ``q75``, ``logq25``, ``logmedian``,
            ``logq75``, ``std``, ``stdlog`` (log of the std), ``logstd`` (std of log(S)),
            ``mom3`` and ``logmom3`` (skewness).

        Autocorrelation of the spectrum across frequency
            ``ac1``, ``ac2``, ``tau`` (first zero-crossing of the autocorrelation, in units of
            w), and ``logac1``, ``logac2``, ``logtau`` for log(S).

        Cumulative power
            ``wq_5``, ``wq_10``, ``wq_25``, ``wq_50``, ``wq_75``, ``wq_90``,
            ``wq_95``, ``wq_99``: the frequency below which 5%, 10%, ..., 99% of the power
            lies (``wq_50`` is the median frequency).

        Power-weighted moments of frequency
            ``specCentroid``, ``specSpread``, ``specSkew``, ``specKurt``: mean, standard
            deviation, skewness and kurtosis of frequency weighted by power.

        Fits to the cumulative area under S (a running integral over w)
            ``fpoly2csS_p1``, ``fpoly2csS_p2``, ``fpoly2csS_p3``, ``fpoly2_sse``,
            ``fpoly2_r2``, ``fpoly2_rmse``: coefficients and goodness of a quadratic fit
            (``fpoly2_sse`` is the integrated squared error); ``fpolysat_a``, ``fpolysat_b``,
            ``fpolysat_r2``, ``fpolysat_rmse``: parameters and goodness of the least-squares fit of
            ``a*w**2/(b+w**2)``, the global optimum over ``1e-6 <= b/max(w)**2 <= 100`` (see
            ``_polysat_fit``; a curve that keeps rising like ``w**2`` gives the largest ``b``).

        Entropy, flatness and areas
            ``spect_shann_ent`` (differential Shannon entropy of the power distribution over
            frequency), ``spect_shann_ent_norm`` (``exp(spect_shann_ent)`` as a fraction of the
            frequency range), ``sfm`` (spectral flatness measure, 10*log10(geometric mean /
            arithmetic mean)), ``areatopeak`` and ``ylogareatopeak`` (area under S, and under
            log(S), up to the peak).

        Robust linear fits (``a1`` intercept, ``a2`` gradient, ``sigrat`` OLS/robust sigma ratio, ``sigma``, ``sea1`` standard error of the intercept)
            ``linfitloglog_all_{a1,a2,sigrat,sigma,sea1}`` (log(S) against log(w), all
            frequencies), ``linfitloglog_lf_a2`` and ``linfitloglog_mf_a2`` (lower half and
            middle half), ``linfitloglog_hf_{a1,a2,sigrat,sigma,sea1}`` (upper half) and
            ``linfitsemilog_all_{a1,sigrat,sigma,sea1}`` (log(S) against w).

        Power in frequency bands (2 and 5 equal bands, from the lowest frequencies)
            ``area_2_1``, ``area_2_2``, ``logarea_2_1``, ``logarea_2_2`` and ``area_5_1`` ...
            ``area_5_5``, ``logarea_5_1`` ... ``logarea_5_5`` (area under S and log(S) in each
            band); ``statav2_s``, ``statav5_s`` (std across bands of the within-band std of S,
            relative to std(S)); ``logstatav2_m``, ``logstatav2_s``, ``logstatav5_m``,
            ``logstatav5_s`` (the same for the band means and stds of log(S), relative to
            std(log(S))). When the number of bins is not divisible by the number of bands the
            last few bins are dropped.

        Crossings of the log spectrum
            ``ncross_log_f05``, ``ncross_log_f10``, ``ncross_log_f20``, ``ncross_log_f50``:
            crossings of a level 5%, 10%, 20%, 50% of the way from min(log S) to max(log S).
    """

    y = np.asarray(y)
    ny = len(y)

    if np.all(y == y[0]):  # constant series has an all-zero spectrum -> log(0)
        warnings.warn("Constant time series has no spectral structure")
        return np.nan

    # Set window (for periodogram and welch):
    if psd_meth == 'welch':
        # Welch's method needs a window shorter than the series, so that segments are averaged.
        # The segment length is fixed in samples (not a fraction of N), so that the frequency
        # resolution stays fixed and more segments are averaged as N grows. 50% overlap.
        win_length = max(min(256, int(np.floor(ny / 4 + 0.5))), 16)  # round half away from zero
    else:
        win_length = ny
    if window_type in ('none', 'rect', 'boxcar'):
        window = np.ones(win_length)  # no window
    elif window_type == 'hamming':
        window = np.hamming(win_length)
    elif window_type == 'hann':
        window = np.hanning(win_length)
    elif window_type == 'bartlett':
        window = np.bartlett(win_length)
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
        # Drop the DC bin (w = 0). A z-scored series has a zero sum up to rounding, so s[0] is
        # not a spectral estimate but the square of a ~1e-14 residual, and log(s[0]) is a
        # random outlier near -70 that would dominate every log-domain statistic. (The
        # windowed/Welch estimates have a genuine non-zero DC bin from leakage.)
        w = w[1:]
        s = s[1:]

    elif psd_meth == 'welch':
        # Welch power spectral density estimate (Welch 1967): 50% overlap, no detrending, segments
        # zero-padded to nfft = max(256, next power of 2 >= segment length)
        fs = 1
        n_overlap = win_length // 2
        nfft = _padded_nfft(win_length)
        f, s = scipy.signal.welch(y, fs=fs, window=window, nperseg=win_length, noverlap=n_overlap,
                                  nfft=nfft, detrend=False, return_onesided=True, scaling='density')
        w = 2 * np.pi * f  # angular frequency
        s = s / (2 * np.pi)  # adjust so that area remains normalized in angular frequency space
    elif psd_meth == 'periodogram':
        win = window
        nfft = _padded_nfft(ny)
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

    # Floor the spectrum at 1e-12 of its maximum (120 dB below the peak) before taking logs: bins
    # at the rounding level of the estimator (e.g., for a periodic signal that fits the transform
    # length, or a ramp) are otherwise arbitrary values of order 1e-30 or exactly zero, and then
    # set every log-domain statistic. The spectral dynamic range of real-world series is well
    # above this floor.
    s = np.maximum(s, 1e-12 * np.nanmax(s))

    n = len(s)
    log_s = np.log(s)
    dw = w[1] - w[0]  # spacing increment in w

    # Simple measures of the power spectrum
    # Peaks
    out = {}
    i_max_s = np.argmax(s)
    out = {'maxS': s[i_max_s], 'maxw': w[i_max_s]}

    # Half-power (-3 dB) bandwidth of the dominant peak: the frequency interval around the
    # maximum over which the spectrum stays above half its peak value.
    half_power = out['maxS'] / 2
    r = np.flatnonzero(s[i_max_s + 1:] < half_power)
    i_upper = i_max_s + 1 + r[0] if r.size else n - 1  # never drops below half power above the peak
    l = np.flatnonzero(s[:i_max_s] < half_power)
    i_lower = l[-1] if l.size else 0  # never drops below half power below the peak
    out['maxWidth'] = w[i_upper] - w[i_lower]

    # Characterize all peaks, run on log(S) rather than S: a linear-scale power spectrum is
    # heavy-tailed (a single dominant peak can be >100x the mean level), so prominence-based
    # peak detection on raw S buries smaller-but-genuine peaks under the dominant one, and the
    # fixed prominence thresholds below are only meaningfully calibrated on the log scale.
    # These thresholds are calibrated for a Welch-smoothed spectrum (psd_meth='welch'): for
    # 'fft' and 'periodogram' the peak-derived fields are computed but not calibrated.
    min_dist_w = 0.02  # minimum peak separation, in angular frequency
    min_pk_dist = np.ceil(min_dist_w / dw)  # ... in bins
    pk_height, pk_loc = _findpeaks(log_s, min_pk_dist, 'descend')  # tallest first; pk_loc are bin indices
    pk_width = scipy.signal.peak_widths(log_s, pk_loc)[0]
    pk_prom = scipy.signal.peak_prominences(log_s, pk_loc)[0]
    # Linear-domain height of each peak, for the 'power in peaks' fields: detection and
    # prominence use log(S), but height x width only means power on the linear spectrum
    pk_height_lin = s[pk_loc]
    pk_width = pk_width * dw  # peak widths in angular frequency
    pk_loc = w[pk_loc]  # angular frequency of each peak

    # Characterize peak prominence (thresholds in log-power units, calibrated on a Welch null)
    num_peaks = len(pk_height)  # local only: needed for the peakPower_* fields below, not itself an output
    with np.errstate(invalid='ignore', divide='ignore'):
        out['numPromPeaks_3'] = np.sum(pk_prom > 3)  # number of peaks with log-prominence of at least 3
        out['numPromPeaks_5'] = np.sum(pk_prom > 5)  # ... at least 5
        out['numPromPeaks_8'] = np.sum(pk_prom > 8)  # ... at least 8
        # mean peak prominence of those with log-prominence of at least 5
        out['meanProm_5'] = _mean_or_nan(pk_prom[pk_prom > 5])
        out['meanPeakWidth_prom5'] = _mean_or_nan(pk_width[pk_prom > 5])
        out['width_weighted_prom'] = np.sum(pk_width * pk_prom) / np.sum(pk_prom)

        # Power in top N peaks
        nn = lambda x: np.arange(min(x, num_peaks))
        out['peakPower_2'] = np.sum(pk_height_lin[nn(2)] * pk_width[nn(2)])
        out['peakPower_5'] = np.sum(pk_height_lin[nn(5)] * pk_width[nn(5)])
        # power in peaks with log-prominence of at least 5
        out['peakPower_prom5'] = np.sum(pk_height_lin[pk_prom > 5] * pk_width[pk_prom > 5])
        # where are prominent peaks located on average (weighted by prominence)
        out['w_weighted_peak_prom'] = np.sum(pk_loc * pk_prom) / np.sum(pk_prom)

    # Number of peaks required to get to 50% of power in peaks
    peak_power = pk_height_lin * pk_width
    if peak_power.size == 0:  # no peaks found (e.g., a monotonic spectrum)
        out['numPeaks_50power'] = np.nan
        out['peakpower_1'] = np.nan
    else:
        half_idx = np.flatnonzero(np.cumsum(peak_power) > 0.5 * np.sum(peak_power))
        # a count (one-based); NaN if the peak powers are not finite
        out['numPeaks_50power'] = half_idx[0] + 1 if half_idx.size else np.nan
        out['peakpower_1'] = peak_power[0] / np.sum(peak_power)

    # Distribution
    # quantiles
    q25_s, q75_s = np.quantile(s, [0.25, 0.75], method='hazen')
    q25_log, q75_log = np.quantile(log_s, [0.25, 0.75], method='hazen')
    out['iqr'] = q75_s - q25_s
    out['logiqr'] = q75_log - q25_log
    out['q25'] = q25_s
    out['median'] = np.median(s)
    out['q75'] = q75_s
    # log-domain companions (the linear spectrum is heavy-tailed, so these capture different information)
    out['logq25'] = q25_log
    out['logmedian'] = np.median(log_s)
    out['logq75'] = q75_log

    # Moments (the standardized third moment, i.e. the skewness, of the power values)
    out['std'] = np.std(s, ddof=1)
    out['stdlog'] = np.log(out['std'])
    out['logstd'] = np.std(log_s, ddof=1)
    out['mom3'] = moments(s, 3, True)
    out['logmom3'] = moments(log_s, 3, True)

    # Autocorrelation of amplitude spectrum:
    auto_corrs_s = autocorr(s, [1, 2, 3, 4], 'Fourier')
    out['ac1'] = auto_corrs_s[0]
    out['ac2'] = auto_corrs_s[1]
    out['tau'] = first_crossing(s, 'ac', 0, 'continuous') * dw  # first zero crossing, in units of w (not bins)
    # The same for log(S): the autocorrelation of the heavy-tailed linear spectrum is dominated by the
    # distance of its single largest value from the rest, which log(S) compresses
    auto_corrs_log_s = autocorr(log_s, [1, 2, 3, 4], 'Fourier')
    out['logac1'] = auto_corrs_log_s[0]
    out['logac2'] = auto_corrs_log_s[1]
    out['logtau'] = first_crossing(log_s, 'ac', 0, 'continuous') * dw

    # Shape of cumulative sum curve: the cumulative area under the spectrum (a running
    # integral over w, not a bare running sum over bins), which rises to ~1 for a unit-variance
    # series whatever the number of bins
    cs_s = np.cumsum(s) * dw
    f_quantile_w = lambda frac: w[np.where(cs_s >= cs_s[-1] * frac)[0][0]]
    # @ what frequency is csS a fraction p of its maximum?
    out['wq_5'] = f_quantile_w(0.05)
    out['wq_10'] = f_quantile_w(0.1)
    out['wq_25'] = f_quantile_w(0.25)
    out['wq_50'] = f_quantile_w(0.5)
    out['wq_75'] = f_quantile_w(0.75)
    out['wq_90'] = f_quantile_w(0.9)
    out['wq_95'] = f_quantile_w(0.95)
    out['wq_99'] = f_quantile_w(0.99)

    # Power-weighted moments of the frequency distribution: the spectrum (non-negative) is treated as a
    # weighting over frequency, giving the textbook spectral centroid (mean frequency), spread (standard
    # deviation), skewness and kurtosis. Not to be confused with mom3, a moment of the distribution of
    # power *values*, nor with 'wq_50' above, which is the median frequency (the 50% point of the
    # cumulative power).
    s_pos = np.maximum(s, 0)  # guard against any tiny negative values from the estimator
    sum_s = np.sum(s_pos)
    out['specCentroid'] = out['specSpread'] = out['specSkew'] = out['specKurt'] = np.nan
    if sum_s > 0:
        pw = s_pos / sum_s  # normalized weighting over frequency
        out['specCentroid'] = np.sum(pw * w)
        w_dev = w - out['specCentroid']
        spec_var = np.sum(pw * w_dev ** 2)
        out['specSpread'] = np.sqrt(spec_var)
        if spec_var > 0:  # otherwise all the power is in a single bin and the shape is undefined
            out['specSkew'] = np.sum(pw * w_dev ** 3) / spec_var ** 1.5
            out['specKurt'] = np.sum(pw * w_dev ** 4) / spec_var ** 2

    # Fit some functions to this cumulative sum:
    # Quadratic
    a, b, c = np.polyfit(w, cs_s, deg=2)
    out['fpoly2csS_p1'] = a
    out['fpoly2csS_p2'] = b
    out['fpoly2csS_p3'] = c
    quad = lambda x, a, b, c: a * x**2 + b * x + c
    gof = goodness_of_fit(cs_s, quad(w, a, b, c), 3)
    out['fpoly2_sse'] = gof['sse'] * dw  # integrated (not summed) squared error
    out['fpoly2_r2'] = gof['rsquare']
    out['fpoly2_rmse'] = gof['rmse']

    # Fit polysat a*x^2/(b+x^2) (has zero derivative at zero, though)
    f_sat = _polysat_fit(w, cs_s)
    out['fpolysat_a'] = f_sat['a']
    out['fpolysat_b'] = f_sat['b']
    out['fpolysat_r2'] = f_sat['r2']
    out['fpolysat_rmse'] = f_sat['rmse']

    # Shannon spectral entropy, from the spectrum rescaled to exactly unit area, Sn = S/(sum(S)*dw):
    # (i) spect_shann_ent: -integral of Sn log(Sn) dw, the differential Shannon entropy of the
    #     power distribution over frequency
    # (ii) spect_shann_ent_norm: exp(spect_shann_ent) as a fraction of the frequency range N*dw
    #     (1 for a flat spectrum, towards 0 as the power concentrates in a narrow band)
    sn = s / (np.sum(s) * dw)
    out['spect_shann_ent'] = np.sum(-sn * np.log(sn)) * dw
    out['spect_shann_ent_norm'] = np.exp(out['spect_shann_ent']) / (n * dw)

    #"Spectral Flatness Measure"
    #which is given in dB as 10 log_10(gm/am) where gm is the geometric mean and am
    # is the arithmetic mean of the power spectral density
    out['sfm'] = 10 * np.log10(np.exp(np.mean(np.log(s))) / np.mean(s))

    # Areas under power spectrum
    out['areatopeak'] = np.sum(s[0:np.argmax(s) + 1]) * dw
    out['ylogareatopeak'] = np.sum(log_s[0:np.argmax(s) + 1]) * dw  # % (semilogy)

    # Robust Fits (iteratively re-weighted least squares); only the statistics hctsa emits per range
    # across full range
    r_all = w > 0
    out |= _give_me_robust_stats(np.log(w[r_all]), np.log(s[r_all]), 'linfitloglog_all',
                                 ('a1', 'a2', 'sigrat', 'sigma', 'sea1'))
    # across first half (low frequency)
    r_lf = (w > 0)
    r_lf[int(np.floor(n/2)):] = 0 #% remove second half of angular frequenciesf
    out |= _give_me_robust_stats(np.log(w[r_lf]), np.log(s[r_lf]), 'linfitloglog_lf', ('a2',))
    # across second half (high frequency)
    r_hf = np.arange(n // 2, n)
    out |= _give_me_robust_stats(np.log(w[r_hf]), np.log(s[r_hf]), 'linfitloglog_hf',
                                 ('a1', 'a2', 'sigrat', 'sigma', 'sea1'))
    # Middle half (mid-frequencies); n/4 and 3n/4 rounded half away from zero
    start = (n + 2) // 4  # round(n/4)
    stop = (3 * n + 2) // 4  # round(3n/4)
    r_mf = np.arange(start, stop)
    out |= _give_me_robust_stats(np.log(w[r_mf]), np.log(s[r_mf]), 'linfitloglog_mf', ('a2',))
    # Fit linear to semilog plot (across full range)
    out |= _give_me_robust_stats(w, np.log(s), 'linfitsemilog_all', ('a1', 'sigrat', 'sigma', 'sea1'))

    # Power in specific frequency bands
    # % 2 bands
    split = _equal_bands(s, 2)
    out['area_2_1'] = np.sum(split[:, 0]) * dw
    out['logarea_2_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_2_2'] = np.sum(split[:, 1]) * dw
    out['logarea_2_2'] = np.sum(np.log(split[:, 1])) * dw
    out['statav2_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)
    # The same on log(S): on the linear spectrum, whichever band contains the dominant peak swamps these
    split_log = _equal_bands(log_s, 2)
    out['logstatav2_m'] = np.std(np.mean(split_log, axis=0), ddof=1) / np.std(log_s, ddof=1)
    out['logstatav2_s'] = np.std(np.std(split_log, ddof=1, axis=0), axis=0, ddof=1) / np.std(log_s, ddof=1)

    # 5 bands
    split = _equal_bands(s, 5)
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
    out['statav5_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)
    split_log = _equal_bands(log_s, 5)
    out['logstatav5_m'] = np.std(np.mean(split_log, axis=0), ddof=1) / np.std(log_s, ddof=1)
    out['logstatav5_s'] = np.std(np.std(split_log, ddof=1, axis=0), axis=0, ddof=1) / np.std(log_s, ddof=1)

    # Count crossings of the log spectrum with a horizontal line set a fraction of the way from
    # min(log S) to max(log S). On the linear spectrum a threshold at a fixed fraction of max(S) sits far
    # above the noise floor whenever there is one dominant peak; differences in log S are power
    # ratios (dB), so a fraction of the log range is a genuinely relative 'how far up from the
    # noise floor' level.
    log_range = np.max(log_s) - np.min(log_s)
    ncrossfn_rel_log = lambda frac: np.sum(sign_change(log_s - (np.min(log_s) + frac * log_range)))
    out['ncross_log_f05'] = ncrossfn_rel_log(0.05)
    out['ncross_log_f10'] = ncrossfn_rel_log(0.1)
    out['ncross_log_f20'] = ncrossfn_rel_log(0.2)
    out['ncross_log_f50'] = ncrossfn_rel_log(0.5)

    return out

def _equal_bands(x: np.ndarray, n_bands: int) -> np.ndarray:
    """Split ``x`` into ``n_bands`` consecutive bands of equal length ``len(x) // n_bands``
    (columns of the result; the last ``len(x) % n_bands`` values are dropped)."""
    m = len(x) // n_bands
    return np.asarray(x[:m * n_bands]).reshape(n_bands, m).T


def _mean_or_nan(x):
    """Mean of an array; NaN (without a warning) if it is empty.

    Parameters
    ----------
    x : array_like
        1-D sequence of values.

    Returns
    -------
    float

    Notes
    -----
    The mean of no values is undefined, so NaN is returned (rather than a warning from
    ``np.mean``): a statistic over an empty set of peaks is missing, not zero.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float('nan')
    return float(np.mean(a))

def _findpeaks(s, min_pk_dist=0, sort_str='none'):
    """Local maxima of a signal, with minimum peak separation and sorting.

    Parameters
    ----------
    s : array-like
        1-D signal of length L.
    min_pk_dist : float, optional
        Minimum separation of peaks in samples (0 for none). Peaks are visited tallest
        first and each visited peak removes all other peaks within ``min_pk_dist`` samples
        (inclusive), so that surviving peaks are more than ``min_pk_dist`` samples apart.
    sort_str : {'none', 'descend', 'ascend'}, optional
        Order of the output: by position, tallest first, or shortest first (ties by position).

    Returns
    -------
    pk_height : numpy.ndarray
        Peak heights (float64).
    pk_loc : numpy.ndarray
        0-based indices of the peaks (int64), ``pk_height == s[pk_loc]``.

    Notes
    -----
    A thin wrapper of :func:`scipy.signal.find_peaks` (a peak is a sample strictly higher than
    both neighbors; the middle sample of a flat-topped peak). The inclusive minimum separation
    ``d`` is ``distance = floor(d) + 1`` of ``find_peaks``.
    """
    if sort_str not in ('none', 'descend', 'ascend'):
        raise ValueError("sort_str must be 'none', 'descend' or 'ascend'")
    if min_pk_dist < 0:
        raise ValueError("min_pk_dist must be nonnegative")
    s = np.asarray(s, dtype=float).ravel()
    distance = int(np.floor(min_pk_dist)) + 1 if min_pk_dist > 0 else None
    loc, _ = scipy.signal.find_peaks(s, distance=distance)
    loc = loc.astype(np.int64)
    h = s[loc]
    if sort_str == 'descend':
        order = np.argsort(-h, kind='stable')
    elif sort_str == 'ascend':
        order = np.argsort(h, kind='stable')
    else:
        order = np.arange(loc.size)
    return h[order].copy(), loc[order]

def _give_me_robust_stats(x_data: ArrayLike, y_data: ArrayLike, field_name: str,
                          which_stats=('a1', 'a2', 'sigrat', 'sigma', 'sea1')) -> dict:
    """
    Statistics based on a robust linear fit.

    ``which_stats`` selects which of the available statistics to emit: ``a1`` (robust intercept),
    ``a2`` (robust gradient), ``sigrat`` (ratio of the OLS to the robust sigma estimate), ``sigma``
    (residual sigma estimate), ``sea1`` / ``sea2`` (standard error of the intercept / gradient).
    """
    out = {}
    try:
        a, stats = robust_linear_fit(x_data, y_data)
        available = {
            'a1': lambda: a[0],  # robust intercept
            'a2': lambda: a[1],  # robust gradient
            # ratio of sigma estimates between ordinary least squares and the robust fit:
            'sigrat': lambda: stats['ols_s'] / stats['robust_s'],
            # sigma as the larger of robust_s and a weighted average of ols_s and robust_s:
            'sigma': lambda: stats['s'],
            'sea1': lambda: stats['se'][0],  # standard error in intercept
            'sea2': lambda: stats['se'][1],  # standard error in slope
        }
        for key in which_stats:
            out[f'{field_name}_{key}'] = available[key]()
    except Exception:
        for key in which_stats:
            out[f'{field_name}_{key}'] = np.nan
    return out

@dict_output
def phase_amp_coupling(y: ArrayLike, n_bands: int = 5, max_n: Union[int, str] = 'full',
                       n_phase_bins: int = 18) -> dict:
    """
    Cross-frequency phase-amplitude coupling.

    Parameters
    ----------
    y : array-like
        The input time series.
    n_bands : int, optional
        The number of equal-width frequency bands to split the spectrum into
        (default: 5, matching :func:`spectral_summaries`' 5-band split).
        Phase-amplitude pairs are formed from every pair of bands i < j (phase
        from the slower band, amplitude from the faster), giving
        ``comb(n_bands, 2)`` pairs.
    max_n : int or str, optional
        The maximum number of samples to consider; longer series are cropped to
        their first ``max_n`` points. Can be ``'full'`` to disable cropping
        (default).
    n_phase_bins : int, optional
        The number of phase bins used to estimate each band pair's modulation
        index (default: 18, i.e. 20-degree bins, the standard choice from Tort
        et al. 2010).

    Returns
    -------
    dict
        - ``maxMI``: the maximum modulation index across all band pairs -- the
          comodulogram peak, i.e. whether *any* band pair shows real coupling.
        - ``entropyMI``: the normalized Shannon entropy of the MI values across
          pairs (0 = coupling concentrated in a single band pair, 1 = uniformly
          diffuse across all pairs).

        Returns NaN if the time series is too short for the requested number of
        bands.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_bands, n_phase_bins = int(n_bands), int(n_phase_bins)

    n = len(y)
    if isinstance(max_n, str) and max_n == 'full':
        pass # No cropping
    elif n > max_n:
        warnings.warn(f"Time series ({n} samples) exceeds max_n = {int(max_n)}; "
                      f"analyzing the first {int(max_n)} samples")
        y = y[:int(max_n)]
        n = int(max_n)

    # ------------------------------------------------------------------------------
    # Equal-width frequency bands (DC and Nyquist bins excluded, as in
    # spectral_summaries_phase -- neither carries meaningful oscillatory phase)
    # ------------------------------------------------------------------------------
    half_n = n // 2 + 1 # one-sided bins: DC (0) up to Nyquist (half_n - 1, if n even)
    is_nyquist_bin = (n % 2 == 0)
    if is_nyquist_bin:
        usable_bins = np.arange(1, half_n - 1)
    else:
        usable_bins = np.arange(1, half_n)

    min_bins_per_band = 2 # need >=2 FFT bins per band for a non-degenerate analytic signal
    if len(usable_bins) < n_bands * min_bins_per_band:
        warnings.warn(f"Time series too short (N = {n}) for {n_bands} usable frequency bands")
        return np.nan

    edges = _even_split_points(0, len(usable_bins), n_bands + 1)
    band_bins = [usable_bins[edges[b]:edges[b + 1]] for b in range(n_bands)]

    # ------------------------------------------------------------------------------
    # Per-band analytic signal (FFT-domain Hilbert trick, band-limited in one step)
    # ------------------------------------------------------------------------------
    y_fft = scipy.fft.fft(y)
    phase_band, amp_band = [], []
    for b in range(n_bands):
        yb = np.zeros(n, dtype=complex)
        yb[band_bins[b]] = 2 * y_fft[band_bins[b]]
        zb = scipy.fft.ifft(yb)
        phase_band.append(np.angle(zb))
        amp_band.append(np.abs(zb))

    # ------------------------------------------------------------------------------
    # Modulation index (Tort et al. 2010) for every phase(i)-amplitude(j) pair, i < j
    # ------------------------------------------------------------------------------
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    h_max = np.log(n_phase_bins)
    mi = []
    for i in range(n_bands - 1):
        for j in range(i + 1, n_bands):
            phi = phase_band[i]
            amp = amp_band[j]
            bin_idx = np.digitize(phi, phase_edges) - 1
            bin_idx[bin_idx == n_phase_bins] = n_phase_bins - 1 # phi == pi edge case
            counts = np.bincount(bin_idx, minlength=n_phase_bins)
            sums = np.bincount(bin_idx, weights=amp, minlength=n_phase_bins)
            mean_amp = np.divide(sums, counts, out=np.zeros(n_phase_bins), where=counts > 0)
            p = mean_amp / np.sum(mean_amp)
            p = p[p > 0]
            h = -np.sum(p * np.log(p))
            mi.append((h_max - h) / h_max)
    mi = np.asarray(mi)
    num_pairs = len(mi)

    if not np.any(np.isfinite(mi)):
        return np.nan

    # ------------------------------------------------------------------------------
    # Summary statistics across band pairs
    # ------------------------------------------------------------------------------
    out = {}
    out['maxMI'] = np.max(mi)

    # Normalized Shannon entropy of the MI values across pairs (0 = coupling
    # concentrated in one pair, 1 = uniformly diffuse):
    p = mi[mi > 0]
    if p.size == 0:
        out['entropyMI'] = np.nan
    else:
        p = p / np.sum(p)
        out['entropyMI'] = -np.sum(p * np.log(p)) / np.log(num_pairs)

    return out

@dict_output
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
    # Reference the phase to the centre of the series rather than the first sample. This
    # removes a linear term of ~pi*N/NFFT per bin that would otherwise dominate the
    # unwrapped phase (groupDelay ~ N/2 for any stationary series).
    sc = sc * np.exp(1j * w * (ny - 1) / 2)
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
    out['groupDelay'] = -beta[1] / ny  # relative to the series centre, as a fraction of its length
    resid = ph_unwrap - X @ beta
    out['phaseLinearity'] = np.sqrt(np.sum(wgt * resid ** 2)) / np.sqrt(len(ww))

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

@dict_output
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
    # The fit excludes the zero-frequency (DC) bin: a z-scored series has (almost) no
    # power there, so log|X| at DC is a huge negative outlier (about -30) that would
    # otherwise bend the fitted envelope. The detrending below still covers all bins.
    pEnv = np.polyfit(fIdx[1:], halfLogMag[1:], envOrder)
    halfDetrended = halfLogMag - np.polyval(pEnv, fIdx)
    # The zero-frequency bin is set to the envelope (0 after detrending): for a z-scored
    # series |X| there is rounding noise (log|X| anywhere from about -36 to -28), which would
    # add an arbitrary constant, (log|X(0)| - envelope)/NFFT, to the whole cepstrum.
    halfDetrended[0] = 0.0

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

    pFit = np.polyfit(periods.astype(float), cSearch, 1)
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



def _fourier_terms(t: np.ndarray, w: float, n: int) -> np.ndarray:
    """
    Design matrix of an n-term Fourier series with fundamental frequency ``w``.

    Parameters
    ----------
    t : np.ndarray
        Sample positions, shape (N,).
    w : float
        Fundamental angular frequency (radians per unit of ``t``).
    n : int
        Number of harmonics (n >= 1).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2n + 1) with columns ``[1, cos(wt), sin(wt), cos(2wt),
        sin(2wt), ..., cos(nwt), sin(nwt)]``, so that ``X @ [a0, a1, b1, ..., an, bn]``
        evaluates the model ``a0 + sum_i a_i cos(i w t) + b_i sin(i w t)``.

    Notes
    -----
    The n-term Fourier series ``a0 + sum_{i=1}^{n} a_i cos(i w t) + b_i sin(i w t)``.
    """
    t = np.asarray(t, dtype=float)
    X = np.empty((t.size, 2 * n + 1), dtype=float)
    X[:, 0] = 1.0
    for i in range(1, n + 1):
        arg = i * w * t
        X[:, 2 * i - 1] = np.cos(arg)
        X[:, 2 * i] = np.sin(arg)
    return X


def _fourier_start_point(t: np.ndarray, y: np.ndarray, n: int) -> float:
    """
    Starting fundamental frequency for fitting an n-term Fourier series.

    The periodogram peak of ``y`` (DC excluded) is located; because the peak may belong to
    any of the first ``n`` harmonics, each of ``w_peak / j`` (j = 1..n) is tried as the
    fundamental, the linear coefficients are solved by least squares, and the candidate with
    the smallest residual sum of squares is returned.

    Parameters
    ----------
    t : np.ndarray
        Equally spaced, increasing sample positions, shape (N,), N >= 3.
    y : np.ndarray
        Data, shape (N,).
    n : int
        Number of harmonics.

    Returns
    -------
    float
        Start value (radians per unit of ``t``) for the fundamental frequency. For a
        degenerate (all-zero) spectrum, the frequency of FFT bin 1 is returned.

    Notes
    -----
    The frequency grid is that of the N-point DFT of the samples,
    ``2 pi k / (N (t[1] - t[0]))``, k = 1..floor(N/2), and the peak is ``argmax`` of
    ``|rfft(y)|`` over k >= 1. The choice among the harmonics is a small multi-start of the
    nonlinear fit, which is multimodal in the frequency.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    N = y.size
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    w_bin1 = 2.0 * np.pi / (N * dt) if dt != 0 else 1.0
    try:
        amp = np.abs(np.fft.rfft(y))
        if amp.size < 2:
            return float(w_bin1)
        k_star = int(np.argmax(amp[1:])) + 1
        w_peak = w_bin1 * k_star
        if not np.isfinite(w_peak):
            return float(w_bin1)
        best_w, best_sse = w_peak, np.inf
        for j in range(1, n + 1):
            w = w_peak / j
            X = _fourier_terms(t, w, n)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            r = X @ coef - y
            sse = float(r @ r)
            if sse < best_sse:  # strict: smallest j wins ties
                best_sse, best_w = sse, w
        return float(best_w)
    except (np.linalg.LinAlgError, ValueError):
        return float(w_bin1)


@dict_output
def sinusoid_fit(y: ArrayLike, model: str = 'sin1') -> Union[dict, float]:
    """
    Fit sinusoids or a Fourier series to the time series.

    Fits a sum of 1-3 sinusoids, or a Fourier series with 1-3 terms, to the time
    series as a function of its time index ``t = 0..N-1``. The values are fitted in
    the order in which they occur, so the result depends on the temporal ordering
    of the data. (This is the time-series-model branch of the former
    ``DN_SimpleFit``, split off in hctsa as ``SP_SinusoidFit`` because the
    distribution of values is unaffected by temporal ordering, whereas these fits
    are.)

    The fitted models are:

    - ``'sinK'``: a sum of K sinusoids, ``sum_i a_i*sin(2*pi*f_i*t + c_i)``, with free
      amplitudes, phases, and frequencies ``f_i`` in ``[1/(2N), 1/2 - 1/(2N)]`` cycles
      per sample. The amplitudes and phases are linear parameters, found by least squares
      for given frequencies, and the frequencies are searched deterministically (no
      random starts, no iterative optimizer): :func:`pyhctsa.robust.bf_fit_sinusoids`.
      The frequencies are bounded below by ``1/(2N)`` because a sinusoid of lower
      frequency cannot be told apart from a constant plus a linear trend, so that its
      amplitude and phase are not determined.
    - ``'fourierK'``: a K-term Fourier series,
      ``a0 + sum_i (a_i*cos(i*w*t) + b_i*sin(i*w*t))``, with a single fitted
      fundamental frequency ``w``, so that the terms are harmonically related.

    Goodness of fit is summarized by the root mean square error (and R^2), and the
    residuals are characterized by their autocorrelation at lags 1 and 2 and by a
    runs test, which together reveal remaining temporal structure that the model
    has not captured (e.g., whether a periodic component has been fully explained).

    Parameters
    ----------
    y : array-like
        The input time series.
    model : {'sin1', 'sin2', 'sin3', 'fourier1', 'fourier2', 'fourier3'}, optional
        The model to fit. Default is ``'sin1'``.

    Returns
    -------
    dict or float
        - ``r2``: the R^2 of the fit.
        - ``adjr2``: the degrees-of-freedom-adjusted R^2.
        - ``rmse``: the root mean square error of the fit (the residual sum of squares
          divided by the degrees of freedom of the error).
        - ``resAC1``, ``resAC2``: the autocorrelation of the residuals at lags 1
          and 2 (``'Fourier'`` method).
        - ``resrunsz``: the signed z-statistic of a runs test on the residuals
          (:func:`pyhctsa.robust.bf_runs_z`): negative when the residuals have fewer
          runs about their median than expected for a random order (slowly varying
          residuals).

        The three residual outputs are NaN if the fit is exact (nothing but rounding
        error is left). NaN is returned instead of a dict if the model cannot be fitted
        (or, for K sinusoids, if there are fewer than ``3K + 1`` samples).

    Notes
    -----
    The Fourier series are fitted by nonlinear least squares (the linear coefficients and the
    fundamental frequency) with SciPy's trust-region-reflective ``least_squares``, started
    from the periodogram peak of the data (see :func:`_fourier_start_point`). The fit is
    multimodal in the frequency, so the result depends on the start point.
    """
    models = ('sin1', 'sin2', 'sin3', 'fourier1', 'fourier2', 'fourier3')
    if model not in models:
        raise ValueError(f"Invalid time-series model '{model}' specified")
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    t = np.arange(N, dtype=float)
    kind, n = model[:-1], int(model[-1])

    try:
        if kind == 'sin':
            # Sum of K sinusoids: least squares over amplitudes and phases, searched over frequencies
            n_coeff = 3 * n  # amplitude, frequency and phase of each sinusoid
            if N <= n_coeff:
                return np.nan
            fitted, _ = bf_fit_sinusoids(y, n)
        else:
            if N < 3:
                return np.nan

            def fun(p):
                return _fourier_terms(t, p[-1], n) @ p[:-1] - y

            def jac(p):
                w = p[-1]
                X = _fourier_terms(t, w, n)
                J = np.empty((N, 2 * n + 2))
                J[:, :-1] = X
                dw = np.zeros(N)
                for i in range(1, n + 1):
                    dw += (-p[2 * i - 1] * np.sin(i * w * t) + p[2 * i] * np.cos(i * w * t)) * i * t
                J[:, -1] = dw
                return J

            w0 = _fourier_start_point(t, y, n)
            coef = np.linalg.lstsq(_fourier_terms(t, w0, n), y, rcond=None)[0]
            p0 = np.append(coef, w0)
            sol = scipy.optimize.least_squares(fun, p0, jac=jac, method='trf')
            fitted = fun(sol.x) + y
            n_coeff = 2 * n + 2
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.nan  # the model could not be fitted
    if not np.all(np.isfinite(fitted)):
        return np.nan

    res = y - fitted
    sse = np.sum(res ** 2)
    sstot = np.sum((y - np.mean(y)) ** 2)
    dfe = N - n_coeff  # degrees of freedom of the error
    with np.errstate(all='ignore'):
        r2 = 1 - sse / sstot
        adjr2 = 1 - (1 - r2) * (N - 1) / dfe if dfe > 0 else np.nan
        rmse = np.sqrt(sse / dfe) if dfe > 0 else np.nan
    out = {'r2': float(r2), 'adjr2': float(adjr2), 'rmse': float(rmse)}
    # Remaining structure in the residuals
    out['resAC1'], out['resAC2'], out['resrunsz'] = bf_residual_stats(res, sstot)
    return out


def _envelope_summary(env: np.ndarray):
    # Distributional and autocorrelation summaries of an amplitude envelope:
    # (cv, skewness, kurtosis, 1/e autocorrelation timescale), NaN where undefined.
    cv = sk = ku = tau = np.nan
    n = len(env)
    m = np.mean(env)
    if not m > 0:
        return cv, sk, ku, tau
    e = env - m
    s2 = np.mean(e ** 2)
    cv = np.sqrt(s2) / m  # population standard deviation over the mean

    # Weight of the measured skewness, kurtosis and timescale: these are a 0/0 for
    # a constant envelope, so they are shrunk continuously toward conventional
    # values (see the function help); w -> 1 for any envelope that varies.
    cv_scale = 1e-10  # well above the round-off CV of a noiseless sinusoid (~1e-15)
    w = cv ** 2 / (cv ** 2 + cv_scale ** 2)
    tau_max = n // 2  # the largest lag searched below (the envelope never decorrelates)

    sk_meas = ku_meas = tau_meas = np.nan
    if s2 > 0:
        sk_meas = np.mean(e ** 3) / s2 ** 1.5
        ku_meas = np.mean(e ** 4) / s2 ** 2

        # Autocorrelation of the envelope via the FFT (zero-padded, biased estimator)
        nfft = 2 ** int(np.ceil(np.log2(2 * n)))
        F = np.fft.fft(e, nfft)
        acf = np.real(np.fft.ifft(np.abs(F) ** 2))
        acf = acf[:n // 2 + 1] / acf[0]  # lags 0..n/2
        below = np.flatnonzero(acf < np.exp(-1))
        if below.size > 0 and below[0] > 0:
            ic = below[0]
            # linear interpolation between lags ic-1 and ic
            a0, a1 = acf[ic - 1], acf[ic]
            tau_meas = (ic - 1) + (a0 - np.exp(-1)) / (a0 - a1)

    def shrink(measured, limit):
        # limit + w*(measured - limit); an undefined measured value takes the
        # conventional value only when w is negligible (an essentially constant envelope)
        if np.isnan(measured):
            return limit if w < 1e-6 else np.nan
        return limit + w * (measured - limit)

    return cv, shrink(sk_meas, 0), shrink(ku_meas, 3), shrink(tau_meas, tau_max)


def envelope_stats(y: ArrayLike, power_frac: float = 0.5, trim_frac: float = 0.05) -> dict:
    """
    Statistics of the amplitude envelope of the full series and of its dominant oscillation.

    Computes the instantaneous amplitude envelope ``|z(t)|`` of the analytic signal
    ``z(t) = y(t) + i H[y](t)`` (``H`` the Hilbert transform) for (a) the full
    series and (b) the dominant band: the narrowest frequency band centered on the
    largest periodogram peak (DC and Nyquist bins excluded) that holds a fraction
    ``power_frac`` of the total power, and at least 2 bins either side of the peak.
    Its width therefore adapts to the series: it is the width of the dominant peak
    for a narrowband oscillation, and a large part of the spectrum for broadband
    noise. Each analytic signal comes from the FFT, zeroing the bins outside the
    band.

    The envelope summaries describe amplitude modulation: how variable the envelope
    is (coefficient of variation), how asymmetric and heavy-tailed its distribution
    is (kurtosis for the full series; skewness and kurtosis for the dominant band),
    and how long it takes to decorrelate (the 1/e timescale of its autocorrelation
    function). An unmodulated sinusoid has a constant envelope (CV near 0);
    bursting or amplitude-modulated signals have a large CV, and a high kurtosis
    for intermittent bursts.

    Baseline for Gaussian noise: the analytic signal of a stationary Gaussian
    process is complex Gaussian, so its envelope is Rayleigh distributed: CV =
    ``sqrt(4/pi - 1)`` = 0.5227, skewness 0.6311, kurtosis 3.2451. Values of the CV
    below this indicate an envelope steadier than noise (e.g., a sinusoid in noise
    follows a Rice distribution), and above it an envelope more modulated than noise.

    To limit edge effects (the FFT filter is circular, so the series ends wrap
    around), ``trim_frac`` of the samples are dropped from each end of the envelope
    and phase before any summary is computed. Timescales are in samples.

    Parameters
    ----------
    y : array-like
        The input time series.
    power_frac : float, optional
        The fraction of the total (one-sided, DC and Nyquist excluded) spectral
        power that the dominant band, centered on the largest periodogram peak,
        must contain. Default is 0.5.
    trim_frac : float, optional
        The fraction of samples dropped from each end of the analytic signal before
        computing summaries. Default is 0.05.

    Returns
    -------
    dict
        - ``full_cv``: the envelope's coefficient of variation (standard deviation
          over mean), full series.
        - ``full_kurt``: the envelope's kurtosis, full series.
        - ``full_tau``: the 1/e decay time (in samples) of the envelope's
          autocorrelation function, full series.
        - ``dom_cv``, ``dom_skew`` (the envelope's skewness), ``dom_kurt``,
          ``dom_tau``: as above for the dominant band.
        - ``dom_ifspread``: a robust spread of the dominant band's instantaneous
          frequency (1.4826 times the median absolute deviation of the phase
          increments, in cycles per sample).

        All fields are NaN for constant, non-finite, or very short (N < 50) series.
        A 1/e timescale is NaN when the autocorrelation never falls below 1/e
        within N/2 lags. For an essentially constant envelope (as for a sinusoid
        without noise) the skewness, kurtosis and timescale are an undefined 0/0;
        they are set by convention to skewness 0, kurtosis 3 (the Gaussian values)
        and a timescale of the largest lag searched, ``n // 2`` for the ``n``
        samples left after trimming, each shrunk toward that value with weight
        ``w = c^2 / (c^2 + 1e-20)`` where ``c`` is the envelope's CV, so the
        reported values vary continuously with ``c``.

    References
    ----------
    B. Boashash, "Estimating and interpreting the instantaneous frequency of a
    signal. I. Fundamentals", Proc. IEEE 80(4), 520-538 (1992).
    """
    y = np.asarray(y, dtype=float).ravel()
    out = {f: np.nan for f in ('full_cv', 'full_kurt', 'full_tau', 'dom_cv', 'dom_skew',
                               'dom_kurt', 'dom_tau', 'dom_ifspread')}
    N = len(y)
    if N < 50 or not np.all(np.isfinite(y)) or np.std(y, ddof=1) == 0:
        return out
    y = y - np.mean(y)

    # Frequency bins (DC and Nyquist excluded, as in phase_amp_coupling):
    half_n = N // 2 + 1
    usable = np.arange(1, half_n - 1 if N % 2 == 0 else half_n)
    Y = np.fft.fft(y)

    # Samples dropped from each end of the (circularly computed) analytic signal:
    n_trim = max(1, int(np.floor(trim_frac * N + 0.5)))
    keep = slice(n_trim, N - n_trim)

    # (a) Full band
    Yfull = np.zeros(N, dtype=complex)
    Yfull[usable] = 2 * Y[usable]
    env_full = np.abs(np.fft.ifft(Yfull)[keep])
    out['full_cv'], _, out['full_kurt'], out['full_tau'] = _envelope_summary(env_full)

    # (b) Dominant band: the largest periodogram peak +/- the smallest half-width
    # (at least 2 bins) holding power_frac of the power
    pw = np.abs(Y[usable]) ** 2
    n_bins = len(usable)
    i_peak = int(np.argmax(pw))
    cum_pow = np.concatenate(([0.0], np.cumsum(pw)))
    hws = np.arange(2, n_bins + 1)
    band_pow = (cum_pow[np.minimum(n_bins - 1, i_peak + hws) + 1]
                - cum_pow[np.maximum(0, i_peak - hws)])
    ok = np.flatnonzero(band_pow >= power_frac * cum_pow[-1])
    half_width = int(hws[ok[0]]) if ok.size else n_bins
    peak_bin = usable[i_peak]
    band = np.arange(max(usable[0], peak_bin - half_width),
                     min(usable[-1], peak_bin + half_width) + 1)

    Ydom = np.zeros(N, dtype=complex)
    Ydom[band] = 2 * Y[band]
    z_dom = np.fft.ifft(Ydom)
    env_dom = np.abs(z_dom[keep])
    out['dom_cv'], out['dom_skew'], out['dom_kurt'], out['dom_tau'] = _envelope_summary(env_dom)

    # Instantaneous frequency (cycles per sample): the unwrapped phase increments
    # over the trimmed segment, summarized robustly (scaled MAD, which equals the
    # standard deviation for a Gaussian)
    phi = np.unwrap(np.angle(z_dom[keep]))
    inst_freq = np.diff(phi) / (2 * np.pi)
    out['dom_ifspread'] = 1.4826 * np.median(np.abs(inst_freq - np.median(inst_freq)))
    return out


@dict_output
def phase_fluctuation_scaling(y: ArrayLike, half_width_frac: float = 0.01,
                              num_windows: int = 16, max_n: int = 10000) -> Union[dict, float]:
    """
    Multi-scale fluctuation analysis of the instantaneous phase of the dominant oscillation.

    Isolates the dominant oscillatory component of ``y`` (the frequency band
    carrying the most spectral power, excluding DC and Nyquist), takes its
    instantaneous phase via the analytic signal, and asks how the fluctuation of
    that phase about its mean rotation rate grows with window size ``w``:
    ``mean(|dphi(t+w) - dphi(t)|)`` against ``w``, in log-log space (the same logic
    as detrended fluctuation analysis, as in ``fluctuation_analysis``, applied to
    instantaneous phase instead of raw values).

    The method is inspired by the analytic-signal analysis of strange nonchaotic
    dynamics (SNA): for nonchaotic dynamics the curve rises at short windows but
    flattens at longer windows (the largest Lyapunov exponent is negative and the
    phase dynamics are globally stable), whereas for chaotic dynamics it keeps
    rising at long windows. The original method uses empirical mode decomposition
    to isolate the dominant intrinsic mode; this implementation instead bandpasses
    around the single dominant non-edge FFT peak (the band-limited analytic-signal
    trick of ``phase_amp_coupling``), which is simpler and avoids EMD's mode-mixing
    and boundary sensitivities, at the cost of not being a literal reimplementation.

    Parameters
    ----------
    y : array-like
        The input time series.
    half_width_frac : float, optional
        Half-width of the frequency band around the dominant peak, as a fraction of
        the usable (DC- and Nyquist-excluded) one-sided spectrum, and at least 2
        bins. Default is 0.01.
    num_windows : int, optional
        The number of log-spaced window sizes (from 2 samples to N/4) to evaluate,
        split into a short-window half and a long-window half for two separate
        linear fits in log-log space. Default is 16.
    max_n : int, optional
        The maximum number of samples to consider; longer series are cropped to
        their first ``max_n`` points. Default is 10000.

    Returns
    -------
    dict or float
        - ``meanFreq``: the mean rotation frequency of the isolated dominant
          component, in cycles per sample.
        - ``slope_short``, ``slope_long``: log-log slopes of the phase fluctuation
          against window size, over the shorter and the longer half of the window
          sizes (the two fits share one point).
        - ``slope_diff``: ``slope_short - slope_long``.

        NaN is returned if the series is too short for a well-defined dominant band
        (or the phase is non-finite); the slopes are NaN if it is too short for the
        multi-scale analysis.

    Notes
    -----
    The short-window slope does not behave as the SNA reasoning suggests: the phase
    comes from a narrow band, so it is smooth at short lags for any input, and
    ``mean(|dphi(t+w) - dphi(t)|)`` grows linearly with ``w`` there (the trivial
    derivative regime): ``slope_short`` is about 1 regardless of dynamics, and
    ``slope_diff`` is just ``1 - slope_long``. The informative quantity is
    ``slope_long`` alone: near 0 where the phase fluctuation saturates (periodic,
    SNA-like), and positive where it keeps growing (chaotic, noisy).

    References
    ----------
    K. Gupta, A. Prasad, H.P. Singh, R. Ramaswamy, "Analytical signal analysis of
    strange nonchaotic dynamics", Phys. Rev. E 77, 046220 (2008).
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    if N > max_n:
        warnings.warn(f"Time series ({N} samples) exceeds max_n = {max_n}; "
                      f"analyzing the first {max_n} samples")
        y = y[:max_n]
        N = max_n

    # Dominant non-edge spectral peak (DC and Nyquist bins excluded):
    half_n = N // 2 + 1
    usable = np.arange(1, half_n - 1 if N % 2 == 0 else half_n)
    half_width_bins = max(2, int(np.floor(half_width_frac * len(usable) + 0.5)))
    if len(usable) < 4 * half_width_bins:
        warnings.warn(f"Time series too short (N = {N}) for a well-defined dominant frequency band")
        return np.nan

    Y = np.fft.fft(y)
    power = np.abs(Y[usable]) ** 2
    peak_bin = usable[int(np.argmax(power))]
    band = np.arange(max(usable[0], peak_bin - half_width_bins),
                     min(usable[-1], peak_bin + half_width_bins) + 1)

    # Band-limited analytic signal and its instantaneous phase:
    Yband = np.zeros(N, dtype=complex)
    Yband[band] = 2 * Y[band]
    phi = np.unwrap(np.angle(np.fft.ifft(Yband)))
    if not np.all(np.isfinite(phi)):
        warnings.warn("Non-finite instantaneous phase (degenerate band-limited signal?)")
        return np.nan

    # Detrend: remove the mean rotation rate to leave the phase fluctuation
    A = np.column_stack((np.arange(N, dtype=float), np.ones(N)))
    coef = np.linalg.lstsq(A, phi, rcond=None)[0]
    out = {'meanFreq': coef[0] / (2 * np.pi)}  # cycles per sample
    dphi = phi - A @ coef
    out.update(slope_short=np.nan, slope_long=np.nan, slope_diff=np.nan)

    # Multi-scale fluctuation analysis: mean(|dphi(t+w) - dphi(t)|) vs w
    w_min = 2
    w_max = N // 4
    if w_max <= w_min * 4:
        warnings.warn(f"Time series too short (N = {N}) for a meaningful multi-scale "
                      f"phase-fluctuation analysis")
        return out
    windows = np.unique(np.floor(np.logspace(np.log10(w_min), np.log10(w_max), num_windows) + 0.5)
                        ).astype(int)
    if len(windows) < 6:
        return out

    mean_abs_diff = np.array([np.mean(np.abs(dphi[w:] - dphi[:-w])) for w in windows])
    with np.errstate(divide='ignore'):
        log_d = np.log10(mean_abs_diff)
    valid = np.isfinite(log_d) & (mean_abs_diff > 0)
    if np.sum(valid) < 6:
        return out
    log_w = np.log10(windows[valid].astype(float))
    log_d = log_d[valid]

    split = int(np.ceil(len(log_w) / 2))
    short_idx = slice(0, split)
    long_idx = slice(split - 1, None)  # one point of overlap anchors the two fits together

    def fit_slope(x, yv):
        if len(x) < 2:
            return np.nan
        return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), yv, rcond=None)[0][0]

    out['slope_short'] = fit_slope(log_w[short_idx], log_d[short_idx])
    out['slope_long'] = fit_slope(log_w[long_idx], log_d[long_idx])
    out['slope_diff'] = out['slope_short'] - out['slope_long']
    return out


def _bicoherence_grid(y: np.ndarray, step: int, num_seg: int, seg_length: int, half_n: int,
                      win: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    # Segment-averaged squared bicoherence of y at the frequency pairs (pi, pj)
    # (0-based bins of the one-sided spectrum, with pi + pj <= half_n - 1).
    psum = pi + pj
    b_num = np.zeros(len(pi), dtype=complex)  # triple-product sum
    p12 = np.zeros(len(pi))  # sum |X(f1) X(f2)|^2
    p3 = np.zeros(len(pi))  # sum |X(f1+f2)|^2
    for k in range(num_seg):
        seg = y[k * step:k * step + seg_length]
        seg = seg - np.mean(seg)  # demean each segment before windowing
        xh = np.fft.fft(seg * win, seg_length)[:half_n]  # one-sided spectrum, DC to Nyquist
        outer = xh[pi] * xh[pj]
        b_num += outer * np.conj(xh[psum])
        p12 += np.abs(outer) ** 2
        p3 += np.abs(xh[psum]) ** 2
    return np.abs(b_num) ** 2 / (p12 * p3 + np.finfo(float).eps)  # bounded in [0, 1]


@dict_output
def bicoherence(y: ArrayLike, seg_length: int = 64, max_n: Union[int, str] = 'full',
                num_surr: int = 100) -> Union[dict, float]:
    """
    Quadratic phase coupling between frequencies, from the squared bicoherence.

    Estimates the bicoherence, a normalized bispectrum, by segment averaging: the
    series is split into overlapping segments (50% overlap, as many as fit), each
    segment is demeaned, Hamming-windowed and Fourier transformed, and the
    per-segment Fourier coefficients are combined into the bispectrum estimate
    ``B(f1,f2) = <X(f1) X(f2) X*(f1+f2)>``, averaged over segments. The squared
    bicoherence is ``bic2(f1,f2) = |B(f1,f2)|^2 / (<|X(f1)X(f2)|^2> <|X(f1+f2)|^2>)``,
    bounded in [0, 1] by the Cauchy-Schwarz inequality.

    Time-domain nonlinearity statistics (e.g., ``tc3``, the ramping-window asymmetry
    of ``ramping_windows``) collapse all frequency structure into a single number
    per lag, so nonlinear coupling localized to a specific pair of frequency bands
    can average out to near zero. The bicoherence resolves quadratic phase coupling
    per frequency pair, directly detecting whether energy at f1 and f2 is
    phase-coupled to energy at f1+f2 (the frequency-domain signature of a quadratic
    nonlinearity).

    Parameters
    ----------
    y : array-like
        The input time series.
    seg_length : int, optional
        The length (in samples) of each FFT segment (at least 16). Segments overlap
        by 50% and as many as fit are averaged; the number of segments K sets the
        variance of the bicoherence estimate. Fewer than 8 segments gives NaN.
        Default is 64.
    max_n : int or 'full', optional
        The maximum number of samples to consider: longer series are cropped to
        their first ``max_n`` points. ``'full'`` disables cropping. Default is
        ``'full'``.
    num_surr : int, optional
        The number of random-phase surrogates (which preserve the power spectrum but
        destroy phase coupling) used to calibrate the significance threshold
        empirically, in place of its asymptotic approximation. The threshold is the
        95% quantile of squared bicoherence values pooled across all frequency pairs
        and all surrogates (the threshold of a few tens of surrogates depends on which were drawn). Default is 100.

    Returns
    -------
    dict or float
        - ``meanBic``, ``maxBic``, ``stdBic``, ``skewBic``: mean, maximum, standard
          deviation and skewness of the squared bicoherence over the non-redundant
          principal domain of frequency pairs (``0 < f1 <= f2``, ``f1 + f2 <=
          Nyquist``).
        - ``entropy``: the Shannon entropy of the bicoherence surface, normalized
          to [0, 1] by the uniform-distribution entropy: whether coupling is
          concentrated in a few frequency pairs or diffuse across many.
        - ``meanBicDiag``: the mean squared bicoherence on the self-coupling
          diagonal ``f1 = f2`` (quadratic harmonic distortion).
        - ``propSig``: the proportion of frequency pairs exceeding the
          surrogate-calibrated 95% significance threshold.
        - ``threshRatio``: the ratio of that empirical threshold to the standard
          analytic large-K approximation (``K * bic2 ~ Exp(1)`` under the null of a
          linear, ~Gaussian process, giving threshold ``-log(0.05)/K``). A ratio far
          from 1 flags that the asymptotic approximation is untrustworthy for this
          series (e.g., because of non-stationarity).

        NaN is returned if the series is too short for 8 segments.

    Notes
    -----
    The random-phase surrogates are generated with hctsa's portable generator
    (:func:`pyhctsa.robust.bf_random`) at seed 0, as in hctsa's ``SD_MakeSurrogates``
    (``'RP'``), so the output is reproducible.
    """
    from .surrogates import _make_surrogates  # local import: surrogates imports other operations

    y = np.asarray(y, dtype=float).ravel()
    min_seg_length = 16
    if seg_length < min_seg_length:
        raise ValueError(f"seg_length = {seg_length} is too short for a meaningful FFT "
                         f"segment (need >= {min_seg_length})")
    N = len(y)
    if not (isinstance(max_n, str) and max_n == 'full') and N > max_n:
        warnings.warn(f"Time series ({N} samples) exceeds max_n = {max_n}; "
                      f"analyzing the first {max_n} samples")
        y = y[:max_n]
        N = int(max_n)

    # Segment geometry (50% overlap, as many segments as fit), shared by the real
    # series and every surrogate:
    step = seg_length // 2
    min_num_seg = 8  # need enough segments for a meaningful bicoherence estimate
    num_seg = (N - seg_length) // step + 1
    if num_seg < min_num_seg:
        warnings.warn(f"Time series (N = {N}) too short for seg_length = {seg_length} to "
                      f"form >= {min_num_seg} 50%-overlapping segments")
        return np.nan

    half_n = seg_length // 2 + 1  # bin i <-> frequency i/seg_length, up to Nyquist
    win = scipy.signal.windows.hamming(seg_length, sym=True)

    # Non-redundant principal domain (excluding DC): 1 <= i <= j, i + j <= Nyquist bin
    ii, jj = np.meshgrid(np.arange(half_n), np.arange(half_n), indexing='ij')
    mask = (ii + jj <= half_n - 1) & (jj >= ii) & (ii >= 1)
    pi, pj = ii[mask], jj[mask]
    diag = pi == pj  # self-coupling diagonal (f1 = f2)

    # Bicoherence of the real series
    bic = _bicoherence_grid(y, step, num_seg, seg_length, half_n, win, pi, pj)
    if bic.size == 0 or not np.any(np.isfinite(bic)):
        return np.nan

    out = {}
    out['meanBic'] = np.mean(bic)
    out['maxBic'] = np.max(bic)
    out['stdBic'] = np.std(bic, ddof=1)
    out['skewBic'] = scipy.stats.skew(bic)

    # Normalized Shannon entropy of the bicoherence surface (0 = all coupling
    # concentrated in one frequency pair, 1 = uniformly diffuse):
    p = bic[bic > 0]
    p = p / np.sum(p)
    out['entropy'] = -np.sum(p * np.log(p)) / np.log(len(bic))

    out['meanBicDiag'] = np.mean(bic[diag])

    # Surrogate-calibrated significance threshold. Random-phase surrogates preserve
    # the power spectrum (linear structure) but destroy quadratic phase coupling,
    # exactly the null hypothesis a bicoherence significance test needs; the 95%
    # quantile of their pooled bic2 values is the empirical threshold.
    alpha = 0.05
    surrogates = _make_surrogates(y, 'RP', num_surr, random_seed=0)  # hctsa: SD_MakeSurrogates(y,'RP',numSurr,[],'default'), BF_Random seed 0
    null_vals = np.concatenate([
        _bicoherence_grid(surrogates[:, s], step, num_seg, seg_length, half_n, win, pi, pj)
        for s in range(num_surr)])
    null_vals = null_vals[np.isfinite(null_vals)]
    surr_thresh = float(np.quantile(null_vals, 1 - alpha, method='hazen'))
    out['propSig'] = np.mean(bic > surr_thresh)

    # How far the standard asymptotic threshold is from the empirical one:
    analytic_thresh = -np.log(alpha) / num_seg
    out['threshRatio'] = surr_thresh / analytic_thresh
    return out


def _spectrogram_power(y: np.ndarray, window: np.ndarray, noverlap: int) -> tuple:
    """
    One-sided power spectrogram of ``y`` (frequency bins x windows) and bin frequencies.

    Parameters
    ----------
    y : np.ndarray
        1-D series of length N.
    window : np.ndarray
        Window of length L (also the FFT length).
    noverlap : int
        Overlap between consecutive frames, 0 <= noverlap < L.

    Returns
    -------
    P : np.ndarray
        Shape (L//2 + 1, M): ``P[k, m] = c_k |sum_n w[n] y[mH + n] exp(-2 pi i k n / L)|^2
        / (sum w)^2`` with hop ``H = L - noverlap``, M = floor((N - L)/H) + 1 frames
        (only complete frames; no padding or detrending), and ``c_k = 2`` except at DC
        and (for even L) Nyquist, where it is 1.
    fout : np.ndarray
        Bin frequencies ``k / L`` in cycles per sample.

    Notes
    -----
    The one-sided "power spectrum" scaling (``scaling='spectrum'``) of a short-time Fourier
    transform, computed by :func:`scipy.signal.spectrogram` without detrending (so that a
    sinusoid of amplitude A on a bin frequency has peak power ``A**2 / 2``); see Oppenheim &
    Schafer, Discrete-Time Signal Processing.
    """
    y = np.asarray(y, dtype=float)
    window = np.asarray(window, dtype=float)
    L = window.size
    fout, _, P = scipy.signal.spectrogram(y, fs=1.0, window=window, nperseg=L,
                                          noverlap=int(noverlap), nfft=L, detrend=False,
                                          scaling='spectrum', mode='psd')
    return np.ascontiguousarray(P), fout


def _spectral_kurtosis(P: np.ndarray) -> tuple:
    """
    Spectral kurtosis of each frequency bin across windows, and its 95% threshold.

    Parameters
    ----------
    P : np.ndarray
        Power spectrogram, shape (n_bins, M).

    Returns
    -------
    kurt : np.ndarray
        ``(M + 1)/(M - 1) * (mean_m(P^2) / mean_m(P)^2 - 1) - 1`` per bin (NaN for a bin with
        zero mean power, or for M < 2).
    thresh : float
        ``2 * Phi^{-1}(0.975) / sqrt(M)``: the two-sided 95% bound on the spectral
        kurtosis of stationary Gaussian noise.

    Notes
    -----
    Antoni, J. (2006) "The spectral kurtosis: a useful tool for characterising
    non-stationary signals", Mech. Syst. Signal Process. 20(2):282-307, defines the spectral
    kurtosis as the excess kurtosis of the STFT coefficients of each frequency,
    ``<|X|^4>/<|X|^2>^2 - 2`` (zero for Gaussian noise). The ratio of sample averages over M
    windows is biased for small M; the estimator used is the unbiased estimator of
    Nita & Gary (2010, PASP 122:595), ``(M+1)/(M-1) * (M S2/S1^2 - 1)`` (mean 1 for Gaussian
    noise; S1, S2 the sums of |X|^2 and |X|^4 over windows), shifted by -1 to Antoni's
    convention, so that its mean for Gaussian noise is exactly 0 and its variance is 4/M to
    first order. That gives the threshold ``2 * Phi^{-1}(0.975) / sqrt(M)``, a two-sided 95%
    bound.
    """
    P = np.asarray(P, dtype=float)
    M = P.shape[1]
    thresh = float(2.0 * scipy.stats.norm.ppf(0.975) / np.sqrt(M)) if M > 0 else np.nan
    if M < 2:
        return np.full(P.shape[0], np.nan), thresh
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.mean(P ** 2, axis=1) / np.mean(P, axis=1) ** 2
        kurt = (M + 1.0) / (M - 1.0) * (ratio - 1.0) - 1.0
    return kurt, thresh


def _spectral_entropy(P: np.ndarray) -> np.ndarray:
    """
    Normalized spectral entropy of each window of a power spectrogram.

    Parameters
    ----------
    P : np.ndarray
        Power spectrogram, shape (n_bins, M).

    Returns
    -------
    np.ndarray
        Shape (M,): ``-sum_f p_f log2 p_f / log2(n_bins)`` with ``p = P[:, m] / sum_f P[:, m]``
        and ``0 log 0 = 0``; NaN for a window with zero total power.

    Notes
    -----
    Shannon entropy of the normalized spectrum, scaled by the white-noise value
    log2(n_bins) (Pan, Chen & Li 2009, Proc. IMechE C 223(5):1223-1231; Shen, Hung &
    Lee 1998, ICSLP).
    """
    P = np.asarray(P, dtype=float)
    n_bins = P.shape[0]
    tot = P.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = P / tot[None, :]
        plogp = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
        se = -plogp.sum(axis=0) / np.log2(n_bins)
    se = np.where(tot > 0, se, np.nan)
    return se


@dict_output
def spectral_time_freq(y: ArrayLike, num_windows: int = 20) -> Union[dict, float]:
    """
    Time-varying spectral statistics from a spectrogram.

    ``spectral_summaries`` computes statistics from a single, static spectral
    estimate of the whole time series. This function instead divides the series into
    overlapping windows and tracks how the spectral content changes across them:

    - The spectral kurtosis (Antoni 2006) is the kurtosis, across windows, of the
      power in each frequency bin. High values flag a frequency band whose energy is
      concentrated in occasional bursts rather than spread evenly over time (e.g., a
      transient, impulsive fault).
    - The spectral entropy of the power spectrum is computed separately in each
      window, giving one entropy value per window. Variation in this sequence flags
      a time series whose spectral character is not stationary.

    Each window is a Hamming window of ``max(8, round(N/num_windows))`` samples with
    50% overlap, so about ``2*num_windows - 1`` windows result.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_windows : int, optional
        Sets the window length to ``N/num_windows`` samples (at least 8), with 50%
        overlap. If fewer than 4 windows fit, the output is NaN. Default is 20.

    Returns
    -------
    dict or float
        - ``sk_max``, ``sk_mean``, ``sk_std``, ``sk_range``: maximum, mean, standard
          deviation and range, over frequencies, of the spectral kurtosis.
        - ``sk_fracAboveThresh``: the fraction of frequencies whose spectral kurtosis
          exceeds the 95% Gaussian-null threshold (non-Gaussian, bursty behavior).
        - ``sk_freqAtMax``: the angular frequency, in radians per sample (``2*pi``
          times the frequency in cycles per sample, from 0 to pi, matching
          ``spectral_summaries``), at which the spectral kurtosis is largest.
        - ``sk_relSpread``: the relative spread of power across windows: the mean over
          frequencies of the standard deviation across windows of the power in each
          frequency bin, divided by the mean over frequencies of the mean power across
          windows. A dimensionless coefficient of variation of the power, independent
          of the variance of the series and of the window length (about 1 for white
          noise).
        - ``se_mean``, ``se_std``, ``se_max``, ``se_min``, ``se_range``: mean,
          standard deviation, maximum, minimum and range, over windows, of the spectral
          entropy of each window's power spectrum: the Shannon entropy (base 2) of the
          one-sided relative power across frequency bins, divided by its maximum,
          log2 of the number of bins, so each window's value lies in [0, 1]: 1 for a
          flat (white) spectrum, near 0 for a spectrum concentrated in a single
          frequency bin.

    Notes
    -----
    Everything is computed from a one-sided power spectrogram of the series.
    """
    y = np.asarray(y, dtype=float).ravel()
    Ny = len(y)

    # Windows sized as a fraction of the series length, so behavior scales across
    # very different input lengths:
    win_length = max(8, int(np.floor(Ny / num_windows + 0.5)))
    noverlap = int(np.floor(win_length / 2 + 0.5))
    window = scipy.signal.windows.hamming(win_length, sym=True)

    hop = win_length - noverlap
    num_frames = (Ny - win_length) // hop + 1
    if num_frames < 4:
        return np.nan  # too short for across-window statistics to mean anything

    P, fout = _spectrogram_power(y, window, noverlap)  # one-sided power, (frequency bin, window)
    kurt, thresh = _spectral_kurtosis(P)  # spectral kurtosis per frequency bin, and its 95% threshold
    spread = np.std(P, axis=1, ddof=1)
    centroid = np.mean(P, axis=1)

    out = {}
    out['sk_max'] = np.max(kurt)
    out['sk_mean'] = np.mean(kurt)
    out['sk_std'] = np.std(kurt, ddof=1)
    out['sk_range'] = np.max(kurt) - np.min(kurt)
    out['sk_fracAboveThresh'] = np.mean(kurt > thresh)
    out['sk_freqAtMax'] = 2 * np.pi * fout[int(np.argmax(kurt))]
    out['sk_relSpread'] = np.mean(spread) / np.mean(centroid)

    se = _spectral_entropy(P)  # normalized spectral entropy of each window
    out['se_mean'] = np.mean(se)
    out['se_std'] = np.std(se, ddof=1)
    out['se_max'] = np.max(se)
    out['se_min'] = np.min(se)
    out['se_range'] = np.max(se) - np.min(se)
    return out


def _lavi_from_rho(rho: np.ndarray) -> np.ndarray:
    """
    Expected lagged-angle vector index (LAVI) of a Gaussian process, from its lag correlation.

    For a pair of circular complex Gaussian variables with correlation magnitude ``rho``, the mean
    resultant length of their phase difference is
    ``(pi/4) rho 2F1(1/2, 1/2; 2; rho**2) = (E(rho**2) - (1 - rho**2) K(rho**2)) / rho``,
    with ``K`` and ``E`` the complete elliptic integrals of the first and second kind (parameter
    convention ``m = rho**2``). Below ``rho = 1e-6`` the small-``rho`` limit ``pi rho / 4`` is used
    (the closed form is a 0/0 there), and ``rho >= 1`` gives 1.

    Parameters
    ----------
    rho : np.ndarray
        Correlation magnitudes, in [0, 1].

    Returns
    -------
    np.ndarray
        The expected LAVI at each ``rho``.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    out = np.full(rho.shape, np.nan)
    small = rho < 1e-6
    sat = rho >= 1
    mid = ~small & ~sat & ~np.isnan(rho)
    out[small] = np.pi * rho[small] / 4
    out[sat] = 1.0
    m = rho[mid] ** 2
    out[mid] = (scipy.special.ellipe(m) - (1 - m) * scipy.special.ellipk(m)) / rho[mid]
    return out


def _moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average of odd ``width``; the window shrinks to the available samples at the ends."""
    half = width // 2
    n = len(x)
    return np.array([x[max(0, i - half):min(n, i + half + 1)].mean() for i in range(n)])


@dict_output
def rhythmicity(y: ArrayLike, m: float = 5, tau: float = 1.5, n_freq: int = 40,
                f_low: float = 0.02, f_high: float = 0.4, smooth_win: int = 5,
                min_cycles: float = 40) -> Union[dict, float]:
    """
    Phase-persistence ("rhythmicity") spectrum of a time series.

    A power spectrum cannot tell a sustained oscillation apart from a train of brief,
    high-power bursts: both deposit energy at the same frequency. This operation instead asks,
    at each frequency, how predictable the phase is a fixed number of cycles into the future:
    high for an oscillation that keeps ticking (sustained, "rhythmic"), low for one that only
    flickers on intermittently (transient, "bursty").

    The method is the lagged-angle vector index (LAVI) of Karvat et al. (2026), a phase-only
    refinement of the lagged coherence of Fransen et al. (2015):

    1. A complex Morlet wavelet transform at each analysis frequency ``f`` (``m`` cycles per
       wavelet) gives the instantaneous phase ``phi(t, f)``. The transform is an FFT-domain
       convolution.
    2. ``LAVI(f) = |mean_t exp(i [phi(t, f) - phi(t + L, f)])|``, with the lag
       ``L = round(tau / f)`` samples, a fixed ``tau`` cycles. This is the mean resultant length
       of the lagged phase differences, in [0, 1]; it is 1 when the phase advances by a
       perfectly repeatable amount every ``tau`` cycles (a stable rhythm). A coefficient with
       ``|x(t, f)| < 1e-6 max_t |x(t, f)|`` (the maximum over the edge-free samples at that
       frequency; under 1e-12 of the peak power) has no meaningful phase, typically where a spiky
       or skewed series is (nearly) constant across the wavelet's support, down to coefficients
       that are pure round-off. Any pair ``(t, t + L)`` involving one is left out of the mean, so
       that the output does not depend on round-off.
    3. The spectral null. Part of LAVI is set by the power spectrum alone: a Gaussian process
       with the same spectrum (equivalently, the mean over phase-randomized surrogates) has a
       lag-``L`` correlation ``rho(f)`` of its wavelet output, computed here exactly from the
       series' own periodogram, and its expected LAVI is then the mean resultant length of the
       phase difference of a circular complex Gaussian pair,
       ``LAVI0(f) = (pi/4) rho 2F1(1/2, 1/2; 2; rho**2)
       = (E(rho**2) - (1 - rho**2) K(rho**2)) / rho``, with ``K`` and ``E`` the complete
       elliptic integrals. This includes a floor that the method has even for white noise: the
       wavelet overlaps itself across the lag, so ``rho = exp(-pi**2 tau**2 / m**2)`` (0.41
       for the defaults) and ``LAVI0 = 0.33``. The excess, ``LAVI(f) - LAVI0(f)``, is the part
       of the rhythmicity spectrum that the power spectrum does not determine: about zero for
       any linear Gaussian process, positive for a sustained oscillation, negative for bursts.
       This replaces the surrogate "noise ribbon" of Karvat et al., deterministically and at the
       cost of one extra FFT per frequency.

    The amplitude-weighted lagged coherence is deliberately not reported: up to edge effects it
    is the normalized lag-``L`` autocorrelation of the wavelet output, a fixed function of the
    periodogram (exactly ``rho(f)`` above).

    The series is treated as evenly sampled with unit spacing: frequencies are in cycles per
    sample (0 to 0.5) and reported angular frequencies are ``2*pi*f``, as in
    :func:`spectral_summaries`. LAVI is biased upward when few cycles are available, so a
    frequency is used only when the pairs used (free of edge effects, with a defined phase) span
    at least ``min_cycles`` cycles, i.e. number of pairs times ``f`` is at least ``min_cycles``;
    the output is NaN if fewer than 10 frequencies qualify (for the defaults, N below
    about 240).

    Parameters
    ----------
    y : array-like
        The input time series.
    m : float, optional
        Width of the Morlet wavelet, in cycles. Larger values give sharper frequency
        resolution and blunter time resolution. Default is 5, as in Karvat et al.
    tau : float, optional
        The lag at which phase persistence is measured, in cycles. Default is 1.5, as in
        Karvat et al., who report that the relative shape of the rhythmicity spectrum is
        nearly independent of this choice.
    n_freq : int, optional
        Number of log-spaced analysis frequencies. Default is 40.
    f_low : float, optional
        Lowest analysis frequency, in cycles per sample. Default is 0.02.
    f_high : float, optional
        Highest analysis frequency, in cycles per sample; must be below the Nyquist frequency
        of 0.5. Default is 0.4.
    smooth_win : int, optional
        Width, in frequency bins (forced odd), of the moving-average (Daniell) spectral window
        applied before counting rhythmic bands; 1 disables it. Only ``lavi_numPeaks`` is taken
        from the smoothed spectrum; every other statistic uses the raw spectrum. Default is 5.
    min_cycles : float, optional
        The minimum number of cycles of ``f`` that the pairs used must span for ``f`` to be
        used. Default is 40, which keeps the upward bias of LAVI to about 0.02.

    Returns
    -------
    dict or float
        Statistics over the frequencies that qualify. NaN if the series is constant or fewer
        than 10 frequencies qualify. For the rhythmicity spectrum ``LAVI(f)``:

        - ``lavi_mean``, ``lavi_median``, ``lavi_std``, ``lavi_iqr``, ``lavi_max``,
          ``lavi_min``: level and spread of rhythmicity across frequency
          (``lavi_median`` is the baseline).
        - ``lavi_maxOnMedian``: peak rhythmicity relative to the baseline.
        - ``lavi_freqAtMax``, ``lavi_freqAtMin``: angular frequency (``2*pi*f``) of the most
          sustained and most transient band.
        - ``lavi_meanAboveMedian``: mean amount by which above-baseline frequencies exceed the
          baseline.
        - ``lavi_fracSustained``, ``lavi_fracTransient``: fraction of frequencies more than 2
          scaled MAD above / below the baseline.
        - ``lavi_ac1``: lag-1 autocorrelation of the spectrum across frequency.
        - ``lavi_numPeaks``: number of local maxima of the smoothed spectrum that exceed the
          baseline by at least one scaled MAD (distinct rhythmic bands).
        - ``lavi_atPeakPower``: LAVI at the frequency with the most wavelet power.

        For the excess over the spectral null, ``LAVI(f) - LAVI0(f)``:

        - ``ex_mean``, ``ex_std``, ``ex_max``, ``ex_min``: level and spread of the rhythmicity
          that the power spectrum does not account for.
        - ``ex_freqAtMax``, ``ex_freqAtMin``: angular frequency of the largest and smallest
          excess.
        - ``ex_atPeakPower``: the excess at the frequency with the most wavelet power; is the
          dominant oscillation more sustained (> 0) or more bursty (< 0) than its spectrum
          implies?

    Notes
    -----
    Scaled MAD is ``1.4826`` times the median absolute deviation from the median. The wavelet
    at frequency ``f`` is a Gaussian-windowed complex exponential with envelope standard
    deviation ``m / (2 pi f)`` samples, truncated at 3.5 standard deviations, made zero-mean
    and normalized to unit energy.

    For a broadband series with no rhythm, the frequency locations (``*_freqAtMax``) tend to
    fall at the lowest qualifying frequency, where a small residual upward bias of LAVI (about
    0.01 to 0.02 at ``min_cycles = 40``) remains; that frequency depends on N.

    References
    ----------
    Karvat, G., et al. (2026). Universal rhythmic architecture uncovers two modes of neural
    dynamics. Nature Communications, 17.

    Fransen, J. W., et al. (2015). Studying the dynamics of oscillatory bursts with lagged
    coherence. NeuroImage, 111, 158.
    """
    y = np.asarray(y, dtype=float).ravel()
    smooth_win = max(1, int(np.floor(smooth_win + 0.5)))
    if smooth_win % 2 == 0:
        smooth_win += 1  # keep the moving average centered
    if f_high >= 0.5:
        raise ValueError(f"f_high = {f_high:.3f} must be below the Nyquist frequency (0.5)")
    if f_low <= 0 or f_low >= f_high:
        raise ValueError("Need 0 < f_low < f_high")

    n = len(y)
    if y.max() == y.min():
        return np.nan  # a constant series carries no phase information
    y = y - y.mean()

    freqs = np.logspace(np.log10(f_low), np.log10(f_high), n_freq)
    lavi = np.full(n_freq, np.nan)
    rho = np.full(n_freq, np.nan)
    power = np.full(n_freq, np.nan)
    for i, f in enumerate(freqs):
        sigma_t = m / (2 * np.pi * f)  # Gaussian envelope s.d., in samples
        h = int(np.ceil(3.5 * sigma_t))  # wavelet support: +/- 3.5 s.d.
        lag = max(1, int(np.floor(tau / f + 0.5)))  # lag in samples
        n_use = n - 2 * h - lag  # samples free of edge effects, at t and t + lag
        if n_use * f < min_cycles:
            continue  # too few cycles for an unbiased estimate: leave this frequency NaN

        t = np.arange(-h, h + 1)
        psi = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
        psi = psi - psi.mean()  # zero mean (wavelet admissibility)
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))  # unit energy

        # Linear convolution via FFT, then the length-n 'same' segment:
        n_conv = n + psi.size - 1
        Y = scipy.fft.fft(y, n_conv)
        W = scipy.fft.fft(psi, n_conv)
        X = scipy.fft.ifft(Y * W)[h:h + n]

        x1 = X[h:h + n_use]
        x2 = X[h + lag:h + lag + n_use]
        # Coefficients below 1e-6 of the largest edge-free magnitude at this frequency have no
        # meaningful phase (at worst it is round-off): leave out pairs involving one.
        tol = 1e-6 * np.max(np.abs(X[h:n - h]))
        has_phase = (np.abs(x1) >= tol) & (np.abs(x2) >= tol)
        if has_phase.sum() * f < min_cycles:
            continue  # too few cycles with a defined phase
        lavi[i] = np.abs(np.mean(np.exp(1j * (np.angle(x1[has_phase]) - np.angle(x2[has_phase])))))
        power[i] = np.mean(np.abs(x1) ** 2)

        # Spectral null: the lag-correlation of the wavelet output implied by the (zero-padded,
        # so linear) periodogram:
        r = scipy.fft.ifft(np.abs(Y) ** 2 * np.abs(W) ** 2)
        rho[i] = np.abs(r[lag]) / r[0].real
    lavi_null = _lavi_from_rho(rho)

    valid = np.isfinite(lavi) & np.isfinite(lavi_null)
    if valid.sum() < 10:
        return np.nan  # too little of the spectrum is estimable to summarize

    f = freqs[valid]
    S = lavi[valid]
    ex = S - lavi_null[valid]
    i_pow = int(np.argmax(power[valid]))

    # The rhythmicity spectrum LAVI(f):
    med = np.median(S)
    scaled_mad = 1.4826 * np.median(np.abs(S - med))

    out = {}
    out['lavi_mean'] = np.mean(S)
    out['lavi_median'] = med
    out['lavi_std'] = np.std(S, ddof=1)
    q1, q3 = np.quantile(S, [0.25, 0.75], method='hazen')
    out['lavi_iqr'] = q3 - q1
    out['lavi_max'] = np.max(S)
    out['lavi_min'] = np.min(S)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['lavi_maxOnMedian'] = np.max(S) / med
    out['lavi_freqAtMax'] = 2 * np.pi * f[int(np.argmax(S))]
    out['lavi_freqAtMin'] = 2 * np.pi * f[int(np.argmin(S))]

    above = S[S > med] - med
    out['lavi_meanAboveMedian'] = np.mean(above) if above.size else 0.0

    if scaled_mad < np.finfo(float).eps:
        out['lavi_fracSustained'] = 0.0
        out['lavi_fracTransient'] = 0.0
    else:
        out['lavi_fracSustained'] = np.mean(S > med + 2 * scaled_mad)
        out['lavi_fracTransient'] = np.mean(S < med - 2 * scaled_mad)

    # Lag-1 autocorrelation of the spectrum across frequency:
    S0 = S - S.mean()
    denom = np.sum(S0 ** 2)
    out['lavi_ac1'] = np.nan if denom < np.finfo(float).eps else np.sum(S0[:-1] * S0[1:]) / denom

    # Number of rhythmic bands: local maxima of the smoothed spectrum that clear the baseline by
    # at least one scaled MAD (a Daniell, i.e. moving-average, window stands in for Welch-style
    # segment averaging, which the series is too short to afford here):
    S_sm = _moving_mean(S, smooth_win)
    dS = np.diff(S_sm)
    is_peak = np.zeros(len(S_sm), dtype=bool)
    is_peak[1:-1] = (dS[:-1] > 0) & (dS[1:] <= 0)
    out['lavi_numPeaks'] = float(np.sum(is_peak & (S_sm > med + scaled_mad)))
    out['lavi_atPeakPower'] = S[i_pow]

    # The excess over the spectral null:
    out['ex_mean'] = np.mean(ex)
    out['ex_std'] = np.std(ex, ddof=1)
    out['ex_max'] = np.max(ex)
    out['ex_min'] = np.min(ex)
    out['ex_freqAtMax'] = 2 * np.pi * f[int(np.argmax(ex))]
    out['ex_freqAtMin'] = 2 * np.pi * f[int(np.argmin(ex))]
    out['ex_atPeakPower'] = ex[i_pow]
    return out
