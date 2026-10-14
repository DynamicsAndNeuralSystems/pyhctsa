"""Accuracy tests for the spectral, wavelet, pre-processing and change-point operations.

Each test checks an analytic result, a reference library (scipy, PyWavelets) or an invariant.
"""
import numpy as np
import pytest
import pywt
import scipy.signal
from scipy.ndimage import median_filter

from pyhctsa.operations import changepoint as cp
from pyhctsa.operations import pre_process as pp
from pyhctsa.operations import spectral as sp
from pyhctsa.operations import wavelet as wl


def _fgn(n, hurst, rng):
    """Fractional Gaussian noise by circulant embedding (Davies-Harte)."""
    k = np.arange(n + 1)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.fft(row).real
    assert lam.min() > -1e-10
    m = row.size
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(np.maximum(lam, 0) / m) * z).real[:n]


# ------------------------------------------------------------------ index grids
def test_even_split_points_hand_cases():
    assert wl._even_split_points(0, 10, 11).tolist() == list(range(11))
    assert wl._even_split_points(0, 9, 3).tolist() == [0, 5, 9]  # 4.5 rounds up
    assert wl._even_split_points(0, 7, 2).tolist() == [0, 7]
    assert wl._even_split_points(2, 2, 1).tolist() == [2]
    # exact half-integers in the middle of a non-integer range: linspace(1, 40, 25)[4] = 7.5
    pts = wl._even_split_points(1, 40, 25)
    assert pts[4] == 8 and pts[0] == 1 and pts[-1] == 40
    assert np.all(np.diff(pts) > 0)
    # the same 25 points for N = 1000, given as N/25 exactly
    from fractions import Fraction
    assert np.array_equal(wl._even_split_points(1, Fraction(1000, 25), 25), pts)


# ------------------------------------------------------------------ peaks (A7)
def _gaussians(x, centers, heights, width):
    return sum(h * np.exp(-0.5 * ((x - c) / width) ** 2) for c, h in zip(centers, heights))


def test_findpeaks_synthetic_gaussians():
    x = np.arange(400.0)
    centers = [40, 110, 180, 300]
    heights = [3.0, 5.0, 1.5, 4.0]
    s = _gaussians(x, centers, heights, 6.0)
    h, loc = sp._findpeaks(s)
    assert loc.tolist() == centers and loc.dtype == np.int64
    np.testing.assert_allclose(h, s[loc])
    h, loc = sp._findpeaks(s, 0, 'descend')
    assert loc.tolist() == [110, 300, 40, 180]
    assert np.all(np.diff(h) <= 0)
    h, loc = sp._findpeaks(s, 0, 'ascend')
    assert loc.tolist() == [180, 40, 300, 110]


def test_findpeaks_minimum_separation_is_inclusive():
    s = np.array([0, 1, 0, 2, 0, 3, 0.0])
    assert sp._findpeaks(s, 2)[1].tolist() == [1, 5]  # peaks exactly d = 2 apart: the lower goes
    assert sp._findpeaks(s, 1)[1].tolist() == [1, 3, 5]
    assert sp._findpeaks(s, 2, 'descend')[1].tolist() == [5, 1]
    assert sp._findpeaks(s, 6)[1].tolist() == [5]
    # surviving peaks are always more than d apart, and the tallest is kept
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500)
    for d in (3, 10):
        _, loc = sp._findpeaks(y, d, 'none')
        assert np.all(np.diff(loc) > d)
        assert np.argmax(y[1:-1]) + 1 in loc


def test_findpeaks_equals_scipy_and_edge_cases():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(300))
    h, loc = sp._findpeaks(y, 0)
    ref, _ = scipy.signal.find_peaks(y)
    assert np.array_equal(loc, ref)
    assert sp._findpeaks([0.0, 0, 0, 0])[0].size == 0
    assert sp._findpeaks([1, 2, 1], 0)[1].tolist() == [1]
    with pytest.raises(ValueError):
        sp._findpeaks([0, 1, 0], 0, 'sideways')
    with pytest.raises(ValueError):
        sp._findpeaks([0, 1, 0], -1)


# ------------------------------------------------------------------ spectral summaries (A17)
def test_weighted_peak_location_is_the_peak_frequency():
    rng = np.random.default_rng(2)
    n = 4000
    for f0 in (0.08, 0.2, 0.31):
        y = np.sin(2 * np.pi * f0 * np.arange(n)) + 1e-6 * rng.standard_normal(n)
        y = (y - y.mean()) / y.std(ddof=1)
        out = sp.spectral_summaries(y, 'welch', 'hamming')
        win = max(min(256, int(np.floor(n / 4 + 0.5))), 16)
        dw = 2 * np.pi / sp._padded_nfft(win)
        assert abs(out['maxw'] - 2 * np.pi * f0) <= dw
        # the prominence-weighted mean peak location is dominated by the sinusoid's peak
        assert abs(out['w_weighted_peak_prom'] - 2 * np.pi * f0) <= dw


def test_padded_nfft():
    assert sp._MIN_NFFT == 256
    assert sp._padded_nfft(10) == 256
    assert sp._padded_nfft(256) == 256
    assert sp._padded_nfft(257) == 512
    assert sp._padded_nfft(1000) == 1024


def test_spectral_summaries_white_noise_and_rect_none_agree():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(1024)
    y = (y - y.mean()) / y.std(ddof=1)
    a = sp.spectral_summaries(y, 'welch', 'none')
    b = sp.spectral_summaries(y, 'welch', 'rect')
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k
    out = sp.spectral_summaries(y, 'fft')
    # white noise: half the power lies below the middle of the frequency range
    assert out['wq_50'] == pytest.approx(np.pi / 2, abs=0.25)
    assert all(np.isnan(v) for v in sp.spectral_summaries(np.ones(50), 'fft').values())


# ------------------------------------------------------------------ bounded least squares (A18)
def test_bounded_lsq_recovers_noiseless_gaussian():
    x = np.linspace(-1.5, -0.2, 120)
    true = np.array([1.3, -0.8, 0.12])
    model = lambda p, x: p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
    p = sp._bounded_lsq(model, [1.0, -0.7, 0.26], x, model(true, x),
                        lower=[0, x.min(), 0.02], upper=[np.inf, x.max(), 0.5])
    np.testing.assert_allclose(p, true, atol=1e-6)


def test_bounded_lsq_respects_bounds_and_recovers_knee_model():
    f = np.linspace(0.02, 0.5, 200)
    model = lambda p, x: p[0] - np.log10(p[1] + x ** p[2])
    true = np.array([0.5, 0.01, 2.0])
    p = sp._bounded_lsq(model, [0.0, 1e-3, 1.0], f, model(true, f),
                        lower=[-np.inf, 0, 0], upper=[np.inf, np.inf, 10])
    np.testing.assert_allclose(p, true, atol=1e-5)
    q = sp._bounded_lsq(lambda p, x: p[0] * x, [0.5], f, 3.0 * f, lower=[0], upper=[1.0])
    assert q[0] == pytest.approx(1.0)


def test_specparam_recovers_exponent_of_power_law_noise():
    rng = np.random.default_rng(4)
    n = 8192
    freqs = np.fft.rfftfreq(n)[1:]
    spec = np.concatenate([[0], freqs ** -0.5])  # amplitude ~ f^-1/2: power ~ 1/f
    z = spec * (rng.standard_normal(spec.size) + 1j * rng.standard_normal(spec.size))
    y = np.fft.irfft(z, n)
    out = sp.specparam(y / y.std())
    assert out['apExponent'] == pytest.approx(1.0, abs=0.25)
    assert out['numPeaks'] == 0


# ------------------------------------------------------------------ Fourier series fit (A19)
@pytest.mark.parametrize("model, harmonics", [('fourier1', 1), ('fourier2', 2), ('fourier3', 3)])
def test_fourier_fit_recovers_periodic_signal(model, harmonics):
    n = 400
    t = np.arange(1, n + 1)
    w = 2 * np.pi * 0.0731
    y = 0.4 + sum((1.0 / h) * np.sin(h * w * t + 0.3 * h) for h in range(1, harmonics + 1))
    out = sp.sinusoid_fit(y, model)
    assert out['r2'] > 1 - 1e-9
    assert out['rmse'] < 1e-4


def test_fourier_start_point_is_the_periodogram_peak():
    n = 256
    t = np.arange(1.0, n + 1)
    k = 17
    y = np.cos(2 * np.pi * k * t / n + 0.4)
    w0 = sp._fourier_start_point(t, y, 1)
    assert w0 == pytest.approx(2 * np.pi * k / n, rel=1e-12)
    # a harmonic of the fundamental is recognised: the start point is the fundamental
    y2 = np.cos(2 * np.pi * 5 * t / n) + 0.3 * np.cos(2 * np.pi * 10 * t / n + 1.0) + 0.9 * np.cos(2 * np.pi * 15 * t / n)
    w2 = sp._fourier_start_point(t, y2, 3)
    assert w2 == pytest.approx(2 * np.pi * 5 / n, rel=1e-12)
    assert sp._fourier_start_point(t, np.ones(n), 2) > 0


def test_fourier_terms_layout():
    t = np.arange(1.0, 21.0)
    X = sp._fourier_terms(t, 0.3, 3)
    assert X.shape == (20, 7) and np.all(X[:, 0] == 1)
    for i in (1, 2, 3):
        np.testing.assert_allclose(X[:, 2 * i - 1], np.cos(i * 0.3 * t))
        np.testing.assert_allclose(X[:, 2 * i], np.sin(i * 0.3 * t))


# ------------------------------------------------------------------ spectrogram (A20), spectral kurtosis
def test_spectrogram_power_equals_scipy_and_sinusoid_power():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(1000)
    w = scipy.signal.windows.hamming(64, sym=True)
    P, f = sp._spectrogram_power(y, w, 32)
    f_ref, _, P_ref = scipy.signal.spectrogram(y, 1.0, window=w, nperseg=64, noverlap=32, nfft=64,
                                               detrend=False, scaling='spectrum', mode='psd')
    np.testing.assert_array_equal(P, P_ref)
    np.testing.assert_array_equal(f, f_ref)
    assert P.shape[0] == 33
    # a unit-amplitude cosine on a bin frequency has power A^2/2 in that bin
    t = np.arange(1000)
    c = np.cos(2 * np.pi * (8 / 64) * t)
    Pc, fc = sp._spectrogram_power(c, np.ones(64), 0)
    assert np.allclose(Pc[8], 0.5) and np.allclose(np.delete(Pc, 8, axis=0), 0, atol=1e-12)


def test_spectral_kurtosis_is_zero_for_gaussian_noise():
    rng = np.random.default_rng(6)
    for M in (6, 20):
        X = rng.standard_normal((200000, M)) + 1j * rng.standard_normal((200000, M))
        P = (np.abs(X) ** 2).T  # (M windows, many bins) -> transpose to (bins, windows)
        kurt, thresh = sp._spectral_kurtosis(P.T)
        se = kurt.std() / np.sqrt(kurt.size)
        assert abs(kurt.mean()) < 4 * se  # unbiased
        assert kurt.std() == pytest.approx(2 / np.sqrt(M), rel=0.3)  # null sd 2/sqrt(M) (to first order)
        assert thresh == pytest.approx(2 * 1.959964 / np.sqrt(M), rel=1e-5)
    # a bursty bin has large kurtosis
    P = np.ones((1, 40)); P[0, 3] = 100.0
    assert sp._spectral_kurtosis(P)[0][0] > 10


def test_spectral_entropy_extremes():
    P = np.ones((16, 5))
    np.testing.assert_allclose(sp._spectral_entropy(P), 1.0)
    P = np.zeros((16, 3)); P[2, :] = 1.0
    np.testing.assert_allclose(sp._spectral_entropy(P), 0.0, atol=1e-15)
    P[:, 1] = 0
    assert np.isnan(sp._spectral_entropy(P)[1])


def test_spectral_time_freq_white_noise():
    y = np.random.default_rng(7).standard_normal(4000)
    out = sp.spectral_time_freq(y)
    assert 0.9 < out['se_mean'] <= 1.0
    assert abs(out['sk_mean']) < 0.5
    assert out['sk_relSpread'] == pytest.approx(1.0, abs=0.15)


# ------------------------------------------------------------------ continuous wavelet (A21, A22)
def test_cwt_morlet_power_peaks_at_the_scale_of_a_sinusoid():
    n = 2000
    f = 0.05  # cycles per sample
    y = np.sin(2 * np.pi * f * np.arange(n))
    scales = np.arange(1, 33)
    coefs, _ = pywt.cwt(y, scales, 'morl')
    freqs = pywt.scale2frequency('morl', scales)
    best = scales[np.argmax(np.mean(coefs[:, 200:-200] ** 2, axis=1))]
    assert abs(freqs[best - 1] - f) < 0.005
    # and the operation uses the same transform: stats depend only on |c| except meanC
    out = wl.cwt(y, 'morl')
    out_neg = wl.cwt(-y, 'morl')
    for key in out:
        if key == 'meanC':
            assert out_neg[key] == pytest.approx(-out[key], abs=1e-12)
        else:
            assert out_neg[key] == pytest.approx(out[key], rel=1e-9, abs=1e-12)


def test_cwt_discrete_wavelet_runs_and_is_sign_symmetric():
    y = np.random.default_rng(8).standard_normal(500)
    for name in ('db3', 'sym2'):
        a, b = wl.cwt(y, name), wl.cwt(-y, name)
        assert a['meanabsC'] == pytest.approx(b['meanabsC'], rel=1e-12)
        assert a['meanC'] == pytest.approx(-b['meanC'], abs=1e-12)
        assert a['pover99'] == pytest.approx(b['pover99'], rel=1e-9)
    assert not hasattr(wl, '_flip')


# ------------------------------------------------------------------ fBm estimators (A23)
@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_wfbmesti_recovers_hurst_index(hurst):
    rng = np.random.default_rng(100 + int(10 * hurst))
    est = np.array([wl._wfbmesti(np.cumsum(_fgn(4096, hurst, rng))) for _ in range(20)])
    mean = est.mean(axis=0)
    np.testing.assert_allclose(mean[:2], hurst, atol=0.05)
    # the Haar regression has only one vanishing moment and is biased downward at small H
    assert abs(mean[2] - hurst) < 0.07
    out = wl.wfbm(_fgn(4096, hurst, rng))
    assert set(out) == {'H_deriv2', 'H_deriv2Wavelet'}


def test_wfbmesti_degenerate_paths():
    assert np.all(np.isnan(wl._wfbmesti(np.arange(100.0))[:2]))  # straight line: no second differences
    assert np.all(np.isnan(wl._wfbmesti([1.0])))
    assert np.all(np.isnan(wl._wfbmesti([1.0, np.nan, 2.0])))
    out = wl.wfbm(np.ones(100))
    assert np.isnan(out['H_deriv2']) and np.isnan(out['H_deriv2Wavelet'])


def test_wfbmesti_second_difference_uses_all_samples():
    # the quadratic path t^2 has constant second difference 2 a^2 at dilation a: V2/V1 = 16
    p = np.arange(64.0) ** 2
    est = wl._wfbmesti(p)
    assert est[0] == pytest.approx(2.0, abs=1e-12)


# ------------------------------------------------------------------ discrete wavelet bookkeeping (A25)
@pytest.mark.parametrize("name, level, n", [('db3', 4, 500), ('sym2', 3, 257), ('haar', 5, 200), ('db3', 1, 64)])
def test_detail_bands_reconstruct_the_signal(name, level, n):
    x = np.random.default_rng(9).standard_normal(n)
    coeffs = wl._wavedec(x, name, level=level)
    assert len(coeffs) == level + 1
    total = np.zeros(n)
    for k in range(1, level + 1):
        d = wl._detcoef(coeffs, k)
        assert np.array_equal(d, coeffs[level - k + 1])
        r = wl._wrcoef(coeffs, name, k, n)
        assert r.shape == (n,)
        total += r
    approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], name, mode='symmetric')[:n]
    np.testing.assert_allclose(total + approx, x, atol=1e-10)


def test_wavedec_level_errors():
    coeffs = wl._wavedec(np.arange(32.0), 'db3', level=2)
    with pytest.raises(ValueError):
        wl._detcoef(coeffs, 3)
    with pytest.raises(ValueError):
        wl._wrcoef(coeffs, 'db3', 0, 32)


def test_scal_2_freq_pseudo_frequency():
    n = 512
    y = np.sin(2 * np.pi * np.arange(n) / 32)
    for delta in (1, 2):
        out = wl.scal_2_freq(y, 'db3', 5, delta)
        f = pywt.scale2frequency('db3', 2 ** out['lmax']) / delta
        assert out['pf'] == pytest.approx(f)
        assert out['period'] == pytest.approx(1 / f)
    assert wl.scal_2_freq(y, 'db3', 5, 1)['lmax'] in (4, 5)  # period 32 lies between the scales of levels 4 and 5


def test_dwt_coeff_noise_estimate():
    x = np.random.default_rng(10).standard_normal(20000)
    out = wl.dwt_coeff(x, 'db3', 2)
    assert out['noisestd_l1'] == pytest.approx(1.0, rel=0.03)
    assert out['stdd_l1'] == pytest.approx(1.0, rel=0.03)


# ------------------------------------------------------------------ MODWT variance (A24)
def test_modwt_variance_of_white_noise():
    rng = np.random.default_rng(11)
    N, s2 = 8192, 2.5
    acc = []
    for _ in range(10):
        x = np.sqrt(s2) * rng.standard_normal(N)
        W = wl._modwt(x, 'haar', 5)
        acc.append(wl._modwt_var(W, 'haar'))
    v = np.mean(acc, axis=0)
    assert v.shape == (6,)
    # wavelet variance of white noise at level j: s2 / 2^j
    np.testing.assert_allclose(v[:5], s2 / 2.0 ** np.arange(1, 6), rtol=0.04)
    # all levels plus the scaling band account for the variance of the series
    assert v.sum() == pytest.approx(s2, rel=0.04)


def test_modwt_variance_wavelet_levels_are_not_centered():
    N = 256
    W = np.zeros((3, N))
    W[0] = 2.0  # constant detail band: mean square 4 (a centred estimate would give 0)
    W[1] = 1.0
    W[2] = 5.0  # scaling band carries a mean: variance 0
    v = wl._modwt_var(W, 'haar')
    np.testing.assert_allclose(v, [4.0, 1.0, 0.0])


def test_modwt_var_feature_on_a_trend_and_noise():
    rng = np.random.default_rng(12)
    t = np.arange(2000)
    y = np.sin(2 * np.pi * t / 500) + 0.1 * rng.standard_normal(2000)
    out = wl.modwt_var(y, 'db3', 5)
    assert out['scalingFrac'] > 0.8
    assert out['domlevel'] == 6


def test_modwt_is_energy_preserving():
    x = np.random.default_rng(13).standard_normal(300)
    for name in ('db3', 'sym2', 'haar'):
        W = wl._modwt(x, name, 4)
        assert np.sum(W ** 2) == pytest.approx(np.sum(x ** 2), rel=1e-6)


# ------------------------------------------------------------------ spline detrending (A27), median filter
def test_spline_knots_equal_width_pieces():
    knots, npieces = pp._spline_knots(101, 4, 4)
    assert npieces == 4
    np.testing.assert_allclose(knots, [0, 0, 0, 0, 25, 50, 75, 100, 100, 100, 100])
    knots, npieces = pp._spline_knots(10, 20, 4)
    assert npieces == 7  # clamped to N - order + 1


def test_spline_detrend_reproduces_cubic_polynomials_exactly():
    x = np.arange(200, dtype=float)
    y = 0.3 - 0.02 * x + 1e-3 * x ** 2 - 4e-6 * x ** 3
    for npieces in (1, 2, 5):
        assert np.max(np.abs(pp._spline_detrend(y, npieces, 4))) < 1e-9
    # a spline on the same knots is reproduced exactly
    from scipy.interpolate import BSpline
    knots, _ = pp._spline_knots(200, 3, 4)
    coef = np.random.default_rng(14).standard_normal(len(knots) - 4)
    spl = BSpline(knots, coef, 3)(x)
    assert np.max(np.abs(pp._spline_detrend(spl, 3, 4))) < 1e-9
    # noise is left in the residual and the fit is linear
    rng = np.random.default_rng(15)
    e = rng.standard_normal(200)
    r = pp._spline_detrend(y + e, 2, 4)
    np.testing.assert_allclose(r, pp._spline_detrend(e, 2, 4), atol=1e-9)
    np.testing.assert_allclose(pp._spline_detrend(2.5 * e + 7, 2, 4), 2.5 * pp._spline_detrend(e, 2, 4), atol=1e-9)
    # one piece of order 4 is a cubic polynomial fit
    poly = e - np.polyval(np.polyfit(x, e, 3), x)
    np.testing.assert_allclose(pp._spline_detrend(e, 1, 4), poly, atol=1e-8)
    with pytest.raises(ValueError):
        pp._spline_detrend(e, 2, 1)
    # too many pieces is clamped (interpolating fit)
    assert np.max(np.abs(pp._spline_detrend(e[:10], 20, 4))) < 1e-8


def test_preproc_compare_spline_removes_a_smooth_trend():
    rng = np.random.default_rng(16)
    t = np.arange(1000)
    noise = rng.standard_normal(1000)
    y = noise + 0.01 * t + 5 * np.sin(2 * np.pi * t / 2000)
    out = pp.preproc_compare(y, 'spline24')
    assert out['statav2'] < 0  # the non-stationarity of the mean drops after detrending


def test_median_filter_alignment_and_ends():
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    assert pp._med_filt_1d(x, 3).tolist() == [1.0, 2.0, 5.0, 3.0, 3.0]  # ends are not pulled to zero
    assert pp._med_filt_1d(x, 1).tolist() == x.tolist()
    rng = np.random.default_rng(17)
    y = rng.standard_normal(100)
    for k in (3, 5, 9):
        np.testing.assert_allclose(pp._med_filt_1d(y, k), median_filter(y, size=k, mode='nearest'))
    # even window: window of k samples, mean of the two middle values
    got = pp._med_filt_1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4)
    assert got.tolist() == [1.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    assert pp._med_filt_1d([], 3).size == 0
    for bad in (0, -1):
        with pytest.raises(ValueError):
            pp._med_filt_1d([1.0, 2.0], bad)
    with pytest.raises(ValueError):
        pp._med_filt_1d(np.ones((2, 2)), 3)


def test_piecewise_detrend_boundaries_are_even():
    y = np.arange(101.0)  # 101 samples in 4 pieces: boundaries 0, 25, 51, 76, 101 (halves round up)
    r = pp._piecewise_poly_residual(y, 1, 4)
    assert np.max(np.abs(r)) < 1e-9  # a line is removed exactly in every piece


# ------------------------------------------------------------------ change points
def test_stepdetect_on_a_stepped_signal():
    rng = np.random.default_rng(18)
    y = np.concatenate([np.zeros(100), 3 * np.ones(60), -2 * np.ones(40)]) + 0.1 * rng.standard_normal(200)
    out = cp.stepdetect(y, 'l1pwc', 0.5)  # lambda below 1: a fraction of lambda_max
    assert out['nsegments'] * 200 == pytest.approx(3, abs=1.5)
    assert out['rmsoff'] > 0.3
    # a pure constant has one segment of full length
    flat = cp.stepdetect(np.ones(100) + 1e-12 * np.arange(100), 'l1pwc', 0.9)
    assert flat['nsegments'] == pytest.approx(1 / 100)
    assert flat['cvstepint'] == 0.0


def test_stepdetect_change_point_counts_by_half(monkeypatch):
    # a fitted staircase with runs starting at 0, 50, 100, 150 (given directly)
    stairs = np.concatenate([np.zeros(50), np.ones(50), 2 * np.ones(50), 3 * np.ones(50)])
    monkeypatch.setattr(cp, 'l1pwc', lambda y, lam, flag: (stairs, np.array([1.0]), np.array([1.0]), 10.0))
    out = cp.stepdetect(stairs + 0.01, 'l1pwc', 5.0)
    assert out['nsegments'] == pytest.approx(4 / 200)
    # change points (excluding the start of the first run): one in the first half (index 50), two in the second
    assert out['ratn12'] == pytest.approx(1 / 2)
    assert out['diffn12'] == pytest.approx(abs(1 - 2) / 4)
    assert out['pshort_3'] == 0
    assert out['meanstepintgt3'] == 50 and out['medianstepint'] == 50 and out['cvstepint'] == 0
    assert out['lambdamax'] == pytest.approx(10.0 / np.sqrt(200))


def test_variance_segmentation_finds_the_change_and_respects_minimum_length():
    rng = np.random.default_rng(19)
    y = np.concatenate([rng.standard_normal(200), 6 * rng.standard_normal(200)])
    J, t = cp._variance_segmentations(y, 3, 10)
    assert np.all(np.diff(J[np.isfinite(J)]) <= 1e-9)  # more segments never fit worse
    assert abs(t[1][0] - 200) <= 15 and t[1][1] == 400
    for k in range(4):
        ends = np.concatenate([[0], t[k][:k + 1]])
        assert np.all(np.diff(ends) >= 10)
    assert cp._wvarchg(y, 5, 10) >= 1
    # a single variance step is found as one change point in most realizations
    counts = [cp._wvarchg(np.concatenate([rng.standard_normal(200), 6 * rng.standard_normal(200)]), 5, 10)
              for _ in range(30)]
    assert np.mean(np.array(counts) == 1) >= 0.7
    assert cp._wvarchg(np.zeros(5), 3, 10) == 0


def test_wavelet_var_chg_detects_a_variance_change():
    rng = np.random.default_rng(20)
    y = np.concatenate([rng.standard_normal(500), 5 * rng.standard_normal(500)])
    assert cp.wavelet_var_chg(y, 'db3', 2, 5, 0.02) >= 1
    assert cp.wavelet_var_chg(rng.standard_normal(1000), 'db3', 2, 5, 0.02) <= 1
    assert np.isnan(cp.wavelet_var_chg(rng.standard_normal(64), 'db3', 10, 5, 0.02))


def _null_series(kind, n, rng):
    if kind == 'white':
        return rng.standard_normal(n)
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -0.9], rng.standard_normal(n + 200))[200:]


@pytest.mark.parametrize('kind', ['white', 'ar09'])
def test_wavelet_var_chg_false_positive_rate_on_stationary_noise(kind):
    # Calibrated to report at least one change point in about 5% of stationary series
    # (see cleanroom/wvarchg_calibration/report.md); a small simulation has to stay well below
    # the ~75% rate of the earlier Lavielle (2005) S = 0.75 rule.
    rng = np.random.default_rng(2026)
    n_rep = 50
    hits = 0
    for _ in range(n_rep):
        y = _null_series(kind, 500, rng)
        y = (y - y.mean()) / y.std()
        hits += cp.wavelet_var_chg(y, 'db3', 3, 10, 0.01) >= 1
    assert hits / n_rep <= 0.16  # binomial(50, 0.05): P(> 8 hits) < 1%


def test_wavelet_var_chg_recovers_a_large_variance_change():
    rng = np.random.default_rng(2027)
    found = 0
    for _ in range(20):
        y = np.concatenate([rng.standard_normal(500), 3 * rng.standard_normal(500)])  # variance ratio 9
        found += cp.wavelet_var_chg(y, 'db3', 2, 10, 0.01) >= 1
    assert found >= 17


def test_wvarchg_counts_and_locates_known_changes():
    rng = np.random.default_rng(2028)
    counts, errs = [], []
    for _ in range(20):
        y = np.concatenate([rng.standard_normal(400), 3 * rng.standard_normal(400), rng.standard_normal(400)])
        k = cp._wvarchg(y, 10, 12)
        counts.append(k)
        if k == 2:
            J, t = cp._variance_segmentations(y, 2, 12)
            errs.append(np.max(np.abs(t[2][:2] - np.array([400, 800]))))
    assert np.mean(np.array(counts) == 2) >= 0.8
    assert np.median(errs) <= 40


def test_variance_inflation_reflects_dependence():
    rng = np.random.default_rng(2029)
    white = rng.standard_normal(4000)
    assert cp._variance_inflation(white) < 1.6
    from scipy.signal import lfilter
    ar = lfilter([1.0], [1.0, -0.9], rng.standard_normal(4200))[200:]
    assert cp._variance_inflation(ar) > 3.0
    assert cp._variance_inflation(np.zeros(100)) == 1.0


def test_penalized_rule_is_zero_when_contrast_does_not_improve():
    J = np.array([100.0, 99.9, 99.8, np.inf])
    t = np.zeros((4, 4), dtype=int)
    t[0, :1] = [100]
    t[1, :2] = [50, 100]
    t[2, :3] = [30, 60, 100]
    assert cp._num_changes_penalized(J, t, 100, 1.0) == 0


def _brute_force_contrast(x, K, d):
    """Minimal total contrast with k = 0..K change points by exhaustive enumeration."""
    import itertools
    x = x - x.mean()
    N = x.size
    best = np.full(K + 1, np.inf)
    for k in range(K + 1):
        for cuts in itertools.combinations(range(d, N - d + 1), k):
            ends = list(cuts) + [N]
            starts = [0] + list(cuts)
            if any(e - s_ < d for s_, e in zip(starts, ends)):
                continue
            total = 0.0
            for s_, e in zip(starts, ends):
                seg = x[s_:e]
                total += seg.size * np.log(np.sum(seg ** 2) / seg.size)
            best[k] = min(best[k], total)
    return best


def test_variance_segmentation_matches_exhaustive_search():
    rng = np.random.default_rng(21)
    for _ in range(3):
        y = np.concatenate([rng.standard_normal(14), 4 * rng.standard_normal(11)])
        J, t = cp._variance_segmentations(y, 2, 4)
        np.testing.assert_allclose(J, _brute_force_contrast(y, 2, 4), rtol=1e-10)


def test_preproc_compare_medianf3_is_shift_and_scale_invariant():
    # the running median commutes with shifts and scalings, and both series are z-scored, so the
    # change statistics do not depend on the mean or scale of the input (zero-padding the ends
    # would make them depend on the distance of the mean from zero)
    rng = np.random.default_rng(22)
    y = np.cumsum(rng.standard_normal(600)) * 0.2 + rng.standard_normal(600)
    a = pp.preproc_compare(y, 'medianf3')
    b = pp.preproc_compare(3.0 * y + 1000.0, 'medianf3')
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-8, abs=1e-8), k
