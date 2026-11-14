"""Features on inputs where a naive formula degenerates: real-valued spectrogram bins, repeated
values, statistics that are zero up to round-off, and fits with no dynamics."""
import numpy as np
import pytest

from pyhctsa.operations import correlation, model_fit, spectral, stationarity, surrogates
from pyhctsa.utils import z_score


# --- spectral_time_freq: the spectral-kurtosis null holds only for circular complex coefficients --

@pytest.mark.parametrize("N", [1000, 1010, 5000])  # window lengths 50, 51 (odd: last bin half a bin below Nyquist), 250
def test_spectral_kurtosis_of_white_noise_is_not_drawn_to_zero_or_nyquist(N):
    rng = np.random.default_rng(11)
    at_edge, frac_above, n_rep = 0, [], 150
    for _ in range(n_rep):
        out = spectral.spectral_time_freq(z_score(rng.standard_normal(N)))
        f = out['sk_freqAtMax']
        at_edge += (f == 0) or (f > np.pi * (1 - 1.5 / (N // 20)))  # zero, or the last bin
        frac_above.append(out['sk_fracAboveThresh'])
    # the first and last bins have real (or nearly real) coefficients, with spectral kurtosis up to
    # 1, not 0, under the null: scored against the complex-bin null they were the maximum in most
    # realizations
    assert at_edge == 0
    assert 0 < out['sk_freqAtMax'] < np.pi
    # one-sided exceedance of a two-sided 95% bound: a few percent, not the ~8% the two real bins gave
    assert 0.02 < np.mean(frac_above) < 0.05


def test_spectral_kurtosis_still_finds_a_bursty_band():
    rng = np.random.default_rng(5)
    N, period = 4000, 10
    y = rng.standard_normal(N)
    burst = slice(1800, 2000)
    y[burst] += 8 * np.sin(2 * np.pi * np.arange(200) / period)
    out = spectral.spectral_time_freq(z_score(y))
    assert out['sk_freqAtMax'] == pytest.approx(2 * np.pi / period, abs=0.1)
    assert out['sk_max'] > 5


# --- embed2: angles between coincident successive points are undefined, not contagious -----------

def test_embed2_angle_statistics_are_defined_for_series_with_repeated_values():
    rng = np.random.default_rng(2)
    y = z_score(rng.poisson(2, 2000).astype(float))  # many ties, so many 0/0 angles
    out = correlation.embed2(y, 'tau')
    for key in ('theta_ac1', 'theta_ac2', 'theta_ac3', 'theta_mean', 'theta_std'):
        assert np.isfinite(out[key]), key
    assert abs(out['theta_ac1']) < 0.2  # independent counts: no structure in the angle sequence
    assert 0 < out['theta_std'] < np.pi / 2


def test_embed2_angle_statistics_unchanged_without_ties():
    rng = np.random.default_rng(4)
    y = z_score(np.cumsum(rng.standard_normal(500)))
    tau = 3
    m = np.column_stack((y[:-tau], y[tau:]))
    theta = np.arctan(np.diff(m[:, 1]) / np.diff(m[:, 0]))
    out = correlation.embed2(y, tau)
    assert out['theta_mean'] == pytest.approx(np.mean(theta), rel=1e-12)
    assert out['theta_std'] == pytest.approx(np.std(theta, ddof=1), rel=1e-12)


# --- local_global: a ratio to a global statistic that is zero up to round-off is NaN --------------

def test_local_global_skewness_ratio_is_nan_for_a_balanced_two_valued_series():
    rng = np.random.default_rng(8)
    y = z_score(rng.permutation(np.repeat([0.0, 1.0], 500)))  # skewness is exactly 0, 1e-17 computed
    for subset_how, n in (('l', 100), ('p', 0.1), ('unicg', 100)):
        out = stationarity.local_global(y, subset_how, n)
        assert np.isnan(out['skewness']), (subset_how, out['skewness'])
        assert np.isfinite(out['kurtosis'])  # (Pearson kurtosis is at least 1: its ratio is always defined)


def test_local_global_ratios_unchanged_for_ordinary_series():
    from scipy.stats import skew
    rng = np.random.default_rng(9)
    y = z_score(rng.gamma(2.0, size=1000))
    out = stationarity.local_global(y, 'l', 100)
    assert out['skewness'] == pytest.approx(abs(1 - skew(y[:100]) / skew(y)), rel=1e-12)
    assert np.isfinite(out['ac1'])


# --- surrogates: a spread of surrogate values at rounding level is not a spread ------------------

def test_surrogate_distance_in_iqrs_is_nan_when_the_iqr_is_rounding_error():
    base = 0.3
    surr = np.full(50, base)
    surr[::2] += 2e-16  # an 'interquartile range' of one unit in the last place
    surr[:3] = [0.1, 0.5, 0.9]  # a few real outliers, so the standard deviation is not tiny
    out = surrogates._sd_give_me_stats(0.7, surr, 'both')
    assert np.isnan(out['mediqr'])
    assert np.isfinite(out['zscore'])
    spread = np.linspace(0.2, 0.4, 50)
    assert surrogates._sd_give_me_stats(0.7, spread, 'both')['mediqr'] == pytest.approx(
        abs(0.7 - np.median(spread)) / (np.percentile(spread, 75) - np.percentile(spread, 25)), rel=0.1)


@pytest.mark.parametrize("surr_method", [1, 2, 3])
def test_surrogate_statistics_of_a_balanced_binary_series_have_no_astronomical_values(surr_method):
    rng = np.random.default_rng(8)
    y = z_score(rng.permutation(np.repeat([0.0, 1.0], 500)))
    out = surrogates.surrogates(y, tau=1, nsurr=50, surr_method=surr_method, surrfn='trev')
    for key in ('iqrsfrommedian', 'ksiqrsfrommode', 'zsigned'):
        assert np.isnan(out[key]) or abs(out[key]) < 1e6, (key, out[key])


# --- garch_fit: with a constant fitted variance, the residual-test differences are exactly zero ---

def test_garch_fit_differences_are_exactly_zero_without_variance_dynamics():
    rng = np.random.default_rng(0)
    out = model_fit.garch_fit(z_score(rng.standard_normal(1000)))
    if not (out['GARCH_1'] == 0 and out['ARCH_1'] == 0):
        pytest.skip("this realization was fitted with variance dynamics")
    for key in ('engle_mean_diff_p', 'engle_max_diff_p', 'lbq_mean_diff_p', 'lbq_max_diff_p', 'diff_ac1'):
        assert out[key] == 0.0, (key, out[key])  # (were round-off of either sign, 1e-17 to 1e-12)


def test_garch_fit_differences_are_computed_when_the_variance_varies():
    rng = np.random.default_rng(1)
    n, h, y = 3000, 1.0, []
    for _ in range(n):  # ARCH(1): h_t = 0.3 + 0.6 y_{t-1}^2
        y.append(np.sqrt(h) * rng.standard_normal())
        h = 0.3 + 0.6 * y[-1] ** 2
    out = model_fit.garch_fit(z_score(np.array(y)))
    assert out['ARCH_1'] > 0.2
    assert out['diff_ac1'] > 0.05 and out['engle_mean_diff_p'] > 0
