"""Features on inputs where a naive formula degenerates: real-valued spectrogram bins, repeated
values, statistics that are zero up to round-off, and fits with no dynamics."""
import numpy as np
import pytest

from pyhctsa.operations import correlation, model_fit, spectral, stationarity, surrogates
from pyhctsa.utils import _iqr, z_score


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


def _embed2_angles(y, tau):
    m = np.column_stack((y[:-tau], y[tau:]))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.arctan(np.diff(m[:, 1]) / np.diff(m[:, 0]))


def _embed2_angle_outputs(theta):
    """The angle-histogram and autocorrelation outputs of embed2 from its angle sequence, written
    out directly: proportions over the defined angles, lagged products over pairs of defined angles."""
    N = theta.size
    ok = ~np.isnan(theta)
    # (embed2's bin edges, which put angles of exactly 0 in the upper bin whatever the rounding)
    edges = correlation._fixed_edges(10, -np.pi / 2, np.pi / 2)
    px = np.histogram(theta[ok], bins=edges)[0] / ok.sum()
    assert px.sum() == pytest.approx(1.0, abs=1e-12)
    w = np.diff(edges)
    out = {'hist10std': np.std(px, ddof=1),
           'histent': -np.sum(px[px > 0] * np.log(px[px > 0] / w[px > 0]))}
    afifth = N // 5
    props = []
    for i in range(5):
        seg = theta[afifth * i:afifth * (i + 1)]
        seg = seg[~np.isnan(seg)]
        props.append(np.histogram(seg, bins=correlation._fixed_edges(4, -np.pi / 2, np.pi / 2))[0] / seg.size)
    props = np.array(props).T
    assert np.allclose(props.sum(axis=0), 1.0)
    for b in range(4):
        out[f'stdb{b + 1}'] = np.std(props[b], ddof=1)
    d = theta - np.nanmean(theta)
    for k in (1, 2, 3):
        pairs = [d[t] * d[t + k] for t in range(N - k) if ok[t] and ok[t + k]]
        out[f'theta_ac{k}'] = (N - k) / N * np.mean(pairs) / np.nanmean(d ** 2)
    return out


def test_embed2_histograms_and_autocorrelations_unchanged_without_ties():
    # with every angle defined: the proportions over all angles and the standard (Box-Jenkins)
    # autocorrelation sum_t d_t d_{t+k} / sum_t d_t^2
    rng = np.random.default_rng(12)
    y = z_score(rng.standard_normal(403))
    for tau in (1, 4):
        theta = _embed2_angles(y, tau)
        assert not np.isnan(theta).any()
        out = correlation.embed2(y, tau)
        exp = _embed2_angle_outputs(theta)
        d = theta - theta.mean()
        for k in (1, 2, 3):
            assert exp[f'theta_ac{k}'] == pytest.approx(np.sum(d[:-k] * d[k:]) / np.sum(d ** 2), rel=1e-12)
            assert out[f'theta_ac{k}'] == pytest.approx(correlation.autocorr(theta, k), rel=0, abs=0)
        for key, val in exp.items():
            assert out[key] == pytest.approx(val, rel=1e-10, abs=1e-14), key


@pytest.mark.parametrize('tau', [1, 2, 'tau'])
def test_embed2_with_ties_uses_only_defined_angles(tau):
    rng = np.random.default_rng(13)
    y = z_score(rng.poisson(0.5, 1000).astype(float))  # many 0/0 angles
    out = correlation.embed2(y, tau)
    tau_used = int(correlation._resolve_delay(y, tau, cap=True, default='tau'))
    theta = _embed2_angles(y, tau_used)
    assert np.isnan(theta).mean() > 0.1
    exp = _embed2_angle_outputs(theta)  # (checks that the proportions sum to 1)
    for key, val in exp.items():
        assert out[key] == pytest.approx(val, rel=1e-10, abs=1e-14), key
    # a proper differential entropy of a distribution on (-pi/2, pi/2): at most log(pi)
    assert out['histent'] <= np.log(np.pi) + 1e-12


def test_gapped_autocorr_never_pairs_values_across_a_gap():
    from pyhctsa.operations.correlation import _gapped_autocorr
    # alternate defined and undefined values: no two defined values are 1 apart, so lag 1 is
    # undefined; joining the defined values into one sequence would give a lag-1 autocorrelation
    x = np.full(40, np.nan)
    x[::2] = np.sin(np.arange(20) / 3.0)
    ac = _gapped_autocorr(x, [1, 2, 3])
    assert np.isnan(ac[0]) and np.isnan(ac[2]) and np.isfinite(ac[1])
    # lag 2 pairs the defined values at their true separation
    d = x[::2] - x[::2].mean()
    assert ac[1] == pytest.approx((40 - 2) / 40 * np.mean(d[:-1] * d[1:]) / np.mean(d ** 2), rel=1e-12)
    # all defined: the standard estimator, exactly
    z = np.random.default_rng(3).standard_normal(50)
    np.testing.assert_array_equal(_gapped_autocorr(z, [1, 2, 3]), correlation.autocorr(z, [1, 2, 3]))
    # equal defined values: undefined
    assert np.isnan(_gapped_autocorr(np.r_[1.0, np.nan, 1.0, 1.0], [1])).all()


# --- local_global: skewness and ac1 are differences, defined when the global statistic is zero ----

def test_local_global_skewness_is_finite_for_a_balanced_two_valued_series():
    from scipy.stats import skew
    rng = np.random.default_rng(8)
    y = z_score(rng.permutation(np.repeat([0.0, 1.0], 500)))  # skewness is exactly 0, 1e-17 computed
    for subset_how, n in (('l', 100), ('p', 0.1), ('unicg', 100)):
        out = stationarity.local_global(y, subset_how, n)
        # (a ratio to the computed global skewness was about 1e17)
        assert 0 <= out['skewness'] < 1, (subset_how, out['skewness'])
        assert np.isfinite(out['kurtosis'])  # (Pearson kurtosis is at least 1: its ratio is always defined)
    assert stationarity.local_global(y, 'l', 100)['skewness'] == pytest.approx(abs(skew(y[:100])), abs=1e-12)


def test_local_global_definitions():
    from scipy.stats import kurtosis, skew
    rng = np.random.default_rng(9)
    y = z_score(rng.gamma(2.0, size=1000))
    s = y[:100]
    out = stationarity.local_global(y, 'l', 100)
    # differences for the dimensionless statistics that can be zero
    assert out['skewness'] == pytest.approx(abs(skew(s) - skew(y)), rel=1e-12)
    assert out['ac1'] == pytest.approx(abs(correlation.autocorr(s, 1, 'Fourier') - correlation.autocorr(y, 1, 'Fourier')),
                                       rel=1e-12)
    # relative deviations for the two whose full-series value is bounded away from zero
    assert out['kurtosis'] == pytest.approx(abs(1 - kurtosis(s, fisher=False) / kurtosis(y, fisher=False)), rel=1e-12)
    assert out['iqr'] == pytest.approx(abs(1 - _iqr(s) / _iqr(y)), rel=1e-12)


@pytest.mark.parametrize("N", [1000, 5000])
def test_local_global_skewness_and_ac1_are_stable_on_white_noise(N):
    # The full-series skewness and lag-1 autocorrelation of white noise are sampling noise around
    # zero, of size 1/sqrt(N). As ratios to them, both outputs were heavy-tailed (values in the
    # hundreds in a few percent of realizations) with a median that grew with N.
    rng = np.random.default_rng(10)
    outs = [stationarity.local_global(z_score(rng.standard_normal(N)), 'l', 100) for _ in range(200)]
    for key, bound in (('skewness', 1.5), ('ac1', 0.6)):
        v = np.array([o[key] for o in outs])
        assert np.all(np.isfinite(v)) and v.max() < bound, (key, v.max())
        assert np.median(v) < 0.25, (key, np.median(v))


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
