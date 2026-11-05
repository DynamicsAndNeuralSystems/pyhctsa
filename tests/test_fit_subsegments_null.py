"""The spread ratios of fit_subsegments ('ar' and 'ss'): the expected sample variance of estimates from
overlapping segments (exact for segment means of independent values), ratios near 1 at every length for
stationary series, a clear response to drifting dynamics and innovation variance, and the edge cases."""
import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import model_fit as mf
from pyhctsa.utils import _even_split_points, z_score


def _ar(coefs, n, seed, burn=500):
    e = np.random.default_rng(seed).standard_normal(n + burn)
    return lfilter([1.0], np.r_[1.0, -np.asarray(coefs)], e)[burn:]


def _uniform_spans(N, m, l):
    first = _even_split_points(1, N - l + 1, m)
    return np.column_stack([first, first + l - 1])


# ------------------------------------------------------------------------------
# the expected sample variance across overlapping segments
# ------------------------------------------------------------------------------
def test_stationary_sample_var_limits():
    w = np.full(4, 2.0)
    assert mf._stationary_sample_var(w, np.array([[1, 10], [11, 20], [21, 30], [31, 40]])) == pytest.approx(2.0)
    assert mf._stationary_sample_var(w, np.array([[1, 10]] * 4)) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(mf._spread_ratio(np.arange(4.0), w, np.array([[1, 10]] * 4)))


@pytest.mark.parametrize('spans', [_uniform_spans(400, 25, 40),               # the configured geometry
                                   np.array([[1, 30], [11, 60], [41, 70], [61, 120], [100, 140]])])  # unequal
def test_stationary_sample_var_segment_means(spans):
    # for means of independent values the covariance o_ij / (n_i n_j) is exact
    reps, N = 20000, spans[:, 1].max()
    z = np.random.default_rng(1).standard_normal((reps, N))
    means = np.stack([z[:, a - 1:b].mean(axis=1) for a, b in spans], axis=1)
    observed = np.var(means, axis=1, ddof=1).mean()
    expected = mf._stationary_sample_var(1.0 / (spans[:, 1] - spans[:, 0] + 1), spans)
    assert observed == pytest.approx(expected, rel=0.02)


# ------------------------------------------------------------------------------
# stationary series: ratios about 1 at any length
# ------------------------------------------------------------------------------
def _median_ratios(coefs, N, reps, seed):
    keys = ('a_1_stdratio', 'a_2_stdratio', 'logfpe_stdratio')
    vals = np.array([[mf.fit_subsegments(z_score(_ar(coefs, N, seed + i)), 'ar', 2, 'uniform', [25, 0.1])[k]
                      for k in keys] for i in range(reps)])
    return dict(zip(keys, np.median(vals, axis=0)))


@pytest.mark.parametrize('coefs', [[], [0.6], [1.2, -0.7]])
def test_ratios_near_one_at_every_length(coefs):
    short = _median_ratios(coefs, 200, 60, 100)
    long = _median_ratios(coefs, 2000, 60, 200)
    for k in short:
        assert 0.8 < short[k] < 1.15 and 0.85 < long[k] < 1.15, (k, short[k], long[k])
        assert abs(long[k] - short[k]) < 0.15, k


def test_ss_ratio_near_one():
    vals = [mf.fit_subsegments(z_score(_ar([0.6], 1000, 300 + i)), 'ss', 2, 'uniform', [25, 0.1])['logfpe_stdratio']
            for i in range(20)]
    assert 0.8 < np.median(vals) < 1.2


# ------------------------------------------------------------------------------
# non-stationary series
# ------------------------------------------------------------------------------
def test_drifting_ar_coefficient():
    N = 2000
    phi = np.linspace(0.1, 0.8, N)
    e = np.random.default_rng(5).standard_normal(N)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi[t] * x[t - 1] + np.sqrt(1 - phi[t] ** 2) * e[t]   # unit variance throughout
    out = mf.fit_subsegments(z_score(x), 'ar', 2, 'uniform', [25, 0.1])
    assert out['a_1_stdratio'] > 2


def test_drifting_innovation_variance():
    N = 2000
    e = np.random.default_rng(6).standard_normal(N) * np.linspace(0.5, 2.0, N)
    x = lfilter([1.0], [1.0, -0.5], e)
    out = mf.fit_subsegments(z_score(x), 'ar', 2, 'uniform', [25, 0.1])
    assert out['logfpe_stdratio'] > 3
    assert out['a_1_stdratio'] < 2   # the dynamics do not change


# ------------------------------------------------------------------------------
# edge cases
# ------------------------------------------------------------------------------
def test_edge_cases():
    # an exact fit (a sinusoid is an exact AR(2)): no sampling error to compare with
    out = mf.fit_subsegments(np.sin(np.arange(500) * 0.3), 'ar', 2, 'uniform', [25, 0.1])
    assert np.isnan(out['a_1_stdratio']) and np.isnan(out['logfpe_stdratio']) and np.isfinite(out['a_1_mean'])
    # segments of 4 points: no residual degree of freedom for an AR(2) fit
    out = mf.fit_subsegments(z_score(_ar([0.5], 40, 7)), 'ar', 2, 'uniform', [25, 0.1])
    assert np.isnan(out['a_2_stdratio']) and np.isnan(out['logfpe_stdratio']) and np.isfinite(out['fpe_mean'])
    # every segment the same: no spread is expected
    out = mf.fit_subsegments(z_score(_ar([0.5], 200, 8)), 'ar', 2, 'uniform', [5, 200])
    assert np.isnan(out['a_1_stdratio']) and np.isnan(out['logfpe_stdratio'])
    # the raw spreads are no longer reported for 'ar' and 'ss'
    assert 'a_1_std' not in out and 'fpe_std' not in out
