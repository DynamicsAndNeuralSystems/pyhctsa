"""Tests for outputs whose definitions were ill-posed on ordinary inputs (unbounded ratios,
values set by the series length, statistics used outside their domain)."""
import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import distribution as di
from pyhctsa.operations import symbolic as sy


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def _all_nan(out):
    """Every output undefined (a dict of NaNs, or a bare NaN when no series gives outputs with these arguments)."""
    return all(np.isnan(v) for v in out.values()) if isinstance(out, dict) else bool(np.isnan(out))


def _ar1(phi, n, seed, innovations='gauss'):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 500) if innovations == 'gauss' else rng.exponential(size=n + 500) - 1
    x = np.zeros(n + 500)
    for t in range(1, n + 500):
        x[t] = phi * x[t - 1] + e[t]
    return _zscore(x[500:])


# ------------------------------------------------------------------------------
# binary_stats: pupstat2
# ------------------------------------------------------------------------------
def test_pupstat2_is_a_bounded_contrast_of_the_two_halves():
    rng = np.random.default_rng(0)
    up = _zscore(np.r_[-np.ones(250), np.ones(250)] + 0.1 * rng.standard_normal(500))
    assert sy.binary_stats(up, 'mean')['pupstat2'] == 1.0  # every 1 in the second half (was inf)
    assert sy.binary_stats(up[::-1], 'mean')['pupstat2'] == -1.0
    for seed in range(20):
        y = _zscore(np.random.default_rng(seed).standard_normal(501))  # odd length
        for how in ('mean', 'diff', 'iqr'):
            assert -1 <= sy.binary_stats(y, how)['pupstat2'] <= 1


def test_pupstat2_compares_proportions_and_is_antisymmetric():
    # 3 ones among the first 5 symbols and 3 among the last 6: proportions 0.6 and 0.5
    y = np.array([1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1.0])
    y_bin = (y > np.mean(y)).astype(int)
    assert y_bin[:5].sum() == 3 and y_bin[5:].sum() == 3
    assert sy.binary_stats(y, 'mean')['pupstat2'] == pytest.approx((0.5 - 0.6) / (0.5 + 0.6))
    # reversing an even-length series swaps the halves
    y = _zscore(np.random.default_rng(3).standard_normal(400))
    assert sy.binary_stats(y[::-1], 'mean')['pupstat2'] == pytest.approx(-sy.binary_stats(y, 'mean')['pupstat2'])


# ------------------------------------------------------------------------------
# burstiness and the harmonic mean: statistics of non-negative data
# ------------------------------------------------------------------------------
def test_burstiness_is_bounded_on_its_domain_and_nan_outside_it():
    rng = np.random.default_rng(0)
    for x in (rng.exponential(size=2000), rng.lognormal(size=2000), rng.poisson(3, 2000).astype(float),
              np.r_[0.0, rng.gamma(0.2, size=1999)], np.full(50, 2.0) + 1e-3 * rng.random(50)):
        out = di.burstiness(x)
        r = x.std(ddof=1) / x.mean()
        assert out['B'] == pytest.approx((r - 1) / (r + 1), abs=1e-15)
        assert -1 <= out['B'] <= 1 and -1.01 <= out['B_Kim'] <= 1.01
    assert abs(di.burstiness(rng.exponential(size=20000))['B']) < 0.02  # a Poisson process: 0
    # the pole at mean = -std (was 39, -1.8e16 and -41 here), and any other signed series
    z = _zscore(rng.standard_normal(1000))
    for mu in (-0.95, -1.0, -1.05, 0.0, 0.5):
        out = di.burstiness(z + mu)
        assert np.isnan(out['B']) and np.isnan(out['B_Kim'])
    assert all(np.isnan(v) for v in di.burstiness(np.zeros(10)).values())


def test_harmonic_mean_domain():
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=500)
    hm = di.mean(x, 'harm')
    assert hm == pytest.approx(1 / np.mean(1 / x), rel=1e-13)
    assert hm <= di.mean(x, 'geom') <= di.mean(x, 'norm')
    assert di.mean(np.r_[0.0, x], 'harm') == 0.0
    assert di.mean(np.array([1, 2, 4]), 'harm') == pytest.approx(12 / 7)  # integer input
    # signed data: the sum of reciprocals has poles (was between -184 and 2.0 over four samples of mean 0.1)
    z = _zscore(rng.standard_normal(1000))
    assert np.isnan(di.mean(z + 0.1, 'harm')) and np.isnan(di.mean(z - 1.05, 'harm'))
    assert np.isnan(di.mean(-x, 'harm'))


# ------------------------------------------------------------------------------
# time_rev_kld: needs ten Theiler windows
# ------------------------------------------------------------------------------
def test_time_rev_kld_is_nan_when_the_series_spans_too_few_theiler_windows():
    rng = np.random.default_rng(0)
    walk = _zscore(np.cumsum(rng.standard_normal(5000)))  # reversible, but one correlation time long
    out = co.time_rev_kld(walk, 'ac1e', 2)
    assert np.isnan(out['raw']) and np.isnan(out['abs'])  # (was between -2.4 and 2.4 over eight such walks)
    # an explicit window: 10 * (w + k) embedded points are needed
    y = _ar1(0.5, 500, 1)
    assert np.isfinite(co.time_rev_kld(y, 1, 2, k=3, theiler_win=46)['raw'])   # 499 >= 490
    assert np.isnan(co.time_rev_kld(y, 1, 2, k=3, theiler_win=47)['raw'])      # 499 < 500
    # the cap on the number of embedded points counts
    assert _all_nan(co.time_rev_kld(_ar1(0.5, 2000, 2), 1, 2, k=3, theiler_win=40, max_n=400))


def test_time_rev_kld_separates_reversible_from_irreversible_series():
    reversible = [co.time_rev_kld(_ar1(phi, 2000, seed), 'ac1e', 2)['raw']
                  for seed, phi in enumerate((0.0, 0.3, 0.6, 0.8, 0.9, 0.95))]
    assert np.all(np.isfinite(reversible)) and np.max(np.abs(reversible)) < 0.12
    irreversible = [co.time_rev_kld(_ar1(0.5, 2000, seed, 'exp'), 'ac1e', 2)['raw'] for seed in range(4)]
    assert np.min(irreversible) > 0.25
