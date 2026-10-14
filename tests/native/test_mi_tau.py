"""The 'mi' delay: first minimum of the 3-point-smoothed Kraskov AMI of tie-aware ranks."""
import numpy as np
import pytest
from scipy.stats import rankdata

from pyhctsa.utils import _tau_ac1e, _tau_mi, _tie_aware_ranks, get_tau


def _noisy_sine(period, noise, n, seed, quant=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)) + noise * rng.standard_normal(n)
    return x if quant is None else np.round(x / quant) * quant


def _ar2(period, r, n, seed):
    rng = np.random.default_rng(seed)
    th = 2 * np.pi / period
    a1, a2 = 2 * r * np.cos(th), -r * r
    e = rng.standard_normal(n + 500)
    x = np.zeros(n + 500)
    for t in range(2, n + 500):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x[500:]


def _z(x):
    return (x - x.mean()) / x.std(ddof=1)


# ---------------------------------------------------------------- the rank transform
def test_ranks_equal_rankdata_without_ties():
    rng = np.random.default_rng(0)
    x = rng.permutation(500) * 0.01 + 1e-4 * rng.standard_normal(500)   # all gaps >> 1e-5 std
    np.testing.assert_array_equal(_tie_aware_ranks(x), rankdata(x))


def test_exact_repeats_are_ties_with_average_ranks():
    x = np.array([3.0, 1.0, 3.0, 2.0, 3.0, 5.0])
    np.testing.assert_array_equal(_tie_aware_ranks(x), rankdata(x))
    np.testing.assert_array_equal(_tie_aware_ranks(x), [4.0, 1.0, 4.0, 2.0, 4.0, 6.0])


def test_blurred_repeats_stay_ties():
    rng = np.random.default_rng(1)
    q = np.round(rng.standard_normal(600) * 3) / 3        # about 20 levels, 0.33 apart
    blurred = q + 1e-7 * rng.standard_normal(q.size)       # each level split into distinct values
    np.testing.assert_array_equal(_tie_aware_ranks(blurred), rankdata(q))


def test_dense_continuous_data_is_not_merged():
    # a smooth ramp: every gap is tiny but there is no isolated cluster
    x = np.linspace(0, 1, 2000) ** 2
    np.testing.assert_array_equal(_tie_aware_ranks(x), rankdata(x))


def test_ranks_invariant_to_monotone_transform_and_order_preserving():
    rng = np.random.default_rng(2)
    x = np.round(rng.standard_normal(400) * 2) / 2 + 1e-8 * rng.standard_normal(400)
    r = _tie_aware_ranks(x)
    for f in (lambda v: np.exp(v), lambda v: 5 * v - 3, lambda v: v ** 3):
        np.testing.assert_array_equal(_tie_aware_ranks(f(x)), r)
    assert r.min() >= 1 and r.max() <= x.size and np.isclose(r.sum(), x.size * (x.size + 1) / 2)


# ---------------------------------------------------------------- the delay
def test_deterministic():
    x = _z(_noisy_sine(40, 0.3, 1000, 3, quant=0.25))
    a = _tau_mi(x)
    np.random.seed(123)
    b = _tau_mi(x.copy())
    np.random.seed(456)
    c = get_tau(x.copy(), 'mi')
    assert a == b == c


def test_invariant_to_monotone_transform(monkeypatch):
    # The AMI of ranks is unchanged by any strictly increasing transformation of the data. (The
    # ac1e cap is not: it depends on the ACF, so fix it to isolate the AMI minimum.)
    import pyhctsa.utils as utils
    monkeypatch.setattr(utils, '_tau_ac1e', lambda y: 40)
    x = _z(_noisy_sine(30, 0.4, 1000, 4))
    base = _tau_mi(x)
    assert 3 <= base < 40
    for f in (np.exp, lambda v: v ** 3, lambda v: 7 * v + 100, np.arctan):
        assert _tau_mi(_z(f(x))) == base


def test_stable_under_tiny_perturbation_of_quantized_series():
    x = _noisy_sine(60, 0.5, 1500, 5, quant=0.3)
    base = get_tau(_z(x), 'mi')
    assert base > 1
    rng = np.random.default_rng(6)
    for _ in range(6):
        xp = x + 1e-6 * x.std() * rng.standard_normal(x.size)
        assert get_tau(_z(xp), 'mi') == base


def test_known_delay_recovered():
    # AR(2) with an oscillation of period 40 (and high Q): the AMI of the first quarter-period
    # minimum is near lag 10; the delay is capped at ac1e, so compare with min(10, ac1e)
    for seed in range(3):
        x = _z(_ar2(40, 0.97, 2000, seed))
        ref = min(10, _tau_ac1e(x))
        tau = get_tau(x, 'mi')
        assert abs(tau - ref) <= max(1, 0.2 * ref), (seed, tau, ref)
    # a slow noisy sine: the AMI minimum (about P/4 = 25) is below the ac1e cap
    x = _z(_noisy_sine(100, 0.2, 2000, 7))
    ref = min(25, _tau_ac1e(x))
    assert abs(get_tau(x, 'mi') - ref) <= max(1, 0.2 * ref)


def test_never_exceeds_ac1e_and_at_least_one():
    for seed in range(4):
        x = _z(_noisy_sine(25 + 10 * seed, 0.3, 800, seed))
        tau = get_tau(x, 'mi')
        assert 1 <= tau <= _tau_ac1e(x)


def test_short_and_degenerate_series():
    assert np.isnan(get_tau(np.ones(300), 'mi'))
    rng = np.random.default_rng(8)
    assert get_tau(rng.standard_normal(1000), 'mi') == 1   # white noise: ac1e = 1
