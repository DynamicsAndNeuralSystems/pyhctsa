"""Tests for outputs whose definitions were ill-posed on ordinary inputs (unbounded ratios,
values set by the series length, statistics used outside their domain)."""
import numpy as np
import pytest

from pyhctsa.operations import symbolic as sy


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


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
