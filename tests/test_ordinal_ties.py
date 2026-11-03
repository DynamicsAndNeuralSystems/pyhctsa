"""Ordinal patterns of series with tied values: ties are broken in a fixed pseudo-random order,
not by position, so that discrete data do not read as spuriously ordered."""
import numpy as np
import pytest

from pyhctsa.operations.entropy import (_break_ties, _ordinal_pattern_rank, permutation_entropy,
                                         permutation_entropy_complexity)
from pyhctsa.utils import time_delay_embed


def test_no_ties_unchanged():
    y = np.random.default_rng(0).standard_normal(300)
    assert _break_ties(y) is not None and np.array_equal(_break_ties(y), y)
    # the patterns are exactly those of the stable-sort ranking
    for m in (2, 3, 4):
        x = time_delay_embed(y, m, 1)
        assert np.array_equal(_ordinal_pattern_rank(time_delay_embed(_break_ties(y), m, 1)),
                              _ordinal_pattern_rank(x))


def test_ties_keep_the_order_of_distinct_values():
    y = np.array([3.0, 1.0, 1.0, 2.0, 1.0, 5.0, 2.0])
    r = _break_ties(y)
    assert sorted(r) == list(range(len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] < y[j]:
                assert r[i] < r[j]


def test_deterministic():
    y = np.random.default_rng(1).poisson(1, 500).astype(float)
    assert permutation_entropy(y, m=3) == permutation_entropy(y, m=3)


@pytest.mark.parametrize("draw", [lambda rng, n: rng.poisson(1, n).astype(float),
                                  lambda rng, n: rng.integers(0, 2, n).astype(float)])
def test_iid_discrete_reads_as_unpredictable(draw):
    # iid discrete series are as unpredictable as iid Gaussian noise; with ties broken by
    # position their permutation entropy was far lower (binary: ~0.77 vs ~0.99 at m = 3)
    rng = np.random.default_rng(2)
    pe = np.mean([permutation_entropy(draw(rng, 400), m=3)["normPermEn"] for _ in range(100)])
    pe_gauss = np.mean([permutation_entropy(rng.standard_normal(400), m=3)["normPermEn"]
                        for _ in range(100)])
    assert abs(pe - pe_gauss) < 0.01
    h = np.mean([permutation_entropy_complexity(draw(rng, 400), m=3)["hNorm"] for _ in range(50)])
    assert h > 0.98
