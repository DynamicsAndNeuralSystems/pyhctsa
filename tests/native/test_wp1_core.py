"""Core helpers of ``pyhctsa.utils`` and ``pyhctsa.robust`` (native-Python pass, WP1).

The checks are accuracy checks: hand-computed values, exact properties (ties, end points,
conservation), and equality with the numpy/scipy primitives the helpers are built on.
"""
import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from pyhctsa import utils
from pyhctsa.robust import bf_hist_edges, bf_quantile_edges
from pyhctsa.utils import _even_split_points, _sample_std, _zscore_or_zeros


@pytest.fixture(autouse=True)
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ------------------------------------------------------------------ Hazen quantile
# pyhctsa's quantile-based features use numpy's Hazen definition (Hyndman & Fan 1996, def. 5)
# directly; these checks document the two properties the call sites rely on.
def hazen(x, p):
    return np.atleast_1d(np.quantile(np.asarray(x, dtype=float), p, method='hazen'))


def test_quantile_hand_values():
    # Hyndman & Fan definition 5: the k-th of n values is the (k - 1/2)/n quantile
    assert hazen([1, 2, 3, 4], 0.5).tolist() == [2.5]
    assert hazen([1, 2, 3, 4], 0.25).tolist() == [1.5]
    assert hazen([1, 2, 3, 4], 0.125).tolist() == [1.0]
    assert hazen([1, 2, 3, 4], [0, 0.04, 0.96, 1]).tolist() == [1, 1, 4, 4]
    x = [10, 20, 30, 40, 50]
    assert hazen(x, [0.1, 0.3, 0.5, 0.7, 0.9]).tolist() == x
    assert hazen(x, 0.2)[0] == pytest.approx(15.0)


def test_quantile_ties_return_the_tied_value_exactly():
    y = np.full(10, 1 / 3)
    assert np.array_equal(hazen(y, np.linspace(0, 1, 37)), np.full(37, 1 / 3))
    z = np.repeat(np.array([0.1, 2 / 3, 0.7]), 4)
    q = hazen(z, [0.2, 0.21, 0.5, 0.52, 0.8])
    assert q[0] == 0.1 and q[1] == 0.1 and q[3] == 2 / 3 and q[4] == 0.7


# ------------------------------------------------------------------ even split points
def _brute_split(a, b, n):
    return [math.floor(Fraction(a) + Fraction(k * (b - a), n - 1) + Fraction(1, 2)) for k in range(n)]


def test_even_split_points_hand_cases():
    assert _even_split_points(1, 10, 2).tolist() == [1, 10]
    assert _even_split_points(1, 9, 3).tolist() == [1, 5, 9]
    # exact ties are rounded up: the midpoint of [1, 10] is 5.5 -> 6; of [1, 4] with 3 points 2.5 -> 3
    assert _even_split_points(1, 10, 3).tolist() == [1, 6, 10]
    assert _even_split_points(1, 4, 3).tolist() == [1, 3, 4]
    assert _even_split_points(0, 5, 6).tolist() == [0, 1, 2, 3, 4, 5]
    assert _even_split_points(3, 3, 4).tolist() == [3, 3, 3, 3]
    assert _even_split_points(7, 20, 1).tolist() == [7]   # one point: the first (as np.linspace)
    assert _even_split_points(1, 10, 0).size == 0


def test_even_split_points_exact_for_many_inputs():
    rng = np.random.default_rng(3)
    for _ in range(300):
        a = int(rng.integers(0, 50))
        b = a + int(rng.integers(0, 4000))
        n = int(rng.integers(2, 40))
        got = _even_split_points(a, b, n)
        assert got.tolist() == _brute_split(a, b, n)
        assert got[0] == a and got[-1] == b
        assert np.all(np.diff(got) >= 0)
        # each point is within half a unit of the exact real position
        exact = a + np.arange(n) * (b - a) / (n - 1)
        assert np.all(np.abs(got - exact) <= 0.5 + 1e-9)


def test_even_split_points_spacing_covers_range():
    got = _even_split_points(1, 1001, 11)
    assert got.tolist() == list(range(1, 1002, 100))


# ------------------------------------------------------------------ sample standard deviation
def test_sample_std_equals_numpy_ddof1():
    rng = np.random.default_rng(4)
    m = rng.standard_normal((6, 5))
    assert _sample_std(m) == pytest.approx(np.std(m.ravel(), ddof=1), rel=1e-14)
    for ax in (0, 1):
        np.testing.assert_allclose(_sample_std(m, axis=ax), np.std(m, axis=ax, ddof=1), rtol=1e-14)
    assert _sample_std([1, 2, 3, 4]) == pytest.approx(math.sqrt(5 / 3), rel=1e-15)
    assert isinstance(_sample_std([1.0, 2.0]), float)
    assert _sample_std(m, axis=0).shape == (5,)


def test_sample_std_degenerate_inputs():
    assert _sample_std([4.2]) == 0.0          # documented convention (C1): one value -> 0
    assert _sample_std([[1.0], [2.0]], axis=1).tolist() == [0.0, 0.0]
    assert math.isnan(_sample_std([]))
    assert math.isnan(_sample_std([np.nan]))
    assert math.isnan(_sample_std([1.0, np.nan, 2.0]))
    assert _sample_std([3.0, 3.0, 3.0]) == 0.0


def test_zscore_convention():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(100) * 3 + 2
    z = _zscore_or_zeros(x)
    assert z.mean() == pytest.approx(0, abs=1e-14)
    assert np.std(z, ddof=1) == pytest.approx(1, rel=1e-14)
    assert np.array_equal(_zscore_or_zeros(np.full(7, 2.5)), np.zeros(7))   # constant -> zeros
    assert _zscore_or_zeros([3.0]).tolist() == [0.0]
    assert _zscore_or_zeros([]).size == 0
    assert np.isnan(_zscore_or_zeros([1.0, np.nan, 2.0])).all()


def test_round_half_away_convention():
    for k in range(0, 200):
        assert utils._round_half_away(k + 0.5) == k + 1
        assert utils._round_half_away(-k - 0.5) == -k - 1
    assert utils._round_half_away(0.49999999999999994) == 0
    assert utils._round_half_away(2.4) == 2 and utils._round_half_away(-2.6) == -3
    assert math.isnan(utils._round_half_away(float('nan')))
    a = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.2])
    assert utils._round_half_away_array(a).tolist() == [-3, -2, -1, 1, 2, 3, 3]


# ------------------------------------------------------------------ robust.py users
def test_bf_quantile_edges_equiprobable():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(1000)
    edges = bf_quantile_edges(y, 10)
    assert edges.size == 11 and np.all(np.diff(edges) > 0)
    counts, _ = np.histogram(y, edges)
    assert counts.sum() == 1000
    assert np.all(np.abs(counts - 100) <= 2)
    # constant data: a single bin of unit width; empty: undefined
    assert bf_quantile_edges(np.full(5, 3.0), 4).tolist() == [2.5, 3.5]
    assert np.isnan(bf_quantile_edges([], 4)).all()
    # tied values merge bins
    assert bf_quantile_edges(np.array([1.0] * 90 + list(range(2, 12))), 10).size < 11


def test_bf_hist_edges_freedman_diaconis_width():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(2000)
    edges = bf_hist_edges(y, 'fd')
    q1, q3 = np.quantile(y, [0.25, 0.75], method='hazen')
    width = 2 * (q3 - q1) * len(y) ** (-1 / 3)
    nb = edges.size - 1
    assert nb == math.ceil((y.max() - y.min()) / width)
    assert edges[0] < y.min() and edges[-1] > y.max()


# ------------------------------------------------------------------ framing
def test_frames_cover_signal_in_order():
    x = np.arange(1, 11)
    assert utils.frame_signal(x, 4).tolist() == [[1, 5, 9], [2, 6, 10], [3, 7, 0], [4, 8, 0]]
    # overlapping frames advance by n - p and start at x[0]
    y = utils.frame_signal(x, 4, 2)
    assert y.shape == (4, 4)
    assert y[:, 0].tolist() == [1, 2, 3, 4] and y[:, 1].tolist() == [3, 4, 5, 6]
    assert y[:, -1].tolist() == [7, 8, 9, 10]
    # a short signal is a single zero-padded frame; an empty one has no frames
    assert utils.frame_signal([1, 2], 4, 3).tolist() == [[1], [2], [0], [0]]
    assert utils.frame_signal([], 4).shape == (4, 0)
    for bad in [(x, 0), (x, 4, 4), (x, 4, -1)]:
        with pytest.raises(ValueError):
            utils.frame_signal(*bad)


def test_frames_non_overlapping_partition_the_signal():
    rng = np.random.default_rng(9)
    for L, n in [(100, 7), (96, 8), (5, 5), (13, 1)]:
        x = rng.standard_normal(L)
        f = utils.frame_signal(x, n)
        assert np.array_equal(f.T.ravel()[:L], x)
        assert np.all(f.T.ravel()[L:] == 0)
