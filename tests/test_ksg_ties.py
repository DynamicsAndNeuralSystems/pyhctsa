"""KSG algorithm 2 on data with repeated values (issue #64).

In algorithm 2, each column with a repeated value gets tiny fixed-seed Gaussian jitter before the
neighbor search, so the estimate no longer depends on which tied neighbors the k-d tree returns
(platform, fetch count), and is no longer biased down by the samples that share a boundary value.
Columns of distinct values, and algorithm 1, are untouched.
"""
import functools

import numpy as np
import pytest

import pyhctsa.ksg as ksg
from pyhctsa.ksg import ksg_mi
from pyhctsa.robust import bf_random
from pyhctsa.utils import _knn_theiler


def _z(a):
    return (a - a.mean()) / a.std(ddof=1)


def _series():
    e = np.random.default_rng(11).standard_normal(600)
    ar1 = np.zeros(600)
    for t in range(1, 600):
        ar1[t] = 0.8 * ar1[t - 1] + e[t]
    t = np.arange(600)
    sine = np.sin(2 * np.pi * t / 23) + 0.3 * np.random.default_rng(12).standard_normal(600)
    rep = np.round(np.random.default_rng(13).standard_normal(600) * 2) / 2  # half-integer values
    return {'ar1': _z(ar1), 'sine': _z(sine), 'rep': _z(rep)}


SERIES = _series()

# ksg_mi(y[:-2], y[2:], 4, algorithm, theiler), computed on macOS arm64 from main before this
# change (0fa9013): the untied series in both algorithms, and the tied series in algorithm 1,
# must keep these values.
UNCHANGED_PINS = {
    ('ar1', 1, 0): 0.19842680894232245,
    ('ar1', 1, 3): 0.20902619511565224,
    ('ar1', 2, 0): 0.21325317813921352,
    ('ar1', 2, 3): 0.22466264991249663,
    ('sine', 1, 0): 0.43204524048688064,
    ('sine', 1, 3): 0.4474298998475694,
    ('sine', 2, 0): 0.4356920270009548,
    ('sine', 2, 3): 0.4524730936138077,
    ('rep', 1, 0): 7.202397910223058,
    ('rep', 1, 3): 7.178390485779044,
}

# Algorithm 2 on the tied series (rounded white noise, so the true MI at lag 2 is 0), with the
# tie-breaking jitter (macOS arm64). Before it, these were -1.3204804013957627 (theiler 0) and
# -1.2997212290306326 (theiler 3) on macOS, and depended on the platform and on how many
# neighbors the tree was asked for.
TIED_PINS = {0: 0.007668018685738964, 3: 0.020412563749243162}


@pytest.mark.parametrize('key', sorted(UNCHANGED_PINS))
def test_unchanged_values(key):
    name, algorithm, theiler = key
    y = SERIES[name]
    assert ksg_mi(y[:-2], y[2:], 4, algorithm, theiler) == pytest.approx(UNCHANGED_PINS[key],
                                                                         rel=1e-12)


def _fetch_more(extra, z, k, w, metric):
    dist, idx = _knn_theiler(z, k + extra, w, metric=metric)
    return dist[:, :k], idx[:, :k]


@pytest.mark.parametrize('theiler', [0, 3])
def test_tied_value_pinned_and_independent_of_tree_fetch_count(theiler, monkeypatch):
    """On main, asking the tree for more neighbors reordered the tied ones and moved the value;
    with the ties broken there is nothing left to reorder."""
    y = SERIES['rep']
    values = []
    for extra in (0, 3, 10):
        monkeypatch.setattr(ksg, '_knn_theiler', functools.partial(_fetch_more, extra))
        values.append(ksg_mi(y[:-2], y[2:], 4, 2, theiler))
    assert values[0] == values[1] == values[2]
    assert values[0] == pytest.approx(TIED_PINS[theiler], rel=1e-9)


def test_break_ties_touches_only_repeated_entries():
    pair = SERIES['ar1'].copy()
    pair[[10, 20]] = pair[5]  # three entries share a value; the others are distinct
    a = np.column_stack([SERIES['ar1'], SERIES['rep'], np.zeros(600), pair])
    out = ksg._break_ties(a, 5)
    assert np.array_equal(out[:, 0], a[:, 0])  # distinct values: untouched
    assert np.array_equal(out[:, 2], a[:, 2])  # constant: untouched
    # repeated values: fixed-seed jitter, a seed per column (first_seed + column index); the few
    # values of 'rep' that occur once are untouched
    vals, inv, cnt = np.unique(a[:, 1], return_inverse=True, return_counts=True)
    rep = cnt[inv] > 1
    assert 0 < np.sum(~rep) < 20
    assert np.array_equal(out[~rep, 1], a[~rep, 1])
    assert np.array_equal(out[rep, 1], a[rep, 1] + ksg._TIE_JITTER * bf_random(600, 6, 'normal')[rep])
    changed = np.flatnonzero(out[:, 3] != a[:, 3])
    assert changed.tolist() == [5, 10, 20]
    assert np.array_equal(out[changed, 3],
                          a[changed, 3] + ksg._TIE_JITTER * bf_random(600, 8, 'normal')[changed])
    assert all(np.unique(out[:, j]).size == 600 for j in (0, 1, 3))
    assert np.array_equal(a[:, 1], SERIES['rep'])  # the input is not modified


def test_break_ties_resolves_collisions_in_large_tie_blocks():
    """Two values, 10000 copies each: jitter of 1e-10 lands dozens of pairs on the same float at
    round-off resolution, which further passes separate."""
    col = np.repeat([-1.0, 1.0], 10000)[:, None]
    first = col[:, 0] + ksg._TIE_JITTER * bf_random(20000, 0, 'normal')
    assert np.unique(first).size < 20000  # the first pass alone collides
    out = ksg._break_ties(col, 0)
    assert np.unique(out[:, 0]).size == 20000
    assert np.array_equal(out, ksg._break_ties(col, 0))  # deterministic


def test_quantized_gaussian_close_to_unquantized():
    """Rounding a correlated Gaussian pair (true MI 0.2231) to steps of 0.5 and 0.1 left main's
    algorithm 2 at -2.41 and -0.34 (every sample sharing a boundary value was counted); with the
    ties broken the estimate stays near that of the unrounded data (0.2008)."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal(2000)
    w = 0.6 * u + 0.8 * rng.standard_normal(2000)
    unrounded = ksg_mi(u, w, 4, 2)
    for step in (0.5, 0.1, 0.01):
        assert abs(ksg_mi(np.round(u / step) * step, np.round(w / step) * step, 4, 2) - unrounded) < 0.03
