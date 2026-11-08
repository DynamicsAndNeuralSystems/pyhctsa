"""KSG algorithm 2 on data with ties at the k-th joint distance (issue #64).

Algorithm 2 takes its marginal separations ``eps_x``, ``eps_y`` over every sample tied at the
k-th joint distance, not over whichever k of them the k-d tree returns, so its value is a
function of the data alone. Untied data keep their previous values exactly.
"""
import functools

import numpy as np
import pytest

import pyhctsa.ksg as ksg
from pyhctsa.ksg import ksg_mi


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

# ksg_mi(y[:-2], y[2:], 4, algorithm, theiler) for the untied series, computed on macOS arm64 from
# the main checkout before the tie convention (0fa9013); the convention must leave them unchanged.
UNTIED_PINS = {
    ('ar1', 1, 0): 0.19842680894232245,
    ('ar1', 1, 3): 0.20902619511565224,
    ('ar1', 2, 0): 0.21325317813921352,
    ('ar1', 2, 3): 0.22466264991249663,
    ('sine', 1, 0): 0.43204524048688064,
    ('sine', 1, 3): 0.4474298998475694,
    ('sine', 2, 0): 0.4356920270009548,
    ('sine', 2, 3): 0.4524730936138077,
}

# Algorithm 2 on the tied series with the tie convention (macOS arm64). Before it, these were
# -1.3204804013957627 (theiler 0) and -1.2997212290306326 (theiler 3) on macOS, and depended on
# the platform and on how many neighbors the tree was asked for.
TIED_PINS = {0: -1.3576546298534993, 3: -1.342088950940811}


@pytest.mark.parametrize('key', sorted(UNTIED_PINS))
def test_untied_values_unchanged(key):
    name, algorithm, theiler = key
    y = SERIES[name]
    assert ksg_mi(y[:-2], y[2:], 4, algorithm, theiler) == pytest.approx(UNTIED_PINS[key],
                                                                         rel=1e-12)


@pytest.mark.parametrize('theiler', [0, 3])
def test_tied_value_independent_of_tree_fetch_count(theiler, monkeypatch):
    """Asking the tree for more neighbors reorders the tied ones; the value must not move."""
    y = SERIES['rep']
    original = ksg._joint_neighbors
    values = []
    for extra in (0, 3, 10):
        monkeypatch.setattr(ksg, '_joint_neighbors', functools.partial(original, extra=extra))
        values.append(ksg_mi(y[:-2], y[2:], 4, 2, theiler))
    assert values[0] == values[1] == values[2]
    assert values[0] == pytest.approx(TIED_PINS[theiler], rel=1e-12)


@pytest.mark.parametrize('theiler', [0, 3])
def test_tied_value_independent_of_row_order(theiler):
    """With theiler 0 the estimate is a function of the set of (x, y) pairs: permuting the rows
    reorders the tree's ties but must not change the value."""
    y = SERIES['rep']
    x1, x2 = y[:-2], y[2:]
    if theiler:
        # a Theiler window depends on the row order; reversing it keeps the windows the same
        perm = np.arange(len(x1))[::-1]
    else:
        perm = np.random.default_rng(0).permutation(len(x1))
    assert ksg_mi(x1[perm], x2[perm], 4, 2, theiler) == pytest.approx(
        ksg_mi(x1, x2, 4, 2, theiler), rel=1e-12)


def test_hand_example_with_ties():
    """Points (0,0), (1,0), (0,1), (1,1), (3,3); y is a permutation of x, so standardizing
    scales both columns alike and keeps the ties. With k = 1:

    * each unit-square corner has its three other corners tied at joint distance 1, so
      eps_x = eps_y = 1 and n_x = n_y = 3 (the other corners; (3,3) is 2 away in each);
    * (3,3) has the single nearest neighbor (1,1) at 2, so eps_x = eps_y = 2 and n_x = n_y = 2.

    I = psi(1) - 1 + psi(5) - (4 * 2 psi(3) + 2 psi(2)) / 5 = 13/12 - 14/5 = -103/60.
    Taking the separations over a single returned neighbor instead (e.g. (1,0) for (0,0), giving
    eps_y = 0 and n_y = 1) gives a different, order-dependent value.
    """
    x = [0, 1, 0, 1, 3]
    y = [0, 0, 1, 1, 3]
    assert ksg_mi(x, y, 1, 2, 0) == pytest.approx(-103 / 60, rel=1e-12)
    # and the same for any row order
    perm = [3, 0, 4, 2, 1]
    assert ksg_mi(np.take(x, perm), np.take(y, perm), 1, 2, 0) == pytest.approx(-103 / 60,
                                                                                rel=1e-12)


def test_coincident_points_have_zero_marginal_epsilon():
    """Ties at joint distance 0 (coincident points) give eps_x = eps_y = 0."""
    z = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    x, y = z[:, :1], z[:, 1:]
    eps_x, eps_y = ksg._marginal_epsilons(x, y, *ksg._joint_neighbors(z, 2, 0), 0)
    assert eps_x[:3].tolist() == [0.0, 0.0, 0.0]
    assert eps_y[:3].tolist() == [0.0, 0.0, 0.0]
