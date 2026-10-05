"""Tests for the ports of hctsa robust/seed: deterministic spread subsamples and averaged random draws."""
import numpy as np

from pyhctsa.operations import entropy as en
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import stationarity as st
from pyhctsa.robust import bf_random, bf_spread_perm


def _y(n=500, seed=4):
    u = bf_random(n, seed, 'normal')
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + u[i]
    return (y - y.mean()) / y.std(ddof=1)


def test_spread_perm_matches_hctsa():
    p = bf_spread_perm(1000)
    assert list(p[:8]) == [619, 237, 855, 473, 91, 709, 327, 945]  # MATLAB BF_SpreadPerm(1000)(1:8)
    assert sorted(p) == list(range(1, 1001))
    gaps = np.diff(np.concatenate([[0], np.sort(p[:64]), [1000]]))
    assert gaps.max() < 3 * 1000 / 64  # evenly spread prefix


def test_deterministic_subsample_features_ignore_the_seed():
    y = _y()
    assert nl.delay_time(y, ('ac', 10), ('ac', 1), 0) == nl.delay_time(y, ('ac', 10), ('ac', 1), 7)
    assert st.spread_random_local(y, 100, 50, 0) == st.spread_random_local(y, 100, 50, 7)
    assert nl.rqa(y, 1, 3, ('ac', 1), 0.1, 2, 2, 'full', 0) == nl.rqa(y, 1, 3, ('ac', 1), 0.1, 2, 2, 'full', 7)


def test_gp_hyperparameters_averages_draws():
    y = _y(400, 5)
    one, two = (mf.gp_hyperparameters(y, 'covSEiso_covNoise', 1, 50, 'random_i', s, 1) for s in (0, 2))
    avg = mf.gp_hyperparameters(y, 'covSEiso_covNoise', 1, 50, 'random_i', 0, 2)
    assert abs(avg['logh1'] - (one['logh1'] + two['logh1']) / 2) < 1e-12


def test_randomize_one_repeat_is_the_single_draw():
    y = _y(300, 2)
    a = en.randomize(y, 'permute', 3, 1)
    b = en.randomize(y, 'permute', 3, 20)
    assert abs(a['ac1diff'] - b['ac1diff']) < 0.2
