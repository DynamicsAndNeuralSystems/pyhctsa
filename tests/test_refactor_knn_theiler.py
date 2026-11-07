"""Pinned outputs of the operations touched by the Theiler-windowed k-nearest-neighbor refactor
(duplication audit theme #12): `utils._knn_theiler` in place of the hand-written searches in
`nonlinearity.fractal_dimensions`, `nonlinearity.return_time` and `ksg.ksg_mi`. The values were
generated from the base branch (dup-theiler-embed, f5aa8c1, output-identical to main) before the
refactor, so this checks that the refactor left every output unchanged (rtol 1e-12). The 'rep'
series (half-integer values) has many tied distances."""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.ksg import ksg_mi
from pyhctsa.operations import information as IN
from pyhctsa.operations import nonlinearity as NL
from pyhctsa.utils import _knn_theiler


def _series():
    e = np.random.default_rng(11).standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    t = np.arange(600)
    sine = np.sin(2 * np.pi * t / 23) + 0.3 * np.random.default_rng(12).standard_normal(600)
    rep = np.round(np.random.default_rng(13).standard_normal(600) * 2) / 2

    def z(x):
        return (x - x.mean()) / x.std(ddof=1)
    return {'ar1': z(ar), 'sine': z(sine), 'rep': z(rep)}


CALLS = {
    'fractal_dimensions': (NL.fractal_dimensions, dict(kmin=2, kmax=10, nref=-1, gend=5, embed_params=[1, 5])),
    'fractal_dimensions_ref': (NL.fractal_dimensions,
                               dict(kmin=5, kmax=20, nref=0.2, gend=10, embed_params=['ac1e', 3])),
    'return_time': (NL.return_time, dict(nnr=0.05, nref=500, embed_params=[1, 3])),
    'return_time_m8': (NL.return_time, dict(nnr=0.02, nref=500, embed_params=[1, 8])),
    'multivariate_ami': (IN.multivariate_ami, dict()),
}

# (operation, series) -> the first four outputs; ('ksg_mi', series, algorithm, theiler) -> MI
EXPECTED = {
    ('fractal_dimensions', 'ar1'): {'rangeDq': 0.20649967397427904, 'maxDq': 4.608010957891085, 'meanDq': 4.54340673612012, 'maxq': 0.7828355789056216},
    ('fractal_dimensions', 'sine'): {'rangeDq': 0.2662654737269792, 'maxDq': 5.141418935017795, 'meanDq': 5.001301632676803, 'maxq': 0.7948782519483549},
    ('fractal_dimensions', 'rep'): {'rangeDq': 0.6751089558639807, 'maxDq': 5.850069666826896, 'meanDq': 5.504870466417638, 'maxq': 0.8067618179435554},
    ('fractal_dimensions_ref', 'ar1'): {'rangeDq': 0.1945321086091636, 'maxDq': 3.098988866917712, 'meanDq': 3.0390632040764003, 'maxq': 0.6684495894192295},
    ('fractal_dimensions_ref', 'sine'): {'rangeDq': 0.48126333016587797, 'maxDq': 3.448758432383019, 'meanDq': 3.2714195676988345, 'maxq': 0.6630154505552991},
    ('fractal_dimensions_ref', 'rep'): {'rangeDq': 1.782368031406636, 'maxDq': 5.379210107809416, 'meanDq': 4.431947585394019, 'maxq': 0.7219783413454435},
    ('return_time', 'ar1'): {'max': 1.643308114375237, 'std': 0.2725456641585632, 'pzeros': 0.0, 'pg05': 0.72},
    ('return_time', 'sine'): {'max': 5.532314209931661, 'std': 1.303682326633718, 'pzeros': 0.0, 'pg05': 0.12},
    ('return_time', 'rep'): {'max': 1.5491794221943203, 'std': 0.21500118146827304, 'pzeros': 0.0, 'pg05': 0.85},
    ('return_time_m8', 'ar1'): {'max': 2.6749385312565663, 'std': 0.6327573236087799, 'pzeros': 0.0, 'pg05': 0.26},
    ('return_time_m8', 'sine'): {'max': 11.339690002662518, 'std': 2.4302415956759353, 'pzeros': 0.69, 'pg05': 0.07},
    ('return_time_m8', 'rep'): {'max': 3.486201547626797, 'std': 0.5263667783752303, 'pzeros': 0.0, 'pg05': 0.07},
    ('multivariate_ami', 'ar1'): {'tau': 9.0, 'multiAMI': 0.0007927572197626692, 'ami_2tau': 4.842743735939398e-05, 'ami_tau': 0.0007298996258623598},
    ('multivariate_ami', 'sine'): {'tau': 6.0, 'multiAMI': 0.6833545489005013, 'ami_2tau': 0.6577891397692613, 'ami_tau': 0.0019435807579697827},
    ('multivariate_ami', 'rep'): {'tau': 2.0, 'multiAMI': 0.001490570362344082, 'ami_2tau': 0.0012253532462806686, 'ami_tau': 0.0002880647471563712},
    ('ksg_mi', 'ar1', 1, 0): 0.19842680894232245,
    ('ksg_mi', 'ar1', 1, 3): 0.20902619511565224,
    ('ksg_mi', 'ar1', 2, 0): 0.21325317813921352,
    ('ksg_mi', 'ar1', 2, 3): 0.22466264991249663,
    ('ksg_mi', 'sine', 1, 0): 0.43204524048688064,
    ('ksg_mi', 'sine', 1, 3): 0.4474298998475694,
    ('ksg_mi', 'sine', 2, 0): 0.4356920270009548,
    ('ksg_mi', 'sine', 2, 3): 0.4524730936138077,
    ('ksg_mi', 'rep', 1, 0): 7.202397910223058,
    ('ksg_mi', 'rep', 1, 3): 7.178390485779044,
    ('ksg_mi', 'rep', 2, 0): -1.3204804013957627,
    ('ksg_mi', 'rep', 2, 3): -1.2997212290306326,
}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    logging.disable(logging.NOTSET)


def _check(got, want):
    if isinstance(want, dict):
        for k, v in want.items():
            np.testing.assert_allclose(got[k], v, rtol=1e-12, atol=0)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


@pytest.mark.parametrize('name', list(CALLS))
def test_operations_unchanged(name):
    f, kw = CALLS[name]
    for sn, y in _series().items():
        _check(f(y, **kw), EXPECTED[(name, sn)])


def test_ksg_mi_unchanged():
    for sn, y in _series().items():
        for alg in (1, 2):
            for th in (0, 3):
                _check(ksg_mi(y[:-2], y[2:], 4, alg, th), EXPECTED[('ksg_mi', sn, alg, th)])


@pytest.mark.parametrize('metric', ['euclidean', 'chebyshev'])
@pytest.mark.parametrize('tree', ['scipy', 'sklearn'])
def test_knn_theiler_matches_brute_force(metric, tree):
    # half-integer points: distances are exact and heavily tied, so the k-th distances are unique
    X = np.round(np.random.default_rng(3).standard_normal((120, 2)) * 2) / 2
    k, w = 4, 3
    q = np.arange(0, 120, 7)
    dist, idx = _knn_theiler(X, k, w, metric, query=q, tree=tree)
    diff = np.abs(X[None, :, :] - X[q, None, :])
    full = np.sqrt((diff ** 2).sum(-1)) if metric == 'euclidean' else diff.max(-1)
    full[np.abs(np.arange(120)[None, :] - q[:, None]) <= w] = np.inf
    np.testing.assert_array_equal(dist, np.sort(full, axis=1)[:, :k])
    assert np.all(np.abs(idx - q[:, None]) > w)
    np.testing.assert_array_equal(np.take_along_axis(full, idx, axis=1), dist)
    # too few rows outside the window: padded with inf and -1
    d2, i2 = _knn_theiler(X[:6], 4, 2, metric, tree=tree)
    assert np.isinf(d2[0, 3]) and i2[0, 3] == -1 and np.isfinite(d2[0, 2])
