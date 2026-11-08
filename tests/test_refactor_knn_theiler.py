"""Checks for the Theiler-windowed k-nearest-neighbor refactor (duplication audit theme #12):
`utils._knn_theiler` in place of the hand-written searches in `nonlinearity.fractal_dimensions`,
`nonlinearity.return_time` and `ksg.ksg_mi`. The 'rep' series (half-integer values) has many
exactly tied distances.

Two kinds of check:

1. Same-runner reference checks (the ones that matter): verbatim copies of the replaced searches
   (`_ref_*`, copied from main 0fa9013, file and line given with each) against `_knn_theiler`
   with the matching metric and tree, asserting identical distances and indices on all three
   series, `rep` included: built on the same machine, the same tree breaks ties the same way
   (except fractal_dimensions' indices on `rep`, which the old code did not use and which differ
   under ties because it queried the tree for more rows; see the test).
2. Operation-level pins, generated on Ben's Mac (arm64) from the base branch (dup-theiler-embed
   f5aa8c1, output-identical to main) before the refactor, with a tolerance per operation (`TOL`)
   set by what the operation computes, since other platforms reproduce them only approximately.
"""
import logging
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

from pyhctsa.ksg import _standardize, ksg_mi
from pyhctsa.operations import information as IN
from pyhctsa.operations import nonlinearity as NL
from pyhctsa.robust import bf_spread_perm
from pyhctsa.utils import _even_split_points, _knn_theiler, theiler_window, time_delay_embed


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
    ('ksg_mi', 'rep', 2, 0): 0.007668018685738964,
    ('ksg_mi', 'rep', 2, 3): 0.020412563749243162,
}
# ksg_mi on 'rep', algorithm 2: on main the exact ties in the joint max-norm distances made the
# value depend on which tied rows the k-d tree returned first, which differed between platforms
# (main gave -1.3204804013957627 and -1.2997212290306326 on macOS; ubuntu and windows gave
# -1.3213785 at theiler 0). Since #64 (branch fix-ksg-tie-order) algorithm 2 breaks the ties with
# fixed-seed jitter, so the value is portable and is pinned again, at the new value.
KSG_PINNED_SERIES = ('ar1', 'sine', 'rep')

# Per-operation (rtol, atol) for the pins (the values above are unchanged main-checkout values).
T1 = (1e-9, 0)  # tier 1, closed-form arithmetic: return-time histogram statistics
T2 = (1e-8, 0)  # tier 2, fitted slopes: the Dq of fractal_dimensions are least-squares slopes over
                # many neighbor distances (CI drift up to 1.2e-10 in rangeDq)
T3 = (1e-5, 0)  # tier 3, kd-tree neighbor order under near-ties: ksg_mi
TOL = {
    'fractal_dimensions': T2,
    'fractal_dimensions_ref': T2,
    'return_time': T1,
    'return_time_m8': T1,
    # multivariate_ami: tier 1 (neighbor counts and digamma sums), with an atol because the MIs
    # are near zero (5e-5): atol = 1e-9 * 10, the scale being the digamma terms (psi(600) ~ 6.4)
    # whose difference they are
    'multivariate_ami': (1e-9, 1e-8),
    'ksg_mi': T3,
}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    logging.disable(logging.NOTSET)


def _check(got, want, tol):
    rtol, atol = tol
    if isinstance(want, dict):
        for k, v in want.items():
            np.testing.assert_allclose(got[k], v, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize('name', list(CALLS))
def test_operations_unchanged(name):
    f, kw = CALLS[name]
    for sn, y in _series().items():
        _check(f(y, **kw), EXPECTED[(name, sn)], TOL[name])


def test_ksg_mi_unchanged():
    for sn, y in _series().items():
        if sn not in KSG_PINNED_SERIES:
            continue
        for alg in (1, 2):
            for th in (0, 3):
                _check(ksg_mi(y[:-2], y[2:], 4, alg, th), EXPECTED[('ksg_mi', sn, alg, th)],
                       TOL['ksg_mi'])


# ---------------------------------------------------------------------------------------------
# Same-runner reference checks: the replaced searches, copied verbatim from main 0fa9013
# ---------------------------------------------------------------------------------------------

def _ref_joint_neighbors(z, k, w):
    # main 0fa9013 pyhctsa/ksg.py:41-53 (_joint_neighbors, used by ksg_mi at ksg.py:166)
    n = z.shape[0]
    dist, idx = cKDTree(z).query(z, k=k + 2 * w + 1, p=np.inf)
    keep = np.abs(idx - np.arange(n)[:, None]) > w
    keep &= np.cumsum(keep, axis=1) <= k
    return dist[keep].reshape(n, k), idx[keep].reshape(n, k)


def _ref_fractal_search(emb, kmax, past, ref_idx):
    # main 0fa9013 pyhctsa/operations/nonlinearity.py:2139-2150 (fractal_dimensions). The old
    # code kept only the sorted distances; the indices they belong to are recovered here with a
    # stable argsort of the same masked array (`order`, an addition to the copied lines), which
    # keeps the tree's order among ties. None stands for the old NaN return.
    n_emb = emb.shape[0]
    k_fetch = min(n_emb - 1, kmax + 2 * past + 5)
    dist, idx = cKDTree(emb).query(emb[ref_idx], k=k_fetch + 1)
    valid = np.abs(idx - ref_idx[:, None]) > past
    masked = np.where(valid, dist, np.inf)
    order = np.argsort(masked, axis=1, kind='stable')[:, :kmax]
    idx = np.where(np.isfinite(np.take_along_axis(masked, order, axis=1)),
                   np.take_along_axis(idx, order, axis=1), -1)
    dist = np.sort(masked, axis=1)[:, :kmax]
    for ii in np.flatnonzero(valid.sum(axis=1) < kmax):  # fall back to a full search
        all_dists = np.linalg.norm(emb - emb[ref_idx[ii]], axis=1)
        all_dists[np.abs(np.arange(n_emb) - ref_idx[ii]) <= past] = np.inf
        idx[ii] = np.argsort(all_dists, kind='stable')[:kmax]  # (addition, as above)
        all_dists = np.sort(all_dists)
        if np.sum(np.isfinite(all_dists)) < kmax:
            return None
        dist[ii] = all_dists[:kmax]
    return dist, idx


def _ref_return_time_search(Y, nnr, past, refs):
    # main 0fa9013 pyhctsa/operations/nonlinearity.py:3798-3811 (return_time): the distance to
    # (and, an addition, the index of) each reference point's nnr-th neighbor outside the window
    n_emb = Y.shape[0]
    k = min(n_emb, nnr + 2 * past + 1)
    tree = KDTree(Y)
    d_nnr = np.full(refs.size, np.nan)
    i_nnr = np.full(refs.size, -1)
    chunk = max(1, int(2e6 // k))
    for c in range(0, refs.size, chunk):
        the_refs = refs[c:c + chunk]
        dist, idx = tree.query(Y[the_refs], k=k)
        is_valid = np.abs(idx - the_refs[:, None]) > past
        which_col = np.argmax(np.cumsum(is_valid, axis=1) >= nnr, axis=1)
        d_nnr[c:c + chunk] = dist[np.arange(the_refs.size), which_col]
        i_nnr[c:c + chunk] = idx[np.arange(the_refs.size), which_col]
    return d_nnr, i_nnr


@pytest.mark.parametrize('k', [1, 4, 8])
@pytest.mark.parametrize('w', [0, 3])
def test_ref_ksg_joint_neighbors(k, w):
    for sn, y in _series().items():
        for lag in (1, 2, 5):
            z = np.hstack([_standardize(y[:-lag, None]), _standardize(y[lag:, None])])
            d_old, i_old = _ref_joint_neighbors(z, k, w)
            d_new, i_new = _knn_theiler(z, k, w, metric='chebyshev')
            np.testing.assert_array_equal(d_new, d_old, err_msg=f'{sn} lag {lag}')
            np.testing.assert_array_equal(i_new, i_old, err_msg=f'{sn} lag {lag}')


@pytest.mark.parametrize('embed,kmax', [((1, 5), 10), ((1, 3), 20), ((2, 2), 10)])
def test_ref_fractal_dimensions_search(embed, kmax):
    for sn, y in _series().items():
        emb = time_delay_embed(y, embed[1], embed[0])
        n_emb = emb.shape[0]
        for past in (0, 2, int(theiler_window(y, ('ac', 1)))):
            for ref_idx in (np.arange(n_emb), bf_spread_perm(n_emb)[:n_emb // 5]):
                old = _ref_fractal_search(emb, kmax, past, ref_idx)
                d_new, i_new = _knn_theiler(emb, kmax, past, query=ref_idx)
                assert old is not None
                # the distances, all the old code used, are identical on every series
                np.testing.assert_array_equal(d_new, old[0], err_msg=f'{sn} past {past}')
                if sn != 'rep':
                    np.testing.assert_array_equal(i_new, old[1], err_msg=f'{sn} past {past}')
                else:
                    # Under exact ties the indices can differ even on one machine: the old code
                    # asked cKDTree for kmax + 2*past + 6 rows and the helper asks for
                    # kmax + 2*past + 1, and cKDTree's order among tied rows depends on k.
                    # fractal_dimensions never used them; check they are valid neighbors at the
                    # stated distances instead.
                    true_d = np.linalg.norm(emb[i_new] - emb[ref_idx][:, None, :], axis=2)
                    np.testing.assert_allclose(true_d, d_new, rtol=1e-12, atol=1e-12)
                    assert np.all(np.abs(i_new - ref_idx[:, None]) > past)
                    assert all(len(set(r)) == kmax for r in i_new)


@pytest.mark.parametrize('embed,nnr', [((1, 3), 30), ((1, 8), 12), ((2, 2), 1)])
def test_ref_return_time_search(embed, nnr):
    for sn, y in _series().items():
        Y = time_delay_embed(y, embed[1], embed[0])
        n_emb = Y.shape[0]
        for past in (0, 3, int(theiler_window(y, ('ac', 1)))):
            for refs in (np.arange(n_emb), np.unique(_even_split_points(1, n_emb, 500)) - 1):
                d_old, i_old = _ref_return_time_search(Y, nnr, past, refs)
                d_new, i_new = _knn_theiler(Y, nnr, past, query=refs, tree='sklearn')
                np.testing.assert_array_equal(d_new[:, nnr - 1], d_old, err_msg=f'{sn} past {past}')
                np.testing.assert_array_equal(i_new[:, nnr - 1], i_old, err_msg=f'{sn} past {past}')


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
