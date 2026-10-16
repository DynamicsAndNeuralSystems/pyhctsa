"""Edge-case behavior of z_score, preproc_compare/preproc_iterate, the HMM training likelihood, the
false-nearest-neighbors dimension, nlpe, persistent_homology, embed2_angle_tau, tisean_d2 and
preproc_schreiber_denoise."""
import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import pre_process as pp
from pyhctsa.utils import theiler_window, z_score


def _ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# ------------------------------------------------------------------------------
# z_score: scale-free
# ------------------------------------------------------------------------------
def test_zscore_is_scale_free():
    x = _ar1(300, 0.5, 0)
    for scale in (1e-9, 1e-6, 1e6):
        np.testing.assert_allclose(z_score(scale * x), z_score(x), atol=1e-12)


@pytest.mark.parametrize("x", [[7.0, 7.0, 7.0], [0.0, 0.0], [0.3, 0.1 + 0.2, 0.3], [1e8, 1e8 + 1.5e-8]])
def test_zscore_constant_up_to_rounding_raises(x):
    with pytest.raises(ValueError):
        z_score(x)


# ------------------------------------------------------------------------------
# preproc_compare / preproc_iterate
# ------------------------------------------------------------------------------
def test_preproc_compare_exact_fit_residual_is_nan():
    # one sinusoid fits a pure sinusoid up to numerical error (here a variance ratio of about 1e-14)
    t = np.arange(1, 1001)
    y = np.sin(2 * np.pi * t / 20)
    assert all(np.isnan(v) for v in pp.preproc_compare(y, 'sin1').values())
    # an exactly zero residual too
    assert all(np.isnan(v) for v in pp.preproc_compare(np.arange(100.0), 'poly1').values())


@pytest.mark.parametrize("meth", ['diff1', 'medianf3', 'poly1', 'resample_1_2'])
def test_preproc_compare_does_not_depend_on_the_scale(meth):
    y = _ar1(500, 0.8, 1)
    a, b = pp.preproc_compare(y, meth), pp.preproc_compare(1e-6 * y, meth)
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-9), k


def test_preproc_iterate_does_not_depend_on_the_scale():
    y = _ar1(500, 0.9, 2)
    a, b = pp.preproc_iterate(y, 'diff'), pp.preproc_iterate(1e-7 * y, 'diff')
    for k in a:
        if np.isnan(a[k]):
            assert np.isnan(b[k]), k
        else:
            assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-9), k


def test_iterate_stats_nan_for_negligible_processed_variance():
    y = _ar1(300, 0.5, 3)
    assert np.all(np.isnan(pp._iterate_stats(y, 1e-6 * _ar1(300, 0.5, 4))))
    assert not np.all(np.isnan(pp._iterate_stats(y, 1e-4 * _ar1(300, 0.5, 4))))


# ------------------------------------------------------------------------------
# HMM: the training log-likelihood is that of the returned parameters
# ------------------------------------------------------------------------------
def test_hmm_training_loglik_is_of_the_returned_parameters():
    y = z_score(_ar1(400, 0.7, 5))
    n_train = int(np.floor(0.7 * len(y)))
    out = mf.hmm_fit(y, 0.7, 3)
    mu, cov, P, pi, ll = mf._gaussian_hmm_fit(y[:n_train], 3)
    assert out['LLtrainpersample'] == pytest.approx(
        mf._gaussian_hmm_loglik(y[:n_train], mu, cov, P, pi) / n_train, abs=1e-12)
    assert out['LLdifference'] == pytest.approx(out['LLtestpersample'] - out['LLtrainpersample'], abs=1e-12)
    cmp = mf.hmm_compare_n_states(y, 0.6, [2, 3])
    n_train = int(np.floor(0.6 * len(y)))
    lls = [mf._gaussian_hmm_fit(y[:n_train], k)[4] / n_train for k in (2, 3)]
    assert cmp['maxLLtrain'] == pytest.approx(max(lls), abs=1e-12)
    assert cmp['chLLtrain'] == pytest.approx(lls[1] - lls[0], abs=1e-12)


# ------------------------------------------------------------------------------
# False nearest neighbors: the dimension is the smallest one not ruled out
# ------------------------------------------------------------------------------
def test_fnn_dimension_smallest_not_ruled_out():
    # never below the threshold up to maxm: maxm + 1 (a lower bound), as fnn's firstunder*
    t = np.arange(4000)
    x = np.sin(2 * np.pi * t / 37) + np.sin(2 * np.pi * t / (37 * np.sqrt(2))) \
        + 0.5 * np.random.default_rng(6).standard_normal(4000)
    d = nl.fnn(x, 1, 3, ('ac', 1), False, escape_factor=5)
    assert d['firstunder005'] == 4  # not below 5% up to maxm = 3
    assert nl.fnn(x, 1, 3, ('ac', 1), True, 0.05, 5) == 4
    from pyhctsa.toolboxes.Tisean_3_0_1 import tisean
    assert tisean.fnn_embedding_dimension(x, 1, int(theiler_window(x, ['ac', 1], x.size)), 0.05, 3) == 4
    # in general the dimension is firstunder* at the threshold (including TISEAN stopping early)
    for x2 in (np.random.default_rng(7).standard_normal(1000), x):
        d = nl.fnn(x2, 1, 10, ('ac', 1), False, escape_factor=5)
        m = nl.fnn(x2, 1, 10, ('ac', 1), True, 0.05, 5)
        assert m == d['firstunder005']
        assert nl._fnn_embedding_dim(x2, 1, 0.05) == m


# ------------------------------------------------------------------------------
# nlpe: the delay is resolved on the cropped series
# ------------------------------------------------------------------------------
def test_nlpe_resolves_tau_after_crop():
    t = np.arange(3000)
    y = np.sin(2 * np.pi * t / np.where(t < 600, 25.0, 80.0)) + 0.1 * np.random.default_rng(7).standard_normal(3000)
    a = nl.nlpe(y, 2, 'ac1e', 600)
    b = nl.nlpe(y[:600], 2, 'ac1e', 600)
    for k in a:
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == pytest.approx(b[k], abs=1e-12), k


# ------------------------------------------------------------------------------
# persistent_homology: classes alive at the threshold count, the essential H0 class does not
# ------------------------------------------------------------------------------
def test_persistent_homology_alive_classes_die_at_threshold(monkeypatch):
    ripser_mod = pytest.importorskip('ripser')
    from scipy.spatial.distance import pdist
    y = np.random.default_rng(11).standard_normal(100)
    x = np.column_stack((y[:-1], y[1:]))  # tau = 1, m = 2: 99 points, all in the threshold subsample
    thr = np.max(pdist(x))
    inf = np.inf
    dgms = [np.array([[0, 0.5 * thr], [0, inf], [0, inf], [0, 0.25 * thr]]),  # one extra H0 class alive
            np.array([[0.2 * thr, inf], [0.1 * thr, 0.3 * thr]])]  # one H1 class alive at the threshold
    monkeypatch.setattr(ripser_mod, 'ripser', lambda *a, **k: {'dgms': dgms})
    out = nl.persistent_homology(y, 1, 2)
    # H0: 0.5 + 0.25 + 1 (the extra alive class dies at the threshold); the essential class is dropped
    assert out['totalPersistenceH0'] == pytest.approx(1.75)
    # H1: the alive loop dies at the threshold, lifetime 0.8
    assert out['maxPersistenceH1'] == pytest.approx(0.8)
    assert out['totalPersistenceH1'] == pytest.approx(1.0)


# ------------------------------------------------------------------------------
# embed2_angle_tau: undefined angles are removed; NaN treatment is the same for all statistics
# ------------------------------------------------------------------------------
def test_embed2_angle_tau_repeated_values():
    y = np.round(_ar1(600, 0.6, 8))  # many repeated values: many 0/0 angles
    out = co.embed2_angle_tau(y, 20)
    assert all(np.isfinite(v) for v in out.values())


def test_embed2_angle_tau_undefined_tau_gives_nan_for_every_statistic():
    y = np.repeat(np.array([0.0, 1.0, 0.5, 2.0, -1.0]), 2)  # at tau = 1, every other step has 0/0
    y = np.tile(y, 3)
    out = co.embed2_angle_tau(y, 12)
    for j in (1, 2, 3):
        keys = [f'mean_thetaac{j}', f'max_thetaac{j}', f'min_thetaac{j}']
        assert len({bool(np.isnan(out[k])) for k in keys}) == 1


# ------------------------------------------------------------------------------
# tisean_d2: one NaN treatment for every Takens' statistic
# ------------------------------------------------------------------------------
def test_tisean_d2_takens_nan_at_one_dimension(monkeypatch):
    y = z_score(np.sin(np.arange(1500) * 0.1) + 0.1 * np.random.default_rng(9).standard_normal(1500))
    full = nl.tisean_d2(y, 1, 5, ('ac', 1))
    orig = nl._sub_takens

    def with_nan(dat, eup):
        t = orig(dat, eup)
        t[-1] = np.nan
        return t

    monkeypatch.setattr(nl, '_sub_takens', with_nan)
    out = nl.tisean_d2(y, 1, 5, ('ac', 1))
    takens = [k for k in out if k.startswith('takens05')]
    assert len(takens) == 10 and all(np.isnan(out[k]) for k in takens)
    assert all(np.isfinite(full[k]) for k in takens)


# ------------------------------------------------------------------------------
# preproc_schreiber_denoise: neighbor statistics over the points with an embedding vector
# ------------------------------------------------------------------------------
def test_fnn_fraction_needs_enough_points():
    from pyhctsa.toolboxes.Tisean_3_0_1 import tisean
    assert tisean.fnn_min_count(0.05) == 60 and tisean.fnn_min_count(0.5) == 6
    # white noise: the points with a neighbor within the search range dwindle with the dimension,
    # and the last fraction (2 points) is 0; that dimension is not accepted: the dimension is the
    # first one whose fraction rests on fewer than 60 points (a lower bound)
    x = np.random.default_rng(3).standard_normal(1000)
    r = tisean.false_nearest(x, 1, 1, 10, 1, 5.0)
    assert r['count'].shape == r['pfnn'].shape and r['count'][-1] < 60 and r['pfnn'][-1] < 0.05
    first_low = int(np.flatnonzero(r['count'] < 60)[0])
    assert np.all(r['pfnn'][:first_low] >= 0.05)
    assert nl.fnn(x, 1, 10, 1, True, 0.05, 5) == r['dim'][first_low]
    assert tisean.fnn_first_under(r['dim'], r['pfnn'], 0.05) == r['dim'][-1]  # without counts
    assert tisean.fnn_embedding_dimension(x, 1, 1, 0.05) == r['dim'][first_low]
    # firstunder* share the minimum count, so they never decrease as the threshold falls
    d = nl.fnn(x, 1, 10, 1, False, escape_factor=5)
    fu = [d[f'firstunder{k}'] for k in ('09', '08', '07', '06', '05', '04', '03', '02', '01', '005')]
    assert np.all(np.diff(fu) >= 0) and fu[-1] == r['dim'][first_low]
    # 0.05 is the default threshold
    assert nl.fnn(x, 1, 10, 1, True, escape_factor=5) == nl.fnn(x, 1, 10, 1, True, 0.05, 5)


def test_schreiber_neighbor_stats_skip_first_samples():
    y = z_score(np.sin(np.arange(1000) * 2 * np.pi / 40) + 0.05 * np.random.default_rng(10).standard_normal(1000))
    out = pp.preproc_schreiber_denoise(y, 5, 1, 1, 0.3)
    # every point with a full embedding vector has neighbors, and the first 4 samples are not counted
    assert out['fracNoCorrection'] == 0
    assert out['meanNeighbors'] > 1
