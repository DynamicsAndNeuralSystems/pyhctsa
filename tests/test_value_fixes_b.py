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
_T = np.arange(1, 1001.0)
_SIN = 2 + np.sin(2 * np.pi * 0.0523 * _T + 0.3)  # not a whole number of periods
_U = 2 * _T / 1000 - 1
_CUBIC = 2 + np.polyval([1, -2, 0.5, 1], _U)
_WHITE = np.random.default_rng(42).standard_normal(1000)
_COLORED = _ar1(1000, 0.9, 43)


def _all_nan(out):
    return all(np.isnan(v) for v in out.values())


def _all_finite(out):
    return all(np.isfinite(v) for v in out.values())


@pytest.mark.parametrize("y, meth", [
    (_SIN, 'sin1'), (_SIN, 'sin2'), (1e3 + np.sin(2 * np.pi * _T / 3), 'sin1'),
    (2 + 0.37 * _T, 'poly1'), (_CUBIC, 'poly3'), (_CUBIC, 'poly9'), (_CUBIC, 'spline24'),
    (1e3 + 0.37 * _T, 'diff1'), (2 + np.polyval([1, -2, 0.5], _U), 'diff2'), (np.full(500, 7.3), 'medianf3')])
def test_preproc_compare_roundoff_is_nan(y, meth):
    # the processed series is floating-point round-off (an exact fit, the differences of a
    # polynomial, a constant): no statistics
    assert _all_nan(pp.preproc_compare(y, meth))


@pytest.mark.parametrize("noise", [_WHITE, _COLORED])
@pytest.mark.parametrize("level", [1e-2, 1e-4, 1e-6])
def test_preproc_compare_small_genuine_residual_is_computed(noise, level):
    e = level * noise / np.std(noise)
    assert _all_finite(pp.preproc_compare(_SIN + e, 'sin1'))
    assert _all_finite(pp.preproc_compare(_CUBIC + e, 'poly3'))
    assert _all_finite(pp.preproc_compare(_CUBIC + e, 'spline24'))


@pytest.mark.parametrize("bits", [16, 24])
def test_preproc_compare_quantized_sinusoid_is_computed(bits):
    # the quantization noise left by an exact fit (about 2**-bits of the amplitude) is genuine
    q = 2.0 ** (bits - 1)
    y = 2 + np.round(np.sin(2 * np.pi * 0.0523 * _T + 0.3) * q) / q
    assert _all_finite(pp.preproc_compare(y, 'sin1'))


def test_preproc_iterate_roundoff_is_nan():
    quad = 2 + np.polyval([1, -2, 0.5], _U)
    assert _all_nan(pp.preproc_iterate(quad, 'diff'))  # the second differences are constant
    out = pp.preproc_iterate(quad + 1e-6 * _WHITE, 'diff')
    assert all(np.isfinite(v) for k, v in out.items() if not k.startswith(('xc', 'normdiff')))


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


def test_iterate_stats_nan_for_roundoff_processed_series():
    # round-off: a standard deviation at most 10 * N * eps (6.7e-13 for N = 300) of the rms of y
    y = _ar1(300, 0.5, 3)
    z = _ar1(300, 0.5, 4) * np.sqrt(np.mean(y ** 2)) / np.std(_ar1(300, 0.5, 4), ddof=1)
    assert np.all(np.isnan(pp._iterate_stats(y, 1e-13 * z)))
    assert not np.all(np.isnan(pp._iterate_stats(y, 1e-11 * z)))


def test_fit_sinusoids_at_least_one_cycle():
    from pyhctsa.robust import bf_fit_sinusoids
    trend = _T + np.random.default_rng(5).standard_normal(1000)
    for constant in (False, True):
        assert bf_fit_sinusoids(trend, 1, constant=constant)[1][0] >= 1 / 1000 - 1e-12


def test_residual_stats_roundoff_only():
    # an exact fit leaves round-off: no residual statistics; the quantization noise left by an
    # exact fit to 24-bit data is genuine
    from pyhctsa.operations.spectral import sinusoid_fit
    from pyhctsa.robust import bf_residual_stats
    s = np.sin(2 * np.pi * 0.0523 * _T + 0.3)
    out = sinusoid_fit(s, 'sin1')
    assert np.isnan(out['resAC1']) and np.isnan(out['resAC2']) and np.isnan(out['resrunsz'])
    q = 2.0 ** 23
    out = sinusoid_fit(np.round(s * q) / q, 'sin1')
    assert np.isfinite(out['resAC1']) and np.isfinite(out['resAC2']) and np.isfinite(out['resrunsz'])
    e = 1e-9 * np.random.default_rng(8).standard_normal(120)
    assert all(np.isnan(bf_residual_stats(e, np.full(120, 1e7))))
    assert all(np.isfinite(bf_residual_stats(e, np.ones(120))))


def test_fit_sinusoids_exact_sum_reaches_roundoff():
    from pyhctsa.robust import bf_fit_sinusoids
    y = 2 + np.sin(2 * np.pi * 0.0711 * _T) + 0.5 * np.cos(2 * np.pi * 0.21 * _T + 1)
    yfit, f = bf_fit_sinusoids(y, 2, constant=True)
    np.testing.assert_allclose(f, [0.0711, 0.21], rtol=1e-12)
    assert np.std(y - yfit) <= 1000 * np.finfo(float).eps * np.sqrt(np.mean(y ** 2))


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
    # never below the threshold up to maxm: maxm + 1 (a lower bound), as fnn's dim*
    t = np.arange(4000)
    x = np.sin(2 * np.pi * t / 37) + np.sin(2 * np.pi * t / (37 * np.sqrt(2))) \
        + 0.5 * np.random.default_rng(6).standard_normal(4000)
    d = nl.fnn(x, 1, 3, ('ac', 1), False, escape_factor=5)
    assert d['dim005'] == 4  # not below 5% up to maxm = 3
    assert nl.fnn(x, 1, 3, ('ac', 1), True, 0.05, 5) == 4
    from pyhctsa.toolboxes.Tisean_3_0_1 import tisean
    assert tisean.fnn_embedding_dimension(x, 1, int(theiler_window(x, ['ac', 1], x.size)), 0.05, 3) == 4
    # in general the dimension is dim* at the threshold (including TISEAN stopping early)
    for x2 in (np.random.default_rng(7).standard_normal(1000), x):
        d = nl.fnn(x2, 1, 10, ('ac', 1), False, escape_factor=5)
        m = nl.fnn(x2, 1, 10, ('ac', 1), True, 0.05, 5)
        assert m == d['dim005']
        assert nl._fnn_embedding_dim(x2, 1, 0.05) == m


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
    # dim* share the minimum count, so they never decrease as the threshold falls
    d = nl.fnn(x, 1, 10, 1, False, escape_factor=5)
    fu = [d[f'dim{k}'] for k in ('09', '08', '07', '06', '05', '04', '03', '02', '01', '005')]
    assert np.all(np.diff(fu) >= 0) and fu[-1] == r['dim'][first_low]
    # 0.05 is the default threshold
    assert nl.fnn(x, 1, 10, 1, True, escape_factor=5) == nl.fnn(x, 1, 10, 1, True, 0.05, 5)


def _smoothed_noise(n, seed=2):
    z = np.convolve(np.random.default_rng(seed).standard_normal(n + 100), np.ones(20) / 20, 'same')
    return z[50:50 + n]


@pytest.mark.parametrize('n', [250, 1000])
def test_fnn_embedding_dimension_limited_by_length(n):
    # when 'fnn' chooses the dimension for another algorithm it is at most floor(2 log10 N);
    # fnn itself is not capped
    cap = int(np.floor(2 * np.log10(n)))
    assert nl._max_embedding_dim(n) == cap
    x = np.random.default_rng(1).standard_normal(n)
    m_fnn = nl.fnn(x, 1, 10, ('ac', 1), True, 0.05, 5)
    assert nl._embedding_params(x, 1, ['fnn', 0.05]) == (1, min(m_fnn, cap))
    assert nl._embedding_params(x, 1, 'fnn') == nl._embedding_params(x, 1, ['fnn', 0.05])
    # smoothed noise: the fraction never falls below 0.05 and many points stay behind it, so the
    # FNN answer exceeds the limit and the limit binds
    z = _smoothed_noise(n)
    assert nl.fnn(z, 1, 10, ('ac', 1), True, 0.05, 5) > cap
    assert nl._embedding_params(z, 1, ['fnn', 0.05]) == (1, cap)


def test_fnn_threshold_syntax_in_operations():
    x = np.random.default_rng(3).standard_normal(600)
    def same(a, b):
        return all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)
    assert same(nl.rqa(x, 'mi', ['fnn', 0.05]), nl.rqa(x, 'mi', 'fnn'))
    assert same(nl.local_density(x, 5, ('ac', 1), 'ac1e', ['fnn', 0.05]),
                nl.local_density(x, 5, ('ac', 1), 'ac1e', 'fnn'))
    assert same(nl.nlpe(x, ['fnn', 0.05], 'mi'), nl.nlpe(x, 'fnn', 'mi'))
    with pytest.raises(ValueError):
        nl.nlpe(x, ['nonsense', 0.05], 'mi')


def test_nlpe_fnn_limit_uses_cropped_length():
    # nlpe crops to max_n samples first; the limit on an 'fnn' dimension is floor(2 log10 max_n)
    z = z_score(_smoothed_noise(6000))
    assert nl.fnn(z[:1000], 1, 10, ('ac', 1), True, 0.05, 5) > 6
    a = nl.nlpe(z, ['fnn', 0.05], 1, 1000)
    b = nl.nlpe(z, 6, 1, 1000)
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)
    assert a['meanabs'] != nl.nlpe(z, 7, 1, 1000)['meanabs']


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
# tisean_d2: Takens' statistics over the dimensions up to the first without an estimate
# ------------------------------------------------------------------------------
def test_tisean_d2_takens_over_leading_dimensions(monkeypatch):
    y = z_score(np.sin(np.arange(1500) * 0.1) + 0.1 * np.random.default_rng(9).standard_normal(1500))
    full = nl.tisean_d2(y, 1, 5, ('ac', 1))
    takens = [k for k in full if k.startswith('takens05')]
    assert len(takens) == 10 and all(np.isfinite(full[k]) for k in takens)
    orig = nl._sub_takens
    vals = {}

    def drop(k):
        def f(dat, eup, c2, num_pairs):
            t, c = orig(dat, eup, c2, num_pairs)
            vals['t'] = t.copy()
            t[k] = np.nan
            return t, c
        return f

    # an estimate missing at the last dimension: statistics over the first four
    monkeypatch.setattr(nl, '_sub_takens', drop(4))
    out = nl.tisean_d2(y, 1, 5, ('ac', 1))
    assert np.isclose(out['takens05_max'], np.max(vals['t'][:4]))
    assert np.isclose(out['takens05_mean'], np.mean(vals['t'][:4]))
    # missing at the second dimension: fewer than two dimensions, so all ten are NaN
    monkeypatch.setattr(nl, '_sub_takens', drop(1))
    out = nl.tisean_d2(y, 1, 5, ('ac', 1))
    assert all(np.isnan(out[k]) for k in takens)


def test_tisean_d2_takens_needs_33_pairs():
    # white noise, N = 1000: at m = 9, 10 the length scale 0.5 holds only a handful of pairs,
    # whose estimate (up to 20) is not a dimension estimate; the statistics stop before
    y = z_score(np.random.default_rng(0).standard_normal(1000))
    out = nl.tisean_d2(y, 1, 10, ('ac', 1))
    assert out['takens05_max'] < 8


def test_tisean_d2_findmmin_roundoff_constant():
    # estimates equal to within round-off are a constant curve: NaN, as for exactly equal ones,
    # rather than a rescaling of their round-off differences
    flat = 0.22048796773549684 + np.array([0, 1.6e-16, 0, -2.8e-16, 3.3e-16, 1.4e-16, 0, -0.6e-16, 1.9e-16, -2.8e-16])
    for d in (flat, np.full(10, 0.2)):
        r = nl._sub_findmmin(d)
        assert r['ri1'] is None and np.isnan(r['goodness']) and np.isnan(r['linrmserr'])
    r = nl._sub_findmmin(np.array([1.0, 2.0, 2.5, 2.6, 2.6, 2.6]))  # a curve that levels off
    assert r['ri1'] is not None and np.isfinite(r['goodness'])


# ------------------------------------------------------------------------------
# preproc_schreiber_denoise: neighbor statistics over the points with an embedding vector
# ------------------------------------------------------------------------------
def test_schreiber_neighbor_stats_skip_first_samples():
    y = z_score(np.sin(np.arange(1000) * 2 * np.pi / 40) + 0.05 * np.random.default_rng(10).standard_normal(1000))
    out = pp.preproc_schreiber_denoise(y, 5, 1, 1, 0.3)
    # every point with a full embedding vector has neighbors, and the first 4 samples are not counted
    assert out['fracNoCorrection'] == 0
    assert out['meanNeighbors'] > 1
