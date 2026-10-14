"""Accuracy tests for the native implementations in the nonlinear / symbolic / correlation /
information / scaling / entropy / surrogate operations (analytic results, reference libraries and
invariants; no agreement with any other implementation is assumed)."""
import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pyhctsa.operations import entropy, nonlinearity, scaling, symbolic
from pyhctsa.operations.correlation import _acf_fft
from pyhctsa.operations.information import _rcond_1norm
from pyhctsa.fitting import polyfit
from pyhctsa.robust import bf_random


# ------------------------------------------------------------------ symbolic: transition matrices

def _palindrome(seq):
    """seq followed by its reverse: its transition-count matrix is symmetric."""
    return np.concatenate([seq, seq[::-1]])


def test_transition_matrix_counts_transitions():
    yth = np.array([1, 2, 2, 3, 1, 1, 3, 2, 1, 3])
    t = symbolic._transition_matrix(yth, 3)
    counts = np.zeros((3, 3))
    for a, b in zip(yth[:-1], yth[1:]):
        counts[a - 1, b - 1] += 1
    np.testing.assert_allclose(t, counts / (len(yth) - 1), atol=1e-15)
    assert t.sum() == pytest.approx(1.0)


def test_transition_matrix_features_match_linear_algebra():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(600))
    out = symbolic.transition_matrix(y, 'quantile', 3, 1)
    t = np.array([out[f'T{i + 1}'] for i in range(9)]).reshape(3, 3, order='F')
    ev = np.linalg.eigvals(t)
    # trace = sum of the eigenvalues
    assert out['ondiag'] == pytest.approx(np.trace(t), abs=1e-14)
    assert np.sum(ev).real == pytest.approx(out['ondiag'], abs=1e-12)
    assert out['stddiag'] == pytest.approx(np.std(np.diag(t), ddof=1), rel=1e-12)
    assert out['symdiff'] == pytest.approx(np.abs(t - t.T).sum(), abs=1e-14)
    assert out['symsumdiff'] == pytest.approx(np.tril(t, -1).sum() - np.triu(t, 1).sum(), abs=1e-14)
    cov = np.cov(t, rowvar=False)
    assert out['sumdiagcov'] == pytest.approx(np.trace(cov), rel=1e-12, abs=1e-15)
    ec = np.linalg.eigvalsh(cov)
    assert out['stdeigcov'] == pytest.approx(np.std(ec, ddof=1), rel=1e-10, abs=1e-15)
    assert out['maxeigcov'] == pytest.approx(ec.max(), rel=1e-12, abs=1e-15)
    # standard deviation of complex eigenvalues: sqrt(sum |x - mean|^2 / (n - 1)), real
    assert out['stdeig'] == pytest.approx(math.sqrt(np.sum(np.abs(ev - ev.mean()) ** 2) / 2), rel=1e-12)
    # trace of the cov. matrix is also the sum of the column variances
    assert out['sumdiagcov'] == pytest.approx(np.sum(np.var(t, axis=0, ddof=1)), rel=1e-12, abs=1e-15)


def test_symmetric_transition_matrix_has_zero_asymmetry_measures():
    rng = np.random.default_rng(1)
    yth = _palindrome(rng.integers(1, 5, 200))
    t = symbolic._transition_matrix(yth, 4)
    np.testing.assert_allclose(t, t.T, atol=1e-15)
    m = symbolic._transition_measures(yth, 4)
    assert m[3] == pytest.approx(0.0, abs=1e-14)  # sum of |T - T'|


def test_transition_measures_use_numpy_definitions():
    rng = np.random.default_rng(2)
    yth = rng.integers(1, 6, 400)
    t = symbolic._transition_matrix(yth, 5)
    m = symbolic._transition_measures(yth, 5)
    assert m[0] == pytest.approx(np.mean(np.diag(t)))
    assert m[1] == pytest.approx(np.max(np.diag(t)))
    assert m[2] == pytest.approx(np.trace(t))
    assert m[4] == pytest.approx(np.trace(np.cov(t, rowvar=False)))
    ev = np.linalg.eigvals(t)
    assert m[5] == pytest.approx(math.sqrt(np.sum(np.abs(ev - ev.mean()) ** 2) / 4))


def test_transition_p_alphabet_runs_and_is_finite():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(1000)
    out = symbolic.transition_p_alphabet(y, np.arange(2, 11), 1)
    assert np.isfinite(out['trfexp_b']) and out['trfexp_b'] < 0  # the trace decays with alphabet size


# ------------------------------------------------------------------ information
def test_rcond_equals_reciprocal_condition_number():
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.standard_normal((2, 2))
        assert _rcond_1norm(a) == pytest.approx(1.0 / np.linalg.cond(a, 1), rel=1e-12)
    assert _rcond_1norm(np.array([[1.0, 1.0], [1.0, 1.0]])) == 0.0
    assert _rcond_1norm(np.eye(2)) == 1.0
    assert math.isnan(_rcond_1norm(np.array([[1.0, np.nan], [0.0, 1.0]])))
    assert math.isinf(_rcond_1norm(np.zeros((0, 0))))


# ------------------------------------------------------------------ scaling
def test_arange_inclusive_hand_cases():
    f = scaling._arange_inclusive
    assert len(f(0, 0.1, 1)) == 11 and f(0, 0.1, 1)[-1] == 1.0
    assert len(f(0, 0.1, 0.3)) == 4 and f(0, 0.1, 0.3)[-1] == 0.3
    assert len(f(-2, 0.5, 2)) == 9
    assert len(f(-5, 0.1, 5)) == 101
    assert len(f(1, -0.1, 0)) == 11
    assert f(3, 1, 3).tolist() == [3.0]
    assert f(0, 1, 2.9).tolist() == [0.0, 1.0, 2.0]  # the bound is not reached
    assert f(1, 1, 0).size == 0 and f(0, 0, 1).size == 0 and f(np.nan, 1, 3).size == 0
    x = f(-5, 0.1, 5)
    np.testing.assert_allclose(x, np.round(np.arange(-50, 51) / 10, 12), atol=1e-12)
    assert x.max() <= 5.0


def test_polyfit_recovers_polynomial_and_is_nan_when_underdetermined():
    x = np.arange(1.0, 9.0)
    c = np.array([0.5, -2.0, 3.0])
    np.testing.assert_allclose(polyfit(x, np.polyval(c, x), 2), c, atol=1e-9)
    assert np.all(np.isnan(polyfit([1.0, 2.0], [3.0, 4.0], 2)))  # 2 points, 3 coefficients
    assert np.all(np.isnan(polyfit([1.0, 2.0, np.nan], [1.0, 2.0, 3.0], 1)))


def test_mma_runs_on_noise():
    rng = np.random.default_rng(5)
    out = scaling.mma(rng.standard_normal(1200), False, [10, 100], [-2, 2])
    assert np.isfinite(out['meanHurstExponent'])


# ------------------------------------------------------------------ correlation: autocorrelation
@pytest.mark.parametrize('n', [2, 3, 16, 17, 100, 257, 1009, 1024])
def test_acf_fft_matches_direct_sums(n):
    rng = np.random.default_rng(n)
    y = np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n)
    z = y - y.mean()
    direct = np.array([np.dot(z[:n - k], z[k:]) for k in range(n)]) / np.dot(z, z)
    np.testing.assert_allclose(_acf_fft(y), direct, atol=1e-12)


# ------------------------------------------------------------------ entropy
def test_dispersion_entropy_class_boundaries():
    # linear mapping onto [0, 1]; c = 4 classes with boundaries 0.25, 0.5, 0.75 (class = floor(4 y) + 1,
    # and the top of the range belongs to the last class)
    y = np.array([0.0, 0.2, 0.3, 0.6, 0.8, 1.0])
    out = entropy.dispersion_entropy(y, m=1, c=4, tau=1, mapping_how='linear')
    p = np.array([2, 1, 1, 2]) / 6  # classes [1, 1, 2, 3, 4, 4]
    assert out['dispEn'] == pytest.approx(-np.sum(p * np.log(p)), abs=1e-14)
    # a point exactly on a boundary goes to the upper class: 0.25 -> class 2
    y2 = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.1])
    out2 = entropy.dispersion_entropy(y2, m=1, c=4, tau=1, mapping_how='linear')
    p2 = np.array([2, 1, 1, 2]) / 6  # classes [1, 2, 3, 4, 4, 1]
    assert out2['dispEn'] == pytest.approx(-np.sum(p2 * np.log(p2)), abs=1e-14)


def test_wavelet_band_energies_conserve_energy_and_white_noise_entropy():
    rng = np.random.default_rng(6)
    y = rng.standard_normal(2048)
    e = entropy._modwt_band_energies(y, 'sym4', 5)
    assert e.sum() == pytest.approx(np.sum(y ** 2), rel=1e-10)
    ents = [entropy.wavelet_entropy(np.random.default_rng(s).standard_normal(2048)) for s in range(5)]
    assert all(0.6 < v < 0.9 for v in ents)  # white noise: about 0.75
    # an energy-free band is not floored: tiny but nonzero energies are kept
    ys = 1e-6 * y
    assert entropy.wavelet_entropy(ys) == pytest.approx(entropy.wavelet_entropy(y), abs=1e-12)  # scale-free


# ------------------------------------------------------------------ nonlinear: helpers
def test_even_split_points():
    f = nonlinearity._even_split_points
    assert f(1, 9, 5).tolist() == [1, 3, 5, 7, 9]
    assert f(0, 5, 2).tolist() == [0, 5]
    assert f(1, 10, 1).tolist() == [1]
    assert f(0, 3, 4).tolist() == [0, 1, 2, 3]
    # exact half-way points round up: 0, 0.5, 1 over [0, 1] with n = 3
    assert f(0, 1, 3).tolist() == [0, 1, 1]
    for a, b, n in [(1, 1000, 20), (0, 997, 7), (3, 500, 50)]:
        p = f(a, b, n)
        assert p[0] == a and p[-1] == b and np.all(np.diff(p) >= 0)
        exact = a + np.arange(n) * (b - a) / (n - 1)
        assert np.all(np.abs(p - exact) <= 0.5 + 1e-9)


def test_recurrent_pairs_equal_brute_force():
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((300, 3))
    d = squareform(pdist(Y))
    for radius in (0.5, 1.0, 1.7):
        src, dst = nonlinearity._recurrent_pairs(Y, radius)
        got = set(zip(src.tolist(), dst.tolist()))
        ii, jj = np.nonzero(d <= radius)
        assert got == set(zip(ii.tolist(), jj.tolist()))


def _zscore(x):
    return (x - x.mean()) / x.std(ddof=1)


def _two_state_switching(gap, n=2000, p_switch=0.02, seed=0):
    """Unit-variance noise around one of two levels `gap` apart, switching with prob. p_switch."""
    rng = np.random.default_rng(seed)
    switch = rng.random(n) < p_switch
    state = np.cumsum(switch) % 2
    return _zscore(gap * (state - 0.5) + rng.standard_normal(n))


def _gaussian_ar1(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return _zscore(x)


def test_embed_cluster_output_keys():
    out = nonlinearity.embed_cluster(_two_state_switching(4.0), tau=1, m=2)
    assert list(out) == ['sep_pp', 'sep_mahal', 'valley', 'nongauss']


def test_optimal_two_means_split_matches_brute_force():
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal((60, 3)), 1)  # (rounded: includes ties)
    r2, thr = nonlinearity._optimal_two_means_splits(x)
    for c in range(3):
        v = np.sort(x[:, c])
        tss = np.sum((v - v.mean()) ** 2)
        best = max(1 - (np.sum((v[:i] - v[:i].mean()) ** 2) + np.sum((v[i:] - v[i:].mean()) ** 2)) / tss
                   for i in range(1, v.size) if v[i - 1] < v[i])
        assert r2[c] == pytest.approx(best, abs=1e-12)
        assert np.sum(v < thr[c]) > 0 and np.sum(v > thr[c]) > 0


def test_embed_cluster_separation_increases_with_state_gap():
    outs = [nonlinearity.embed_cluster(_two_state_switching(g), tau=1, m=2) for g in (0.0, 2.0, 4.0, 6.0)]
    for key in ('sep_pp', 'sep_mahal', 'nongauss'):
        vals = [o[key] for o in outs]
        assert all(b > a for a, b in zip(vals, vals[1:])), (key, vals)
    assert outs[0]['valley'] == 0 and outs[-1]['valley'] > 0.9
    assert outs[-1]['sep_pp'] > 0.9


def test_embed_cluster_gaussian_ar_has_no_cluster_structure():
    # For Gaussian data the best 1-D two-means split explains 2/pi of the variance
    out = nonlinearity.embed_cluster(_gaussian_ar1(0.8, 6000), tau=1, m=2)
    assert out['sep_pp'] == pytest.approx(2 / np.pi, abs=0.03)
    assert out['valley'] == 0 and abs(out['nongauss']) < 0.05


def test_embed_cluster_two_point_masses_have_unit_sep_pp():
    rng = np.random.default_rng(2)
    y = _zscore(rng.integers(0, 2, 1000).astype(float))
    assert nonlinearity.embed_cluster(y, tau=1, m=2)['sep_pp'] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize('tau, m', [(1, 2), ('ac1e', 2), (1, 3)])
def test_embed_cluster_is_deterministic(tau, m):
    y = _two_state_switching(3.0)
    a = nonlinearity.embed_cluster(y, tau=tau, m=m)
    b = nonlinearity.embed_cluster(y.copy(), tau=tau, m=m)
    assert a == b


def test_embed_cluster_is_stable_to_tiny_perturbation():
    y = _two_state_switching(4.0)
    rng = np.random.default_rng(5)
    a = nonlinearity.embed_cluster(y, tau=1, m=2)
    b = nonlinearity.embed_cluster(y + 1e-6 * rng.standard_normal(y.size), tau=1, m=2)
    for key in ('sep_pp', 'sep_mahal', 'valley', 'nongauss'):
        assert b[key] == pytest.approx(a[key], abs=1e-4)


def test_knn_negentropy_gaussian_and_point_masses():
    from scipy.special import digamma
    # Near 0 for a Gaussian cloud
    rng = np.random.default_rng(3)
    assert abs(nonlinearity._knn_negentropy(rng.standard_normal((4000, 2)))) < 0.05
    # Two point masses: every nearest-neighbor distance is 0, floored at 0.01, so the value is finite
    pts = np.repeat(np.array([[-1.0, -1.0], [1.0, 1.0]]), 500, axis=0)
    h = digamma(1000) - digamma(5) + np.log(np.pi) + 2 * np.log(0.01)
    assert nonlinearity._knn_negentropy(pts) == pytest.approx(np.log(2 * np.pi * np.e) - h, rel=1e-12)


def test_valley_depth_unimodal_and_gap():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(2000)
    assert nonlinearity._valley_depth(x / x.std(ddof=1), 0.0) < 0.05
    x = np.r_[rng.standard_normal(1000) - 5, rng.standard_normal(1000) + 5]
    x = (x - x.mean()) / x.std(ddof=1)
    assert nonlinearity._valley_depth(x, 0.0) > 0.99


def test_embed_cluster_sep_mahal_nan_without_a_two_component_fit():
    # A single spike: every split (two-means or median) isolates the two embedded points containing
    # it, too few for a full-rank covariance, so no 2-component fit is possible; the other
    # statistics are still defined
    y = np.zeros(1000)
    y[500] = 50.0
    out = nonlinearity.embed_cluster(_zscore(y), tau=1, m=2)
    assert out['sep_pp'] == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(out['sep_mahal'])
    assert np.isfinite(out['valley']) and np.isfinite(out['nongauss'])


# ------------------------------------------------------------------ DVV: draws and statistics
def test_dvv_draws_are_valid_reproducible_and_seed_dependent():
    n, n_dv, nsub, ns = 300, 290, 40, 6
    refs, perms = nonlinearity._dvv_draws(3, n, n_dv, nsub, ns)
    assert len(refs) == ns + 1 and len(perms) == ns
    for r in refs:
        assert r.size == nsub and len(set(r.tolist())) == nsub and r.min() >= 0 and r.max() < n_dv
    for p in perms:
        assert sorted(p.tolist()) == list(range(n))
    refs2, perms2 = nonlinearity._dvv_draws(3, n, n_dv, nsub, ns)
    assert all(np.array_equal(a, b) for a, b in zip(refs, refs2))
    assert all(np.array_equal(a, b) for a, b in zip(perms, perms2))
    refs3, _ = nonlinearity._dvv_draws(4, n, n_dv, nsub, ns)
    assert not np.array_equal(refs[0], refs3[0])
    # blocks within one call are distinct draws
    assert not np.array_equal(refs[0], refs[1]) and not np.array_equal(perms[0], perms[1])
    # the stream is the portable generator
    assert np.array_equal(refs[0], bf_random(n_dv, 3 * (2 * ns + 1), 'perm')[:nsub])


def _ar1(n, phi, seed):
    e = np.random.default_rng(seed).standard_normal(n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def test_dvv_reproducible_and_in_range_across_seeds():
    y = _ar1(800, 0.9, 0)
    first = nonlinearity.dvv(y, m=3, num_dvs=40, num_surr=4, random_seed=1)
    again = nonlinearity.dvv(y, m=3, num_dvs=40, num_surr=4, random_seed=1)
    assert first == again
    vals = []
    for seed in range(5):
        o = nonlinearity.dvv(y, m=3, num_dvs=40, num_surr=4, random_seed=seed)
        assert isinstance(o, dict)
        assert o['min'] <= o['mean'] <= o['max']
        assert -1 <= o['dataSurrCorr'] <= 1 and 0 <= o['meanNormCDF'] <= 1
        vals.append(o['rmsDiffSurr'])
    assert len(set(vals)) > 1  # different draws, different surrogate comparison


def test_dvv_target_variance_white_noise_vs_predictable_process():
    wn = np.random.default_rng(9).standard_normal(1500)
    ow = nonlinearity.dvv(wn, m=3, num_dvs=60, num_surr=3, random_seed=0)
    # the targets of white noise are independent of the delay vector: normalized target variance ~ 1
    assert 0.85 < ow['mean'] < 1.15
    ar = _ar1(1500, 0.95, 10)
    oa = nonlinearity.dvv(ar, m=3, num_dvs=60, num_surr=3, random_seed=0)
    # a strongly autocorrelated process is predictable from its recent past: smaller target variance
    assert oa['min'] < 0.5 * ow['min'] or oa['mean'] < ow['mean'] - 0.1


# ------------------------------------------------------------------ largest Lyapunov exponent
def test_largest_lyap_henon_growth_rate():
    # the Henon map (a = 1.4, b = 0.3) has a largest Lyapunov exponent of about 0.42 per iteration: the mean
    # log divergence of nearby trajectories (p1..p5 at steps 1..5) grows by about that much per step,
    # whereas for white noise it saturates at once
    a, b = 1.4, 0.3
    x = [0.1, 0.1]
    for _ in range(1500):
        x.append(1 - a * x[-1] ** 2 + b * x[-2])
    x = np.array(x[102:])
    x = (x - x.mean()) / x.std(ddof=1)
    d = nonlinearity.largest_lyap(x, -1, 10, 1, 3, (1, 2))
    inc = np.diff([d[f'p{i}'] for i in range(1, 6)])
    assert np.all(np.abs(inc[1:] - 0.42) < 0.08), inc
    w = nonlinearity.largest_lyap(np.random.default_rng(0).standard_normal(1500), -1, 10, 1, 3, (1, 2))
    assert abs(w['p5'] - w['p3']) < 0.1
