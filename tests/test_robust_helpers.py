"""Tests for the shared robust helpers in ``pyhctsa.robust`` (ports of hctsa ``BF_*``).

Expected values were generated with current hctsa (branch robust/all) in MATLAB R2026a on
inputs stored in ``tests/data/robust_helpers.json`` (regenerate with the scripts in
``tests/data/robust_helpers/``). ``bf_random`` uniforms and permutations are bit-identical to
MATLAB's ``BF_Random`` (a permutation is the stable argsort of n uniforms); deterministic closed forms agree to rounding; the searches
(``bf_exp_fit`` golden section, ``bf_fit_sinusoids`` zooming grids) agree to the resolution
at which their criterion stops changing (about 1e-7 in the parameters, 1e-12 in the fit).
"""
import json
import os

import numpy as np
import pytest

from pyhctsa.robust import (bf_exp_fit, bf_fit_density_curve, bf_fit_sinusoids, bf_gauss_mix2,
                            bf_half_sample_mode, bf_hist_edges, bf_ks_density, bf_quantile_edges,
                            bf_random, bf_random_seed, bf_residual_stats, bf_runs_z, bf_theil_sen,
                            bf_tie_break_noise)

with open(os.path.join(os.path.dirname(__file__), 'data', 'robust_helpers.json')) as fh:
    _FIX = json.load(fh)
S = {k: np.asarray(v, dtype=float) for k, v in _FIX['inputs'].items()}
E = {k: np.asarray(v, dtype=float) for k, v in _FIX['expected'].items()}


def _eq(py, key, atol=0.0, rtol=0.0):
    """Compare with the MATLAB value, NaNs in the same places."""
    ml = E[key]
    py = np.asarray(py, dtype=float).reshape(ml.shape)
    assert np.array_equal(np.isnan(py), np.isnan(ml)), key
    np.testing.assert_allclose(py, ml, atol=atol, rtol=rtol, equal_nan=True, err_msg=key)


# ------------------------------------------------------------------------------
class TestBFRandom:
    def test_reference_checksums(self):
        # L'Ecuyer's RngStream.c (default state, all 12345): published reference values
        u = bf_random(1000000, [12345] * 6)
        np.testing.assert_array_equal(u[:5], [0.12701112204657714, 0.3185275653967945, 0.30918601558327008,
                                              0.82584686292711362, 0.2216299157820229])
        assert u[999] == 0.98607848680213228
        assert u[99999] == 0.69628910995743587
        assert u[999999] == 0.37578835621568801
        np.testing.assert_array_equal(bf_random(3, [1] * 6),
                                      [0.0003395772237870988, 0.55588071598279964, 0.014204660652803588])

    def test_uniform_matches_matlab_bitwise(self):
        _eq(bf_random(1000, [12345] * 6), 'rand_ref')
        _eq(bf_random(8, 0), 'rand_s0')
        _eq(bf_random(5000, 7), 'rand_s7')
        _eq(bf_random(6, 12345), 'rand_s12345')
        _eq(bf_random(5, [1, 2, 3, 4, 5, 6]), 'rand_vec')

    def test_normal_matches_matlab(self):
        _eq(bf_random(5001, 7, 'normal'), 'rand_normal7', atol=1e-14)  # odd n: last Box-Muller pair is cut
        _eq(bf_random(6, 0, 'normal'), 'rand_normal0', atol=1e-14)

    def test_perm_matches_matlab(self):
        p = bf_random(50, 3, 'perm')
        _eq(p, 'rand_perm3')
        assert sorted(p) == list(range(1, 51))
        _eq(bf_random(1, 0, 'perm'), 'rand_perm0_1')
        _eq(bf_random(4000, 7, 'perm'), 'rand_perm7_4000')
        _eq(bf_random(12, [1, 2, 3, 4, 5, 6], 'perm'), 'rand_perm_vec')
        u = bf_random(50, 3)
        np.testing.assert_array_equal(bf_random(50, 3, 'perm'), np.argsort(u, kind='stable') + 1)

    def test_seed(self):
        py = [bf_random_seed('default'), bf_random_seed(None), bf_random_seed(7.6), bf_random_seed(2.5),
              bf_random_seed(-3), bf_random_seed(5e9 + 0.4), bf_random_seed(0.5), bf_random_seed(42)]
        _eq(py, 'seed_vals')
        assert 0 <= bf_random_seed('none') < 4e9
        with pytest.raises(ValueError):
            bf_random_seed('bogus')

    def test_properties(self):
        assert bf_random(0, 0).size == 0
        u = bf_random(20000, 11)
        assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01
        z = bf_random(20000, 11, 'normal')
        assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
        np.testing.assert_array_equal(bf_random(100, 4), bf_random(100, 4))  # same seed, same stream
        assert not np.array_equal(bf_random(100, 4), bf_random(100, 5))
        with pytest.raises(ValueError):
            bf_random(3, 0, 'bogus')

    def test_leaves_global_state(self):
        np.random.seed(5)
        before = np.random.get_state()[1].copy()
        bf_random(10, 0)
        np.testing.assert_array_equal(np.random.get_state()[1], before)


class TestTieBreakNoise:
    def test_matches_matlab(self):
        _eq(bf_tie_break_noise(S['quant'], 0), 'tb_quant0', atol=1e-20)
        _eq(bf_tie_break_noise(S['quant'], 1), 'tb_quant1', atol=1e-20)
        _eq(bf_tie_break_noise(S['binary'], 5), 'tb_binary', atol=1e-20)

    def test_untouched_when_few_ties(self):
        np.testing.assert_array_equal(bf_tie_break_noise(S['white'], 0), S['white'])
        np.testing.assert_array_equal(bf_tie_break_noise(S['const'], 0), S['const'])  # no scale: unchanged
        np.testing.assert_array_equal(bf_tie_break_noise([3.0], 0), [3.0])

    def test_noise_is_bf_random(self):
        y = S['quant']
        np.testing.assert_allclose((bf_tie_break_noise(y, 1) - y) / (1e-10 * np.std(y, ddof=1)),
                                   bf_random(y.size, 1, 'normal'), atol=1e-4)  # (y itself carries rounding)
        assert np.unique(bf_tie_break_noise(y, 1)).size == y.size  # ties broken

    def test_matrix_shape(self):
        y = np.round(S['ar_pos'][:20]).reshape(4, 5)
        assert bf_tie_break_noise(y, 2).shape == (4, 5)

    def test_information_module_uses_it(self):
        from pyhctsa.operations import information
        np.testing.assert_array_equal(information._tie_break_noise(S['quant'], 1), bf_tie_break_noise(S['quant'], 1))


# ------------------------------------------------------------------------------
class TestRunsAndResiduals:
    NAMES = ['ar_pos', 'ar_neg', 'white', 'trend', 'quant', 'binary', 'mostmax', 'const', 'two', 'small',
             'nan_series', 'hsm_even']

    def test_runs_z(self):
        _eq([bf_runs_z(S[n]) for n in self.NAMES], 'runs', atol=1e-12)

    def test_runs_z_signs(self):
        assert bf_runs_z(S['trend']) < -5  # slowly varying: too few runs
        assert bf_runs_z(S['ar_neg']) > 2  # alternating: too many
        assert np.isnan(bf_runs_z(np.ones(10)))

    def test_residual_stats(self):
        r = S['res']
        _eq(bf_residual_stats(r, np.sum((r - r.mean()) ** 2) * 3), 'resstats', atol=1e-12)
        out = bf_residual_stats(S['res_exact'], 1e6)  # exact fit: nothing but numerical error left
        assert all(np.isnan(out))
        _eq(out, 'resstats_exact')


class TestTheilSen:
    def test_matches_matlab(self):
        _eq(bf_theil_sen(S['x'], S['y_lin_out']), 'ts1', atol=1e-12)
        _eq(bf_theil_sen(S['x_ties'], S['y_ties']), 'ts2', atol=1e-12)
        _eq(bf_theil_sen(S['x'], S['y_decay']), 'ts4', atol=1e-12)

    def test_degenerate_and_robust(self):
        _eq(bf_theil_sen(np.ones(5), np.arange(1.0, 6)), 'ts3')
        slope, icpt = bf_theil_sen(S['x'], S['y_lin_out'])  # true line 2x + 1 with three gross outliers
        assert abs(slope - 2) < 0.1 and abs(icpt - 1) < 0.15
        np.testing.assert_allclose(bf_theil_sen([0, 1, 2, 3], [1, 3, 5, 7]), [2, 1])


# ------------------------------------------------------------------------------
class TestExpFit:
    CASES = ['y_decay', 'y_grow', 'y_nearlin', 'y_step', 'y_lin_out']

    def test_matches_matlab(self):
        rows = []
        for c in self.CASES:
            for wo in (True, False):
                for mr in (20, 5):
                    q = bf_exp_fit(S['x'], S[c], wo, mr)
                    assert list(q) == ['a', 'b', 'c', 'r2', 'adjr2', 'rmse']
                    rows.append([q[k] for k in q])
        rows, ml = np.array(rows), E['exp']
        # the golden-section tail compares SSEs below rounding: the rate agrees to ~1e-7 (flat minimum),
        # r2 and rmse to rounding; a and c inherit the rate's error (large when they are ill-determined)
        np.testing.assert_allclose(rows[:, 1], ml[:, 1], atol=3e-7)
        np.testing.assert_allclose(rows[:, 3:], ml[:, 3:], atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(rows[:, 0], ml[:, 0], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(rows[:, 2], ml[:, 2], rtol=1e-3, atol=1e-5)

    def test_nan_cases(self):
        for q, key in ((bf_exp_fit(S['x'], np.ones(60)), 'exp_const'), (bf_exp_fit([1, 2, 3], [1, 3, 2]), 'exp_small')):
            assert all(np.isnan(v) for v in q.values())
            _eq([q[k] for k in q], key)

    def test_recovers_exponential(self):
        x = np.linspace(0, 2, 40)
        q = bf_exp_fit(x, 2 * np.exp(-3 * x) + 0.5)
        assert abs(q['a'] - 2) < 1e-5 and abs(q['b'] + 3) < 1e-6 and abs(q['c'] - 0.5) < 1e-5
        assert q['r2'] > 1 - 1e-12


class TestDensityFits:
    def test_gauss_mix2(self):
        dx = S['xc'][1] - S['xc'][0]
        _eq(np.concatenate(bf_gauss_mix2(S['xc'], S['p_bimod'], dx)), 'gm_bimod', atol=1e-9)
        _eq(np.concatenate(bf_gauss_mix2(S['xc'], S['p_gauss'], dx)), 'gm_gauss', atol=1e-9)

    def test_gauss_mix2_recovers_components(self):
        w, mu, sg = bf_gauss_mix2(S['xc'], S['p_bimod'], S['xc'][1] - S['xc'][0])
        np.testing.assert_allclose(mu, [-1.5, 1.8], atol=0.1)
        np.testing.assert_allclose(w, [0.6, 0.4], atol=0.05)
        assert mu[0] < mu[1]

    @pytest.mark.parametrize('model, x, p, key', [
        ('gauss', 'xc', 'p_gauss', 'fd_gauss_g'), ('gauss', 'xc', 'p_bimod', 'fd_gauss_b'),
        ('gauss2', 'xc', 'p_bimod', 'fd_gauss2_b'), ('gauss2', 'xc', 'p_gauss', 'fd_gauss2_g'),
        ('exp', 'xe', 'p_exp', 'fd_exp'), ('power', 'xp', 'p_power', 'fd_power'),
        ('exp', 'xc', 'p_gauss', 'fd_exp_g')])
    def test_fit_density_curve(self, model, x, p, key):
        _eq(bf_fit_density_curve(S[x], S[p], model), key, atol=1e-7)

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            bf_fit_density_curve(S['xc'], S['p_gauss'], 'cauchy')


class TestFitSinusoids:
    @pytest.mark.parametrize('name, y, k', [('sin2', 'sin2', 2), ('sin1', 'sin2', 1), ('sinn', 'sin_noise', 3)])
    def test_matches_matlab(self, name, y, k):
        yf, fr = bf_fit_sinusoids(S[y], k)
        _eq(fr, name + '_f', atol=1e-8)  # (zooming grid: ~1e-10 on N = 1000 series)
        _eq(yf, name + '_fit', atol=1e-6)
        assert np.all(np.diff(fr) >= 0)

    def test_recovers_frequencies(self):
        yf, fr = bf_fit_sinusoids(S['sin2'], 2)
        np.testing.assert_allclose(fr, [0.071, 0.23], atol=2e-3)


# ------------------------------------------------------------------------------
class TestKSDensity:
    def test_matches_matlab(self):
        f, xi, h = bf_ks_density(S['bimodal'])
        _eq(f, 'ks_bimodal', atol=1e-13); _eq(xi, 'ks_bimodal_xi', atol=1e-13); _eq(h, 'ks_bimodal_h', atol=1e-15)
        f, xi, h = bf_ks_density(S['skew'], np.linspace(0, 6, 25))
        _eq(f, 'ks_skew', atol=1e-13); _eq(h, 'ks_skew_h', atol=1e-15)
        f, xi, h = bf_ks_density(S['quant'], [-3, 0, 2.5], 0.7)
        _eq(f, 'ks_quant', atol=1e-13); assert h == 0.7
        f, xi, h = bf_ks_density(S['mostmax'])  # MAD = 0: falls back to the standard deviation
        _eq(f, 'ks_mostmax', atol=1e-13); _eq(h, 'ks_mostmax_h', atol=1e-15)
        f, xi, h = bf_ks_density(S['nan_series'], [0, 1])  # NaNs ignored
        _eq(f, 'ks_nan', atol=1e-13); _eq(h, 'ks_nan_h', atol=1e-15)

    def test_constant_is_nan(self):
        f, xi, h = bf_ks_density(S['const'])
        assert np.isnan(h) and np.all(np.isnan(f)) and np.all(np.isnan(xi)) and xi.size == 100

    def test_integrates_to_one(self):
        f, xi, h = bf_ks_density(S['skew'], np.linspace(-20, 40, 6001))
        assert abs(np.trapezoid(f, xi) - 1) < 1e-6


class TestEdges:
    NAMES = ['white', 'quant', 'lattice', 'skew', 'bimodal', 'binary', 'const', 'small', 'mostmax', 'nan_series']

    @pytest.mark.parametrize('name', NAMES)
    @pytest.mark.parametrize('rule', ['auto', 'sqrt', 'sturges', 'fd'])
    def test_hist_edges(self, name, rule):
        _eq(bf_hist_edges(S[name], rule), f'he_{name}_{rule}', atol=1e-14, rtol=1e-14)

    def test_hist_edges_options(self):
        _eq(bf_hist_edges(S['lattice'], 10), 'he_lattice_10', atol=1e-14)
        _eq(bf_hist_edges(S['lattice'], 5, [0, 10]), 'he_lattice_lim', atol=1e-14)
        _eq(bf_hist_edges(S['white'], 'sqrt', [-5, 5]), 'he_white_limrule', atol=1e-14)
        with pytest.raises(ValueError):
            bf_hist_edges(S['white'], 'nope')

    def test_hist_edges_lattice_values_fall_in_upper_bin(self):
        edges = bf_hist_edges(S['lattice'], 10)  # values 0, 0.1, ..., 9.9 on a 10-bin grid
        counts = np.histogram(S['lattice'], edges)[0]
        np.testing.assert_array_equal(counts, 10)

    @pytest.mark.parametrize('name', NAMES)
    def test_quantile_edges(self, name):
        _eq(bf_quantile_edges(S[name], 10), f'qe_{name}_10', atol=1e-14, rtol=1e-14)

    def test_quantile_edges_options(self):
        _eq(bf_quantile_edges(S['quant'], 3), 'qe_quant_3', atol=1e-14)
        _eq(bf_quantile_edges(S['white'], 7), 'qe_white_7', atol=1e-14)
        _eq(bf_quantile_edges(S['lattice'], 4), 'qe_lattice_4', atol=1e-14)
        e = bf_quantile_edges(S['white'], 10)
        counts = np.histogram(S['white'], e)[0]
        assert counts.sum() == S['white'].size and counts.min() >= 24


class TestHalfSampleMode:
    NAMES = ['white', 'quant', 'lattice', 'skew', 'bimodal', 'binary', 'const', 'two', 'small', 'mostmax',
             'nan_series', 'hsm_even', 'hsm_tie', 'hsm_tie2', 'hsm_5']

    def test_matches_matlab(self):
        _eq([bf_half_sample_mode(S[n]) for n in self.NAMES], 'hsm', atol=1e-14)

    def test_cases(self):
        assert bf_half_sample_mode([5.0]) == 5.0
        assert bf_half_sample_mode([1, 2, 4]) == 1.5
        assert bf_half_sample_mode([1, 2, 3]) == 2
        assert abs(bf_half_sample_mode(S['bimodal'])) < 0.6  # the larger mode, near 0
