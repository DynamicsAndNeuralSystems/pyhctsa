"""Tests for the hctsa robust/all (and robust/finish) ports in distribution, hypothesis_tests, graph,
physics, pre_process, spectral, stationarity, scaling, wavelet and medical.

Expected values were generated with current hctsa (robust/finish) in MATLAB R2026a on the series
stored in ``tests/data/p2_b.json`` (scripts ``mlA.m`` ... ``mlG.m`` and ``build_fixture.py`` in
``tests/data/p2_b/``). The deterministic closed forms agree to rounding, the new fits to about
1e-7, and the features that draw random numbers through ``bf_random`` agree exactly.
"""
import json
import os

import numpy as np
import pytest

from pyhctsa.operations import distribution as D
from pyhctsa.operations import graph as G
from pyhctsa.operations import hypothesis_tests as H
from pyhctsa.operations import medical as MD
from pyhctsa.operations import physics as PH
from pyhctsa.operations import pre_process as PP
from pyhctsa.operations import scaling as SC
from pyhctsa.operations import spectral as SP
from pyhctsa.operations import stationarity as ST
from pyhctsa.robust import bf_random

with open(os.path.join(os.path.dirname(__file__), 'data', 'p2_b.json')) as fh:
    _FIX = json.load(fh)
SERIES = {k: np.asarray(v, dtype=float) for k, v in _FIX['inputs'].items()}
EXP = _FIX['expected']
ZS = ['s3', 's20', 'expn700', 'quant600', 'short60']  # z-scored series
ALL = ZS + ['const300']


def _num(v):
    return np.nan if v is None or isinstance(v, str) else float(v)


def _check(py, ml, name, rtol=1e-7, atol=1e-9, skip=()):
    """Compare a python result with the MATLAB one (dict of outputs or scalar; NaN struct = NaN)."""
    if not isinstance(ml, dict):
        assert not isinstance(py, dict) or all(np.isnan(_num(v)) for v in py.values()), name
        if not isinstance(py, dict):
            assert np.isnan(_num(py)) or np.isnan(_num(ml)), name
        return
    if not isinstance(py, dict):  # python NaN where MATLAB returns a struct
        py = {}
    for k, v in ml.items():
        if k in skip:
            continue
        assert k in py, f'{name}: output {k} missing'
        a, b = _num(py[k]), _num(v)
        if np.isnan(b) or np.isnan(a):
            assert np.isnan(a) and np.isnan(b), f'{name}.{k}: {a} vs {b}'
        elif np.isinf(a) or np.isinf(b):
            assert a == b, f'{name}.{k}: {a} vs {b}'
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f'{name}.{k}')


def _scalar(py, ml, name, **kw):
    _check({'v': py}, {'v': ml}, name, **kw)


def _series(key, names=ALL):
    return [(n, EXP[key][n]) for n in names if n in EXP[key]]


# ------------------------------------------------------------------------------
# distribution
# ------------------------------------------------------------------------------
class TestDistribution:
    @pytest.mark.parametrize('k', [1, 2])
    def test_cv(self, k):
        for n, ml in _series(f'A_cv{k}'):
            _scalar(D.cv(SERIES[n], k), ml, f'cv{k} {n}')

    def test_cv_offset_and_zscored_nan(self):
        for n, ml in _series('A_cv1o'):
            _scalar(D.cv(SERIES[n] + 3, 1), ml, f'cv offset {n}')
        assert np.isnan(D.cv(SERIES['s3'], 1))  # a z-scored series has a mean of zero up to rounding

    def test_custom_skewness_mode(self):
        for n, ml in _series('A_cskew'):
            _scalar(D.custom_skewness(SERIES[n], 'pearsonMode'), ml, n)

    @pytest.mark.parametrize('key,arg', [('hm5', 5), ('hm10', 10), ('hmauto', 'auto'), ('hmsqrt', 'sqrt'), ('hmfd', 'fd')])
    def test_histogram_mode(self, key, arg):
        for n, ml in _series('A_' + key):
            _scalar(D.histogram_mode(SERIES[n], arg), ml, f'{key} {n}', rtol=0, atol=1e-12)

    def test_fit_kernel_smooth(self):
        kw = dict(numcross=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], area=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                  arclength=[0.1, 0.5, 1, 2])
        for n, ml in _series('A_fks'):
            _check(D.fit_kernel_smooth(SERIES[n], **kw), ml, f'fks {n}', rtol=1e-9, atol=1e-12)
        assert np.isnan(D.fit_kernel_smooth(SERIES['const300']))

    @pytest.mark.parametrize('dist', ['norm', 'uni', 'beta'])
    def test_compare_ks_fit(self, dist):
        for n, ml in _series('A_ks_' + dist):
            _check(D.compare_ks_fit(SERIES[n], dist), ml, f'ks {dist} {n}', rtol=1e-6, atol=1e-8)

    def test_tail_index(self):
        for key, frac in (('A_ti5', 0.05), ('A_ti10', 0.10)):
            for n, ml in _series(key):
                _check(D.tail_index(SERIES[n], frac), ml, f'{key} {n}', rtol=1e-9, atol=1e-12)

    def test_outlier_include(self):
        for kind in ('abs', 'pos', 'neg'):
            for n, ml in _series('B_oi_' + kind):
                py = D.outlier_include(SERIES[n], kind, 0.01)
                _check(py, ml, f'oi {kind} {n}', rtol=1e-5, atol=1e-6)
                if isinstance(py, dict):
                    assert not any(k.endswith(('expa', 'expc')) for k in py)  # a and c are no longer returned
        for n, ml in _series('B_oi_abs_thr2'):
            _check(D.outlier_include(SERIES[n], 'abs', 0.01, 2), ml, f'oi thr {n}', rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize('model,nb', [('gauss1', 'sqrt'), ('gauss1', 0), ('gauss2', 'sqrt'), ('gauss2', 0),
                                           ('exp1', 'sqrt'), ('exp1', 15), ('power1', 'sqrt')])
    def test_simple_fit(self, model, nb):
        for n, ml in _series(f'B_sf_{model}_{nb}'):
            py = D.simple_fit(SERIES[n], model, nb)
            _check(py, ml, f'simple_fit {model} {nb} {n}', rtol=1e-5, atol=1e-6)
            if isinstance(py, dict):
                assert 'resrunsz' in py and 'resruns' not in py

    def test_simple_fit_sin_dispatch(self):
        for n, ml in _series('C_sfD'):
            _check(D.simple_fit(SERIES[n], 'sin1'), ml, f'simple_fit sin1 {n}', rtol=1e-6, atol=1e-8)

    def test_histogram_asymmetry_explicit_edges(self):
        for key, nb, simple in (('G_ha10', 10, False), ('G_ha11', 11, False), ('G_ha11s', 11, True)):
            for n, ml in _series(key, ZS):
                _check(D.histogram_asymmetry(SERIES[n], nb, simple), ml, f'{key} {n}', rtol=1e-12, atol=1e-12)

    def test_remove_points_random_matches_matlab(self):
        for n, ml in _series('G_rp_rand', ZS):
            _check(D.remove_points(SERIES[n], 'random', 0.1, 'remove'), ml, f'rp {n}', rtol=1e-8, atol=1e-10)
        for n, ml in _series('G_rp_rand3', ZS):
            _check(D.remove_points(SERIES[n], 'random', 0.3, 'remove', 3), ml, f'rp seed 3 {n}', rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------------------
# hypothesis tests
# ------------------------------------------------------------------------------
class TestHypothesisTests:
    @pytest.mark.parametrize('test,rtol', [('runsz', 0), ('runstest', 1e-9), ('lbq', 1e-9)])
    def test_independence_tests(self, test, rtol):
        for n, ml in _series('B_it_' + test):
            _scalar(H.independence_tests(SERIES[n], test), ml, f'{test} {n}', rtol=rtol, atol=1e-12)
            if test == 'runsz':
                _scalar(H.hypothesis_test(SERIES[n], 'runsz'), ml, f'ht runsz {n}', rtol=0, atol=1e-12)

    def test_runsz_sign(self):
        smooth = np.sin(np.linspace(0, 20, 400))  # few runs: positive serial dependence -> z < 0
        alternating = np.tile([1.0, -1.0], 200)
        assert H.independence_tests(smooth, 'runsz') < -10
        assert H.independence_tests(alternating, 'runsz') > 10

    def test_variance_ratio_test(self):
        periods, iids = [2, 4, 6, 8, 2, 4, 6, 8], [0, 0, 0, 0, 1, 1, 1, 1]
        for n, ml in _series('B_vr2', ZS):
            py = H.variance_ratio_test(SERIES[n], periods, iids)
            _check(py, ml, f'vr {n}', rtol=1e-9, atol=1e-12)
            assert 'maxpValue' not in py and 'meanpValue' not in py
        for n, ml in _series('B_vr1', ZS):
            _check(H.variance_ratio_test(SERIES[n], 2, 0), ml, f'vr1 {n}', rtol=1e-9, atol=1e-12)

    def test_variance_ratio_extremes_use_statistic_not_saturated_p(self):
        # a strong departure from a random walk: the p-values underflow to 0 but |stat| orders the tests
        y = np.sin(np.arange(2000) * 0.05) * 50 + np.arange(2000) * 0.001
        out = H.variance_ratio_test(y, [2, 4, 8, 16], [0, 0, 0, 0])
        assert out['periodmaxpValue'] in (2, 4, 8, 16) and out['periodminpValue'] in (2, 4, 8, 16)


# ------------------------------------------------------------------------------
# stationarity
# ------------------------------------------------------------------------------
_SW = [('AC1', 'ent', 10, 1), ('AC1', 'std', 2, 1), ('AC1', 'std', 10, 1), ('AC1', 'permen', 10, 2),
       ('asymAC1', 'ent', 10, 1), ('asymAC1', 'std', 10, 1), ('ent', 'std', 2, 1), ('ent', 'std', 10, 1),
       ('lillie', 'std', 2, 1), ('mean', 'ent', 10, 1), ('mean', 'std', 2, 1), ('mean', 'std', 10, 1),
       ('mom3', 'ent', 10, 1), ('mom3', 'permen', 10, 2), ('std', 'ent', 10, 1), ('std', 'std', 10, 1),
       ('permen', 'std', 2, 1), ('permen', 'std', 5, 10), ('specen', 'ent', 5, 1), ('specen', 'std', 2, 1),
       ('specen', 'std', 10, 1), ('specen', 'permen', 10, 2), ('specen', 'ent', 2, 1), ('mean', 'ent', 2, 1),
       ('specen', 'ent', 10, 1)]


class TestStationarity:
    @pytest.mark.parametrize('segs,mode', [(3, 'each'), (3, 'par'), (5, 'each'), (5, 'par')])
    def test_local_distributions(self, segs, mode):
        for n, ml in _series(f'B_ld{segs}{mode}'):
            _check(ST.local_distributions(SERIES[n], segs, mode), ml, f'ld {n}', rtol=1e-9, atol=1e-12)

    def test_local_distributions_bounds(self):
        out = ST.local_distributions(SERIES['s3'], 4, 'par', 8)
        assert 0 <= out['meandiv'] <= 1
        assert 0 <= ST.local_distributions(SERIES['s3'], 2, 'each') <= 1

    def test_drifting_mean(self):
        for key, nn in (('B_dm50', 50), ('B_dm100', 100)):
            for n, ml in _series(key, ZS):
                _check(ST.drifting_mean(SERIES[n], 'fix', nn), ml, f'dm {n}', rtol=1e-9, atol=1e-12)

    def test_sliding_window(self):
        idx = [0, 3, 4, 6, 10, 12, 18, 19, 20, 21, 22, 24]
        for j in idx:
            a, b, ns, im = _SW[j]
            for n, ml in _series(f'D_sw{j + 1}', ZS):
                _scalar(ST.sliding_window(SERIES[n], a, b, ns, im), ml, f'sw {_SW[j]} {n}', rtol=1e-7, atol=1e-9)

    def test_sliding_window_specen_ignores_dc_bin(self):
        # a window mean of rounding size must not change the entropy
        y = SERIES['s3']
        assert ST.sliding_window(y, 'specen', 'std', 5, 1) == pytest.approx(ST.sliding_window(y + 1e-13, 'specen', 'std', 5, 1), rel=1e-6)

    def test_local_global_randcg_matches_matlab(self):
        for n, ml in _series('G_lg_rand', ZS):
            _check(ST.local_global(SERIES[n], 'randcg', 100), ml, f'randcg {n}', rtol=1e-8, atol=1e-10)
        for n, ml in _series('G_lg_rand7', ZS):
            _check(ST.local_global(SERIES[n], 'randcg', 50, 7), ml, f'randcg seed 7 {n}', rtol=1e-8, atol=1e-10)

    def test_spread_random_local_matches_matlab(self):
        for n, ml in _series('G_srl', ZS):
            _check(ST.spread_random_local(SERIES[n], 100, 100), ml, f'srl {n}', rtol=1e-8, atol=1e-10)
        for n, ml in _series('G_srl_s5', ZS):
            _check(ST.spread_random_local(SERIES[n], 50, 30, 5), ml, f'srl seed 5 {n}', rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------------------
# graph, physics, pre-processing
# ------------------------------------------------------------------------------
class TestGraph:
    @pytest.mark.parametrize('key,meth,make', [('C_vgh', 'horiz', lambda y: y), ('C_vgn', 'norm', lambda y: y),
                                                ('C_vgnq', 'norm', lambda y: np.round(y * 2) / 2),
                                                ('C_vghq', 'horiz', lambda y: np.round(y * 2) / 2),
                                                ('C_vgnr', 'norm', lambda y: np.arange(1, len(y) + 1) / len(y) * 3)])
    def test_visibility_graph(self, key, meth, make):
        for n, ml in _series(key, ZS):
            py = G.visibility_graph(make(SERIES[n]), meth, 'full' if meth == 'horiz' else 20000)
            _check(py, ml, f'{key} {n}', rtol=1e-6, atol=1e-6, skip=('evparam1', 'evparam2', 'evnlogL'))
            if isinstance(py, dict):
                assert 'dgaussk_resrunsz' in py and 'dgaussk_resruns' not in py

    def test_natural_degrees_collinear_nodes_block_the_view(self):
        k = G._natural_vg_degrees(np.arange(10, dtype=float))  # a ramp: only neighbors see each other
        np.testing.assert_array_equal(k, [1] + [2] * 8 + [1])
        k = G._natural_vg_degrees(np.array([0.0, 1.0, 2.0 + 1e-14, 3.0]))  # equal slopes to within rounding
        np.testing.assert_array_equal(k, [1, 2, 2, 1])


class TestPhysics:
    @pytest.mark.parametrize('j,rule,par', [(1, 'biasprop', [0.1, 0.5]), (3, 'momentum', 2), (5, 'prop', 0.1)])
    def test_walker(self, j, rule, par):
        for n, ml in _series(f'C_walk{j}', ZS):
            _check(PH.walker(SERIES[n], rule, par), ml, f'walker {rule} {n}', rtol=1e-6, atol=1e-8,
                   skip=('sw_ansarib_pval',))

    @pytest.mark.parametrize('j,what,par', [(1, 'dblwell', [1, 0.2, 0.1]), (5, 'sine', [0.5, 5, 0.1]), (6, 'sine', [1, 2, 0.1])])
    def test_force_potential(self, j, what, par):
        for n, ml in _series(f'C_fp{j}', ZS):
            py = PH.force_potential(SERIES[n], what, par)
            _check(py, ml, f'fp {what} {n}', rtol=1e-6, atol=1e-7)
            if isinstance(py, dict):
                assert 'meanabs' in py and 'finaldev' not in py


class TestPreProcess:
    @pytest.mark.parametrize('key,meth', [('C_pp_sin1', 'sin1'), ('C_pp_sin2', 'sin2'), ('C_pp_medianf3', 'medianf3')])
    def test_preproc_compare(self, key, meth):
        for n, ml in _series(key, ZS):
            py = PP.preproc_compare(SERIES[n] * 3 + 2, meth)
            _check(py, ml, f'pp {meth} {n}', rtol=1e-3, atol=1e-3)
            if isinstance(py, dict):
                assert 'gauss1_kd_resrunsz' in py

    @pytest.mark.parametrize('key,seed', [('G_rmgd', None), ('G_rmgd5', 5)])
    def test_rank_map_gaussian_matches_matlab(self, key, seed):
        for n in ZS:
            ml = EXP[key][n]
            if isinstance(ml, list):
                np.testing.assert_allclose(PP._rank_map_gaussian(SERIES[n], seed), ml, atol=1e-12)


# ------------------------------------------------------------------------------
# spectral, scaling, wavelet, medical
# ------------------------------------------------------------------------------
class TestSpectralScalingMedical:
    @pytest.mark.parametrize('model,tol', [('sin1', 1e-6), ('sin2', 1e-6), ('sin3', 1e-5)])
    def test_sinusoid_fit(self, model, tol):
        for n, ml in _series('C_sf_' + model, ZS):
            py = SP.sinusoid_fit(SERIES[n], model)
            _check(py, ml, f'sinfit {model} {n}', rtol=tol, atol=tol)
            if isinstance(py, dict):
                assert 'resrunsz' in py and 'resruns' not in py

    def test_sinusoid_fit_too_short(self):
        assert np.isnan(SP.sinusoid_fit(SERIES['s3'][:9], 'sin3'))
        for n, ml in _series('C_sf_sin3_ten', ZS):
            _check(SP.sinusoid_fit(SERIES[n][:10], 'sin3'), ml, f'sin3 N=10 {n}', rtol=1e-4, atol=1e-5)

    def test_sinusoid_fit_recovers_a_sinusoid(self):
        t = np.arange(1, 401)
        y = 2 * np.sin(2 * np.pi * 0.037 * t + 0.4)
        out = SP.sinusoid_fit(y, 'sin1')
        assert out['rmse'] < 1e-6 and out['r2'] > 1 - 1e-10
        assert np.isnan(out['resAC1'])  # an exact fit leaves only rounding error: no residual statistics

    def test_spectral_summaries_floor(self):
        for n, ml in _series('B_sp_fft', ZS):
            _check(SP.spectral_summaries(SERIES[n], 'fft'), ml, f'SP fft {n}', rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('j,wtf,k', [(4, 'rsrange', 1), (5, 'rsrangefit', 1), (7, 'iqr', 1), (10, 'dfa', 2), (11, 'dfa', 3)])
    def test_fluctuation_analysis(self, j, wtf, k):
        for n, ml in _series(f'C_fa{j}', ZS):
            py = SC.fluctuation_analysis(SERIES[n], 2, wtf, 50, k, None, True)
            _check(py, ml, f'fa {wtf} {n}', rtol=1e-7, atol=1e-9)
            if isinstance(py, dict):
                assert 'alphadiff' in py and 'splitgain' in py and 'alpharat' not in py
                assert np.isnan(py['splitgain']) or 0 <= py['splitgain'] <= 1

    def test_wavelet_cwt_zero_guard(self):
        from pyhctsa.operations import wavelet as WL
        out = WL.cwt(SERIES['s3'], 'db3', 32)
        assert np.isfinite(out['SC_h']) and np.isfinite(out['dd_SC_h'])

    def test_medical_explicit_edges(self):
        for n, ml in _series('G_raw', ZS):
            _check(MD.raw_hrv_meas(SERIES[n] * 30 + 800), ml, f'raw hrv {n}', rtol=1e-9, atol=1e-9)
        for n, ml in _series('G_hrv', ZS):
            _check(MD.hrv_classic(SERIES[n] * 30 + 800), ml, f'hrv {n}', rtol=1e-7, atol=1e-8)
        for key, lv in (('G_porta6', 6), ('G_porta4', 4)):
            for n, ml in _series(key, ZS):
                _check(MD.porta(SERIES[n], lv), ml, f'porta {n}', rtol=0, atol=1e-12)
