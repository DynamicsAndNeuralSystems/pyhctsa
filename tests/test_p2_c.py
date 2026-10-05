"""Tests for the model-fitting features redefined in hctsa's robustness work (hctsa robust/all).

Expected values in ``data/p2_c.json`` are outputs of hctsa (MATLAB R2026a, branch robust/all) on the stored
series: z-scored rows of the stationary1000 corpus (``s<row>``), an exact sinusoid and a 30-sample series.
"""
import json
import pathlib

import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf

DATA = json.loads((pathlib.Path(__file__).parent / 'data' / 'p2_c.json').read_text())
SERIES = {k: np.array(v) for k, v in DATA['series'].items()}
REAL = ['s0', 's250', 's700', 's900']
COV = 'covSEiso_covNoise'

# label: (function of the series, series names, relative tolerance, absolute tolerance)
CASES = {
    'h073': (lambda y: mf.hmm_fit(y, 0.7, 3), list(SERIES), 1e-8, 1e-9),
    'h082': (lambda y: mf.hmm_fit(y, 0.8, 2), list(SERIES), 1e-8, 1e-9),
    'cmp': (lambda y: mf.hmm_compare_n_states(y, 0.6, [2, 3, 4]), list(SERIES), 1e-8, 1e-9),
    'loopm': (lambda y: mf.loop_local_simple(y, 'mean'), REAL, 1e-6, 1e-7),
    'arfit18': (lambda y: mf.ar_fit(y, 1, 8, 'sbc'), list(SERIES), 1e-6, 1e-8),
    'arcov3': (lambda y: mf.ar_cov(y, 3), list(SERIES), 1e-8, 1e-9),
    'car05': (lambda y: mf.compare_ar(y, np.arange(1, 11), 0.5), REAL + ['x_short'], 1e-7, 1e-9),
    'carall': (lambda y: mf.compare_ar(y, np.arange(1, 11), 'all'), REAL, 1e-7, 1e-9),
    # the GP fits run the minimize optimizer for 50 evaluations: agreement is to optimizer noise
    # random subsamples and starts now come from BF_Random: the same draws as hctsa
    'fs_ar': (lambda y: mf.fit_subsegments(y, 'arsbc', None, 'rand', [25, 0.1], 'default'), REAL, 1e-8, 1e-9),
    'fs_ar3': (lambda y: mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 3), REAL, 1e-8, 1e-9),
    'cts_ar': (lambda y: mf.compare_test_sets(y, 'ar', 4, 'rand', [25, 0.1], 1, 'default'), REAL, 1e-6, 1e-8),
    'gh_ri2': (lambda y: mf.gp_hyperparameters(y, COV, 1, 200, 'random_i', 4), REAL, 1e-2, 1e-2),
    'gh_rb': (lambda y: mf.gp_hyperparameters(y, COV, 1, 200, 'random_both', 2), REAL, 1e-2, 1e-2),
    'gl_rg': (lambda y: mf.gp_local_prediction(y, COV, 10, 3, 20, 'randomgap', 'default'), REAL, 1e-2, 1e-2),
    'gpfa': (lambda y: mf.gp_fit_across(y, COV, 20), REAL, 1e-2, 1e-2),
    'gphp_first': (lambda y: mf.gp_hyperparameters(y, COV, 1, 200, 'first'), REAL, 1e-2, 1e-2),
    'gplp_fb': (lambda y: mf.gp_local_prediction(y, COV, 10, 3, 20, 'frombefore'), REAL, 1e-2, 1e-2),
}

PARAMS = [(label, name) for label, (_, names, _, _) in CASES.items() for name in names]


def _is_nan_output(out):
    return not isinstance(out, dict) or all(np.isnan(v) for v in out.values())


@pytest.mark.parametrize('label,name', PARAMS)
def test_matches_hctsa_robust(label, name):
    fn, _, rtol, atol = CASES[label]
    expected = DATA['cases'][label][name]
    out = fn(SERIES[name])
    if expected is None:  # hctsa returns NaN
        assert _is_nan_output(out), f'{label} on {name}: hctsa gives NaN, got {out}'
        return
    assert isinstance(out, dict), f'{label} on {name}: expected outputs, got {out}'
    for key, exp in expected.items():
        assert key in out, f'{label} on {name}: missing output {key}'
        if exp is None:
            assert np.isnan(out[key]), f'{label}/{key} on {name}: hctsa gives NaN, got {out[key]}'
        else:
            np.testing.assert_allclose(out[key], exp, rtol=rtol, atol=atol,
                                       err_msg=f'{label}/{key} on {name}')


# ------------------------------------------------------------------------------
# HMM
# ------------------------------------------------------------------------------
def test_hmm_is_deterministic_and_has_no_seed():
    y = SERIES['s250']
    a, b = mf.hmm_fit(y, 0.7, 3), mf.hmm_fit(y, 0.7, 3)
    assert a == b
    assert 'nit' not in a and 'meanP' not in a
    with pytest.raises(TypeError):
        mf.hmm_fit(y, 0.7, 3, random_seed=1)
    with pytest.raises(TypeError):
        mf.hmm_compare_n_states(y, 0.6, [2, 3], random_seed=1)


def test_hmm_fit_keeps_the_best_start():
    y = SERIES['s0'][:700]
    mu, cov, p_matrix, pi, ll = mf._zg_hmm_fit(y, 3)
    xs = np.sort(y)
    for means in (xs[np.ceil(700 * (np.arange(1, 4) - 0.5) / 3).astype(int) - 1],
                  y.mean() + y.std(ddof=1) * np.linspace(-1, 1, 3)):
        for rho in (0.9, 0.5, 0.99):
            p0 = 0.5 * (1 - rho) * np.ones((3, 3)) + (rho - 0.5 * (1 - rho)) * np.eye(3)
            ll_start = mf._zg_hmm_em(y, means, np.var(y, ddof=1), p0, np.ones(3) / 3, 30, 1e-4,
                                     0.01 * np.var(y, ddof=1))[4]
            assert ll[-1] >= ll_start[-1] - 1e-9


def test_hmm_variance_floor():
    # a series sitting on a few repeated values would otherwise drive the variance to zero
    rng = np.random.RandomState(0)
    y = rng.choice([-1.0, 0.0, 1.0], size=600) + 1e-9 * rng.randn(600)
    out = mf.hmm_fit(y, 0.8, 3)
    assert out['Cov'] >= 0.01 * np.var(y[:480], ddof=1) * (1 - 1e-9)
    assert np.isfinite(out['LLtestpersample'])


# ------------------------------------------------------------------------------
# Gaussian process noise floor and constant windows
# ------------------------------------------------------------------------------
def test_gp_noise_is_bounded_below():
    t = np.arange(1, 61, dtype=float)
    yt = np.sin(t / 8)  # smooth: the unbounded fit drives the noise to e^-12 or less
    theta = mf._gp_learn_hyperp(t, yt, mf.CovSEisoNoise)
    floor = np.log(0.01 * np.std(yt, ddof=1))
    assert theta[2] >= floor - 1e-12 and theta[3] >= floor - 1e-12


def test_gp_local_prediction_skips_constant_windows():
    rng = np.random.RandomState(3)
    y = np.r_[np.zeros(400), rng.randn(600)]  # the first windows have constant training data
    out = mf.gp_local_prediction(y, COV, 10, 3, 20, 'frombefore')
    assert all(np.isfinite(v) for v in out.values())
    out_const = mf.gp_local_prediction(np.ones(500), COV, 10, 3, 20, 'frombefore')
    assert all(np.isnan(v) for v in out_const.values())


# ------------------------------------------------------------------------------
# AR fits
# ------------------------------------------------------------------------------
def test_exactly_predictable_series_give_nan():
    t = np.arange(1000)
    y = np.sin(2 * np.pi * t / 40)
    y = (y - y.mean()) / y.std(ddof=1)
    assert np.isnan(mf.ar_cov(y, 3)) and np.isnan(mf.ar_fit(y, 1, 8, 'sbc'))
    assert np.isnan(mf.ar_cov(y, 2))  # a sinusoid is an exact AR(2) process


def test_compare_ar_short_series_is_nan():
    rng = np.random.RandomState(1)
    assert np.isnan(mf.compare_ar(rng.randn(30), np.arange(1, 11), 0.5))
    assert isinstance(mf.compare_ar(rng.randn(300), np.arange(1, 11), 0.5), dict)


# ------------------------------------------------------------------------------
# FC_LoopLocalSimple
# ------------------------------------------------------------------------------
def test_loop_local_simple_outputs():
    out = mf.loop_local_simple(SERIES['s0'], 'mean')
    for gone in ('sws_fexp_a', 'sws_fexp_c'):
        assert gone not in out
    for kept in ('sws_fexp_b', 'sws_fexp_r2', 'sws_fexp_adjr2', 'sws_fexp_rmse'):
        assert kept in out and np.isfinite(out[kept])
    assert 0 <= out['sws_fexp_r2'] <= 1


def test_gp_matern_noise_position_and_floor():
    # a degree-parameterized component has two hyperparameters: covNoise follows at index 2
    comps = [('covMaterniso', 3), ('covNoise', None)]
    assert mf._gp_noise_pos(comps) == [2]
    t = np.arange(1, 61, dtype=float)
    yt = np.sin(t / 8)
    init = mf._gp_init_hyp(comps, t)
    assert init[2] == np.log(0.1) and init[0] == np.log(1.0) and init[1] == 0
    out = mf.gp_hyperparameters(SERIES['s0'], 'covMaterniso3_covNoise', 1, 200, 'first')
    assert out['logh3'] >= np.log(0.01 * np.std(SERIES['s0'][:200], ddof=1)) - 1e-9


def test_random_draws_are_portable_and_leave_global_state():
    y = SERIES['s250']
    state = np.random.get_state()[1].copy()
    a = mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 5)
    b = mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 5)
    c = mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 6)
    assert a == b and a != c
    assert mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], None) == \
        mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 'default')
    np.testing.assert_array_equal(np.random.get_state()[1], state)
