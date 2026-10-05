"""Model-fit ports (wave 6c): the 'ss'/'arma' paths of steps_ahead / compare_test_sets,
fit_subsegments ('ss', 'arma', 'rand'), and the shared whitening / HMM helpers.

Expected values were generated with MATLAB hctsa (R2026a) on the deterministic series below."""
import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf


def _x(n_samples=400):
    n = np.arange(1, 401)
    u = np.mod(np.sin(n * 12.9898) * 43758.5453, 1)  # deterministic pseudo-noise
    x = np.sin(0.3 * n) + 0.5 * np.cos(1.7 * n) ** 3 + 0.6 * (u - 0.5) + 0.2 * np.sin(0.011 * n ** 1.5)
    x = (x - x.mean()) / x.std(ddof=1)
    return x[:n_samples]


def _check(out, expected, tol=1e-7):
    for key, value in expected.items():
        assert abs(out[key] - value) < tol, key


def test_steps_ahead_ss():
    out = mf.steps_ahead(_x(), 'ss', 'best', 3)
    _check(out, {'stde_h1': 0.515123948444, 'meanabs_h2': 0.402126421171, 'ac1_h3': 0.551491967754,
                 'stde_meanabs_diff': 0.068729959788, 'stde_stddiff': 0.013125743198})
    assert out['stde_ndown'] == 0
    _check(mf.steps_ahead(_x(), 'ss', 2, 3),
           {'stde_h1': 0.988955465476, 'stde_h3': 0.93579420125, 'ac1_h2': 0.237710099915})


def test_compare_test_sets_ss():
    out = mf.compare_test_sets(_x(), 'ss', 2, 'rand', [15, 0.1], 2, 'default')
    _check(out, {'stde_mean': 0.791842451482, 'stde_iqr': 0.160026270653, 'ac1_mean': 0.112570982982,
                 'meane_mean': 0.139714942505, 'stdrat_median': 0.983921846276})


def test_predictor_models_run():
    # every model type gives a predictor; the ARMA fit no longer needs a fallback
    y = _x()
    for model, order in (('ar', 2), ('ar', 'best'), ('arma', [2, 1]), ('ss', 2), ('ss', 'best')):
        predict_errors = mf._fit_predictor_model(y, model, order)
        assert predict_errors is not None
        assert predict_errors(y[:50], 2).shape == (50,)
    with pytest.raises(ValueError):
        mf._fit_predictor_model(y, 'nope', 2)


def test_fit_subsegments_rand_matches_matlab_draws():
    y = _x()
    _check(mf.fit_subsegments(y, 'ar', 2, 'rand', [10, 0.1], 'default'),
           {'fpe_mean': 0.379634519252, 'fpe_range': 0.102596762019, 'a_1_mean': -0.864566529011,
            'a_2_std': 0.048453762369})
    _check(mf.fit_subsegments(y, 'arsbc', None, 'rand', [10, 0.1], 'default'),
           {'orders_mean': 8.2, 'sbcs_std': 0.291284688166, 'sbcs_min': -2.181912461189})


def test_fit_subsegments_ss():
    y = _x()
    _check(mf.fit_subsegments(y, 'ss', 2, 'rand', [10, 0.15], 'default'),
           {'fpe_mean': 0.504523045973, 'fpe_std': 0.038456149331, 'fpe_max': 0.554694513916})
    _check(mf.fit_subsegments(y, 'ss', 'best', 'uniform', [8, 0.2]),
           {'fpe_mean': 0.113022755098, 'fpe_range': 0.070104747615, 'fpe_min': 0.08019030167})


def test_fit_subsegments_arma_and_seed():
    y = _x()
    out = mf.fit_subsegments(y, 'arma', [2, 1], 'uniform', [10, 0.2])
    assert set(out) >= {'fpe_std', 'fpe_mean', 'fpe_max', 'fpe_min', 'fpe_range', 'p_1_std',
                        'p_2_mean', 'q_1_max', 'q_1_min'}
    assert 'q_2_std' not in out and np.all(np.isfinite(list(out.values())))
    # the segments chosen at random depend on the seed only
    a = mf.fit_subsegments(y, 'ar', 2, 'rand', [10, 0.1], 3)
    b = mf.fit_subsegments(y, 'ar', 2, 'rand', [10, 0.1], 3)
    c = mf.fit_subsegments(y, 'ar', 2, 'rand', [10, 0.1], 4)
    assert a == b and a != c
    with pytest.raises(ValueError):
        mf.fit_subsegments(y, 'ar', 2, 'sideways', [10, 0.1])


def test_whiten_rounds_piece_boundaries_as_matlab():
    # lengths at which the piece boundaries fall on halves (BF_Whiten 'ar' picked p1_10 / p1_20)
    for n, ref in ((150, [-0.8921875322, 0.7435597543, -0.2828722015, 0.7759966957]),
                   (270, [-0.8677188156, 0.7301680883, -0.4856759918, 1.9476948228])):
        w = mf._whiten(_x(n), 'ar', 0)
        assert len(w) == n
        np.testing.assert_allclose(w[[0, 7, 100, n - 1]], ref, atol=1e-8)
        assert abs(np.sum(w ** 2) - (n - 1)) < 1e-8


def test_hmm_fit_uses_shared_em_loop():
    y = _x()
    out = mf.hmm_fit(y, 0.7, 3, 0)
    rng = np.random.RandomState(5489)
    model, LL = mf._zg_hmm_fit(y[:int(np.floor(0.7 * len(y)))], 3, rng)
    assert out['nit'] == len(LL)
    assert abs(out['LLtrainpersample'] - np.max(LL) / int(np.floor(0.7 * len(y)))) < 1e-12
    assert abs(out['Cov'] - model.covars_.flatten()[0]) < 1e-12
