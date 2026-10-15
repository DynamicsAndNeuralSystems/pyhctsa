"""Model-fit ports (wave 6c): the 'ss'/'arma' paths of steps_ahead / compare_test_sets,
fit_subsegments ('ss', 'arma', 'rand'), and the shared whitening / HMM helpers.

Expected values of the unchanged features were generated with hctsa (R2026a; random starts from BF_Random,
robust/all) on the deterministic series below."""
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


# The state-space ('ss') features changed definition (stationary Kalman start of the predictions, errors averaged over
# all samples predicted, 2n parameters, horizons from the AIC order of an AR model; see
# cleanroom/native-wp4-report.md). hctsa's values are no longer the reference; accuracy checks of these features
# against simulated systems are in tests/native/test_wp4_sysid.py. The checks below are sanity checks on the
# deterministic series.
def test_steps_ahead_ss():
    out = mf.steps_ahead(_x(), 'ss', 'best', 3)
    assert all(np.isfinite(v) for v in out.values())
    assert 0 < out['stde_h1'] < 1.2 and 0 < out['stde_h3'] < 1.5  # (z-scored series: below the series' own sd)
    out2 = mf.steps_ahead(_x(), 'ss', 2, 3)
    assert all(np.isfinite(v) for v in out2.values()) and 0 < out2['stde_h1'] < 1.2
    assert out['stde_h1'] <= out2['stde_h1'] * 1.05  # (the best of orders 1..10 predicts at least as well)


def test_compare_test_sets_ss():
    out = mf.compare_test_sets(_x(), 'ss', 2, 'rand', [15, 0.1], 2, 'default')
    assert all(np.isfinite(v) for v in out.values()) and 0 < out['stde_mean'] < 1.5


def test_predictor_models_run():
    # every model type gives a predictor; the ARMA fit no longer needs a fallback
    y = _x()
    for model, order in (('ar', 2), ('ar', 'best'), ('arma', [2, 1]), ('ss', 2), ('ss', 'best')):
        predict_errors = mf._fit_predictor_model(y, model, order)
        assert predict_errors is not None
        assert predict_errors(y[:50], 2).shape == (50,)
    with pytest.raises(ValueError):
        mf._fit_predictor_model(y, 'nope', 2)


def test_fit_subsegments_rand_draws():
    y = _x()
    # (hctsa's final prediction errors are divided by the segment length l = 40; they are now means over the
    # l - p predicted samples, so fpe_* are hctsa's times l / (l - p))
    f = 40 / 38
    _check(mf.fit_subsegments(y, 'ar', 2, 'rand', [10, 0.1], 'default'),
           {'fpe_mean': 0.39482533738942843 * f, 'fpe_range': 0.16510385440096348 * f,
            'a_1_mean': -0.88754626825997, 'a_2_std': 0.038102305197980246})
    _check(mf.fit_subsegments(y, 'arsbc', None, 'rand', [10, 0.1], 'default'),
           {'orders_mean': 8.7, 'sbcs_std': 0.2789350693032891, 'sbcs_min': -2.12873710590734})


def test_fit_subsegments_ss():
    y = _x()
    out = mf.fit_subsegments(y, 'ss', 2, 'rand', [10, 0.15], 'default')
    assert all(np.isfinite(v) for v in out.values()) and out['fpe_min'] <= out['fpe_mean'] <= out['fpe_max']
    out = mf.fit_subsegments(y, 'ss', 'best', 'uniform', [8, 0.2])
    assert all(np.isfinite(v) for v in out.values()) and out['fpe_min'] <= out['fpe_mean'] <= out['fpe_max']


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
    # hmm_fit is the shared deterministic multi-start EM (_gaussian_hmm_fit) on the training part
    y = _x()
    n_train = int(np.floor(0.7 * len(y)))
    mu, cov, P, pi, ll = mf._gaussian_hmm_fit(y[:n_train], 3)
    out = mf.hmm_fit(y, 0.7, 3)
    assert abs(out['LLtrainpersample'] - ll / n_train) < 1e-12
    # the training log-likelihood is that of the returned parameters
    assert abs(ll - mf._gaussian_hmm_loglik(y[:n_train], mu, cov, P, pi)) < 1e-9
    assert abs(out['Cov'] - np.ravel(cov)[0]) < 1e-12
    assert mf.hmm_fit(y, 0.7, 3) == out  # deterministic