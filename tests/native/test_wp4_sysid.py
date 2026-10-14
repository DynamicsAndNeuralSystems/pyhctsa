"""Accuracy tests for the model-fitting and system-identification helpers (model_fit.py).

Checks use analytic results, simulations with known parameters, closed forms, and the reference
implementations in numpy / scipy / statsmodels. No test compares with another implementation of the
same features.
"""
import numpy as np
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from pyhctsa.operations import model_fit as mf
from pyhctsa.operations.model_fit import (
    _ar_fb, _arcov, _arch_test_pvalues, _arx_losses, _even_split_points, _kstep_residuals,
    _kstep_residuals_ss, _lbq_test_pvalues, _n4_arx_order, _n4_core, _n4_fpe, _n4_horizons,
    _n4_state_space, _defined_errors, _diffuse_predictions, _nanmax_or_nan, _nanmin_or_nan, _poly_to_innovations, _prediction_errors,
    _ss_initial_state, _stabilize_matrix,
)


def _arma(a, c, n, seed, burn=500):
    """Simulate A(q) y = C(q) e with unit-variance Gaussian innovations."""
    rng = np.random.default_rng(seed)
    return lfilter(c, a, rng.normal(size=n + burn))[burn:]


# ------------------------------------------------------------------ index grids (A3)
def test_even_split_points_exact_ties_and_bounds():
    # N = 11, 2 pieces: the boundary 5.5 rounds up to 6 (half up), never to 5
    assert list(_even_split_points(0, 11, 3)) == [0, 6, 11]
    assert list(_even_split_points(0, 5, 2)) == [0, 5]
    # an exact integer grid point is never moved by floating-point error
    for b in range(2, 200):
        pts = _even_split_points(1, b, 3, 'floor')
        assert pts[0] == 1 and pts[-1] == b and pts[1] == 1 + (b - 1) // 2
    for n in (2, 5, 20):
        pts = _even_split_points(1, 1000, n)
        assert pts[0] == 1 and pts[-1] == 1000 and np.all(np.diff(pts) >= 0)
        np.testing.assert_allclose(pts, np.linspace(1, 1000, n), atol=0.5 + 1e-9)
    assert len(_even_split_points(1, 10, 0)) == 0 and list(_even_split_points(4, 10, 1)) == [4]
    with pytest.raises(ValueError):
        _even_split_points(0, 10, 3, 'nearest')


def test_nanmax_nanmin():
    a = np.array([1.0, np.nan, 3.0])
    assert _nanmax_or_nan(a) == 3 and _nanmin_or_nan(a) == 1
    assert np.isnan(_nanmax_or_nan(np.full(3, np.nan))) and np.isnan(_nanmin_or_nan([np.nan]))
    np.testing.assert_array_equal(_nanmax_or_nan(np.array([[1, np.nan], [np.nan, np.nan]]), axis=0), [1, np.nan])


# ------------------------------------------------------------------ AR fits (A30, A31)
def test_ar_fb_recovers_ar2_and_fpe_formula():
    y = _arma([1, -0.6, 0.2], [1], 20000, 0)
    a, fpe = _ar_fb(y, 2)
    np.testing.assert_allclose(a, [-0.6, 0.2], atol=0.03)       # polynomial convention: 1 + a1 q^-1 + ...
    assert fpe == pytest.approx(1.0 * (1 + 2 / 20000) / (1 - 2 / 20000), rel=0.03)  # innovation variance 1
    # the FPE uses the mean squared residual over the N - p samples with a full set of past values
    phi, sse = mf._fb_lsq(y, 2)
    v = sse / (len(y) - 2)
    assert fpe == pytest.approx(v * (1 + 2 / len(y)) / (1 - 2 / len(y)), rel=1e-12)


def test_arcov_equals_autoreg_least_squares():
    for seed, p in ((1, 1), (2, 3), (3, 5)):
        y = _arma([1, -0.5, 0.3], [1, 0.4], 600, seed)
        phi, nv = _arcov(y, p)
        res = AutoReg(y, lags=p, trend='n').fit()
        np.testing.assert_allclose(phi, res.params, atol=1e-10)
        assert nv == pytest.approx(res.sigma2, rel=1e-10)
        resid = y[p:] - np.column_stack([y[p - k:len(y) - k] for k in range(1, p + 1)]) @ phi
        assert nv == pytest.approx(np.mean(resid ** 2) * (len(y) - p) / (len(y) - p), rel=1e-10)
    with pytest.raises(ValueError):
        _arcov(np.ones(3), 3)


def test_ar_cov_residuals_cover_the_fitted_samples():
    y = _arma([1, -0.6, 0.2], [1], 800, 4)
    out = mf.ar_cov(y, 2)
    phi, nv = _arcov(y, 2)
    err = np.column_stack([y[1:-1], y[:-2]]) @ phi - y[2:]
    assert out['meane'] == pytest.approx(np.mean(err), rel=1e-10)
    assert out['stde'] == pytest.approx(np.std(err, ddof=1), rel=1e-10)
    assert out['a2'] == pytest.approx(-phi[0], rel=1e-12)


def test_arx_losses_true_order_matches_innovation_variance():
    y = _arma([1, -0.6, 0.2], [1], 30000, 5)
    orders = [1, 2, 3, 4]
    loss, n_scored = _arx_losses(y, y, orders)
    assert n_scored == len(y) - 4
    assert loss[1] == pytest.approx(1.0, rel=0.03)
    assert loss[0] > loss[1] * 1.03                              # too low an order is worse
    assert np.all(np.diff(loss) <= 1e-12)                        # in-sample loss never rises with order
    # equals the mean squared residual of a least-squares fit on the same samples
    m = 4
    for na, l in zip(orders, loss):
        X = np.column_stack([y[m - k:len(y) - k] for k in range(1, na + 1)])
        r = y[m:] - X @ np.linalg.lstsq(X, y[m:], rcond=None)[0]
        assert l == pytest.approx(np.mean(r ** 2), rel=1e-10)


def test_arx_losses_out_of_sample_and_errors():
    y = _arma([1, -0.7, 0.1], [1], 6000, 6)
    loss, n = _arx_losses(y[:3000], y[3000:], [1, 2, 6])
    assert n == 3000 - 6 and loss[1] == pytest.approx(1.0, rel=0.1)
    with pytest.raises(ValueError):
        _arx_losses(np.arange(10.0), np.arange(10.0), [1, 2, 3, 4, 5, 6])


def test_compare_ar_properties():
    y = _arma([1, -0.6, 0.2], [1], 3000, 7)
    y = (y - y.mean()) / y.std(ddof=1)
    out = mf.compare_ar(y, np.arange(1, 11), 'all')
    assert out['best_n'] >= 2 and out['aic_n'] == 2
    assert 0 < out['propgain1min'] < 1 and 0 < out['medonmax'] <= 1
    out_half = mf.compare_ar(y, np.arange(1, 11), 0.5)
    assert out_half['aic_n'] in (2, 3)
    short = mf.compare_ar(y[:20], np.arange(1, 11), 0.5)           # too short for the largest order
    assert all(np.isnan(v) for v in short.values()) if isinstance(short, dict) else np.isnan(short)


# ------------------------------------------------------------------ ARCH / Ljung-Box (A32)
def _arch_reference(x, lag):
    x2 = x ** 2
    from scipy.stats import chi2
    win = np.lib.stride_tricks.sliding_window_view(x2, lag + 1)
    target = win[:, lag]
    design = np.column_stack([np.ones(len(target)), win[:, :lag]])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    r2 = 1 - np.sum((target - design @ beta) ** 2) / np.sum((target - target.mean()) ** 2)
    return chi2.sf(len(target) * r2, lag)


def test_arch_pvalues_textbook_regression():
    rng = np.random.default_rng(8)
    x = rng.normal(size=400)
    p = _arch_test_pvalues(x, 6)
    np.testing.assert_allclose(p, [_arch_reference(x, L) for L in range(1, 7)], rtol=1e-9, atol=1e-12)


def test_arch_detects_volatility_clustering_and_not_white_noise():
    rng = np.random.default_rng(9)
    n = 3000
    e, h = rng.normal(size=n), np.ones(n)
    x = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.2 + 0.7 * x[t - 1] ** 2
        x[t] = np.sqrt(h[t]) * e[t]
    assert np.all(_arch_test_pvalues(x, 5) < 1e-6)
    assert np.mean(_arch_test_pvalues(rng.normal(size=n), 20) < 0.01) < 0.1


def test_lbq_pvalues_closed_form_and_detection():
    from scipy.stats import chi2
    rng = np.random.default_rng(10)
    x = _arma([1, -0.5], [1], 500, 11)
    n = len(x)
    xc = x - x.mean()
    rho = np.array([xc[k:] @ xc[:n - k] for k in range(1, 11)]) / (xc @ xc)
    q = n * (n + 2) * np.cumsum(rho ** 2 / (n - np.arange(1, 11)))
    np.testing.assert_allclose(_lbq_test_pvalues(x, 10), chi2.sf(q, np.arange(1, 11)), rtol=1e-8, atol=1e-14)
    assert np.all(_lbq_test_pvalues(x, 10) < 1e-6)
    assert np.mean(_lbq_test_pvalues(rng.normal(size=2000), 20) < 0.01) < 0.1


def test_arch_lbq_degenerate_input():
    const = np.ones(100)
    assert np.all(np.isnan(_arch_test_pvalues(const))) and np.all(np.isnan(_lbq_test_pvalues(const)))
    bad = np.r_[np.ones(10), np.nan, np.zeros(10)]
    assert np.all(np.isnan(_arch_test_pvalues(bad, 3))) and np.all(np.isnan(_lbq_test_pvalues(bad, 3)))
    with pytest.raises(ValueError):
        _arch_test_pvalues(np.arange(21.0))
    with pytest.raises(ValueError):
        _lbq_test_pvalues(np.arange(10.0), 10)


# ------------------------------------------------------------------ stabilization (A29)
def test_stabilize_matrix_reflects_to_conjugate_reciprocal():
    stable = np.array([[0.5, 0.2], [-0.1, 0.9]])
    assert np.array_equal(_stabilize_matrix(stable), stable)
    out = _stabilize_matrix(np.array([[1.25]]))
    assert out[0, 0] == pytest.approx(0.8, rel=1e-14)
    # a rotation-scaling block: eigenvalues 1.5 exp(+-i 0.7) -> exp(+-i 0.7) / 1.5
    th = 0.7
    m = 1.5 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    st = _stabilize_matrix(m)
    assert np.isrealobj(st)
    ev = np.sort_complex(np.linalg.eigvals(st))
    ref = np.sort_complex(np.array([np.exp(1j * th), np.exp(-1j * th)]) / 1.5)
    np.testing.assert_allclose(ev, ref, atol=1e-12)
    rng = np.random.default_rng(0)
    big = rng.normal(size=(5, 5))
    assert np.max(np.abs(np.linalg.eigvals(big))) > 1.2
    assert np.max(np.abs(np.linalg.eigvals(_stabilize_matrix(big)))) <= 1 + 1e-9


# ------------------------------------------------------------------ prediction from the stationary state (init='stationary')
def test_stationary_prediction_ar1_closed_form():
    phi = 0.8
    y = _arma([1, -phi], [1], 200, 12)
    a, k, c = _poly_to_innovations([1, -phi], [1])
    for steps in (1, 2, 3):
        err = _prediction_errors((a, k, c), y, steps, init='stationary')
        expect = np.zeros(len(y))
        expect[steps:] = phi ** steps * y[:-steps]
        np.testing.assert_allclose(err, expect - y, atol=1e-12)    # prior mean 0 before any data
    # the same through the polynomial interface
    np.testing.assert_allclose(_kstep_residuals(np.array([1, -phi]), np.array([1.0]), y, 1)[1:],
                               (phi * y[:-1] - y[1:]), atol=1e-12)     # (the default, diffuse start)


def test_stationary_prediction_matches_exact_arma_filter():
    """The time-varying Kalman prediction equals the exact (stationary-start) ARMA(1,1) filter."""
    phi, theta = 0.7, 0.4
    y = _arma([1, -phi], [1, theta], 300, 13)
    res = ARIMA(y, order=(1, 0, 1), trend='n').smooth([phi, theta, 1.0])
    one_step = np.asarray(res.fittedvalues)
    model = _poly_to_innovations([1, -phi], [1, theta])
    err1 = _prediction_errors(model, y, 1, init='stationary')
    np.testing.assert_allclose(y + err1, one_step, atol=1e-9)
    # k steps ahead: yhat(t | t-k) = phi^(k-1) * yhat(t-k+1 | t-k)
    for k in (2, 3):
        errk = _prediction_errors(model, y, k, init='stationary')
        np.testing.assert_allclose((y + errk)[k - 1:], phi ** (k - 1) * one_step[:len(y) - k + 1], atol=1e-9)
        np.testing.assert_allclose((y + errk)[:k - 1], 0, atol=1e-12)


def test_prediction_initial_state_rules():
    phi = 0.9
    y = _arma([1, -phi], [1], 400, 14)
    model = _poly_to_innovations([1, -phi], [1])
    stat = _prediction_errors(model, y, 1, init='stationary')
    diff = _prediction_errors(model, y, 1)
    zero = _prediction_errors(model, y, 1, init='zero')
    est = _prediction_errors(model, y, 1, init='estimate')
    # for an AR(1) the first prediction is the mean (0) under both rules and the later ones coincide
    np.testing.assert_allclose(stat, zero, atol=1e-12)
    # the diffuse start predicts nothing for sample 0 (one state to determine) and then agrees: an AR(1) state
    # is known exactly after one observation
    assert np.isnan(diff[0]) and not np.any(np.isnan(diff[1:]))
    np.testing.assert_allclose(diff[1:], stat[1:], atol=1e-10)
    # the estimated initial state removes the first error (a free parameter), nothing else
    assert abs(est[0]) < 1e-12 and np.sum(est ** 2) <= np.sum(stat ** 2) + 1e-12
    with pytest.raises(ValueError):
        _prediction_errors(model, y, 1, init='nope')
    # a model without stationary distribution falls back to the zero state
    unstable = _poly_to_innovations([1, -1.02], [1])
    np.testing.assert_allclose(_prediction_errors(unstable, y, 1, init='stationary'),
                               _prediction_errors(unstable, y, 1, init='zero'))


def test_stationary_start_first_error_variance():
    """Over many realizations the first one-step error has the stationary variance C P C' + 1 = 1/(1 - a^2) for an AR(1)."""
    phi = 0.8
    rng = np.random.default_rng(15)
    model = _poly_to_innovations([1, -phi], [1])
    first, later = [], []
    for _ in range(3000):
        y = lfilter([1], [1, -phi], rng.normal(size=300))[100:]
        # (the realization starts in the stationary regime)
        e = _prediction_errors(model, y, 1, init='stationary')
        first.append(e[0])
        later.append(e[5])
    # prediction of sample 0 is the mean: error variance 1/(1 - phi^2); later samples: innovation variance 1
    assert np.var(first) == pytest.approx(1 / (1 - phi ** 2), rel=0.1)
    assert np.var(later) == pytest.approx(1.0, rel=0.1)


def test_prediction_similarity_invariance_and_degenerate():
    rng = np.random.default_rng(3)
    a = np.array([[0.8, 0.3], [-0.2, 0.5]])
    kg, cv = np.array([0.5, 0.1]), np.array([1.0, 0.5])
    y = rng.normal(size=60)
    t = np.array([[1.0, 0.4], [-0.3, 2.0]])
    ti = np.linalg.inv(t)
    for k in (1, 3):
        np.testing.assert_allclose(_kstep_residuals_ss(a, kg, cv, y, k),
                                   _kstep_residuals_ss(t @ a @ ti, t @ kg, cv @ ti, y, k), atol=1e-9)
    np.testing.assert_allclose(_kstep_residuals(np.array([1.0]), np.array([1.0]), y, 2), -y)  # no states
    y_nan = y.copy()
    y_nan[5] = np.nan
    assert np.all(np.isnan(_kstep_residuals_ss(a, kg, cv, y_nan, 1)))
    x0, err = _ss_initial_state(a, kg, cv, y[:4], 10)             # horizons beyond the data: free run only
    np.testing.assert_allclose(y[:4] - err, mf._ss_predict(a, kg, cv, y[:4], x0, 10), atol=1e-10)


# ------------------------------------------------------------------ N4SID (A28)
def test_n4_arx_order_and_horizons():
    y = _arma([1, -0.6, 0.2], [1], 2000, 16)
    assert 2 <= _n4_arx_order(y) <= 6
    r, s = _n4_horizons(y, 2)
    assert r == s >= 3
    # the data matrix keeps at least as many columns as rows, shrinking the horizons for short series
    for n_obs in (30, 60, 120):
        r, s = _n4_horizons(y[:n_obs], 2)
        assert (s == 3) or (n_obs - r - s + 1 >= r + s + 1)
    # horizons are never below order + 1
    assert _n4_horizons(y, 6)[1] >= 7


def test_n4sid_recovers_poles_and_innovation_variance():
    a_poly, c_poly = [1, -0.9, 0.5], [1, 0.4]     # poles 0.45 +- 0.54i
    true_poles = np.sort_complex(np.roots(a_poly))
    for seed in (17, 18):
        y = _arma(a_poly, c_poly, 8000, seed)
        fit = _n4_state_space(y, 2)
        poles = np.sort_complex(np.linalg.eigvals(fit['A']))
        np.testing.assert_allclose(poles, true_poles, atol=0.06)
        assert fit['loss'] == pytest.approx(1.0, rel=0.05)         # one-step loss = innovation variance
        assert np.all(fit['C'] >= 0) and fit['K'].shape == (2, 1)


def test_n4sid_best_order_selection_prefers_true_order():
    y = _arma([1, -0.9, 0.5], [1, 0.4], 3000, 20)
    assert _n4_state_space(y, 'best')['order'] in (2, 3, 4)
    wn = _arma([1], [1], 3000, 21)
    assert _n4_state_space(wn, 'best')['order'] <= 3


def test_n4_core_noiseless_two_state_system():
    """Noiseless data from a damped oscillator: the identified state matrix has the true poles and the
    model predicts the series."""
    r0, w0 = 0.95, 0.3
    t = np.arange(400)
    y = r0 ** t * np.cos(w0 * t) + 0.5 * r0 ** t * np.sin(w0 * t)
    # make the process slightly noisy at machine-precision scale so the Riccati equation is regular
    y = y + 1e-8 * np.random.default_rng(22).normal(size=len(y))
    fit = _n4_core(y, 2, 4, 4)
    poles = np.sort_complex(np.linalg.eigvals(fit['A']))
    ref = np.sort_complex(np.array([r0 * np.exp(1j * w0), r0 * np.exp(-1j * w0)]))
    np.testing.assert_allclose(poles, ref, atol=1e-4)
    # (the decaying signal is not stationary: score it with the estimated initial state)
    err = _prediction_errors((fit['A'], fit['K'][:, 0], fit['C'][0]), y, 1, init='estimate')
    assert np.mean(err ** 2) < 1e-6 * np.var(y)


def test_n4_fpe_and_errors():
    assert _n4_fpe(2.0, 2, 100) == pytest.approx(2.0 * 1.04 / 0.96)    # d = 2n
    assert _n4_fpe(1.0, 8, 15) == np.inf
    rng = np.random.default_rng(23)
    y = lfilter([1, 0.4], [1, -0.7], rng.normal(size=300))
    with pytest.raises(ValueError):
        _n4_state_space(np.ones(100), 2)
    with pytest.raises(ValueError):
        _n4_state_space(np.r_[y[:50], np.nan], 2)
    with pytest.raises(ValueError):
        _n4_state_space(y[:8], 3)


def test_state_space_n4sid_outputs():
    y = _arma([1, -0.9, 0.5], [1, 0.4], 4000, 24)
    y = (y - y.mean()) / y.std(ddof=1)
    out = mf.state_space_n4sid(y, 2, 0.5, 1)
    n_obs = len(y)
    assert out['np'] == 2 * 2 + 2 * 2 and out['Ts'] == 1
    assert out['noisevar'] == pytest.approx(out['lossfn'] * n_obs / (n_obs - 4), rel=1e-12)
    assert out['fpe'] == pytest.approx(out['lossfn'] * (1 + 4 / n_obs) / (1 - 4 / n_obs), rel=1e-12)
    # held-out one-step errors of the true-order model have the innovation scale
    assert out['stde'] == pytest.approx(np.sqrt(out['lossfn']), rel=0.1)
    assert abs(out['ac1']) < 0.1                                  # the one-step errors of the true order are white
    assert mf.state_space_n4sid(y, 'best', 0.5, 1)['bestorder'] >= 1
    short = mf.state_space_n4sid(y[:10], 2)
    assert all(np.isnan(v) for v in short.values())


def test_state_space_comp_order_uses_two_n_parameters():
    y = _arma([1, -0.9, 0.5], [1, 0.4], 800, 25)
    out = mf.state_space_comp_order(y, 4)
    n_obs = len(y)
    losses = [_n4_state_space(y, k)['loss'] for k in range(1, 5)]
    aic = np.log(losses) + 2 * 2 * np.arange(1, 5) / n_obs
    assert out['minaic'] == pytest.approx(aic.min(), rel=1e-10) and out['aicopt'] == int(np.argmin(aic)) + 1
    assert out['lossfnopt'] == int(np.argmin(losses)) + 1


# ------------------------------------------------------------------ the predictor features
def test_steps_ahead_ar1_ratio_to_trivial_predictor():
    phi = 0.8
    y = _arma([1, -phi], [1], 20000, 26)
    y = (y - y.mean()) / y.std(ddof=1)
    out = mf.steps_ahead(y, 'ar', 1, 4)
    # one step ahead: model error sd sqrt(1 - phi^2); best trivial predictor (previous value) sqrt(2 (1 - phi))
    assert out['stde_h1'] == pytest.approx(np.sqrt(1 - phi ** 2) / np.sqrt(2 * (1 - phi)), rel=0.03)
    assert out['stde_ndown'] == 0 and out['stde_meandiff'] > 0      # the model's error grows with the horizon
    for model, order in (('arma', [1, 1]), ('ss', 2), ('ss', 'best')):
        o = mf.steps_ahead(y[:3000], model, order, 3)
        assert o['stde_h1'] == pytest.approx(out['stde_h1'], rel=0.1) and np.all(np.isfinite(list(o.values())))


def test_compare_test_sets_one_step_error_is_innovation_scale():
    y = _arma([1, -0.8], [1], 2000, 27)
    sd_innov = 1.0
    out = mf.compare_test_sets(y, 'ar', 1, 'rand', [60, 30], 1, 1)
    assert out['stde_mean'] == pytest.approx(sd_innov, rel=0.1)
    assert abs(out['ac1_mean']) < 0.15
    assert out['stdrat_median'] < 1.0               # one-step predictions are smoother than the data
    out_ss = mf.compare_test_sets(y, 'ss', 1, 'uniform', [40, 30], 1)
    assert out_ss['stde_mean'] == pytest.approx(sd_innov, rel=0.15)


def test_fit_subsegments_ar_fpe_is_innovation_variance():
    y = _arma([1, -0.6, 0.2], [1], 20000, 28)
    out = mf.fit_subsegments(y, 'ar', 2, 'uniform', [20, 1000])
    assert out['fpe_mean'] == pytest.approx(1.0 * (1 + 4 / 1000), rel=0.1)
    out_ss = mf.fit_subsegments(y[:6000], 'ss', 2, 'uniform', [8, 600])
    assert out_ss['fpe_mean'] == pytest.approx(1.0, rel=0.1)


def test_fit_subsegments_arcrosspred_square_matrix_scale():
    y = _arma([1, -0.6, 0.2], [1], 5000, 29)
    out = mf.fit_subsegments(y, 'arcrosspred', 2, 'uniform', 5)
    # every model predicts every segment with a root-mean-square error near the innovation sd (1)
    assert out['range'] < 0.2 and out['stdmean'] < 0.1 and np.isfinite(out['mineig'])
    short = mf.fit_subsegments(y[:40], 'arcrosspred', 2, 'uniform', 5)             # segments too short
    assert all(np.isnan(v) for v in short.values()) if isinstance(short, dict) else np.isnan(short)


# ------------------------------------------------------------------ diffuse initial state (C4, revised)
def _statsmodels_diffuse_forecasts(a, k, c, y):
    """One-step forecasts of the innovations-form model from statsmodels' exact diffuse state-space filter.

    The model is written in Harvey's form with the state (x(t), e(t)): x(t+1) = A x(t) + K e(t),
    y(t) = C x(t) + e(t); the first n states are diffuse and e(0) is N(0, 1)."""
    from statsmodels.tsa.statespace.initialization import Initialization
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    n = len(k)
    mod = MLEModel(y, k_states=n + 1, k_posdef=1)
    mod['design', 0, :n] = c
    mod['design', 0, n] = 1.0
    mod['obs_cov', 0, 0] = 0.0
    mod['transition', :n, :n] = a
    mod['transition', :n, n] = k
    mod['selection', n, 0] = 1.0
    mod['state_cov', 0, 0] = 1.0
    init = Initialization(n + 1)
    init.set((0, n), 'diffuse')
    init.set((n, n + 1), 'known', constant=[0.0], stationary_cov=[[1.0]])
    mod.ssm.initialize(init)
    res = mod.smooth([])
    return np.asarray(res.filter_results.forecasts[0]), res.filter_results.nobs_diffuse


@pytest.mark.parametrize("a_poly,c_poly", [
    ([1, -0.7], [1, 0.4]),                  # ARMA(1,1): one state
    ([1, -0.7, 0], [1, 0.4, 0]),            # the same process with a redundant second state
    ([1, -1.5, 0.6], [1, 0.3, -0.2]),       # ARMA(2,2)
    ([1, -1.0], [1, -0.5]),                 # local level (integrated MA(1)): A = 1
])
def test_diffuse_filter_matches_statsmodels(a_poly, c_poly):
    y = _arma(a_poly if a_poly[-1] != 0 and a_poly[1] != -1.0 else [1, -0.9], c_poly, 150, 30) + 50.0
    a, k, c = _poly_to_innovations(a_poly, c_poly)
    ref, n_diffuse = _statsmodels_diffuse_forecasts(a, k, c, y)
    pred = _diffuse_predictions(a, k, c, y, 1)
    undefined = np.isnan(pred)
    assert undefined.sum() == n_diffuse and np.all(undefined[:n_diffuse])    # as many as the diffuse states
    np.testing.assert_allclose(pred[~undefined], ref[~undefined], atol=1e-7, rtol=1e-9)


def test_diffuse_random_walk_with_large_offset():
    """With an unknown initial level, a random walk's one-step errors are its innovations after the first sample,
    whatever the offset."""
    rng = np.random.default_rng(31)
    e = rng.normal(size=500)
    model = _poly_to_innovations([1, -1.0], [1])               # y(t) = y(t-1) + e(t)
    for offset in (0.0, 1e3, 1e6):
        y = offset + np.cumsum(e)
        err = _prediction_errors(model, y, 1)
        assert np.isnan(err[0])
        np.testing.assert_allclose(err[1:], -e[1:], atol=1e-9 * max(1.0, offset) )
        assert np.std(err[1:]) == pytest.approx(1.0, rel=0.1)
    # the stationary start, in contrast, would not even define a prior for this model (unit root)
    assert mf._stationary_state_cov(*model[:2]) is None


def test_diffuse_near_unit_root_with_offset():
    """A near-unit-root AR(1) with a large offset: the stationary start makes a huge error at sample 0, the diffuse start none."""
    phi = 0.995
    rng = np.random.default_rng(32)
    y = 1000.0 + lfilter([1], [1, -phi], rng.normal(size=400))
    model = _poly_to_innovations([1, -phi], [1])
    diff = _defined_errors(_prediction_errors(model, y, 1))
    stat = _prediction_errors(model, y, 1, init='stationary')
    # (the model has no constant, so the offset leaves a steady bias (1 - phi) * 1000 = 5 in every error)
    assert np.sqrt(np.mean(diff ** 2)) == pytest.approx(5.0, abs=1.0)
    assert abs(stat[0]) > 900 and np.sqrt(np.mean(stat ** 2)) > 5 * np.sqrt(np.mean(diff ** 2))


def test_diffuse_k_step_defined_samples_and_ar_closed_form():
    phi = 0.8
    y = _arma([1, -phi], [1], 100, 33) + 5.0
    a, k, c = _poly_to_innovations([1, -phi], [1])
    for steps in (1, 2, 3):
        err = _prediction_errors((a, k, c), y, steps)
        assert np.all(np.isnan(err[:steps])) and not np.any(np.isnan(err[steps:]))
        np.testing.assert_allclose(err[steps:], phi ** steps * y[:-steps] - y[steps:], atol=1e-10)
    # a second-order model: the first n + k - 1 = 4 samples of a 3-step prediction are undefined
    a2, k2, c2 = _poly_to_innovations([1, -0.5, 0.2], [1])
    e3 = _prediction_errors((a2, k2, c2), y, 3)
    assert list(np.flatnonzero(np.isnan(e3))) == [0, 1, 2, 3]
    # undefined samples are dropped by the helper; with nothing defined a single NaN is returned
    assert len(_defined_errors(e3)) == len(y) - 4
    assert np.isnan(_defined_errors(np.full(5, np.nan))).all() and len(_defined_errors(np.full(5, np.nan))) == 1


def test_diffuse_unobservable_state_does_not_delay_predictions():
    a = np.array([[0.5, 0.0], [0.0, 0.3]])
    k, c = np.array([0.4, 0.2]), np.array([1.0, 0.0])     # the second state never reaches the output
    y = _arma([1, -0.5], [1, 0.4], 80, 34)
    pred = _diffuse_predictions(a, k, c, y, 1)
    assert list(np.flatnonzero(np.isnan(pred))) == [0]
    ref = _diffuse_predictions(a[:1, :1], k[:1], c[:1], y, 1)
    np.testing.assert_allclose(pred[1:], ref[1:], atol=1e-10)
