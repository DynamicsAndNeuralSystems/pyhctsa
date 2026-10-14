"""Accuracy tests of the fitting toolbox: curve fits, polynomial fits, goodness of fit, robust
regression, distribution fits and the gpml optimizers.

Checks are analytic (exact recovery of noiseless models, closed forms, score equations),
against reference libraries (numpy, scipy, statsmodels) or by simulation (coverage).
"""
import numpy as np
import pytest
from scipy import special, stats

from pyhctsa.toolboxes.distribution_fits.distfits import betafit, evfit, gamfit, wblfit
from pyhctsa.fitting import fit_curve, fit_exp, goodness_of_fit, polyfit
from pyhctsa.fitting import robust_linear_fit as robustfit
from pyhctsa.toolboxes.gpml import optimizers


# ------------------------------------------------------------------ fit_curve / fit_exp
def test_fit_curve_recovers_noiseless_exponential():
    x = np.linspace(0, 5, 40)
    p = fit_curve(lambda c, t: c[0] * np.exp(c[1] * t), [1.0, -0.2], x, 2.5 * np.exp(-0.7 * x))
    np.testing.assert_allclose(p, [2.5, -0.7], rtol=1e-8)


def test_fit_curve_recovers_saturating_and_linear_models():
    x = np.linspace(0.1, 4, 30)
    p = fit_curve(lambda c, t: c[0] * (1 - np.exp(c[1] * t)), [1.0, -0.5], x, 3.0 * (1 - np.exp(-1.3 * x)))
    np.testing.assert_allclose(p, [3.0, -1.3], rtol=1e-7)
    p = fit_curve(lambda c, t: c[0] * t + c[1], [0.1, 0.0], x, 4 * x - 2)
    np.testing.assert_allclose(p, [4.0, -2.0], atol=1e-8)


def test_fit_curve_noisy_optimum_has_zero_gradient():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 60)
    y = 2.0 * np.exp(-0.4 * x) + 0.05 * rng.standard_normal(60)
    a, b = fit_exp(x, y, (1.0, -0.2))
    r = a * np.exp(b * x) - y
    grad = np.array([np.sum(r * np.exp(b * x)), np.sum(r * a * x * np.exp(b * x))])
    assert np.all(np.abs(grad) < 1e-4 * np.sum(np.abs(r)))       # relative to the scale of the residual sums
    # no worse than the true parameters and than a perturbed optimum
    assert np.sum(r ** 2) <= np.sum((2.0 * np.exp(-0.4 * x) - y) ** 2) + 1e-12
    for da, db in ((1e-3, 0), (0, 1e-3), (-1e-3, 1e-3)):
        assert np.sum(r ** 2) <= np.sum(((a + da) * np.exp((b + db) * x) - y) ** 2)


def test_fit_exp_returns_floats():
    x = np.linspace(0, 3, 20)
    y = 1.5 * np.exp(-0.5 * x)
    a, b = fit_exp(x, y)
    assert isinstance(a, float) and isinstance(b, float)
    np.testing.assert_allclose([a, b], [1.5, -0.5], rtol=1e-7)


def test_fit_curve_errors():
    with pytest.raises(ValueError):
        fit_exp([1, 2, 3], [1, np.nan, 2])
    with pytest.raises(ValueError):
        fit_exp([1, 2, 3], [1, np.inf, 2])
    with pytest.raises(ValueError):
        fit_exp([1.0], [1.0])


# ------------------------------------------------------------------ polyfit
@pytest.mark.parametrize("deg", [0, 1, 2, 3, 5])
def test_polyfit_recovers_polynomial_and_equals_numpy(deg):
    rng = np.random.default_rng(deg)
    coef = rng.uniform(-2, 2, deg + 1)
    x = np.linspace(-1, 2, 25)
    y = np.polyval(coef, x)
    np.testing.assert_allclose(polyfit(x, y, deg), coef, atol=1e-8)
    yn = y + 0.1 * rng.standard_normal(25)
    np.testing.assert_allclose(polyfit(x, yn, deg), np.polyfit(x, yn, deg), rtol=1e-10, atol=1e-12)


def test_polyfit_degenerate_inputs():
    assert np.all(np.isnan(polyfit([1, 2, np.nan], [1, 2, 3], 1)))
    assert np.all(np.isnan(polyfit([1, 2, 3], [1, np.inf, 3], 1)))
    assert np.all(np.isnan(polyfit([], [], 2)))  # no data: not identified (C7)
    # underdetermined: the slope is not identified
    assert np.all(np.isnan(polyfit([1.0, 2.0], [1.0, 2.0], 2)))
    assert np.all(np.isnan(polyfit([3.0], [1.0], 1)))
    # constant y
    np.testing.assert_allclose(polyfit(np.arange(10.0), np.full(10, 4.0), 1), [0.0, 4.0], atol=1e-12)


# ------------------------------------------------------------------ goodness_of_fit
def test_goodness_of_fit_hand_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yh = np.array([1.1, 1.9, 3.2, 3.8, 5.0])
    g = goodness_of_fit(y, yh, 2)
    sse = 0.1 ** 2 + 0.1 ** 2 + 0.2 ** 2 + 0.2 ** 2
    sst = 10.0
    assert g["sse"] == pytest.approx(sse)
    assert g["rsquare"] == pytest.approx(1 - sse / sst)
    assert g["adjrsquare"] == pytest.approx(1 - sse / sst * 4 / 3)
    assert g["rmse"] == pytest.approx(np.sqrt(sse / 3))
    assert g["dfe"] == 3


def test_goodness_of_fit_perfect_fit_and_degenerate():
    g = goodness_of_fit([1, 2, 3, 5], [1, 2, 3, 5], 2)
    assert g["sse"] == 0 and g["rsquare"] == 1 and g["rmse"] == 0
    g = goodness_of_fit([1, 2], [1, 2.1], 2)               # no degrees of freedom left
    assert np.isnan(g["adjrsquare"]) and np.isnan(g["rmse"]) and np.isfinite(g["rsquare"])
    for yh in ([1, 1, 1, 1], [1, 1, 1, 2]):                  # constant y: R^2 undefined
        g = goodness_of_fit([1, 1, 1, 1], yh, 1)
        assert np.isnan(g["rsquare"]) and np.isnan(g["adjrsquare"])
    assert np.isnan(goodness_of_fit([1, 2, np.nan], [1, 2, 3], 1)["rsquare"])


# ------------------------------------------------------------------ robust regression
def test_robustfit_recovers_exact_line_without_standard_error_floor():
    x = np.arange(30.0)
    b, st_ = robustfit(x, 3.0 - 0.25 * x)
    np.testing.assert_allclose(b, [3.0, -0.25], atol=1e-12)
    assert st_["s"] < 1e-12 and np.all(st_["se"] < 1e-12)       # no artificial floor on se
    assert np.all(st_["w"] > 0)


def test_robustfit_resists_outliers_and_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = 1.0 + 2.0 * x + 0.3 * rng.standard_normal(300)
    y[:15] += rng.choice([-1, 1], 15) * 20
    b, st_ = robustfit(x, y)
    assert abs(b[0] - 1.0) < 0.1 and abs(b[1] - 2.0) < 0.1
    assert np.all(st_["w"][:15] == 0)
    ref = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.TukeyBiweight(4.685)).fit().params
    np.testing.assert_allclose(b, ref, atol=0.02)
    # ordinary least squares is dragged away by the contamination
    assert abs(np.polyfit(x, y, 1)[1] - 1.0) > abs(b[0] - 1.0)


def test_robustfit_large_tuning_constant_is_least_squares():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(80)
    y = 0.5 + x + rng.standard_normal(80)
    b, _ = robustfit(x, y, tune=1e9)
    np.testing.assert_allclose(b, np.polyfit(x, y, 1)[::-1], rtol=1e-8)


def test_robustfit_is_equivariant_to_affine_changes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(60)
    y = 1 + x + rng.standard_normal(60)
    y[3] += 15
    b, st_ = robustfit(x, y)
    b2, st2 = robustfit(3 * x + 1, 5 * y - 2)
    np.testing.assert_allclose(b2[1], 5 * b[1] / 3, rtol=1e-6)
    np.testing.assert_allclose(st2["s"], 5 * st_["s"], rtol=1e-6)


def test_robustfit_standard_errors_have_nominal_coverage():
    rng = np.random.default_rng(3)
    reps, cover = 400, np.zeros(2)
    for _ in range(reps):
        x = rng.standard_normal(60)
        y = 1.0 + 2.0 * x + rng.standard_normal(60)
        b, st_ = robustfit(x, y)
        tq = stats.t.ppf(0.975, st_["dfe"])
        cover += np.abs(b - [1.0, 2.0]) <= tq * st_["se"]
    cover /= reps
    assert np.all((cover > 0.91) & (cover < 0.99)), cover


def test_robustfit_leverage_one_point_is_handled():
    x = np.r_[np.ones(12), 100.0]
    y = np.r_[np.linspace(0, 1, 12), 7.0]
    b, st_ = robustfit(x, y)
    assert np.all(np.isfinite(b)) and np.isfinite(st_["s"])
    assert st_["h"][-1] == pytest.approx(1.0) and st_["w"][-1] == 1.0


def test_robustfit_iteration_limit_warns(monkeypatch):
    from pyhctsa.fitting import robust_regression
    monkeypatch.setattr(robust_regression, "MAX_ITER", 1)
    rng = np.random.default_rng(4)
    x = rng.standard_normal(30)
    y = x + rng.standard_normal(30)
    y[:3] += 12
    with pytest.warns(RuntimeWarning, match="Iteration limit"):
        robustfit(x, y)


# ------------------------------------------------------------------ distribution fits
@pytest.mark.parametrize("seed", range(4))
def test_evfit_satisfies_score_equations_and_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    x = -rng.gumbel(rng.normal(), rng.uniform(0.3, 5), int(rng.choice([30, 400, 4000])))
    mu, sigma = evfit(x)
    z = (x - mu) / sigma
    assert abs(np.mean(1 - np.exp(z))) < 1e-9
    assert abs(np.mean(-1 - z + z * np.exp(z))) < 1e-9
    loc, scale = stats.gumbel_l.fit(x)
    np.testing.assert_allclose([mu, sigma], [loc, scale], rtol=1e-7)


@pytest.mark.parametrize("seed", range(4))
def test_gamfit_satisfies_score_equations(seed):
    rng = np.random.default_rng(seed)
    x = rng.gamma(rng.uniform(0.5, 8), rng.uniform(0.5, 5), int(rng.choice([30, 400, 4000])))
    a, b = gamfit(x)
    assert abs(np.log(a) - special.digamma(a) - (np.log(np.mean(x)) - np.mean(np.log(x)))) < 1e-9
    assert a * b == pytest.approx(np.mean(x), rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_wblfit_satisfies_score_equations(seed):
    rng = np.random.default_rng(seed)
    x = rng.weibull(rng.uniform(0.6, 5), int(rng.choice([30, 400, 4000]))) * rng.uniform(0.5, 100)
    scale, shape = wblfit(x)
    lx = np.log(x)
    w = (x / scale) ** shape
    assert abs(np.sum(w * lx) / np.sum(w) - 1 / shape - np.mean(lx)) < 1e-10      # d/d shape (profile)
    assert np.mean(w) == pytest.approx(1.0, rel=1e-10)                           # d/d scale


@pytest.mark.parametrize("seed", range(4))
def test_betafit_satisfies_score_equations(seed):
    rng = np.random.default_rng(seed)
    x = rng.beta(rng.uniform(0.5, 6), rng.uniform(0.5, 6), int(rng.choice([30, 400, 4000])))
    a, b = betafit(x)
    assert abs(special.digamma(a) - special.digamma(a + b) - np.mean(np.log(x))) < 1e-7
    assert abs(special.digamma(b) - special.digamma(a + b) - np.mean(np.log1p(-x))) < 1e-7


def test_distribution_fits_recover_known_parameters_from_large_samples():
    rng = np.random.default_rng(7)
    n = 100000
    mu, sigma = evfit(-rng.gumbel(1.5, 2.0, n))
    assert mu == pytest.approx(-1.5, rel=0.01) and sigma == pytest.approx(2.0, rel=0.01)
    a, b = gamfit(rng.gamma(2.5, 3.0, n))
    assert a == pytest.approx(2.5, rel=0.01) and b == pytest.approx(3.0, rel=0.01)
    scale, shape = wblfit(2.0 * rng.weibull(1.7, n))
    assert scale == pytest.approx(2.0, rel=0.01) and shape == pytest.approx(1.7, rel=0.01)
    a, b = betafit(rng.beta(2.0, 5.0, n))
    assert a == pytest.approx(2.0, rel=0.01) and b == pytest.approx(5.0, rel=0.01)


def test_distribution_fits_return_nan_for_degenerate_input():
    for f in (evfit, gamfit, wblfit):
        for bad in ([], [2.0], [3.0, 3.0, 3.0], [1.0, np.nan, 2.0], [1.0, np.inf, 2.0]):
            assert all(np.isnan(f(bad))), (f.__name__, bad)
    for f in (gamfit, wblfit):                                  # outside (0, inf)
        assert all(np.isnan(f([1.0, -2.0, 3.0])))
        assert all(np.isnan(f([0.0, 1.0, 3.0])))
    for bad in ([0.1, 0.2, 1.5], [0.0, 0.5, 0.7], [0.3, 0.3, 0.3], [0.3, np.nan, 0.4]):
        r = betafit(bad)
        assert r.shape == (2,) and np.all(np.isnan(r))


def test_betafit_output_type():
    r = betafit(np.random.default_rng(0).beta(2, 3, 50))
    assert r.shape == (2,) and r.dtype == np.float64


# ------------------------------------------------------------------ optimizers
def _rosenbrock(x):
    f = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    g = np.zeros_like(x)
    g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return f, g


def test_minimize_solves_a_convex_quadratic_exactly():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    A = M @ M.T + 4 * np.eye(4)
    c = rng.standard_normal(4)
    x, fx, i = optimizers.minimize(np.zeros(4), lambda v: (0.5 * v @ A @ v - c @ v, A @ v - c), 100)
    np.testing.assert_allclose(x, np.linalg.solve(A, c), atol=1e-8)
    assert all(b <= a for a, b in zip(fx, fx[1:]))


def test_minimize_budget_and_progress_on_rosenbrock():
    calls = []

    def f(x):
        calls.append(1)
        return _rosenbrock(x)

    x0 = np.array([-1.2, 1.0])
    x, fx, i = optimizers.minimize(x0, f, -50)
    assert len(calls) <= 50 + 20                      # the budget is checked between line-search steps
    assert fx[-1] < fx[0] and _rosenbrock(x)[0] <= fx[-1] + 1e-12     # returned point is the best found
    x2, fx2, _ = optimizers.minimize(x0, _rosenbrock, 500)
    np.testing.assert_allclose(x2, [1.0, 1.0], atol=1e-4)


def test_minimize_handles_objectives_with_failed_evaluations():
    # f is undefined (NaN) for x > 1: extrapolation steps landing there are bisected
    def f(x):
        if x[0] > 1.0:
            return np.nan, np.array([np.nan])
        return (x[0] - 3.0) ** 2, np.array([2 * (x[0] - 3.0)])

    x, fx, _ = optimizers.minimize(np.array([0.0]), f, -50)
    assert np.isfinite(x[0]) and x[0] <= 1.0 and x[0] > 0.9


def test_minimize_cubic_without_real_stationary_point_stays_finite():
    # non-convex functions drive the cubic step to a negative discriminant
    def f(x):
        return np.sum(np.sin(3 * x) + 0.1 * x ** 2), 3 * np.cos(3 * x) + 0.2 * x

    rng = np.random.default_rng(0)
    for _ in range(50):
        x0 = rng.normal(size=3) * 3
        x, fx, _ = optimizers.minimize(x0, f, -50)
        assert np.all(np.isfinite(x)) and fx[-1] <= fx[0] + 1e-12
