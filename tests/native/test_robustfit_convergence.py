"""Convergence of the bisquare IRLS when the plain iteration fails.

With the scale re-estimated at every step, the IRLS map ``T`` (weighted least squares with the
bisquare weights of the residuals of ``b``) can have a fixed point at which its Jacobian has an
eigenvalue below -1 (the plain iteration overshoots and wanders chaotically, so the coefficient
after 200 steps depends on input rounding) or close to +1 (the plain iteration creeps and stops
at the limit short of the fixed point). The cases below are Higuchi log-log points (log(1/k),
log L(k), k = 1..8) of benchmark series with each behavior.
"""
import warnings

import numpy as np
import pytest

from pyhctsa.fitting import BisquareIRLS, robust_linear_fit
from pyhctsa.fitting import robust_regression
from pyhctsa.fitting.robust_regression import bisquare_weights

X = -np.log(np.arange(1, 9))
OSCILLATING = {   # eigenvalue of the Jacobian of T at the fixed point ~ -1.2 to -1.5
    "telegraph": [2.0730353317295425, 0.9449213348409864, 0.24637743560304512, -0.16932142659692542,
                  -0.508903368515749, -0.9546787575790956, -1.5318893907297668, -2.473742412968138],
    "saturated_ou": [4.329947528945863, 3.100441009404432, 2.4095829157739357, 1.8800262216977457,
                     1.300286503430168, 0.8808792501887549, 0.5804854480127664, 0.3267639133527076],
}
SLOW = {   # eigenvalue ~ 0.987: the plain iteration needs ~800 steps
    "noisy_oscillator": [5.178381269986604, 4.154577337283978, 3.3699021854225424, 2.7265654095017093,
                         2.141840727084089, 1.7353244168157989, 1.470088475346259, 1.2855207398780335],
}
CASES = {**OSCILLATING, **SLOW}


def _fit(y):
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no "Iteration limit reached."
        m = BisquareIRLS(X, np.asarray(y))
        b, st = m.fit()
    return m, b, st


@pytest.mark.parametrize("name", CASES)
def test_converges_to_a_fixed_point_of_the_irls_map(name):
    m, b, st = _fit(CASES[name])
    assert m.converged and m.fallback
    tb = m._solve(bisquare_weights(m._scaled_residuals(b)))
    np.testing.assert_allclose(tb, b, rtol=0, atol=1e-8)
    np.testing.assert_allclose(st["w"], bisquare_weights(m._scaled_residuals(b)), atol=1e-7)


@pytest.mark.parametrize("name", OSCILLATING)
def test_insensitive_to_rounding_level_perturbations(name):
    y = np.asarray(CASES[name])
    _, b, st = _fit(y)
    rng = np.random.default_rng(1)
    for _ in range(5):
        _, bp, stp = _fit(y + 1e-12 * rng.standard_normal(y.size))
        np.testing.assert_allclose(bp, b, rtol=0, atol=1e-9)
        np.testing.assert_allclose(stp["se"], st["se"], rtol=1e-8)


def test_plain_iteration_is_unchanged_when_it_converges(monkeypatch):
    """Without the fallback the result is that of the plain iteration from least squares."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40)
    y = 1 + 2 * x + 0.3 * rng.standard_normal(40)
    y[:4] += 6
    m = BisquareIRLS(x, y)
    b, _ = m.fit()
    assert not m.fallback
    bb = m._solve()
    for _ in range(robust_regression.MAX_ITER):
        bn = m._solve(bisquare_weights(m._scaled_residuals(bb)))
        if np.max(np.abs(bn - bb)) <= robust_regression.CONVERGENCE_TOL * (1 + np.max(np.abs(bn))):
            bb = bn
            break
        bb = bn
    assert np.array_equal(b, bb)


def test_higuchi_points_give_a_stable_slope():
    """The fitted slope of the oscillating case no longer depends on the iteration limit."""
    b, _ = robust_linear_fit(X, OSCILLATING["telegraph"])
    b2, _ = robust_linear_fit(X, np.asarray(OSCILLATING["telegraph"]) * (1 + 1e-15))
    assert abs(b[1] - b2[1]) < 1e-9
