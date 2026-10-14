"""Tests for the ARMA fit (``_armax_fit``, conditional least squares from a Hannan-Rissanen start) and the
forward-backward AR fit.

The tests of the other system-identification helpers are in ``tests/native/test_wp4_sysid.py``.
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import model_fit as mf
from pyhctsa.operations.model_fit import _ar_fb, _armax_fit


def test_ar_fb_time_reversal_and_errors():
    rng = np.random.default_rng(1)
    y = lfilter([1], [1, -0.6, 0.2], rng.normal(size=200))
    np.testing.assert_allclose(_ar_fb(y, 3)[0], _ar_fb(y[::-1], 3)[0], atol=1e-10)
    with pytest.raises(ValueError):
        _ar_fb(y[:3], 3)


# --------------------------------------------------------------------------- armax
def _css_residuals(theta, y, p, q):
    """Conditional residuals e(t), t >= max(p, q), written out as a recursion."""
    r = max(p, q)
    e = np.zeros(len(y))
    for t in range(r, len(y)):
        e[t] = y[t] + sum(theta[k - 1] * y[t - k] for k in range(1, p + 1)) \
            - sum(theta[p + k - 1] * e[t - k] for k in range(1, q + 1))
    return e[r:]


@pytest.mark.parametrize('orders', [(1, 1), (2, 2), (3, 1)])
def test_armax_fit_minimizes_the_conditional_sum_of_squares(orders):
    from scipy.optimize import least_squares
    rng = np.random.default_rng(3)
    y = lfilter([1, 0.4], [1, -0.7, 0.2], rng.normal(size=600))
    p, q = orders
    a, c, loss, cov = _armax_fit(y, p, q)
    theta = np.r_[a[1:], c[1:]]
    assert np.all(np.abs(np.roots(c)) < 1)
    # a stationary point of V: an independent least-squares solver started there does not move
    fit = least_squares(lambda th: _css_residuals(th, y, p, q), theta, method='lm', xtol=1e-14, ftol=1e-14)
    assert 2 * fit.cost == pytest.approx(np.sum(_css_residuals(theta, y, p, q) ** 2), rel=1e-9)
    np.testing.assert_allclose(fit.x, theta, atol=1e-5)
    # the loss is the mean squared one-step error from a diffuse start (first max(p, q) samples not predicted)
    err = mf._kstep_residuals(a, c, y, 1)
    assert np.all(np.isnan(err[:max(p, q)])) and np.all(np.isfinite(err[max(p, q):]))
    assert loss == pytest.approx(np.nanmean(err ** 2), rel=1e-12)
    # the standard errors are those of the linearized least-squares problem at the optimum
    J = fit.jac
    s2 = 2 * fit.cost / (len(y) - max(p, q) - p - q)
    np.testing.assert_allclose(cov, s2 * np.linalg.inv(J.T @ J), rtol=1e-3, atol=1e-10)


def test_armax_fit_recovers_an_arma11_process():
    rng = np.random.default_rng(4)
    y = lfilter([1, 0.5], [1, -0.6], rng.normal(size=20000))
    a, c, loss, cov = _armax_fit(y, 1, 1)
    assert a[1] == pytest.approx(-0.6, abs=0.03) and c[1] == pytest.approx(0.5, abs=0.03)
    assert loss == pytest.approx(1.0, abs=0.03)
    assert np.all(np.sqrt(np.diag(cov)) < 0.03)


def test_armax_fit_reflects_a_noninvertible_ma_part():
    rng = np.random.default_rng(5)
    e = rng.normal(size=4000)
    y = lfilter([1, -0.9], [1], e)          # MA(1) with its zero at 0.9: fitted as ARMA(1, 1)
    a, c, loss, cov = _armax_fit(y, 1, 1)
    assert np.all(np.abs(np.roots(c)) < 1)
    # a reflected zero gives the same spectrum shape; the reflection itself
    assert np.allclose(mf._reflect_poly([1, -2.0]), [1, -0.5])


def test_armax_edges():
    rng = np.random.default_rng(2)
    y = lfilter([1, 0.5], [1, -0.6], rng.normal(size=400))
    a0, c0, loss0, cov0 = _armax_fit(np.zeros(60), 2, 1)
    assert np.array_equal(a0, [1, 0, 0]) and np.array_equal(c0, [1, 0]) and loss0 == 0
    with pytest.raises(ValueError):
        _armax_fit(y[:3], 2, 1)
    with pytest.raises(ValueError):
        _armax_fit(np.r_[y[:20], np.nan], 1, 1)
    # pure AR: ordinary least squares on the samples that have p past values
    a_ar, c_ar, _, _ = _armax_fit(y, 2, 0)
    assert np.array_equal(c_ar, [1.0])
    X = np.column_stack([y[1:-1], y[:-2]])
    np.testing.assert_allclose(a_ar[1:], -np.linalg.lstsq(X, y[2:], rcond=None)[0], atol=1e-10)
    # a series too short for the orders gives NaN outputs from armax
    out = mf.armax(y[:12], [3, 1], 0.5)
    assert all(np.isnan(v) for v in out.values())


# --------------------------------------------------------------------------- residual summary
@pytest.mark.parametrize('kind', ['tiny', 'offset', 'zero'])
def test_residual_analysis_of_rounding_level_residuals(kind):
    rng = np.random.default_rng(9)
    y = rng.normal(size=300)
    e = {'tiny': 1e-13 * rng.normal(size=300), 'offset': np.full(300, 0.3), 'zero': np.zeros(300)}[kind]
    out = mf.residual_analysis(e, y, 'full')
    assert all(np.isfinite(out[k]) for k in ('meane', 'meanabs', 'stde'))
    nan_keys = set(out) - {'meane', 'meanabs', 'stde'}
    assert nan_keys == {'maxonstd', 'ac1', 'ac2', 'ac3', 'propbth', 'taurat', 'sws', 'swm', 'ftbth',
                        'normksstat', 'popt', 'minsbc'}
    assert all(np.isnan(out[k]) for k in nan_keys)
    ordinary = mf.residual_analysis(rng.normal(size=300), y, 'full')
    assert all(np.isfinite(v) for v in ordinary.values())
