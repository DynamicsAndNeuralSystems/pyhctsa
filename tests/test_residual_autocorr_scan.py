"""residual_autocorr_scan: the largest autocorrelation of AR residuals (and of their squares) over a wide
range of lags, and the Ljung-Box and McLeod-Li statistics as z-scores.

Exact pieces (the prewhitening AR fit), null calibration on white and short-range AR series, detection of a
single long-lag dependence in sign or in amplitude at the right lag, and degenerate inputs."""

import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import correlation as co

FIELDS = {"max_z", "peak_lag", "lb_z", "sq_max_z", "sq_lb_z"}


def _ar(n, coefs, rng, burn=500):
    e = rng.standard_normal(n + burn)
    return lfilter([1.0], np.r_[1.0, -np.asarray(coefs)], e)[burn:]


def _long_lag(n, lag, phi, rng, short=(0.5, -0.3), burn=1000):
    # u_t = e_t + phi u_{t-lag}, then a short-range AR(2) filter: dependence at one long lag under
    # strong short-range structure
    e = rng.standard_normal(n + burn)
    u = lfilter([1.0], np.r_[1.0, np.zeros(lag - 1), -phi], e)
    return lfilter([1.0], np.r_[1.0, -np.asarray(short)], u)[burn:]


def test_prewhiten_recovers_ar_order_and_coefficients():
    rng = np.random.default_rng(0)
    x = _ar(20_000, [0.6, -0.3], rng)
    e, p = co._ar_prewhiten(x, 10)
    assert p >= 2
    assert e.size == x.size - p
    # the residuals are (close to) the innovations: white, and the AR(2) explains the rest
    r = co._acf_fft(e)[1:6]
    assert np.all(np.abs(r) < 0.03)
    # white noise needs no AR term
    _, p0 = co._ar_prewhiten(np.random.default_rng(1).standard_normal(5000), 10)
    assert p0 <= 2


def test_output_fields_and_scale_invariance():
    rng = np.random.default_rng(2)
    x = _ar(1000, [0.5], rng)
    out = co.residual_autocorr_scan(x)
    assert set(out) == FIELDS
    assert 1 <= out["peak_lag"] <= 250
    out2 = co.residual_autocorr_scan(3 * x - 7)
    for k in FIELDS:
        assert out2[k] == pytest.approx(out[k], rel=1e-9)


@pytest.mark.parametrize("make", [lambda n, r: r.standard_normal(n), lambda n, r: _ar(n, [0.9], r)])
def test_null_calibration(make):
    # white noise and a strongly autocorrelated AR(1): after prewhitening the Ljung-Box z-score is
    # about standard normal and the largest |z_k| stays near its extreme-value scale
    rng = np.random.default_rng(3)
    outs = [co.residual_autocorr_scan(make(2000, rng)) for _ in range(60)]
    lb = np.array([o["lb_z"] for o in outs])
    mz = np.array([o["max_z"] for o in outs])
    assert abs(lb.mean()) < 0.5 and 0.5 < lb.std() < 1.6
    assert np.median(mz) < 3.8 and mz.max() < 5.0


def test_detects_one_long_lag_under_short_range_structure():
    rng = np.random.default_rng(4)
    hits = [co.residual_autocorr_scan(_long_lag(1000, 120, 0.4, rng)) for _ in range(10)]
    assert np.median([h["max_z"] for h in hits]) > 7
    assert np.mean([h["peak_lag"] == 120 for h in hits]) >= 0.8
    # the null with the same short-range filter and no long-lag term
    nulls = [co.residual_autocorr_scan(_long_lag(1000, 120, 0.0, rng)) for _ in range(10)]
    assert max(h["max_z"] for h in nulls) < min(h["max_z"] for h in hits)


def test_squared_residuals_see_lagged_amplitude_dependence():
    # u_t = e_t (1 + g(|u_{t-L}| - E|u|)): uncorrelated, but the size of u_t depends on |u_{t-L}|
    rng = np.random.default_rng(5)
    L, n = 60, 3000
    e = rng.standard_normal(n + 500)
    u = np.zeros_like(e)
    for t in range(e.size):
        u[t] = e[t] * (1 + 0.8 * (abs(u[t - L]) - 0.8)) if t >= L else e[t]
        u[t] = np.clip(u[t], -20, 20)
    out = co.residual_autocorr_scan(u[500:])
    assert out["sq_max_z"] > 6 and out["sq_lb_z"] > 3
    assert out["max_z"] < 4.5


def test_degenerate_inputs_are_nan():
    for y in (np.ones(100), np.arange(5.0), [1.0, 2.0, 1.0]):
        out = co.residual_autocorr_scan(y)
        assert set(out) == FIELDS
        assert all(np.isnan(v) for v in out.values())


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    assert "resid_ac_scan" in calc.feature_funcs
    out = calc.feature_funcs["resid_ac_scan"](np.random.default_rng(6).standard_normal(500))
    assert set(out) == FIELDS - {"peak_lag"}   # excluded from the default set
