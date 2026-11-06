"""extremogram: the largest lagged conditional exceedance probability of |x - mean| (Davis & Mikosch 2009) over
a range of lags.

Exact values on a constructed series, null calibration on i.i.d. data, detection of extremes that recur at a
long lag while the |x| autocorrelation stays near zero, invariances, and degenerate inputs."""

import numpy as np
import pytest

from pyhctsa.operations import extreme_events as ex

FIELDS = {"max_chi", "max_z", "peak_lag"}


def test_exact_periodic_spikes():
    # one large value every 7 samples on a constant background (so the threshold ties with the background
    # and exactly the spikes exceed it): every exceedance is followed by another exactly 7 samples later,
    # and by none at any other lag up to 13
    n = 700
    y = np.zeros(n)
    y[::7] = 5.0
    out = ex.extremogram(y, quantile=0.8, max_lag=13)
    assert out["max_chi"] == pytest.approx(1.0)
    assert out["peak_lag"] == 7
    p = 100 / n
    n7 = 99   # spikes at 0, 7, ..., 693: all but the last have a sample 7 steps later
    assert out["max_z"] == pytest.approx((n7 - n7 * p) / np.sqrt(n7 * p))


def test_iid_null():
    rng = np.random.default_rng(1)
    outs = [ex.extremogram(rng.standard_t(4, 4000)) for _ in range(40)]
    chi = np.array([o["max_chi"] for o in outs])
    z = np.array([o["max_z"] for o in outs])
    # the largest of 100 conditional rates of ~0.05 (each from ~200 exceedances), and of 100 z-scores
    assert np.all(chi < 0.13)
    assert 1.5 < np.median(z) < 4.0


def _recurring_extremes(n, lag, p, rng):
    # i.i.d. heavy-tailed series in which each top-5% value is repeated (in size, with a random sign) lag
    # samples later with probability p, the displaced value moving to a random time: the values barely
    # change, but extremes recur at one lag
    y = rng.standard_t(4, n)
    a = np.abs(y)
    u = np.quantile(a, 0.95)
    for t in np.flatnonzero(a > u):
        if t + lag < n and rng.random() < p and abs(y[t + lag]) <= u:
            s = rng.integers(n)
            y[s], y[t + lag] = y[t + lag], np.sign(rng.standard_normal()) * abs(y[t])
    return y


def test_detects_extremes_recurring_at_a_long_lag():
    rng = np.random.default_rng(2)
    hits = [ex.extremogram(_recurring_extremes(4000, 40, 0.3, rng)) for _ in range(10)]
    nulls = [ex.extremogram(_recurring_extremes(4000, 40, 0.0, rng)) for _ in range(10)]
    assert np.mean([h["peak_lag"] == 40 for h in hits]) >= 0.8
    assert min(h["max_z"] for h in hits) > max(h["max_z"] for h in nulls)
    assert np.median([h["max_chi"] for h in hits]) > 0.25


def test_scale_and_shift_invariant():
    y = np.random.default_rng(3).standard_normal(1000)
    a, b = ex.extremogram(y), ex.extremogram(4 * y + 9)
    for k in FIELDS:
        assert a[k] == pytest.approx(b[k])


def test_degenerate_inputs():
    for y in (np.arange(9.0), np.ones(50)):
        out = ex.extremogram(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())
    for q in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            ex.extremogram(np.arange(100.0), quantile=q)


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    for lab in ("extremogram_0p95", "extremogram_0p98"):
        out = calc.feature_funcs[lab](np.random.default_rng(4).standard_normal(500))
        assert set(out) == FIELDS
