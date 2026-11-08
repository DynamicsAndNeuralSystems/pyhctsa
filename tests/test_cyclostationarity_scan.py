"""cyclostationarity_scan: spectral lines in the lag products of AR residuals (Gardner's spectral-line
regeneration), scanned over cyclic frequency and lag.

Null calibration on stationary processes (including a sharp spectral peak), detection of amplitude modulation
and of a periodically varying AR coefficient at non-integer periods, invariances, and degenerate inputs."""

import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import correlation as co

FIELDS = {"max_score", "lag0_score", "peak_alpha", "peak_lag"}


def _ar2_peak(n, r, f, rng):
    # a stationary AR(2) with a spectral peak of width ~(1 - r) at frequency f
    return lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * f), r * r], rng.standard_normal(n + 1000))[1000:]


@pytest.mark.parametrize("kind", ["white", "ar2_peak"])
def test_null_calibrated(kind):
    # stationary: the score is approximately standard Gumbel (mean 0.58, 5% above 2.97), even for a spectral
    # peak much narrower than the smoothing window, since prewhitening removes it
    rng = np.random.default_rng(1)
    make = (lambda: rng.standard_normal(1000)) if kind == "white" else (lambda: _ar2_peak(1000, 0.99, 0.05, rng))
    outs = [co.cyclostationarity_scan(make()) for _ in range(60)]
    s = np.array([o["max_score"] for o in outs])
    assert -0.2 < np.mean(s) < 1.4
    assert np.mean(s > 2.97) <= 0.15
    # with no significant line there is no rhythm to locate: peak_alpha and peak_lag are NaN exactly when
    # max_score is below the 5% threshold
    for o in outs:
        assert np.isnan(o["peak_alpha"]) == np.isnan(o["peak_lag"]) == (o["max_score"] <= 2.9702)


def test_detects_amplitude_modulation():
    # noise with a periodic variance (period 37.3 samples): no spectral peak in x, but a line at alpha = 1/37.3
    # in the squared residuals
    rng = np.random.default_rng(2)
    n = 2000
    for _ in range(5):
        y = (1 + 0.5 * np.cos(2 * np.pi * np.arange(n) / 37.3 + rng.uniform(0, 2 * np.pi))) * rng.standard_normal(n)
        out = co.cyclostationarity_scan(y)
        assert out["max_score"] > 10 and out["lag0_score"] > 10
        assert out["peak_alpha"] == pytest.approx(1 / 37.3, abs=2 / n) and out["peak_lag"] == 0


def test_detects_periodic_ar_coefficient():
    # AR(1) with a coefficient that varies with period 9.7: the lag-1 autocovariance is periodic
    rng = np.random.default_rng(3)
    n = 4000
    phi = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(n) / 9.7)
    scores = []
    for _ in range(5):
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi[t] * y[t - 1] + e[t]
        scores.append(co.cyclostationarity_scan(y)["max_score"])
    assert min(scores) > 2.97


def test_scale_and_shift_invariant():
    y = np.random.default_rng(4).standard_t(5, 800)
    a, b = co.cyclostationarity_scan(y), co.cyclostationarity_scan(-2 * y + 5)
    assert set(a) == FIELDS
    for k in FIELDS:
        assert a[k] == pytest.approx(b[k], nan_ok=True)


def test_degenerate_inputs():
    for y in (np.ones(300), np.random.default_rng(5).standard_normal(50)):
        out = co.cyclostationarity_scan(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())
    with pytest.raises(ValueError):
        co.cyclostationarity_scan(np.random.default_rng(0).standard_normal(200), n_smooth=0)
    for level in (0.0, 1.0):
        with pytest.raises(ValueError):
            co.cyclostationarity_scan(np.random.default_rng(0).standard_normal(200), peak_level=level)


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    fc = FeatureCalculator()
    assert [lab for lab in fc.min_lengths if lab.startswith("cyclo_scan")] == ["cyclo_scan"]
    assert fc.min_lengths["cyclo_scan"] == 128
