"""reversibility_scan: the studentised Ramsey-Rothman asymmetry gamma_{i,j}(k) on Gaussian ranks, scanned over lags.

Symmetries (time reversal, monotone transforms), null calibration on a reversible process, detection of a
fourth-order-only asymmetry by (3, 1) at its own lag, the classic (2, 1) case, and degenerate inputs."""

import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import correlation as co

FIELDS = {"max_abs_z", "peak_z", "peak_lag", "sumsq_z"}


def _ar1(n, phi, rng, innov=None, burn=300):
    e = rng.standard_normal(n + burn) if innov is None else innov(n + burn)
    return lfilter([1.0], [1.0, -phi], e)[burn:]


def _quartic_echo(n, k, a, rng):
    # x_t = e_t + a g(e_{t-k}), g odd with g^2 = 1 and E[e g(e)] = 0: white, white squares, no third-order lag
    # moment; the lowest asymmetric lag moment is <x_t x_{t-k}^3> (and <x_t^3 x_{t-k}> = 0)
    theta = np.sqrt(2 * np.log(2))
    e = rng.standard_normal(n + k)
    g = np.sign(e) * np.where(np.abs(e) < theta, 1.0, -1.0)
    return e[k:] + a * g[:-k]


def test_time_reversal_and_monotone_invariance():
    rng = np.random.default_rng(0)
    y = _ar1(800, 0.5, rng, innov=lambda m: rng.exponential(size=m))
    out = co.reversibility_scan(y)
    assert set(out) == FIELDS
    rev = co.reversibility_scan(y[::-1])   # every T_k changes sign
    for k in FIELDS - {"peak_z"}:
        assert rev[k] == pytest.approx(out[k])
    assert rev["peak_z"] == pytest.approx(-out["peak_z"]) and abs(out["peak_z"]) == pytest.approx(out["max_abs_z"])
    # distinct, evenly spaced values with the copula of y, so strictly increasing transforms keep every gap
    # far above the tie tolerance of the ranks
    r = np.argsort(np.argsort(y)).astype(float)
    out = co.reversibility_scan(r)
    for z in (np.exp(r / 50), np.log1p(r), 2 * r - 1):
        mon = co.reversibility_scan(z)
        for k in FIELDS:
            assert mon[k] == pytest.approx(out[k])


@pytest.mark.parametrize("ij", [(2, 1), (3, 1)])
def test_reversible_null(ij):
    rng = np.random.default_rng(1)
    outs = [co.reversibility_scan(_ar1(2000, 0.7, rng), *ij) for _ in range(40)]
    s = np.array([o["sumsq_z"] for o in outs])
    m = np.array([o["max_abs_z"] for o in outs])
    assert abs(np.median(s)) < 1.0
    assert np.median(m) < 3.3 and m.max() < 4.5


def test_fourth_order_asymmetry_at_its_lag():
    rng = np.random.default_rng(2)
    for k in (3, 9):
        y = _quartic_echo(8000, k, 0.8, rng)
        o31 = co.reversibility_scan(y, 3, 1)
        o21 = co.reversibility_scan(y, 2, 1)
        assert o31["max_abs_z"] > 5 and o31["peak_lag"] == k
        assert o21["max_abs_z"] < 4.5   # nothing at third order


def test_third_order_asymmetry():
    # AR(1) with exponential innovations: sharp rises, slow decays
    rng = np.random.default_rng(3)
    y = _ar1(2000, 0.6, rng, innov=lambda m: rng.exponential(size=m))
    out = co.reversibility_scan(y, 2, 1)
    assert out["max_abs_z"] > 6 and out["sumsq_z"] > 5
    # -3 gamma_{2,1} is the trev numerator, so the signs are opposite
    assert out["peak_lag"] == 1 and out["peak_z"] < 0 < co.trev_studentized(y, 1)


def test_degenerate_inputs():
    for y in (np.ones(100), [1.0, 2.0, 3.0]):
        out = co.reversibility_scan(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())
    for ij in ((1, 1), (0, 1), (2, -1), (1.5, 1)):
        with pytest.raises(ValueError):
            co.reversibility_scan(np.arange(50.0), *ij)


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    for lab in ("rev_scan_2_1", "rev_scan_3_1"):
        out = calc.feature_funcs[lab](np.random.default_rng(4).standard_normal(500))
        assert set(out) == FIELDS
