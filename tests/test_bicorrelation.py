"""bicorrelation: Hinich's bicorrelation plane of AR residuals (portmanteau, largest cell, largest reversal-
asymmetric cell).

Exact values against an independent computation, null calibration, detection of one third-order cell at a lag
pair, the reversal asymmetry, and degenerate inputs."""

import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import correlation as co

FIELDS = {"H_z", "max_abs", "asym_max_abs"}


def _bilinear(n, p, q, c, rng):
    # x_t = e_t + c e_{t-p} e_{t-q}: white, with one nonzero third-order cell, at (q - p, q)
    e = rng.standard_normal(n + q)
    return e[q:] + c * e[q - p:n + q - p] * e[:n]


def test_exact_against_direct_computation():
    rng = np.random.default_rng(0)
    y = rng.standard_t(6, 300)
    out = co.bicorrelation(y, max_lag=6, max_ar_order=0)
    z = (y - y.mean()) / y.std()
    G = {}
    for s in range(2, 7):
        for r in range(1, s):
            prod = np.array([z[t] * z[t + r] * z[t + s] for t in range(300 - s)])
            G[r, s] = prod.mean() / (prod.std() / np.sqrt(prod.size))
    g = np.array(list(G.values()))
    assert out["H_z"] == pytest.approx((np.sum(g ** 2) - 15) / np.sqrt(30))
    assert out["max_abs"] == pytest.approx(np.max(np.abs(g)))
    a = [(G[r, s] - G[s - r, s]) / np.sqrt(2) for s in range(3, 7) for r in range(1, s) if r < s - r]
    assert out["asym_max_abs"] == pytest.approx(np.max(np.abs(a)))


def test_lag_rule():
    # L = floor(m^0.4), capped at 20: 15 cells at m = 100 (L = 6), 190 at large m (L = 20)
    rng = np.random.default_rng(1)
    for n, L in ((100, 6), (5000, 20)):
        y = rng.standard_normal(n)
        assert co.bicorrelation(y, max_ar_order=0) == pytest.approx(co.bicorrelation(y, max_lag=L, max_ar_order=0))


@pytest.mark.parametrize("phi", [0.0, 0.8])
def test_null_calibration(phi):
    # white noise, and a strongly autocorrelated AR(1) (removed by the prewhitening)
    rng = np.random.default_rng(2)
    outs = [co.bicorrelation(lfilter([1.0], [1.0, -phi], rng.standard_normal(2300))[300:]) for _ in range(40)]
    h = np.array([o["H_z"] for o in outs])
    m = np.array([o["max_abs"] for o in outs])
    assert abs(np.median(h)) < 0.7
    assert np.median(m) < 3.8 and m.max() < 5.0


def test_detects_one_cell():
    rng = np.random.default_rng(3)
    hits = [co.bicorrelation(_bilinear(2000, 3, 11, 0.3, rng)) for _ in range(10)]
    nulls = [co.bicorrelation(rng.standard_normal(2000)) for _ in range(10)]
    assert min(h["max_abs"] for h in hits) > max(h["max_abs"] for h in nulls)
    assert np.median([h["max_abs"] for h in hits]) > 6


def test_reversal_asymmetry():
    # a bilinear source is irreversible; (u + u')/sqrt2 keeps the asymmetric cell and
    # (u + reversed u')/sqrt2 balances it with its mirror
    rng = np.random.default_rng(4)
    fwd, sym = [], []
    for _ in range(10):
        u, u2 = _bilinear(3000, 2, 9, 0.4, rng), _bilinear(3000, 2, 9, 0.4, rng)
        fwd.append(co.bicorrelation((u + u2) / np.sqrt(2))["asym_max_abs"])
        sym.append(co.bicorrelation((u + u2[::-1]) / np.sqrt(2))["asym_max_abs"])
    assert min(fwd) > max(sym)


def test_degenerate_inputs():
    for y in (np.ones(100), np.random.default_rng(5).standard_normal(12)):
        out = co.bicorrelation(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    out = calc.feature_funcs["bicorr"](np.random.default_rng(6).standard_normal(500))
    assert set(out) == FIELDS
