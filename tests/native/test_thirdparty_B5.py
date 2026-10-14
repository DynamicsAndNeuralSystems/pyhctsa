"""Accuracy tests for entropy.rpde (recurrence period density entropy, Little et al. 2007) and
its close-returns kernel. Independent checks plus the MATLAB black-box fixture (evidence only)."""
import json
import os

import numpy as np
import pytest

from pyhctsa.operations import entropy

SP = "/Users/benfulcher/ClaudeCode/pyhctsa-cleanroom-work"
SPEC = os.path.join(SP, "cleanroom", "specs", "thirdparty")
KEYS = ["H", "H_norm", "propNonZero", "meanNonZero", "maxRPD"]


def _embed(y, m, tau):
    n = len(y) - (m - 1) * tau
    return np.column_stack([y[k * tau:k * tau + n] for k in range(m)])


def _close_returns_naive(v, eps):
    """Direct n x n implementation of the first-return definition."""
    n = len(v)
    d = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(-1))
    h = np.zeros(n, dtype=np.int64)
    for i in range(n):
        j = i + 1
        while j < n and d[i, j] < eps:
            j += 1
        while j < n and d[i, j] >= eps:
            j += 1
        if j < n:
            h[j - i] += 1
    return h


def test_close_returns_hand_example():
    # 1-D trajectory 0, 0.1, 1, 1, 0.05, 2, 0.02 with eps = 0.5, m = 1
    v = np.array([0.0, 0.1, 1.0, 1.0, 0.05, 2.0, 0.02])[:, None]
    h = entropy._close_returns_hist(v, 0.5)
    # i=0: inside at j=1, out at j=2,3, back inside at j=4 -> T=4
    # i=1: out at j=2,3, back inside at j=4 -> T=3
    # i=2, 3 (value 1.0): out from j=4 on, never return; i=4: out at j=5, back at j=6 -> T=2
    # i=5, 6: no later points / no return
    expected = np.zeros(7, dtype=np.int64)
    expected[[4, 3, 2]] = 1
    np.testing.assert_array_equal(h, expected)


@pytest.mark.parametrize("m,tau,eps", [(2, 1, 0.3), (3, 2, 0.5), (1, 1, 0.1)])
def test_close_returns_matches_naive(m, tau, eps):
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(300))
    y = (y - y.mean()) / y.std()
    v = _embed(y, m, tau)
    np.testing.assert_array_equal(entropy._close_returns_hist(v, eps), _close_returns_naive(v, eps))


def test_rpde_periodic_vs_noise():
    t = np.arange(2000)
    s = np.sin(2 * np.pi * t / 20)
    s = (s - s.mean()) / s.std(ddof=1)
    per = entropy.rpde(s, 2, 5, 0.12)
    assert per["H_norm"] < 0.1
    rng = np.random.default_rng(0)
    w = rng.standard_normal(2000)
    w = (w - w.mean()) / w.std(ddof=1)
    assert entropy.rpde(w, 2, 1, 0.12)["H_norm"] > 0.6


def test_rpde_outputs_from_histogram():
    rng = np.random.default_rng(5)
    y = rng.standard_normal(400)
    y = (y - y.mean()) / y.std(ddof=1)
    out = entropy.rpde(y, 3, 1)
    h = _close_returns_naive(_embed(y, 3, 1), 0.12)
    rpd = h / h.sum()
    nz = rpd > 0
    H = -np.sum(rpd[nz] * np.log(rpd[nz]))
    nb = len(rpd)
    ref = [H, H / np.log(nb), nz.mean(), rpd[nz].mean() * nb, rpd.max() * nb]
    np.testing.assert_allclose([out[k] for k in KEYS], ref, rtol=1e-12)


def test_rpde_tmax_truncates():
    rng = np.random.default_rng(6)
    y = rng.standard_normal(500)
    y = (y - y.mean()) / y.std(ddof=1)
    h = _close_returns_naive(_embed(y, 2, 1), 0.12)[:50]
    rpd = h / h.sum()
    out = entropy.rpde(y, 2, 1, 0.12, 50)
    assert out["propNonZero"] == pytest.approx(np.mean(rpd > 0))
    assert out["maxRPD"] == pytest.approx(rpd.max() * 50)


def test_rpde_nan_cases():
    assert all(np.isnan(v) for v in entropy.rpde(np.ones(100), 3, "ac").values())  # tau undetermined
    assert all(np.isnan(v) for v in entropy.rpde(np.array([0.0, 1.0]), 2, 1).values())  # empty embedding
    # a monotone ramp never returns to a ball it has left
    assert all(np.isnan(v) for v in entropy.rpde(np.arange(100.0), 2, 1).values())
    assert set(entropy.rpde(np.sin(np.arange(300) / 3.0), 2, 5)) == set(KEYS)


# ------------------------------------------------------------------ MATLAB black-box fixture

def _fixture():
    fx = os.path.join(SPEC, "thirdparty_blackbox.json")
    inp = os.path.join(SPEC, "gen", "thirdparty_inputs.json")
    if not (os.path.exists(fx) and os.path.exists(inp)):
        pytest.skip("black-box fixture not available")
    return json.load(open(fx))["rpde"], json.load(open(inp))


def test_fixture_close_returns():
    rows, inp = _fixture()
    for r in rows:
        if "close_returns_hist" not in r:
            continue
        v = _embed(np.asarray(inp[r["series"]], float), int(r["m"]), int(r["tau"]))
        h = entropy._close_returns_hist(v, float(r["epsilon"]))
        np.testing.assert_array_equal(h, np.atleast_1d(r["close_returns_hist"]).astype(int))


def test_fixture_rpde():
    rows, inp = _fixture()
    for r in rows:
        if "rpde_3_1" not in r:
            continue
        y = np.asarray(inp[r["series"]], float)
        for key, args in [("rpde_3_1", (3, 1)), ("rpde_2_1", (2, 1, 0.12)), ("rpde_3_ac", (3, "ac"))]:
            got = entropy.rpde(y, *args)
            np.testing.assert_allclose([got[k] for k in KEYS], [r[key][k] for k in KEYS],
                                       rtol=1e-12, atol=1e-12, err_msg=f"{r['series']} {key}")
