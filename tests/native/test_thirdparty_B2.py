"""Accuracy tests for the clean-room Lempel-Ziv complexity, Shannon block entropy and
nearest-neighbour prediction-residual (nlpe) code.

Fixtures (fixtures_thirdparty_B2.json) are outputs recorded from hctsa in MATLAB R2026a; agreement is
evidence, not a requirement, so the Lempel-Ziv offset (published count c vs hctsa's c - 1) is applied
explicitly. All other tests use the published definitions and brute-force references written here."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import entropy, extreme_events, nonlinearity
from pyhctsa.operations._nlpe import _nn_prediction_residuals
from pyhctsa.operations._symbolic_complexity import (
    _equiprobable_symbols, _lempel_ziv_phrase_count, _normalized_lz_complexity,
    _shannon_block_entropy)

FX = json.loads((Path(__file__).parent / "fixtures_thirdparty_B2.json").read_text())
INP = {k: np.asarray(v, dtype=float) for k, v in FX["inputs"].items()}


# ---------------------------------------------------------------- Lempel-Ziv

def _brute_force_lz76(s):
    """Exhaustive-history component count, straight from the definition (Lempel and Ziv 1976)."""
    s = list(s)
    n, p, c = len(s), 0, 0
    while p < n:
        length = 1
        # extend while s[p:p+length] is reproducible from the text before its last symbol
        while p + length <= n and _is_substring(s[p:p + length], s[:p + length - 1]):
            length += 1
        c += 1
        p += length
    return c


def _is_substring(w, text):
    return any(text[i:i + len(w)] == w for i in range(len(text) - len(w) + 1))


@pytest.mark.parametrize("seq, c", [("0001101001000101", 6), ("0101010101010101", 3),
                                    ("0000000011111111", 3), ("0110101110010001101", 7),
                                    ("012012012012", 4), ("0120112200210", 8),
                                    ("01", 2), ("1", 1), ("00000000", 2)])
def test_lz_phrase_count_known(seq, c):
    assert _lempel_ziv_phrase_count(np.array([int(ch) for ch in seq])) == c


def test_lz_phrase_count_edge_and_random():
    assert _lempel_ziv_phrase_count(np.array([], dtype=int)) == 0
    rng = np.random.default_rng(1)
    for n in range(1, 61):
        for k in (2, 3):
            s = rng.integers(0, k, n)
            assert _lempel_ziv_phrase_count(s) == _brute_force_lz76(s), (n, k, s)
    for s in itertools.product([0, 1], repeat=8):  # exhaustive for short binary strings
        assert _lempel_ziv_phrase_count(np.array(s)) == _brute_force_lz76(s)


def test_equiprobable_symbols():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000)
    for nb in (2, 3, 7, 10):
        s = _equiprobable_symbols(x, nb)
        counts = np.bincount(s, minlength=nb + 1)[1:]
        assert s.min() == 1 and s.max() == nb and counts.max() - counts.min() <= 1
    xt = np.array([1., 1., 2., 3., 3., 3., 4., 5.])
    s = _equiprobable_symbols(xt, 3)
    assert s[0] == s[1] and s[3] == s[4] == s[5]
    assert np.all(np.diff(s) >= 0)  # monotone in the data


def test_normalized_lz_complexity_bs_fixture():
    for row in FX["MS_complexitybs"]:
        sym = np.atleast_1d(np.asarray(row["symbols"]))
        c = _lempel_ziv_phrase_count(sym)
        n, k = sym.size, int(sym.max()) + 1
        got = _normalized_lz_complexity(sym)
        if n == 1:
            assert got == 0.0
        elif k == 1:
            assert math.isinf(got)
        else:
            # published count c; hctsa's value is the same expression with c - 1
            assert got == pytest.approx(c * math.log(n) / (n * math.log(k)), abs=1e-12)
            assert (c - 1) * math.log(n) / (n * math.log(k)) == pytest.approx(row["complexitybs"], abs=1e-12)


def test_lempel_ziv_complexity_fixture():
    for row in FX["EN_LZComplexity"]:
        y = INP[row["series"]]
        for key, pre in (("lz", None), ("lz_diff", "diff")):
            for n_bits, ref in zip(row["n"], row[key]):
                got = entropy.lempel_ziv_complexity(y, n_bits, pre)
                z = y if pre is None else (np.diff(y) - np.diff(y).mean()) / np.diff(y).std(ddof=1)
                sym = _equiprobable_symbols(z, n_bits)
                c = _lempel_ziv_phrase_count(sym)
                k = sym.max()
                assert got == pytest.approx(c * math.log(z.size) / (z.size * math.log(k)), abs=1e-12)
                # hctsa does not count the final component
                assert (c - 1) * math.log(z.size) / (z.size * math.log(k)) == pytest.approx(ref, abs=1e-12)


def test_extreme_event_order_lz_fixture():
    for row in FX["EX_ExtremeEventOrder"]:
        y = INP[row["series"]]
        for thr, key in ((0.1, "lz_01"), (0.05, "lz_005")):
            got = extreme_events.extreme_event_order(y, thr)["lzComplexity"]
            # got = c ln N/(N ln 2) (published count), hctsa = (c - 1) ln N/(N ln 2): the offset is
            # one component, ln N/(N ln 2), for the integer number of events N
            assert _solve_n(got - row[key]) >= 10


def _solve_n(diff):
    for n in range(10, 5000):
        if abs(math.log(n) / (n * math.log(2)) - diff) < 1e-12:
            return n
    raise AssertionError("no N reproduces the offset")


# ---------------------------------------------------------------- Shannon block entropy

def test_shannon_fixture():
    for row in FX["shannon"]:
        y = np.asarray(row.get("x", INP.get(row["series"])), dtype=float)
        ref = np.asarray(row["MS_shannon"])
        for a, nb in enumerate(row["num_bins"]):
            for b, d in enumerate(row["depth"]):
                assert _shannon_block_entropy(y, nb, d) == pytest.approx(ref[a, b], abs=1e-12)


def test_shannon_wrapper_fixture():
    for row in FX["shannon"]:
        if "EN_Shannon_2_3" not in row:
            continue
        y = INP[row["series"]]
        assert entropy.shannon_entropy(y, 2, 3) == pytest.approx(row["EN_Shannon_2_3"], abs=1e-12)
        got = entropy.shannon_entropy(y, list(range(2, 11)), 2)
        for k, v in row["EN_Shannon_bins_2to10_d2"].items():
            assert got[k] == pytest.approx(v, abs=1e-12)


def test_shannon_hand_cases():
    y = np.arange(1, 9, dtype=float)
    expect = -(2 * (3 / 7) * math.log(3 / 7) + (1 / 7) * math.log(1 / 7))
    assert _shannon_block_entropy(y, 2, 2) == pytest.approx(expect, abs=1e-14)
    assert _shannon_block_entropy(y, 1, 3) == 0.0
    assert _shannon_block_entropy(y, 2, 8) == 0.0  # a single word
    assert np.isnan(_shannon_block_entropy(y, 2, 9))
    assert np.isnan(_shannon_block_entropy(np.array([1., np.nan, 2.]), 2, 1))
    # tie-free: balanced symbols, depth 1 -> ln(num_bins)
    assert _shannon_block_entropy(np.random.default_rng(0).standard_normal(600), 6, 1) == pytest.approx(math.log(6))


def test_shannon_large_alphabet_words():
    # n_bins ** depth beyond int64: falls back to row-unique counting
    y = np.random.default_rng(5).standard_normal(300)
    h = _shannon_block_entropy(y, 40, 12)
    assert h == pytest.approx(math.log(300 - 12 + 1))  # every word distinct


# ---------------------------------------------------------------- nlpe

def _brute_nlpe(y, de, tau, th):
    y = np.asarray(y, float)
    span = (de - 1) * tau
    t = list(range(span, len(y) - 1))
    v = [np.array([y[ti - k * tau] for k in range(de)]) for ti in t]
    res = []
    for i in range(len(t)):
        best, bj = np.inf, None
        for j in range(len(t)):
            if abs(i - j) <= th:
                continue
            d = np.sum((v[i] - v[j]) ** 2)
            if d < best:
                best, bj = d, j
        if bj is None:  # no neighbour outside the Theiler window: not predicted
            continue
        res.append(y[t[bj] + 1] - y[t[i] + 1])
    return np.array(res)


def test_nn_prediction_residuals_fixture():
    for row in FX["nlpe"]:
        if "res" not in row:
            continue
        got = _nn_prediction_residuals(INP[row["series"]], row["de"], row["tau"], row["theiler"])
        np.testing.assert_allclose(got, row["res"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("de, tau, th", [(2, 1, 0), (3, 1, 2), (2, 3, 5), (4, 2, 10)])
def test_nn_prediction_residuals_ties_and_block(de, tau, th, monkeypatch):
    y = np.round(np.random.default_rng(7).standard_normal(300), 1)
    ref = _brute_nlpe(y, de, tau, th)
    np.testing.assert_allclose(_nn_prediction_residuals(y, de, tau, th), ref, atol=1e-12)
    from pyhctsa.operations import _nlpe
    monkeypatch.setattr(_nlpe, "_BLOCK_BYTES", 8 * 300 * 7)  # tiny blocks
    np.testing.assert_allclose(_nn_prediction_residuals(y, de, tau, th), ref, atol=1e-12)


def test_nn_prediction_residuals_nan_cases():
    y = np.random.default_rng(3).standard_normal(50)
    assert np.isnan(_nn_prediction_residuals(y[:3], 3, 1))     # fewer than two states
    assert np.isnan(_nn_prediction_residuals(y, 2, 1, 100))    # window excludes everything
    assert _nn_prediction_residuals(y, 3, 2).shape == (50 - 1 - 4,)


def test_nn_prediction_residuals_skips_states_without_neighbours():
    y = np.random.default_rng(11).standard_normal(60)
    # m = 59 states; with a window of 35, states 23..35 have no neighbour outside it
    got = _nn_prediction_residuals(y, 1, 1, 35)
    ref = _brute_nlpe(y, 1, 1, 35)
    assert got.shape == (59 - 13,) and np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, atol=1e-12)
    # too few predictable states (here 17 of 59: under half): NaN
    assert np.isnan(_nn_prediction_residuals(y, 1, 1, 50))


def test_nlpe_finite_when_some_states_lack_neighbours():
    y = np.cumsum(np.random.default_rng(12).standard_normal(120))
    out = nonlinearity.nlpe(y, 2, 1, 5000, 60)  # m = 118: states 57..60 lack neighbours
    assert np.isfinite(out["msqerr"]) and np.isfinite(out["ac1"])


def test_nlpe_msqerr_fixture():
    # the mean squared residual part of the nlpe feature (no ARFIT needed)
    from pyhctsa.utils import theiler_window
    for row in FX["nlpe"]:
        if "res" in row:
            continue
        y = INP[row["series"]]
        th = int(theiler_window(y, ("ac", 1), y.size))
        got = np.mean(_nn_prediction_residuals(y, 3, 1, th) ** 2)
        assert got == pytest.approx(row["nlpe_3_1"]["msqerr"], abs=1e-8)


def test_nlpe_feature_fixture():
    for row in FX["nlpe"]:
        if "res" in row:
            continue
        y = INP[row["series"]]
        for key, args in (("nlpe_3_1", (3, 1, 5000, ("ac", 1))), ("nlpe_3_ac1e", (3, "ac1e", 5000, ("ac", 1))),
                          ("nlpe_2_2_th0", (2, 2, 5000, 0))):
            try:
                got = nonlinearity.nlpe(y, *args)
            except NotImplementedError:
                pytest.skip("residual_analysis depends on ARFIT (another work package)")
            for k, v in row[key].items():
                assert got[k] == pytest.approx(v, abs=1e-8), (row["series"], key, k)
