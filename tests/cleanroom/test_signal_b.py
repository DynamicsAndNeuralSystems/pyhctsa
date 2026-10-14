"""Cross-implementation checks of the MODWT pyramid and the Shannon entropy cost against
reference outputs stored in ``fixtures/*_b.json``. (Accuracy tests of the spectral, wavelet,
pre-processing and change-point operations are in ``tests/native/test_wp3_signal.py``.)
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import entropy as en
from pyhctsa.operations import wavelet as wl

FIX = Path(__file__).parent / "fixtures"


def load(name):
    with open(FIX / name) as f:
        return json.load(f)


def nan_array(a):
    return np.array([np.nan if v is None else v for v in a], dtype=float)


# ---------------------------------------------------------------- MODWT
MOD = load("modwt_b.json")


@pytest.mark.parametrize("case", MOD["cases"], ids=lambda c: f"{c['wname']}-J{c['level']}-N{c['N']}")
def test_modwt(case):
    x = np.array(case["x"])
    W = wl._modwt(x, case["wname"], case["level"])
    ref = np.array(case["w"])
    tol = 1e-9 if case["wname"].startswith("db") else 1e-12
    assert W.shape == ref.shape
    np.testing.assert_allclose(W, ref, atol=tol, rtol=0)
    # energy conservation
    np.testing.assert_allclose(np.sum(W ** 2), np.sum(x ** 2), rtol=1e-10)  # limited by the published filter precision


def test_modwt_impulse_alignment():
    ip = MOD["impulse_probe"]
    x = np.array(ip["x"], float)
    np.testing.assert_allclose(wl._modwt(x, "db3", 2), ip["db3_J2"], atol=1e-9)
    np.testing.assert_allclose(wl._modwt(x, "haar", 2), ip["haar_J2"], atol=1e-12)


MWE = load("modwt_wentropy_b.json")


@pytest.mark.parametrize("case", MWE, ids=lambda c: f"{c['kind']}-N{c['N']}-{c['wname']}")
def test_modwt_band_energies_and_entropy(case):
    if case["error"]:
        assert np.isnan(en.wavelet_entropy(np.random.default_rng(0).standard_normal(case["N"]),
                                           case["wname"], case["level"]))
        return
    y = np.array(case["y"])
    E = en._modwt_band_energies(y, case["wname"], case["level"])
    assert E.shape == (case["level"] + 1,)
    # the MODWT conserves energy (Percival & Walden 2000, eq. 171a), and no band is floored
    assert E.sum() == pytest.approx(np.sum(y ** 2), rel=1e-10)
    assert np.all(E >= 0)
    # entropy of the relative band energies, normalized by log2(level + 1)
    p = E / E.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p)) / np.log2(case["level"] + 1)
    assert en.wavelet_entropy(y, case["wname"], case["level"]) == pytest.approx(ent, abs=1e-12)
    assert 0 <= ent <= 1 + 1e-12


# ---------------------------------------------------------------- Shannon entropy cost
WEN = load("wentropy_b.json")


@pytest.mark.parametrize("i", range(len(WEN["cases"])))
def test_shannon_entropy_cases(i):
    c = WEN["cases"][i]
    got = wl._shannon_entropy(np.array(c["x"], float))
    assert abs(got - c["ent"]) <= 1e-12 + 1e-12 * abs(c["ent"])


def test_shannon_entropy_probe_and_properties():
    p = WEN["wpdec_probe"]
    for coeffs, ent in zip(p["node_coeffs"], p["ent"]):
        got = wl._shannon_entropy(np.array(coeffs, float))
        assert abs(got - ent) <= 1e-9 * abs(ent) + 1e-12
    assert abs(wl._shannon_entropy(np.array([0.5] * 4)) - 4 * 0.25 * np.log(4)) < 1e-14
    assert wl._shannon_entropy(np.array([])) == 0.0
    assert wl._shannon_entropy(np.zeros(5)) == 0.0
    assert np.isnan(wl._shannon_entropy(np.array([1.0, np.nan])))
    a, b = np.array([0.3, -1.2]), np.array([2.0, 0.0, 0.1])
    assert abs(wl._shannon_entropy(np.concatenate([a, b]))
               - wl._shannon_entropy(a) - wl._shannon_entropy(b)) < 1e-14
