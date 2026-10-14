"""Tests for ``pyhctsa.operations.spectral.rhythmicity`` (the LAVI phase-persistence spectrum and its analytic
spectral null): reference values from hctsa's SP_Rhythmicity, edge cases, and analytic checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.special

from pyhctsa.operations import spectral as sp
from pyhctsa.utils import z_score

REF = json.loads((Path(__file__).parent / "data" / "rhythmicity_ref.json").read_text())
NAMES = REF["output_names"]


def _ar1(n, phi, seed):
    e = np.random.default_rng(seed).standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def test_output_names_and_order():
    out = sp.rhythmicity(z_score(_ar1(1000, 0.5, 1)))
    assert list(out) == NAMES
    assert len(NAMES) == 22 and NAMES[0] == "lavi_mean" and NAMES[-1] == "ex_atPeakPower"


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: c["name"])
def test_matches_hctsa_reference(case):
    """Values from hctsa's SP_Rhythmicity on the same (z-scored) series: a 4-series sample of the 1000 x 1000
    dataset at the registered settings, synthetic series of other lengths, and a non-default parameter set."""
    y = np.asarray(case["y"])
    y = z_score(y) if y.std() > 0 else y
    out = sp.rhythmicity(y, **case["params"])
    expected = np.array([np.nan if v is None else v for v in case["expected"]])
    got = np.array([out[k] for k in NAMES])
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-8)


def test_too_short_series_is_nan_like_hctsa():
    # fewer than 10 frequencies clear the edge-free-cycles criterion below about 240 samples
    y = z_score(np.random.default_rng(3).standard_normal(200))
    out = sp.rhythmicity(y)
    assert list(out) == NAMES and all(np.isnan(v) for v in out.values())
    assert not np.isnan(sp.rhythmicity(z_score(np.random.default_rng(3).standard_normal(400)))["lavi_mean"])


def test_constant_series_is_nan():
    for y in (np.full(500, 3.0), np.zeros(500)):
        out = sp.rhythmicity(y)
        assert list(out) == NAMES and all(np.isnan(v) for v in out.values())


def test_invalid_frequency_limits_raise():
    y = np.random.default_rng(0).standard_normal(1000)
    with pytest.raises(ValueError):
        sp.rhythmicity(y, f_high=0.5)
    with pytest.raises(ValueError):
        sp.rhythmicity(y, f_low=0.0)
    with pytest.raises(ValueError):
        sp.rhythmicity(y, f_low=0.3, f_high=0.2)


def test_invariant_to_scale_offset_and_orientation():
    y = _ar1(1000, 0.6, 2)
    a = sp.rhythmicity(y)
    b = sp.rhythmicity(5.0 * y + 7.0)
    c = sp.rhythmicity(y.reshape(1, -1))
    for k in NAMES:
        assert b[k] == pytest.approx(a[k], abs=1e-9)
        assert c[k] == a[k]


def test_null_matches_hypergeometric_form():
    rho = np.array([0.0, 1e-8, 1e-3, 0.2, 0.41, 0.8, 0.999, 1.0])
    got = sp._lavi_from_rho(rho)
    want = np.pi / 4 * rho * scipy.special.hyp2f1(0.5, 0.5, 2.0, rho ** 2)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-15)
    assert got[-1] == 1.0
    # the white-noise floor quoted for the defaults: rho = exp(-pi^2 tau^2 / m^2) = 0.41, LAVI0 = 0.33
    r0 = np.exp(-np.pi ** 2 * 1.5 ** 2 / 5 ** 2)
    assert r0 == pytest.approx(0.41, abs=0.005)
    assert sp._lavi_from_rho(r0)[0] == pytest.approx(0.33, abs=0.005)


def test_white_noise_has_no_excess_over_the_null():
    y = z_score(np.random.default_rng(11).standard_normal(5000))
    out = sp.rhythmicity(y)
    assert abs(out["ex_mean"]) < 0.02
    assert 0.3 < out["lavi_median"] < 0.37  # the white-noise floor


def test_sustained_oscillation_is_rhythmic_at_its_frequency():
    rng = np.random.default_rng(5)
    t = np.arange(3000)
    y = z_score(np.sin(2 * np.pi * 0.1 * t) + 0.5 * rng.standard_normal(t.size))
    out = sp.rhythmicity(y)
    assert out["lavi_max"] > 0.9
    assert out["lavi_freqAtMax"] == pytest.approx(2 * np.pi * 0.1, rel=0.1)
    assert out["lavi_atPeakPower"] > 0.9  # the dominant oscillation is the most rhythmic band


def test_bursty_oscillation_is_less_rhythmic_than_a_sustained_one():
    rng = np.random.default_rng(6)
    n = 4000
    t = np.arange(n)
    carrier = np.sin(2 * np.pi * 0.1 * t)
    gate = np.convolve((rng.random(n) < 0.004).astype(float), np.ones(60), "same") > 0
    noise = rng.standard_normal(n)
    burst = sp.rhythmicity(z_score(gate * carrier + 0.4 * noise))
    steady = sp.rhythmicity(z_score(0.4 * carrier + 0.4 * noise))
    assert burst["lavi_atPeakPower"] < steady["lavi_atPeakPower"]
    assert burst["ex_atPeakPower"] < steady["ex_atPeakPower"]
