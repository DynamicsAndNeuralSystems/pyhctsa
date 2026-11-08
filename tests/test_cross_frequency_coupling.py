"""cross_frequency_coupling: phase-amplitude coupling (modulation index) and lagged amplitude envelope
correlation between frequency bands, with a null-calibrated peak score.

Null calibration on linear Gaussian processes, detection of phase-amplitude coupling and of envelopes coupled
with a delay, invariances, and degenerate inputs."""

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.special import ndtr, ndtri

from pyhctsa.operations import spectral as sp

FIELDS = {"maxMI", "entropyMI", "maxAEC0", "maxAEC", "zAEC"}


def _ar1(n, phi, rng):
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 500))[500:]


@pytest.mark.parametrize("n", [500, 4000])
def test_null_calibrated_at_any_length(n):
    # for a linear Gaussian process the envelopes of disjoint bands are asymptotically independent, so zAEC
    # (corrected for the lags and band pairs scanned) stays near the same small value at every length
    rng = np.random.default_rng(n)
    z = [sp.cross_frequency_coupling(y)["zAEC"]
         for y in [rng.standard_normal(n) for _ in range(15)] + [_ar1(n, 0.8, rng) for _ in range(15)]]
    assert 0.3 < np.median(z) < 2.2
    assert np.mean(np.array(z) > 3.5) <= 0.1


def _lognormal_envelope(n, tau, rng):
    a = np.exp(-1.0 / tau)
    return np.exp(0.8 * lfilter([np.sqrt(1 - a * a)], [1.0, -a], rng.standard_normal(n + 2000))[2000:])


def _rhythms(n, lag, coupled, rng):
    # two rhythms in the middles of different bands, each with a slowly varying lognormal amplitude, in white
    # noise strong enough to cover each rhythm's spectral spread into its neighbouring bands; when coupled, the
    # fast rhythm's amplitude follows the slow rhythm's with a delay of `lag` samples
    slow_env = _lognormal_envelope(n + lag, 100, rng)
    fast_env = slow_env[:n] if coupled else _lognormal_envelope(n, 100, rng)
    t = np.arange(n)
    return (slow_env[lag:] * np.cos(2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi))
            + fast_env * np.cos(2 * np.pi * 0.35 * t + rng.uniform(0, 2 * np.pi))
            + 1.5 * rng.standard_normal(n))


def test_detects_delayed_envelope_coupling():
    rng = np.random.default_rng(1)
    hits = [sp.cross_frequency_coupling(_rhythms(4000, 250, True, rng)) for _ in range(12)]
    nulls = [sp.cross_frequency_coupling(_rhythms(4000, 250, False, rng)) for _ in range(12)]
    z_hit = np.array([h["zAEC"] for h in hits])
    z_null = np.array([h["zAEC"] for h in nulls])
    # the lognormal envelopes vary on a timescale of 100 samples, so each record holds only ~40 independent
    # envelope epochs and the scores spread widely; compare the two distributions
    assert np.median(z_hit) > 4.5 and np.median(z_null) < 2.5
    assert np.mean(z_hit > 3.5) >= 0.75
    # the coupling is at a delay of 250 samples (several envelope correlation times), so the classical
    # zero-lag AEC barely sees it, while the lagged peak does
    assert np.median([h["maxAEC"] - h["maxAEC0"] for h in hits]) > 0.3


def test_detects_phase_amplitude_coupling():
    # a fast rhythm whose amplitude follows the phase of a slow one, against the same rhythms uncoupled
    rng = np.random.default_rng(2)
    t = np.arange(2000)

    def signal(coupled):
        slow = np.cos(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
        amp = 1 + 0.8 * slow if coupled else 1 + 0.8 * np.cos(2 * np.pi * 0.0123 * t)
        return slow + amp * np.cos(2 * np.pi * 0.3 * t) + 0.5 * rng.standard_normal(len(t))

    hit = [sp.cross_frequency_coupling(signal(True))["maxMI"] for _ in range(5)]
    null = [sp.cross_frequency_coupling(signal(False))["maxMI"] for _ in range(5)]
    assert min(hit) > 3 * max(null)


def test_scale_and_shift_invariant():
    y = np.random.default_rng(3).standard_t(4, 1000)
    a, b = sp.cross_frequency_coupling(y), sp.cross_frequency_coupling(-3 * y + 7)
    assert set(a) == FIELDS
    for k in FIELDS:
        assert a[k] == pytest.approx(b[k])


def test_max_normal_score():
    # one draw: unchanged; the max of n draws is shifted down; stable far into the tail
    assert sp._max_normal_score(1.3, 1) == pytest.approx(1.3)
    assert sp._max_normal_score(3.0, 100) == pytest.approx(ndtri(ndtr(3.0) ** 100))
    assert np.isfinite(sp._max_normal_score(40.0, 1e4)) and sp._max_normal_score(40.0, 1e4) < 40


def test_degenerate_inputs():
    for y in (np.arange(15.0), np.ones(200)):
        out = sp.cross_frequency_coupling(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())
    for bad in ({"max_lag_frac": 0.0}, {"max_lag_frac": 1.0}, {"n_bands": 1}):
        with pytest.raises(ValueError):
            sp.cross_frequency_coupling(np.random.default_rng(0).standard_normal(200), **bad)


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    fc = FeatureCalculator()
    labels = [lab for lab in fc.min_lengths if lab.startswith("cfc")]
    assert labels == ["cfc_5"]
