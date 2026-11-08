"""doubled_phase_coherence: the record-long coherence of the squared Morlet band signal, with its Gaussian
spectral null.

Exact limits (a sinusoid), the Isserlis null against Monte Carlo, calibration of the excess on linear Gaussian
processes and its independence of length, the sign invariance at h = 2 (bursts of random sign on one clock
against bursts with independent phases), degenerate inputs and invalid parameters, the calculator entry, and
a pin of rhythmicity's outputs, which shares its wavelet transform and lag covariance."""
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import spectral as sp

FIELDS = ["coh_max", "coh_freqAtMax", "coh_mean", "coh_atPeakPower", "ex_max", "ex_mean", "ex_atPeakPower"]
W = 2 * np.pi / 12.3  # angular frequency of the test oscillations


def _ar1(n, phi, rng):
    e = rng.standard_normal(n + 200)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def _bursts(n, rng, polarity):
    """Bursts of one oscillation (60-200 samples, sin**2 envelope) separated by silent gaps of 150-400 samples,
    plus a little noise. With ``polarity`` one clock runs through the gaps and each burst takes a random sign;
    otherwise each burst has a fresh uniform phase."""
    y = np.zeros(n)
    t, clock = 0, rng.uniform(0, 2 * np.pi)
    while True:
        t += rng.integers(150, 400)
        L = rng.integers(60, 200)
        if t >= n:
            break
        s = np.arange(t, min(n, t + L))
        phase = clock + np.pi * rng.integers(0, 2) if polarity else rng.uniform(0, 2 * np.pi)
        y[s] += np.sin(np.pi * (s - t) / L) ** 2 * np.cos(W * s + phase)
        t += L
    return y + 0.1 * rng.standard_normal(n)


def test_output_names():
    out = sp.doubled_phase_coherence(np.random.default_rng(0).standard_normal(1500))
    assert list(out) == FIELDS
    assert all(np.isfinite(v) for v in out.values())
    assert all(0 <= out[k] <= 1 for k in ("coh_max", "coh_mean", "coh_atPeakPower"))


def test_pure_sinusoid_is_fully_coherent():
    y = np.cos(W * np.arange(4000) + 0.3)
    for h in (1, 2):
        out = sp.doubled_phase_coherence(y, harmonic=h)
        assert out["coh_atPeakPower"] == pytest.approx(1.0, abs=2e-3)
        assert out["coh_max"] <= 1 + 1e-12
    out = sp.doubled_phase_coherence(y)
    # C**2 = 1 against a null of about 0.25: a Gaussian process with a line spectrum is itself narrowband, so
    # few DFT ordinates of its X**2 share the band and its expected maximum is large
    assert 0.6 < out["ex_atPeakPower"] < 0.9


@pytest.mark.parametrize("h", [2, 3])
def test_isserlis_null_matches_monte_carlo(h):
    # X_t = e_t + a e_{t-1}, e circular complex Gaussian: c(0) = 1 + |a|**2, c(1) = a. A complex a makes the
    # expected periodogram of X**h asymmetric in nu, so a wrong lag sign or conjugate would show.
    rng = np.random.default_rng(5)
    a, n_e, n_fft, reps = 0.6 * np.exp(0.8j), 48, 96, 20000
    e = (rng.standard_normal((reps, n_e + 1)) + 1j * rng.standard_normal((reps, n_e + 1))) / np.sqrt(2)
    X = e[:, 1:] + a * e[:, :-1]
    mc = np.mean(np.abs(np.fft.fft(X ** h, n_fft, axis=1)) ** 2, axis=0)
    null = sp._isserlis_power_null(np.array([1 + abs(a) ** 2, a]), h, n_e, n_fft)
    assert np.max(np.abs(mc / null - 1)) < 0.06
    assert np.mean(mc) / np.mean(null) == pytest.approx(1, abs=0.01)


def test_expected_max_of_exponentials():
    # for J equal means s the expected maximum is s H_J, H_J the J-th harmonic number
    for J in (1, 10, 1000):
        assert sp._expected_max_exponential(np.full(J, 2.0)) == pytest.approx(2 * np.sum(1 / np.arange(1, J + 1)), rel=3e-3)
    assert sp._expected_max_exponential(np.array([1.0, 1e-9])) == pytest.approx(1.0, rel=3e-3)


@pytest.mark.parametrize("gen", ["white", "ar1"])
def test_excess_is_zero_for_linear_gaussian_processes(gen):
    outs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(2048) if gen == "white" else _ar1(2048, 0.9, rng)
        outs.append(sp.doubled_phase_coherence(y))
    # per-series spread is about 0.01 (ex_mean) and 0.05 (ex_atPeakPower)
    assert abs(np.mean([o["ex_mean"] for o in outs])) < 0.02
    assert abs(np.mean([o["ex_atPeakPower"] for o in outs])) < 0.04


def test_no_length_drift_of_excess_on_noise():
    med = {}
    for n in (1000, 8000):
        outs = [sp.doubled_phase_coherence(np.random.default_rng(100 + s).standard_normal(n)) for s in range(6)]
        med[n] = {k: np.median([o[k] for o in outs]) for k in FIELDS}
    for k in ("ex_mean", "ex_atPeakPower"):
        assert abs(med[1000][k] - med[8000][k]) < 0.04
    # the raw coherence carries a noise floor that falls as N**-1/2 (why it is out of the default set)
    assert med[1000]["coh_mean"] - med[8000]["coh_mean"] > 0.1


def test_random_polarity_is_coherent_at_h2_only():
    def med(polarity, h, key):
        return np.median([sp.doubled_phase_coherence(_bursts(4096, np.random.default_rng(k), polarity),
                                                     harmonic=h)[key] for k in range(5)])
    # h = 2: one clock modulo pi (about 0.8) against independent phases (about 0.4)
    assert med(True, 2, "ex_atPeakPower") > med(False, 2, "ex_atPeakPower") + 0.25
    # h = 1: the random signs cancel, so the two are alike (both spectrally determined)
    assert abs(med(True, 1, "ex_atPeakPower") - med(False, 1, "ex_atPeakPower")) < 0.05


def test_degenerate_inputs_are_nan():
    rng = np.random.default_rng(0)
    for y in (np.ones(1000), np.r_[rng.standard_normal(999), np.nan], np.r_[rng.standard_normal(999), np.inf],
              rng.standard_normal(285)):  # 286 samples are needed for 10 qualifying frequencies
        out = sp.doubled_phase_coherence(y)
        assert list(out) == FIELDS and all(np.isnan(v) for v in out.values())
    assert np.isfinite(sp.doubled_phase_coherence(rng.standard_normal(286))["ex_mean"])


@pytest.mark.parametrize("kwargs", [dict(f_high=0.4), dict(m=3, f_high=0.3), dict(harmonic=0), dict(harmonic=1.5),
                                    dict(pad=0), dict(f_low=0.3), dict(f_low=0), dict(m=0)])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        sp.doubled_phase_coherence(np.random.default_rng(0).standard_normal(1000), **kwargs)


def test_calculator_entry(tmp_path):
    from pyhctsa.calculator import FeatureCalculator
    # the registered entry, as written in hctsa.yaml (which has custom tags, so it is cut out as text)
    text = (Path(__file__).resolve().parents[1] / "pyhctsa" / "configurations" / "hctsa.yaml").read_text()
    entry = text[text.index("  doubled_phase_coherence:"):].split("\n\n")[0]
    path = tmp_path / "dpc.yaml"
    path.write_text("spectral:\n" + entry + "\n")
    df = FeatureCalculator(config_path=str(path)).extract(np.random.default_rng(0).standard_normal(1500))
    assert list(df.columns) == ["dpc_2.ex_max", "dpc_2.ex_mean", "dpc_2.ex_atPeakPower"]
    assert np.all(np.isfinite(df.to_numpy()))


# ---- rhythmicity shares _morlet_transform and _band_lag_sums: its outputs are pinned to their values before
# the helpers were factored out (pyhctsa main 0fa9013) ----

def _rhythm_cases():
    rng = np.random.default_rng(20261107)
    e = rng.standard_normal(1200)
    ar = np.zeros(1200)
    for t in range(1, 1200):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    t = np.arange(1500)
    sine = np.cos(2 * np.pi * t / 11.7) + 0.5 * rng.standard_normal(1500)
    walk = np.cumsum(rng.standard_normal(600))
    spiky = rng.standard_normal(900) ** 3
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    return {"ar1": (z(ar), {}), "sine": (z(sine), {}), "walk": (z(walk), {}), "spiky": (z(spiky), {}),
            "ar1_params": (z(ar), dict(m=7, tau=3, n_freq=25, f_low=0.03, f_high=0.35, smooth_win=3, min_cycles=30))}


RHYTHM_PINNED = {
    "ar1": [0.33230328874117027, 0.32078233937027184, 0.06649133973149546, 0.06844252439773618, 0.48779199392377287, 0.17568480560575825, 1.5206323230928418, 0.4295025629099799, 0.315883386824165, 0.061896327403143644, 0.0967741935483871, 0.03225806451612903, 0.4861677035099384, 1.0, 0.36660994356051213, 0.014039298152008987, 0.05786142890105578, 0.15613784568800526, -0.1074610477779164, 0.4637944028115589, 0.315883386824165, 0.03142572615215983],
    "sine": [0.4768535983865546, 0.35228547178774616, 0.2825961122392189, 0.40929940764686723, 0.9909163167815515, 0.16720652626530927, 2.812821975748644, 0.5408103445168216, 0.21514342015905452, 0.32387003872687103, 0.2647058823529412, 0.0, 0.88649266522163, 1.0, 0.9909163167815515, 0.035167412605262474, 0.07774537759040878, 0.25789746082487686, -0.062015528508208434, 0.3977461789705834, 1.711753175841694, 0.04729880089821292],
    "walk": [0.3209403140893543, 0.3165752271007704, 0.07114342913211218, 0.11586478282169277, 0.46870803651866744, 0.2071102189165997, 1.4805581624662973, 0.925898103337763, 1.1658493231355112, 0.06204120303638596, 0.0, 0.0, 0.420020743804407, 0.0, 0.38648422838533036, 0.0017360698992696225, 0.04574401453078706, 0.07073381214119873, -0.08441766024368144, 0.925898103337763, 0.6809641059061708, 0.06600420439112131],
    "spiky": [0.379508688984041, 0.372175793290009, 0.08740272219837317, 0.08687574247397095, 0.592605152559982, 0.14240807556445356, 1.5922721553741912, 0.4295025629099799, 0.341103777722771, 0.07344742440281168, 0.07407407407407407, 0.037037037037037035, 0.4075746645731947, 1.0, 0.592605152559982, 0.04468168774812658, 0.040835727687677396, 0.11834802229129288, -0.029929014662572917, 1.8484209653405093, 0.341103777722771, 0.09183024023583841],
    "ar1_params": [0.17586977086868721, 0.1676975811320179, 0.09174229026608914, 0.14457857038612257, 0.3465697464767534, 0.026915233772473103, 2.0666353333022767, 0.790108529824932, 0.8752716594233417, 0.08771235270935185, 0.0, 0.0, -0.10353639938773024, 0.0, 0.2890227016949445, 0.02985423057299208, 0.04690638480787211, 0.1107257468064752, -0.046381348434458375, 0.3483676512722629, 0.28387398082888277, 0.06636251209779617],
}


@pytest.mark.parametrize("name", list(RHYTHM_PINNED))
def test_rhythmicity_unchanged_by_shared_helpers(name):
    y, params = _rhythm_cases()[name]
    got = np.array(list(sp.rhythmicity(y, **params).values()))
    np.testing.assert_allclose(got, RHYTHM_PINNED[name], rtol=1e-12, atol=1e-13)
