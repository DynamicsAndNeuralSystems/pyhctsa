"""long_memory_d: the local Whittle estimate (Künsch 1987; Robinson 1995) of the memory parameter d on the
Fourier frequencies below 1/128 and 1/32 cycles per sample.

The value against a brute-force minimization of the local Whittle objective, calibration on ARFIMA(0, d, 0)
simulated exactly (circulant embedding), short memory and the fixed-band design (no drift with length), the
nonstationary range, the NaN rules, invariances, and the registry entry."""

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.special import gammaln

from pyhctsa.operations import scaling as sc

FIELDS = {"d", "d_32", "d_diff"}


def _fractional_noise(d, n, rng):
    # ARFIMA(0, d, 0) by circulant embedding of its autocovariance (Davies & Harte 1987), using
    # gamma(0) = Gamma(1 - 2d) / Gamma(1 - d)^2 and gamma(k) = gamma(k - 1) (k - 1 + d) / (k - d)
    k = np.arange(1, n + 1)
    g = np.exp(gammaln(1 - 2 * d) - 2 * gammaln(1 - d)) * np.r_[1.0, np.cumprod((k - 1 + d) / (k - d))]
    ev = np.clip(np.fft.fft(np.r_[g, g[-2:0:-1]]).real, 0, None)
    z = rng.standard_normal(ev.size) + 1j * rng.standard_normal(ev.size)
    return np.fft.fft(np.sqrt(ev / ev.size) * z).real[:n]


def _ar1(phi, n, rng, burn=300):
    return lfilter([1], [1, -phi], rng.standard_normal(n + burn))[burn:]


def _brute_force_lw(y, m):
    # the local Whittle objective written out directly: periodogram by an explicit DFT sum, R(d) on a fine grid
    n = len(y)
    t = np.arange(n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    dft = np.exp(-1j * np.outer(lam, t)) @ (y - y.mean())
    pgram = np.abs(dft) ** 2 / (2 * np.pi * n)
    grid = np.linspace(-0.99, 1.99, 298001)  # spacing 1e-5
    r = np.log(np.mean(lam[None, :] ** (2 * grid[:, None]) * pgram[None, :], axis=1)) \
        - 2 * grid * np.mean(np.log(lam))
    return grid[np.argmin(r)]


def test_output_names():
    out = sc.long_memory_d(np.random.default_rng(0).standard_normal(2048))
    assert set(out) == FIELDS and all(np.isfinite(v) for v in out.values())


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_brute_force_objective(seed):
    # N = 1024: d uses m = 8 frequencies, d_32 uses m = 32
    y = _ar1(0.6, 1024, np.random.default_rng(seed)) + 0.3 * np.cumsum(np.random.default_rng(seed + 10).
                                                                       standard_normal(1024)) / 32
    out = sc.long_memory_d(y)
    assert out["d"] == pytest.approx(_brute_force_lw(y, 8), abs=5e-5)
    assert out["d_32"] == pytest.approx(_brute_force_lw(y, 32), abs=5e-5)


@pytest.mark.parametrize("d", [-0.3, 0.0, 0.3])
def test_calibrated_on_fractional_noise(d):
    # N = 4096: m = 32 (d) and 128 (d_32), asymptotic standard errors 0.088 and 0.044. Over 20 series the
    # mean's standard error is about 0.025 and 0.01; the tolerances are about three times those.
    rng = np.random.default_rng(int(10 * d) + 20)
    outs = [sc.long_memory_d(_fractional_noise(d, 4096, rng)) for _ in range(20)]
    est = np.array([o["d"] for o in outs])
    est32 = np.array([o["d_32"] for o in outs])
    assert abs(est.mean() - d) < 0.08 and 0.05 < est.std() < 0.18
    assert abs(est32.mean() - d) < 0.035 and 0.025 < est32.std() < 0.08


def test_short_memory_ar1():
    # AR(1), coefficient 0.8: d = 0, but its spectrum bends within the 1/32 band (by about +0.08)
    rng = np.random.default_rng(3)
    outs = [sc.long_memory_d(_ar1(0.8, 8192, rng)) for _ in range(20)]
    d = np.array([o["d"] for o in outs])
    d_diff = np.array([o["d_diff"] for o in outs])
    assert abs(np.median(d)) < 0.05
    assert 0.02 < np.median(d_diff) < 0.15


def test_fixed_band_has_no_length_drift_under_short_memory():
    # with bands fixed in cycles per sample, neither expectation moves with N on a short-memory process
    rng = np.random.default_rng(1)
    med = {n: np.median([[sc.long_memory_d(_ar1(0.8, n, rng))[k] for k in ("d", "d_32")] for _ in range(40)],
                        axis=0)
           for n in (2048, 16384)}
    assert abs(med[2048][0] - med[16384][0]) < 0.08  # d: m = 16 and 128
    assert abs(med[2048][1] - med[16384][1]) < 0.03  # d_32: m = 64 and 512


def test_random_walk_gives_about_one():
    rng = np.random.default_rng(2)
    outs = [sc.long_memory_d(np.cumsum(rng.standard_normal(8192))) for _ in range(5)]
    assert all(0.8 < o["d"] < 1.2 and 0.9 < o["d_32"] < 1.1 for o in outs)


def test_length_thresholds():
    rng = np.random.default_rng(6)
    for n in (256, 1023):
        out = sc.long_memory_d(rng.standard_normal(n))
        assert np.isnan(out["d"]) and np.isnan(out["d_diff"]) and np.isfinite(out["d_32"])
    assert all(np.isfinite(v) for v in sc.long_memory_d(rng.standard_normal(1024)).values())


def test_degenerate_inputs_give_nan():
    rng = np.random.default_rng(6)
    for y in (rng.standard_normal(255), np.full(4096, 0.1), np.r_[np.nan, rng.standard_normal(4095)],
              np.r_[np.inf, rng.standard_normal(4095)]):
        out = sc.long_memory_d(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())


def test_spectral_line_in_band_is_nan():
    # a sinusoid of period 500 lies inside the 1/128 band: R(d) is minimized at a bound, so d is NaN
    x = np.sin(2 * np.pi * np.arange(8192) / 500) + 0.5 * np.random.default_rng(3).standard_normal(8192)
    out = sc.long_memory_d(x)
    assert np.isnan(out["d"]) and np.isnan(out["d_diff"]) and np.isfinite(out["d_32"])


def test_d_diff_is_exactly_the_difference():
    out = sc.long_memory_d(_ar1(0.5, 4096, np.random.default_rng(4)))
    assert out["d_diff"] == out["d_32"] - out["d"]


def test_scale_and_shift_invariant():
    y = np.random.default_rng(5).standard_normal(4096)
    a, b = sc.long_memory_d(y), sc.long_memory_d(3 * y - 7)
    for k in FIELDS:
        assert a[k] == pytest.approx(b[k], abs=1e-5)


def test_registered_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    out = calc.feature_funcs["long_memory_d"](np.random.default_rng(7).standard_normal(512))
    assert set(out) == FIELDS and np.isnan(out["d"]) and np.isfinite(out["d_32"])
    sub = FeatureCalculator.__new__(FeatureCalculator)  # extract with this one operation only
    sub.feature_funcs = {"long_memory_d": calc.feature_funcs["long_memory_d"]}
    df = sub.extract(np.random.default_rng(8).standard_normal(2048))
    assert sorted(df.columns) == ["long_memory_d.d", "long_memory_d.d_32", "long_memory_d.d_diff"]
    assert np.isfinite(df.to_numpy(dtype=float)).all()
