"""Accuracy tests for detrended fluctuation analysis (SC_FluctAnal with wtf='dfa')."""
import numpy as np
import pytest

from pyhctsa.operations.scaling import (
    _detrend_windows, fluctuation_analysis)


def fgn(n, hurst, seed):
    """Fractional Gaussian noise by the Davies-Harte circulant embedding method."""
    rng = np.random.default_rng(seed)
    k = np.arange(n + 1, dtype=float)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([gamma, gamma[-2:0:-1]])  # first row of the circulant, length 2n
    lam = np.fft.fft(row).real
    assert lam.min() > -1e-10
    lam = np.clip(lam, 0, None)
    m = row.size
    w = np.zeros(m, dtype=complex)
    w[0] = np.sqrt(lam[0] / m) * rng.standard_normal()
    w[m // 2] = np.sqrt(lam[m // 2] / m) * rng.standard_normal()
    j = np.arange(1, m // 2)
    a, b = rng.standard_normal(j.size), rng.standard_normal(j.size)
    w[j] = np.sqrt(lam[j] / (2 * m)) * (a + 1j * b)
    w[m - j] = np.conj(w[j])
    return np.fft.fft(w).real[:n]


def alpha(x, **kw):
    return fluctuation_analysis(x, wtf='dfa', tau_step=50, **kw)['alpha']


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7, 0.9])
def test_fgn_exponent_matches_hurst(hurst):
    est = [alpha(fgn(2000, hurst, seed), k=1) for seed in range(10)]
    assert abs(np.mean(est) - hurst) < 0.05


def test_white_noise_exponent_is_half():
    est = [alpha(np.random.default_rng(s).standard_normal(2000)) for s in range(10)]
    assert abs(np.mean(est) - 0.5) < 0.05


def test_cumulated_fgn_has_exponent_h_plus_one():
    est = [alpha(np.cumsum(fgn(2000, 0.5, s)), k=2) for s in range(10)]
    assert abs(np.mean(est) - 1.5) < 0.1


def test_k0_is_not_dfa_and_saturates():
    # no detrending: the exponent of a nonstationary series cannot exceed about 1
    assert alpha(np.cumsum(fgn(2000, 0.8, 1)), k=0) < 1.15


@pytest.mark.parametrize("hurst", [0.4, 0.7])
def test_mfdfa_monofractal_h_is_constant_in_q(hurst):
    qs = [-2, 0, 0.5, 2, 4]
    h = np.array([[alpha(fgn(2000, hurst, s), q=q) for q in qs] for s in range(10)]).mean(axis=0)
    assert np.all(np.abs(h - hurst) < 0.08), h
    assert np.ptp(h) < 0.1


def test_noninteger_q_between_neighbours():
    x = fgn(2000, 0.7, 3)
    a = alpha(x, q=2.5)
    assert np.isfinite(a) and abs(a - 0.7) < 0.2


def test_q2_is_rms_of_window_detrended_profile():
    # F_2 from the MF-DFA formula equals the pointwise RMS of all detrended profile points
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal((40, 25)) , axis=0)
    r = _detrend_windows(y, 1)
    from pyhctsa.operations.scaling import _dfa_fluctuation
    assert np.isclose(_dfa_fluctuation(r, 2), np.sqrt(np.mean(r ** 2)))
    var = np.mean(r ** 2, axis=0)
    assert np.isclose(_dfa_fluctuation(r, 4), np.mean(var ** 2) ** 0.25)
    assert np.isclose(_dfa_fluctuation(r, 0), np.exp(0.5 * np.mean(np.log(var))))
    assert np.isclose(_dfa_fluctuation(r, -1.5), np.mean(var ** -0.75) ** (-1 / 1.5))


def test_mean_offset_invariance():
    x = fgn(1500, 0.6, 5)
    for k in (0, 1, 2):
        ref = fluctuation_analysis(x - x.mean(), wtf='dfa', tau_step=50, k=k)
        off = fluctuation_analysis(x + 1000.0, wtf='dfa', tau_step=50, k=k)
        for key in ('alpha', 'linfitint', 'r1_alpha', 'r2_alpha'):
            assert np.isclose(ref[key], off[key], rtol=1e-8, atol=1e-8), (k, key)
    # with a lag the decimated series has its own mean
    a = fluctuation_analysis(x, wtf='dfa', tau_step=50, k=1, lag=2)['alpha']
    b = fluctuation_analysis(x + 3.0, wtf='dfa', tau_step=50, k=1, lag=2)['alpha']
    assert np.isclose(a, b, rtol=1e-8)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("tau", [5, 17, 100])
def test_vectorised_detrend_equals_polyfit_loop(k, tau):
    rng = np.random.default_rng(tau * 10 + k)
    y = np.cumsum(rng.standard_normal((tau, 30)), axis=0)
    ref = np.empty_like(y)
    tt = np.arange(tau)
    for j in range(y.shape[1]):
        ref[:, j] = y[:, j] - np.polyval(np.polyfit(tt, y[:, j], k), tt)
    np.testing.assert_allclose(_detrend_windows(y, k), ref, rtol=0, atol=1e-12)


def test_constant_series_does_not_raise():
    out = fluctuation_analysis(np.ones(500), wtf='dfa', tau_step=50, q=-2)
    assert isinstance(out, dict)
