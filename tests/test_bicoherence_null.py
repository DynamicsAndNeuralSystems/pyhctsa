"""The null-corrected summaries of bicoherence: the Gaussian null of the squared bicoherence is Beta(1, K_eff - 1),
so the corrected outputs are near 0 for linear Gaussian processes at any length and clearly positive under quadratic
phase coupling."""
import numpy as np
import pytest
import scipy.signal

from pyhctsa.operations import spectral as sp

L, STEP = 64, 32


def _num_seg(N):
    return (N - L) // STEP + 1


def _ar1(phi, N, seed):
    e = np.random.default_rng(seed).standard_normal(N + 200)
    return scipy.signal.lfilter([1.0], [1.0, -phi], e)[200:]


def _iid(N, seed):
    return np.random.default_rng(seed).standard_normal(N)


def _coupled(N, seed):
    # y = g + 0.4 (g^2 - 1) with g a unit-variance AR(2) oscillator (quadratic nonlinearity)
    e = np.random.default_rng(seed).standard_normal(N + 300)
    g = scipy.signal.lfilter([1.0], [1.0, -2 * 0.9 * np.cos(2 * np.pi / 12), 0.81], e)[300:]
    g = g / g.std()
    return g + 0.4 * (g ** 2 - 1)


def test_k_eff_matches_simulated_null():
    win = scipy.signal.windows.hamming(L, sym=True)
    N = 1000
    K = _num_seg(N)
    k_eff = sp._bicoherence_k_eff(win, STEP, K)
    assert 1.015 < K / k_eff < 1.03
    half_n = L // 2 + 1
    ii, jj = np.meshgrid(np.arange(half_n), np.arange(half_n), indexing='ij')
    mask = (ii + jj <= half_n - 1) & (jj >= ii) & (ii >= 2) & (jj > ii) & (ii + jj < half_n - 1)  # interior bins
    pi, pj = ii[mask], jj[mask]
    means = [np.mean(sp._bicoherence_grid(_iid(N, s), STEP, K, L, half_n, win, pi, pj)) for s in range(200)]
    assert np.mean(means) * k_eff == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("make", [_iid, lambda N, s: _ar1(0.7, N, s)])
def test_mean_bic_near_zero_and_flat_in_n(make):
    med = {}
    for N in (400, 4000):
        med[N] = np.median([sp.bicoherence(make(N, s))['meanBic'] for s in range(20)])
        assert abs(med[N]) < 0.006
    assert abs(med[400] - med[4000]) < 0.006
    # the uncorrected mean would be about 1/K_eff
    assert 1 / _num_seg(400) > 0.05


@pytest.mark.parametrize("N", [400, 1000, 4000])
def test_prop_sig_and_max_bic_under_null(N):
    outs = [sp.bicoherence(_iid(N, 100 + s)) for s in range(20)]
    assert np.mean([o['propSig'] for o in outs]) == pytest.approx(0.05, abs=0.02)
    assert abs(np.mean([o['maxBic'] for o in outs])) < 0.12
    assert abs(np.median([o['maxBic'] for o in outs])) < 0.1


@pytest.mark.parametrize("N", [400, 1000, 4000])
def test_quadratic_coupling_is_positive_at_every_n(N):
    outs = [sp.bicoherence(_coupled(N, 200 + s)) for s in range(10)]
    assert np.median([o['meanBic'] for o in outs]) > 0.04
    assert np.median([o['maxBic'] for o in outs]) > 0.3
    assert np.median([o['propSig'] for o in outs]) > 0.2


def test_outputs_and_determinism():
    x = _ar1(0.5, 1000, 1)
    out = sp.bicoherence(x)
    assert set(out) == {'meanBic', 'maxBic', 'meanBicDiag', 'propSig', 'entropy'}
    for k in ('stdBic', 'skewBic', 'threshRatio'):
        assert k not in out
    assert sp.bicoherence(x) == out


def test_too_short_is_nan():
    with pytest.warns(UserWarning):
        out = sp.bicoherence(_iid(250, 0))
    assert set(out) == {'meanBic', 'maxBic', 'meanBicDiag', 'propSig', 'entropy'}
    assert all(np.isnan(v) for v in out.values())
