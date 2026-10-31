"""Tests for the guards on the sampling-rate normalisation (shared with hctsa's BF_PreProcess
and BF_GetTau):

- ``pre_process(y, 'decimate_ac1e')`` gives NaN if fewer than 50 points would remain;
- ``pre_process(y, 'rescale_tau')`` (used by ``multi_scale_entropy``) coarse-grains at the
  'ac1e' delay, not at the first zero crossing of the ACF;
- ``get_tau(y, 'mi')`` takes the first minimum of the smoothed AMI only if it is deeper than
  twice the AMI estimator noise, so noise on a flat AMI curve no longer gives ``tau = 1``.
"""
import numpy as np
import pytest

import pyhctsa.ksg
import pyhctsa.utils as utils
from pyhctsa.operations.entropy import multi_scale_entropy
from pyhctsa.utils import (_DECIMATE_MIN_POINTS, _ami_first_min, _tau_ac1e, _tau_mi,
                           _zscore_or_zeros, decimate_ac1e, get_tau, make_buffer, pre_process,
                           z_score)

T = np.arange(1000.0)


def _noisy_random_walk(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    return z_score(np.cumsum(rng.standard_normal(n)) * 0.1 + rng.standard_normal(n))


def _ar1(phi, seed=0, n=1000):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return z_score(x)


# ------------------------------------------------------------------------------------------
# A. Decimation length floor
# ------------------------------------------------------------------------------------------
def test_decimation_floor_constant():
    assert _DECIMATE_MIN_POINTS == 50


def test_decimation_plain_subsample_when_long_enough():
    y = _ar1(0.9)
    tau = get_tau(y, 'ac1e')
    assert tau > 1 and -(-y.size // tau) >= _DECIMATE_MIN_POINTS
    np.testing.assert_array_equal(pre_process(y, 'decimate_ac1e'), y[::tau])
    np.testing.assert_allclose(decimate_ac1e(y), _zscore_or_zeros(y[::tau]))


def test_decimation_nan_below_floor():
    # sine of period 200: ac1e = 38, so 1000 samples decimate to 27 points
    y = z_score(np.sin(2 * np.pi * T / 200))
    assert get_tau(y, 'ac1e') == 38
    assert np.isnan(pre_process(y, 'decimate_ac1e'))
    assert np.isnan(decimate_ac1e(y))
    assert np.isnan(decimate_ac1e(y, rezscore=False))


@pytest.mark.parametrize('n, n_dec', [(245, 49), (246, 50), (250, 50), (251, 51)])
def test_decimation_floor_exact_boundary(n, n_dec):
    # sine of period 30: ac1e = 5 at these lengths, so ceil(n / 5) points remain
    y = z_score(np.sin(2 * np.pi * T[:n] / 30))
    assert get_tau(y, 'ac1e') == 5
    d = pre_process(y, 'decimate_ac1e')
    if n_dec < _DECIMATE_MIN_POINTS:
        assert np.isscalar(d) and np.isnan(d)
    else:
        np.testing.assert_array_equal(d, y[::5])
        assert d.size == n_dec


@pytest.mark.parametrize('n', [49, 50])
def test_decimation_floor_iterated_map(n):
    # tau = 1 (white noise): the series is returned unchanged if it has at least 50 points
    y = z_score(np.random.default_rng(1).standard_normal(n))
    assert get_tau(y, 'ac1e') == 1
    d = pre_process(y, 'decimate_ac1e')
    if n < _DECIMATE_MIN_POINTS:
        assert np.isnan(d)
    else:
        np.testing.assert_array_equal(d, y)


# ------------------------------------------------------------------------------------------
# B. rescale_tau coarse-grains at the 'ac1e' delay
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('y', [_ar1(0.9), _ar1(0.97, seed=3), _noisy_random_walk(),
                               z_score(np.sin(2 * np.pi * T / 40))],
                         ids=['ar09', 'ar097', 'rw_noise', 'sine40'])
def test_rescale_tau_uses_ac1e(y):
    tau = get_tau(y, 'ac1e')
    expected = np.mean(make_buffer(y, tau), axis=1)
    np.testing.assert_array_equal(pre_process(y, 'rescale_tau'), expected)
    assert len(pre_process(y, 'rescale_tau')) == y.size // tau


def test_rescale_tau_not_the_zero_crossing():
    # monotonically decaying ACF: the zero crossing is several times the 1/e time
    y = _ar1(0.9)
    tau_ac, tau_ac1e = get_tau(y, 'ac'), get_tau(y, 'ac1e')
    assert tau_ac > 2 * tau_ac1e
    assert len(pre_process(y, 'rescale_tau')) == y.size // tau_ac1e != y.size // tau_ac


def test_rescale_tau_nan_when_ac1e_undefined():
    assert np.isnan(pre_process(np.ones(300), 'rescale_tau'))


def test_multi_scale_entropy_rescale_tau_matches_ac1e_coarse_graining():
    y = _ar1(0.9, seed=2, n=3000)
    tau = get_tau(y, 'ac1e')
    assert tau > 1 and tau != get_tau(y, 'ac')
    y_cg = _zscore_or_zeros(np.mean(make_buffer(y, tau), axis=1))
    a = multi_scale_entropy(y, range(1, 11), 2, 0.15, pre_process_how='rescale_tau')
    b = multi_scale_entropy(y_cg, range(1, 11), 2, 0.15)
    assert isinstance(a, dict) and a.keys() == b.keys()
    for k in a:
        np.testing.assert_equal(a[k], b[k])
    assert np.isfinite(a['meanSampEn'])


# ------------------------------------------------------------------------------------------
# C. Noise-gated AMI minimum for get_tau(y, 'mi')
# ------------------------------------------------------------------------------------------
def test_mi_noisy_random_walk_not_one():
    # the smoothed AMI of a noisy random walk is flat and noisy; it happens to rise from lag 1
    # to lag 2, which the old rule took as a minimum at lag 1 (tau = 1, against ac1e = 57)
    y = _noisy_random_walk(seed=0)
    ac1e = get_tau(y, 'ac1e')
    assert ac1e > 3
    tau = get_tau(y, 'mi')
    assert tau != 1
    assert tau == ac1e  # no minimum deeper than the noise: capped by ac1e


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_mi_noisy_random_walks_never_one(seed):
    y = _noisy_random_walk(seed=seed)
    ac1e = get_tau(y, 'ac1e')
    if ac1e > 3:
        assert get_tau(y, 'mi') > 3


def test_mi_genuine_minimum_below_ac1e():
    # a small fast oscillation (period 20) on a slow one (period 300), plus noise: the ACF stays
    # above 1/e to lag ~47, but the AMI has a clear minimum near half the fast period
    rng = np.random.default_rng(1)
    y = z_score(0.6 * np.sin(2 * np.pi * T / 20) + 1.5 * np.sin(2 * np.pi * T / 300)
                + 0.2 * rng.standard_normal(1000))
    ac1e = get_tau(y, 'ac1e')
    tau = get_tau(y, 'mi')
    assert ac1e > 30
    assert 8 <= tau <= 15 < ac1e


def test_ami_first_min_gate():
    lags = np.arange(1, 41)
    rng = np.random.default_rng(0)
    # a clear minimum at lag 10 on a decaying curve with small noise
    clear = 0.5 + 0.3 * np.cos(2 * np.pi * lags / 20) + 0.005 * rng.standard_normal(40)
    tau_clear = _ami_first_min(clear)
    assert 8 <= tau_clear <= 12
    # a flat curve of noise: no minimum is deeper than twice the noise
    flat = 0.05 + 0.01 * rng.standard_normal(40)
    assert np.isnan(_ami_first_min(flat))
    # a monotonically decaying curve plus noise: no minimum either
    decay = np.exp(-lags / 15) + 0.005 * rng.standard_normal(40)
    assert np.isnan(_ami_first_min(decay))


def test_ami_first_min_never_lag_one():
    # rises from lag 1 to lag 2 (the old rule's lag-1 minimum), then decays
    ami = np.r_[0.50, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    tau = _ami_first_min(ami)
    assert np.isnan(tau) or tau >= 2
    assert np.isnan(_ami_first_min(np.r_[0.5, 0.6]))  # two lags: nothing to search


def test_ami_first_min_short_curve_delta_zero():
    # with fewer than 4 lags there is no noise estimate (delta = 0): any strict minimum of the
    # smoothed curve at lag 2 qualifies, however shallow
    assert _ami_first_min(np.r_[1.0, 1.2, 0.9]) == 2   # smoothed: 1.1, 1.033, 1.05
    assert _ami_first_min(np.r_[1.0, 1.0 + 3e-6, 1.0 - 1e-6]) == 2
    assert np.isnan(_ami_first_min(np.r_[1.0, 0.8, 0.6]))  # monotone
    assert np.isnan(_ami_first_min(np.r_[1.0, 1.0, 1.0]))  # no strict minimum


@pytest.mark.parametrize('curve, expected', [((1.0, 1.2, 0.9), 2), ((1.0, 0.8, 0.6), 3)])
def test_tau_mi_short_series_delta_zero(monkeypatch, curve, expected):
    # 35 samples: the AMI is evaluated at lags 1..3 only (N // 10 = 3), so delta = 0; the
    # Kraskov estimator is replaced by a fixed curve to exercise this path deterministically
    y = z_score(np.sin(2 * np.pi * T[:35] / 20))
    assert _tau_ac1e(y) == 3
    n = y.size

    def fake_ksg_mi(x1, x2, k, alg):
        return curve[n - len(x1) - 1]  # len(x1) = n - lag

    monkeypatch.setattr(pyhctsa.ksg, 'ksg_mi', fake_ksg_mi)
    assert _tau_mi(y) == expected   # a minimum at lag 2, or else capped by ac1e = 3


def test_mi_unchanged_early_returns():
    # ac1e = 1 (iterated map / white noise) returns 1 before any AMI is computed
    y = z_score(np.random.default_rng(5).standard_normal(1000))
    assert get_tau(y, 'ac1e') == 1 and get_tau(y, 'mi') == 1
    # a constant series has no delay
    assert np.isnan(get_tau(np.ones(300), 'mi'))
    assert utils._TAU_RULES == ('ac', 'ac1e', 'mi', 'mi-gaussian')
