"""Pinned outputs of the FFT autocovariance sites (duplication audit theme: one ``_autocov_fft``).

``utils._autocov_fft`` now computes the autocovariance behind ``utils._acf_fft`` (formerly
``correlation._acf_fft``), ``utils._acf_fourier``, ``changepoint._mean_square_inflation``
(formerly ``changepoint._variance_inflation``) and ``stationarity._segment_acf``. The expected
values below were generated from the main checkout before the refactor (commit 0fa9013), on
seeded series, on one machine (arm64 macOS).

Two kinds of check:

1. Same-runner reference (the equivalence check): verbatim copies of the three pre-refactor FFT
   blocks, one per padding variant (``'fast'``, ``'2n'``, ``'pow2'``), are compared with the code
   that now calls ``_autocov_fft`` at rtol 1e-12, atol 0, on whatever machine runs the test.
2. Operation-level pins: the main-checkout values, with tolerances set by the kind of computation
   (other BLAS/FFT builds reproduce them only to about 1e-12; see TOLERANCES).
"""
import numpy as np
import pytest
from scipy import fft as sp_fft

from pyhctsa import utils
from pyhctsa.operations import changepoint as cp
from pyhctsa.operations import correlation as co
from pyhctsa.operations import stationarity as st
from pyhctsa.utils import z_score


def _series():
    rng = np.random.default_rng(20261107)
    w = rng.standard_normal(1000)
    e = rng.standard_normal(1000)
    a = np.zeros(1000)
    for t in range(1, 1000):
        a[t] = 0.8 * a[t - 1] + e[t]
    s = np.sin(2 * np.pi * np.arange(500) / 37) + 0.3 * rng.standard_normal(500)
    v = np.concatenate([rng.standard_normal(300), 3 * rng.standard_normal(300), rng.standard_normal(300)])
    return {'white': w, 'ar1': a, 'sine': s, 'varstep': v}


EXPECTED = {'white': {'acf_fft': [0.03831410182037328,
                       0.008887982527080559,
                       0.006310353612425469,
                       -0.0007382608351774559,
                       0.040948507428543966],
           'acf_fourier': [0.03831410182037328, 0.01985586398915666, -0.024719762497355118],
           'tau_ac': 6,
           'tau_ac1e': 1,
           'autocorr': [0.03831410182037328,
                        0.008887982527080559,
                        0.006310353612425469,
                        -0.0007382608351774559],
           'resid_ac_scan': {'max_z': 2.9086479449860425,
                             'peak_lag': 172.0,
                             'lb_z': -1.21089704063216,
                             'sq_max_z': 3.7200173647957486,
                             'sq_lb_z': -0.3902755636410686},
           'mean_square_inflation': 1.149029062695968,
           'wavelet_var_chg': 0.0,
           'segment_inflation': 1.1808770549471432,
           'local_dist_par': {'ex_meandiv': 0.015640099182240713,
                              'ex_meandiv_ess': 0.01380020228825371,
                              'stddiv': 0.029453352950046283},
           'local_dist_each': {'ex_meandiv': 0.03440434779273069,
                               'ex_meandiv_ess': 0.032229550769967905,
                               'stddiv': 0.017075797907139107}},
 'ar1': {'acf_fft': [0.7983829985088108,
                     0.6180333202004858,
                     0.2854569792336218,
                     0.05195081473710569,
                     -0.0401110582197317],
         'acf_fourier': [0.7983829985088108, 0.47993408956877753, -0.031328349226849114],
         'tau_ac': 14,
         'tau_ac1e': 3,
         'autocorr': [0.7983829985088108,
                      0.6180333202004858,
                      0.2854569792336218,
                      0.05195081473710569],
         'resid_ac_scan': {'max_z': 2.839229431010375,
                           'peak_lag': 179.0,
                           'lb_z': -0.6747332741755772,
                           'sq_max_z': 2.5372425408759547,
                           'sq_lb_z': -1.322860703396296},
         'mean_square_inflation': 4.688553628727757,
         'wavelet_var_chg': 0.0,
         'segment_inflation': 5.116932490154418,
         'local_dist_par': {'ex_meandiv': 0.018640099182240702,
                            'ex_meandiv_ess': -0.007689631188247878,
                            'stddiv': 0.010839741694339407},
         'local_dist_each': {'ex_meandiv': 0.019389332777715693,
                             'ex_meandiv_ess': -0.015314233880825825,
                             'stddiv': 0.03632682144469281}},
 'sine': {'acf_fft': [0.8235469552302292,
                      0.7934962375569105,
                      0.5710400136155915,
                      -0.08379023148475152,
                      -0.44346675847008216],
          'acf_fourier': [0.8235469552302292, 0.7386440797762435, -0.7867314928927782],
          'tau_ac': 10,
          'tau_ac1e': 6,
          'autocorr': [0.8235469552302292,
                       0.7934962375569105,
                       0.5710400136155915,
                       -0.08379023148475152],
          'resid_ac_scan': {'max_z': 3.1403473968474267,
                            'peak_lag': 12.0,
                            'lb_z': 2.6346931661245394,
                            'sq_max_z': 2.637185403698516,
                            'sq_lb_z': 0.8144186608937568},
          'mean_square_inflation': 38.241254168676505,
          'wavelet_var_chg': 0.0,
          'segment_inflation': 2.5603367558523913,
          'local_dist_par': {'ex_meandiv': -0.010104847827020882,
                             'ex_meandiv_ess': -0.04377426256288505,
                             'stddiv': 0.015165750888103104},
          'local_dist_each': {'ex_meandiv': -0.01080325940951353,
                              'ex_meandiv_ess': -0.05195052271381058,
                              'stddiv': 0.020868082018902133}},
 'varstep': {'acf_fft': [0.0090632917706866,
                         -0.03960284430110232,
                         -0.010418686139900688,
                         -0.09527001749781303,
                         -0.023428503243205313],
             'acf_fourier': [0.0090632917706866, -0.03210529036856604, 0.05853941765643475],
             'tau_ac': 2,
             'tau_ac1e': 1,
             'autocorr': [0.0090632917706866,
                          -0.03960284430110232,
                          -0.010418686139900688,
                          -0.09527001749781303],
             'resid_ac_scan': {'max_z': 3.378609138227483,
                               'peak_lag': 17.0,
                               'lb_z': 4.74809580773264,
                               'sq_max_z': 8.813660910572011,
                               'sq_lb_z': 161.75775642908206},
             'mean_square_inflation': 3.1189433415732095,
             'wavelet_var_chg': 2.0,
             'segment_inflation': 0.8152474114394279,
             'local_dist_par': {'ex_meandiv': 0.08060271374049471,
                                'ex_meandiv_ess': 0.07462981535035634,
                                'stddiv': 0.12846433909748295},
             'local_dist_each': {'ex_meandiv': 0.2589803662449287,
                                 'ex_meandiv_ess': 0.25922570093570296,
                                 'stddiv': 0.22258581366395402}}}


# ------------------------------------------------------------------------------
# 1. Same-runner reference: verbatim copies of the pre-refactor FFT blocks
# ------------------------------------------------------------------------------

def _ref_acf_fft(y):
    # pad 'fast': verbatim body of correlation._acf_fft, main 0fa9013, correlation.py lines
    # 2948-2958 (scipy.fft, next_fast_len(2N - 1)).
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    if N == 0:
        return np.zeros(0)
    if not np.all(np.isfinite(y)) or np.ptp(y) == 0:
        return np.full(N, np.nan)
    z = y - y.mean()
    nfft = sp_fft.next_fast_len(2 * N - 1, real=True)
    f = sp_fft.rfft(z, nfft)
    a = sp_fft.irfft(f * np.conj(f), nfft)[:N]
    return np.real(a / a[0])


def _ref_variance_inflation(x, max_lag=200):
    # pad '2n': verbatim body of changepoint._variance_inflation (now _mean_square_inflation),
    # main 0fa9013, changepoint.py lines 290-303 (numpy.fft at exactly 2N).
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    xc = x - np.mean(x)
    v = np.mean(xc ** 2)
    if N < 8 or not v > 0:
        return 1.0
    L = int(min(max_lag, N // 4))
    f = np.fft.rfft(xc, 2 * N)
    a = np.fft.irfft(f * np.conj(f))[:L + 1]
    rho = a[1:] / a[0]
    h = np.arange(1, L + 1)
    tau = 1.0 + 2.0 * np.sum((1.0 - h / (L + 1.0)) * rho ** 2)
    kurt = np.mean(xc ** 4) / v ** 2
    return float(max(tau, 1.0) * max(1.0, (kurt - 1.0) / 2.0))


def _ref_segment_acf(z, num_segs, n):
    # pad 'pow2': verbatim body of stationarity._segment_acf, main 0fa9013, stationarity.py
    # lines 217-221 (numpy.fft with abs(F)**2 at the power of two >= 2n).
    Z = z[:num_segs * n].reshape(num_segs, n)
    Z = Z - Z.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))  # zero-padding to >= 2n - 1: linear autocovariance
    acov = np.fft.irfft(np.abs(np.fft.rfft(Z, nfft, axis=1)) ** 2, nfft, axis=1)[:, :n].sum(axis=0)
    return acov / acov[0] if acov[0] > 0 else np.full(n, np.nan)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_pad_fast_matches_reference(name):
    y = _series()[name]
    # Same scipy.fft calls in the same order: identical, lag by lag.
    np.testing.assert_allclose(utils._acf_fft(y), _ref_acf_fft(y), rtol=1e-12, atol=0)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_pad_2n_matches_reference(name):
    z = z_score(_series()[name])
    for max_lag in (200, 20):
        assert np.isclose(cp._mean_square_inflation(z, max_lag), _ref_variance_inflation(z, max_lag),
                          rtol=1e-12, atol=0)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_pad_pow2_matches_reference(name, monkeypatch):
    z = z_score(_series()[name])
    for num_segs in (3, 5):
        n = z.size // num_segs
        r_new, r_ref = st._segment_acf(z, num_segs, n), _ref_segment_acf(z, num_segs, n)
        # The ACF vector itself: numpy.fft with abs(F)**2 (old) and scipy.fft with F * conj(F)
        # (new) round differently, by up to ~6e-17 on near-zero lags, so lags are compared with
        # atol = 1e-12 * r[0] (r[0] = 1, the scale of an autocorrelation).
        np.testing.assert_allclose(r_new, r_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(st._variance_inflation(r_new, n), st._variance_inflation(r_ref, n),
                                   rtol=1e-12, atol=0)
    # The feature that consumes it: local_distributions with the old block patched in.
    new = st.local_distributions(z, 5, 'par')
    monkeypatch.setattr(st, '_segment_acf', _ref_segment_acf)
    ref = st.local_distributions(z, 5, 'par')
    for key in new:
        np.testing.assert_allclose(new[key], ref[key], rtol=1e-12, atol=0, err_msg=key)


# ------------------------------------------------------------------------------
# 2. Operation-level pins (main-checkout values) with per-output tolerances
# ------------------------------------------------------------------------------

# Every pinned output is closed-form arithmetic on FFT autocovariances (ACF values, sums of
# squared ACF, kurtosis, Ljung-Box statistics, segment moments), so all use the closed-form tier,
# rtol 1e-9: BLAS/FFT ordering moved them by at most 1.6e-12 across runners (local_dist_par).
# tau_ac/tau_ac1e, peak_lag and wavelet_var_chg are integers (exact in practice). Outputs whose
# value can be near zero (ACF at long lags, ex_meandiv_ess, lb_z) also get an absolute tolerance
# of 1e-9 times the scale they are computed from: ACF values are bounded by r[0] = 1, and the
# local_distributions and residual_autocorr_scan outputs are differences of O(1) quantities on a
# z-scored series (scale 1), so ATOL = 1e-9 * 1.
RTOL = 1e-9
ATOL = 1e-9 * 1.0


def _outputs(y):
    z = z_score(y)
    n = z.size // 5
    return {
        'acf_fft': utils._acf_fft(y)[[1, 2, 5, 10, 50]].tolist(),
        'acf_fourier': utils._acf_fourier(y)[[1, 3, 20]].tolist(),
        'tau_ac': utils.get_tau(y, 'ac'),
        'tau_ac1e': utils.get_tau(y, 'ac1e'),
        'autocorr': co.autocorr(y, [1, 2, 5, 10], 'Fourier').tolist(),
        'resid_ac_scan': co.residual_autocorr_scan(z),
        'mean_square_inflation': cp._mean_square_inflation(z),
        'wavelet_var_chg': cp.wavelet_var_chg(z, 'db3', 3, 10),
        'segment_inflation': st._variance_inflation(st._segment_acf(z, 5, n), n),
        'local_dist_par': st.local_distributions(z, 5, 'par'),
        'local_dist_each': st.local_distributions(z, 3, 'each'),
    }


@pytest.mark.parametrize('name', list(EXPECTED))
def test_acf_fft_sites_pinned(name):
    got = _outputs(_series()[name])
    for key, want in EXPECTED[name].items():
        g = got[key]
        if isinstance(want, dict):
            assert set(g) == set(want), key
            g, want = [g[k] for k in want], list(want.values())
        np.testing.assert_allclose(np.asarray(g, dtype=float), np.asarray(want, dtype=float),
                                   rtol=RTOL, atol=ATOL, err_msg=f'{name} {key}')


def test_autocov_fft_rows_and_padding():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((4, 37))
    direct = np.array([[np.sum((r[:37 - k] - r.mean()) * (r[k:] - r.mean())) for k in range(37)] for r in Z])
    for pad in ('fast', '2n', 'pow2'):
        np.testing.assert_allclose(utils._autocov_fft(Z, pad=pad), direct, rtol=0, atol=1e-12)
        np.testing.assert_allclose(utils._autocov_fft(Z[1], 5, pad=pad), direct[1, :5], rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        utils._autocov_fft(Z, pad='circular')
