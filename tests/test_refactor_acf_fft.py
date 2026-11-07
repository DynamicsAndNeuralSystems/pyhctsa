"""Pinned outputs of the FFT autocovariance sites (duplication audit theme: one ``_autocov_fft``).

``utils._autocov_fft`` now computes the autocovariance behind ``utils._acf_fft`` (formerly
``correlation._acf_fft``), ``utils._acf_fourier``, ``changepoint._mean_square_inflation``
(formerly ``changepoint._variance_inflation``) and ``stationarity._segment_acf``. The expected
values below were generated from the main checkout before the refactor (commit 0fa9013), on
seeded series, and are checked at rtol 1e-12.
"""
import numpy as np
import pytest

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
                                   rtol=1e-12, atol=0, err_msg=f'{name} {key}')


def test_autocov_fft_rows_and_padding():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((4, 37))
    direct = np.array([[np.sum((r[:37 - k] - r.mean()) * (r[k:] - r.mean())) for k in range(37)] for r in Z])
    for pad in ('fast', '2n', 'pow2'):
        np.testing.assert_allclose(utils._autocov_fft(Z, pad=pad), direct, rtol=0, atol=1e-12)
        np.testing.assert_allclose(utils._autocov_fft(Z[1], 5, pad=pad), direct[1, :5], rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        utils._autocov_fft(Z, pad='circular')
