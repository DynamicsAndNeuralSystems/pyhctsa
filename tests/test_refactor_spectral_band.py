"""Pinned outputs of the operations touched by the spectral analytic-band refactor.

The analytic signal, usable-bin selection, dominant-band selection and max_n cropping of
spectral.py were factored into module-private helpers, _padded_nfft moved to utils (and is used
by medical.hrv_classic and nonlinearity._period_normalized_tau), and nonlinearity's inline max_n
crops now call _check_max_n. None of this may change an output: the values below were generated
from the main checkout (0fa9013) before the refactor and are compared at rtol 1e-12.
"""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.operations import medical as md
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import spectral as sp


def _series():
    rng = np.random.default_rng(7)
    n = 1200
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    sin = np.sin(2 * np.pi * np.arange(n + 1) / 23.7) + 0.4 * rng.standard_normal(n + 1)  # odd N
    return {'ar1': ar, 'sin_odd': sin}


SERIES = _series()

CALLS = {
    'phase_amp_coupling': (sp.phase_amp_coupling, {}),
    'phase_amp_coupling_max_n': (sp.phase_amp_coupling, {'max_n': 700, 'n_bands': 4}),
    'envelope_stats': (sp.envelope_stats, {}),
    'phase_fluctuation_scaling': (sp.phase_fluctuation_scaling, {'max_n': 900}),
    'bicoherence': (sp.bicoherence, {'max_n': 800}),
    'spectral_summaries_welch': (sp.spectral_summaries, {'psd_meth': 'welch'}),
    'hrv_classic': (md.hrv_classic, {}),
    'zero_one_test': (nl.zero_one_test, {'max_n': 600}),
    'nlpe': (nl.nlpe, {'max_n': 500}),
    'embed_cluster': (nl.embed_cluster, {'max_n': 300}),
    'embed_kernel_pca': (nl.embed_kernel_pca, {'max_n': 300}),
}

PINNED = {
    'phase_amp_coupling': {
        'ar1': {
            'maxMI': 0.0019463062346649022,
            'entropyMI': 0.928498802527566,
        },
        'sin_odd': {
            'maxMI': 0.0008695159348339754,
            'entropyMI': 0.9771054167147875,
        },
    },
    'phase_amp_coupling_max_n': {
        'ar1': {
            'maxMI': 0.0033204237143018823,
            'entropyMI': 0.9010631302983428,
        },
        'sin_odd': {
            'maxMI': 0.001199950034117532,
            'entropyMI': 0.9666245704500985,
        },
    },
    'envelope_stats': {
        'ar1': {
            'full_cv': 0.5246614478475472,
            'full_kurt': 3.072060821424426,
            'dom_cv': 0.5488229077973847,
            'dom_kurt': 2.582744449508469,
            'dom_ifspread': 0.006855560811158193,
        },
        'sin_odd': {
            'full_cv': 0.35638175239959974,
            'full_kurt': 2.899025885040078,
            'dom_cv': 0.08218266644905439,
            'dom_kurt': 5.313460719412068,
            'dom_ifspread': 0.00026455008098093616,
        },
    },
    'phase_fluctuation_scaling': {
        'ar1': {
            'meanFreq': 0.027943646441605623,
            'slope_short': 1.0057981579291562,
            'slope_long': 0.8058075799704743,
            'slope_diff': 0.19999057795868191,
        },
        'sin_odd': {
            'meanFreq': 0.04220170257008161,
            'slope_short': 0.991060325269172,
            'slope_long': 0.6102606944094591,
            'slope_diff': 0.3807996308597129,
        },
    },
    'bicoherence': {
        'ar1': {
            'meanBic': 0.045186423892335216,
            'maxBic': 0.23967121389081927,
            'skewBic': 1.5876372238264476,
            'meanBicDiag': 0.031573921076840955,
            'threshRatio': 0.995615912796739,
        },
        'sin_odd': {
            'meanBic': 0.04296040859262361,
            'maxBic': 0.4086824169286797,
            'skewBic': 2.83244350528225,
            'meanBicDiag': 0.023280156353465347,
            'threshRatio': 0.9909822547782275,
        },
    },
    'spectral_summaries_welch': {
        'ar1': {
            'maxS': 13.643921523814466,
            'logstd': 1.389451831981044,
            'fpoly2_r2': 0.8239741667739016,
            'linfitsemilog_all_sigma': 0.6157699250328597,
            'ncross_log_f50': 9.0,
        },
        'sin_odd': {
            'maxS': 14.707338053612684,
            'logstd': 0.8170941202649162,
            'fpoly2_r2': 0.654978995748215,
            'linfitsemilog_all_sigma': 0.4304900808864068,
            'ncross_log_f50': 2.0,
        },
    },
    'hrv_classic': {
        'ar1': {
            'pnnrel025': 0.8015012510425354,
            'pnnrel1': 0.3219349457881568,
            'lfhf': 0.5382233231864144,
            'hf': 33.36432224649133,
            'SD2': 2300.3659571954777,
        },
        'sin_odd': {
            'pnnrel025': 0.7908333333333334,
            'pnnrel1': 0.33166666666666667,
            'lfhf': 0.013843261662781424,
            'hf': 79.10401558546683,
            'SD2': 1094.20628317549,
        },
    },
    'zero_one_test': {
        'ar1': {
            'K': 0.9970494944460515,
            'Kstd': 0.0061092897482743925,
            'D': 0.5028596145899138,
            'Dstd': 0.7258279863282748,
        },
        'sin_odd': {
            'K': 0.990164685946041,
            'Kstd': 0.09299999307238939,
            'D': 0.16321255656806943,
            'Dstd': 0.04485092426252303,
        },
    },
    'nlpe': {
        'ar1': {
            'msqerr': 1.9221495919908007,
            'stde': 1.386604430175024,
            'ac3': 0.009024102742685964,
            'swm': 0.12244984454334751,
            'minsbc': 0.013230449339834749,
        },
        'sin_odd': {
            'msqerr': 0.593816823224132,
            'stde': 0.7713595763787796,
            'ac3': 0.04447896565328907,
            'swm': 0.11574162331978662,
            'minsbc': 0.020145685381460332,
        },
    },
    'embed_cluster': {
        'ar1': {
            'sep_pp': 0.6995662865272382,
            'sep_mahal': 0.9324923307820927,
            'valley': 0.0,
            'nongauss': 0.14591679817578074,
        },
        'sin_odd': {
            'sep_pp': 0.7277188301930316,
            'sep_mahal': 1.3099115233799308,
            'valley': 0.06815496129737497,
            'nongauss': 0.16254122461134957,
        },
    },
    'embed_kernel_pca': {
        'ar1': {
            'perc_1': 0.20742145426063172,
            'min': 0.15750470315035303,
            'nto80': 8.0,
            'fb001': 15.0,
            'std_ratio': 0.6067547691577171,
        },
        'sin_odd': {
            'perc_1': 0.31475794474870256,
            'min': 0.10179135987260728,
            'nto80': 6.0,
            'fb001': 12.0,
            'std_ratio': 0.44042472812291966,
        },
    },
    '_period_normalized_tau': {
        'ar1': 1.0,
        'sin_odd': 5.0,
    },
}


@pytest.mark.parametrize('op', list(CALLS))
@pytest.mark.parametrize('name', list(SERIES))
def test_pinned_outputs(op, name):
    f, kwargs = CALLS[op]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = f(SERIES[name], **kwargs)
    for key, expected in PINNED[op][name].items():
        np.testing.assert_allclose(out[key], expected, rtol=1e-12, atol=0, err_msg=f'{op}.{key}')


@pytest.mark.parametrize('name', list(SERIES))
def test_period_normalized_tau(name):
    assert nl._period_normalized_tau(SERIES[name]) == PINNED['_period_normalized_tau'][name]


def test_crop_messages(caplog):
    y = SERIES['ar1']
    with pytest.warns(UserWarning, match=r'^Time series \(1200 samples\) exceeds max_n = 700; '
                                         r'analyzing the first 700 samples$'):
        sp.phase_amp_coupling(y, max_n=700)
    with caplog.at_level(logging.INFO, logger='pyhctsa'):
        nl.nlpe(y, max_n=500)
        nl.zero_one_test(y, max_n=600)
        nl.embed_kernel_pca(y, max_n=300)
    messages = [r.getMessage() for r in caplog.records]
    assert "Michael Small's nlpe code is only being evaluated on the first 500 (/1200) samples." in messages
    assert 'Time series (1200 samples) exceeds maxN=600; analyzing the first 600 samples.' in messages
    assert any(m.startswith('Cropping to the first 300 of ') and m.endswith('for kernel PCA (memory/time '
               'cap, not a convergence point)') for m in messages)
    with pytest.raises(ValueError, match="maxN must be an integer or 'full'"):
        nl.zero_one_test(y, max_n='half')
    with pytest.raises(ValueError, match="max_n must be an integer or 'full', got 'half'"):
        nl.embed_cluster(y, max_n='half')


def test_spectral_helpers():
    assert list(sp._usable_bins(8)) == [1, 2, 3]
    assert list(sp._usable_bins(9)) == [1, 2, 3, 4]
    assert sp._padded_nfft(257) == 512 and sp._MIN_NFFT == 256
