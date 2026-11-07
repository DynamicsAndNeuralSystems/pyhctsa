"""Regression pins for the single-file duplication cleanups (audit theme #21).

The refactor (``symbolic._matrix_measures`` shared by ``transition_matrix`` and
``_transition_measures``; ``pre_process._transform`` shared by ``preproc_compare`` and
``preproc_iterate``; the 2- and 5-band loop in ``spectral.spectral_summaries``; the AAFT
Gaussian rank map through ``pre_process._rank_map_gaussian``) is output-preserving.
The values below were generated from the main checkout before the refactor (commit 0fa9013).
"""
import warnings

import numpy as np
import pytest

from pyhctsa.operations import pre_process, spectral, surrogates, symbolic


def _series():
    rng = np.random.default_rng(11)
    e = rng.standard_normal(300)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.8 * x[t - 1] + e[t]
    rng = np.random.default_rng(12)
    t = np.arange(300)
    s = np.sin(2 * np.pi * t / 23.7) + 0.4 * rng.standard_normal(300)
    return {name: (y - y.mean()) / y.std(ddof=1) for name, y in {'ar1': x, 'sine': s}.items()}


CALLS = {
    'tm_q3': lambda z: symbolic.transition_matrix(z, 'quantile', 3, 1),
    'tm_q5_t2': lambda z: symbolic.transition_matrix(z, 'quantile', 5, 2),
    'tpa': lambda z: symbolic.transition_p_alphabet(z),
    'ppc_poly2': lambda z: pre_process.preproc_compare(z, 'poly2'),
    'ppc_spline24': lambda z: pre_process.preproc_compare(z, 'spline24'),
    'ppc_diff1': lambda z: pre_process.preproc_compare(z, 'diff1'),
    'ppc_medianf3': lambda z: pre_process.preproc_compare(z, 'medianf3'),
    'ppc_rav3': lambda z: pre_process.preproc_compare(z, 'rav3'),
    'ppc_resample_2_1': lambda z: pre_process.preproc_compare(z, 'resample_2_1'),
    'ppi_diff': lambda z: pre_process.preproc_iterate(z, 'diff'),
    'ppi_rav': lambda z: pre_process.preproc_iterate(z, 'rav'),
    'ppi_resampledown': lambda z: pre_process.preproc_iterate(z, 'resampledown'),
    'ss_fft': lambda z: spectral.spectral_summaries(z, 'fft'),
    'ss_welch': lambda z: spectral.spectral_summaries(z, 'welch'),
}


def _aaft(z):
    surr = surrogates._make_surrogates(z, 'AAFT', 2, 0)
    return {'s0_0': surr[0, 0], 's1_7': surr[7, 1], 's0_150': surr[150, 0],
            'sum_abs_diff': np.sum(np.abs(np.diff(surr, axis=0)))}


CALLS['aaft'] = _aaft

PINS = {('ar1', 'aaft'): {'s0_0': -0.89053590008085,
                   's0_150': -1.1088554414548837,
                   's1_7': -0.14423793297261175,
                   'sum_abs_diff': 321.0306014398218},
 ('ar1', 'ppc_diff1'): {'olbt_m5': -0.004695540477370447,
                        'statav2': -0.49299785681868713,
                        'swms5_1': -0.8440439017153872},
 ('ar1', 'ppc_medianf3'): {'statav2': 0.07351547861016382, 'swss10_1': 0.09862292132937572},
 ('ar1', 'ppc_poly2'): {'olbt_m2': -0.0001985513910114584,
                        'statav2': -0.8771503581062706,
                        'swss5_1': -0.0006279644207190213},
 ('ar1', 'ppc_rav3'): {'statav2': -0.09330006798937752, 'swms10_1': 0.02508146942766271},
 ('ar1', 'ppc_resample_2_1'): {'olbt_m2': 0.000264033910530901, 'statav2': 0.032182930383142835},
 ('ar1', 'ppc_spline24'): {'kscn_olapint': -0.002499017419227327,
                           'olbt_s5': 0.006316542697736803,
                           'statav2': -0.1566146016910181},
 ('ar1', 'ppi_diff'): {'ol_trend': 0.36514556556377564,
                       'statav5_trend': -2.4620666826341835,
                       'swms5_2_jump': 3.844546185660663},
 ('ar1', 'ppi_rav'): {'normdiff_trend': 3.499183654371031,
                      'statav5_trend': 3.2348014284627133,
                      'xc1_jump': 8.167896116653964},
 ('ar1', 'ppi_resampledown'): {'gauss1_kd_jump': 6.33993099494131, 'statav5_trend': 2.422665148112958},
 ('ar1', 'ss_fft'): {'area_2_1': 0.9235045547140299,
                     'area_5_3': 0.05116707705069407,
                     'logarea_2_2': -5.730415016750034,
                     'logarea_5_5': -2.509223439750405,
                     'logstatav2_m': 0.7790162911148765,
                     'logstatav5_s': 0.08061816344480602,
                     'statav2_s': 0.9103938853924036,
                     'statav5_s': 0.7942432754811674},
 ('ar1', 'ss_welch'): {'area_5_1': 0.8551539699986171,
                       'logstatav2_s': 0.33110471025333804,
                       'logstatav5_m': 1.0161563656327823},
 ('ar1', 'tm_q3'): {'maxeig': 0.3333284863445011,
                    'maxeigcov': 0.023653347728466723,
                    'ondiag': 0.6053511705685619,
                    'stddiag': 0.04733751113837713,
                    'stdeig': 0.14004288737964057,
                    'sumdiagcov': 0.025141404085711198,
                    'symdiff': 0.013377926421404705},
 ('ar1', 'tm_q5_t2'): {'ondiag': 0.3724832214765101,
                       'stdeig': 0.08464095762443101,
                       'sumdiagcov': 0.00428922120625197,
                       'symdiff': 0.14093959731543626,
                       'transEntropy': 1.4193248735350872},
 ('ar1', 'tpa'): {'maxdiagfexp_r2': 0.9475961923812124,
                  'meandiagfexp_b': -0.4807734505758499,
                  'trfexp_b': -0.1590524155360214,
                  'trflin5_adjr2': 0.8994532811083845},
 ('sine', 'aaft'): {'s0_0': -1.1922321143778,
                    's0_150': -0.12874462072887535,
                    's1_7': -0.7746263241939627,
                    'sum_abs_diff': 361.7399708787202},
 ('sine', 'ppc_diff1'): {'olbt_m5': 0.008217161892132382,
                         'statav2': 0.5356773638021818,
                         'swms5_1': -0.7370061540140986},
 ('sine', 'ppc_medianf3'): {'statav2': -0.4447468259603433, 'swss10_1': 0.061660285625295234},
 ('sine', 'ppc_poly2'): {'olbt_m2': -0.0010587479850067247,
                         'statav2': -0.1155379663734372,
                         'swss5_1': -0.013834287927298754},
 ('sine', 'ppc_rav3'): {'statav2': 0.564690775745768, 'swms10_1': -0.013189420284764811},
 ('sine', 'ppc_resample_2_1'): {'olbt_m2': 0.00017318075243662268, 'statav2': -0.6421137789357829},
 ('sine', 'ppc_spline24'): {'kscn_olapint': 0.00012319494167525343,
                            'olbt_s5': 0.0029557139117370058,
                            'statav2': 0.27624550322977176},
 ('sine', 'ppi_diff'): {'ol_trend': -1.2254912098011226,
                        'statav5_trend': -1.858312251256477,
                        'swms5_2_jump': 3.9531763839307597},
 ('sine', 'ppi_rav'): {'normdiff_trend': 3.5364277295250774,
                       'statav5_trend': -2.6339034875828515,
                       'xc1_jump': 9.728958060116113},
 ('sine', 'ppi_resampledown'): {'gauss1_kd_jump': 1.8497235845198898, 'statav5_trend': 2.4842912417144865},
 ('sine', 'ss_fft'): {'area_2_1': 0.8894188195138792,
                      'area_5_3': 0.051195476035843346,
                      'logarea_2_2': -5.205763540479876,
                      'logarea_5_5': -2.047959990460173,
                      'logstatav2_m': 0.24631245111035757,
                      'logstatav5_s': 0.22560492283128442,
                      'statav2_s': 0.9762217926898484,
                      'statav5_s': 0.9758337141926346},
 ('sine', 'ss_welch'): {'area_5_1': 0.8327529569270591,
                        'logstatav2_s': 0.6935671510429072,
                        'logstatav5_m': 0.5840815951008361},
 ('sine', 'tm_q3'): {'maxeig': 0.3333307622665454,
                     'maxeigcov': 0.026620955975199777,
                     'ondiag': 0.6354515050167224,
                     'stddiag': 0.04638270694027808,
                     'stdeig': 0.13172899357963366,
                     'sumdiagcov': 0.029190575795199902,
                     'symdiff': 0.04682274247491642},
 ('sine', 'tm_q5_t2'): {'ondiag': 0.32550335570469796,
                        'stdeig': 0.09597462718751737,
                        'sumdiagcov': 0.004700238727985226,
                        'symdiff': 0.15436241610738255,
                        'transEntropy': 1.3711530330987605},
 ('sine', 'tpa'): {'maxdiagfexp_r2': 0.9842223976724699,
                   'meandiagfexp_b': -0.5458298523934537,
                   'trfexp_b': -0.1904908822424008,
                   'trflin5_adjr2': 0.8386379838134774}}

SERIES = _series()


@pytest.mark.parametrize('key', sorted(PINS), ids=lambda k: f'{k[0]}-{k[1]}')
def test_pinned_outputs(key):
    series_name, call = key
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = CALLS[call](SERIES[series_name])
    for name, expected in PINS[key].items():
        np.testing.assert_allclose(out[name], expected, rtol=1e-12, atol=0, err_msg=f'{call}.{name}')


def test_transition_measures_shared():
    """transition_matrix and _transition_measures report the same shared measures."""
    z = SERIES['ar1']
    for ng in (2, 4, 7):
        yth = symbolic.coarse_grain(z, 'quantile', ng)
        m = symbolic._transition_measures(yth, ng, 1)
        out = symbolic.transition_matrix(z, 'quantile', ng, 1)
        assert [out['ondiag'], out['symdiff'], out['sumdiagcov'], out['stdeig']] == [m[2], m[3], m[4], m[5]]
