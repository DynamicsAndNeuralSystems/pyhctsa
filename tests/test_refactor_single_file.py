"""Regression pins for the single-file duplication cleanups (audit theme #21).

The refactor (``symbolic._matrix_measures`` shared by ``transition_matrix`` and
``_transition_measures``; ``pre_process._transform`` shared by ``preproc_compare`` and
``preproc_iterate``; the 2- and 5-band loop in ``spectral.spectral_summaries``; the AAFT
Gaussian rank map through ``pre_process._rank_map_gaussian``) is output-preserving.

Two checks:

1. Same-runner references (the equivalence checks that matter): each code path the refactor
   replaced is copied verbatim from main 0fa9013 below (``_ref_*``, with the file and lines it
   came from) and must agree with the new code at rtol 1e-12 on whatever machine runs the test.
2. Operation-level pins. The values were generated from the main checkout before the refactor
   (commit 0fa9013, on Ben's Mac); the tolerance is set per operation (see ``_rtol``), since
   other platforms reproduce some of them (the exponential fits above all) only to about 1e-8.
"""
import re
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import lfilter, resample_poly

from pyhctsa.operations import pre_process, spectral, surrogates, symbolic
from pyhctsa.operations.pre_process import _med_filt_1d, _spline_detrend
from pyhctsa.utils import _even_split_points, _sample_std


# --- verbatim copies of the replaced code paths (main 0fa9013) --------------------------------

def _ref_transition_measures(T):
    # copied from pyhctsa/operations/symbolic.py lines 839-857 (body of _transition_measures,
    # after T = _transition_matrix(...)) on main 0fa9013
    out = np.zeros(6)
    #   (i) diagonal elements
    diag_t = np.diag(T)
    out[0] = np.mean(diag_t)
    out[1] = np.max(diag_t)
    out[2] = np.sum(diag_t)  # trace

    #  (ii) measures of symmetry:
    out[3] = np.sum(np.abs(T - T.T))  # sum of differences of individual elements

    # (iii) measures from covariance matrix:
    out[4] = np.trace(np.cov(T, rowvar=False))  # trace of the covariance matrix

    # (iv) measures from eigenvalues of T
    eig_t = np.linalg.eigvals(T)
    out[5] = _sample_std(eig_t)

    return out


def _ref_transition_matrix_shared(T):
    # copied from pyhctsa/operations/symbolic.py lines 755, 759, 771-772, 805-806 (inside
    # transition_matrix) on main 0fa9013: the four outputs now taken from _matrix_measures
    out = {}
    diag_t = np.diag(T)
    out['ondiag'] = np.sum(diag_t)  # trace
    out['symdiff'] = np.sum(np.abs(T - T.T))  # sum of differences of individual elements
    eig_t = np.linalg.eigvals(T)
    out['stdeig'] = _sample_std(eig_t)  # std of eigenvalues
    cov_t = np.cov(T, rowvar=False)
    out['sumdiagcov'] = np.trace(cov_t)  # trace of covariance matrix
    return out


def _ref_preproc_compare_transform(y, detrend_meth):
    # copied from pyhctsa/operations/pre_process.py lines 235-270 (inside preproc_compare) on
    # main 0fa9013, the parameterized branches only (sin, logr and boxcox are unchanged)
    N = len(y)
    if (m := re.fullmatch(r'poly([1-9])', detrend_meth)):
        # 1) Polynomial detrend
        order = int(m.group(1))
        r = np.arange(N, dtype=float)
        y_d = y - np.polynomial.Polynomial.fit(r, y, order)(r)

    elif (m := re.fullmatch(r'spline(\d)(\d)', detrend_meth)):
        # 3) Spline detrend
        y_d = _spline_detrend(y, int(m.group(1)), int(m.group(2)))

    elif (m := re.fullmatch(r'diff(\d)', detrend_meth)):
        # 4) Differencing
        y_d = np.diff(y, n=int(m.group(1)))

    elif (m := re.fullmatch(r'medianf(\d+)', detrend_meth)):
        # 5) Median filter
        y_d = _med_filt_1d(y, int(m.group(1)))

    elif (m := re.fullmatch(r'rav(\d+)', detrend_meth)):
        # 6) Running average
        n = int(m.group(1))
        y_d = lfilter(np.ones(n) / n, [1], y)

    elif (m := re.fullmatch(r'resample_(\d+)_(\d+)', detrend_meth)):
        # 7) Resample
        y_d = resample_poly(y, int(m.group(1)), int(m.group(2)))
    return y_d


def _ref_preproc_iterate_transform(y, dt_meth, n):
    # copied from pyhctsa/operations/pre_process.py lines 498-509 (loop body of preproc_iterate)
    # on main 0fa9013
    if dt_meth == 'spline':
        y_d = _spline_detrend(y, n, 4)  # n pieces, cubic
    elif dt_meth == 'diff':
        y_d = np.diff(y, n=n)
    elif dt_meth == 'medianf':
        y_d = _med_filt_1d(y, n)
    elif dt_meth == 'rav':
        y_d = lfilter(np.ones(n) / n, [1], y)
    elif dt_meth == 'resampleup':
        y_d = resample_poly(y, n, 1)
    else:  # resampledown
        y_d = resample_poly(y, 1, n)
    return y_d


def _ref_band_measures(s, log_s, dw):
    # copied from pyhctsa/operations/spectral.py lines 747-775 (inside spectral_summaries) on
    # main 0fa9013
    _equal_bands = spectral._equal_bands
    out = {}
    # Power in specific frequency bands
    # % 2 bands
    split = _equal_bands(s, 2)
    out['area_2_1'] = np.sum(split[:, 0]) * dw
    out['logarea_2_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_2_2'] = np.sum(split[:, 1]) * dw
    out['logarea_2_2'] = np.sum(np.log(split[:, 1])) * dw
    out['statav2_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)
    # The same on log(S): on the linear spectrum, whichever band contains the dominant peak swamps these
    split_log = _equal_bands(log_s, 2)
    out['logstatav2_m'] = np.std(np.mean(split_log, axis=0), ddof=1) / np.std(log_s, ddof=1)
    out['logstatav2_s'] = np.std(np.std(split_log, ddof=1, axis=0), axis=0, ddof=1) / np.std(log_s, ddof=1)

    # 5 bands
    split = _equal_bands(s, 5)
    out['area_5_1'] = np.sum(split[:, 0]) * dw
    out['logarea_5_1'] = np.sum(np.log(split[:, 0])) * dw
    out['area_5_2'] = np.sum(split[:, 1]) * dw
    out['logarea_5_2'] = np.sum(np.log(split[:, 1])) * dw
    out['area_5_3'] = np.sum(split[:, 2]) * dw
    out['logarea_5_3'] = np.sum(np.log(split[:, 2])) * dw
    out['area_5_4'] = np.sum(split[:, 3]) * dw
    out['logarea_5_4'] = np.sum(np.log(split[:, 3])) * dw
    out['area_5_5'] = np.sum(split[:, 4]) * dw
    out['logarea_5_5'] = np.sum(np.log(split[:, 4])) * dw
    out['statav5_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1) / np.std(s, ddof=1)
    split_log = _equal_bands(log_s, 5)
    out['logstatav5_m'] = np.std(np.mean(split_log, axis=0), ddof=1) / np.std(log_s, ddof=1)
    out['logstatav5_s'] = np.std(np.std(split_log, ddof=1, axis=0), axis=0, ddof=1) / np.std(log_s, ddof=1)
    return out


def _ref_aaft_rank_remap(x, noise):
    # copied from pyhctsa/operations/surrogates.py lines 167-175 (inside _make_surrogates, AAFT)
    # on main 0fa9013, for one surrogate column
    ix = np.argsort(x, kind='stable')
    x_ro = np.argsort(ix, kind='stable')  # rank-ordered permutation
    n_sort = np.sort(noise)
    y = n_sort[x_ro]
    return y


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


def _rtol(call, name):
    # Tier 3 (iterative optimizer): transition_p_alphabet's ``*fexp_*`` keys are golden-section
    # exponential fits, which amplify ulp-level platform differences (CI drift up to 5e-8).
    if 'fexp' in name:
        return 1e-5
    # Tier 2 (linear algebra / least squares): eigenvalues of transition and covariance
    # matrices, tpa's linear fits, polynomial and spline detrending (lstsq), and preproc_iterate's
    # trend/jump fits across the iteration.
    if (name in ('stdeig', 'maxeig', 'maxeigcov') or call == 'tpa'
            or call in ('ppc_poly2', 'ppc_spline24') or call.startswith('ppi_')):
        return 1e-8
    # Tier 1 (closed-form arithmetic): traces, sums, standard deviations, band areas of an FFT
    # or Welch spectrum, filters, resampling, and the AAFT surrogate values (rank remap + FFT).
    return 1e-9


@pytest.mark.parametrize('key', sorted(PINS), ids=lambda k: f'{k[0]}-{k[1]}')
def test_pinned_outputs(key):
    series_name, call = key
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = CALLS[call](SERIES[series_name])
    for name, expected in PINS[key].items():
        np.testing.assert_allclose(out[name], expected, rtol=_rtol(call, name), atol=0,
                                   err_msg=f'{call}.{name}')


def _ref_series():
    # the two pinned series plus a heavy-tailed one with repeated values (ties)
    rng = np.random.default_rng(21)
    ties = np.round(rng.standard_t(3, 300) * 2) / 2
    return {**SERIES, 'ties': (ties - ties.mean()) / ties.std(ddof=1)}


def _recorder(monkeypatch, module, name):
    # wrap module.name so that every call's arguments are recorded
    calls = []
    orig = getattr(module, name)

    def wrapped(*args, **kwargs):
        calls.append(args)
        return orig(*args, **kwargs)
    monkeypatch.setattr(module, name, wrapped)
    return calls


def test_matrix_measures_match_reference(monkeypatch):
    """_matrix_measures against the replaced code of _transition_measures and transition_matrix."""
    rng = np.random.default_rng(5)
    for z in _ref_series().values():
        for ng in (2, 3, 4, 7):
            for tau in (1, 2):
                yth = symbolic.coarse_grain(z, 'quantile', ng)
                T = symbolic._transition_matrix(yth, ng, tau)
                np.testing.assert_allclose(symbolic._transition_measures(yth, ng, tau),
                                           _ref_transition_measures(T), rtol=1e-12, atol=0)
    for ng in (2, 3, 5):  # random row-stochastic matrices (complex eigenvalues)
        T = rng.random((ng, ng)); T /= T.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(symbolic._matrix_measures(T), _ref_transition_measures(T),
                                   rtol=1e-12, atol=0)
    calls = _recorder(monkeypatch, symbolic, '_matrix_measures')
    for z in _ref_series().values():
        for ng, tau in ((2, 1), (3, 1), (5, 2), (7, 1)):
            calls.clear()
            out = symbolic.transition_matrix(z, 'quantile', ng, tau)
            want = _ref_transition_matrix_shared(calls[0][0])
            for name, value in want.items():
                np.testing.assert_allclose(out[name], value, rtol=1e-12, atol=0, err_msg=name)


@pytest.mark.parametrize('meth', ['poly1', 'poly2', 'poly9', 'spline24', 'spline53', 'diff1',
                                  'diff3', 'medianf3', 'medianf12', 'rav3', 'rav10',
                                  'resample_1_2', 'resample_2_1', 'resample_3_7'])
def test_preproc_compare_transform_matches_reference(meth):
    for z in _ref_series().values():
        got = pre_process._transform(z, *pre_process._parse_transform(meth))
        np.testing.assert_allclose(got, _ref_preproc_compare_transform(z, meth), rtol=1e-12, atol=0)


def _iterate_n_range(dt_meth, N):
    # copied from pyhctsa/operations/pre_process.py lines 485-491 (preproc_iterate) on main 0fa9013
    if dt_meth in ('spline', 'resampleup', 'resampledown'):
        n_range = np.arange(1, 21)
    elif dt_meth == 'diff':
        n_range = np.arange(1, 6)
    elif dt_meth in ('medianf', 'rav'):
        n_range = _even_split_points(1, Fraction(N, 25), 25)  # 25 window lengths from 1 to N/25
    return n_range


@pytest.mark.parametrize('dt_meth', ['spline', 'diff', 'medianf', 'rav', 'resampleup', 'resampledown'])
def test_preproc_iterate_transform_matches_reference(monkeypatch, dt_meth):
    # record the (y, y_d) pairs preproc_iterate hands to _iterate_stats, and check each y_d
    # against the replaced loop body on the same y and n
    calls = _recorder(monkeypatch, pre_process, '_iterate_stats')
    for z in _ref_series().values():
        calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pre_process.preproc_iterate(z, dt_meth)
        n_range = _iterate_n_range(dt_meth, len(z))
        assert len(calls) == len(n_range)
        for (y, y_d), n in zip(calls, n_range):
            np.testing.assert_allclose(y_d, _ref_preproc_iterate_transform(y, dt_meth, int(n)),
                                       rtol=1e-12, atol=0, err_msg=f'n={n}')


@pytest.mark.parametrize('psd_meth', ['fft', 'welch'])
def test_band_loop_matches_reference(monkeypatch, psd_meth):
    # record the spectrum s and log_s that spectral_summaries splits into bands, and run the
    # replaced block on them; dw is the bin spacing of the angular frequencies (pi / (n_bins)
    # for the FFT spectrum without its DC bin, pi / (n_bins - 1) for Welch with it), which may
    # differ from spectral_summaries' w[1] - w[0] in the last bit only
    calls = _recorder(monkeypatch, spectral, '_equal_bands')
    for z in _ref_series().values():
        calls.clear()
        out = spectral.spectral_summaries(z, psd_meth)
        s, log_s = calls[0][0], calls[1][0]
        dw = np.pi / len(s) if psd_meth == 'fft' else np.pi / (len(s) - 1)
        for name, value in _ref_band_measures(s, log_s, dw).items():
            np.testing.assert_allclose(out[name], value, rtol=1e-12, atol=0, err_msg=name)


def test_aaft_rank_remap_matches_reference():
    rng = np.random.default_rng(9)
    for z in _ref_series().values():  # includes a series with ties
        for _ in range(3):
            noise = rng.standard_normal(len(z))
            np.testing.assert_array_equal(pre_process._rank_map_gaussian(z, draws=noise),
                                          _ref_aaft_rank_remap(z, noise))


def test_transition_measures_shared():
    """transition_matrix and _transition_measures report the same shared measures."""
    z = SERIES['ar1']
    for ng in (2, 4, 7):
        yth = symbolic.coarse_grain(z, 'quantile', ng)
        m = symbolic._transition_measures(yth, ng, 1)
        out = symbolic.transition_matrix(z, 'quantile', ng, 1)
        assert [out['ondiag'], out['symdiff'], out['sumdiagcov'], out['stdeig']] == [m[2], m[3], m[4], m[5]]
