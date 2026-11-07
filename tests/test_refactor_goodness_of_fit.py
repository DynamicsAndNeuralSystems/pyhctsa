"""Pinned outputs of the operations whose R^2 / adjusted R^2 / RMSE now come from
:func:`pyhctsa.fitting.goodness_of_fit` (duplication-audit theme 7).

The reference values were generated from the main checkout at 0fa9013, before the refactor, so
these tests check that routing the inline goodness-of-fit code through the shared function left
every output unchanged (including the -inf of a zero total sum of squares).
"""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.fitting import goodness_of_fit
from pyhctsa.operations import distribution, graph, scaling, spectral, symbolic


def _series():
    r = np.random.default_rng(7)
    wn = r.standard_normal(600)
    e = r.standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    return {'wn': wn, 'ar1': ar}


CASES = {
    'spectral_summaries': lambda y: spectral.spectral_summaries(y),
    'sinusoid_fit_sin1': lambda y: spectral.sinusoid_fit(y, 'sin1'),
    'sinusoid_fit_fourier1': lambda y: spectral.sinusoid_fit(y, 'fourier1'),
    'simple_fit_gauss1': lambda y: distribution.simple_fit(y, 'gauss1'),
    'simple_fit_exp1_ks': lambda y: distribution.simple_fit(y, 'exp1', 0),
    'outlier_include': lambda y: distribution.outlier_include(y),
    'visibility_graph': lambda y: graph.visibility_graph(y, 'horiz'),
    'transition_p_alphabet': lambda y: symbolic.transition_p_alphabet(y),
    'mfdfa': lambda y: scaling.mfdfa(y),
}

# Generated from the main checkout (0fa9013), before the refactor.
EXPECTED = {
    ('spectral_summaries', 'wn'): {'fpolysat_r2': 0.9810690730312851, 'fpolysat_rmse': 0.03413046016877067, 'fpoly2_r2': 0.9987726868073356, 'fpoly2_rmse': 0.008698815144143513},
    ('sinusoid_fit_sin1', 'wn'): {'r2': 0.0014937324451347322, 'adjr2': -0.0018513471781647084, 'rmse': 0.922775328752735},
    ('sinusoid_fit_fourier1', 'wn'): {'r2': 0.022396254165364282, 'adjr2': 0.017475429941364418, 'rmse': 0.9138313255143474},
    ('simple_fit_gauss1', 'wn'): {'r2': 0.9168221256096379, 'adjr2': 0.9092605006650596, 'rmse': 0.04494544570221991},
    ('simple_fit_exp1_ks', 'wn'): {'r2': 0.004903534590019687, 'adjr2': -0.005250510975388201, 'rmse': 0.13638393981312666},
    ('outlier_include', 'wn'): {'nflr2': 0.9485279796940483, 'nflrmse': 6.354357228287337, 'stdrflr2': 0.8334423855477887, 'stdrflrmse': 3.7441908156958306},
    ('visibility_graph', 'wn'): {'dgaussk_r2': 0.7687129256831513, 'dgaussk_adjr2': 0.7398020413935452, 'dgaussk_rmse': 0.10742332995895273, 'dexpk_r2': 0.45992427266315505, 'dexpk_adjr2': 0.428155112231576, 'dexpk_rmse': 0.15925232799392475, 'dpowerk_r2': 0.2385123324434938, 'dpowerk_adjr2': 0.1937189402342876, 'dpowerk_rmse': 0.1890992895590311},
    ('transition_p_alphabet', 'wn'): {'trflin5_adjr2': 0.7844188036710243, 'trflin10adjr2': 0.7844188036710243},
    ('mfdfa', 'wn'): {'meanR2': 0.981515248346813},
    ('spectral_summaries', 'ar1'): {'fpolysat_r2': 0.9806546281820142, 'fpolysat_rmse': 0.08676843718559242, 'fpoly2_r2': 0.7759450219267804, 'fpoly2_rmse': 0.2955811471653132},
    ('sinusoid_fit_sin1', 'ar1'): {'r2': 0.15206778735488746, 'adjr2': 0.14922714342642818, 'rmse': 1.670930286807655},
    ('sinusoid_fit_fourier1', 'ar1'): {'r2': 0.15363678086667654, 'adjr2': 0.14937656332070348, 'rmse': 1.6707835489397702},
    ('simple_fit_gauss1', 'ar1'): {'r2': 0.9484996958868672, 'adjr2': 0.9438178500584006, 'rmse': 0.032014166331249774},
    ('simple_fit_exp1_ks', 'ar1'): {'r2': 0.007916881614282167, 'adjr2': -0.002206415512102655, 'rmse': 0.13324873476439322},
    ('outlier_include', 'ar1'): {'nflr2': 0.9428431376251895, 'nflrmse': 7.073868383570861, 'stdrflr2': 0.8817177567577196, 'stdrflrmse': 3.3010726377579376},
    ('visibility_graph', 'ar1'): {'dgaussk_r2': 0.8415462965525552, 'dgaussk_adjr2': 0.8127365322893834, 'dgaussk_rmse': 0.07998391658633684, 'dexpk_r2': 0.39600470321992254, 'dexpk_adjr2': 0.34567176182158277, 'dexpk_rmse': 0.14951128268332914, 'dpowerk_r2': 0.17434714204398594, 'dpowerk_adjr2': 0.10554273721431817, 'dpowerk_rmse': 0.17480583821464138},
    ('transition_p_alphabet', 'ar1'): {'trflin5_adjr2': 0.9217186292026659, 'trflin10adjr2': 0.9217186292026659},
    ('mfdfa', 'ar1'): {'meanR2': 0.9546859292192903},
}


@pytest.fixture(scope='module')
def outputs():
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {(name, s): f(y) for s, y in _series().items() for name, f in CASES.items()}
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize('case', list(EXPECTED))
def test_pinned_outputs(outputs, case):
    out = outputs[case]
    for key, val in EXPECTED[case].items():
        np.testing.assert_allclose(out[key], val, rtol=1e-12, atol=0, err_msg=f'{case} {key}')


def test_zero_total_sum_of_squares():
    # Equal histogram counts (SST = 0) and a constant series keep the plain floating-point
    # result -inf, not the NaN of goodness_of_fit's default guard (values from main, 0fa9013).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = distribution.simple_fit(np.arange(400.0), 'gauss2')
        assert o['r2'] == -np.inf and o['adjr2'] == -np.inf
        np.testing.assert_allclose(o['rmse'], 2.440727683641595e-06, rtol=1e-12, atol=0)
        o = spectral.sinusoid_fit(np.ones(50), 'sin1')
        assert o['r2'] == -np.inf and o['adjr2'] == -np.inf
        np.testing.assert_allclose(o['rmse'], 0.9831736241435396, rtol=1e-12, atol=0)


def test_goodness_of_fit_conventions():
    y = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    yh = np.array([1.2, 2.1, 3.3, 3.9, 4.6])
    sse = np.sum((y - yh) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    g = goodness_of_fit(y, yh, 2)
    assert g['adjrsquare'] == 1 - sse * 4 / (sst * 3)
    assert goodness_of_fit(y, yh, 2, adj_from_rsquare=True)['adjrsquare'] == 1 - (1 - r2) * 4 / 3
    assert g['rsquare'] == r2 and g['rmse'] == np.sqrt(sse / 3)
    c = np.ones(5)
    assert np.isnan(goodness_of_fit(c, yh, 2)['rsquare'])
    with np.errstate(all='ignore'):
        assert goodness_of_fit(c, yh, 2, guard_sst=False)['rsquare'] == -np.inf
        assert np.isnan(goodness_of_fit(c, c, 2, guard_sst=False)['rsquare'])
    assert np.isnan(goodness_of_fit(y, yh, 5, guard_sst=False)['adjrsquare'])
