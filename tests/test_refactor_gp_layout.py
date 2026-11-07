"""
Pinned outputs for the Gaussian-process hyperparameter layout refactor (one ``_gp_layout`` in
place of the per-role position walkers in ``model_fit.py``).

The expected values were generated from the main checkout (0fa9013) before the refactor, so the
test checks that the refactor leaves every GP output unchanged (rtol 1e-12).
"""
import numpy as np
import pytest

import pyhctsa.operations.model_fit as mf

RTOL = 1e-12


def _series():
    rng = np.random.default_rng(7)
    wn = rng.standard_normal(300)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    return {k: (y - y.mean()) / y.std(ddof=1) for k, y in (('wn', wn), ('ar1', ar))}


SERIES = _series()

CASES = [
    ('ar1', lambda y: mf.gp_fit_across(y, 'covSEiso_covNoise'),
     {'stde': 0.9177860630020508, 'meanS': 0.723678877103032, 'nlml': 1.300436272482262,
      'logh2': -0.08815140550692771, 'logh3': -0.5452414046928566}),
    ('wn', lambda y: mf.gp_fit_across(y, 'covSEiso_covNoise'),
     {'stde': 1.0226552606920438, 'stdmu': 0.018258212994875116, 'logh3': -0.35719140570889624}),
    ('ar1', lambda y: mf.gp_hyperparameters(y, 'covSEiso_covNoise', 1, 200, 'first'),
     {'logh1': 0.6171109040249246, 'logh2': -0.16161561102675429, 'logh3': -0.8616060751511386,
      'nlml': 1.029397269591294, 'meanS': 0.4989946447392767}),
    ('ar1', lambda y: mf.gp_hyperparameters(y, 'covMaterniso3_covNoise', 1, 200, 'first'),
     {'logh1': 0.8960543012209071, 'logh2': -0.11666246313149084, 'logh3': -0.9909786549775478,
      'stde': 0.2548620189842625}),
    ('wn', lambda y: mf.gp_hyperparameters(y, 'covRQiso_covNoise', 1, 200, 'first'),
     {'logh1': 0.0, 'logh2': -1.9962119439868855, 'logh3': np.nan, 'logh4': -0.06755834772024763,
      'nlml': 1.3603913867845026}),
    ('wn', lambda y: mf.gp_hyperparameters(y, 'covSEiso_covPeriodic_covNoise', 1, 100, 'first'),
     {'logh1': 0.0, 'logh2': -1.4736004215830034, 'logh3': np.nan, 'logh4': np.nan,
      'logh5': -0.4964149089149668, 'logh6': -0.36405912244996264, 'nlml': 1.3378221522489182}),
]


@pytest.mark.parametrize('series, fn, expected', CASES)
def test_gp_outputs_pinned(series, fn, expected):
    out = fn(SERIES[series])
    for k, v in expected.items():
        np.testing.assert_allclose(out[k], v, rtol=RTOL, atol=0, equal_nan=True, err_msg=k)


def test_gp_layout_mixed_components():
    comps = [('covPeriodic', None), ('covRQiso', None), ('covMaterniso', 5), ('covNoise', None),
             ('covSEiso', None)]
    layout = mf._gp_layout(comps)
    assert layout['alpha'] == [5]
    assert layout['amp'] == [2, 4, 7, 10]
    assert layout['shape'] == [0, 1, 3, 5, 6, 9]
    assert layout['len'] == [3, 6, 9]
    assert layout['noise'] == [8]
    tt = np.arange(1, 81, dtype=float) * 0.7
    bounds = mf._gp_shape_bounds(comps, tt)
    assert list(bounds) == [0, 1, 3, 5, 6, 9]
    assert np.isnan([bounds[0], bounds[1], bounds[5]]).all()
    np.testing.assert_allclose([bounds[3], bounds[6], bounds[9]], -0.3566749439387323, rtol=RTOL, atol=0)
    comps2 = [('covNoise', None), ('covRQiso', None), ('covNoise', None), ('covPeriodic', None)]
    np.testing.assert_allclose(
        mf._gp_init_hyp(comps2, tt),
        [-2.3025850929940455, -0.3566749439387323, 0.0, 0.0, -2.3025850929940455,
         -0.3566749439387323, 1.7101878155342434, 0.0, -2.3025850929940455], rtol=RTOL, atol=0)
