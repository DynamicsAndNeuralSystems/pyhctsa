"""Pinned outputs for the run-length refactor (duplication audit theme #20).

``binary_stats``, ``binary_stats_ar1``, ``pol_var`` and ``pol_var_resid`` now share
``utils._run_lengths``; the expected values below were generated from the main checkout
(0fa9013) before the refactor, so these tests check that the outputs are unchanged.
"""
import numpy as np
import pytest

from pyhctsa.operations.medical import pol_var, pol_var_resid
from pyhctsa.operations.symbolic import binary_stats, binary_stats_ar1
from pyhctsa.utils import _run_lengths


def _series():
    rng = np.random.default_rng(7)
    wn = rng.standard_normal(300)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    rep = np.round(rng.standard_normal(300) * 2)
    return {'wn': wn, 'ar1': ar, 'rep': rep}


_KEYS = ['pstretch1', 'longstretch0', 'longstretch1norm', 'meanstretch0', 'meanstretch1norm',
         'stdstretch0', 'stdstretch1norm', 'meanstretchdiff', 'stdstretchdiff', 'diff21stretch0',
         'diff21stretch1']
_POL_VAR_ARGS = [(1.0, 3, 1), (0.5, 5, 1), (0.1, 6, 1), (1.0, 3, 'ac1e'), (1.0, 3, 4)]

# generated from the main checkout (0fa9013), before the refactor
_EXPECTED = {
    ('wn', 'binary_stats', 'mean'): [0.25, 7.0, 0.02666666666666667, 1.9324324324324325, 0.006977777777777777, 1.2641644712776015, 0.004250890503185194, 0.0005363363363363361, 3.7008932259855835e-05, -0.27027027027027023, -0.013333333333333364],
    ('wn', 'binary_stats', 'diff'): [0.34782608695652173, 4.0, 0.013377926421404682, 1.4757281553398058, 0.004727296115255982, 0.7389003513989855, 0.0022645321808376084, -0.00020824955477681369, -0.00020670645260381434, -0.4077669902912622, -0.4134615384615385],
    ('wn', 'binary_stats', 'iqr'): [0.25666666666666665, 7.0, 0.016666666666666666, 1.9736842105263157, 0.006493506493506493, 1.2855471904785791, 0.004254206858277208, -8.544087491455891e-05, -3.095044331805556e-05, -0.21052631578947367, -0.3246753246753246],
    ('wn', 'binary_stats_ar1', 'mean'): [0.25, 1.9324324324324325, 2.0933333333333333],
    ('wn', 'binary_stats_ar1', 'diff'): [0.34782608695652173, 1.4757281553398058, 1.4134615384615385],
    ('wn', 'pol_var', ''): [0.1605351170568562, 0.06688963210702341, 0.13043478260869565, 0.1605351170568562, 0.16216216216216217],
    ('wn', 'pol_var_resid', ''): [0.14429530201342283, 0.016778523489932886],
    ('ar1', 'binary_stats', 'mean'): [0.11, 26.0, 0.04666666666666667, 4.71875, 0.015050505050505052, 6.165640353421689, 0.011699988488871715, -0.0006786616161616147, -0.008852146022533916, -0.3125, -0.06060606060606061],
    ('ar1', 'binary_stats', 'diff'): [0.27759197324414714, 5.0, 0.016722408026755852, 1.8433734939759037, 0.005883064028690011, 1.0058599974205227, 0.0033277319972515843, -0.00028206471370431613, -3.634826167992955e-05, -0.1566265060240964, -0.3012048192771084],
    ('ar1', 'binary_stats', 'iqr'): [0.18666666666666668, 15.0, 0.03333333333333333, 2.727272727272727, 0.008928571428571428, 2.280793765779844, 0.007523891719645267, -0.0001623376623376623, -7.87541662875461e-05, -0.1272727272727273, -0.2678571428571428],
    ('ar1', 'binary_stats_ar1', 'mean'): [0.11, 4.71875, 4.515151515151516],
    ('ar1', 'binary_stats_ar1', 'diff'): [0.27759197324414714, 1.8433734939759037, 1.7590361445783131],
    ('ar1', 'pol_var', ''): [0.14715719063545152, 0.043478260869565216, 0.11371237458193979, 0.1750841750841751, 0.15202702702702703],
    ('ar1', 'pol_var_resid', ''): [0.15100671140939598, 0.020134228187919462],
    ('rep', 'binary_stats', 'mean'): [0.22666666666666666, 6.0, 0.03333333333333333, 1.7681159420289856, 0.008725490196078432, 1.2502344196217043, 0.006839292675421068, 0.002831770389315146, 0.002671844610015387, -0.463768115942029, -0.2352941176470588],
    ('rep', 'binary_stats', 'diff'): [0.3311036789297659, 5.0, 0.010033444816053512, 1.6868686868686869, 0.004459308807134894, 0.9107797135765286, 0.0019111323459149545, -0.001182392486740313, -0.0011349536526687534, -0.30303030303030304, -0.4848484848484848],
    ('rep', 'binary_stats', 'iqr'): [0.24666666666666667, 8.0, 0.02, 2.4133333333333336, 0.00536036036036036, 1.7249624620540285, 0.003062890645668974, -0.002684084084084085, -0.002686984227844454, -0.21333333333333335, -0.2837837837837837],
    ('rep', 'binary_stats_ar1', 'mean'): [0.22666666666666666, 1.7681159420289856, 2.6176470588235294],
    ('rep', 'binary_stats_ar1', 'diff'): [0.3311036789297659, 1.6868686868686869, 1.3333333333333333],
    ('rep', 'pol_var', ''): [0.24414715719063546, 0.11705685618729098, 0.0802675585284281, 0.24414715719063546, 0.21621621621621623],
    ('rep', 'pol_var_resid', ''): [0.12416107382550336, 0.003355704697986577],
}


def _compute(name, op, arg):
    x = _series()[name]
    if op == 'binary_stats':
        out = binary_stats(x, arg)
        return [out[k] for k in _KEYS]
    if op == 'binary_stats_ar1':
        out = binary_stats_ar1(x, arg)
        return [out[k] for k in ['pstretch1', 'meanstretch0', 'meanstretch1']]
    if op == 'pol_var':
        return [pol_var(x, d, D, tau) for d, D, tau in _POL_VAR_ARGS]
    return [pol_var_resid(x, D) for D in [3, 6]]


@pytest.mark.parametrize('key', list(_EXPECTED), ids=lambda k: '-'.join(filter(None, k)))
def test_pinned_outputs(key):
    np.testing.assert_allclose(np.asarray(_compute(*key), dtype=float), _EXPECTED[key],
                               rtol=1e-12, atol=0)


def test_run_lengths_edges():
    v, n = _run_lengths(np.array([1, 1, 0, 1, 0, 0, 0]))
    np.testing.assert_array_equal(v, [1, 0, 1, 0])
    np.testing.assert_array_equal(n, [2, 1, 1, 3])  # runs touching both ends counted in full
    v, n = _run_lengths(np.array([True]))
    np.testing.assert_array_equal(n, [1])
    v, n = _run_lengths(np.array([]))
    assert v.size == 0 and n.size == 0


def test_binary_stats_no_runs_of_a_symbol():
    out = binary_stats(np.arange(20.0), 'diff')  # all increases: no runs of 0s
    assert out['longstretch0'] == 0 and out['meanstretch0'] == 0
    assert np.isnan(out['stdstretch0']) and out['longstretch1'] == 19
