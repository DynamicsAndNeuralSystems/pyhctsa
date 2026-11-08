"""Pins fit_subsegments and compare_test_sets across the _segment_ranges refactor.

The expected values were generated from the main checkout (0fa9013) before the
uniform / spread / rand segment-range construction was shared, so they check that
the shared helper reproduces both sites exactly, including the seeded random starts.
"""
import warnings

import numpy as np
import pytest

from pyhctsa.operations.model_fit import _segment_ranges, compare_test_sets, fit_subsegments
from pyhctsa.utils import z_score


def _series(name):
    if name == 'ar1':
        e = np.random.default_rng(1).standard_normal(600)
        y = np.zeros(600)
        for t in range(1, 600):
            y[t] = 0.8 * y[t - 1] + e[t]
        return y
    return np.sin(2 * np.pi * np.arange(600) / 37) + 0.3 * np.random.default_rng(2).standard_normal(600)


FS = fit_subsegments
CT = compare_test_sets
CASES = [
    (FS, dict(model='ar', order=2, subset_how='uniform', sample_p=[25, 0.1]),
     {'ar1': {'fpe_mean': 0.41607953130668085, 'a_2_mean': -0.022662963565340286, 'a_1_stdratio': 1.337125071689334},
      'sin': {'fpe_mean': 0.2860780965958622, 'a_2_mean': -0.3358313033200091, 'a_1_stdratio': 0.8148791327296538}}),
    (FS, dict(model='arcrosspred', order=2, subset_how='uniform', sample_p=5),
     {'ar1': {'excess_ratio': 2.287510439031922, 'stdmean': 0.07492928183171964},
      'sin': {'excess_ratio': 0.5423361891586471, 'stdmean': 0.024193303952209127}}),
    (FS, dict(model='ss', order=2, subset_how='uniform', sample_p=[25, 0.1]),
     {'ar1': {'logfpe_stdratio': 1.0491812179979811, 'fpe_mean': 0.44375072682412586},
      'sin': {'logfpe_stdratio': 1.0361011121924995, 'fpe_mean': 0.2954591674030229}}),
    (FS, dict(model='ar', order=2, subset_how='spread', sample_p=[25, 0.1]),
     {'ar1': {'fpe_mean': 0.4101149422135633, 'a_1_mean': -0.6916263171636774},
      'sin': {'fpe_mean': 0.2744715507427893, 'a_1_mean': -0.5359405543081046}}),
    (FS, dict(model='ar', order=2, subset_how='rand', sample_p=[10, 80], random_seed=3),
     {'ar1': {'fpe_mean': 0.42135313937539703, 'a_1_mean': -0.7080235926235988},
      'sin': {'fpe_mean': 0.2829564900669965, 'a_1_mean': -0.5304447922736599}}),
    (CT, dict(the_model='ar', ord=4, subset_how='spread', sample_p=[25, 0.1], steps=1),
     {'ar1': {'stde_mean': 0.6166337815550367, 'stdrat_mean': 0.7626194608167779},
      'sin': {'stde_mean': 0.5186745564045431, 'stdrat_mean': 0.8311745104703396}}),
    (CT, dict(the_model='ss', ord=2, subset_how='uniform', sample_p=[25, 0.1], steps=1),
     {'ar1': {'stde_mean': 0.6103007928975493, 'stdrat_mean': 0.7568206269363807},
      'sin': {'stde_mean': 0.5276568346374799, 'stdrat_mean': 0.9359994171812714}}),
    (CT, dict(the_model='ar', ord=2, subset_how='rand', sample_p=[25, 0.1], steps=1),
     {'ar1': {'stde_mean': 0.6871660377751476, 'meane_mean': 0.10365905223327912},
      'sin': {'stde_mean': 0.5062411772549266, 'meane_mean': 0.14253039608450707}}),
]


@pytest.mark.parametrize('fn, kw, expected', CASES)
@pytest.mark.parametrize('name', ['ar1', 'sin'])
def test_segment_outputs_pinned(fn, kw, expected, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = fn(z_score(_series(name)), **kw)
    for key, val in expected[name].items():
        np.testing.assert_allclose(out[key], val, rtol=1e-12, atol=0)


@pytest.mark.parametrize('args, expected', [
    ((200, [4, 30], 'rand', 'default', None), [[24, 53], [130, 159], [99, 128], [71, 100]]),
    ((200, [4, 0.5], 'rand', 5, (10, 20)), [[56, 75], [172, 191], [22, 41], [108, 127]]),
    ((200, [4, 0.1], 'spread', None, None), [[112, 131], [43, 62], [155, 174], [86, 105]]),
    ((200, [3], 'uniform', None, None), [[1, 67], [68, 133], [134, 200]]),
    ((200, [4, 0.3], 'uniform', None, (10, 20)), [[1, 20], [61, 80], [121, 140], [181, 200]]),
])
def test_segment_ranges(args, expected):
    np.testing.assert_array_equal(_segment_ranges(*args), expected)
