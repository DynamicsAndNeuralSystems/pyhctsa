"""Agreement with hctsa of the model-fitting and system-identification features (state-space, ARMA, AR,
prediction and exponential-smoothing families), whose algorithms are specified identically in both.

The expected values (``tests/data/mf_parity.json``) are hctsa's outputs (MATLAB R2026a) on the series of
``tests/data/p2_c.json``. Both implementations follow the same documented algorithms (subspace identification
with a fixed horizon rule, a structured-doubling Riccati solver, diffuse-start Kalman predictions, the
Hannan-Rissanen ARMA regression, forward-backward AR fits, grid-plus-golden-section exponential smoothing), so
they agree to rounding (1e-6 here), except where a quantity is itself at the level of rounding error (some fits
of the exactly periodic ``x_sine``), which are not compared.
"""
import json
import math
import os

import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, 'data', 'p2_c.json')) as f:
    SERIES = {k: np.array(v) for k, v in json.load(f)['series'].items()}
with open(os.path.join(HERE, 'data', 'mf_parity.json')) as f:
    EXPECTED = json.load(f)['cases']

CASES = {
    'arcov3': lambda y: mf.ar_cov(y, 3),
    'car05': lambda y: mf.compare_ar(y, np.arange(1, 11), 0.5),
    'carall': lambda y: mf.compare_ar(y, np.arange(1, 11), 'all'),
    'fs_ar3': lambda y: mf.fit_subsegments(y, 'ar', 2, 'rand', [25, 0.1], 3),
    'fs_ss2': lambda y: mf.fit_subsegments(y, 'ss', 2, 'uniform', [25, 0.1]),
    'fs_xp': lambda y: mf.fit_subsegments(y, 'arcrosspred', 2, 'uniform', 5),
    'ss1': lambda y: mf.state_space_n4sid(y, 1, 0.5, 1),
    'ss3': lambda y: mf.state_space_n4sid(y, 3, 0.5, 1),
    'ssbest': lambda y: mf.state_space_n4sid(y, 'best', 0.5, 1),
    'sco8': lambda y: mf.state_space_comp_order(y, 8),
    'armax11': lambda y: mf.armax(y, [1, 1], 0.5, 1),
    'armax31': lambda y: mf.armax(y, [3, 1], 0.5, 1),
    'sa_ar2': lambda y: mf.steps_ahead(y, 'ar', 2, 6),
    'sa_arma': lambda y: mf.steps_ahead(y, 'arma', [3, 1], 6),
    'sa_ss': lambda y: mf.steps_ahead(y, 'ss', 'best', 6),
    'cts_ar': lambda y: mf.compare_test_sets(y, 'ar', 4, 'spread', [25, 0.1], 1),
    'cts_ss2': lambda y: mf.compare_test_sets(y, 'ss', 2, 'uniform', [25, 0.1], 1),
    'cts_ssb': lambda y: mf.compare_test_sets(y, 'ss', 'best', 'uniform', [25, 0.1], 1),
    'expsm': lambda y: mf.exp_smoothing(y, 0.5, 'best'),
}
# fits of the exactly periodic series x_sine that are not determined by the data (an exact fit leaves some model
# parameters, e.g. a noise gain or an MA coefficient, set by rounding)
SINE_SKIP = {'ss1', 'ss3', 'ssbest', 'armax31'}


def _is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


@pytest.mark.parametrize('label', list(CASES))
@pytest.mark.parametrize('name', list(SERIES))
def test_matches_hctsa(label, name):
    if name == 'x_sine' and label in SINE_SKIP:
        pytest.skip('parameters not determined by an exactly periodic series')
    expected = EXPECTED[label][name]
    assert not isinstance(expected, str), expected   # (hctsa raised an error)
    out = CASES[label](SERIES[name])
    if _is_nan(expected):
        assert all(np.isnan(v) for v in out.values())
        return
    for key, exp in expected.items():
        if key not in out:
            continue   # (an output that pyhctsa does not compute, e.g. np/Ts constants)
        got = out[key]
        if _is_nan(exp):
            assert np.isnan(got), key
        else:
            np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-8, err_msg=f'{label}.{key}')
