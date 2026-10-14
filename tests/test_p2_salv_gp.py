"""Tests for gp_local_prediction's deterministic 'spreadgap' splits and its robust tail outputs
(hctsa branch robust/salv-gp).

Expected values in ``data/p2_salv_gp.json`` are hctsa's (MATLAB R2026a) on the z-scored rows of the
stationary1000 corpus stored in ``data/p2_c.json``.
"""
import json
import pathlib

import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf

HERE = pathlib.Path(__file__).parent / 'data'
REF = json.loads((HERE / 'p2_salv_gp.json').read_text())
SERIES = {k: np.array(v) for k, v in json.loads((HERE / 'p2_c.json').read_text())['series'].items()}
COV = 'covSEiso_covNoise'
CALLS = {'small': lambda y: mf.gp_local_prediction(y, COV, 10, 3, 6, 'spreadgap', None, 2),
         'full': lambda y: mf.gp_local_prediction(y, COV, 10, 3, 20, 'spreadgap'),
         'gph2': lambda y: mf.gp_hyperparameters(y, COV, 1, 50, 'random_i', 0, 2),
         'gphboth': lambda y: mf.gp_hyperparameters(y, COV, 1, 200, 'random_both', 1, 2),
         'gph20': lambda y: mf.gp_hyperparameters(y, COV, 1, 50, 'random_i', 'default')}
PARAMS = [(c, s) for c in CALLS for s in REF[c]]


@pytest.mark.parametrize('call,name', PARAMS)
def test_spreadgap_matches_hctsa(call, name):
    # the GP fits run the minimize optimizer for 50 evaluations: agreement is to optimizer noise
    # (as in test_p2_c; the noise hyperparameter, often at its floor, is the loosest), and
    # tighter for the new tail summaries
    out = CALLS[call](SERIES[name])
    tight = {'q90abs_run', 'q10abs_run', 'q90abs_std_run', 'low25abs_std_run', 'high25errbar', 'q90nlml', 'oosmedabserr'}
    for key, exp in REF[call][name].items():
        if exp is None:  # hctsa gives NaN
            assert np.isnan(out[key]), f'{call} {name} {key}'
            continue
        tol = 1e-3 if key in tight else 1e-2
        assert out[key] == pytest.approx(exp, rel=tol, abs=tol), f'{call} {name} {key}'


def test_spreadgap_ignores_the_seed():
    y = SERIES['s250']
    a = mf.gp_local_prediction(y, COV, 10, 3, 4, 'spreadgap', 0, 2)
    b = mf.gp_local_prediction(y, COV, 10, 3, 4, 'spreadgap', 7, 2)
    assert a == b


def test_tail_summaries_are_ordered():
    o = mf.gp_local_prediction(SERIES['s0'], COV, 10, 3, 6, 'spreadgap', None, 4)
    assert o['meanabs_run'] <= o['q90abs_run'] + 1e-12 <= o['maxabs_run'] + 1e-12
    assert o['minabs_run'] <= o['q10abs_run'] <= o['meanabs_run'] + 1e-12
    assert o['q90abs_std_run'] <= o['maxabs_std_run']
    assert o['minabs_std_run'] <= o['low25abs_std_run'] <= o['meanabs_std_run']
    assert o['meanerrbar'] <= o['high25errbar'] <= o['maxerrbar']
    assert o['minnlml'] <= o['q90nlml'] <= o['maxnlml']


def test_spreadgap_cycles_through_the_test_sets():
    # 2 windows x 8 splits = 16 fits, more than the C(5, 2) = 10 possible test sets
    o = mf.gp_local_prediction(SERIES['s700'], COV, 3, 2, 2, 'spreadgap', None, 8)
    assert np.isfinite(o['q90abs_run'])


def test_out_of_sample_error_is_nan_without_unseen_points():
    o = mf.gp_hyperparameters(SERIES['s0'][:300], COV, 1, 200, 'first')
    assert np.isnan(o['oosmedabserr'])
