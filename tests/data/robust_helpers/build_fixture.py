"""Merge the inputs (gen_inputs.py -> inputs.json) and the MATLAB results (ml_run.m -> results.mat,
run with hctsa's PeripheryFunctions on the path) into ../robust_helpers.json, which
tests/test_robust_helpers.py reads. Run from this directory:
    python gen_inputs.py; matlab -batch "run('ml_run.m')"; python build_fixture.py
"""
import json
import numpy as np
import scipy.io as sio

inputs = json.load(open('inputs.json'))
res = {k: np.atleast_1d(np.asarray(v).squeeze()).tolist()
       for k, v in sio.loadmat('results.mat').items() if not k.startswith('__')}
json.dump({'inputs': inputs, 'expected': res}, open('../robust_helpers.json', 'w'), separators=(',', ':'))
