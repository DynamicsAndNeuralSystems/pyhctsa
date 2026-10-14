"""Build tests/data/p2_b.json from the MATLAB outputs (mlA-mlD.json and mlG.json, written by mlA.m ... mlG.m with
current hctsa, branch robust/all, on the series of inputs.mat made by gen_inputs.py) for a few series."""
import json, sys
import numpy as np
import scipy.io as sio

D = sys.argv[1]  # directory with inputs.mat and mlA.json ... mlD.json
KEEP = ['s3', 's20', 'expn700', 'quant600', 'const300', 'short60', 'posmean500']
S = sio.loadmat(f'{D}/inputs.mat')
ml = {k: json.load(open(f'{D}/ml{k}.json')) for k in 'ABCDG'}
fix = {'inputs': {n: S[n].ravel().tolist() for n in KEEP}, 'expected': {}}
for letter in 'ABCDG':
    for n in KEEP:
        for key, val in ml[letter][n].items():
            fix['expected'].setdefault(letter + '_' + key, {})[n] = val
json.dump(fix, open('p2_b.json', 'w'), allow_nan=True)
