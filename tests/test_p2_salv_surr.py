"""surrogates (hctsa's SD_Surrogates): the meannumsurr output against MATLAB, same seed."""
import json
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import surrogates as su
from pyhctsa.operations.correlation import tc3, trev

FIX = json.loads((Path(__file__).parent / 'data' / 'salv_surr.json').read_text())
CASES = [(s, c) for s in FIX for c in FIX[s] if c != 'y']


@pytest.mark.parametrize('s,c', CASES)
def test_meannumsurr_equal_matlab(s, c):
    x = np.array(FIX[s]['y'], dtype=float)
    d = FIX[s][c]
    out = su.surrogates(x, d['tau'], 20, d['m'], d['fn'], 42)
    for k in ('meannumsurr', 'meansurr', 'stdsurr'):
        assert out[k] == pytest.approx(d[k], rel=1e-7, abs=1e-9), (s, c, k)


def test_meannumsurr_is_mean_of_numerator():
    x = np.array(FIX['s1']['y'], dtype=float)
    z = su._make_surrogates(x, 'AAFT', 20, 3)
    for fn, f in (('tc3', tc3), ('trev', trev)):
        out = su.surrogates(x, 2, 20, 2, fn, 3)
        assert out['meannumsurr'] == pytest.approx(np.mean([f(z[:, i], 2)['num'] for i in range(20)]), abs=1e-12)
