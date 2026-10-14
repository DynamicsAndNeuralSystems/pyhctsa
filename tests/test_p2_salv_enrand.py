"""randomize_ordinal (hctsa's EN_RandomizeOrdinal): the exact expected ordinal-pattern decay
under 'statdist' randomization, against MATLAB values and brute-force enumeration."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import entropy as en

FIX = json.loads((Path(__file__).parent / 'data' / 'salv_enrand.json').read_text())


def _zs(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


@pytest.mark.parametrize('case', FIX['cases'], ids=[f"s{c['series']}-tau{c['tau']}" for c in FIX['cases']])
def test_matches_matlab(case):
    out = en.randomize_ordinal(_zs(FIX['series'][case['series']]), case['tau'])
    for k, v in case['expected'].items():
        assert out[k] == pytest.approx(v, rel=1e-11, abs=1e-12), k


def test_pattern_dists_brute_force():
    # every kept subset, by enumerating all draws, on a short series with repeated values
    y = np.array([2., 1., 2., 3., 1., 2., 2.])
    n = y.size
    S = en._ro_pattern_dists(y, 1)
    W = np.column_stack([y[:-2], y[1:-1], y[2:]])
    for k, K in enumerate(en._RO_SUBSETS):
        free = [j for j in range(3) if j not in K]
        B = np.zeros(6)
        for w in W:
            for d in itertools.product(range(n), repeat=len(free)):
                z = w.copy()
                for j, i in zip(free, d):
                    z[j] = y[i]
                B[en._ordinal_pattern_rank(z[None, :])[0]] += n ** -len(free)
        np.testing.assert_allclose(S[k], B / len(W), atol=1e-13)


def test_deterministic_ordinal_and_edge_cases():
    t = np.arange(1, 301)
    y = np.sin(0.2 * t) + 0.3 * np.cos(0.05 * t ** 1.1)
    a = en.randomize_ordinal(y)
    assert a == en.randomize_ordinal(y)
    assert en.randomize_ordinal(np.exp(2 * y) + 5)['halftime'] == pytest.approx(a['halftime'], abs=1e-12)
    assert a['normPermEn0'] == pytest.approx(en.permutation_entropy(y, 3, 1)['normPermEn'], abs=1e-12)
    assert np.isnan(en.randomize_ordinal(np.ones(50))['halftime'])
    assert np.isnan(en.randomize_ordinal(np.arange(6.)))
