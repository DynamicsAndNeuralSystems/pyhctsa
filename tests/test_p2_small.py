"""Tests for the small hctsa robustness changes: bounded loss ratios of compare_ar and the rounding-level
guards of falling_sticks. Expected values in ``data/p2_small.json`` are outputs of hctsa (MATLAB R2026a, branch
robust/small) on the stored series."""
import json
import pathlib

import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import model_fit as mf

DATA = json.loads((pathlib.Path(__file__).parent / 'data' / 'p2_small.json').read_text())
SERIES = {k: np.array(v) for k, v in DATA['series'].items()}


@pytest.mark.parametrize('name', list(SERIES))
def test_falling_sticks_matches_hctsa(name):
    out = co.falling_sticks(SERIES[name])
    for key, exp in DATA['fs'][name].items():
        if exp is None:
            assert np.isnan(out[key]), f'{key} on {name}: hctsa gives NaN, got {out[key]}'
        else:
            np.testing.assert_allclose(out[key], exp, rtol=1e-8, atol=1e-9, err_msg=f'{key} on {name}')


def test_falling_sticks_flat_branches_are_nan():
    # a branch whose sticks all fall flat has constant angles (rounding noise only): no persistence statistics
    out = co.falling_sticks(SERIES['flatboth'])
    for key in ('tau_p', 'ac1_p', 'tau_n', 'ac1_n', 'skewness_all', 'kurtosis_all'):
        assert np.isnan(out[key]), key
    assert out['std_all'] == 0 and out['propFlat_all'] == 1
    out = co.falling_sticks(SERIES['s350'])  # only the negative branch is flat
    assert np.isnan(out['tau_n']) and np.isnan(out['ac1_n']) and np.isfinite(out['tau_p'])


def test_compare_ar_bounded_ratios():
    rng = np.random.RandomState(1)
    t = np.arange(600)
    for y in (rng.randn(600), np.cumsum(rng.randn(600)), np.sin(0.1 * t) + 0.5 * rng.randn(600)):
        y = (y - y.mean()) / y.std(ddof=1)
        for how in (0.5, 'all'):
            out = mf.compare_ar(y, np.arange(1, 11), how)
            assert 0 <= out['propgain1min'] <= 1 and 0 < out['medonmax'] <= 1
            assert 'firstonmin' not in out and 'maxonmed' not in out
    # a noiseless series is predicted exactly by a higher order: the limits 1 and 0
    y = np.sin(0.1 * t) + np.sin(0.37 * t)
    out = mf.compare_ar(y / y.std(ddof=1), np.arange(1, 11), 'all')
    assert out['propgain1min'] > 1 - 1e-9 and out['medonmax'] < 1e-9
