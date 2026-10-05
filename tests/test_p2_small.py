"""Tests for the small hctsa robustness changes: bounded loss ratios of compare_ar and the rounding-level
guards of falling_sticks. Expected values in ``data/p2_small.json`` are outputs of hctsa (MATLAB R2026a, branch
robust/small) on the stored series."""
import numpy as np

from pyhctsa.operations import model_fit as mf


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
