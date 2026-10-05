"""Entropy functions ported from hctsa (expected values generated with MATLAB hctsa)."""
import numpy as np
import pytest

from pyhctsa.operations.entropy import (
    approximate_entropy, dispersion_entropy, distribution_entropy,
    multi_scale_entropy, permutation_entropy, sample_entropy)


def _x():
    n = np.arange(1, 301)
    x = np.sin(0.3 * n) + 0.5 * np.cos(1.7 * n) ** 3 + 0.2 * np.sin(0.011 * n ** 1.5)
    return (x - x.mean()) / x.std(ddof=1)


def _all_nan(out):
    """True for NaN, or for the dict of NaN fields a failed dict-valued function returns."""
    return all(np.isnan(v) for v in out.values()) if isinstance(out, dict) else bool(np.isnan(out))


def test_dispersion_entropy():
    d = dispersion_entropy(_x(), 2, 6, 1)
    np.testing.assert_allclose(
        [d['dispEn'], d['normDispEn'], d['fDispEn'], d['normFDispEn']],
        [2.94501554195247, 0.821822234661126, 1.50216767357667, 0.626452577231876], rtol=1e-10)
    d = dispersion_entropy(_x(), 3, 4, 'ac1e', 'linear')
    np.testing.assert_allclose(
        [d['dispEn'], d['normDispEn'], d['fDispEn'], d['normFDispEn']],
        [3.25592248023321, 0.78288386929189, 2.80272781425558, 0.720158588929768], rtol=1e-10)
    assert _all_nan(dispersion_entropy(np.ones(100)))
    assert _all_nan(dispersion_entropy(_x()[:5], 3, 6, 2))


def test_multi_scale_entropy():
    o = multi_scale_entropy(_x(), range(1, 11), 2, 0.15)
    np.testing.assert_allclose(
        [o['sampen_s3'], o['meanSampEn'], o['slope'], o['slopeSE']],
        [0.816877861512244, 1.02179390506592, -0.0235069683559523, 0.0433827817660596], rtol=1e-9)
    o = multi_scale_entropy(_x(), range(1, 11), 2, 0.15, what_entropy='dispen', num_classes=6)
    np.testing.assert_allclose(
        [o['dispen_s4'], o['meanDispEn'], o['slope']],
        [0.823471080718485, 0.769342711043293, -0.0181231642192005], rtol=1e-9)
    assert _all_nan(multi_scale_entropy(_x()[:10]))
    with pytest.raises(ValueError):
        multi_scale_entropy(_x(), what_entropy='nope')


def test_sample_entropy_nan_without_matches():
    o = sample_entropy(_x()[:40], 5, 0.02)
    assert o['sampen1'] == 0
    assert np.isnan(o['sampen2']) and np.isnan(o['sampen3'])


def test_delay_rules():
    x = _x()
    np.testing.assert_allclose(approximate_entropy(x, 2, 0.2, 'ac1e'), 0.741563785040266, rtol=1e-10)
    o = permutation_entropy(x, 4, 'ac1e')
    np.testing.assert_allclose([o['permEn'], o['normWPE']], [3.41955364314743, 0.714765170552043], rtol=1e-10)
    assert np.isnan(approximate_entropy(np.ones(100), 2, 0.2, 'ac1e'))


def test_distribution_entropy_ks_grid():
    x = _x()
    assert np.isfinite(distribution_entropy(x, 'ks', '', 0))
    assert np.isnan(distribution_entropy(np.ones(100), 'ks', '', 0))
