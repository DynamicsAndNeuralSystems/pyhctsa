"""Entropy functions ported from hctsa (expected values generated with MATLAB hctsa)."""
import numpy as np
import pytest

from pyhctsa.operations.entropy import (
    approximate_entropy, dispersion_entropy, distribution_entropy,
    multi_scale_entropy, permutation_entropy, sample_entropy)
from pyhctsa.toolboxes.physionet import sampen as _sampen_c


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
        [o['sampen_s3'], o['meanSampEn'], o['slope']],
        [0.810930216216329, 0.999876654612307, -0.0302895740419296], rtol=1e-9)
    # the clean-room robust (bisquare) fit agrees with MATLAB's robustfit to ~1e-8 relative in the
    # standard error (iteration stopping tolerance)
    np.testing.assert_allclose(o['slopeSE'], 0.0416403276927917, rtol=1e-7)
    o = multi_scale_entropy(_x(), range(1, 11), 2, 0.15, what_entropy='dispen', num_classes=6)
    np.testing.assert_allclose(
        [o['dispen_s4'], o['meanDispEn'], o['slope']],
        [0.823471080718485, 0.769342711043293, -0.0181231642192005], rtol=1e-9)
    assert _all_nan(multi_scale_entropy(_x()[:10]))
    with pytest.raises(ValueError):
        multi_scale_entropy(_x(), what_entropy='nope')


def test_sample_entropy_nan_without_matches():
    o = sample_entropy(_x()[:40], 5, 0.02)
    # matches at length 1 (B = 9 pairs) but none at length 2 (A = 0): NaN, as with no matches
    _, a, b = _sampen_c.calculate(_x()[:40], 6, 0.02, True)
    assert a[1] == 0 and b[1] == 9
    assert np.isnan(o['sampen1'])
    assert np.isnan(o['sampen2']) and np.isnan(o['sampen3'])
    # sampen0 is NaN when no two values are within r
    assert np.isnan(sample_entropy(np.arange(6.0), 2, 0.5)['sampen0'])


def _counts(y, k, r):
    """Template-match counts of length k + 1 and k (independent of the C code)."""
    n = len(y)
    pairs = [(i, j) for i in range(n - k) for j in range(i + 1, n - k)]
    match = lambda i, j, L: all(abs(y[i + t] - y[j + t]) < r for t in range(L))
    return sum(match(i, j, k + 1) for i, j in pairs), sum(match(i, j, k) for i, j in pairs)


def test_multi_scale_entropy_refined_composite():
    # at each scale, -log of the match counts pooled over the coarse-graining offsets
    y = np.random.default_rng(3).standard_normal(250)
    o = multi_scale_entropy(y, range(1, 11), 2, 0.15)
    for s in (1, 4, 10):
        a = b = 0
        for off in range(s):
            nw = (len(y) - off) // s
            ai, bi = _counts(y[off:off + nw * s].reshape(nw, s).mean(axis=1), 2, 0.15)
            a, b = a + ai, b + bi
        assert o[f'sampen_s{s}'] == pytest.approx(-np.log(a / b), rel=1e-12)


def test_multi_scale_entropy_strict_summaries():
    # scales 8-10 have no matches of length m + 1 (NaN), so every summary is NaN
    o = multi_scale_entropy(_x()[:150], range(1, 11), 2, 0.15)
    assert not np.isnan(o['sampen_s7']) and np.isnan(o['sampen_s8'])
    for k in ['maxSampEn', 'maxScale', 'minSampEn', 'minScale', 'meanSampEn', 'stdSampEn',
              'cvSampEn', 'meanch', 'slope', 'slopeSE']:
        assert np.isnan(o[k])
    # a robust fit needs at least 3 scales
    o = multi_scale_entropy(_x(), [1, 2], 2, 0.15)
    assert np.isnan(o['slope']) and o['meanSampEn'] == pytest.approx(1.37086621228729, rel=1e-12)


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
