"""The sample standard deviation convention (utils._sample_std / _sample_var).

One observation gives NaN (the N-1 spread is undefined), unless a site states that its values are the
complete set of components of one object (``single=0.0``). For N >= 2 the helpers equal
``numpy.std/var(ddof=1)`` bitwise. Feature values below were generated from the cons-sample-std branch;
the ordinary-series values equal those of main 0fa9013, and the degenerate-input ones are the changes
the convention makes (main gave 0.0).
"""
import math

import numpy as np
import pytest

from pyhctsa.operations.changepoint import stepdetect
from pyhctsa.operations.model_fit import ar_fit
from pyhctsa.operations.stationarity import local_extrema
from pyhctsa.operations.symbolic import binary_stats
from pyhctsa.operations.wavelet import wpd_best_tree
from pyhctsa.utils import _sample_std, _sample_var, _zscore, z_score


def test_helpers_equal_numpy_bitwise_for_two_or_more():
    m = np.random.default_rng(3).standard_normal((7, 5))
    assert _sample_std(m) == np.std(m, ddof=1)
    assert _sample_var(m) == np.var(m, ddof=1)
    for ax in (0, 1):
        assert np.array_equal(_sample_std(m, axis=ax), np.std(m, axis=ax, ddof=1))
        assert np.array_equal(_sample_var(m, axis=ax), np.var(m, axis=ax, ddof=1))
    c = m + 1j * m[::-1]
    assert _sample_std(c) == np.std(c, ddof=1)


def test_one_and_zero_observations():
    assert math.isnan(_sample_std([2.5])) and math.isnan(_sample_var([2.5]))
    assert math.isnan(_sample_std([])) and math.isnan(_sample_var([]))
    assert np.isnan(_sample_std(np.ones((1, 4)), axis=0)).all()
    assert _sample_std(np.zeros((0, 3)), axis=0).shape == (3,)
    assert _sample_std([2.5], single=0.0) == 0.0 and _sample_var([2.5], single=0.0) == 0.0
    assert math.isnan(_sample_std([np.inf], single=0.0))  # a non-finite value is still NaN
    assert _sample_std([4.0, 4.0]) == 0.0                 # constant data: a real zero spread
    assert _zscore([5.0]).tolist() == [0.0]      # its contract: constant input -> zeros


def test_samples_of_one_are_nan():
    short = z_score(np.random.default_rng(6).standard_normal(100))
    out = local_extrema(short, 'l', 100)  # one window: one local maximum
    assert all(math.isnan(out[k]) for k in ('stdmax', 'stdmin', 'stdext'))
    assert out['meanmax'] == pytest.approx(0.8295744111995121, rel=1e-12)
    n5 = np.random.default_rng(8).standard_normal(5)
    out = binary_stats(n5, 'iqr')
    assert all(math.isnan(out[k]) for k in ('stdstretch0', 'stdstretch1', 'stdstretchdiff'))
    assert math.isnan(stepdetect(n5, 'l1pwc', 10)['cvstepint'])


def test_sample_of_many_unchanged():
    wn = z_score(np.random.default_rng(1).standard_normal(1000))
    out = local_extrema(wn, 'l', 100)
    assert out['stdmax'] == pytest.approx(0.5519510396313049, rel=1e-12)
    assert out['stdmin'] == pytest.approx(0.39435724155244006, rel=1e-12)
    assert out['stdext'] == pytest.approx(3.1379130495036955, rel=1e-12)


def test_components_of_one_object_keep_zero():
    wn = np.random.default_rng(1).standard_normal(1000)
    out = ar_fit(wn, 1, 8, 'sbc')  # an AR(1) is selected: one coefficient, one mode
    assert out['A1'] == pytest.approx(-0.013642281907338854, rel=1e-12)
    for k in ('stdA', 'stdabsLambda', 'stdper', 'stdtau', 'stdexctn'):
        assert out[k] == 0.0
    heavy = z_score(np.random.default_rng(5).standard_t(2, 1000))
    out = wpd_best_tree(heavy)  # the best tree is a single leaf
    assert out['numLeaves'] == 1 and out['stdDepth'] == 0.0
