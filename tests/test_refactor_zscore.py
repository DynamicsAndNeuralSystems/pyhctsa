"""utils._zscore(x, constant='zeros' | 'none') replaces the pair _zscore_or_zeros / _zscore_or_none.

The two old functions are copied verbatim below (from branch cons-sample-std at 8911753, utils.py)
and compared with the new helper on every kind of input the old contracts distinguished, on the
machine that runs the test, so this file has no platform-dependent pins.
"""
import numpy as np
import pytest

from pyhctsa.utils import _sample_std, _zscore


def _ref_zscore_or_zeros(x):
    # Verbatim from utils._zscore_or_zeros (cons-sample-std 8911753).
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a.copy()
    sd = float(_sample_std(a, single=0.0))  # one value is a constant input: zeros
    if sd == 0:
        sd = 1.0
    with np.errstate(invalid='ignore'):
        return (a - np.mean(a)) / sd


def _ref_zscore_or_none(x):
    # Verbatim from utils._zscore_or_none (cons-sample-std 8911753).
    sd = _sample_std(x)
    if not sd > 0:
        return None
    return (x - np.mean(x)) / sd


def _corpus():
    rng = np.random.default_rng(5)
    return {
        'normal': rng.standard_normal(300),
        'ar1': np.cumsum(rng.standard_normal(200)) * 0.1 + rng.standard_normal(200),
        'tiny_spread': 1e3 + 1e-12 * rng.standard_normal(50),
        'two_values': np.array([1.5, -2.0]),
        'constant': np.full(7, 2.5),
        'single': np.array([3.0]),
        'empty': np.array([]),
        'with_nan': np.array([1.0, np.nan, 2.0]),
        'with_inf': np.array([1.0, np.inf, 2.0]),
        'ints': np.arange(10),
    }


@pytest.mark.parametrize('name', list(_corpus()))
def test_zeros_matches_reference(name):
    x = _corpus()[name]
    got, ref = _zscore(x), _ref_zscore_or_zeros(x)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)  # bit-identical, NaN positions included


@pytest.mark.parametrize('name', list(_corpus()))
def test_none_matches_reference(name):
    x = np.asarray(_corpus()[name], dtype=float)
    with np.errstate(invalid='ignore'):
        got, ref = _zscore(x, constant='none'), _ref_zscore_or_none(x)
    if ref is None:
        assert got is None
    else:
        np.testing.assert_array_equal(got, ref)


def test_constant_rule_is_the_only_difference():
    # On data with spread the two rules agree bit for bit; they differ only on degenerate input.
    x = np.random.default_rng(9).standard_normal(100)
    np.testing.assert_array_equal(_zscore(x), _zscore(x, constant='none'))
    assert np.array_equal(_zscore(np.full(5, 1.0)), np.zeros(5))
    assert _zscore(np.full(5, 1.0), constant='none') is None
    assert _zscore([3.0]).tolist() == [0.0]
    assert _zscore([3.0], constant='none') is None
    assert np.isnan(_zscore([1.0, np.nan, 2.0])).all()
    assert _zscore([1.0, np.nan, 2.0], constant='none') is None
    assert _zscore([]).size == 0
    assert _zscore([], constant='none') is None


def test_bad_constant_rejected():
    with pytest.raises(ValueError, match="constant must be"):
        _zscore(np.ones(3), constant='nan')
