"""Tests for utils.detrend_linear and the ``preprocess: detrend`` input of the calculator (the *_lindt
features)."""
import numpy as np
import pytest
from scipy.signal import detrend, lfilter

from pyhctsa.calculator import FeatureCalculator, _transform_input
from pyhctsa.operations.correlation import autocorr, first_crossing
from pyhctsa.utils import detrend_linear, z_score

N = 1000
T = np.linspace(0.0, 1.0, N)


def _ar1(phi, seed, n=N, burn=500):
    rng = np.random.default_rng(seed)
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + burn))[burn:]




# ---- detrend_linear ----
def test_detrend_linear_matches_scipy_and_removes_any_line():
    x = _ar1(0.8, 1) + 5.0 + 3.0 * T
    r = detrend_linear(x)
    np.testing.assert_allclose(r, detrend(x, type='linear'), atol=1e-9)
    assert abs(r.mean()) < 1e-10
    assert abs(np.polyfit(np.arange(N), r, 1)[0]) < 1e-12
    # adding a line to the input changes nothing in the output
    np.testing.assert_allclose(detrend_linear(x - 20.0 * T + 7.0), r, atol=1e-9)
    # short inputs: a line passes through any two points
    np.testing.assert_array_equal(detrend_linear([1.0, 5.0]), np.zeros(2))
    assert detrend_linear([]).size == 0


# ---- the calculator input ----
def test_detrend_input_is_the_rezscored_residual():
    x = _ar1(0.8, 2) + 3.0 * T
    xd = _transform_input(x, True, False, 'detrend')
    np.testing.assert_allclose(xd, z_score(detrend_linear(z_score(x))), atol=1e-10)
    assert abs(xd.mean()) < 1e-10 and abs(xd.std(ddof=1) - 1) < 1e-10
    # without zscore the residual is handed over as it is
    np.testing.assert_allclose(_transform_input(x, False, False, 'detrend'), detrend_linear(x), atol=1e-10)
    # abs comes after the detrend
    np.testing.assert_allclose(_transform_input(x, True, True, 'detrend'), np.abs(xd), atol=1e-10)


def test_detrend_input_is_undefined_for_a_line():
    assert np.isnan(_transform_input(np.arange(50.0), True, False, 'detrend'))   # an exact ramp
    assert np.isnan(_transform_input(np.ones(50), False, False, 'detrend'))      # a constant (z-scoring rejects it first)
    assert np.isnan(_transform_input(np.array([1.0, 2.0]), False, False, 'detrend'))
    # a ramp with the smallest noise above round-off is not a line
    x = np.arange(50.0) + 1e-9 * np.random.default_rng(0).standard_normal(50)
    assert isinstance(_transform_input(x, True, False, 'detrend'), np.ndarray)


def test_lindt_features_are_invariant_to_an_added_line_and_match_the_manual_route():
    base = _ar1(0.8, 3)
    fc = FeatureCalculator()
    fc.feature_funcs = {k: f for k, f in fc.feature_funcs.items()
                        if k in ('ac_1', 'ac_1_lindt', 'ac_10_lindt', 'first_crossing_ac_1/e_both_lindt', 'autocorr_shape_drown_lindt')}
    out = fc.extract([base, base + 4.0 * T - 2.0, base * 3.0 + 1.0])
    # the originals move with the trend, the _lindt forms do not
    assert abs(out['ac_1'].iloc[1] - out['ac_1'].iloc[0]) > 0.05
    dt_cols = [c for c in out.columns if c.split('.')[0].endswith('_lindt')]
    assert len(dt_cols) > 3
    for c in dt_cols:
        np.testing.assert_allclose(out[c].iloc[1], out[c].iloc[0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(out[c].iloc[2], out[c].iloc[0], rtol=1e-8, atol=1e-10)
    # and equal the original operation applied to the detrended, re-z-scored series
    xd = z_score(detrend_linear(z_score(base)))
    np.testing.assert_allclose(out['ac_10_lindt'].iloc[0], autocorr(xd, 10), atol=1e-10)
    np.testing.assert_allclose(out['first_crossing_ac_1/e_both_lindt.firstCrossing'].iloc[0],
                               first_crossing(xd, 'ac', '1/e', 'both')['firstCrossing'])


def test_lindt_features_of_a_line_are_nan_not_error():
    fc = FeatureCalculator()
    fc.feature_funcs = {k: f for k, f in fc.feature_funcs.items() if k in ('ac_1', 'ac_1_lindt')}
    out = fc.extract(np.arange(100.0))
    assert np.isnan(out['ac_1_lindt'].iloc[0])


def test_default_set_has_the_lindt_family_only():
    fc = FeatureCalculator()
    dt = sorted(k for k in fc.feature_funcs if k.endswith('_lindt'))
    assert len(dt) == 52
    stems = {k.split('_')[0] for k in dt}
    assert stems == {'ac', 'partial', 'first', 'autocorr', 'histogram'}
    for k in dt:   # every _lindt call has its original beside it
        assert k[:-len('_lindt')] in fc.feature_funcs, k
