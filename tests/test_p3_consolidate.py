"""Output shapes and shared helpers after the consolidation pass (feature names stay 1:1 with hctsa)."""
import numpy as np
import yaml

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.operations import correlation, distribution, model_fit, nonlinearity, scaling
from pyhctsa.robust import bf_random, bf_remove_points
from pyhctsa.utils import dict_output, nan_outputs, z_score

rng = np.random.default_rng(7)
Y = np.zeros(600)
for i in range(1, 600):
    Y[i] = 0.6 * Y[i - 1] + rng.standard_normal()
Y = z_score(Y)


# ------------------------------------------------------------------------------
# autocorr: one lag gives a number
# ------------------------------------------------------------------------------
def test_autocorr_scalar_lag_is_a_float():
    for method in ('Fourier', 'TimeDomain', 'TimeDomainStat'):
        assert isinstance(correlation.autocorr(Y, 3, method), float)
        assert correlation.autocorr(Y, [3], method).shape == (1,)
    assert np.isnan(correlation.autocorr(Y, np.nan))
    assert correlation.autocorr(Y, []).shape == (600,)
    assert correlation.autocorr(Y, 3) == correlation.autocorr(Y, [1, 2, 3])[2]


def test_calculator_names_for_single_lags(module_config):
    fc = FeatureCalculator(module_config('correlation'))
    cols = [c for c in fc.feature_funcs if c.startswith('ac_')]
    df = fc.extract(Y, labels='s')[['ac_1', 'ac_100', 'ac_10_abs']]
    assert list(df.columns) == ['ac_1', 'ac_100', 'ac_10_abs'] and df.notna().all().all()
    assert not any(c.endswith('_0') for c in df.columns) and 'ac_1' in cols


# ------------------------------------------------------------------------------
# dict_output / nan_outputs: a failing function keeps its field names
# ------------------------------------------------------------------------------
@dict_output
def _needs_positive(y, k=2):
    if np.min(y) <= 0:
        return np.nan
    return {f'a{i}': float(np.mean(y)) for i in range(k)}


@dict_output
def _never_works(y):
    return np.nan


def test_dict_output_gives_nan_fields():
    assert _needs_positive(np.ones(50)) == {'a0': 1.0, 'a1': 1.0}
    out = _needs_positive(-np.ones(50), k=3)  # the field names follow the arguments
    assert list(out) == ['a0', 'a1', 'a2'] and all(np.isnan(v) for v in out.values())
    assert np.isnan(_never_works(Y)) and not isinstance(_never_works(Y), dict)  # nothing to learn the names from
    assert np.isnan(nan_outputs(lambda y: 1.0))  # not a dict-valued function


def test_failed_functions_return_all_fields():
    neg = Y  # (zero-mean data: the exponential and Rayleigh fits to the distribution cannot be made)
    ok = distribution.compare_ks_fit(np.exp(Y), 'exp')
    bad = distribution.compare_ks_fit(neg, 'exp')
    assert isinstance(bad, dict) and list(bad) == list(ok) and all(np.isnan(v) for v in bad.values())
    # a series on which the dimension estimates cannot be made still gives the full set of fields
    for f in (nonlinearity.dimensions, nonlinearity.gp_corr_sum):
        out = f(np.tile([1.0, -1.0, 2.0], 100))
        assert isinstance(out, dict) and len(out) > 5


def test_calculator_decimation_failure_keeps_names(tmp_path):
    cfg = {'spectral': {'cepstrum': {'base_name': 'cepstrum', 'configs': [
        {'zscore': True, 'preprocess': 'decimate_ac1e'}, {'zscore': True}], 'ordered_args': []}}}
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump(cfg))
    row = FeatureCalculator(str(path)).extract(np.linspace(0, 1, 400) + 0.01 * np.sin(np.arange(400)))
    dec = [c for c in row.columns if '_dec.' in c]
    assert dec and all(np.isnan(row[c].iloc[0]) for c in dec)  # (a trend never falls to 1/e)
    assert not any(c.endswith('_dec') for c in row.columns)


# ------------------------------------------------------------------------------
# failures that are genuinely inappropriate inputs give NaN, not an exception
# ------------------------------------------------------------------------------
def test_short_series_give_nan_outputs():
    for order, n in (('best', 25), (2, 10)):  # (too few samples for the model, or for training it on half)
        out = model_fit.state_space_n4sid(Y[:n], order)
        assert isinstance(out, dict) and all(np.isnan(v) for v in out.values())
    assert np.isnan(correlation.nonlinear_autocorr(np.arange(5.0), [0, 3, 6]))
    assert np.isnan(scaling.fast_dfa(np.arange(5.0)))


def test_falling_sticks_near_constant_angles_do_not_raise():
    assert isinstance(correlation.falling_sticks(np.exp(np.random.RandomState(1).randn(500))), dict)


# ------------------------------------------------------------------------------
# bf_remove_points (hctsa BF_RemovePoints): the random ordering is bf_random's permutation
# ------------------------------------------------------------------------------
def test_remove_points_random_uses_bf_random():
    keep = np.sort(bf_random(600, 5, 'perm')[:420] - 1)
    np.testing.assert_array_equal(bf_remove_points(Y, 'random', 0.3, 'remove', 5), Y[keep])
    np.testing.assert_array_equal(bf_remove_points(Y, 'random', 0.3, 'remove'), bf_remove_points(Y, 'random', 0.3, 'remove', 'default'))
    assert list(distribution.remove_points(Y, 'absfar', 0.1)) == ['mean', 'median', 'std', 'skewnessdiff', 'kurtosisrat']
