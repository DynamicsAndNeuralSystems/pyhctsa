"""The input transforms a config can request (hctsa's x, x_z, abs(x_z), diff(x_z), zscore(abs(x_z)), zscore(sign(x_z)),
zscore(BF_PreProcess(x_z,'decimate_ac1e')))."""
import numpy as np
import pandas as pd
import pytest
import yaml

from pyhctsa.calculator import (FeatureCalculator, _build_label, _preprocess_decorator, _standardise_inputs,
                                _transform_input)
from pyhctsa.operations import distribution
from pyhctsa.utils import decimate_ac1e, z_score

rng = np.random.default_rng(3)
X = np.cumsum(rng.standard_normal(400)) * 3.0 + 10.0  # not z-scored, mostly nonzero values
XZ = z_score(X)


def _mz(v):
    """MATLAB zscore: N-1 std."""
    return (v - v.mean()) / v.std(ddof=1)


def test_transforms_match_hctsa_inputs():
    np.testing.assert_allclose(_transform_input(X), X)                                   # x
    np.testing.assert_allclose(_transform_input(X, True), XZ)                            # x_z
    np.testing.assert_allclose(_transform_input(X, True, True), np.abs(XZ))              # abs(x_z)
    np.testing.assert_allclose(_transform_input(X, True, False, 'diff1'), np.diff(XZ))   # diff(x_z)
    np.testing.assert_allclose(_transform_input(X, True, False, 'zscore_abs'), _mz(np.abs(XZ)))
    np.testing.assert_allclose(_transform_input(X, True, False, 'zscore_sign'), _mz(np.sign(XZ)))
    np.testing.assert_allclose(_transform_input(X, True, False, 'decimate_ac1e'), decimate_ac1e(XZ))


def test_zscore_abs_is_not_abs_of_zscore():
    # SC_FluctAnal 'mag': the magnitudes are re-z-scored, so the mean is 0 (abs(x_z) has a positive mean)
    mag = _transform_input(X, True, False, 'zscore_abs')
    assert abs(mag.mean()) < 1e-12 and abs(mag.std(ddof=1) - 1) < 1e-12
    assert np.abs(XZ).mean() > 0.5
    # a constant result (the signs of a one-sided series) becomes zeros, as MATLAB's zscore
    np.testing.assert_array_equal(_transform_input(np.arange(1.0, 50.0), False, False, 'zscore_sign'), np.zeros(49))


def test_diff1_shortens_and_is_not_rezscored():
    d = _transform_input(X, True, False, 'diff1')
    assert d.size == X.size - 1
    assert abs(d.std(ddof=1) - 1) > 1e-3


def test_decorator_hands_over_transformed_series():
    seen = {}
    f = _preprocess_decorator(True, False, 'zscore_sign')(lambda x: seen.setdefault('x', x).size)
    assert f(X) == X.size
    np.testing.assert_allclose(seen['x'], _mz(np.sign(XZ)))
    with pytest.raises(ValueError):
        _preprocess_decorator(True, False, 'nope')


def test_labels():
    for pre, suffix in [('diff1', '_diff1'), ('zscore_abs', '_mag'), ('zscore_sign', '_sign'), ('decimate_ac1e', '_dec'),
                        ('detrend', '_lindt')]:
        assert _build_label('f', {'a': 2}, ['a'], True, False, pre) == f'f_2{suffix}'
    assert _build_label('f', {}, [], True, True, 'decimate_ac1e') == 'f_abs_dec'
    assert _build_label('f', {}, [], False, True, 'detrend') == 'f_raw_abs_lindt'


def test_calculator_applies_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'echo_input', lambda x, k=0: {'n': float(len(x)), 'mean': float(np.mean(x)),
                                                                     'std': float(np.std(x, ddof=1))}, raising=False)
    cfg = {'distribution': {'echo_input': {'base_name': 'echo', 'ordered_args': ['k'], 'configs': [
        {'zscore': True},
        {'zscore': True, 'abs': True},
        {'zscore': True, 'preprocess': 'diff1'},
        {'zscore': True, 'preprocess': 'zscore_abs', 'k': 2},
        {'zscore': True, 'preprocess': 'zscore_sign'},
        {'zscore': False, 'preprocess': 'diff1'},
        {'zscore': True, 'preprocess': 'detrend'},
    ]}}}
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    assert list(fc.feature_funcs) == ['echo', 'echo_abs', 'echo_diff1', 'echo_2_mag', 'echo_sign', 'echo_raw_diff1',
                                      'echo_lindt']
    out = fc.extract(X).iloc[0]
    assert out['echo_lindt.n'] == X.size
    np.testing.assert_allclose(out['echo_lindt.mean'], 0, atol=1e-12)
    np.testing.assert_allclose(out['echo_lindt.std'], 1)
    assert out['echo.n'] == X.size and out['echo_diff1.n'] == X.size - 1
    np.testing.assert_allclose(out['echo_abs.mean'], np.abs(XZ).mean())
    np.testing.assert_allclose(out['echo_2_mag.mean'], 0, atol=1e-12)
    np.testing.assert_allclose(out['echo_2_mag.std'], 1)
    np.testing.assert_allclose(out['echo_sign.std'], 1)
    np.testing.assert_allclose(out['echo_raw_diff1.mean'], np.diff(X).mean())


# --- accepted input containers --------------------------------------------------------------------

def _three_series(n=60):
    rng = np.random.default_rng(3)
    return [rng.standard_normal(n) for _ in range(3)]


@pytest.mark.parametrize("wrap", [list, tuple, np.asarray, pd.Series], ids=lambda f: f.__name__)
def test_list_of_series_in_any_container_is_one_series_each(wrap):
    xs = _three_series()
    out = _standardise_inputs([wrap(x) for x in xs])
    assert len(out) == 3
    for got, x in zip(out, xs):
        assert got.ndim == 1 and got.dtype == float
        np.testing.assert_array_equal(got, x)


def test_tuple_of_series_and_unequal_lengths():
    xs = [np.arange(5.0), np.arange(8.0), np.arange(3.0)]
    out = _standardise_inputs(tuple(pd.Series(x) for x in xs))
    assert [len(o) for o in out] == [5, 8, 3]


def test_single_pandas_series_is_one_series():
    x = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10))
    out = _standardise_inputs(x)
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], np.arange(10.0))


def test_inputs_are_not_modified_or_aliased_to_ints():
    x = np.arange(10)  # integer dtype
    out = _standardise_inputs(x)
    assert out[0].dtype == float and x.dtype != float


@pytest.mark.parametrize("data, message", [
    (pd.DataFrame(np.zeros((3, 4))), "DataFrame"),
    ([np.zeros((2, 5)), np.zeros((2, 5))], "not 1D"),
    ([1.0, 2.0, [3.0, 4.0]], "mixes numbers and sequences"),
    ([], "empty"),
    (np.array([1 + 2j, 3 + 0j, 2 - 1j]), "complex"),
    ([np.array(["a", "b", "c"])], "not numeric"),
])
def test_ambiguous_or_invalid_inputs_raise(data, message):
    with pytest.raises(ValueError, match=message):
        _standardise_inputs(data)


def test_tall_2d_array_warns_about_orientation():
    with pytest.warns(UserWarning, match="one series per row"):
        out = _standardise_inputs(np.zeros((6, 4)))
    assert len(out) == 6


def test_extract_gives_the_same_rows_for_arrays_and_pandas_series(tmp_path):
    config = tmp_path / "small.yaml"
    config.write_text("correlation:\n  autocorr:\n    base_name: ac\n    configs:\n"
                      "      - {tau: [1, 2], method: 'Fourier', zscore: True}\n"
                      "    ordered_args: ['tau']\n", encoding="utf-8")
    calc = FeatureCalculator(str(config))
    xs = _three_series()
    from_arrays = calc.extract(xs)
    from_series = calc.extract([pd.Series(x) for x in xs])
    assert from_arrays.shape == (3, 2)
    pd.testing.assert_frame_equal(from_arrays, from_series)
    single = calc.extract(pd.Series(xs[0]))
    np.testing.assert_array_equal(single.to_numpy()[0], from_arrays.to_numpy()[0])


def test_labels_pandas_cannot_index_fail_before_computing(tmp_path, monkeypatch):
    config = tmp_path / "small.yaml"
    config.write_text("correlation:\n  autocorr:\n    base_name: ac\n    configs:\n"
                      "      - {tau: 1, method: 'Fourier', zscore: True}\n"
                      "    ordered_args: ['tau']\n", encoding="utf-8")
    calc = FeatureCalculator(str(config))
    called = []
    monkeypatch.setattr("pyhctsa.calculator._extract_features_single_series",
                        lambda ts, funcs: called.append(1))
    with pytest.raises((TypeError, ValueError)):
        calc.extract(_three_series(), labels=[("s", 1), ("s", 2), ("s", 3)])
    assert not called


# --- warnings -------------------------------------------------------------------------------------

def test_building_and_running_the_default_set_leaves_warning_filters_alone():
    import warnings
    before = list(warnings.filters)
    calc = FeatureCalculator()
    assert warnings.filters == before
    funcs = dict(list(calc.feature_funcs.items())[:20])
    calc.feature_funcs = funcs
    calc.extract(_three_series(100)[0])
    assert warnings.filters == before
    with pytest.warns(RuntimeWarning):  # the user's own warnings still show
        np.log(np.array([-1.0]))


def test_features_skipped_for_a_missing_dependency_are_reported_as_a_warning(tmp_path):
    config = tmp_path / "small.yaml"
    config.write_text("correlation:\n  autocorr:\n    base_name: ac\n"
                      "    dependencies: [a_package_that_does_not_exist]\n    configs:\n"
                      "      - {tau: 1, method: 'Fourier', zscore: True}\n"
                      "    ordered_args: ['tau']\n"
                      "  first_crossing:\n    base_name: fc\n    configs:\n"
                      "      - {corr_fun: 'ac', threshold: 0, what_out: 'discrete', zscore: True}\n",
                      encoding="utf-8")
    with pytest.warns(UserWarning, match="a_package_that_does_not_exist"):
        calc = FeatureCalculator(str(config))
    assert list(calc.feature_funcs) and all(not f.startswith("ac") for f in calc.feature_funcs)
