"""The input transforms a config can request (hctsa's x, x_z, abs(x_z), diff(x_z), zscore(abs(x_z)), zscore(sign(x_z)),
zscore(BF_PreProcess(x_z,'decimate_ac1e')))."""
import numpy as np
import pytest
import yaml

from pyhctsa import calculator
from pyhctsa.calculator import FeatureCalculator, _build_label, _preprocess_decorator, _transform_input
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
    for pre, suffix in [('diff1', '_diff1'), ('zscore_abs', '_mag'), ('zscore_sign', '_sign'), ('decimate_ac1e', '_dec')]:
        assert _build_label('f', {'a': 2}, ['a'], True, False, pre) == f'f_2{suffix}'
    assert _build_label('f', {}, [], True, True, 'decimate_ac1e') == 'f_abs_dec'


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
    ]}}}
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    assert list(fc.feature_funcs) == ['echo', 'echo_abs', 'echo_diff1', 'echo_2_mag', 'echo_sign', 'echo_raw_diff1']
    out = fc.extract(X).iloc[0]
    assert out['echo.n'] == X.size and out['echo_diff1.n'] == X.size - 1
    np.testing.assert_allclose(out['echo_abs.mean'], np.abs(XZ).mean())
    np.testing.assert_allclose(out['echo_2_mag.mean'], 0, atol=1e-12)
    np.testing.assert_allclose(out['echo_2_mag.std'], 1)
    np.testing.assert_allclose(out['echo_sign.std'], 1)
    np.testing.assert_allclose(out['echo_raw_diff1.mean'], np.diff(X).mean())
