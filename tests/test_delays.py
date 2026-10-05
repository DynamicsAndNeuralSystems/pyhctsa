"""Tests for the shared delay / Theiler-window / decimation helpers (hctsa BF_GetTau,
BF_TheilerWindow, BF_PreProcess 'decimate_ac1e').

Expected values were generated with current hctsa (MATLAB) on the same analytic series.
"""
import numpy as np
import pytest
import yaml

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.utils import (_preprocess_decorator, decimate_ac1e, get_tau, pre_process,
                           theiler_window, time_delay_embed, z_score)

T = np.arange(1000.0)


def _zs(y):
    return (y - y.mean()) / y.std(ddof=1)


def _logistic():
    x = np.zeros(1000)
    x[0] = 0.3
    for i in range(1, 1000):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return _zs(x)


SERIES = {
    'sine200': _zs(np.sin(2 * np.pi * T / 200)),
    'twosine': _zs(np.sin(2 * np.pi * T / 50) + 0.5 * np.sin(2 * np.pi * T / 13.7)),
    'am': _zs(np.sin(2 * np.pi * T / 90) * np.sin(2 * np.pi * 7 * T / 1000) + 0.3 * np.cos(2 * np.pi * T / 31)),
    'logistic': _logistic(),
}
# (ac, ac1e, mi, mi-gaussian, W['ac',1], W['ac',2.5], W['ac1e',3], decimated length)
EXPECTED = {
    'sine200': (52, 38, 2, 51, 52, 130, 114, 27),
    'twosine': (15, 6, 5, 14, 15, 38, 18, 167),
    'am': (21, 12, 12, 20, 21, 53, 36, 84),
    'logistic': (1, 1, 1, 3, 1, 3, 3, 1000),
}


@pytest.mark.parametrize('name', list(SERIES))
def test_matches_hctsa(name):
    y = SERIES[name]
    ac, ac1e, mi, mig, w_ac1, w_ac25, w_ac1e3, n_dec = EXPECTED[name]
    assert get_tau(y, 'ac') == ac
    assert get_tau(y, 'ac1e') == ac1e
    assert get_tau(y, 'mi') == mi
    assert get_tau(y, 'mi-gaussian') == mig
    assert theiler_window(y, ['ac', 1]) == w_ac1
    assert theiler_window(y, ('ac', 2.5)) == w_ac25
    assert theiler_window(y, ['ac1e', 3]) == w_ac1e3
    assert len(decimate_ac1e(y)) == n_dec


def test_integer_passthrough_and_errors():
    y = SERIES['sine200']
    assert get_tau(y, 3) == 3 and get_tau(y, np.int64(4)) == 4 and get_tau(y, 2.0) == 2
    with pytest.raises(ValueError):
        get_tau(y, 'bogus')
    with pytest.raises(ValueError):
        get_tau(y, 2.5)


def test_constant_series_gives_nan():
    y = np.ones(300)
    for rule in ('ac', 'ac1e', 'mi', 'mi-gaussian'):
        assert np.isnan(get_tau(y, rule))
    assert np.isnan(theiler_window(y, ['ac', 1]))
    assert np.isnan(theiler_window(y, ['ac1e', 3]))
    assert np.isnan(decimate_ac1e(y))
    assert theiler_window(y, 5) == 5  # numeric specs need no ACF


def test_theiler_numeric_specs():
    assert theiler_window(None, 7.5, N=100) == 8      # round half away from zero
    assert theiler_window(None, 2.5, N=100) == 3
    assert theiler_window(None, 0) == 0
    assert theiler_window(np.zeros(250), 0.1) == 25   # proportion of N
    assert theiler_window(None, 0.5, N=10) == 5
    assert theiler_window(None, 0.25, N=10) == 3      # 2.5 -> 3
    for bad in (-1, ['ac'], ['mi', 1], ['ac', -1], 'ac'):
        with pytest.raises(ValueError):
            theiler_window(SERIES['sine200'], bad)


def test_decimation_values():
    y = SERIES['sine200']
    d = pre_process(y, 'decimate_ac1e')
    np.testing.assert_array_equal(d, y[::38])
    dz = decimate_ac1e(y)
    assert abs(dz.mean()) < 1e-12 and abs(dz.std(ddof=1) - 1) < 1e-12
    # pre_process other cases
    np.testing.assert_allclose(pre_process(y, 'diff1'), np.diff(y))
    assert len(pre_process(y, 'rescale_tau')) == 1000 // 52
    assert pre_process(y, None) is not None
    with pytest.raises(ValueError):
        pre_process(y, 'nope')


def test_time_delay_embed_accepts_rule():
    y = SERIES['sine200']
    np.testing.assert_array_equal(time_delay_embed(y, 3, 'ac1e'), time_delay_embed(y, 3, 38))
    with pytest.raises(ValueError):
        time_delay_embed(np.ones(200), 3, 'ac')


def test_decorator_decimation():
    y = z_score(np.sin(2 * np.pi * T / 200) + 0.1 * np.cos(T))
    seen = {}
    f = _preprocess_decorator(True, False, 'decimate_ac1e')(lambda x: seen.setdefault('x', x) is x and len(x))
    assert f(y) == len(decimate_ac1e(y)) < len(y)
    np.testing.assert_allclose(seen['x'], decimate_ac1e(z_score(y)))
    # an undefined delay gives NaN instead of calling the operation
    h = _preprocess_decorator(False, False, 'decimate_ac1e')(lambda x: 1.0)
    assert np.isnan(h(np.ones(300)))
    with pytest.raises(ValueError):
        _preprocess_decorator(True, False, 'nope')


def test_calculator_preprocess_key_and_labels(tmp_path):
    cfg = {'scaling': {'fast_dfa': {'base_name': 'fast_dfa', 'configs': [
        {'zscore': True},
        {'zscore': True, 'preprocess': 'decimate_ac1e'},
        {'zscore': True, 'abs': True, 'preprocess': 'decimate_ac1e'},
    ]}}}
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    assert list(fc.feature_funcs) == ['fast_dfa', 'fast_dfa_dec', 'fast_dfa_abs_dec']
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.safe_dump({'scaling': {'fast_dfa': {'base_name': 'fast_dfa',
                                   'configs': [{'zscore': True, 'preprocess': 'nope'}]}}}))
    with pytest.raises(ValueError):
        FeatureCalculator(str(bad))
