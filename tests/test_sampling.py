"""Tests for pyhctsa.sampling: sampling_report and resample_dataset."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from pyhctsa.sampling import resample_dataset, sampling_report


def _ar1(phi, n, seed, burn=500):
    rng = np.random.default_rng(seed)
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + burn))[burn:]


def _has(warnings, text):
    return any(text in w for w in warnings)


def test_report_white_noise():
    y = np.random.default_rng(0).standard_normal(5000)
    rep = sampling_report(y)
    assert rep['n'] == 5000
    assert rep['tau_ac1e'] == 1
    assert rep['decimation_factor'] == 1
    assert rep['decimated_length'] == 5000
    assert abs(rep['acf1']) < 0.05
    assert np.isfinite(rep['eta'])
    assert _has(rep['warnings'], 'near-critically sampled')
    assert not _has(rep['warnings'], 'noise floor')
    assert 'fs' not in rep


def test_report_ar1():
    # ACF 0.95^k crosses 1/e at k = 19.5, so tau_ac1e ~ 19 and factor floor(19/3) = 6
    rep = sampling_report(_ar1(0.95, 5000, seed=6))
    assert 15 <= rep['tau_ac1e'] <= 25
    assert rep['tau_ac1e'] == 19  # this realisation
    assert rep['decimation_factor'] == rep['tau_ac1e'] // 3 == 6
    assert rep['decimated_length'] == 5000 // 6 == 833
    assert rep['tau_mi'] <= rep['tau_ac1e']
    assert rep['acf1'] > 0.9
    assert rep['eta'] > 10  # Toker et al.'s oversampling threshold
    assert rep['warnings'] == []


def test_report_target_tau():
    y = _ar1(0.95, 5000, seed=6)
    assert sampling_report(y, target_tau=1)['decimation_factor'] == 19
    assert sampling_report(y, target_tau=100)['decimation_factor'] == 1
    with pytest.raises(ValueError):
        sampling_report(y, target_tau=0)


def test_report_short_after_decimation():
    rep = sampling_report(_ar1(0.95, 2000, seed=6))
    assert rep['decimation_factor'] > 1
    assert rep['decimated_length'] < 500
    assert _has(rep['warnings'], 'would leave only')


def test_report_random_walk_disagreement():
    y = np.cumsum(np.random.default_rng(0).standard_normal(5000))
    rep = sampling_report(y)
    taus = [rep['tau_ac1e'], rep['tau_ac'], rep['tau_mi']]
    assert max(taus) / min(taus) > 10
    assert _has(rep['warnings'], 'delays disagree')


def test_report_noise_floor():
    # a slow sinusoid buried in white measurement noise: correlated at lag 1, below 1/e at lag 2
    rng = np.random.default_rng(1)
    y = np.sin(np.arange(3000) / 3.0) + rng.standard_normal(3000)
    rep = sampling_report(y)
    assert rep['tau_ac1e'] == 1 and rep['acf1'] > 0.2
    assert _has(rep['warnings'], 'noise floor')


def test_report_with_fs():
    fs = 250.0
    rep = sampling_report(_ar1(0.95, 5000, seed=6), fs=fs)
    assert rep['fs'] == fs
    assert rep['fs_suggested'] == pytest.approx(fs / rep['decimation_factor'])
    assert rep['nyquist_suggested'] == pytest.approx(rep['fs_suggested'] / 2)
    assert rep['tau_ac1e_seconds'] == pytest.approx(rep['tau_ac1e'] / fs)


def test_report_input_types():
    y = _ar1(0.9, 2000, seed=2)
    ref = sampling_report(y)
    for alt in (list(y), pd.Series(y)):
        rep = sampling_report(alt)
        for key in ('n', 'tau_ac1e', 'tau_ac', 'tau_mi', 'decimation_factor'):
            assert rep[key] == ref[key]


def test_report_constant_series_is_nan():
    rep = sampling_report(np.ones(200), fs=100.0)
    assert np.isnan(rep['tau_ac1e']) and np.isnan(rep['decimation_factor'])
    assert np.isnan(rep['fs_suggested'])
    assert _has(rep['warnings'], 'undefined')


def test_resample_two_lengths():
    rng = np.random.default_rng(3)
    data = [rng.standard_normal(1000), rng.standard_normal(601)]
    out = resample_dataset(data, 4)
    assert isinstance(out, list) and len(out) == 2
    assert [len(x) for x in out] == [250, 151]  # ceil(n / factor)


def test_resample_single_series_and_pandas():
    y = np.random.default_rng(4).standard_normal(800)
    out = resample_dataset(pd.Series(y), 2)
    assert len(out) == 1 and len(out[0]) == 400
    np.testing.assert_allclose(out[0], resample_dataset(y, 2)[0])


def test_resample_factor_one_passthrough():
    rng = np.random.default_rng(5)
    data = [rng.standard_normal(100), rng.standard_normal(57)]
    out = resample_dataset(data, 1)
    for x, y in zip(out, data):
        np.testing.assert_array_equal(x, y)


def test_resample_antialiasing():
    # a tone above the new Nyquist frequency is removed, one well below it is kept
    t = np.arange(4000)
    fast = np.sin(2 * np.pi * 0.2 * t)    # 0.2 cycles/sample > 0.5 / 4
    slow = np.sin(2 * np.pi * 0.02 * t)   # 0.02 cycles/sample < 0.5 / 4
    out_fast, out_slow = resample_dataset([fast, slow], 4)
    assert np.std(out_fast[50:-50]) < 0.01
    assert np.std(out_slow[50:-50]) == pytest.approx(np.std(slow), rel=0.05)


def test_resample_large_factor_staged():
    y = np.random.default_rng(6).standard_normal(5000)
    for factor in (26, 17, 100):
        out = resample_dataset(y, factor)[0]
        assert len(out) == -(-5000 // factor)
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize('factor', [0, -2, 2.5, np.nan, '3', True, None])
def test_resample_invalid_factor(factor):
    with pytest.raises(ValueError):
        resample_dataset(np.zeros(100), factor)
