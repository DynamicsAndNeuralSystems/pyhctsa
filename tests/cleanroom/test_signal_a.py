"""Tests of the FFT autocorrelation, the running median and the spectral peak finder."""
import numpy as np
import pytest

from pyhctsa.operations.correlation import _acf_fft
from pyhctsa.operations.pre_process import _med_filt_1d
from pyhctsa.operations.spectral import _findpeaks


# ---------------------------------------------------------------- autocorrelation
def test_acf_fft_properties():
    from statsmodels.tsa.stattools import acf
    rs = np.random.RandomState(4)
    for n in [2, 3, 10, 100, 257]:
        y = np.cumsum(rs.randn(n)) + rs.randn(n)
        a = _acf_fft(y)
        assert a[0] == pytest.approx(1.0) and np.all(np.abs(a) <= 1 + 1e-12)
        np.testing.assert_allclose(a, acf(y, nlags=n - 1, fft=False, adjusted=False), atol=1e-12)
    assert _acf_fft([3.0, 5.0]).tolist() == pytest.approx([1.0, -0.5])
    assert np.all(np.isnan(_acf_fft(np.full(10, 0.1))))
    assert np.all(np.isnan(_acf_fft([1.0, np.nan, 2.0])))
    assert np.all(np.isnan(_acf_fft([1.0, np.inf, 2.0])))
    assert np.isnan(_acf_fft([4.0])).all()


# ---------------------------------------------------------------- medfilt1
def test_med_filt_1d_edges():
    assert _med_filt_1d([1, np.nan, 3, 4, 5, 2, 1], 4).tolist()[4:] == [3.5, 3, 1.5]
    assert _med_filt_1d([5.0, 6.0], 1).tolist() == [5.0, 6.0]
    assert _med_filt_1d([], 3).size == 0
    for bad in [0, -1]:
        with pytest.raises(ValueError):
            _med_filt_1d([1.0, 2.0], bad)
    with pytest.raises(ValueError):
        _med_filt_1d(np.ones((2, 2)), 3)


# ---------------------------------------------------------------- findpeaks
def test_findpeaks_known():
    assert _findpeaks([0, 1, 0, 2, 0, 3, 0], 2)[1].tolist() == [1, 5]  # distance exactly d is removed
    assert _findpeaks([1, 2, 1], 0)[1].tolist() == [1]
    h, loc = _findpeaks([0, 0, 0, 0])
    assert h.size == 0 and loc.size == 0
    with pytest.raises(ValueError):
        _findpeaks([0, 1, 0], 0, 'sideways')


