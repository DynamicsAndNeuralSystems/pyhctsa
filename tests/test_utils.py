import numpy as np
import pytest
import os
from pathlib import Path

from pyhctsa.utils import get_dataset, z_score, _check_optional_deps, _validate_data, sign_change
import pyhctsa.utils

# 1. Dataset loading tests
class TestDataLoader:
    def test_get_dataset_default(self):
        # test default behaviour
        data = get_dataset()
        assert isinstance(data, list)
        assert len(data) == 1000
        assert all(isinstance(ts, np.ndarray) for ts in data)
        assert all(isinstance(x, float) for ts in data for x in ts)

    def test_get_data_e1000(self):
        # test whether the empirical1000 dataset can be loaded
        data = get_dataset(which="e1000")
        assert data, "Nothing returned"
        assert len(data) == 1000, "Expected list of length 1000"
        assert all(isinstance(ts, np.ndarray) for ts in data)
        assert all(isinstance(x, float) for ts in data for x in ts)

    def test_get_dataset_unknown(self):
        # test error thrown when dataset not found
        with pytest.raises(NotImplementedError):
            get_dataset("unknown_dataset")

# 2. z-score function tests
class TestZScoring:
    def test_zscore_basic(self):
        # basic test - is the data zero mean and unit variance
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        x2 = np.array([10, 20, 30, 14, 32, 4])
        z = z_score(x)
        z2 = z_score(x2)
        assert isinstance(z, np.ndarray) # check that numpy array is returned
        assert isinstance(z2, np.ndarray)
        np.testing.assert_almost_equal(np.mean(z), 0, decimal=7)
        np.testing.assert_almost_equal(np.mean(z2), 0, decimal=7)
        np.testing.assert_almost_equal(np.std(z, ddof=1), 1, decimal=7)
        np.testing.assert_almost_equal(np.std(z2, ddof=1), 1, decimal=7)

    def test_zscore_constant_values(self):
        # check behaviour when values are constant
        x = [7, 7, 7, 7]
        with pytest.raises(ValueError):
            z_score(x)
    
    def test_zscore_nonumeric(self):
        # check handling of non-numeric values
        x = ['val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7']
        with pytest.raises(TypeError) as excinfo:
            z_score(x)
        assert "Input cannot be converted to numeric array" in str(excinfo.value)

    @pytest.mark.parametrize("x", [
        [1, 2, np.nan, 4],
        [1, 2, -np.nan, 3],
        [1, np.inf, 10, 4, 5],
        [1, 2, 3, -np.inf, 9, 8, 11]
    ])
    def test_zscore_with_nonfinite(self, x):
        # check behaviour when nonfinite values (nan/infs) are passed into zscore
        with pytest.raises(ValueError):
            z_score(x)

    def test_zscore_with_empty(self):
        # test behaviour when empty list/array is passed
        x = []
        with pytest.raises(ValueError):
            z_score(x)

# 3. Optional dep checks
class TestOptionalDepChecks:
    def test_optional_dep_check_basic(self):
        assert _check_optional_deps('numpy') is True
        assert _check_optional_deps('test') is False
        assert _check_optional_deps('jpype1') is True
    
# 4. Validate data checks
class TestValidateData:
    def test_data_too_short(self):
        dat = np.random.randn(10)
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_constant(self):
        dat = np.ones(1000)
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_nan(self):
        dat = np.random.randn(1000)
        dat[5] = np.nan
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_inf(self):
        dat = np.random.randn(1000)
        dat[5] = np.inf
        assert _validate_data(dat) is False, "Expected False to be returned"

# 5. Periphery function checks
class TestPeripheryFuncs:
    def test_sign_change_default(self):
        d1 = np.array([0.1, 2, 1, 5, -1, 2, 3, -1, -0.1, -2, 3])
        expected1 = np.array([False, False, False,  True,  True, False,
                              True, False, False,True])
        assert np.array_equal(sign_change(d1), expected1)
        d2 = np.array([1, 2, 3, 4])
        expected2 = np.array([False, False, False])
        assert np.array_equal(sign_change(d2), expected2)
    def test_sign_change_idx(self):
        d1 = np.array([0.1, 2, 1, 5, -1, 2, 3, -1, -0.1, -2, 3])
        expected1 = np.array([3, 4, 6, 9])
        assert np.array_equal(sign_change(d1, 1), expected1)
        d2 = np.array([1, 2, 3, 4])
        assert np.array_equal(sign_change(d2, 1), np.array([]))
    def test_histc(self):
        data = get_dataset(which='sinusoid')[0]
        bin_edges10 = np.linspace(data.min(), data.max(), 10)
        expected10 = np.array([218.,  94.,  79.,  72.,  70.,  74.,  79.,  97., 215.,   2.])
        res10 = pyhctsa.utils.histc(data, bin_edges10)
        assert np.array_equal(res10, expected10), "mismatch between expected counts and actual counts"
        assert len(res10) == 10, "Expected 10 counts for 10 bins"
        bin_edges20 = np.linspace(data.min(), data.max(), 20)
        expected20 = np.array([148.,  63.,  50.,  41.,  41.,  36.,  34.,  37.,  31.,  33.,  36.,
            31.,  37.,  37.,  38.,  45.,  51.,  63., 146.,   2.])
        res20 = pyhctsa.utils.histc(data, bin_edges20)
        assert np.array_equal(res20, expected20), "mismatch between expected counts and actual counts"
        assert len(res20) == 20, "Expected 20 counts for 20 bins"
    def test_preproc_decorator(self):
        simple_test_func = lambda x: sum(x)
        zs_true_abs_false = pyhctsa.utils._preprocess_decorator(True, False)
        zs_true_abs_true = pyhctsa.utils._preprocess_decorator(True, True)
        zs_false_abs_true = pyhctsa.utils._preprocess_decorator(False, True)
        d = np.array([1, -3, -4, -8, 2, 3, 4, 2, 45, 10, 8, 6])
        assert simple_test_func(d) == 66, "Unexpected output for simple test function"
        assert pytest.approx(0) == zs_true_abs_false(simple_test_func)(d), "Expected near 0 output"
        assert np.isclose(zs_true_abs_true(simple_test_func)(d), 6.997516651978067), "Expected near 6.9975 output"
        assert zs_false_abs_true(simple_test_func)(d) == 96, "Expected output to be 96"
    def test_point_of_crossing(self):
        ds = get_dataset(which='e1000')
        d = ds[0]
        fc_0p1, poc_0p1 = pyhctsa.utils.point_of_crossing(d, 0.1)
        assert fc_0p1 == 1, "Expected first crossing index to be 1."
        assert np.isclose(poc_0p1, 0.009535835889354775)
        fc_0p3, poc_0p3 = pyhctsa.utils.point_of_crossing(d, 0.3)
        assert fc_0p3 == 1, "Expected first crossing index to be 1."
        assert np.isclose(poc_0p3, 0.36128190336832033)
        d2 = ds[5]
        fc_0p9, poc_0p9 = pyhctsa.utils.point_of_crossing(d2, 0.9)
        assert fc_0p9 == 6, "Expected first crossing index to be 6."
        assert np.isclose(poc_0p9, 5.695366116499348)
    def test_make_buffer(self):
        d = get_dataset(which='sinusoid')[0]
        b3 = pyhctsa.utils.make_buffer(d, 3)
        assert b3.shape[0] == 333, "expected there to be 333 rows"
        assert b3.shape[1] == 3, "expected there to be 3 columns"
        b2 = pyhctsa.utils.make_buffer(d, 2)
        assert b2.shape[0] == 500, "expected there to be "
    def test_binarize(self):
        d = get_dataset(which='e1000')[0]
        # test defaults
        res_default = pyhctsa.utils.binarize(d)
        assert int(np.sum(res_default)) == 4961
        res_diff = pyhctsa.utils.binarize(d, binarize_how='mean')
        assert int(np.sum(res_diff)) == 4219
        res_median = pyhctsa.utils.binarize(d, binarize_how='median')
        assert int(np.sum(res_median)) == 5000
        res_iqr = pyhctsa.utils.binarize(d, binarize_how='iqr')
        assert int(np.sum(res_iqr)) == 5000
        with pytest.raises(ValueError):
            pyhctsa.utils.binarize(d, binarize_how='test')
    def test_xcorr(self):
        d1, d2 = get_dataset(which='e1000')[0], get_dataset(which='e1000')[1]
        # basic test, 5 lags
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, max_lags=5)
        assert out1[0] == -5
        assert out1[-1] == 5
        assert len(out2) == 11, "expected 11 values for lag 5"
        assert np.isclose(sum(out2), 8.025332010437447)
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, normed=False, max_lags=5)
        assert np.isclose(sum(out2), 7704.728115530612)
        # test invalid input i.e., mismtach in lengths
        with pytest.raises(ValueError) as excinfo:
            pyhctsa.utils.x_corr(d1, np.random.randn(100), max_lags=5)
        assert 'x and y must be equal length' in str(excinfo.value)
        # test invalid max lags
        with pytest.raises(ValueError) as excinfo2:
            pyhctsa.utils.x_corr(d1, d2, max_lags=-5)
        assert 'max_lags must be None or strictly positive' in str(excinfo2.value)
        # set max lags to none
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, max_lags=None)
        d1_len = len(d1)
        assert abs(out1[0]) == (d1_len - 1), "expected max lag to be nx - 1 where nx is length of data"
