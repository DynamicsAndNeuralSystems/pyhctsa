import numpy as np
import pytest
import os
from pathlib import Path

from pyhctsa.utils import get_dataset, z_score, check_optional_deps, validate_data, _load_csv

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
        assert check_optional_deps('numpy') is True
        assert check_optional_deps('test') is False
        assert check_optional_deps('jpype1') is True
    
# 4. Validate data checks
class TestValidateData:
    def test_data_too_short(self):
        dat = np.random.randn(10)
        assert validate_data(dat) is False, "Expected False to be returned"
    def test_data_constant(self):
        dat = np.ones(1000)
        assert validate_data(dat) is False, "Expected False to be returned"
    def test_data_nan(self):
        dat = np.random.randn(1000)
        dat[5] = np.nan
        assert validate_data(dat) is False, "Expected False to be returned"
    def test_data_inf(self):
        dat = np.random.randn(1000)
        dat[5] = np.inf
        assert validate_data(dat) is False, "Expected False to be returned"
