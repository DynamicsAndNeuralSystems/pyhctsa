import numpy as np
import pytest

from pyhctsa.utils import get_dataset, z_score

#----------------- Data loader tests ------------------

def test_get_dataset_default():
    # test default behaviour
    data = get_dataset()
    assert isinstance(data, list)
    assert len(data) == 1000
    assert all(isinstance(ts, list) for ts in data)
    assert all(isinstance(x, float) for ts in data for x in ts)

def test_get_data_e1000():
    # test whether the empirical1000 dataset can be loaded
    data = get_dataset(which="e1000")
    assert data, "Nothing returned"
    assert len(data) == 1000, "Expected list of length 1000"
    assert all(isinstance(ts, list) for ts in data)
    assert all(isinstance(x, float) for ts in data for x in ts)

def test_get_dataset_unknown():
    # test error thrown when dataset not found
    with pytest.raises(NotImplementedError):
        get_dataset("unknown_dataset")

#----------------- zscore function tests ------------------

def test_zscore_basic():
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

def test_zscore_constant_values():
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
def test_zscore_with_nonfinite(x):
    # check behaviour when nonfinite values (nan/infs) are passed into zscore
    with pytest.raises(ValueError):
        z_score(x)

def test_zscore_with_empty():
    # test behaviour when empty list/array is passed
    x = []
    with pytest.raises(ValueError):
        z_score(x)

#----------------- zscore function tests ------------------