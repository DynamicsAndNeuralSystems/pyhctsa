from pyhctsa.Utilities.utils import get_dataset
import pytest
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

def test_zscore():
    # basic test - is the data zero mean and unit variance
    
