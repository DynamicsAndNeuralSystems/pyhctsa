from pyhctsa.Utilities.utils import get_dataset
from pyhctsa.FeatureCalculator.calculator import FeatureCalculator
import numpy as np
import os
import pytest

#----------------- High-level module tests ------------------
# does the calculator run on module functions (yes/no)?
@pytest.mark.parametrize("x", [
    "medical", "extreme", "criticality", "correlation", "information", "entropy",
    "stationarity", "distribution"])
def test_module_basic(x):
    # basic checks on medical module
    data = get_dataset(which="sinusoid")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pyhctsa", "Configurations", f"{x}.yaml")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    calc = FeatureCalculator(config_path)
    fvec = calc.extract(data)
    # Check that something is returned and it's not empty
    assert fvec is not None, "No output returned"
    assert isinstance(fvec, list), "Output should be a list (of feature dicts)"
    assert len(fvec) > 0, "Output list is empty"
    assert isinstance(fvec[0], dict), "Each element should be a dict of features"
    assert len(fvec[0]) > 0, "Feature dict is empty"
    # check output for multiple time series
    data2 = get_dataset("e1000")[1:3] # just the first 3 time-series instances
    fvec2 = calc.extract(data2)
    assert fvec2 is not None, "No output returned for multiple time-series input"
    assert isinstance(fvec2, list), "Output should be a list (of feature dicts)"
    assert len(fvec2) == len(data2)
