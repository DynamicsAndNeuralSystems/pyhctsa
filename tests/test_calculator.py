from pyhctsa.Utilities.utils import get_dataset
from pyhctsa.FeatureCalculator.calculator import FeatureCalculator
import numpy as np
import os
import pandas as pd
import pytest

#----------------- High-level module tests ------------------
# does the calculator run on module functions (yes/no)?
@pytest.mark.parametrize("x", [
    "medical", "extreme", "criticality", "correlation", "information", "entropy",
    "stationarity", "distribution", "scaling", "symbolic", "wavelet", 
    "hypothesis", "spectral", "modelfit", "graph", "physics", "preprocess",
    "surrogates"])
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
    assert isinstance(fvec, pd.DataFrame),  "Output should be a dataframe."

    # check output for multiple time series
    data2 = get_dataset("e1000")[0:3] # just the first 3 time-series instances
    fvec2 = calc.extract(data2)
    assert fvec2 is not None, "No output returned for multiple time-series input"
    assert isinstance(fvec2, pd.DataFrame), "Output should be a list (of feature dicts)"
    assert fvec2.shape[0] == 3, "Expected three rows of features for three time series."
