import os

import numpy as np
import pandas as pd
import pytest

from pyhctsa.calculator import FeatureCalculator, classify_output, _standardise_inputs
from pyhctsa.utils import get_dataset

#----------------- High-level module tests ------------------
# does the calculator run on module functions (yes/no)?
@pytest.mark.parametrize("x", [
    "medical", "extreme_events", "criticality", "correlation", "information", "entropy",
    "stationarity", "distribution", "scaling", "symbolic", "wavelet", 
    "hypothesis", "spectral", "model_fit", "graph", "physics", "pre_process",
    "surrogates", "nonlinear"])
def test_module_basic(x):
    # basic checks on medical module
    data = get_dataset(which="sinusoid")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pyhctsa", "configurations", "module_configs", f"{x}.yaml")
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

def test_output_classification():
    test_res = [0.3, 12.3, np.nan, np.inf, -np.inf]
    expected_out = np.array([0, 0, 2, 3, 4])
    actual_out = []
    for i in test_res:
        r = classify_output(i)
        actual_out.append(r)
    actual_out = np.array(actual_out)
    assert (actual_out == expected_out).all()

def test_standardise_inputs():
    # try each input type
    d = get_dataset()
    dat = d[0]
    # numpy array, 1d
    np_input = np.array(dat)
    np_standardised = _standardise_inputs(np_input)
    assert isinstance(np_standardised, list)
    assert isinstance(np_standardised[0], np.ndarray), "Expected numpy array to be returned as numpy array"
    # numpy array 2d
    np_input2d = np.random.randn(2,1000)
    np_2dstandardised = _standardise_inputs(np_input2d)
    assert isinstance(np_2dstandardised, list)
    assert len(np_2dstandardised) == 2
    assert isinstance(np_2dstandardised[0], np.ndarray), "Expected 2d numpy array to be returned as list of numpy arrays"
    # as a list
    list_standardised = _standardise_inputs(dat)
    assert isinstance(list_standardised, list)
    assert isinstance(list_standardised[0], np.ndarray), "Expected list to be returned as numpy array"
    test_tuple = tuple(dat)
    tuple_standardised = _standardise_inputs(test_tuple)
    assert isinstance(tuple_standardised, list)
    assert isinstance(tuple_standardised[0], np.ndarray), "Expected tuple to be returned as numpy array"

