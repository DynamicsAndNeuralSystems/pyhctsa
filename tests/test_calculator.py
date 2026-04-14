import os
import yaml

import numpy as np
import pandas as pd
import pytest

from pyhctsa.calculator import FeatureCalculator, classify_output, _standardise_inputs, _apply_selection_wrapper
from pyhctsa.utils import get_dataset

# 1. Time-series feature module tests
class TestOperations:
    @pytest.mark.parametrize("x", [
        "medical", "extreme_events", "criticality", "correlation", "information", "entropy",
        "stationarity", "distribution", "scaling", "symbolic", "wavelet", 
        "hypothesis", "spectral", "model_fit", "graph", "physics", "pre_process",
        "surrogates"])
    def test_module_basic(self, x):
        # basic checks on each module
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

# 2. Test the calculator
class TestCalculator:
    # test the default configuration
    def test_calculator_defaults(self):
        calc = FeatureCalculator()
        # check that default feature set is loaded
        assert len(calc.config) > 0, "Expected a non-zero number of modules loaded into the FeatureCalculator."
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyhctsa", "configurations", "hctsa.yaml")
        # load yaml for comparison
        with open(config_path, 'rb') as f:
            hctsa_yaml = yaml.safe_load(f)
        assert len(hctsa_yaml) == len(calc.config), f"Expected {len(hctsa_yaml)} modules to be loaded, got {len(hctsa_yaml)} instead."
    # test the loading of a single module e.g. correlation
    def test_calculator_custom(self):
        confpath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyhctsa", "configurations", "module_configs", "correlation.yaml")
        calc = FeatureCalculator(config_path=confpath)
        assert len(calc.config) == 1, f"Expected only a single module, got {len(calc.config)} instead."
    # test sucessful instantiation of calculator
    def test_instantiation(self):
        calc = FeatureCalculator()
        assert isinstance(calc, FeatureCalculator), "Failed to instantiate FeatureCalculator object."
    # test basic feature extraction
    def test_basic_feature_extraction(self):
        data = get_dataset(which='sinusoid')
        calc = FeatureCalculator()
        result = calc.extract(data)
        assert result is not None
        assert isinstance(result, pd.DataFrame), "Expected a DataFrame to be returned."
        assert result.shape[0] == 1, "Expected a single row as output."
        assert result.shape[1] > 0, "Expected multiple columns as output."
    # test label functionality
    def test_output_labelling(self):
        data = get_dataset(which='sinusoid')
        calc = FeatureCalculator()
        result_ls = calc.extract(data, labels=['ts1']) # assign as a list
        assert result_ls.index.to_list() == ['ts1'], "mismatch between labels in DataFrame and assigned labels (list)."
        result_str = calc.extract(data, labels='ts1') # assign as a str
        assert result_str.index.to_list() == ['ts1'], "mismatch between labels in DataFrame and assigned labels (str)."
        result_int = calc.extract(data, labels=1) # assign as an integer
        assert result_int.index.tolist()[0] == 1, "mismatch between labels in DataFrame and assigned labels (int)."
        with pytest.raises(ValueError) as excinfo:
            # fewer labels than series
            labels1 = ['ts1']
            data2 = [get_dataset(which='sinusoid')[0], get_dataset(which='noise')[0]]
            calc.extract(data2, labels=labels1)
        assert f"Length of labels ({len(labels1)}) must equal the number of series ({len(data2)})" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            # more labels than series
            labels2 = ['ts1', 'ts2']
            data1 = get_dataset(which='sinusoid')
            calc.extract(data1, labels=labels2)
        assert f"Length of labels ({len(labels2)}) must equal the number of series ({len(data1)})" in str(excinfo.value)
        # multiple series
        data2 = [get_dataset(which='sinusoid')[0], get_dataset(which='noise')[0]]
        labels2 = ['ts1', 'ts2']
        result2 = calc.extract(data2, labels=labels2)
        assert result2.index.to_list() == labels2, "mismatch between labels in DataFrame and assigned labels for two series."
    # test the handling of invalid input data to the calculator
    def test_invalid_data_shape(self):
        with pytest.raises(ValueError) as excinfo:
            # invalid data type (tensor)
            calc = FeatureCalculator()
            calc.extract(np.random.randn(3, 3, 3))
        assert "NumPy array must be 1D or 2D" in str(excinfo.value)
    # time series contains NaNs or special vals
    def test_nan_in_data(self):
        data = get_dataset(which='sinusoid')
        data[0][5] = np.nan
        with pytest.raises(ValueError) as excinfo:
            calc = FeatureCalculator()
            calc.extract(data)
        assert "One or more time series instances are invalid" in str(excinfo.value)
    def test_inf_in_data(self):
        data = get_dataset(which='sinusoid')
        data[0][5] = np.inf
        with pytest.raises(ValueError) as excinfo:
            calc = FeatureCalculator()
            calc.extract(data)
        assert "One or more time series instances are invalid" in str(excinfo.value)
    # data type is invalid e.g. dict
    def test_invalid_data_type(self):
        data = get_dataset(which='sinusoid')
        data_dict = {'ts1': data}
        with pytest.raises(ValueError) as excinfo:
            calc = FeatureCalculator()
            calc.extract(data_dict)
        assert "Input must be a 1D series, a list of 1D series, or a 2D array with shape (n_series, n_samples)" in str(excinfo.value)

# 2. Feature output classification tests
class TestOutputClassifier:
    def test_output_classification(self):
        test_res = [0.3, 12.3, np.nan, np.inf, -np.inf, None, 0.4j]
        expected_out = np.array([0, 0, 2, 3, 4, 6, 5])
        actual_out = []
        for i in test_res:
            r = classify_output(i)
            actual_out.append(r)
        actual_out = np.array(actual_out)
        assert (actual_out == expected_out).all()

# 3. Input data type handling checks
class TestStandardiseInputs:
    def test_numpy_arr(self):
        # try each input type
        d = get_dataset()
        dat = d[0]
        # numpy array, 1d
        np_input = np.array(dat)
        np_standardised = _standardise_inputs(np_input)
        assert isinstance(np_standardised, list)
        assert isinstance(np_standardised[0], np.ndarray), "Expected numpy array to be returned as numpy array"
    def test_numpy2d(self):
        # numpy array 2d
        np_input2d = np.random.randn(2,1000)
        np_2dstandardised = _standardise_inputs(np_input2d)
        assert isinstance(np_2dstandardised, list)
        assert len(np_2dstandardised) == 2
        assert isinstance(np_2dstandardised[0], np.ndarray), "Expected 2d numpy array to be returned as list of numpy arrays"
    def test_list(self):
        # as a list
        d = get_dataset()
        dat = d[0]
        list_standardised = _standardise_inputs(dat)
        assert isinstance(list_standardised, list)
        assert isinstance(list_standardised[0], np.ndarray), "Expected list to be returned as numpy array"
    def test_tuple(self):
        d = get_dataset()
        dat = d[0]
        test_tuple = tuple(dat)
        tuple_standardised = _standardise_inputs(test_tuple)
        assert isinstance(tuple_standardised, list)
        assert isinstance(tuple_standardised[0], np.ndarray), "Expected tuple to be returned as numpy array"

# 4. Test the selection wrapper for filtering features
class TestFeatureFiltering:
    def test_sel_wrapper(self):
        data = get_dataset(which='sinusoid')
        calc = FeatureCalculator()
        # pick a partialed function which returns a dict with multiple keys
        test_func = calc.feature_funcs['first_crossing_ac_0_both']
        filtered_func = _apply_selection_wrapper(test_func, filter_keys='firstCrossing')
        assert callable(filtered_func), "Expected the filtered function to be callable"
        # check the output
        modified_output = filtered_func(data[0])
        modified_output_keys = list(modified_output.keys())
        assert len(modified_output_keys) == 1, "Expected there to only be a single key in the output"
        assert modified_output_keys[0] == 'firstCrossing', "Expected isolated key to match specified key"
    def test_sel_wrapper_both(self):
        data = get_dataset(which='sinusoid')
        calc = FeatureCalculator()
        test_func = calc.feature_funcs['force_potential_dblwell_1_0p5_0p2']
        filtered_func_keep = _apply_selection_wrapper(test_func, filter_keys='pcross', keep=True)
        # check the output for a single key
        modified_output_keep = filtered_func_keep(data[0])
        modified_output_keys_keep = list(modified_output_keep.keys())
        assert len(modified_output_keys_keep) == 1, "Expected there to only be a single key in the output"
        assert modified_output_keys_keep[0] == 'pcross', "Expected isolated key to match specified key"
        # check discarding
        filtered_func_discard = _apply_selection_wrapper(test_func, filter_keys='pcross', keep=False)
        modified_output_discard = filtered_func_discard(data[0])
        modified_output_keys_discard = list(modified_output_discard.keys())
        assert len(modified_output_keys_discard) > 1
        assert 'pcross' not in modified_output_keys_discard

    
