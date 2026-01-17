import importlib
import time
from functools import partial
from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from tqdm import tqdm

from .utils import preprocess_decorator, validate_data, check_optional_deps

def range_constructor(loader, node) -> list:
    """Construct a range from a YAML config."""
    start, end = loader.construct_sequence(node)
    return list(range(start, end + 1))
yaml.SafeLoader.add_constructor("!range", range_constructor)

def unfold_results(res : list) -> dict:
    # unfold the results from an extraction
    feature_dict = dict()
    for k in res:
        if isinstance(res[k], dict):
            # unfold
            for i in res[k]:
                feature = f"{k}.{i}"
                feature_dict[feature] = res[k][i]
        else:
            feature_dict[k] = res[k]
    # make into a pandas dataframe
    return pd.DataFrame([feature_dict])

def classify_output(res) -> int:
    # classify the type of output
    if isinstance(res, str) and res.startswith("Error:"):
        return 1
    elif res is None:
        return 6
    elif np.iscomplexobj(res):
        # non-zero imaginary component
        return 5
    elif np.isnan(res):
        return 2
    elif np.isposinf(res):
        return 3
    elif np.isneginf(res):
        return 4
    else:
        return 0

def standardise_inputs(data) -> list[np.ndarray]:
     # standardize the input into a list of 1D float arrays
    if isinstance(data, pd.Series):
        return [np.asarray(data.to_numpy(), dtype=float)]
    elif isinstance(data, pd.DataFrame):
        return [np.asarray(r, dtype=float) for _, r in data.iterrows()]
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [np.asarray(data, dtype=float)]
        elif data.ndim == 2:
            return [np.asarray(row, dtype=float) for row in data]
        else:
            raise ValueError("NumPy array must be 1D or 2D.")
    # list/tuple/array-likes
    elif isinstance(data, (list, tuple)):
        # if it looks like a list of series, coerce each
        # otherwise treat as a single series.
        if len(data) > 0 and all(isinstance(ts, (list, tuple, np.ndarray, pd.Series)) for ts in data):
            out = []
            for ts in data:
                ts = ts.to_numpy() if isinstance(ts, pd.Series) else ts
                out.append(np.asarray(ts, dtype=float))
            return out
        # single series
        return [np.asarray(data, dtype=float)]
    else:
        raise ValueError(
        "Input must be a 1D series, a list of 1D series, a 2D array "
        "with shape (n_series, n_samples), or a pandas Series/DataFrame.")
  
def _format_param_value(val : Union[int, float, list]) -> str:
    """
    Format parameter value for label:
    - For floats/ints: as before.
    - For lists: if contiguous range, show as 'start_end', else join all values.
    """
    if isinstance(val, list):
        # Check if it's a contiguous range
        if len(val) > 1 and all(isinstance(x, (int, float)) for x in val):
            diffs = [val[i+1] - val[i] for i in range(len(val)-1)]
            if all(d == 1 for d in diffs):  # contiguous integer range
                return f"{_format_param_value(val[0])}_{_format_param_value(val[-1])}"
        # Otherwise, join all values
        return "_".join(_format_param_value(x) for x in val)
    if isinstance(val, float) or isinstance(val, int):
        if val < 0:
            return 'm' + _format_param_value(-val)
        elif val == int(val):
            return str(int(val))
        elif 0 < val < 1:
            return '0p' + str(val).split(".")[1].rstrip('0')
        else:
            return str(val).replace('.', 'p').rstrip('0').rstrip('p')
    return str(val)

class FeatureCalculator:
    def __init__(self, config_path : Union[str, None] = None):
        """
        Initialises a FeatureCalculator instance.  

        Parameters
        ----------
        configPath : str or None, optional
            Path to the YAML configuration file. If None, uses the default configuration.
        """
        # set the default config path
        if config_path is None:
            ROOT_DIR = Path(__file__).resolve().parent
            config_path = ROOT_DIR / "configurations" / "hctsa.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self._operations_package = "pyhctsa.operations" # abs path
        self.feature_funcs = self._build_feature_funcs()
        print(f"Loaded {len(self.feature_funcs)} master operations.")

    def _build_feature_funcs(self):
        feature_funcs = {}
        skipped_functions = []
        
        for module_key in self.config.keys():
            try:
                module = importlib.import_module(f"{self._operations_package}.{module_key}")
            except ImportError as e:
                print(f"Failed to import module '{module_key}': {e}")
                # Skip all functions in this module since we can't import it
                for feature_name in self.config[module_key].keys():
                    skipped_functions.append((f"{module_key}.{feature_name}", ["import_error"]))
                continue
            
            # Process features from this module
            for feature_name, feature_config in self.config[module_key].items():
                # Check if this specific function has dependencies
                function_dependencies = feature_config.get("dependencies", None)
                missing_deps = []
                
                if function_dependencies is not None:
                    # Handle both string and list of dependencies
                    if isinstance(function_dependencies, str):
                        deps_to_check = [function_dependencies]
                    elif isinstance(function_dependencies, list):
                        deps_to_check = function_dependencies
                    else:
                        deps_to_check = []
                    
                    # Check each dependency for this function
                    for dep in deps_to_check:
                        if not check_optional_deps(dep):
                            missing_deps.append(dep)
                    
                    if missing_deps:
                        print(f"Skipping function '{module_key}.{feature_name}' - missing dependencies: {', '.join(missing_deps)}")
                        skipped_functions.append((f"{module_key}.{feature_name}", missing_deps))
                        continue
                op_func = getattr(module, feature_name, None)
                if op_func is None:
                    continue
                base_name = feature_config.get("base_name", f"{module_key}_{feature_name}")
                ordered_args = feature_config.get("ordered_args", [])
                configs = feature_config.get("configs", [{}])
                if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                    # Check if zscore and abs vary
                    zscore_values = [conf.get("zscore", False) for conf in configs]
                    abs_values = [conf.get("abs", False) for conf in configs]
                    zscore_varies = len(set(zscore_values)) > 1
                    abs_varies = len(set(abs_values)) > 1
                    for conf in configs:
                        zscore = conf.pop("zscore", False) if "zscore" in conf else False
                        absval = conf.pop("abs", False) if "abs" in conf else False
                        if conf:
                            keys, values = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in conf.items()])
                            for combo in product(*values):
                                combo_dict = dict(zip(keys, combo))
                                label = base_name
                                if ordered_args:
                                    parts = []
                                    for arg in ordered_args:
                                        if arg in combo_dict:
                                            parts.append(_format_param_value(combo_dict[arg]))
                                    if parts:
                                        label += "_" + "_".join(parts)
                                else:
                                    label += "_" + "_".join(f"{k}{_format_param_value(v)}" for k, v in combo_dict.items())
                                # Only append "_raw" if zscore varies and zscore is False
                                if zscore_varies and not zscore:
                                    label += "_raw"
                                # Only append "_abs" if abs varies and abs is True
                                if abs_varies and absval:
                                    label += "_abs"
                                decorated_func = preprocess_decorator(zscore, absval)(op_func)
                                feature_funcs[label] = partial(decorated_func, **combo_dict)
                        else:
                            label = base_name
                            # Only append "_raw" if zscore varies and zscore is False
                            if zscore_varies and not zscore:
                                label += "_raw"
                            # Only append "_abs" if abs varies and abs is True
                            if abs_varies and absval:
                                label += "_abs"
                            decorated_func = preprocess_decorator(zscore, absval)(op_func)
                            feature_funcs[label] = decorated_func
                else:
                    zscore, absval = False, False
                    if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                        zscore = configs[0].pop("zscore", False)
                        absval = configs[0].pop("abs", False)
                    label = f"{module_key}_{feature_name}"
                    # If abs is explicitly set True on the single config, include suffix to make it explicit
                    if absval:
                        label += "_abs"
                    decorated_func = preprocess_decorator(zscore, absval)(op_func)
                    feature_funcs[label] = decorated_func
        
        # Store information about skipped functions for later reference
        self._skipped_functions = skipped_functions
        if skipped_functions:
            print(f"Total functions skipped due to missing dependencies: {len(skipped_functions)}")
        
        return feature_funcs

    def _extract_single(self, ts : ArrayLike):
        results = {}
        for name, func in self.feature_funcs.items():
            # for each partialed function
            try:
                results[name] = func(ts)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results
    
    def summary(self):
        """
        Generate a summary of the last feature extraction call. 
        Currently generates a summary for all instances.
        """
        # Check that extract has already been called. Otherwise return nothing...
        print(f"Time taken to compute {len(self.feature_funcs)} master operations: {self._last_elapsed:.4f} seconds.")
        codings = { "succesful" : 0, "fatal error(s)" : 1, "NaN(s)": 2, "+inf(s)": 3, "-inf(s)": 4, "complex": 5, "empty": 6}
        e_arr = self._errors.to_numpy()
        for c in codings:
            print(f"{c} : {np.sum(e_arr == codings[c])}")
            
        return e_arr
    
    def extract(self, data, verbose = True) -> pd.DataFrame:
        """
        Run the configured feature extractor over one or more time series and
        return a single tidy `pandas.DataFrame`.

        Parameters
        ----------
        data : ArrayLike
            The input time series data. Two forms are accepted:

            * **Single series**: a 1-D array-like of real values
            (e.g., list[float] or `np.ndarray` of shape ``(n_samples,)``).
            * **Multiple series**: an array-like of 1-D array-likes
            (e.g., list[np.ndarray] or `np.ndarray` of dtype=object), where
            each element is a 1-D real-valued series of shape ``(n_samples_i,)``.
        verbose : Bool, optional
            Whether to show a progress bar of the features being computed.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per input series and one column per computed
            feature. 

        Examples
        --------
        >>> fc = FeatureCalculator()
        >>> x = np.random.randn(1000)
        >>> df = fc.extract(x)
        Evaluating 128 partialed functions. Strap in!...
        Feature extraction completed in 0.237 seconds.
        >>> df.shape
        (1, 128)
        """
        series_list = standardise_inputs(data)
        isValid = np.array([validate_data(t) for t in series_list]) # check each time series to see if valid...
        invalid = np.argwhere(isValid == False)
        if invalid.size > 0:
            raise ValueError(f"One or more time series instances are invalid: {invalid.flatten()}")

        n_funcs = len(self.feature_funcs)
        print(f"Evaluating {n_funcs} partialed functions. Strap in!...")
        start_time = time.perf_counter()
        rows: list[dict] = []

        pbar = tqdm if verbose else (lambda x, **kwargs: x)

        for ts in pbar(series_list, desc="Instances", unit="instance"):
            row = {}
            for name, func in pbar(self.feature_funcs.items(), desc="Features", unit="feature", leave=False, colour="green"):
                try:
                    val = func(ts)
                    # flatten if the feature returns a dict
                    if isinstance(val, dict):
                        for k, v in val.items():
                            # check the output for quality
                            row[f"{name}.{k}"] = v
                    # flatten small 1D arrays: name_0, name_1, ...
                    elif isinstance(val, np.ndarray) and val.ndim == 1 and val.size <= 16:
                        for i, v in enumerate(val):
                            row[f"{name}_{i}"] = v
                    else:
                        row[name] = val
                except Exception as e:
                    row[name] = f"Error: {e}"
            rows.append(row)

        elapsed = time.perf_counter() - start_time
        print(f"Feature extraction completed in {elapsed:.3f} seconds.")
        df = pd.json_normalize(rows)
        # run output quality checks
        df_errs = df.map(lambda x: classify_output(x))
        self._last_elapsed = elapsed
        self._last_result = df
        self._errors = df_errs

        return df
