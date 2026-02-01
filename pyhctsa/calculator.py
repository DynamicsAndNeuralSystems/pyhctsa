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
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .utils import check_optional_deps, preprocess_decorator, validate_data
from .distribute import _compute_features_for_chunk, _extract_features_single_series

def range_constructor(loader, node) -> list:
    """Construct a range from a YAML config."""
    start, end = loader.construct_sequence(node)
    return list(range(start, end + 1))
yaml.SafeLoader.add_constructor("!range", range_constructor)

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

def apply_selection_wrapper(func, keep_keys):
    """
    Wraps a function to filter its output dictionary to specific keys.
    
    :param func: Original function
    :param keep_keys: Features to keep as keys in a dict.
    :return: Wrapped function.
    :rtype: Any
    """
    keys_list = [keep_keys] if isinstance(keep_keys, str) else keep_keys
    
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # if result is a dict, then filter it according to the keys
        if isinstance(result, dict):
            return {k: result[k] for k in keep_keys if k in result}
        return result
    return wrapper

def _standardise_inputs(data) -> list[np.ndarray]:
     # standardize the input into a list of 1D float arrays
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [np.asarray(data, dtype=float)]
        elif data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                # notify the user to check that the shapes make sense
                print(f"Check that the shape of the 2D input is such that (n_series, n_samples). Got shape: {data.shape}")
            return [np.asarray(row, dtype=float) for row in data]
        else:
            raise ValueError("NumPy array must be 1D or 2D.")
    # list/tuple/array-likes
    elif isinstance(data, (list, tuple)):
        # if it looks like a list of series, coerce each
        # otherwise treat as a single series.
        if len(data) > 0 and all(isinstance(ts, (list, np.ndarray)) for ts in data):
            out = []
            for ts in data:
                out.append(np.asarray(ts, dtype=float))
            return out
        # single series
        return [np.asarray(data, dtype=float)]
    else:
        raise ValueError(
        "Input must be a 1D series, a list of 1D series, or a 2D array "
        "with shape (n_series, n_samples)")
      
def _format_param_value(val, key=None) -> str: 
    """ 
    Format parameter value for label: 
    - For bools: if True, return the key name.
    - For floats/ints: as before. 
    - For lists: if contiguous range, show as 'start_end', else join all values. 
    """
    # New Logic for Booleans
    if isinstance(val, bool):
        return key if val and key else ""

    if isinstance(val, list): 
        # Check if it's a contiguous range 
        if len(val) > 1 and all(isinstance(x, (int, float)) for x in val): 
            diffs = [val[i+1] - val[i] for i in range(len(val)-1)] 
            if all(d == 1 for d in diffs): 
                # Pass key down for recursion if needed, though usually lists aren't bools
                return f"{_format_param_value(val[0])}_{_format_param_value(val[-1])}" 
        return "_".join(_format_param_value(x) for x in val) 

    if isinstance(val, (float, int)): 
        if val < 0: 
            return 'm' + _format_param_value(-val) 
        elif val == int(val): 
            return str(int(val)) 
        elif 0 < val < 1: 
            return '0p' + str(val).split(".")[1].rstrip('0') 
        else: 
            return str(val).replace('.', 'p').rstrip('0').rstrip('p') 
            
    return str(val)

def _build_label(base_name, combo_dict, ordered_args, do_zscore, do_absval):
    """Constructs the feature string based on params and flags."""
    parts = []
    
    # Process parameters
    if ordered_args:
        for arg in ordered_args:
            if arg in combo_dict:
                formatted = _format_param_value(combo_dict[arg], key=arg)
                if formatted: parts.append(formatted)
    else:
        for k, v in combo_dict.items():
            formatted_v = _format_param_value(v, key=k)
            parts.append(formatted_v if isinstance(v, bool) else f"{k}{formatted_v}")
    
    # join base and parts, filter out empty strings
    label = "_".join([base_name] + [p for p in parts if p])
    
    # append suffix flags
    if not do_zscore: label += '_raw'
    if do_absval:    label += '_abs'
    return label

class FeatureCalculator:
    def __init__(self, config_path : Union[str, None] = None):
        """
        Initialises a FeatureCalculator instance.  

        Parameters
        ----------
        config_path : str or None, optional
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

    def _check_deps(self, module_key, feature_name, config):
        raw_deps = config.get("dependencies")
        if not raw_deps:
            return True
        deps_to_check = [raw_deps] if isinstance(raw_deps, str) else raw_deps
        missing = [dep for dep in deps_to_check if not check_optional_deps(dep)]
        if missing:
            full_name = f"{module_key}.{feature_name}"
            print(f"Skipping function '{full_name}' - missing dependencies: {', '.join(missing)}")
            self._skipped_functions.append((full_name, missing))
            return False
        
        return True
    
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
                op_func = getattr(module, feature_name)
                base_name = feature_config.get("base_name", feature_name)
                ordered_args = feature_config.get("ordered_args", [])
                
                for config_item in feature_config.get("configs", [{}]):
                    # extract and clean meta params
                    do_zscore = config_item.pop('zscore', False)
                    do_absval = config_item.pop('abs', False)
                    select_features = config_item.pop('_select', None)
                    
                    # setup base function
                    master_func = preprocess_decorator(do_zscore, do_absval)(op_func)
                    
                    # standardise config_item into a list of combinations
                    # if config_item is empty, product(*) returns [()], allowing us to loop once
                    keys = list(config_item.keys())
                    values = [v if isinstance(v, list) else [v] for v in config_item.values()]
                    
                    for combo_values in product(*values):
                        combo_dict = dict(zip(keys, combo_values))
                        
                        # generate label and apply wrappers
                        label = _build_label(base_name, combo_dict, ordered_args, do_zscore, do_absval)
                        
                        final_func = partial(master_func, **combo_dict)
                        if select_features:
                            final_func = apply_selection_wrapper(final_func, select_features)
                        
                        feature_funcs[label] = final_func
        
        # store information about skipped functions for later reference
        self._skipped_functions = skipped_functions
        if skipped_functions:
            print(f"Total functions skipped due to missing dependencies: {len(skipped_functions)}")
        
        return feature_funcs
    
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
    
    def extract(self, data : Union[ArrayLike, list[ArrayLike]], labels: Union[ArrayLike, list[ArrayLike], None] = None,
                verbose : bool = True, distributor = None) -> pd.DataFrame:
        """
        Run the configured feature extractor over one or more time series and
        return a single tidy `pandas.DataFrame`.

        Parameters
        ----------
        data : ArrayLike
            The input time series data. Two forms are accepted:

            * **Single series**: a 1-D array-like of real values
            (e.g., list[float] or `np.ndarray` of shape ``(n_samples,)``).
            * **Multiple series**: a 2-D `np.ndarray` of shape ``(n_series, n_samples)``
            or a list of 1-D array-likes (e.g., list[np.ndarray]), where
            each element is a 1-D real-valued series of shape ``(n_samples_i,)``.
        labels : array-like, list, str, int, or None, optional
            Labels for each time series. The order of labels is assumed to match the order 
            of the time series as passed in the `data` argument. Can be:
            
            * A single label (str or int) for a single series.
            * A list or array of labels, one per series.
            * None (default), in which case series are labeled as 'ts_1', 'ts_2', etc.
        verbose : bool, optional
            Whether to show a progress bar of the features being computed.
        distributor : object, optional
            Optional distributor for parallel computation. The distributor must
            implement a `map(func, iterable, **kwargs)` method that applies
            `func` to chunks of `iterable` (or to items) and returns a
            flattened list of results.

            Two implementations provided in this package are
            `pyhctsa.distribute.LocalDistributor` (uses `pathos.ProcessPool`)
            and `pyhctsa.distribute.DaskDistributor` (uses
            `dask.distributed`). If ``None`` (default), extraction runs
            sequentially in the current process. Distributor objects may also
            provide an optional `close()` method for cleaning up resources.
        
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
        series_list = _standardise_inputs(data)
        is_valid = np.array([validate_data(t) for t in series_list]) # check each time series to see if valid...
        invalid = np.argwhere(is_valid == False)
        if invalid.size > 0:
            raise ValueError(f"One or more time series instances are invalid: {invalid.flatten()}")
        
        # check labels if provided
        if labels is not None:
            # allow single label for single series, otherwise require one label per series
            if isinstance(labels, (str, int)):
                labels_list = [labels]
            else:
                labels_list = list(labels)

            if len(labels_list) != len(series_list):
                raise ValueError(f"Length of labels ({len(labels_list)}) must equal the number of series ({len(series_list)}).")
        else:
            # default names
            n = len(series_list)
            labels_list = [f"ts_{i}" for i in range(1, n + 1)]

        print(f"Evaluating {len(self.feature_funcs)} partialed functions. Strap in!...")
        start_time = time.perf_counter()
        
        if distributor:
            # parallel execution (local or cluster)
            rows = distributor.map(
                _compute_features_for_chunk,
                series_list, 
                feature_funcs=self.feature_funcs
            )
        else:
            # sequential fallback
            if verbose:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None), # 'None' makes it expand to full width
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("Sequential Extraction", total=len(series_list))
                    rows = []
                    for ts in series_list:
                        rows.append(_extract_features_single_series(ts, self.feature_funcs))
                        progress.advance(task)
            else:
                rows = [_extract_features_single_series(ts, self.feature_funcs) for ts in series_list]

        elapsed = time.perf_counter() - start_time
        print(f"Feature extraction completed in {elapsed:.3f} seconds.")
        df = pd.json_normalize(rows)
        # assign row names
        df.index = pd.Index(labels_list, name="instance")
        
        # meta data for summary
        self._last_elapsed = elapsed
        self._errors = df.map(classify_output)

        return df
