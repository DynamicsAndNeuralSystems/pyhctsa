import yaml
from functools import partial
import importlib
import numpy as np
import time
from numpy.typing import ArrayLike
from itertools import product
from ..Utilities.utils import preprocess_decorator

def range_constructor(loader, node):
    start, end = loader.construct_sequence(node)
    return list(range(start, end + 1))
yaml.SafeLoader.add_constructor("!range", range_constructor)

def _format_param_value(val):
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
    def __init__(self, configPath):
        with open(configPath) as f:
            self.config = yaml.safe_load(f)
        self.operations_package = "pyhctsa.Operations" # abs path
        self.feature_funcs = self._build_feature_funcs()

    def _build_feature_funcs(self):
        feature_funcs = {}
        for module_key in self.config.keys():
            # Dynamically import the module based on the config key
            module = importlib.import_module(f"{self.operations_package}.{module_key}")
            for feature_name, feature_config in self.config[module_key].items():
                op_func = getattr(module, feature_name, None)
                if op_func is None:
                    continue
                base_name = feature_config.get("base_name", f"{module_key}_{feature_name}")
                ordered_args = feature_config.get("ordered_args", [])
                configs = feature_config.get("configs", [{}])
                if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                    for conf in configs:
                        zscore = conf.pop("zscore", False) if "zscore" in conf else False
                        absval = conf.pop("abs", False) if "abs" in conf else False
                        label = f"{module_key}_{feature_name}"
                        if conf:
                            keys, values = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in conf.items()])
                            for combo in product(*values):
                                combo_dict = dict(zip(keys, combo))
                                if ordered_args:
                                    label = base_name + "_" + "_".join(
                                        _format_param_value(combo_dict[arg]) for arg in ordered_args)
                                else:
                                    label = base_name + "_" + "_".join(f"{k}{v}" for k, v in combo_dict.items())
                                decorated_func = preprocess_decorator(zscore, absval)(op_func)
                                feature_funcs[label] = partial(decorated_func, **combo_dict)
                        else:
                            label = base_name
                            decorated_func = preprocess_decorator(zscore, absval)(op_func)
                            feature_funcs[label] = decorated_func
                else:
                    zscore, absval = False, False
                    if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                        zscore = configs[0].pop("zscore", False)
                        absval = configs[0].pop("abs", False)
                    label = f"{module_key}_{feature_name}"
                    decorated_func = preprocess_decorator(zscore, absval)(op_func)
                    feature_funcs[label] = decorated_func
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

    def extract(self, data : ArrayLike):
        # Single time series: 1D array or list of numbers
        print(f"Evaluating {len(self.feature_funcs)} partialed functions. Strap in!...")
        start_time = time.perf_counter()
        results = []
        if isinstance(data, (np.ndarray, list)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in data):
            results = self._extract_single(np.asarray(data, dtype=float))
        # List of time series: list/array of lists/arrays
        elif isinstance(data, (list, np.ndarray)) and all(isinstance(ts, (list, np.ndarray)) for ts in data):
            for ts in data:
                results.append(self._extract_single(np.asarray(ts, dtype=float)))
        else:
            raise ValueError("Input must be a 1D array-like (single time series) or a list of 1D array-likes (multiple time series).")
        elapsed = time.perf_counter() - start_time
        print(f"Feature extraction completed in {elapsed:.3f} seconds.")
        return results
    