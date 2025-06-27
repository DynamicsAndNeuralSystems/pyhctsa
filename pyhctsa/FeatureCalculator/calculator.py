import yaml
from functools import partial
import importlib
from ..Operations import Medical
from itertools import product
from ..Utilities.utils import preprocess_decorator

class FeatureCalculator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.feature_funcs = self._build_feature_funcs()

    def _build_feature_funcs(self):
        feature_funcs = {}
        for feature_name, feature_config in self.config["Medical"].items():
            op_func = getattr(Medical, feature_name, None)
            print(op_func)
            if op_func is None:
                continue
            configs = feature_config.get("configs", [{}])
            if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                for conf in configs:
                   # extract the preprocessing flags
                    zscore = conf.pop("zscore", False) if "zscore" in conf else False
                    absval = conf.pop("abs", False) if "abs" in conf else False
                    label = f"{feature_name}"
                    if zscore:
                        label += "_zscoreTrue"
                    if absval:
                        label += "_absTrue"
                    # Handle empty conf (no additional params)
                    if conf:
                        keys, values = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in conf.items()])
                        for combo in product(*values):
                            combo_dict = dict(zip(keys, combo))
                            label_full = label
                            if combo_dict:
                                label_full += "_" + "_".join(f"{k}{v}" for k, v in combo_dict.items())
                            decorated_func = preprocess_decorator(zscore, absval)(op_func)
                            feature_funcs[label_full] = partial(decorated_func, **combo_dict)
                    else:
                        decorated_func = preprocess_decorator(zscore, absval)(op_func)
                        feature_funcs[label] = decorated_func
            else:
                # no configs i.e., no additional function params
                zscore, absval = False, False
                if isinstance(configs, list) and configs and isinstance(configs[0], dict):
                    zscore = configs[0].pop("zscore", False)
                    absval = configs[0].pop("abs", False)
                label = f"{feature_name}"
                if zscore:
                    label += "_zscoreTrue"
                if absval:
                    label += "_absTrue"
                decorated_func = preprocess_decorator(zscore, absval)(op_func)
                feature_funcs[label] = decorated_func
        return feature_funcs
    
    def extract(self, data):
        results = {}
        for name, func in self.feature_funcs.items():
            try:
                results[name] = func(data)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results
    