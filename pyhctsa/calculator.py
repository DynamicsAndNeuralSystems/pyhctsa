import importlib
import re
import time
from functools import partial
from itertools import product
from pathlib import Path
from typing import Union, Any, Callable
import logging

logger = logging.getLogger('pyhctsa')
logger.setLevel(logging.CRITICAL) # only log critical warnings by default
logger.addHandler(logging.NullHandler())

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .utils import _check_optional_deps, _preprocess_decorator, _validate_data
from .distribute import _extract_features_single_series

class RangeList(list):
    """A list produced by the ``!range`` YAML constructor.

    Behaves exactly like a plain ``list`` but remembers the ``(start, stop, step)``
    it was built from so it can be rendered compactly in feature labels.
    """
    def __init__(self, values, start, stop, step):
        super().__init__(values)
        self.start = start
        self.stop = stop
        self.step = step

def range_constructor(loader, node) -> RangeList:
    """Construct an (inclusive) range from a YAML config.

    Usage in YAML:
        !range [start, stop]          # integer step of 1, inclusive of stop
        !range [start, stop, step]    # custom step (int or float), inclusive of stop

    Examples:
        !range [1, 40]          -> [1, 2, ..., 40]
        !range [0, 0.95, 0.05]  -> [0.0, 0.05, ..., 0.95]
    """
    seq = loader.construct_sequence(node)
    if len(seq) == 2:
        start, stop = seq
        step = 1
    elif len(seq) == 3:
        start, stop, step = seq
    else:
        raise ValueError(
            "!range expects [start, stop] or [start, stop, step], "
            f"got {len(seq)} values"
        )
    if step == 0:
        raise ValueError("!range step must be non-zero")
    n = int(round((stop - start) / step)) + 1
    # round to mitigate floating-point drift (e.g. 0.30000000000000004)
    values = [round(start + i * step, 10) for i in range(n)]
    return RangeList(values, start, stop, step)
yaml.SafeLoader.add_constructor("!range", range_constructor)

#: Status codes used by :func:`classify_output` and ``FeatureCalculator.errors``.
OK, ERRORED, NAN, POSINF, NEGINF, COMPLEX, NONE = range(7)

def classify_output(res) -> int:
    """Classify a single feature value into a status code.

    Returns one of ``OK``, ``NAN``, ``POSINF``, ``NEGINF``, ``COMPLEX`` or
    ``NONE``. ``ERRORED`` is not returned here: a feature that raised
    contributes no value at all, and is marked separately from the error
    record collected during extraction.
    """
    if res is None:
        return NONE
    if isinstance(res, str):
        # a string is a legitimate feature value, not an error marker
        return OK
    try:
        arr = np.asarray(res)
    except Exception:
        return OK
    if arr.dtype == object or arr.size != 1:
        # array- and object-valued cells carry no single status
        return OK
    if np.iscomplexobj(arr):
        # non-zero imaginary component
        return COMPLEX
    try:
        if np.isnan(arr):
            return NAN
        if np.isposinf(arr):
            return POSINF
        if np.isneginf(arr):
            return NEGINF
    except TypeError:
        # dtype has no notion of NaN/Inf (e.g. datetime, str)
        return OK
    return OK

def _feature_label_for_column(column: str, labels) -> Union[str, None]:
    """Recover the feature label that produced a given DataFrame column.

    Inverts the naming scheme applied in
    :func:`pyhctsa.distribute._flatten_into`: ``name`` for a scalar,
    ``name.key`` for a dict entry and ``name_i`` for an inlined array element.
    An exact match wins, so that a label which itself ends in ``_<digits>``
    is not mistaken for an array element of a shorter label.
    """
    if column in labels:
        return column
    head = column.split(".", 1)[0]
    if head in labels:
        return head
    match = re.fullmatch(r"(.+)_\d+", column)
    if match and match.group(1) in labels:
        return match.group(1)
    return None

def _apply_selection_wrapper(func: Callable, filter_keys: Union[str, list[str]],
                             keep: bool = True) -> Callable:
    """
    Wraps a function to selectively filter keys from its dict output.

    If the wrapped function does not return a dict, the result is passed through unchanged.

    Parameters
    ----------
        func (Callable): The function to wrap.
        filter_keys (str or list[str]): Key or keys to keep or discard from the output dict.
        keep (bool): If True, only the specified keys are retained. If False, the specified
                    keys are discarded and all remaining keys are returned. Defaults to True.

    Returns
    -------
        Callable: Wrapped function with filtered dict output.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
    
        if isinstance(filter_keys, str):
            keys = [filter_keys]
        elif isinstance(filter_keys, list):
            keys = filter_keys
        else:
            raise ValueError(f'Unexpected type for filter_keys: {type(filter_keys)}. Must be str or list[str]')

        if isinstance(result, dict):
            missing = [k for k in keys if k not in result] # log all of the missing keys
            if missing:
                logger.info(f'Warning: time-series features for func {func} not found {missing}')
            if keep:
                return {k: result[k] for k in keys if k in result}
            else:
                # discard instead of keep
                return {k: v for k, v in result.items() if k not in keys}
        return result
    return wrapper

def _standardise_inputs(data) -> list[np.ndarray]:
     # standardise the input into a list of 1D float arrays
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [np.asarray(data, dtype=float)]
        elif data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                # notify the user to check that the shapes make sense
                logger.warning(f"Check that the shape of the 2D input is such "
                      f"that (n_series, n_samples). Got shape: {data.shape}")
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
   
def _format_range_bound(val) -> str:
    """Render a single !range bound for a label by dropping the decimal point.

    0 -> '0', 0.05 -> '005', 0.95 -> '095', 1.0 -> '1'.
    """
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    if isinstance(val, int):
        return str(val)
    return str(val).replace(".", "")

def _format_param_value(val, key=None) -> str:
    """ 
    Format parameter value for label: 
    - For bools: if True, return the key name.
    - For floats/ints: as before. 
    - For lists: if contiguous range, show as 'start_to_end', else join all values. 
    """
    # New Logic for Booleans
    if isinstance(val, bool):
        return key if val and key else ""

    # A !range with a known start/stop/step renders compactly as
    # 'start_step_stop', dropping the decimal point (e.g. 0 to 0.95 in
    # steps of 0.05 -> '0_005_095').
    if isinstance(val, RangeList):
        return "_".join(_format_range_bound(v) for v in (val.start, val.step, val.stop))

    if isinstance(val, list):
        # Check if it's a contiguous range
        if len(val) > 1 and all(isinstance(x, (int, float)) for x in val):
            diffs = [val[i+1] - val[i] for i in range(len(val)-1)]
            if all(d == 1 for d in diffs):
                # Pass key down for recursion if needed, though usually lists aren't bools
                return f"{_format_param_value(val[0])}_to_{_format_param_value(val[-1])}"
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
                if formatted:
                    parts.append(formatted)
    else:
        for k, v in combo_dict.items():
            formatted_v = _format_param_value(v, key=k)
            parts.append(formatted_v if isinstance(v, bool) else f"{k}{formatted_v}")
    # join base and parts, filter out empty strings
    label = "_".join([base_name] + [p for p in parts if p])
    # append suffix flags
    if not do_zscore:
        label += '_raw'
    if do_absval:
        label += '_abs'
    return label

class FeatureCalculator:
    """
    A configuration-driven feature extraction calculator for time series data.
    
    Attributes
    ----------
    config: dict
        The parsed YAML configuration containing feature definitions.
    feature_funcs: dict
        A dictionary mapping feature labels to their corresponding callable functions.
    
    Parameters
    ----------
    config_path: str or None, optional
        Path to the YAML configuration file. If None, uses the default 'hctsa.yaml'
        configuration file in the package configurations directory.
    modules: list[str] or None, optional
        Restrict the calculator to these modules (top-level keys of the
        configuration), e.g. ``['correlation', 'entropy']``. If None, every
        module in the configuration is used.

    Examples
    --------
    >>> fc = FeatureCalculator()  # Load default configuration
    >>> x = np.random.randn(1000)
    >>> df = fc.extract(x)

    >>> fc = FeatureCalculator(modules=['correlation'])  # correlation features only
    """
    def __init__(self, config_path: Union[str, None] = None,
                 modules: Union[list[str], None] = None):
        """
        Initialises a FeatureCalculator instance.

        Parameters
        ----------
        config_path : str or None, optional
            Path to the YAML configuration file. If None, uses the default configuration.
        modules : list[str] or None, optional
            Restrict the calculator to these modules (top-level keys of the
            configuration). If None, every module in the configuration is used.

        Raises
        ------
        ValueError
            If any name in `modules` is not a top-level key of the configuration.
        """
        # set the default config path
        if config_path is None:
            ROOT_DIR = Path(__file__).resolve().parent
            config_path = ROOT_DIR / "configurations" / "hctsa.yaml"
        self.config_path = config_path
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        if modules is not None:
            if isinstance(modules, str):
                modules = [modules]
            unknown = [m for m in modules if m not in self.config]
            if unknown:
                raise ValueError(
                    f"Unknown module(s) {unknown} for config '{config_path}'. "
                    f"Available modules: {sorted(self.config)}")
            self.config = {m: self.config[m] for m in modules}
        self.modules = modules
        self._operations_package = "pyhctsa.operations" # abs path
        self.feature_funcs = self._build_feature_funcs()
        print(f"Loaded {len(self.feature_funcs)} master operations.")
    
    def _check_deps(self, module_key, feature_name, config):
        raw_deps = config.get("dependencies", None)
        if not raw_deps:
            return True
        deps_to_check = [raw_deps] if isinstance(raw_deps, str) else raw_deps
        missing = [dep for dep in deps_to_check if not _check_optional_deps(dep)]
        if missing:
            full_name = f"{module_key}.{feature_name}"
            logger.info(f"Skipping function '{full_name}' - missing dependencies: {', '.join(missing)}")
            self._skipped_functions.append((full_name, missing))
            return False
        return True
    
    def _build_feature_funcs(self):
        feature_funcs = {}
        self._skipped_functions = []      
        for module_key in self.config.keys():

            try:
                module = importlib.import_module(f"{self._operations_package}.{module_key}")
            except ImportError as e:
                logger.warning(f"Failed to import module '{module_key}': {e}")
                # Skip all functions in this module since we can't import it
                for feature_name in self.config[module_key].keys():
                    self._skipped_functions.append((f"{module_key}.{feature_name}", ["import_error"]))
                continue

            # Process features from this module
            for feature_name, feature_config in self.config[module_key].items():
                if not self._check_deps(module_key, feature_name, feature_config):
                    continue
                op_func = getattr(module, feature_name)
                base_name = feature_config.get("base_name", feature_name)
                ordered_args = feature_config.get("ordered_args", [])
                
                for config_item in feature_config.get("configs", [{}]):
                    # extract and clean meta params
                    do_zscore = config_item.pop('zscore', False)
                    do_absval = config_item.pop('abs', False)
                    select_features = config_item.pop('select', None)
                    exclude_features = config_item.pop('exclude', None)
                    # setup base function
                    master_func = _preprocess_decorator(do_zscore, do_absval)(op_func)
                    
                    # standardise config_item into a list of combinations
                    # if config_item is empty, product(*) returns [()], allowing us to loop once
                    keys = list(config_item.keys())
                    values = [v if isinstance(v, list) else [v] for v in config_item.values()]
                    
                    for combo_values in product(*values):
                        combo_dict = dict(zip(keys, combo_values))
                        
                        # generate label and apply wrappers
                        label = _build_label(base_name, combo_dict, ordered_args, do_zscore, do_absval)
                        
                        final_func = partial(master_func, **combo_dict)
                        # filter out certain features if specified by the user with the select/include keys
                        if select_features:
                            final_func = _apply_selection_wrapper(final_func, filter_keys=select_features, keep=True)
                        elif exclude_features:
                            final_func = _apply_selection_wrapper(final_func, filter_keys=exclude_features, keep=False)
                        
                        feature_funcs[label] = final_func
        
        if self._skipped_functions:
            logger.info(f"Total functions skipped due to missing dependencies: {len(self._skipped_functions)}")
        return feature_funcs

    def extract(self, data: Union[ArrayLike, list[ArrayLike]],
                labels: Union[ArrayLike, list[ArrayLike], None] = None,
                verbose: bool = False, distributor: Any = None) -> pd.DataFrame:
        """
        Run the configured feature extractor over one or more time series and
        return a single tidy `pandas.DataFrame`.

        Parameters
        ----------
        data: ArrayLike or list[ArrayLike]
            The input time series data. Two forms are accepted:
            
            * **Single series**: a 1-D array-like of real values.
            * **Multiple series**: a 2-D array or list of 1-D array-likes.
        labels: ArrayLike, list, str, int, or None, optional
            Labels for each time series. The order of labels is assumed to match 
            the order of the time series as passed in the `data` argument.
        verbose: bool, optional
            Whether to show a progress bar. Honoured by both the sequential and
            the parallel path. Default is ``False``.
        distributor: object, optional
            Optional distributor for parallel computation, e.g.
            :class:`pyhctsa.distribute.LocalDistributor`.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per input series and one column per feature.
            The column set is determined by the configuration alone: a feature
            that raises yields ``NaN`` rather than a differently-shaped row. The
            failures themselves are recorded in ``self._error_messages`` and
            flagged as ``ERRORED`` in ``self._errors``.
        """
        series_list = _standardise_inputs(data)
        # check each time series to see if valid...
        is_valid = np.array([_validate_data(t) for t in series_list])
        invalid = np.argwhere(~is_valid)
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
                raise ValueError(f"Length of labels ({len(labels_list)}) must equal "
                                 f"the number of series ({len(series_list)}).")
        else:
            # default names
            n = len(series_list)
            labels_list = [f"ts_{i}" for i in range(1, n + 1)]

        print(f"Evaluating {len(self.feature_funcs)} partialed functions. Strap in!...")
        start_time = time.perf_counter()
        if distributor:
            # parallel execution
            results = distributor.map(
                _extract_features_single_series,
                series_list,
                progress=verbose,
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
                    results = []
                    for ts in series_list:
                        results.append(_extract_features_single_series(ts, self.feature_funcs))
                        progress.advance(task)
            else:
                results = [_extract_features_single_series(ts, self.feature_funcs)
                           for ts in series_list]

        elapsed = time.perf_counter() - start_time
        print(f"Feature extraction completed in {elapsed:.3f} seconds.")
        value_rows = [values for values, _ in results]
        error_rows = [errors for _, errors in results]
        df = pd.json_normalize(value_rows)
        # assign row names
        df.index = pd.Index(labels_list, name="instance")
        # metadata
        self._last_elapsed = elapsed
        self._errors = self._build_error_frame(df, error_rows)
        self._error_messages = {
            label: errors for label, errors in zip(labels_list, error_rows) if errors
        }
        if self._error_messages:
            n_failures = sum(len(e) for e in self._error_messages.values())
            print(f"{n_failures} feature evaluation(s) failed across "
                  f"{len(self._error_messages)} series; their columns are NaN. "
                  f"See <calculator>._error_messages for the reasons.")
            for label, errors in self._error_messages.items():
                for feature, message in errors.items():
                    logger.warning("[%s] %s: %s", label, feature, message)

        return df

    def _build_error_frame(self, df: pd.DataFrame, error_rows: list[dict]) -> pd.DataFrame:
        """Build the per-cell status frame aligned with ``df``.

        Every cell is classified by :func:`classify_output`; cells belonging to
        a feature that raised for that series are then overwritten with
        ``ERRORED``.
        """
        status = df.map(classify_output)
        if not any(error_rows):
            return status
        # group the produced columns, by position, under the label that made them
        labels = set(self.feature_funcs)
        positions_by_label: dict[str, list[int]] = {}
        for position, column in enumerate(df.columns):
            label = _feature_label_for_column(column, labels)
            if label is not None:
                positions_by_label.setdefault(label, []).append(position)
        for row, errors in enumerate(error_rows):
            for label in errors:
                for column in positions_by_label.get(label, ()):
                    status.iat[row, column] = ERRORED
        return status
