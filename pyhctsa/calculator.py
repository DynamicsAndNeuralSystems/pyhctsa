import copy
import importlib
import time
from functools import partial, wraps
from itertools import product
from pathlib import Path
from typing import Union, Any, Callable
import logging
import warnings

logger = logging.getLogger('pyhctsa')

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .utils import _check_optional_deps, _input_problem, _zscore_or_zeros, decimate_ac1e, nan_outputs, z_score
from .quality import Quality, classify, quality_summary
from .results import ERROR_COLUMNS, save_results
from .distribute import (_compute_config_features_for_chunk, _compute_features_for_chunk,
                         _extract_features_single_series)

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

# The input transforms a config can request with ``preprocess:`` (besides ``zscore:`` and
# ``abs:``), the hctsa input each one reproduces, and the suffix it adds to the feature label.
# ``x_z`` is the z-scored series (``zscore: True``); the label suffix is hctsa's.
_PREPROCESS_LABELS = {
    'decimate_ac1e': '_dec',     # zscore(BF_PreProcess(x_z, 'decimate_ac1e'))
    'diff1': '_diff1',           # diff(x_z)
    'zscore_abs': '_mag',        # zscore(abs(x_z)): the magnitudes, re-z-scored
    'zscore_sign': '_sign',      # zscore(sign(x_z)): the signs, re-z-scored
}

def _transform_input(x, zscore: bool = False, absval: bool = False,
                     preprocess: Union[str, None] = None):
    """The series a feature function is called on: z-score, then ``preprocess``, then abs.

    Returns a scalar ``nan`` when ``decimate_ac1e`` cannot determine its delay or would leave
    fewer than 50 samples (the feature is then undefined, as in hctsa).
    """
    if zscore:
        x = z_score(x)
    if preprocess == 'decimate_ac1e':
        x = decimate_ac1e(x, rezscore=zscore)
        if not isinstance(x, np.ndarray):
            return np.nan
    elif preprocess == 'diff1':
        x = np.diff(np.asarray(x, dtype=float))
    elif preprocess == 'zscore_abs':
        x = _zscore_or_zeros(np.abs(x))
    elif preprocess == 'zscore_sign':
        x = _zscore_or_zeros(np.sign(x))
    if absval:
        x = np.abs(x)
    return x

def _preprocess_decorator(zscore: bool = False, absval: bool = False,
                          preprocess: Union[str, None] = None) -> Callable:
    """
    Decorator that gives a feature function the input its config asks for.

    The order is: z-score (``zscore``), then the named ``preprocess`` step, then the absolute
    value (``absval``). ``preprocess`` is one of the keys of ``_PREPROCESS_LABELS``:

    - ``'decimate_ac1e'``: one sample per floored 1/e autocorrelation time, then z-scored again
      if ``zscore`` (see :func:`pyhctsa.utils.decimate_ac1e`); if the delay cannot be
      determined, or fewer than 50 samples would remain, the function is not called and its
      outputs are all NaN (:func:`~pyhctsa.utils.nan_outputs`);
    - ``'diff1'``: incremental differences, ``diff(x_z)`` (not re-z-scored);
    - ``'zscore_abs'`` / ``'zscore_sign'``: ``zscore(abs(x_z))`` / ``zscore(sign(x_z))``.
    """
    if preprocess is not None and preprocess not in _PREPROCESS_LABELS:
        raise ValueError(f"Unknown preprocess setting '{preprocess}'; "
                         f"supported: {sorted(_PREPROCESS_LABELS)}")

    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            x = _transform_input(x, zscore, absval, preprocess)
            if np.ndim(x) == 0:  # no 1/e time: every output undefined, like hctsa's NaN
                return nan_outputs(func, *args, **kwargs)
            return func(x, *args, **kwargs)
        return wrapper
    return decorator

def classify_output(res) -> int:
    """The :class:`~pyhctsa.quality.Quality` code of one raw output, as an int.

    Kept for compatibility; :func:`pyhctsa.quality.classify` also returns the stored value.
    A string starting with ``"Error:"`` (the old way of recording an exception) is ``ERROR``.
    """
    if isinstance(res, str) and res.startswith("Error:"):
        return int(Quality.ERROR)
    return int(classify(res)[1])

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

def _build_label(base_name, combo_dict, ordered_args, do_zscore, do_absval, preprocess=None):
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
    if preprocess:
        label += _PREPROCESS_LABELS[preprocess]
    return label

# import name -> PyPI distribution name, where the two differ
_DIST_NAMES = {"sklearn": "scikit-learn", "pywt": "PyWavelets", "yaml": "pyyaml"}


def _import_failure_message(module_key: str, exc: ImportError) -> str:
    """Message for an operations module that failed to import (a missing required dependency)."""
    name = (getattr(exc, "name", None) or "").split(".")[0]
    dist = _DIST_NAMES.get(name, name)
    if name and not name.startswith("pyhctsa"):
        return (f"pyhctsa cannot import its operations module '{module_key}' because the required "
                f"package '{dist}' is not installed. Install it with `pip install {dist}` "
                f"(or reinstall pyhctsa with `pip install pyhctsa` to get all required dependencies). "
                f"Original error: {exc}")
    return (f"pyhctsa cannot import its operations module '{module_key}': {exc}. "
            f"This is a required module, so no features can be loaded without it.")


def _assemble(records: list, problems: list, index: pd.Index):
    """Build the values, quality, errors and calc_time tables of one extraction.

    ``records`` holds one worker record per series (``None`` for a series rejected by the input
    check, whose reason is in ``problems``). Columns are the union of the computed outputs, in
    order of first appearance; a computed series missing one of them has ``Quality.LINK_ERROR``
    there, and a rejected series has ``Quality.REJECTED`` everywhere.
    """
    computed = [r for r in records if r is not None]
    columns = list(dict.fromkeys(c for r in computed for c in r["values"]))
    funcs = list(dict.fromkeys(f for r in computed for f in r["times"]))
    n, p = len(records), len(columns)
    values = np.full((n, p), np.nan)
    quality = np.full((n, p), int(Quality.LINK_ERROR), dtype=np.uint8)
    times = np.full((n, len(funcs)), np.nan)
    col_idx = {c: j for j, c in enumerate(columns)}
    f_idx = {f: j for j, f in enumerate(funcs)}
    errors = []
    for i, (r, problem) in enumerate(zip(records, problems)):
        if r is None:
            quality[i] = int(Quality.REJECTED)
            errors.append((index[i], None, "InvalidInput", problem))
            continue
        for c, v in r["values"].items():
            values[i, col_idx[c]] = v
            quality[i, col_idx[c]] = r["quality"][c]
        for f, t in r["times"].items():
            times[i, f_idx[f]] = t
        errors.extend((index[i], f, etype, msg) for f, etype, msg in r["errors"])
    return (pd.DataFrame(values, index=index, columns=columns),
            pd.DataFrame(quality, index=index, columns=columns),
            pd.DataFrame(errors, columns=ERROR_COLUMNS),
            pd.DataFrame(times, index=index, columns=funcs))

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
    
    Notes
    -----
    Besides the operation's own arguments, each entry of a feature's ``configs`` list may
    carry these meta keys (not passed to the function):

    - ``zscore: True``: z-score the series first (label gets no suffix; without it, ``_raw``);
    - ``abs: True``: take the absolute value last (suffix ``_abs``);
    - ``preprocess``: one more input transform, applied to the (z-scored) series before ``abs``;
      its label suffix follows ``_abs``. The values, with the hctsa input each reproduces:

      * ``decimate_ac1e`` (suffix ``_dec``): ``zscore(BF_PreProcess(x_z, 'decimate_ac1e'))``.
        The z-scored series is decimated to one sample per floored 1/e autocorrelation time and
        z-scored again. If that time cannot be determined (e.g. the ACF never falls to 1/e), or
        fewer than 50 samples would remain, the feature is NaN. See
        :func:`pyhctsa.utils.decimate_ac1e`;
      * ``diff1`` (``_diff1``): ``diff(x_z)``;
      * ``zscore_abs`` (``_mag``): ``zscore(abs(x_z))``; ``zscore_sign`` (``_sign``):
        ``zscore(sign(x_z))``. (``abs: True`` alone is ``abs(x_z)``.)
    - ``select`` / ``exclude``: keep or drop keys of a dict-valued output.

    Examples
    --------
    >>> fc = FeatureCalculator()  # Load default configuration
    >>> x = np.random.randn(1000)
    >>> df = fc.extract(x)
    """
    def __init__(self, config_path: Union[str, None] = None):
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
        self.config_path = config_path
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self._load_config(config)

    @classmethod
    def _from_config(cls, config: dict) -> "FeatureCalculator":
        """A calculator built from an already-parsed configuration (used by parallel workers)."""
        calc = cls.__new__(cls)
        calc.config_path = None
        calc._load_config(config)
        return calc

    def _load_config(self, config: dict):
        # building consumes the meta params (zscore, abs, ...) of self.config, so keep the
        # parsed configuration for parallel workers to rebuild the features from
        self._parsed_config = copy.deepcopy(config)
        self.config = config
        self._operations_package = "pyhctsa.operations" # abs path
        self.feature_funcs = self._build_feature_funcs()
        # the functions as built from self.config, so extract() can tell whether
        # feature_funcs has since been changed (see _parallel_task)
        self._config_funcs = dict(self.feature_funcs)
        # results of the last extract(): see extract() and save()
        self.quality = None
        self.errors = None
        self.calc_time = None
        logger.info(f"Loaded {len(self.feature_funcs)} master operations.")
    
    def _check_deps(self, module_key, feature_name, config):
        raw_deps = config.get("dependencies", None)
        if not raw_deps:
            return True
        deps_to_check = [raw_deps] if isinstance(raw_deps, str) else raw_deps
        missing = [dep for dep in deps_to_check if not _check_optional_deps(dep)]
        if missing:
            full_name = f"{module_key}.{feature_name}"
            logger.debug(f"Skipping function '{full_name}' - missing optional dependencies: {', '.join(missing)}")
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
                # An operations module that cannot be imported would silently drop all of its
                # features, so this is always an error. Genuinely optional dependencies are
                # declared per feature in hctsa.yaml and skipped in _check_deps, never here.
                raise ImportError(_import_failure_message(module_key, e)) from e

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
                    preprocess = config_item.pop('preprocess', None)
                    select_features = config_item.pop('select', None)
                    exclude_features = config_item.pop('exclude', None)
                    # setup base function
                    master_func = _preprocess_decorator(do_zscore, do_absval, preprocess)(op_func)
                    
                    # standardise config_item into a list of combinations
                    # if config_item is empty, product(*) returns [()], allowing us to loop once
                    keys = list(config_item.keys())
                    values = [v if isinstance(v, list) else [v] for v in config_item.values()]
                    
                    for combo_values in product(*values):
                        combo_dict = dict(zip(keys, combo_values))
                        
                        # generate label and apply wrappers
                        label = _build_label(base_name, combo_dict, ordered_args, do_zscore, do_absval, preprocess)
                        
                        final_func = partial(master_func, **combo_dict)
                        # filter out certain features if specified by the user with the select/include keys
                        if select_features:
                            final_func = _apply_selection_wrapper(final_func, filter_keys=select_features, keep=True)
                        elif exclude_features:
                            final_func = _apply_selection_wrapper(final_func, filter_keys=exclude_features, keep=False)
                        
                        feature_funcs[label] = final_func
        
        if self._skipped_functions:
            details = ", ".join(f"{name} (needs {', '.join(deps)})" for name, deps in self._skipped_functions)
            missing_all = sorted({d for _, deps in self._skipped_functions for d in deps})
            logger.warning(f"Skipped {len(self._skipped_functions)} feature set(s) because optional "
                           f"dependencies are not installed: {details}. "
                           f"Install with: pip install {' '.join(missing_all)}")
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
            Whether to print progress and timing information. Default is ``False``.
        distributor: object, optional
            Optional distributor for parallel computation.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per input series and one column per feature. Every value is
            a float: a real, finite number, or NaN. Why a value is NaN is in
            :attr:`quality`, which this call also sets, with :attr:`errors` and
            :attr:`calc_time`:

            * ``quality``: a ``uint8`` DataFrame of the same shape, one
              :class:`~pyhctsa.quality.Quality` code per value (0 good, 1 error, 2 NaN, 3 Inf,
              4 -Inf, 5 complex, 6 empty, 7 link error, 8 non-scalar, 9 rejected input; codes 0-7
              are hctsa's ``TS_Quality``).
            * ``errors``: one row per feature function that raised (``instance``, ``feature``,
              ``error_type``, ``message``), and one per series rejected by the input check
              (``feature`` is ``None``).
            * ``calc_time``: seconds per series and feature function.

            A series that fails the input check (too short, constant, or containing NaN or Inf) is not
            computed, and its row is coded ``Quality.REJECTED``. If every series fails it, a
            ``ValueError`` is raised.
        """
        # forget the previous extraction, so save() cannot pair new values with old labels
        self.quality = self.errors = self.calc_time = None
        series_list = _standardise_inputs(data)
        # check each time series: an invalid one is not computed and its row is coded
        # Quality.REJECTED; only if every series is invalid is there nothing to return
        problems = [_input_problem(t) for t in series_list]
        invalid = [i for i, p in enumerate(problems) if p is not None]
        if len(invalid) == len(series_list):
            details = "; ".join(f"{i}: {problems[i]}" for i in invalid[:10])
            raise ValueError(f"One or more time series instances are invalid: {np.array(invalid)} "
                             f"({details})")
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

        if invalid:
            warnings.warn(f"{len(invalid)} of {len(series_list)} time series are invalid and were not "
                          f"computed (their rows have quality Quality.REJECTED; reasons in "
                          f"FeatureCalculator.errors).", stacklevel=2)
        valid_series = [t for t, p in zip(series_list, problems) if p is None]

        if verbose:
            print(f"Evaluating {len(self.feature_funcs)} partialed functions. Strap in!...")
        start_time = time.perf_counter()
        if distributor:
            # parallel execution
            task, task_kwargs = self._parallel_task()
            rows = distributor.map(task, valid_series, **task_kwargs)
        else:
            # sequential fallback
            if verbose:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None), # 'None' makes it expand to full width
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("Sequential Extraction", total=len(valid_series))
                    rows = []
                    for ts in valid_series:
                        rows.append(_extract_features_single_series(ts, self.feature_funcs))
                        progress.advance(task)
            else:
                rows = [_extract_features_single_series(ts, self.feature_funcs)
                        for ts in valid_series]

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"Feature extraction completed in {elapsed:.3f} seconds.")
        index = pd.Index(labels_list, name="instance")
        if len(rows) != len(problems) - len(invalid):
            raise RuntimeError(f"Expected results for {len(problems) - len(invalid)} series, got {len(rows)}")
        it = iter(rows)
        records = [None if p is not None else next(it) for p in problems]
        df, self.quality, self.errors, self.calc_time = _assemble(records, problems, index)
        self._last_elapsed = elapsed
        if verbose:
            print(f"Quality of the {df.size} values: {quality_summary(self.quality.to_numpy())}")
        return df

    def save(self, path, values: pd.DataFrame, meta: Union[dict, None] = None) -> Path:
        """Write ``values`` (the DataFrame the last :meth:`extract` returned) together with its
        :attr:`quality`, :attr:`errors` and :attr:`calc_time` to ``path``.

        The format follows the suffix: ``.npz`` (no extra dependency) or ``.parquet`` (needs
        pyarrow). Read it back with :func:`pyhctsa.load_results`.
        """
        if self.quality is None:
            raise RuntimeError("Nothing to save: call extract() first.")
        if not values.index.equals(self.quality.index) or not values.columns.equals(self.quality.columns):
            raise ValueError("values must be the DataFrame returned by the last extract()")
        info = {"n_feature_functions": len(self.feature_funcs),
                "config_path": None if self.config_path is None else str(self.config_path)}
        info.update(meta or {})
        return save_results(path, values, self.quality, self.errors, self.calc_time, info)

    def _parallel_task(self) -> tuple[Callable, dict]:
        """The chunk function, and its keyword arguments, to hand to a distributor.

        Workers rebuild the feature functions from the parsed configuration rather than receiving
        them pickled: they are closures over their operations modules, which ``dill`` must pickle
        by value, globals and all, and that fails when Python is embedded (e.g. in R through
        reticulate) because the standard library then no longer looks importable to ``dill``. Only
        if ``feature_funcs`` has been changed to hold functions the configuration does not build
        are the functions themselves sent.
        """
        funcs = self.feature_funcs
        if all(self._config_funcs.get(label) is func for label, func in funcs.items()):
            # None: all of them, in configuration order
            labels = None if list(funcs) == list(self._config_funcs) else list(funcs)
            return _compute_config_features_for_chunk, {"config": self._parsed_config, "labels": labels}
        return _compute_features_for_chunk, {"feature_funcs": funcs}
