from abc import ABC, abstractmethod
from typing import Union
import logging
import multiprocessing as mp
import time
from functools import partial
import numpy as np
from rich import get_console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from pathos.helpers import mp as pathos_mp

import warnings
from .quality import Quality, classify
from .utils import _reference_series

# (feature name, function identity) -> the output columns of a function that raised
_FAILED_COLUMNS: dict = {}

def _function_key(func, depth=0):
    """A key for a feature function that is the same for an unpickled or rebuilt copy of it: its
    module and name, and (for a ``partial`` or a decorator's closure) what it wraps and with what."""
    if depth > 8:
        return repr(func)
    if isinstance(func, partial):
        return (_function_key(func.func, depth + 1), repr(func.args),
                repr(sorted(func.keywords.items())))
    cells = []
    for cell in getattr(func, "__closure__", None) or ():
        try:
            c = cell.cell_contents
        except ValueError:  # an empty cell
            continue
        cells.append(_function_key(c, depth + 1) if callable(c) else repr(c))
    return getattr(func, "__module__", None), getattr(func, "__qualname__", repr(func)), tuple(cells)

def _failed_columns(name, func):
    """The columns of feature function ``name``, for a series on which it raised.

    They are learnt from its output on well-behaved reference series (once per function), so that
    they match its columns on the series where it works, including array outputs (``name_0``, ...).
    The global numpy random state is restored afterwards, so that later features are unaffected.
    """
    key = (name, _function_key(func))
    if key not in _FAILED_COLUMNS:
        columns = [name]
        state = np.random.get_state()
        try:
            for ref in _reference_series():
                try:
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out = func(ref)
                except Exception:
                    continue
                columns = [c for c, _ in _output_items(name, out)]
                break
        finally:
            np.random.set_state(state)
        _FAILED_COLUMNS[key] = columns
    return _FAILED_COLUMNS[key]

def _output_items(name, val):
    """The ``(column, raw value)`` pairs of one feature function's output."""
    if isinstance(val, dict):
        return [(f"{name}.{k}", v) for k, v in val.items()]
    if isinstance(val, np.ndarray) and val.ndim == 1 and val.size <= 16:
        return [(f"{name}_{i}", v) for i, v in enumerate(val)]
    return [(name, val)]


def _extract_features_single_series(ts, feature_funcs):
    """
    Worker task: every feature function on one series.

    Returns a dict with ``values`` (column -> float; NaN unless the output is a real, finite
    number), ``quality`` (column -> :class:`~pyhctsa.quality.Quality` code), ``times`` (feature
    function -> seconds) and ``errors`` (a list of ``(feature function, exception type, message)``).
    A function that raises gives every one of its output columns (from :func:`_failed_columns`)
    the code ``Quality.ERROR``, so the columns do not depend on which functions failed.
    """
    values, quality, times, errors = {}, {}, {}, []
    for name, func in feature_funcs.items():
        t0 = time.perf_counter()
        try:
            val = func(ts)
        except Exception as e:
            times[name] = time.perf_counter() - t0
            for col in _failed_columns(name, func):
                values[col], quality[col] = np.nan, int(Quality.ERROR)
            errors.append((name, type(e).__name__, str(e)))
            continue
        times[name] = time.perf_counter() - t0
        for col, raw in _output_items(name, val):
            values[col], q = classify(raw)
            quality[col] = int(q)
    return {"values": values, "quality": quality, "times": times, "errors": errors}

def _compute_features_for_chunk(chunk, feature_funcs):
    return [_extract_features_single_series(ts, feature_funcs) for ts in chunk]

# the configuration this worker process last built feature functions from, and those functions
_worker_config = None
_worker_funcs = None

def _feature_funcs_from_config(config):
    """The feature functions a configuration builds, built once per worker process."""
    global _worker_config, _worker_funcs
    if _worker_funcs is None or config != _worker_config:
        import copy
        from .calculator import FeatureCalculator
        logger = logging.getLogger('pyhctsa')
        level = logger.level
        logger.setLevel(logging.ERROR)  # the calling process has already reported skipped features
        try:
            # a copy: building consumes the config's meta params, and config is the cache key
            _worker_funcs = FeatureCalculator._from_config(copy.deepcopy(config)).feature_funcs
        finally:
            logger.setLevel(level)
        _worker_config = config
    return _worker_funcs

def _compute_config_features_for_chunk(chunk, config, labels=None):
    """
    Worker task that builds its feature functions from ``config`` (all of them, or those named in
    ``labels``, in that order) instead of receiving them pickled.
    """
    funcs = _feature_funcs_from_config(config)
    if labels is not None:
        funcs = {label: funcs[label] for label in labels}
    return _compute_features_for_chunk(chunk, funcs)

class BaseDistributor(ABC):
    """
    BaseDistributor abstract class.
    """
    def calculate_chunk_size(self, n_items, n_workers):
        """
        Calculate the number of chunks to distribute amongst the workers.
        """
        if n_items == 0:
            return 1
        return max(1, n_items // (n_workers*5))
    @abstractmethod
    def map(self, func, iterable, **kwargs):
        """Apply func to every element in iterable and return results"""
        pass
    def close(self):
        """Cleanup resources"""
        pass

class LocalDistributor(BaseDistributor):
    """Local process-based distributor using a pathos ProcessPool.

    This distributor parallelises work across local worker processes using
    `pathos.multiprocessing.ProcessPool`. 

    Parameters
    ----------
    n_workers: int, optional
        Number of worker processes to use. If ``None``, defaults to
        ``multiprocessing.cpu_count()``.
    """
    def __init__(self, n_workers: Union[int, None] = None):
        self.n_workers = n_workers or mp.cpu_count()
        try:
            pathos_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # already set
            pass
        from pathos.multiprocessing import ProcessPool
        self.pool = ProcessPool(nodes=self.n_workers)
    
    def map(self, func, data, **kwargs):
        c_size = self.calculate_chunk_size(len(data), self.n_workers)
        data_chunks = [data[i:i + c_size] for i in range(0, len(data), c_size)]
        
        worker_task = partial(func, **kwargs)
        iterator = self.pool.imap(worker_task, data_chunks)
        
        # a console that can't encode '•' (e.g. ASCII stdout) gets a plain separator
        separator = "-" if get_console().options.ascii_only else "•"
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn(separator),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            chunked_results = list(progress.track(
                iterator, 
                total=len(data_chunks), 
                description=f"Parallel (Pathos, cores={self.n_workers})"
            ))
       
        return [row for chunk in chunked_results for row in chunk]

    def close(self):
        self.pool.clear()
        self.pool.close()
        self.pool.join()
