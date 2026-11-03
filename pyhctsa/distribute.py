from abc import ABC, abstractmethod
from typing import Union
import logging
import multiprocessing as mp
import numpy as np
from functools import partial
from rich import get_console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from pathos.helpers import mp as pathos_mp

def _extract_features_single_series(ts, feature_funcs):
    """
    Worker task.
    """
    row = {}
    for name, func in feature_funcs.items():
        try:
            val = func(ts)
            if isinstance(val, dict):
                for k, v in val.items():
                    row[f"{name}.{k}"] = v
            elif isinstance(val, np.ndarray) and val.ndim == 1 and val.size <= 16:
                for i, v in enumerate(val):
                    row[f"{name}_{i}"] = v
            else:
                row[name] = val
        except Exception as e:
            row[name] = f"Error: {e}"
    return row

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
