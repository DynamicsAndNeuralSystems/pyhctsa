from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
from functools import partial
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, track
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

class BaseDistributor(ABC):
    def calculate_chunk_size(self, n_items, n_workers):
        if n_items == 0: return 1
        return max(1, n_items // (n_workers*5))
    @abstractmethod
    def map(self, func, iterable, **kwargs):
        """Apply func to every element in iterable and return results"""
        pass
    def close(self):
        """Cleanup resources"""
        pass

class LocalDistributor(BaseDistributor):
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
        try:
            pathos_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set
            pass
        from pathos.multiprocessing import ProcessPool
        self.pool = ProcessPool(nodes=self.n_workers)
    
    def map(self, func, data, **kwargs):
        c_size = self.calculate_chunk_size(len(data), self.n_workers)
        data_chunks = [data[i:i + c_size] for i in range(0, len(data), c_size)]
        
        worker_task = partial(_compute_features_for_chunk, **kwargs)
        iterator = self.pool.imap(worker_task, data_chunks)
        
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
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
