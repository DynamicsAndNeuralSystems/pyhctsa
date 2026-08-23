"""Parallel execution backends for :class:`~pyhctsa.calculator.FeatureCalculator`."""
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Union

import dill
import numpy as np
from pathos.helpers import mp as _mp
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)

logger = logging.getLogger('pyhctsa')

# Environment variables read by the threading layers of the BLAS/OpenMP
# backends that NumPy, SciPy and friends link against. They are consulted when
# the backend is *loaded*, so they must be set before the worker interpreter
# imports NumPy -- see ``_thread_env``.
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# 1-D array outputs no longer than this are spread across one column per
# element; anything larger is left as a single array-valued cell.
_MAX_INLINE_ARRAY = 16


def _flatten_into(values: dict, name: str, val: Any) -> None:
    """Write a single feature output into ``values`` under one or more keys."""
    if isinstance(val, dict):
        for k, v in val.items():
            values[f"{name}.{k}"] = v
    elif isinstance(val, np.ndarray) and val.ndim == 1 and val.size <= _MAX_INLINE_ARRAY:
        for i, v in enumerate(val):
            values[f"{name}_{i}"] = v
    else:
        values[name] = val


def _extract_features_single_series(ts, feature_funcs) -> tuple[dict, dict]:
    """
    Worker task.

    Returns a ``(values, errors)`` pair. A feature that raises contributes
    nothing to ``values`` and one entry to ``errors``, keyed by feature label.
    Keeping failures out of ``values`` is what makes the resulting column set
    depend only on the configuration, not on which series happened to fail.
    """
    values: dict = {}
    errors: dict = {}
    for name, func in feature_funcs.items():
        try:
            val = func(ts)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            # the traceback only exists here, in the process that raised
            logger.debug("Feature '%s' failed", name, exc_info=True)
            continue
        _flatten_into(values, name, val)
    return values, errors


@contextmanager
def _thread_env(n_threads: Union[int, None]):
    """Temporarily pin the BLAS/OpenMP thread-count environment variables.

    Worker processes inherit the environment as it stands when they are
    spawned, so wrapping pool creation in this context manager is what stops
    ``n_workers`` processes each starting their own thread pool and
    oversubscribing the machine.
    """
    if n_threads is None:
        yield
        return
    saved = {v: os.environ.get(v) for v in _BLAS_THREAD_VARS}
    os.environ.update({v: str(n_threads) for v in _BLAS_THREAD_VARS})
    try:
        yield
    finally:
        for var, old in saved.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def _default_n_workers() -> int:
    """Number of usable CPUs, honouring cgroup and CPU-affinity limits."""
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python >= 3.13
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# --------------------------------------------------------------------------
# Worker-side state.
#
# The payload (the mapped function together with its bound keyword arguments)
# is deserialised once per worker process by the pool initialiser and cached
# here. Binding it into the task callable instead would make the pool
# re-serialise it for every single task.
# --------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(payload: bytes, n_threads: Union[int, None]) -> None:
    if n_threads is not None:
        try:
            import threadpoolctl
            # keep a reference: the limits are lifted when this object is restored
            _WORKER["_threadpool"] = threadpoolctl.threadpool_limits(limits=n_threads)
        except Exception:
            # ``_thread_env`` has already covered the common (spawned) case
            pass
    _WORKER["func"], _WORKER["kwargs"] = dill.loads(payload)


def _run_worker(item):
    return _WORKER["func"](item, **_WORKER["kwargs"])


class BaseDistributor(ABC):
    """
    BaseDistributor abstract class.
    """
    @abstractmethod
    def map(self, func: Callable, data, *, progress: bool = False, **kwargs) -> list:
        """Apply ``func`` to every element of ``data`` and return the results.

        Results are returned in the same order as ``data``. Any additional
        keyword arguments are passed through to ``func`` on every call.
        """

    def close(self):
        """Cleanup resources"""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class LocalDistributor(BaseDistributor):
    """Local process-based distributor.

    This distributor parallelises work across local worker processes. The
    mapped function is sent to each worker once, when the pool starts, rather
    than once per task; for feature extraction the function carries the whole
    set of configured feature callables, so the difference is substantial.

    The pool is created on first use and owned solely by this instance, so
    constructing a distributor is cheap and ``close()`` really does shut its
    workers down.

    Parameters
    ----------
    n_workers: int, optional
        Number of worker processes to use. If ``None``, defaults to the number
        of CPUs available to this process (respecting CPU affinity and cgroup
        limits).
    worker_threads: int or None, optional
        Threads each worker may use for BLAS/OpenMP operations. Defaults to
        ``1``, so that ``n_workers`` processes do not oversubscribe the machine
        with nested thread pools. Pass ``None`` to leave the ambient
        configuration untouched.
    start_method: str, optional
        Process start method, one of ``'spawn'``, ``'fork'`` or
        ``'forkserver'``. Defaults to ``'spawn'``, which is the only method
        that is safe in the presence of threads. Note that under ``'spawn'``
        the calling module is re-imported by every worker, so scripts must
        guard their entry point with ``if __name__ == '__main__':``.
    chunksize: int, optional
        Number of series handed to a worker at a time. Defaults to ``1``:
        per-series feature cost varies by orders of magnitude, and larger
        chunks let one slow series stall a whole batch.

    Examples
    --------
    >>> with LocalDistributor(n_workers=4) as dist:      # doctest: +SKIP
    ...     df = calc.extract(data, distributor=dist)
    """
    def __init__(self, n_workers: Union[int, None] = None,
                 worker_threads: Union[int, None] = 1,
                 start_method: str = "spawn",
                 chunksize: int = 1):
        self.n_workers = _default_n_workers() if n_workers is None else int(n_workers)
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        if worker_threads is not None and int(worker_threads) < 1:
            raise ValueError(f"worker_threads must be >= 1 or None, got {worker_threads}")
        if chunksize < 1:
            raise ValueError(f"chunksize must be >= 1, got {chunksize}")
        self.worker_threads = None if worker_threads is None else int(worker_threads)
        self.start_method = start_method
        self.chunksize = chunksize
        self._ctx = _mp.get_context(start_method)
        self._pool = None
        self._payload = None

    def _ensure_pool(self, payload: bytes):
        """Start the pool, or restart it if the mapped function has changed."""
        if self._pool is not None and payload == self._payload:
            return self._pool
        self.close()
        self._payload = payload
        with _thread_env(self.worker_threads):
            self._pool = self._ctx.Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(payload, self.worker_threads),
            )
        return self._pool

    def map(self, func: Callable, data, *, progress: bool = False, **kwargs) -> list:
        data = list(data)
        if not data:
            return []

        # serialise the function and its bound arguments exactly once
        payload = dill.dumps((func, kwargs))
        pool = self._ensure_pool(payload)

        try:
            iterator = pool.imap(_run_worker, data, chunksize=self.chunksize)
            if not progress:
                return list(iterator)
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                expand=True
            ) as bar:
                return list(bar.track(
                    iterator,
                    total=len(data),
                    description=f"Parallel (workers={self.n_workers})"
                ))
        except BaseException:
            # a half-consumed pool cannot be reused; tear it down rather than
            # leaving orphaned workers behind (notably on KeyboardInterrupt)
            self.terminate()
            raise

    def terminate(self):
        """Stop the workers immediately, without waiting for pending tasks."""
        if self._pool is None:
            return
        pool, self._pool, self._payload = self._pool, None, None
        pool.terminate()
        pool.join()

    def close(self):
        """Shut the pool down, letting in-flight tasks finish. Idempotent."""
        if self._pool is None:
            return
        pool, self._pool, self._payload = self._pool, None, None
        pool.close()
        pool.join()
