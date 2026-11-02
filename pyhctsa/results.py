"""Saving and loading feature values together with their quality labels.

``.npz`` (the default) needs only numpy and holds everything in one file, in the same layout as
pyspi's ``Calculator.save``. ``.parquet`` can be read from R (``arrow``) and MATLAB
(``parquetread``) too. It needs ``pyarrow`` (``pip install pyhctsa[parquet]``), and since a
parquet file holds one table, it writes one file per table next to the values:
``<name>.parquet`` (values), ``<name>.quality.parquet``, ``<name>.errors.parquet`` and
``<name>.calc_time.parquet``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd

from . import __version__

__all__ = ["Results", "save_results", "load_results"]

NPZ_SCHEMA = "pyhctsa.results/1"
ERROR_COLUMNS = ["instance", "feature", "error_type", "message"]


class Results(NamedTuple):
    """Feature values and everything recorded about them, as returned by :func:`load_results`."""

    values: pd.DataFrame
    """Series x features, float64; NaN wherever the quality is not ``Quality.GOOD``."""
    quality: pd.DataFrame
    """Same index and columns as ``values``; one ``uint8`` :class:`~pyhctsa.quality.Quality` code each."""
    errors: pd.DataFrame
    """One row per feature function that raised, or series rejected by the input check."""
    calc_time: Union[pd.DataFrame, None]
    """Seconds per series and feature function (``None`` if not saved)."""
    meta: dict
    """pyhctsa version, sizes, and any metadata passed to :func:`save_results`."""


def _meta(values: pd.DataFrame, extra: Union[dict, None]) -> dict:
    meta = {"pyhctsa_version": __version__, "n_series": int(values.shape[0]),
            "n_features": int(values.shape[1])}
    meta.update(extra or {})
    return meta


def save_results(path, values: pd.DataFrame, quality: pd.DataFrame,
                 errors: Union[pd.DataFrame, None] = None,
                 calc_time: Union[pd.DataFrame, None] = None,
                 meta: Union[dict, None] = None) -> Path:
    """Write feature values and their quality labels to ``path`` (format from the suffix).

    Parameters
    ----------
    path : str or Path
        Ends in ``.npz`` or ``.parquet``.
    values, quality : pandas.DataFrame
        As returned by :meth:`FeatureCalculator.extract` and set in
        :attr:`FeatureCalculator.quality`: same index and columns.
    errors, calc_time : pandas.DataFrame, optional
        :attr:`FeatureCalculator.errors` and :attr:`FeatureCalculator.calc_time`.
    meta : dict, optional
        Extra metadata to store (JSON-serialisable).

    Returns
    -------
    Path
        The path of the values file.
    """
    path = Path(path)
    if list(quality.index) != list(values.index) or list(quality.columns) != list(values.columns):
        raise ValueError("values and quality must have the same index and columns")
    errors = pd.DataFrame(columns=ERROR_COLUMNS) if errors is None else errors
    info = _meta(values, meta)
    if path.suffix == ".npz":
        arrays = dict(
            schema=np.array(NPZ_SCHEMA),
            values=values.to_numpy(dtype=np.float64),
            quality=quality.to_numpy(dtype=np.uint8),
            # JSON keeps the index labels' types (str or int); dtype 'U' arrays load without
            # allow_pickle, which object arrays would need
            instances=np.array(json.dumps(list(values.index), default=str)),
            features=np.array(list(values.columns), dtype="U"),
            errors=np.array(errors.to_json(orient="records")),
            meta=np.array(json.dumps(info, default=str)),
        )
        if calc_time is not None:
            arrays["calc_time"] = calc_time.to_numpy(dtype=np.float64)
            arrays["calc_time_functions"] = np.array(list(calc_time.columns), dtype="U")
        np.savez_compressed(path, **arrays)
    elif path.suffix == ".parquet":
        _require_pyarrow()
        stem = path.with_suffix("")
        v = values.copy()
        v.attrs = {}
        v.attrs["pyhctsa"] = json.dumps(info, default=str)
        v.to_parquet(path)
        quality.astype(np.uint8).to_parquet(f"{stem}.quality.parquet")
        errors.astype({"instance": str}).to_parquet(f"{stem}.errors.parquet", index=False)
        if calc_time is not None:
            calc_time.to_parquet(f"{stem}.calc_time.parquet")
    else:
        raise ValueError(f"Unsupported suffix '{path.suffix}'. Use '.npz' or '.parquet'.")
    return path


def load_results(path) -> Results:
    """Read results written by :func:`save_results` or :meth:`FeatureCalculator.save`.

    Returns a :class:`Results` named tuple, ``(values, quality, errors, calc_time, meta)``, as
    pyspi's ``load_results`` does. (The quality matrix is not put in ``values.attrs``: pandas
    copies ``attrs`` on every operation and cannot concatenate frames holding DataFrames there.)
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            schema = str(f["schema"])
            if schema != NPZ_SCHEMA:
                raise ValueError(f"{path}: unknown schema '{schema}' (expected '{NPZ_SCHEMA}')")
            index = pd.Index(json.loads(str(f["instances"])), name="instance")
            columns = [str(c) for c in f["features"]]
            values = pd.DataFrame(f["values"], index=index, columns=columns)
            quality = pd.DataFrame(f["quality"], index=index, columns=columns)
            errors = pd.read_json(_io(str(f["errors"])), orient="records")
            calc_time = None
            if "calc_time" in f.files:
                calc_time = pd.DataFrame(f["calc_time"], index=index,
                                         columns=[str(c) for c in f["calc_time_functions"]])
            meta = json.loads(str(f["meta"]))
    elif path.suffix == ".parquet":
        _require_pyarrow()
        stem = path.with_suffix("")
        values = pd.read_parquet(path)
        meta = json.loads(values.attrs.get("pyhctsa", "{}"))
        quality = pd.read_parquet(f"{stem}.quality.parquet")
        errors = pd.read_parquet(f"{stem}.errors.parquet")
        ct = Path(f"{stem}.calc_time.parquet")
        calc_time = pd.read_parquet(ct) if ct.exists() else None
    else:
        raise ValueError(f"Unsupported suffix '{path.suffix}'. Use '.npz' or '.parquet'.")
    if errors.empty:
        errors = pd.DataFrame(columns=ERROR_COLUMNS)
    values.attrs = {}
    return Results(values, quality, errors, calc_time, meta)


def _io(text: str):
    from io import StringIO
    return StringIO(text)


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError("Saving or loading .parquet needs pyarrow: "
                          "pip install 'pyhctsa[parquet]' (or pip install pyarrow)") from e
