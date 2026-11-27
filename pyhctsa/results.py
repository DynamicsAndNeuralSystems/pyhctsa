"""Saving and loading feature values together with their quality labels.

``.npz`` (the default) needs only numpy and holds everything in one file, in the same layout as
pyspi's ``Calculator.save``. ``.parquet`` can be read from R (``arrow``) and MATLAB
(``parquetread``) too. It needs ``pyarrow`` (``pip install pyhctsa[parquet]``), and since a
parquet file holds one table, it writes one file per table next to the values:
``<name>.parquet`` (values), ``<name>.quality.parquet``, ``<name>.errors.parquet`` and
``<name>.calc_time.parquet``.

Every file also records the feature set that produced the values (the pyhctsa version, the
configuration's name and content hash, the optional dependencies that were missing, ...): in the
``meta`` array of a ``.npz`` file, and in the key-value schema metadata (key ``pyhctsa``) of each
``.parquet`` file. :func:`load_results` returns it as ``meta``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd

from . import __version__

__all__ = ["Results", "save_results", "load_results"]

NPZ_SCHEMA = "pyhctsa.results/1"
ERROR_COLUMNS = ["instance", "feature", "error_type", "message"]
PARQUET_META_KEY = b"pyhctsa"


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
    """What produced the values: pyhctsa version, sizes, time saved, the feature set (written by
    :meth:`FeatureCalculator.save`), and any metadata passed to :func:`save_results`. Keys a file
    does not hold (older files, or files saved without them) are absent."""


def _meta(values: pd.DataFrame, extra: Union[dict, None]) -> dict:
    meta = {"pyhctsa_version": __version__, "n_series": int(values.shape[0]),
            "n_features": int(values.shape[1]),
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
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
        Extra metadata to store (JSON-serialisable). The pyhctsa version, the number of series
        and features, and the time saved (UTC) are always stored; :meth:`FeatureCalculator.save`
        adds the feature set (configuration, its SHA-256, skipped optional dependencies).

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
            instances=np.array(json.dumps(list(values.index), default=_json_default)),
            features=np.array(list(values.columns), dtype="U"),
            errors=np.array(json.dumps(errors[ERROR_COLUMNS].to_dict("records"), default=_json_default)),
            meta=np.array(json.dumps(info, default=_json_default)),
        )
        if calc_time is not None:
            arrays["calc_time"] = calc_time.to_numpy(dtype=np.float64)
            arrays["calc_time_functions"] = np.array(list(calc_time.columns), dtype="U")
        np.savez_compressed(path, **arrays)
    elif path.suffix == ".parquet":
        _require_pyarrow()
        stem = path.with_suffix("")
        meta_json = json.dumps(info, default=_json_default)
        _write_parquet(values, path, meta_json)
        _write_parquet(quality.astype(np.uint8), f"{stem}.quality.parquet", meta_json)
        e = errors[ERROR_COLUMNS].copy()
        # a parquet column holds one type: keep the instance labels' own type when they share one
        if pd.api.types.infer_dtype(e["instance"], skipna=True) not in ("integer", "string", "empty"):
            e["instance"] = e["instance"].astype(str)
        _write_parquet(e, f"{stem}.errors.parquet", meta_json, index=False)
        ct = Path(f"{stem}.calc_time.parquet")
        if calc_time is not None:
            _write_parquet(calc_time, ct, meta_json)
        elif ct.exists():  # do not leave an earlier save's timings beside these values
            ct.unlink()
    else:
        raise ValueError(f"Unsupported suffix '{path.suffix}'. Use '.npz' or '.parquet'.")
    return path


def load_results(path) -> Results:
    """Read results written by :func:`save_results` or :meth:`FeatureCalculator.save`.

    Returns a :class:`Results` named tuple, ``(values, quality, errors, calc_time, meta)``, as
    pyspi's ``load_results`` does. (The quality matrix is not put in ``values.attrs``: pandas
    copies ``attrs`` on every operation and cannot concatenate frames holding DataFrames there.)
    ``meta`` says what produced the values (see :class:`Results`); it is ``{}`` for a file that
    holds no pyhctsa metadata, such as a parquet file written by pandas.
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
            errors = pd.DataFrame(json.loads(str(f["errors"])), columns=ERROR_COLUMNS)
            calc_time = None
            if "calc_time" in f.files:
                calc_time = pd.DataFrame(f["calc_time"], index=index,
                                         columns=[str(c) for c in f["calc_time_functions"]])
            meta = json.loads(str(f["meta"])) if "meta" in f.files else {}
    elif path.suffix == ".parquet":
        _require_pyarrow()
        import pyarrow.parquet as pq
        stem = path.with_suffix("")
        values = pd.read_parquet(path)
        stored = pq.read_schema(path).metadata or {}
        meta = json.loads(stored.get(PARQUET_META_KEY, b"{}"))
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


def _write_parquet(df: pd.DataFrame, path, meta_json: str, index: Union[bool, None] = None):
    """``df`` to ``path``, with ``meta_json`` in the file's own key-value schema metadata, which
    any parquet reader (and any pandas version) keeps."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = df.copy()
    df.attrs = {}
    table = pa.Table.from_pandas(df, preserve_index=index)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           PARQUET_META_KEY: meta_json})
    pq.write_table(table, path)


def _json_default(o):
    """numpy scalars as the Python numbers they hold (so integer labels stay integers)."""
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError("Saving or loading .parquet needs pyarrow: "
                          "pip install 'pyhctsa[parquet]' (or pip install pyarrow)") from e
