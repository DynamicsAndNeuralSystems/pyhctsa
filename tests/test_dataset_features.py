"""get_dataset_features: the precomputed feature files of the downloadable datasets.

No test touches the network. Small synthetic files written with save_results cover the logic; the
real 2.0.0 files are checked only where a local copy exists (``PYHCTSA_DATASET_FEATURES_DIR``,
default ``~/Downloads/pyhctsa_dataset_features_2.0.0``), and are skipped otherwise, as in CI.
"""
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyhctsa.calculator
import pyhctsa.utils
from pyhctsa import Quality, Results, save_results
from pyhctsa.utils import _DATASET_FEATURES, _config_sha256, get_dataset, get_dataset_features
from conftest import e1000_available, s1000_available

LOCAL_DIR = Path(os.environ.get("PYHCTSA_DATASET_FEATURES_DIR",
                                Path.home() / "Downloads" / "pyhctsa_dataset_features_2.0.0"))


def _write(path, which, n=3, **meta):
    """A small features file as published for ``which``, with ``meta`` overriding its metadata."""
    labels = [f"ts_{i + 1}" for i in range(n)]
    values = pd.DataFrame(np.arange(2.0 * n).reshape(n, 2), index=labels, columns=["op.a", "op.b"])
    quality = pd.DataFrame(Quality.GOOD, index=labels, columns=values.columns, dtype=np.uint8)
    record = {"pyhctsa_version": pyhctsa.utils.__version__, "config_sha256": _config_sha256(),
              "dataset": which, "dataset_file_sha256": _DATASET_FEATURES[which]["series"]["sha256"]}
    record.update(meta)
    record = {k: v for k, v in record.items() if v is not None}
    return save_results(path, values, quality, meta=record)


@pytest.fixture
def publish(monkeypatch, tmp_path):
    """Point a dataset's record at a local file (a ``file://`` URL with its size and digest), with
    the download cache in a temporary directory."""
    monkeypatch.setenv("PYHCTSA_DATA_DIR", str(tmp_path / "cache"))

    def _publish(which, path, verify=True):
        path = Path(path)
        record = _DATASET_FEATURES[which]
        monkeypatch.setitem(record, "url", path.as_uri())
        if not verify:
            return
        monkeypatch.setitem(record, "size", path.stat().st_size)
        monkeypatch.setitem(record, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    return _publish


def _no_warnings(which):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return get_dataset_features(which)


@pytest.mark.parametrize("which", ["e1000", "1000x1000"])
def test_downloads_verifies_caches_and_loads(which, publish, tmp_path):
    src = _write(tmp_path / "published.npz", which)
    publish(which, src)
    res = _no_warnings(which)
    assert isinstance(res, Results)
    assert res.values.shape == (3, 2) and list(res.values.index) == ["ts_1", "ts_2", "ts_3"]
    assert res.meta["dataset"] == which
    cached = tmp_path / "cache" / _DATASET_FEATURES[which]["filename"]
    assert cached.read_bytes() == src.read_bytes()
    src.unlink()  # later calls read the verified cached copy
    assert _no_warnings(which).values.equals(res.values)


def test_rejects_a_file_that_does_not_verify(publish, tmp_path):
    src = _write(tmp_path / "published.npz", "e1000")
    publish("e1000", src)
    src.write_bytes(src.read_bytes() + b"x")
    with pytest.raises(OSError, match="verification"):
        get_dataset_features("e1000")


@pytest.mark.parametrize("which", ["e1000", "1000x1000"])
def test_clear_error_while_the_url_is_unset(which, monkeypatch):
    monkeypatch.setitem(_DATASET_FEATURES[which], "url", None)
    with pytest.raises(RuntimeError, match="URL .* not set"):
        get_dataset_features(which)


@pytest.mark.parametrize("which", ["sinusoid", "noise", "E1000", "unknown"])
def test_only_the_downloadable_datasets_have_features(which):
    with pytest.raises(ValueError, match="No precomputed features"):
        get_dataset_features(which)


def test_warns_when_the_configuration_differs(publish, tmp_path):
    publish("e1000", _write(tmp_path / "published.npz", "e1000", config_sha256="0" * 64))
    with pytest.warns(UserWarning, match="feature set may differ") as record:
        res = get_dataset_features("e1000")
    assert "0" * 64 in str(record[0].message) and _config_sha256() in str(record[0].message)
    assert res.values.shape == (3, 2)  # returned anyway


def test_warns_when_the_version_differs(publish, tmp_path):
    publish("e1000", _write(tmp_path / "published.npz", "e1000", pyhctsa_version="1.9.0"))
    with pytest.warns(UserWarning, match=r"pyhctsa 1\.9\.0, but pyhctsa .* is installed"):
        get_dataset_features("e1000")


def test_one_warning_lists_every_difference(publish, tmp_path):
    publish("1000x1000", _write(tmp_path / "published.npz", "1000x1000",
                                pyhctsa_version="1.9.0", config_sha256=None, dataset_file_sha256=None))
    with pytest.warns(UserWarning) as record:
        get_dataset_features("1000x1000")
    assert len(record) == 1
    message = str(record[0].message)
    assert message.count("not recorded") == 2  # save_results always records the version
    assert "pyhctsa 1.9.0" in message and "feature set may differ" in message
    assert "rows may not match" in message


@pytest.mark.parametrize("meta", [{"dataset_file_sha256": "f" * 64}, {"dataset": "1000x1000"}])
def test_warns_when_the_series_file_differs(meta, publish, tmp_path):
    publish("e1000", _write(tmp_path / "published.npz", "e1000", **meta))
    with pytest.warns(UserWarning, match="rows may not match its series"):
        get_dataset_features("e1000")


def test_records_point_at_the_series_files_get_dataset_uses():
    assert _DATASET_FEATURES["e1000"]["series"] is pyhctsa.utils._E1000
    assert _DATASET_FEATURES["1000x1000"]["series"] is pyhctsa.utils._S1000
    for record in _DATASET_FEATURES.values():
        assert record["filename"].endswith(".npz")  # load_results reads one file


def test_config_hash_is_the_one_feature_calculator_records(module_config):
    # the hash of the default configuration, without building the default calculator (~10 s)
    default = Path(pyhctsa.calculator.__file__).resolve().parent / "configurations" / "feature_config.yaml"
    assert Path(pyhctsa.utils._DEFAULT_CONFIG_PATH).resolve() == default
    path = module_config("medical")
    assert _config_sha256(path) == pyhctsa.calculator.FeatureCalculator(path)._config_sha256


# The real 2.0.0 files, where a local copy exists.

def _local(which):
    path = LOCAL_DIR / _DATASET_FEATURES[which]["filename"]
    if not path.exists():
        pytest.skip(f"no local copy of {path.name}")
    return path


@pytest.mark.parametrize("which", ["e1000", "1000x1000"])
def test_real_file(which, publish):
    publish(which, _local(which), verify=False)  # the size and digest in the record must match
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = get_dataset_features(which)
    # only an installed version other than the file's (a development install) may warn
    assert all(f"pyhctsa {res.meta['pyhctsa_version']}, but" in str(w.message)
               and "feature set" not in str(w.message) and "rows" not in str(w.message)
               for w in caught)
    assert res.meta["config_sha256"] == _config_sha256()
    assert res.meta["dataset_file_sha256"] == _DATASET_FEATURES[which]["series"]["sha256"]
    assert res.values.shape == res.quality.shape == (1000, 6406)
    assert list(res.values.index) == [f"ts_{i + 1}" for i in range(1000)]
    assert res.calc_time.shape == (1000, res.meta["n_feature_functions"])
    good = res.quality.to_numpy() == Quality.GOOD
    assert np.array_equal(np.isfinite(res.values.to_numpy()), good)


@pytest.mark.parametrize("which", [
    pytest.param("e1000", marks=pytest.mark.skipif(not e1000_available(), reason="Empirical1000 not cached")),
    pytest.param("1000x1000", marks=pytest.mark.skipif(not s1000_available(),
                                                       reason="1000x1000 collection not cached")),
])
def test_real_file_rows_are_get_dataset_series_in_order(which, publish, module_config):
    path = _local(which)
    series = get_dataset(which)  # from the user cache, before publish() moves the cache
    publish(which, path, verify=False)
    # recompute the cheap medical features of a few rows from the series get_dataset returns
    rows = [0, 1, 499, 998, 999]
    calc = pyhctsa.calculator.FeatureCalculator(module_config("medical"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a version mismatch; repeated values in some series
        res = get_dataset_features(which)
        got = calc.extract([series[i] for i in rows], labels=[f"ts_{i + 1}" for i in rows])
    assert len(series) == len(res.values)
    stored = res.values.loc[got.index, got.columns].to_numpy()
    np.testing.assert_allclose(got.to_numpy(), stored, rtol=1e-10, atol=0, equal_nan=True)
    # the rows differ from each other, so a shifted or reordered series would not match
    assert np.isfinite(stored).any() and len(np.unique(np.nan_to_num(stored), axis=0)) == len(rows)
