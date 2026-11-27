"""Saved results record the feature set that produced them (version, configuration, skipped
optional dependencies), and files without that record still load."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyhctsa import __version__, load_results
from pyhctsa.calculator import FeatureCalculator

CONFIG = ("correlation:\n"
          "  autocorr:\n    base_name: ac\n    configs:\n"
          "      - {tau: [1, 2], method: 'Fourier', zscore: True}\n"
          "    ordered_args: ['tau']\n"
          "  first_crossing:\n    base_name: fc\n    configs:\n"
          "      - {corr_fun: 'ac', threshold: 0, what_out: 'discrete', zscore: True}\n")

MISSING = ("  ami_dep:\n    base_name: ami\n"
           "    dependencies: [a_package_that_does_not_exist, numpy]\n    configs:\n"
           "      - {tau: [1, 2, 3], zscore: True}\n      - {tau: 5, zscore: False}\n"
           "    ordered_args: ['tau']\n")


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _extract(calc):
    rng = np.random.default_rng(0)
    return calc.extract([rng.standard_normal(200), rng.standard_normal(250)], labels=["a", "b"])


def _need(suffix):
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")


@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_saved_files_record_the_feature_set(tmp_path, suffix):
    _need(suffix)
    config = _write(tmp_path / "small.yaml", CONFIG)
    calc = FeatureCalculator(config)
    df = _extract(calc)
    r = load_results(calc.save(tmp_path / f"res{suffix}", df, meta={"note": "x"}))
    m = r.meta
    assert m["pyhctsa_version"] == __version__
    assert m["config_path"] == config and m["config_name"] == "small.yaml"
    assert m["config_sha256"] == hashlib.sha256(CONFIG.encode("utf-8")).hexdigest()
    assert m["n_features"] == df.shape[1] and m["n_series"] == 2
    assert m["n_feature_functions"] == 3 and m["feature_functions_modified"] is False
    assert m["missing_dependencies"] == [] and m["skipped_operations"] == {}
    assert m["n_skipped_feature_functions"] == 0
    assert datetime.fromisoformat(m["saved_utc"]).utcoffset().total_seconds() == 0
    assert m["note"] == "x"
    # the values themselves are untouched, and what extract returned has no metadata attached
    pd.testing.assert_frame_equal(r.values, df, check_names=False)
    assert df.attrs == {} and r.values.attrs == {}


def test_every_parquet_file_carries_the_record(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    path = calc.save(tmp_path / "res.parquet", _extract(calc))
    stem = path.with_suffix("")
    for p in [path] + [Path(f"{stem}.{t}.parquet") for t in ("quality", "errors", "calc_time")]:
        meta = json.loads(pq.read_schema(p).metadata[b"pyhctsa"])
        assert meta["config_sha256"] == calc._config_sha256, p


def test_hash_follows_the_content_not_the_path(tmp_path):
    a = FeatureCalculator(_write(tmp_path / "a.yaml", CONFIG))
    same = FeatureCalculator(_write(tmp_path / "b.yaml", CONFIG))
    edited = FeatureCalculator(_write(tmp_path / "c.yaml", CONFIG.replace("Fourier", "TimeDomainStat")))
    rec = a._feature_set_record()
    assert rec["config_sha256"] == same._feature_set_record()["config_sha256"]
    assert rec["config_sha256"] != edited._feature_set_record()["config_sha256"]
    # an edit that keeps the feature names is still detected
    assert list(a.feature_funcs) == list(edited.feature_funcs)


def test_default_configuration_is_identified():
    calc = FeatureCalculator()
    rec = calc._feature_set_record()
    path = Path(calc.config_path)
    assert rec["config_name"] == path.name
    assert rec["config_sha256"] == hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_skipped_dependencies_are_recorded(tmp_path, suffix):
    _need(suffix)
    config = _write(tmp_path / "deps.yaml", CONFIG + MISSING)
    with pytest.warns(UserWarning, match="a_package_that_does_not_exist"):
        calc = FeatureCalculator(config)
    r = load_results(calc.save(tmp_path / f"res{suffix}", _extract(calc)))
    assert r.meta["missing_dependencies"] == ["a_package_that_does_not_exist"]
    assert r.meta["skipped_operations"] == {
        "correlation.ami_dep": {"missing": ["a_package_that_does_not_exist"], "n_feature_functions": 4}}
    assert r.meta["n_skipped_feature_functions"] == 4
    assert r.meta["n_feature_functions"] == 3


def test_changed_feature_funcs_are_flagged(tmp_path):
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    calc.feature_funcs = dict(list(calc.feature_funcs.items())[:2])
    r = load_results(calc.save(tmp_path / "res.npz", _extract(calc)))
    assert r.meta["feature_functions_modified"] is True and r.meta["n_feature_functions"] == 2


def test_npz_without_metadata_loads(tmp_path):
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    df = _extract(calc)
    path = calc.save(tmp_path / "res.npz", df)
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "meta"}
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    r = load_results(old)
    assert r.meta == {}
    pd.testing.assert_frame_equal(r.values, df, check_names=False)


def test_parquet_without_metadata_loads(tmp_path):
    pytest.importorskip("pyarrow")
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    df = _extract(calc)
    # written by plain pandas, as a user (or an older pyhctsa) might
    df.to_parquet(tmp_path / "old.parquet")
    calc.quality.to_parquet(tmp_path / "old.quality.parquet")
    calc.errors.to_parquet(tmp_path / "old.errors.parquet", index=False)
    r = load_results(tmp_path / "old.parquet")
    assert r.meta == {} and r.calc_time is None
    pd.testing.assert_frame_equal(r.values, df)
