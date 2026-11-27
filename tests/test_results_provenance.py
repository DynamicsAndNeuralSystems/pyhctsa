"""Saved results record the feature set that produced them (version, configuration, skipped
optional dependencies), and files without that record still load."""
import hashlib
import json
import os
from datetime import datetime
from functools import partial
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
    assert "config_path" not in m and m["config_name"] == "small.yaml"
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


def _stored_meta(path: Path, suffix: str) -> list:
    """The metadata record as written in every file of one save (read without load_results)."""
    if suffix == ".npz":
        with np.load(path) as f:
            return [json.loads(str(f["meta"]))]
    import pyarrow.parquet as pq
    stem = path.with_suffix("")
    files = [path] + [Path(f"{stem}.{t}.parquet") for t in ("quality", "errors", "calc_time")]
    return [json.loads(pq.read_schema(p).metadata[b"pyhctsa"]) for p in files]


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)
    elif o is not None:
        yield str(o)


@pytest.mark.parametrize("custom", [False, True], ids=["default_config", "custom_config"])
@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_saved_metadata_reveals_no_paths_or_user(tmp_path, monkeypatch, suffix, custom):
    """A shared file must not expose where, or by whom, it was written."""
    _need(suffix)
    import getpass
    import os
    work = tmp_path / "project"
    work.mkdir()
    monkeypatch.chdir(work)
    if custom:
        calc = FeatureCalculator(_write(work / "small.yaml", CONFIG))
    else:
        calc = FeatureCalculator()
        calc.feature_funcs = dict(list(calc.feature_funcs.items())[:5])
    # a feature function whose error message names a file in the home directory
    calc.feature_funcs["reads_a_file"] = partial(_raise_file_not_found,
                                                 path=str(Path.home() / "project" / "data.csv"))
    path = calc.save(work / f"res{suffix}", _extract(calc))
    forbidden = {p for p in (str(Path.home()), os.getcwd(), str(work), str(tmp_path)) if len(p) > 1}
    try:
        user = getpass.getuser()
    except Exception:  # no user name to find (e.g. some containers)
        user = None
    for meta in _stored_meta(path, suffix):
        assert "config_path" not in meta
        assert meta["config_name"] == ("small.yaml" if custom else "hctsa.yaml")
        for s in _strings(meta):
            for f in forbidden:
                assert f not in s, (s, f)
            if user and len(user) > 2:  # a very short name would match by chance
                assert user not in s, (s, user)
    # nor do the error messages, once the home directory is replaced by ~
    messages = list(load_results(path).errors["message"])
    assert sum("~" in m and "data.csv" in m for m in messages) == 2  # one per series
    for s in messages:
        for f in forbidden:
            assert f not in s, (s, f)
        if user and len(user) > 2:
            assert user not in s, (s, user)
    # the calculator itself still knows where its configuration came from
    assert calc.config_path is not None


def test_old_files_with_config_path_still_load(tmp_path):
    """Files written before config_path was dropped hold it; they load, and keep it in meta."""
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    df = _extract(calc)
    path = calc.save(tmp_path / "res.npz", df, meta={"config_path": "/somewhere/small.yaml"})
    r = load_results(path)
    assert r.meta["config_path"] == "/somewhere/small.yaml"
    pd.testing.assert_frame_equal(r.values, df, check_names=False)


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


# --- the home directory in recorded error messages ------------------------------------------------

def _raise_file_not_found(ts, path):
    raise FileNotFoundError(2, "No such file or directory", path)


def _raise_with_message(ts, message):
    raise ValueError(message)


def _set_home(monkeypatch, value):
    """Point the home directory at ``value`` (HOME on POSIX, USERPROFILE on Windows)."""
    from pyhctsa import utils
    monkeypatch.setenv("HOME", value)
    monkeypatch.setenv("USERPROFILE", value)
    utils._home_pattern.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_home_pattern():
    """The home pattern is cached per process: recompute it for, and after, each test."""
    from pyhctsa import utils
    utils._home_pattern.cache_clear()
    yield
    utils._home_pattern.cache_clear()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    _set_home(monkeypatch, str(home))
    return home


@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_error_messages_hide_the_home_directory(tmp_path, fake_home, suffix):
    _need(suffix)
    calc = FeatureCalculator(_write(tmp_path / "small.yaml", CONFIG))
    data = fake_home / "project" / "data.csv"
    calc.feature_funcs = dict(calc.feature_funcs,
                              reads_a_file=partial(_raise_file_not_found, path=str(data)),
                              says_why=partial(_raise_with_message, message="series too smooth"))
    path = calc.save(tmp_path / f"res{suffix}", _extract(calc))
    expected = str(FileNotFoundError(2, "No such file or directory",
                                     os.path.join("~", "project", "data.csv")))
    for errors in (calc.errors, load_results(path).errors):
        assert len(errors) == 4  # each function, on each of the two series
        for f, m in zip(errors["feature"], errors["message"]):
            # a message without a path is unchanged
            assert m == (expected if f == "reads_a_file" else "series too smooth")
            assert str(fake_home) not in m


def test_home_written_with_other_separators(fake_home):
    from pyhctsa.utils import _scrub_home
    other = "\\" if os.sep == "/" else "/"
    h = str(fake_home)
    # as written, with the other separator, and with backslashes doubled by repr
    for form in (h, h.replace(os.sep, other), h.replace(os.sep, "\\\\")):
        assert _scrub_home(f"cannot open '{form}{os.sep}x.txt'") == f"cannot open '~{os.sep}x.txt'"
        assert _scrub_home(f"found in {form}.") == "found in ~."
        assert _scrub_home(form) == "~"
    # a directory whose name only starts with the home directory's is left alone
    for name in (h + "2", h + "-old", h + ".bak"):
        assert _scrub_home(f"see {name}{os.sep}x") == f"see {name}{os.sep}x"
    if os.name == "nt":  # Windows paths ignore case (drive letter included)
        assert _scrub_home(h.swapcase() + "\\x") == "~\\x"


def test_resolved_home_is_also_replaced(tmp_path, monkeypatch):
    from pyhctsa.utils import _scrub_home
    real = tmp_path / "real_home"
    real.mkdir()
    link = tmp_path / "linked_home"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create a symbolic link here")
    _set_home(monkeypatch, str(link))
    for home in (str(link), os.path.realpath(str(link))):
        assert _scrub_home(os.path.join(home, "x")) == os.path.join("~", "x")


@pytest.mark.parametrize("home", ["", "/", "\\", "C:\\", "relative_dir", "~"])
def test_degenerate_home_changes_nothing(monkeypatch, home):
    from pyhctsa.utils import _scrub_home
    _set_home(monkeypatch, home)
    msg = "No such file: '/data/x.csv' or 'C:\\data\\x.csv' (relative_dir/x, ~/y)"
    assert _scrub_home(msg) == msg


def test_scrubbing_never_raises(monkeypatch):
    from pyhctsa import utils

    def broken():
        raise RuntimeError("no home directory")
    monkeypatch.setattr(utils, "_home_pattern", broken)
    assert utils._scrub_home("a message") == "a message"
    monkeypatch.undo()
    assert utils._scrub_home(None) is None
