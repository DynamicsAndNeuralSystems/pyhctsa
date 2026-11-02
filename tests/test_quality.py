"""Quality labels: classification, extraction, and saving/loading with the values."""
import numpy as np
import pandas as pd
import pytest

from pyhctsa import Quality, classify, load_results, quality_labels, quality_summary
from pyhctsa.calculator import FeatureCalculator
from pyhctsa.quality import quality_from_values


# 1. Classifying single outputs and arrays
@pytest.mark.parametrize("raw, value, code", [
    (1.5, 1.5, Quality.GOOD),
    (True, 1.0, Quality.GOOD),
    (np.int64(3), 3.0, Quality.GOOD),
    (np.complex128(4 + 0j), 4.0, Quality.GOOD),
    (np.array([7.0]), 7.0, Quality.GOOD),
    (np.nan, np.nan, Quality.NAN),
    (np.inf, np.nan, Quality.INF),
    (-np.inf, np.nan, Quality.NEG_INF),
    (2 + 3j, np.nan, Quality.COMPLEX),
    (None, np.nan, Quality.EMPTY),
    (np.array([]), np.nan, Quality.EMPTY),
    ("hello", np.nan, Quality.NON_SCALAR),
    (np.arange(20.0), np.nan, Quality.NON_SCALAR),
    ([1, 2], np.nan, Quality.NON_SCALAR),
])
def test_classify(raw, value, code):
    v, q = classify(raw)
    assert q == code
    assert (np.isnan(v) and np.isnan(value)) or v == value


def test_quality_from_values():
    x, q = quality_from_values(np.array([[1.0, np.nan], [np.inf, 2 + 1j]]))
    assert q.dtype == np.uint8
    assert q.tolist() == [[Quality.GOOD, Quality.NAN], [Quality.INF, Quality.COMPLEX]]
    assert x[0, 0] == 1.0 and np.isnan(x).sum() == 3


def test_codes_match_hctsa():
    # hctsa's TS_Quality codes (TS_CalculateFeatureVector.m); must never change
    assert [int(Quality[k]) for k in ("GOOD", "ERROR", "NAN", "INF", "NEG_INF", "COMPLEX", "EMPTY",
                                      "LINK_ERROR")] == list(range(8))


def test_labels_and_summary():
    q = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    assert quality_labels(q).tolist() == [["good", "error"], ["NaN", "NaN"]]
    assert quality_summary(q) == {"NaN": 2, "good": 1, "error": 1}


# 2. Extraction
def _toy_calculator(funcs):
    """A calculator whose feature functions are ``funcs`` (name -> callable)."""
    calc = FeatureCalculator.__new__(FeatureCalculator)
    calc.config_path = None
    calc.feature_funcs = dict(funcs)
    calc._config_funcs = dict(funcs)
    calc.quality = calc.errors = calc.calc_time = None
    return calc


def _two_outputs(y):
    return {"a": float(np.mean(y)), "b": float(np.std(y))}


def _raises_when_negative_mean(y):
    if np.mean(y) < 0:
        raise ValueError("negative mean")
    return {"a": 1.0, "b": 2.0}


def test_raising_feature_keeps_columns_and_codes_error():
    calc = _toy_calculator({"ok": _two_outputs, "flaky": _raises_when_negative_mean,
                            "inf": lambda y: np.inf, "cplx": lambda y: 1 + 1j})
    rng = np.random.default_rng(0)
    pos, neg = rng.standard_normal(200) + 5, rng.standard_normal(200) - 5
    df = calc.extract([pos, neg], labels=["pos", "neg"])
    assert list(df.columns) == ["ok.a", "ok.b", "flaky.a", "flaky.b", "inf", "cplx"]
    assert df.dtypes.eq(np.float64).all()
    q = calc.quality
    assert q.dtypes.eq(np.uint8).all()
    assert (q.loc["pos", ["flaky.a", "flaky.b"]] == Quality.GOOD).all()
    assert (q.loc["neg", ["flaky.a", "flaky.b"]] == Quality.ERROR).all()
    assert (q["inf"] == Quality.INF).all() and (q["cplx"] == Quality.COMPLEX).all()
    # values are finite exactly where the quality is GOOD
    assert (np.isfinite(df.to_numpy()) == (q.to_numpy() == Quality.GOOD)).all()
    assert calc.errors.to_dict("records") == [
        {"instance": "neg", "feature": "flaky", "error_type": "ValueError", "message": "negative mean"}]
    assert list(calc.calc_time.columns) == ["ok", "flaky", "inf", "cplx"]
    assert calc.calc_time.notna().all().all()


def test_invalid_series_is_rejected_not_fatal():
    calc = _toy_calculator({"ok": _two_outputs})
    rng = np.random.default_rng(1)
    with pytest.warns(UserWarning, match="1 of 2 time series are invalid"):
        df = calc.extract([rng.standard_normal(200), np.ones(200)], labels=["x", "const"])
    assert df.loc["x"].notna().all() and df.loc["const"].isna().all()
    assert (calc.quality.loc["const"] == Quality.REJECTED).all()
    err = calc.errors.set_index("instance").loc["const"]
    assert err["error_type"] == "InvalidInput" and err["message"] == "constant"


def test_all_series_invalid_raises():
    calc = _toy_calculator({"ok": _two_outputs})
    with pytest.raises(ValueError, match="One or more time series instances are invalid"):
        calc.extract([np.ones(200)])


def test_real_features_have_stable_columns(module_config):
    # a library module, on a series where some of its functions raise
    calc = FeatureCalculator(module_config("spectral"))
    rng = np.random.default_rng(2)
    clean = calc.extract(rng.standard_normal(500))
    short = calc.extract(rng.standard_normal(101))
    assert list(short.columns) == list(clean.columns)
    assert (np.isfinite(short.to_numpy()) == (calc.quality.to_numpy() == Quality.GOOD)).all()


# 3. Saving and loading
@pytest.mark.parametrize("suffix", [".npz", ".parquet"])
def test_save_load_round_trip(tmp_path, suffix):
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    calc = _toy_calculator({"ok": _two_outputs, "flaky": _raises_when_negative_mean,
                            "inf": lambda y: np.inf})
    rng = np.random.default_rng(3)
    df = calc.extract([rng.standard_normal(200) + 5, rng.standard_normal(200) - 5], labels=[1, 2])
    path = calc.save(tmp_path / f"res{suffix}", df, meta={"note": "test"})
    r = load_results(path)
    pd.testing.assert_frame_equal(r.values, df, check_names=False, check_index_type=False)
    assert (r.quality.to_numpy() == calc.quality.to_numpy()).all()
    assert r.quality.dtypes.eq(np.uint8).all()
    assert list(r.quality.columns) == list(df.columns)
    assert len(r.errors) == 1 and r.errors.iloc[0]["feature"] == "flaky"
    assert r.calc_time.shape == calc.calc_time.shape
    assert r.meta["note"] == "test" and r.meta["n_features"] == df.shape[1]
    if suffix == ".npz":
        assert list(r.values.index) == [1, 2]  # index label types survive
    # loaded frames are plain DataFrames: they concatenate
    pd.concat([r.values, r.values])


def test_save_needs_extract(tmp_path):
    calc = _toy_calculator({"ok": _two_outputs})
    with pytest.raises(RuntimeError, match="call extract"):
        calc.save(tmp_path / "x.npz", pd.DataFrame())


def test_save_rejects_unknown_suffix(tmp_path):
    calc = _toy_calculator({"ok": _two_outputs})
    df = calc.extract(np.random.default_rng(4).standard_normal(200))
    with pytest.raises(ValueError, match="Unsupported suffix"):
        calc.save(tmp_path / "x.csv", df)


def test_quality_from_values_object_array():
    x, q = quality_from_values(np.array([1.0, None, 2 + 1j, "x", np.inf], dtype=object))
    assert q.tolist() == [Quality.GOOD, Quality.EMPTY, Quality.COMPLEX, Quality.NON_SCALAR, Quality.INF]
    assert x[0] == 1.0 and np.isnan(x[1:]).all()
