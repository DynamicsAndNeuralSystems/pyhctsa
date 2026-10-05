"""scripts/check_hctsa_yaml.py (the comparison of hctsa.yaml with hctsa's INP_mops / INP_ops) and the shape of the
default feature set."""
import importlib.util
import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from pyhctsa.calculator import FeatureCalculator, _PREPROCESS_LABELS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_hctsa_yaml.py"
HCTSA_YAML = Path(__file__).resolve().parents[1] / "pyhctsa" / "configurations" / "hctsa.yaml"
spec = importlib.util.spec_from_file_location("check_hctsa_yaml", SCRIPT)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)


# ---- parsing hctsa's feature-set files ----
def test_split_args_respects_nesting_and_quotes():
    assert chk.split_args("x_z,1,'Fourier'") == ["x_z", "1", "'Fourier'"]
    assert chk.split_args("x_z,{'ac',1},[1,2,3],'a,b'") == ["x_z", "{'ac',1}", "[1,2,3]", "'a,b'"]
    assert chk.split_args("zscore(BF_PreProcess(x_z,'decimate_ac1e')),5,0.2") == \
        ["zscore(BF_PreProcess(x_z,'decimate_ac1e'))", "5", "0.2"]


def test_call_parsing():
    c = chk.Call("EN_SampEn(zscore(BF_PreProcess(x_z,'decimate_ac1e')),5,0.2)", "EN_SampEn_5_02_dec")
    assert c.func == "EN_SampEn" and c.args == ["5", "0.2"]
    assert c.input == "zscore(BF_PreProcess(x_z,'decimate_ac1e'))"
    assert chk.INPUTS[c.input] == (True, False, "decimate_ac1e")


def test_every_input_has_a_calculator_preprocess():
    for z, ab, pre in chk.INPUTS.values():
        assert pre is None or pre in _PREPROCESS_LABELS


def test_mvalue():
    assert chk.mvalue("[]") is None and chk.mvalue("true") is True and chk.mvalue("'ac1e'") == "ac1e"
    assert chk.mvalue("1:4") == [1.0, 2.0, 3.0, 4.0] and chk.mvalue("[0,3,6]") == [0.0, 3.0, 6.0]
    assert chk.mvalue("{'ac',1}") == ["ac", 1.0] and chk.mvalue("1/exp(1)") == pytest.approx(1 / 2.718281828, rel=1e-8)
    assert chk.norm(chk.mvalue("'4'")) == chk.norm(4) and chk.norm("") is None


# ---- matching, on a small made-up hctsa tree ----
@pytest.fixture
def tiny_hctsa(tmp_path):
    h = tmp_path / "hctsa"
    (h / "FeatureSets").mkdir(parents=True)
    (h / "Operations").mkdir()
    (h / "FeatureSets" / "INP_mops_hctsa.txt").write_text(
        "CO_AutoCorr(x_z,1,'Fourier')\tAC_1\n"
        "CO_AutoCorr(x_z,2,'Fourier')\tAC_2\n"
        "CO_AutoCorr(abs(x_z),10,'Fourier')\tAC_abs_10\n"
        "DN_Mean(x,'arith')\tDN_Mean_arith\n"        # a function pyhctsa implements with no yaml config here
        "ZZ_NotPorted(x_z)\tZZ_NotPorted\n")
    (h / "FeatureSets" / "INP_ops_hctsa.txt").write_text(
        "AC_1\tAC_1\tcorrelation\nAC_2\tAC_2\tcorrelation\nAC_abs_10\tAC_abs_10\tcorrelation\n"
        "DN_Mean_arith\tDN_Mean_arith\tlocation\nZZ_NotPorted\tZZ_NotPorted\tx\n")
    (h / "Operations" / "CO_AutoCorr.m").write_text(
        "function out = CO_AutoCorr(y, tau, whatMethod)\nif nargin < 3 || isempty(whatMethod)\n"
        "    whatMethod = 'Fourier';\nend\n")
    (h / "Operations" / "DN_Mean.m").write_text("function out = DN_Mean(y, meanType)\n")
    return h


def _write_yaml(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_compare_finds_missing_and_extra(tiny_hctsa, tmp_path):
    cfg = {"correlation": {"autocorr": {"base_name": "autocorr", "ordered_args": ["tau"], "legacy_name": "CO_AutoCorr",
                                        "configs": [{"tau": 1, "zscore": True}, {"tau": 3, "zscore": True}]}}}
    res = chk.compare(tiny_hctsa, _write_yaml(tmp_path, cfg), log=lambda *a: None)
    missing = sorted(c.label for c in res["missing"]["CO_AutoCorr"])
    assert missing == ["AC_2", "AC_abs_10"]            # AC_abs_10 needs abs: True; lag 2 is not registered
    assert [e[0] for e in res["extra"]] == ["autocorr_3"]   # lag 3 is not in hctsa
    assert res["not_implemented"] == ["ZZ_NotPorted"]
    assert "DN_Mean" in res["unregistered"]                  # implemented (distribution.mean) but not in the yaml


def test_compare_clean_when_registered_the_same(tiny_hctsa, tmp_path):
    cfg = {"correlation": {"autocorr": {"base_name": "autocorr", "ordered_args": ["tau"], "legacy_name": "CO_AutoCorr",
                                        "configs": [{"tau": [1, 2], "zscore": True},
                                                    {"tau": 10, "zscore": True, "abs": True}]}}}
    res = chk.compare(tiny_hctsa, _write_yaml(tmp_path, cfg), log=lambda *a: None)
    assert not res["extra"] and not res["missing"].get("CO_AutoCorr")
    assert set(res["matched_labels"]) == {"autocorr_1", "autocorr_2", "autocorr_10_abs"}


def test_compare_reports_deregistered_function_and_duplicate_labels(tiny_hctsa, tmp_path):
    cfg = {"correlation": {"autocorr": {"base_name": "autocorr", "ordered_args": ["tau"], "legacy_name": "CO_AutoCorr",
                                        "configs": [{"tau": 1, "zscore": True}, {"tau": 1, "zscore": True}]}},
           "stationarity": {"dyn_win": {"base_name": "dyn_win", "legacy_name": "SY_DynWin", "configs": [{"zscore": True}]}}}
    res = chk.compare(tiny_hctsa, _write_yaml(tmp_path, cfg), log=lambda *a: None)
    assert [d[:3] for d in res["deregistered"]] == [("stationarity", "dyn_win", "SY_DynWin")]
    assert [d[0] for d in res["duplicate_labels"]] == ["autocorr_1"]   # the second config replaces the first


def test_suggest_writes_matching_configs(tiny_hctsa, tmp_path):
    cfg = {"correlation": {"autocorr": {"base_name": "autocorr", "ordered_args": ["tau"], "legacy_name": "CO_AutoCorr",
                                        "configs": [{"tau": 1, "zscore": True}]}}}
    p = _write_yaml(tmp_path, cfg)
    res = chk.compare(tiny_hctsa, p, log=lambda *a: None)
    ents = chk.suggest_entries(tiny_hctsa, res, chk.py_signatures(chk.REPO / "pyhctsa" / "operations"))
    lines = [cfg_ for e in ents if e["name"] == "autocorr" for cfg_, _ in e["lines"]]
    assert "{tau: 2, zscore: True}" in lines and "{tau: 10, zscore: True, abs: True}" in lines


# ---- the default feature set itself ----
def _load():
    return yaml.safe_load(HCTSA_YAML.read_text(encoding="utf-8"))


def test_labels_are_unique():
    # two configs with the same label silently replace each other in the calculator
    cfg = _load()
    from pyhctsa.calculator import _build_label
    seen = Counter()
    for module, fns in cfg.items():
        for name, meta in fns.items():
            for _, kw, z, ab, pre, _, _ in chk.expand(meta):
                seen[_build_label(meta.get("base_name"), kw, meta.get("ordered_args") or [], z, ab, pre)] += 1
    assert not [k for k, v in seen.items() if v > 1]


def test_every_function_names_its_hctsa_function_and_known_preprocess():
    for module, fns in _load().items():
        for name, meta in fns.items():
            assert meta.get("legacy_name"), f"{module}.{name} has no legacy_name"
            for item in meta["configs"]:
                assert item.get("preprocess") in (None, *_PREPROCESS_LABELS), f"{module}.{name}: {item}"
                assert not ("select" in item and "exclude" in item), f"{module}.{name}: both select and exclude"


@pytest.mark.skipif(not os.environ.get("HCTSA_DIR"), reason="set HCTSA_DIR to a hctsa checkout to compare against it")
def test_default_feature_set_matches_hctsa():
    res = chk.compare(Path(os.environ["HCTSA_DIR"]), HCTSA_YAML, log=lambda *a: None)
    assert not any(res["missing"].values()), {f: [c.label for c in v] for f, v in res["missing"].items() if v}
    assert not res["extra"] and not res["deregistered"]
    assert not res.get("duplicate_labels")


def test_default_calculator_builds_every_config():
    n = sum(len(list(chk.expand(meta))) for fns in _load().values() for meta in fns.values())
    assert len(FeatureCalculator().feature_funcs) == n
