"""A missing REQUIRED dependency must raise; only OPTIONAL ones (declared per feature in hctsa.yaml)
lead to skipping just the features that need them, with a single clear warning."""
import importlib

import pytest

import pyhctsa.calculator as calc_mod
from pyhctsa.calculator import FeatureCalculator

OPTIONAL = "ripser"


def test_missing_required_dependency_raises(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.endswith(".wavelet"):
            raise ModuleNotFoundError("No module named 'pywt'", name="pywt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(calc_mod.importlib, "import_module", fake_import)
    with pytest.raises(ImportError) as err:
        FeatureCalculator()
    msg = str(err.value)
    assert "PyWavelets" in msg and "pip install PyWavelets" in msg and "wavelet" in msg


def test_missing_optional_dependency_skips_only_its_features(monkeypatch):
    real_check = calc_mod._check_optional_deps
    monkeypatch.setattr(calc_mod, "_check_optional_deps",
                        lambda dep: False if dep == OPTIONAL else real_check(dep))
    with pytest.warns(UserWarning) as record:  # a warning the user sees, not a log record
        calc = calc_mod.FeatureCalculator()
    skipped = calc._skipped_functions
    assert skipped and all(deps == [OPTIONAL] for _, deps in skipped)
    assert [n for n, _ in skipped] == ["nonlinearity.persistent_homology"]
    assert not any(label.startswith("persistent_homology") for label in calc.feature_funcs)
    assert len(calc.feature_funcs) > 100  # everything else still loaded
    assert len(record) == 1
    message = str(record[0].message)
    assert "persistent_homology" in message and f"pip install {OPTIONAL}" in message
    assert record[0].filename == __file__  # attributed to the code that built the calculator


def test_nothing_skipped_when_all_installed():
    assert FeatureCalculator()._skipped_functions == []
