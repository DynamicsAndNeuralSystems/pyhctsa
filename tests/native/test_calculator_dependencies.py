"""A missing REQUIRED dependency must raise; only OPTIONAL ones (declared per feature in hctsa.yaml)
lead to skipping just the features that need them, with a single clear warning."""
import importlib
import logging

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


def test_missing_optional_dependency_skips_only_its_features(monkeypatch, caplog):
    real_check = calc_mod._check_optional_deps
    monkeypatch.setattr(calc_mod, "_check_optional_deps",
                        lambda dep: False if dep == OPTIONAL else real_check(dep))
    with caplog.at_level(logging.WARNING, logger=calc_mod.logger.name):
        calc = calc_mod.FeatureCalculator()
    skipped = calc._skipped_functions
    assert skipped and all(deps == [OPTIONAL] for _, deps in skipped)
    assert [n for n, _ in skipped] == ["nonlinearity.persistent_homology"]
    assert not any(label.startswith("persistent_homology") for label in calc.feature_funcs)
    assert len(calc.feature_funcs) > 100  # everything else still loaded
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "persistent_homology" in warnings[0].getMessage() and f"pip install {OPTIONAL}" in warnings[0].getMessage()


def test_nothing_skipped_when_all_installed():
    assert FeatureCalculator()._skipped_functions == []
