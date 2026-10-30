"""Parallel extraction with LocalDistributor gives exactly the serial result, including when Python is
embedded in another program (e.g. R through reticulate), where dill no longer sees the standard
library as importable."""
import sys

import numpy as np
import pandas as pd
import pytest

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.distribute import (LocalDistributor, _compute_config_features_for_chunk,
                                _compute_features_for_chunk)

rng = np.random.default_rng(7)
SERIES = [rng.standard_normal(200), np.cumsum(rng.standard_normal(200)),
          np.sin(np.arange(200) / 5) + 0.2 * rng.standard_normal(200)]


@pytest.fixture(scope="module")
def distributor():
    dist = LocalDistributor(n_workers=2)
    yield dist
    dist.close()


@pytest.fixture
def calc(module_config):
    # the entropy module is fast and uses !range, select/exclude and preprocess
    return FeatureCalculator(module_config("entropy"))


def test_parallel_matches_serial(calc, distributor):
    serial = calc.extract(SERIES)
    pd.testing.assert_frame_equal(calc.extract(SERIES, distributor=distributor), serial, check_exact=True)


def test_parallel_matches_serial_when_embedded(calc, distributor, tmp_path, monkeypatch):
    serial = calc.extract(SERIES)
    # embedded interpreters (reticulate) report the virtualenv as the base prefix, so dill's
    # _is_builtin_module no longer recognises stdlib modules and tries to pickle them by value
    for name in ("base_prefix", "base_exec_prefix", "exec_prefix", "prefix"):
        monkeypatch.setattr(sys, name, str(tmp_path))
    pd.testing.assert_frame_equal(calc.extract(SERIES, distributor=distributor), serial, check_exact=True)


def test_parallel_with_a_subset_of_features(calc, distributor):
    # a reordered subset of the configured features is rebuilt in the workers by label
    labels = list(calc.feature_funcs)[::-3]
    calc.feature_funcs = {label: calc.feature_funcs[label] for label in labels}
    task, kwargs = calc._parallel_task()
    assert task is _compute_config_features_for_chunk and kwargs["labels"] == labels
    serial = calc.extract(SERIES)
    pd.testing.assert_frame_equal(calc.extract(SERIES, distributor=distributor), serial, check_exact=True)


def test_functions_not_from_the_config_are_sent_as_is(calc):
    calc.feature_funcs["series_length"] = len
    task, kwargs = calc._parallel_task()
    assert task is _compute_features_for_chunk and kwargs["feature_funcs"] is calc.feature_funcs


def test_parsed_config_rebuilds_the_same_features(calc):
    # the copy kept for the workers still has the meta params (zscore, abs, select, ...) that
    # building removes from calc.config
    _, kwargs = calc._parallel_task()
    assert list(FeatureCalculator._from_config(kwargs["config"]).feature_funcs) == list(calc.feature_funcs)
