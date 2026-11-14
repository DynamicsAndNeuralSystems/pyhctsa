"""A mistake in a configuration file is reported when the calculator is built, naming the entry."""
import warnings

import numpy as np
import pytest

from pyhctsa.calculator import FeatureCalculator

GOOD = """
correlation:
  autocorr:
    base_name: ac
    configs:
      - {tau: [1, 2, 3], method: 'Fourier', zscore: True}
    ordered_args: ['tau']
"""


def _build(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return FeatureCalculator(str(path))


def test_a_valid_config_builds_without_warnings(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calc = _build(tmp_path, GOOD)
    assert list(calc.feature_funcs) == ['ac_1', 'ac_2', 'ac_3']


@pytest.mark.parametrize("text, message", [
    # a parameter the function does not have (a misspelled parameter or meta key)
    (GOOD.replace("tau: [1, 2, 3]", "taus: [1, 2, 3]"), r"sets 'taus'.*Did you mean 'tau'"),
    (GOOD.replace("zscore: True", "zcore: True"), r"sets 'zcore'.*Did you mean 'zscore'"),
    # a function the module does not have
    (GOOD.replace("autocorr:", "autocor:"), r"no function 'autocor'.*Did you mean 'autocorr'"),
    # meta keys of the wrong type: 'False' in quotes is a (truthy) string
    (GOOD.replace("zscore: True", "zscore: 'False'"), r"'zscore'.*must be true or false"),
    (GOOD.replace("zscore: True", "zscore: True, select: []"), r"'select'.*non-empty list"),
    (GOOD.replace("zscore: True", "zscore: True, select: ['a'], exclude: ['b']"), "both 'select' and 'exclude'"),
    # two configs with one label: ordered_args leaves out the parameter that differs
    (GOOD.replace("ordered_args: ['tau']", "ordered_args: ['method']"), r"Two configs give the feature label 'ac_\w+'.*ordered_args"),
    # the entry's keys at the indentation of its name
    ("correlation:\n  autocorr:\n  base_name: ac\n  configs:\n    - {tau: 1}\n", "must be a mapping"),
    ("correlation:\n  autocorr:\n    configs: {tau: 1}\n", "'configs'.*list of mappings"),
    # nothing in the file, or in a module
    ("", "configuration is empty"),
    ("correlation:\n", "lists no operations"),
])
def test_config_mistakes_raise_when_the_calculator_is_built(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, text)


def test_unknown_entry_key_warns_and_legacy_name_is_accepted(tmp_path):
    with pytest.warns(UserWarning, match=r"Unknown key 'ordered_arg'.*Did you mean 'ordered_args'"):
        _build(tmp_path, GOOD.replace("ordered_args: ['tau']", "ordered_arg: ['tau']").replace("[1, 2, 3]", "1"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(tmp_path, GOOD.replace("base_name: ac", "base_name: ac\n    legacy_name: CO_AutoCorr"))


SELECT = """
symbolic:
  binary_stats:
    base_name: binary_stats
    configs:
      - {binary_method: 'mean', zscore: True, KEY: ['pupstat2', 'NAME']}
    ordered_args: ['binary_method']
"""


@pytest.mark.parametrize("key", ["select", "exclude"])
def test_select_or_exclude_name_that_is_not_an_output_warns(tmp_path, key):
    y = np.random.default_rng(0).standard_normal(200)
    calc = _build(tmp_path, SELECT.replace("KEY", key).replace("NAME", "pstretch1"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = calc.extract(y)
    calc = _build(tmp_path, SELECT.replace("KEY", key).replace("NAME", "pstrech1"))
    with pytest.warns(UserWarning, match=rf"binary_stats_mean: 'pstrech1' \(named in {key}\) not among its outputs"):
        typo = calc.extract(y)
    if key == "select":
        assert list(ok.columns) == ['binary_stats_mean.pupstat2', 'binary_stats_mean.pstretch1']
        assert list(typo.columns) == ['binary_stats_mean.pupstat2']
    else:
        assert typo.shape[1] == ok.shape[1] + 1


def test_default_configuration_passes_every_check():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calc = FeatureCalculator()
    assert len(calc.feature_funcs) == len(set(calc.feature_funcs)) > 1000
