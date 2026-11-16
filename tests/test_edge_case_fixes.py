"""Edge cases outside the default feature set (direct calls with unusual arguments or very short
series), and docstrings that must render as written."""
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import pyhctsa
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import scaling as sc


# ------------------------------------------------------------------------------
# fluctuation_analysis: q <= 0 for the methods other than DFA
# ------------------------------------------------------------------------------
METHODS = ['std', 'iqr', 'range', 'endptdiff', 'rsrange', 'rsrangefit']


@pytest.mark.parametrize('wtf', METHODS)
def test_fluctuations_at_q_zero_are_the_geometric_mean_limit(wtf):
    y = np.cumsum(np.random.default_rng(0).standard_normal(2000))
    taus = np.array([10, 20, 40, 80])
    f0 = sc._fluctuations(y, taus, wtf, 0, 1)  # (raised ZeroDivisionError)
    assert np.all(np.isfinite(f0)) and np.all(f0 > 0)
    for q in (-1e-6, 1e-6):
        np.testing.assert_allclose(sc._fluctuations(y, taus, wtf, q, 1), f0, rtol=1e-4)
    # power means increase with q
    assert np.all(sc._fluctuations(y, taus, wtf, -1, 1) <= f0)
    assert np.all(f0 <= sc._fluctuations(y, taus, wtf, 2, 1))


def test_fluctuations_are_nan_for_q_at_most_zero_if_a_window_does_not_fluctuate():
    y = np.cumsum(np.random.default_rng(1).standard_normal(2000))
    y[100:300] = 0.0  # a flat stretch of the profile: windows of exactly zero spread
    taus = np.array([10, 20, 40, 80])
    for q in (0, -1):  # as for DFA (q = -1 gave F = 0)
        assert np.all(np.isnan(sc._fluctuations(y, taus, 'std', q, 1)))
    assert np.all(sc._fluctuations(y, taus, 'std', 2, 1) > 0)


def test_fluctuation_analysis_runs_at_q_zero():
    x = np.random.default_rng(2).standard_normal(3000)
    for wtf in ('std', 'rsrange', 'dfa'):
        out = sc.fluctuation_analysis(x, 0, wtf)
        assert 0.3 < out['alpha'] < 0.8  # white noise: about 0.5


# ------------------------------------------------------------------------------
# steps_ahead: the best null predictor among those that are defined
# ------------------------------------------------------------------------------
def test_steps_ahead_uses_the_null_predictors_defined_at_each_horizon():
    # 7 samples, 6 steps ahead: one value 6 steps before another, so the last-value predictor is
    # defined but the two-value sliding mean is not (NaN)
    y = np.random.default_rng(0).standard_normal(7)
    y = (y - y.mean()) / y.std(ddof=1)
    out = mf.steps_ahead(y, 'ar', 1, 6)
    assert np.isfinite(out['stde_h6']) and np.isfinite(out['meanabs_h6'])
    # the sample-mean predictor is always defined, so every horizon the model reaches has a benchmark
    for h in range(1, 7):
        assert np.isfinite(out[f'stde_h{h}']) and out[f'stde_h{h}'] > 0


# ------------------------------------------------------------------------------
# docstrings
# ------------------------------------------------------------------------------
def _documented_objects():
    for info in pkgutil.walk_packages(pyhctsa.__path__, 'pyhctsa.'):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional dependency
            continue
        yield info.name, module
        for name, obj in vars(module).items():
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, '__module__', None) == info.name:
                yield f'{info.name}.{name}', obj


def test_no_docstring_has_control_characters():
    # LaTeX in a docstring that is not a raw string needs doubled backslashes: a single one turns
    # \tau, \frac, \rho, \beta, \alpha and \varepsilon into a tab, form feed, carriage return,
    # backspace, bell and vertical tab
    bad = [name for name, obj in _documented_objects()
           if any(c in (getattr(obj, '__doc__', None) or '') for c in '\t\a\b\f\r\v')]
    assert bad == []


def test_persistent_homology_docstring_names_its_outputs():
    doc = nl.persistent_homology.__doc__
    for key in ('maxPersistenceH1', 'totalPersistenceH1', 'persistenceEntropyH1', 'totalPersistenceH0'):
        assert key in doc
    assert 'totH0' not in doc and 'entH1`' not in doc
