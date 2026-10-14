"""Tests of ``hypothesis_tests._fit_distribution_cdf`` against stored reference values.

The curve-fit, polynomial, goodness-of-fit, robust-fit and distribution-fit tests live in
``tests/native/test_wp2_fitting.py``.
"""
import json
import os

import numpy as np
import pytest

from pyhctsa.operations.hypothesis_tests import _fit_distribution_cdf

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "fit")
_SPECIAL = {"NaN": np.nan, "Inf": np.inf, "-Inf": -np.inf}


def dec(v):
    if isinstance(v, str) and v in _SPECIAL:
        return _SPECIAL[v]
    if isinstance(v, list):
        return [dec(i) for i in v]
    if isinstance(v, dict):
        return {k: dec(i) for k, i in v.items()}
    return v


def load(name):
    with open(os.path.join(FIX, name + ".json")) as fh:
        return dec(json.load(fh))


def arr(v):
    return np.atleast_1d(np.asarray(v, dtype=float))


# ---------------------------------------------------------------- distribution fits
@pytest.mark.parametrize("c", load("distribution_fits")["fit_distribution_cdf_cases"], ids=lambda c: c["name"])
def test_fit_distribution_cdf(c):
    z = arr(c["z"])
    for d in ("norm", "ev", "uni", "beta", "rayleigh", "exp", "gamma", "logn", "wbl"):
        ref = c[d]
        x = arr(ref["x_rescaled"]) if d == "beta" else arr(c["x"])
        zz = arr(ref["z"]) if (d == "beta" and "z" in ref) else z
        cdf, npar = _fit_distribution_cdf(x, d)
        assert npar == ref["num_params"]
        assert np.max(np.abs(cdf(zz) - arr(ref["cdf"]))) <= 1e-6, d


def test_fit_distribution_cdf_unknown():
    with pytest.raises(ValueError):
        _fit_distribution_cdf(np.arange(1.0, 5.0), "nope")
