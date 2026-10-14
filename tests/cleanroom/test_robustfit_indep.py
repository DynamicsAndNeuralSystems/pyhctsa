"""Tests of the bisquare robust regression (``robustfit``).

The estimator is the published Holland-Welsch / DuMouchel-O'Brien bisquare IRLS. The stored
cases in ``tests/cleanroom/data`` are outputs of a reference implementation of the same estimator,
used here only as a cross-check of the algorithm on ordinary data (coefficients to 1e-7
relative, scales and standard errors to 1e-7); accuracy properties (exact recovery, outlier
resistance, the least-squares limit, coverage) are tested in ``tests/native/test_wp2_fitting.py``.
"""
import json
import os
import warnings

import numpy as np
import pytest

from pyhctsa.operations.entropy import _robust_linear_fit_se
from pyhctsa.fitting.robust_regression import BisquareIRLS, bisquare_psi_slope, bisquare_weights
from pyhctsa.fitting import robust_linear_fit as robustfit

DATA = os.path.join(os.path.dirname(__file__), "data")
SQRT_EPS = np.sqrt(np.finfo(float).eps)
REL = 1e-7   # sigma estimates and standard errors (observed worst case ~6e-8)


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)["cases"]


def _arr(v):
    return np.array([np.nan if e is None or e == "NaN" else e for e in np.asarray(v, dtype=object).ravel()], float)


def _x(v):
    rows = [[np.nan if e is None or e == "NaN" else e for e in r] if isinstance(r, list) else
            (np.nan if r is None or r == "NaN" else r) for r in v]
    return np.array(rows, float)


def _by_name(fname):
    return {c["name"]: c for c in _load(fname)}


MAIN = _load("robustfit.json")
EDGE = _by_name("robustfit_edge.json")
LEV = _load("robustfit_leverage1.json")
FLOOR = _load("robustfit_se_floor.json")
PROBES = _load("robustfit_probes.json")


def _fit(case, **opt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BisquareIRLS(_x(case["x"]), _arr(case["y"]), **opt).fit()


# ----------------------------------------------------------------------------- well-conditioned cases
@pytest.mark.parametrize("case", [c for c in MAIN if c["name"] != "exact_line"] + LEV,
                         ids=lambda c: c["name"])
def test_agrees_with_reference_implementation(case):
    """Cross-check on ordinary data: same estimator, so b, scales and se agree to ~1e-7."""
    b, st = robustfit(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], rtol=1e-7, atol=1e-9)
    for key in ("ols_s", "robust_s", "mad_s", "s"):
        np.testing.assert_allclose(st[key], case[key], rtol=REL, err_msg=key)
    np.testing.assert_allclose(st["se"], case["se"], rtol=REL)
    assert st["dfe"] == case["dfe"]


def test_exact_line_is_recovered():
    case = next(c for c in MAIN if c["name"] == "exact_line")
    b, st = robustfit(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], rtol=1e-9, atol=1e-6)
    assert st["s"] < 1e-13 and st["dfe"] == 8
    assert np.all(st["se"] < 1e-13) and np.all(st["w"] > 0.5)


# ----------------------------------------------------------------------------- scale and sigma rules
def test_mad_scale_drops_p_minus_1_smallest_adjusted_residuals():
    """mad_s = median(|r_adj| without its p-1 smallest) / 0.6745 at the final coefficients."""
    for case in MAIN[:3]:
        m = BisquareIRLS(_arr(case["x"]), _arr(case["y"]))
        b, st = m.fit()
        a = np.sort(np.abs((m.y - m.X @ b) / np.sqrt(1 - m.h)))
        np.testing.assert_allclose(st["mad_s"], np.median(a[m.rank - 1:]) / 0.6745, rtol=1e-12)


def test_final_sigma_is_variance_weighted_with_p2_to_n():
    """s = max(robust_s, sqrt((p^2 ols_s^2 + n robust_s^2) / (p^2 + n)))."""
    for case in PROBES:
        if "ols_s" not in case or case["name"].startswith("symmetric"):
            continue
        _, st = _fit_case(case)
        p = len(case["b"])
        n = int(np.sum(~np.isnan(_arr(case["y"]))))
        expected = max(st["robust_s"], np.sqrt((p * p * st["ols_s"] ** 2 + n * st["robust_s"] ** 2) / (p * p + n)))
        np.testing.assert_allclose(st["s"], expected, rtol=1e-13)
        np.testing.assert_allclose(case["s"], expected, rtol=1e-6)   # the reference implementation obeys the same identity


def test_se_is_s_times_unweighted_inverse_gram_diagonal():
    for case in MAIN[:4]:
        x = _arr(case["x"])
        X = np.column_stack([np.ones_like(x), x])
        b, st = robustfit(x, _arr(case["y"]))
        np.testing.assert_allclose(st["se"], st["s"] * np.sqrt(np.diag(np.linalg.inv(X.T @ X))), rtol=1e-12)
        np.testing.assert_allclose(st["covb"], st["s"] ** 2 * np.linalg.inv(X.T @ X), rtol=1e-12)


def _fit_case(case):
    opts = dict(case.get("options", {}))
    x = _x(case["x"])
    y = _arr(case["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BisquareIRLS(x, y, **opts).fit()


@pytest.mark.parametrize("case", [c for c in PROBES if "ols_s" in c], ids=lambda c: c["name"])
def test_designs_options_and_nan_rows(case):
    b, st = _fit_case(case)
    np.testing.assert_allclose(b, case["b"], rtol=1e-7, atol=1e-9)
    for key in ("ols_s", "robust_s", "mad_s", "s"):
        np.testing.assert_allclose(st[key], case[key], rtol=REL, err_msg=key)
    np.testing.assert_allclose(st["se"], case["se"], rtol=REL)
    np.testing.assert_allclose(st["w"], _arr(case["w"]), atol=1e-6)
    np.testing.assert_allclose(st["h"], _arr(case["h"]), atol=1e-6)
    assert st["dfe"] == case["dfe"]


@pytest.mark.parametrize("name", ["probe4_full_stats", "probe4_nan_row", "probe4_noconst", "probe4_tune2p5",
                                  "probe4_two_predictors"])
def test_secondary_statistics(name):
    case = next(c for c in PROBES if c["name"] == name)
    _, st = _fit_case(case)
    np.testing.assert_allclose(st["resid"], _arr(case["resid"]), atol=1e-6)
    np.testing.assert_allclose(st["rstud"], _arr(case["rstud"]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(st["t"], _arr(case["t"]), rtol=1e-6)
    np.testing.assert_allclose(st["p"], _arr(case["p"]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(st["covb"], np.array(case["covb"]).reshape(st["covb"].shape), rtol=1e-7)
    np.testing.assert_allclose(st["coeffcorr"], np.array(case["coeffcorr"]).reshape(st["coeffcorr"].shape),
                               atol=1e-6)


def test_nan_row_gives_nan_in_per_observation_outputs():
    case = EDGE["nan_row"]
    b, st = robustfit(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], rtol=1e-7, atol=1e-9)
    assert np.isnan(st["w"][2]) and np.isnan(st["h"][2]) and np.isnan(st["resid"][2])
    assert np.sum(np.isnan(st["w"])) == 1
    assert st["dfe"] == 4
    np.testing.assert_allclose(st["w"], _arr(case["w"]), atol=1e-6)


def test_too_few_points_raises():
    case = EDGE["n2"]
    with pytest.raises(ValueError, match="Too few observations"):
        robustfit(_arr(case["x"]), _arr(case["y"]))
    with pytest.raises(ValueError):
        robustfit([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])   # only two valid rows


def test_rank_deficient_design_gives_basic_solution():
    case = EDGE["constant_x"]
    with pytest.warns(RuntimeWarning, match="rank deficient, rank = 1"):
        b, st = robustfit(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], atol=1e-14)
    assert b[0] == 0 and st["se"][0] == 0
    np.testing.assert_allclose(st["se"][1], case["se"][1], rtol=REL)
    np.testing.assert_allclose(st["s"], case["s"], rtol=REL)
    assert st["dfe"] == case["dfe"] == 7


def test_majority_identical_y():
    case = EDGE["majority_identical_y"]
    b, st = robustfit(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], atol=1e-14)
    # majority of exactly-fitted points keep (rounding-level-noise) weights near 1, the others get 0
    assert np.all(st["w"][:8] > 0.9) and np.all(st["w"][8:] == 0)
    np.testing.assert_allclose(st["s"], case["s"], rtol=1e-12)
    np.testing.assert_allclose(st["se"], case["se"], rtol=1e-12)
    np.testing.assert_allclose(st["ols_s"], case["ols_s"], rtol=1e-12)


# ----------------------------------------------------------------------------- building blocks
def test_bisquare_functions():
    u = np.array([0.0, 0.5, 1.0, -0.5, 2.0])
    np.testing.assert_allclose(bisquare_weights(u), [1.0, 0.5625, 0.0, 0.5625, 0.0])
    np.testing.assert_allclose(bisquare_psi_slope(u), [1.0, 0.75 * 0.0 + (1 - .25) * (1 - 1.25), 0.0,
                                                       (1 - .25) * (1 - 1.25), 0.0])
    # psi' really is d/du of u (1-u^2)^2
    v = np.array([0.1, 0.3, 0.7]); d = 1e-6
    f = lambda t: t * (1 - t * t) ** 2
    np.testing.assert_allclose(bisquare_psi_slope(v), (f(v + d) - f(v - d)) / (2 * d), rtol=1e-8)


def test_entropy_helper_returns_coefficients_and_standard_errors():
    case = MAIN[0]
    b, se = _robust_linear_fit_se(_arr(case["x"]), _arr(case["y"]))
    np.testing.assert_allclose(b, case["b"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(se, case["se"], rtol=REL)


def test_input_forms():
    case = MAIN[2]
    x, y = _arr(case["x"]), _arr(case["y"])
    b0, _ = robustfit(x, y)
    b1, _ = robustfit(x[None, :], y[None, :])           # row vectors
    b2, _ = robustfit(list(x), list(y))
    b3, _ = robustfit(x[:, None], y)
    np.testing.assert_allclose([b1, b2, b3], [b0] * 3, rtol=1e-13)


def test_outlier_resistance_and_ols_limit():
    rng = np.random.default_rng(0)
    x = np.arange(40.0)
    y = 1.0 + 0.5 * x + 0.1 * rng.standard_normal(40)
    y[[5, 20]] += 50
    b, st = robustfit(x, y)
    assert abs(b[0] - 1.0) < 0.2 and abs(b[1] - 0.5) < 0.01
    assert st["w"][5] == 0 and st["w"][20] == 0
    # very large tuning constant -> ordinary least squares
    b_big, _ = robustfit(x, y, tune=1e9)
    np.testing.assert_allclose(b_big, np.polyfit(x, y, 1)[::-1], rtol=1e-8)
