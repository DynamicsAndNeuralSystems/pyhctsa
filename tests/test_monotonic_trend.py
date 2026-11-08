"""monotonic_trend: the Mann-Kendall tau of a series against time and the Theil-Sen slope.

Exact values against direct computations, robustness to an outlier where the least-squares gradient is
not, the subsampled slope on a long series, degenerate inputs, and the calculator's labels."""

import numpy as np
import pytest
from scipy.stats import kendalltau

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.operations import stationarity as st
from pyhctsa.utils import z_score

FIELDS = {"tau", "sen_slope"}


def _sen_direct(y):
    n = len(y)
    return np.median([(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, n)])


def test_exact_against_direct_computation():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(60)) + 0.1 * np.arange(60)
    out = st.monotonic_trend(y)
    assert set(out) == FIELDS
    assert out["tau"] == pytest.approx(kendalltau(np.arange(60), y)[0])
    assert out["sen_slope"] == pytest.approx(_sen_direct(y))


def test_pure_ramp_and_its_mirror():
    y = 0.3 * np.arange(50) + 2.0
    out = st.monotonic_trend(y)
    assert out["tau"] == pytest.approx(1.0) and out["sen_slope"] == pytest.approx(0.3)
    down = st.monotonic_trend(-y)
    assert down["tau"] == pytest.approx(-1.0) and down["sen_slope"] == pytest.approx(-0.3)


def test_robust_to_an_outlier_where_the_least_squares_gradient_is_not():
    rng = np.random.default_rng(1)
    y = 0.05 * np.arange(200) + rng.standard_normal(200)
    clean = st.monotonic_trend(y)["sen_slope"]
    y_out = y.copy()
    y_out[-1] += 200.0
    assert abs(st.monotonic_trend(y_out)["sen_slope"] - clean) < 0.005
    assert abs(st.trend(y_out)["gradient"] - st.trend(y)["gradient"]) > 0.01


def test_white_noise_has_no_trend():
    rng = np.random.default_rng(2)
    taus = [st.monotonic_trend(rng.standard_normal(500))["tau"] for _ in range(20)]
    assert abs(np.mean(taus)) < 0.03 and np.max(np.abs(taus)) < 0.15


def test_long_series_subsamples_the_slope_but_not_tau():
    rng = np.random.default_rng(3)
    y = 0.002 * np.arange(6000) + 0.1 * rng.standard_normal(6000)
    out = st.monotonic_trend(y)
    assert out["sen_slope"] == pytest.approx(0.002, abs=2e-5)
    assert out["tau"] == pytest.approx(kendalltau(np.arange(6000), y)[0])
    ramp = 1.5 * np.arange(6000)
    assert st.monotonic_trend(ramp)["sen_slope"] == pytest.approx(1.5)   # exact whatever the subsample
    assert st.monotonic_trend(ramp, max_points=100)["sen_slope"] == pytest.approx(1.5)


def test_degenerate_inputs_are_nan():
    for y in (np.ones(30), np.array([1.0, 2.0]), np.array([0.0, 1.0, np.nan, 3.0])):
        out = st.monotonic_trend(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())


def test_calculator_labels_and_zscored_units():
    rng = np.random.default_rng(4)
    y = 0.02 * np.arange(300) + rng.standard_normal(300)
    res = FeatureCalculator().extract([y])
    cols = [c for c in res.columns if c.startswith("monotonic_trend")]
    assert set(cols) == {f"monotonic_trend.{f}" for f in FIELDS}
    z = z_score(y)
    assert res["monotonic_trend.sen_slope"].iloc[0] == pytest.approx(st.monotonic_trend(z)["sen_slope"])
    assert res["monotonic_trend.tau"].iloc[0] == pytest.approx(st.monotonic_trend(y)["tau"])   # rank-based: unchanged by z-scoring
