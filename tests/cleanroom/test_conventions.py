"""Tests for the small convention helpers (clean-room rewrite, group ``conventions``).

Expected values are black-box MATLAB R2026a outputs stored in fixtures/conventions.json.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations import information
from pyhctsa.operations.distribution import _nanargmax_or_zero as _dist_nanargmax, _nanmax_or_nan as _dist_nanmax
from pyhctsa.operations.model_fit import _nanmax_or_nan, _nanmin_or_nan, _tail_mean
from pyhctsa.operations.spectral import _mean_or_nan
from pyhctsa import utils

FIX = json.loads((Path(__file__).parent / "fixtures" / "conventions.json").read_text())


def dec(a):
    """Decode a fixture array to (ndarray in MATLAB shape)."""
    re = [float(s) for s in a["re"]]
    if "im" in a:
        re = [complex(r, float(i)) for r, i in zip(re, a["im"])]
    return np.array(re).reshape(a["size"], order="F")


def ins(case, key):
    return dec(case["inputs"][key])


def out(case, key):
    return dec(case["outputs"][key])


def same(a, b):
    """exact equality with NaN == NaN"""
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or a == b


@pytest.fixture(autouse=True)
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------- round
def test_round_fixture():
    c = FIX["round"][0]
    x = ins(c, "x").ravel()
    y = out(c, "y").ravel()
    vec = utils._round_half_away_array(x)
    assert vec.shape == x.shape
    for xi, yi, vi in zip(x, y, vec):
        assert same(utils._round_half_away(xi), yi), xi
        assert same(vi, yi), xi


def test_round_scalar_and_ties():
    assert isinstance(utils._round_half_away(2.5), float)
    assert utils._round_half_away_array(2.5).shape == ()
    for k in list(range(0, 1000)) + [10**6 - 1, 10**6]:
        assert utils._round_half_away(k + 0.5) == k + 1
        assert utils._round_half_away(-k - 0.5) == -k - 1
    ks = np.arange(0, 10**6 + 1)
    assert np.array_equal(utils._round_half_away_array(ks + 0.5), ks + 1)
    assert np.array_equal(utils._round_half_away_array(-ks - 0.5), -ks - 1)


# ---------------------------------------------------------------- std
def test_std_fixture():
    for c in FIX["std"]:
        cmd = c["command"]
        x = ins(c, "x")
        y = out(c, "y")
        if "M,0,1" in cmd or "R,0,1" in cmd:
            got = utils._sample_std(x, axis=0)
            assert got.shape == (x.shape[1],)
            np.testing.assert_allclose(got, y.ravel(), rtol=1e-14, atol=1e-15)
        elif "M,0,2" in cmd:
            got = utils._sample_std(x, axis=1)
            np.testing.assert_allclose(got, y.ravel(), rtol=1e-14, atol=1e-15)
        else:
            exp = float(y[0, 0])
            xf = x.ravel()
            for got in (float(utils._sample_std(xf)), float(utils._sample_std(x))):
                if math.isnan(exp) or exp == 0:
                    assert same(got, exp), cmd
                else:
                    assert got == pytest.approx(exp, rel=1e-14, abs=0), cmd


def test_std_nested_sd_and_types():
    assert isinstance(utils._sample_std([1, 2, 3]), float)
    assert isinstance(utils._sample_std(np.array([1.0, 2.0])), float)
    assert utils._sample_std(np.zeros((3, 4)), axis=None).shape == ()
    assert utils._sample_std(np.arange(12.0).reshape(3, 4), axis=1).shape == (3,)


def test_std_axis_matches_slices():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((6, 5))
    for ax in (0, 1):
        got = utils._sample_std(m, ax)
        ref = [utils._sample_std(np.take(m, i, axis=1 - ax)) for i in range(m.shape[1 - ax])]
        np.testing.assert_allclose(got, ref, rtol=1e-14)


# ---------------------------------------------------------------- zscore
def test_zscore_fixture():
    for c in FIX["zscore"]:
        x = ins(c, "x")
        z = out(c, "z")
        got = utils._zscore_or_zeros(x.ravel())
        assert got.shape == (z.size,)
        if z.size == 0:
            continue
        zf = z.ravel()
        assert np.array_equal(np.isnan(got), np.isnan(zf))
        ok = ~np.isnan(zf)
        np.testing.assert_allclose(got[ok], zf[ok], rtol=0, atol=1e-14)


# ---------------------------------------------------------------- mean
def test_mean_or_nan_fixture():
    for c in FIX["mean"]:
        x = ins(c, "x").ravel()
        exp = float(out(c, "y")[0, 0])
        got = _mean_or_nan(x)
        assert isinstance(got, float)
        if math.isnan(exp):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(exp, rel=1e-15, abs=0)


# ---------------------------------------------------------------- max/min
def test_maxmin_fixture():
    for c in FIX["maxmin"]:
        x = ins(c, "x")
        xf = x.ravel()
        mx = out(c, "max")
        mn = out(c, "min")
        if xf.size == 0:
            assert math.isnan(_dist_nanmax(xf)) and _dist_nanargmax(xf) == 0
            continue
        if "imax_1based" in c["outputs"]:
            assert same(_dist_nanmax(xf), mx[0, 0])
            assert _dist_nanargmax(xf) == int(out(c, "imax_1based")[0, 0]) - 1
        assert same(_nanmax_or_nan(xf), mx[0, 0])
        assert same(_nanmin_or_nan(xf), mn[0, 0])
        assert same(_nanmax_or_nan(x), mx[0, 0])  # 2-D, axis=None


def test_maxmin_nan_first_and_axis():
    assert _dist_nanargmax(np.array([np.nan, -np.inf])) == 1
    a = np.array([[1.0, np.nan], [np.nan, np.nan]])
    r = _nanmax_or_nan(a, axis=0)
    assert r[0] == 1.0 and math.isnan(r[1])
    r = _nanmin_or_nan(a, axis=1)
    assert r[0] == 1.0 and math.isnan(r[1])


# ---------------------------------------------------------------- tail mean
def test_tail_mean_fixture():
    for c in FIX["tail_mean"]:
        x = ins(c, "x").ravel()
        for which in ("high", "low"):
            exp = float(out(c, which)[0, 0])
            got = _tail_mean(x, which)
            if math.isnan(exp):
                assert math.isnan(got)
            else:
                assert got == pytest.approx(exp, rel=1e-15, abs=0)
    with pytest.raises(ValueError):
        _tail_mean(np.arange(4.0), "mid")
    assert _tail_mean(np.arange(16.0).reshape(4, 4), "high") == pytest.approx(13.5)


# ---------------------------------------------------------------- Pearson r
def test_pearson_fixture():
    for c in FIX["corr"]:
        a = ins(c, "a").ravel()
        b = ins(c, "b").ravel()
        exp = float(out(c, "r")[0, 0])
        got = information._pearson_r(a, b)
        assert isinstance(got, float)
        if math.isnan(exp):
            assert math.isnan(got), c["inputs"]
        else:
            assert abs(got - exp) <= 1e-14


def test_pearson_properties():
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(3, 500))
        a = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4) + rng.standard_normal() * 100
        c = rng.choice([-3.7, 3.7])
        r = information._pearson_r(a, c * a + 11)
        assert abs(r) <= 1.0
        assert abs(r - np.sign(c)) <= 1e-14
    assert math.isnan(information._pearson_r(np.arange(4.0), np.full(4, 5.0)))
    assert information._pearson_r(np.ones(3), np.arange(3.0)) != information._pearson_r(np.ones(3), np.arange(3.0))


# ---------------------------------------------------------------- rcond
def test_rcond_1norm_matches_exact_condition_number():
    """information._rcond_1norm = 1 / cond_1(A) (exact); 0 for singular, NaN for NaN input, inf for empty."""
    for c in FIX["rcond"]:
        a = ins(c, "A")
        got = information._rcond_1norm(a)
        assert isinstance(got, float)
        if a.size == 0:
            assert math.isinf(got)
        elif not np.all(np.isfinite(a)):
            assert math.isnan(got)
        elif np.linalg.matrix_rank(a) < a.shape[0] and np.linalg.cond(a, 1) > 1e15:
            assert got < 1e-15
        else:
            assert got == pytest.approx(1.0 / np.linalg.cond(a, 1), rel=1e-9, abs=1e-18)
    assert information._rcond_1norm(np.zeros((3, 3))) == 0.0


def test_profile_trend_jump_uses_sd():
    from pyhctsa.operations.pre_process import _profile_trend_jump
    f = np.random.default_rng(0).standard_normal(50).cumsum()
    trend, jump = _profile_trend_jump(f)
    assert np.isfinite(trend) and np.isfinite(jump)
