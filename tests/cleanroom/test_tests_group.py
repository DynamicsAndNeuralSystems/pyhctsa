"""Tests for the hypothesis-test helpers (group `tests`) whose implementation is plain textbook code.

The tests of the helpers that wrap scipy/arch (KS, variance ratio, sign, signed rank, Jarque-Bera,
Kendall, Phillips-Perron) are in tests/native/test_wp5_stats.py.
"""
import json
import os

import numpy as np
from scipy import stats
from scipy.special import comb

from pyhctsa.operations import hypothesis_tests as ht
from pyhctsa.operations import stationarity as st

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')


def load(name):
    with open(os.path.join(FIX, name + '.json')) as f:
        return json.load(f)


def arr(v):
    return np.atleast_1d(np.array([np.nan if e is None else e for e in np.atleast_1d(v)], dtype=float))


def close(a, b, rtol=0.0, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


# --------------------------------------------------------------------------- KS / Lilliefors
def test_lilliefors_cases():
    for c in load('ks_statistic')['lilliefors_cases']:
        tol = 1e-6 if c['distn'] == 'ev' else 1e-12
        d = ht._lilliefors_statistic(np.array(c['x']), c['distn'])
        assert abs(d - c['kstat']) < tol, (c['distn'], len(c['x']))


def test_lilliefors_small_n():
    for c in load('ks_extra')['lillietest_small_n']:
        x = np.array(c['x'])
        d = ht._lilliefors_statistic(x, c['distn'])
        if c['kstat'] is None:
            assert np.isnan(d)
        else:
            tol = 1e-6 if c['distn'] == 'ev' else 1e-12
            assert abs(d - c['kstat']) < tol


def test_lilliefors_matches_statsmodels():
    from statsmodels.stats.diagnostic import lilliefors
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40)
    assert abs(ht._lilliefors_statistic(x, 'norm') - lilliefors(x, dist='norm')[0]) < 1e-12
    e = rng.exponential(2.0, 40)
    assert abs(ht._lilliefors_statistic(e, 'exp') - lilliefors(e, dist='exp')[0]) < 1e-12


def test_kstest_effect_nan_when_no_finite_cdf():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isnan(ht._kstest_effect(x, lambda z: np.full_like(z, np.nan)))


# --------------------------------------------------------------------------- chi2gof
def _chi2_cdf(case):
    n = case['cdf']
    p = case.get('params')
    if n == 'normcdf':
        if p[1] == 0:
            return lambda z: stats.norm.cdf(z, p[0], 1e-300)
        return lambda z: stats.norm.cdf(z, p[0], p[1])
    if n == 'unifcdf':
        return lambda z: stats.uniform.cdf(z, p[0], p[1] - p[0])
    probs = {'cp': [0.10, 0.05, 0.20, 0.20, 0.20, 0.25],
             'cp2': [0.25, 0.20, 0.20, 0.20, 0.05, 0.10],
             'cp3': [0.12, 0.20, 0.20, 0.20, 0.20, 0.08]}
    key = [k for k in ('cp3', 'cp2', 'cp') if k in n][0]
    ed = np.linspace(0, 10, 7)
    cp = np.concatenate([[0], np.cumsum(probs[key])])
    return lambda z: np.clip(np.interp(z, ed, cp), 0, 1)


def test_chi2gof_cases():
    for c in load('chi2gof')['cases']:
        x = np.array(c['x'])
        if c['distn'] == 'norm':
            f = lambda z, p=c['params']: stats.norm.cdf(z, p[0], p[1])
        elif c['distn'] == 'exp':
            f = lambda z, p=c['params']: stats.expon.cdf(z, scale=p)
        elif c['distn'] == 'uni':
            f = lambda z, p=c['params']: stats.uniform.cdf(z, p[0], p[1] - p[0])
        else:
            f = lambda z, p=c['params']: stats.gumbel_l.cdf(z, p[0], p[1])
        edges, O, E = ht._chi2gof_bins(x, f, c['nbins'])
        assert np.array_equal(O, c['O']), c['name']
        assert close(E, c['E'], rtol=1e-9), c['name']
        assert close(edges, c['edges'], atol=1e-12, rtol=1e-12), c['name']
        assert close(ht._chi2gof_effect(x, f, c['nbins']), c['effect_chi2_over_n'], rtol=1e-9, atol=1e-12)


def test_chi2gof_extra_cases():
    for c in load('chi2gof_extra')['cases']:
        x = np.atleast_1d(np.array(c['x'], dtype=float))
        f = _chi2_cdf(c)
        edges, O, E = ht._chi2gof_bins(x, f, c['nbins'])
        assert np.array_equal(O, c['O']), c['name']
        assert close(E, c['E'], rtol=1e-9, atol=1e-12), c['name']
        assert close(edges, c['edges'], atol=1e-9, rtol=1e-9), c['name']
        ref = np.nan if c['effect_chi2_over_n'] is None else c['effect_chi2_over_n']
        got = ht._chi2gof_effect(x, f, c['nbins'])
        assert close(got, ref, rtol=1e-9, atol=1e-12), (c['name'], got, ref)


# --------------------------------------------------------------------------- variance ratio
def test_vratiotest_constant():
    out = ht._vratiotest(np.ones(50), 2, True)
    assert all(np.isnan(out))


# --------------------------------------------------------------------------- sign / signrank
# --------------------------------------------------------------------------- vartest
def test_vartest_cases():
    for c in load('vartest')['cases']:
        x = arr(c['x'])
        p = ht._vartest_pvalue(x, 1.0)
        # the recorded p of the chi2(29) tail case is off by 2.7e-8 relative from the exact value
        # (checked with mpmath), so the tolerance is 1e-7 rather than the spec's 1e-9
        assert abs(p - c['p']) <= 1e-7 * c['p'] + 1e-300, c['p']


def test_vartest_edge():
    assert ht._vartest_pvalue(np.array([3.0]), 1.0) == 0.0
    assert ht._vartest_pvalue(4 * np.ones(10), 1.0) == 0.0
    assert np.isnan(ht._vartest_pvalue(np.array([]), 1.0))


# --------------------------------------------------------------------------- runs test
def test_runstest_cases():
    for c in load('runstest')['cases']:
        if c['error']:
            continue
        x = arr(c['x'])
        p = ht.runstest_pvalue(x)
        assert abs(p - c['p']) <= 1e-9 * c['p'] + 1e-15, c


def test_runstest_pmf_sums_to_one():
    rng = np.random.default_rng(2)
    for _ in range(5):
        n1, n0 = rng.integers(2, 40, 2)
        N = n1 + n0
        r = np.arange(2, N + 1)
        m = r // 2
        even = ht._log_choose(n1 - 1, m - 1) + ht._log_choose(n0 - 1, m - 1) + np.log(2)
        odd = np.logaddexp(ht._log_choose(n1 - 1, m) + ht._log_choose(n0 - 1, m - 1),
                           ht._log_choose(n1 - 1, m - 1) + ht._log_choose(n0 - 1, m))
        pmf = np.exp(np.where(r % 2 == 0, even, odd) - ht._log_choose(N, n1))
        assert abs(pmf.sum() - 1) < 1e-12


def test_log_choose():
    for n in range(0, 61):
        for k in range(0, n + 1):
            assert abs(np.exp(ht._log_choose(n, k)) / comb(n, k, exact=True) - 1) < 1e-12
    assert ht._log_choose(5, -1) == -np.inf
    assert ht._log_choose(5, 6) == -np.inf
    assert np.shape(ht._log_choose(5, 2)) == ()
    assert ht._log_choose(np.array([4, 5]), 2).shape == (2,)


def test_runstest_edge():
    assert ht.runstest_pvalue(np.ones(10)) == 1.0
    assert ht.runstest_pvalue(np.array([])) == 1.0


# --------------------------------------------------------------------------- Kendall / Pearson
def test_kendall_properties():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(30)
    y = 0.5 * x + rng.standard_normal(30)
    t1, p1 = st._kendall(x, y)
    t2, p2 = st._kendall(y, x)
    t3, p3 = st._kendall(x, -y)
    assert np.isclose(t1, t2) and np.isclose(p1, p2)
    assert np.isclose(t3, -t1) and np.isclose(p3, p1)
    assert all(np.isnan(st._kendall(x, np.where(np.arange(30) == 3, np.nan, y))))


def test_pearson_cases():
    for c in load('kendall_extra')['pearson_cases']:
        x, y = arr(c['x']), arr(c['y'])
        r, p = st._pearson(x, y)
        ref_r = np.nan if c['r'] is None else c['r']
        ref_p = np.nan if c['p'] is None else c['p']
        assert close(r, ref_r, atol=1e-12)
        if abs(ref_r) > 1 - 1e-12:
            assert p < 1e-7
        else:
            assert close(p, ref_p, rtol=1e-8, atol=1e-14), (p, ref_p)

