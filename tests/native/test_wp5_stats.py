"""Accuracy tests for the hypothesis-test, stationarity and distribution helpers (WP5).

References are scipy / statsmodels / arch, analytic results and simulation; none depends on MATLAB.
"""
import warnings

import numpy as np
import pytest
from scipy import stats

from pyhctsa.operations import distribution as D
from pyhctsa.operations import extreme_events as EE
from pyhctsa.operations import hypothesis_tests as H
from pyhctsa.operations import stationarity as S
from pyhctsa.utils import _sample_std


def _ar1(n, phi, rng):
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def _uniform_ks_p(p):
    return stats.kstest(np.asarray(p), 'uniform').pvalue


# --------------------------------------------------------------------------- KS and Lilliefors
def test_kstest_effect_equals_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = np.round(rng.standard_normal(200), 1)            # many tied values
    cdf = stats.norm(0.1, 1.2).cdf
    assert H._kstest_effect(x, cdf) == pytest.approx(stats.kstest(x, cdf).statistic, abs=1e-15)


def test_kstest_effect_hand_value_and_nan():
    # one observation at 0.5 against U(0,1): D = max(F_n - F) = 0.5
    assert H._kstest_effect(np.array([0.5]), stats.uniform.cdf) == pytest.approx(0.5)
    assert np.isnan(H._kstest_effect(np.array([1.0, 2.0, 3.0]), lambda z: np.full_like(z, np.nan)))


def test_distribution_ks_small_for_correct_family():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    assert H.distribution_test(x, 'ks', 'norm') < 0.03
    assert H.distribution_test(x, 'ks', 'exp') != H.distribution_test(x, 'ks', 'norm')
    assert H.distribution_test(rng.uniform(size=5000), 'ks', 'uni') < 0.03
    assert H.distribution_test(x, 'ks', 'uni') > 0.1          # a normal sample is not uniform


def test_ev_mle_recovers_parameters_and_stationary():
    rng = np.random.default_rng(1)
    x = stats.gumbel_l.rvs(loc=0.5, scale=2.0, size=100000, random_state=rng)
    mu, sd = H._ev_mle(x)
    assert mu == pytest.approx(0.5, rel=0.02) and sd == pytest.approx(2.0, rel=0.01)
    # maximum of the likelihood: perturbations lower it
    ll = lambda m, s: stats.gumbel_l.logpdf(x, m, s).sum()
    for dm, ds in [(1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)]:
        assert ll(mu, sd) >= ll(mu + dm, sd + ds)
    assert all(np.isnan(H._ev_mle(np.ones(10))))


# --------------------------------------------------------------------------- variance ratio
def test_variance_ratio_equals_arch():
    from arch.unitroot import VarianceRatio
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(400))
    for q in (2, 5, 12):
        for iid in (True, False):
            p, z, r = H._vratiotest(y, q, iid)
            ref = VarianceRatio(y, lags=q, trend='c', debiased=True, robust=not iid, overlap=True)
            assert (p, z, r) == pytest.approx((ref.pvalue, ref.stat, ref.vr), rel=1e-12, abs=1e-15)


def test_variance_ratio_random_walk_and_ar_increments():
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.standard_normal(100000))
    assert H._vratiotest(rw, 4, True)[2] == pytest.approx(1.0, abs=0.03)
    # increments AR(1) with phi: VR(2) = 1 + phi
    y = np.cumsum(_ar1(100000, 0.5, rng))
    assert H._vratiotest(y, 2, True)[2] == pytest.approx(1.5, abs=0.03)
    # negatively autocorrelated increments (white noise level series): ratio below 1
    assert H._vratiotest(rng.standard_normal(5000), 4, True)[2] < 0.5


def test_variance_ratio_pvalues_uniform_under_random_walk():
    rng = np.random.default_rng(4)
    p = [H._vratiotest(np.cumsum(rng.standard_normal(300)), 3, False)[0] for _ in range(300)]
    assert _uniform_ks_p(p) > 0.01


def test_variance_ratio_edges():
    assert all(np.isnan(H._vratiotest(np.ones(50), 2, True)))
    with pytest.raises(ValueError):
        H._vratiotest(np.arange(10.0), 1, True)
    with pytest.raises(ValueError):
        H._vratiotest(np.arange(10.0), 5, True)
    out = H.variance_ratio_test(np.cumsum(np.random.default_rng(0).standard_normal(300)),
                                [2, 3, 4], [0, 1, 0])
    assert out['minratio'] <= out['meanratio'] <= out['maxratio']


# --------------------------------------------------------------------------- sign tests
def test_signtest_equals_binomial():
    rng = np.random.default_rng(5)
    for n in (7, 40, 99, 100, 101, 400):
        x = rng.standard_normal(n) + 0.1
        npos = int(np.sum(x > 0))
        assert H._signtest_pvalue(x) == pytest.approx(stats.binomtest(npos, n, 0.5).pvalue, rel=1e-12)
        assert H._signtest_pvalue(rng.permutation(x)) == pytest.approx(H._signtest_pvalue(x), rel=1e-12)
        assert H._signtest_pvalue(-x) == pytest.approx(H._signtest_pvalue(x), rel=1e-12)


def test_signtest_hand_values_and_edges():
    assert H._signtest_pvalue(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.25)    # 2 * (1/2)^3
    assert H._signtest_pvalue(np.array([1.0, -1.0])) == pytest.approx(1.0)
    assert H._signtest_pvalue(np.array([0.0, 0.0, np.nan, 1.0, 2.0, 3.0])) == pytest.approx(0.25)  # 0 and NaN dropped
    assert H._signtest_pvalue(np.zeros(5)) == 1.0
    assert H._signtest_pvalue(np.array([])) == 1.0


def test_signrank_equals_scipy():
    rng = np.random.default_rng(6)
    for n in (5, 12, 15, 16, 40, 120):
        x = rng.standard_normal(n) + 0.3
        ref = stats.wilcoxon(x, zero_method='wilcox', correction=False, method='auto').pvalue
        assert H._signrank_pvalue(x) == pytest.approx(ref, rel=1e-12)
        assert H._signrank_pvalue(-x) == pytest.approx(ref, rel=1e-12)
    # exact distribution for a small sample without ties
    x = rng.standard_normal(10) + 0.5
    assert H._signrank_pvalue(x) == pytest.approx(stats.wilcoxon(x, method='exact').pvalue, rel=1e-12)
    # ties: normal approximation with the tie-corrected variance
    xt = np.round(rng.standard_normal(60), 1) + 0.2
    xt = xt[xt != 0]
    ref = stats.wilcoxon(xt, zero_method='wilcox', correction=False, method='approx').pvalue
    assert H._signrank_pvalue(xt) == pytest.approx(ref, rel=1e-12)


def test_signrank_hand_values_and_edges():
    # all positive, n = 5 (W+ = 15 or 0): p = 2 / 2^5
    assert H._signrank_pvalue(np.arange(1.0, 6.0)) == pytest.approx(2 / 32)
    assert H._signrank_pvalue(np.zeros(5)) == 1.0
    assert np.isnan(H._signrank_pvalue(np.array([np.nan])))
    assert H._signrank_pvalue(np.array([2.5])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- Jarque-Bera
@pytest.mark.parametrize('n', [4, 10, 30, 100, 500, 1999, 2000, 4000])
def test_jbtest_pvalues_uniform_under_normality(n):
    rng = np.random.default_rng(100 + n)
    p = [H._jbtest_pvalue(rng.standard_normal(n)) for _ in range(1500)]
    assert _uniform_ks_p(p) > 0.005
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.015)


def test_jbtest_pvalues_are_not_clamped():
    rng = np.random.default_rng(8)
    assert H._jbtest_pvalue(rng.exponential(size=500)) < 1e-6              # far below the old 0.001 floor
    p = np.array([H._jbtest_pvalue(rng.standard_normal(100)) for _ in range(400)])
    assert p.max() > 0.9 and p.min() < 0.01                                 # spans (0, 1), not [0.001, 0.5]


def test_jbtest_monotone_in_statistic_and_asymptotic():
    # p-value decreases with the statistic at fixed n (scale the kurtosis through a mixture)
    rng = np.random.default_rng(9)
    base = rng.standard_normal(200)
    ps = [H._jbtest_pvalue(np.where(np.abs(base) > 1.5, base * s, base)) for s in (1.0, 1.5, 2.0, 3.0)]
    assert all(np.diff(ps) <= 1e-12)
    x = rng.standard_t(5, size=3000)
    d = x - x.mean()
    jb = len(x) / 6 * (np.mean(d ** 3) ** 2 / np.mean(d ** 2) ** 3
                       + (np.mean(d ** 4) / np.mean(d ** 2) ** 2 - 3) ** 2 / 4)
    assert H._jbtest_pvalue(x) == pytest.approx(stats.chi2.sf(jb, 2), rel=1e-12)   # chi2(2) for n >= 2000
    assert H._jbtest_pvalue(x) == pytest.approx(stats.jarque_bera(x).pvalue, rel=1e-9)


def test_jbtest_table_integrity():
    from pyhctsa.toolboxes.distribution_fits import _jbtest_null_table as T
    assert T.ALPHAS.min() == pytest.approx(1e-4) and T.ALPHAS.max() == pytest.approx(1 - 1e-4)
    assert np.all(np.diff(T.ALPHAS) > 0)
    assert np.all(np.diff(T.CRITICAL_VALUES, axis=1) <= 1e-12)             # decreasing in alpha
    i = list(T.N_NODES).index(2000)
    j = int(np.argmin(np.abs(T.ALPHAS - 0.05)))
    assert T.CRITICAL_VALUES[i, j] == pytest.approx(stats.chi2.isf(T.ALPHAS[j], 2), rel=0.03)
    cv, alphas = H._jb_critical_values(57)
    assert np.all(np.diff(cv) <= 0)


def test_jbtest_edge_cases():
    assert H._jbtest_pvalue(np.array([1.0, 2.0])) == 1.0
    assert H._jbtest_pvalue(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0
    p = H._jbtest_pvalue(np.array([1.0, 2.0, np.nan, 4.0, 5.0, 7.0]))
    assert p == H._jbtest_pvalue(np.array([1.0, 2.0, 4.0, 5.0, 7.0]))
    for bad in (np.array([5.0]), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError):
            H._jbtest_pvalue(bad)


# --------------------------------------------------------------------------- Kendall
def test_kendall_equals_scipy():
    rng = np.random.default_rng(10)
    for n in (4, 9, 12, 40, 80):
        x = rng.standard_normal(n)
        y = 0.4 * x + rng.standard_normal(n)
        for (a, b) in [(x, y), (np.round(x, 1), np.round(y, 1))]:
            tau, p = S._kendall(a, b)
            ref = stats.kendalltau(a, b, variant='b', method='auto')
            assert tau == pytest.approx(ref.statistic, rel=1e-12)
            assert p == pytest.approx(ref.pvalue, rel=1e-12)


def test_kendall_properties():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(30)
    y = 0.5 * x + rng.standard_normal(30)
    t1, p1 = S._kendall(x, y)
    t2, p2 = S._kendall(y, x)
    t3, p3 = S._kendall(x, -y)
    assert np.isclose(t1, t2) and np.isclose(p1, p2)
    assert np.isclose(t3, -t1) and np.isclose(p3, p1)
    assert all(np.isnan(S._kendall(x, np.where(np.arange(30) == 3, np.nan, y))))
    assert all(np.isnan(S._kendall(x, np.ones(30))))
    assert S._kendall(np.arange(8.0), np.arange(8.0))[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- Phillips-Perron
@pytest.mark.parametrize('model,trend', [('ar', 'n'), ('ard', 'c'), ('ts', 'ct')])
@pytest.mark.parametrize('stat,tt', [('t1', 'tau'), ('t2', 'rho')])
def test_pp_test_equals_arch(model, trend, stat, tt):
    from arch.unitroot import PhillipsPerron
    rng = np.random.default_rng(11)
    y = np.cumsum(rng.standard_normal(300)) + 0.02 * np.arange(300)
    for q in (0, 3):
        out = S.pp_test(y, lags=q, model=model, test_statistic=stat)
        ref = PhillipsPerron(y, lags=q, trend=trend, test_type=tt)
        assert out['stat'] == pytest.approx(ref.stat, rel=1e-12)
        assert out['pvalue'] == pytest.approx(ref.pvalue, rel=1e-12)


def test_pp_test_regression_summaries_match_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(12)
    y = np.cumsum(rng.standard_normal(250))
    n = len(y)
    for model, cols in (('ar', 1), ('ard', 2), ('ts', 3)):
        T = n - 1
        X = [y[:-1]]
        if model in ('ard', 'ts'):
            X.insert(0, np.ones(T))
        if model == 'ts':
            X.insert(1, np.arange(T, dtype=float))   # trend regressor t = 0, ..., T-1
        res = sm.OLS(y[1:], np.column_stack(X)).fit()
        out = S.pp_test(y, lags=2, model=model)
        assert out['coeff1'] == pytest.approx(res.params[0], rel=1e-9)
        assert out['rmse'] == pytest.approx(np.sqrt(res.scale), rel=1e-9)
        # Gaussian log-likelihood at the unbiased variance, per observation of the series
        k = cols
        ll = -0.5 * T * np.log(2 * np.pi * res.scale) - 0.5 * (T - k)
        assert out['loglikelihood'] == pytest.approx(ll / n, rel=1e-9)
        assert out['AIC'] == pytest.approx((2 * k - 2 * ll) / n, rel=1e-9)
        assert out['BIC'] == pytest.approx((k * np.log(T) - 2 * ll) / n, rel=1e-9)


def test_pp_pvalues_uniform_under_unit_root_and_reject_stationary():
    rng = np.random.default_rng(13)
    p = [S.pp_test(np.cumsum(rng.standard_normal(250)), lags=3, model='ard')['pvalue'] for _ in range(300)]
    assert _uniform_ks_p(p) > 0.01
    assert S.pp_test(_ar1(500, 0.5, rng), lags=3, model='ard')['pvalue'] < 0.01
    assert S.pp_test(np.cumsum(rng.standard_normal(2000)), lags=3)['pvalue'] > 0.05
    # not clamped: strongly stationary data give p far below 0.001
    assert S.pp_test(rng.standard_normal(2000), lags=0, model='ard')['pvalue'] < 1e-10


def test_pp_test_contracts_and_errors():
    rng = np.random.default_rng(14)
    y = _ar1(200, 0.5, rng)
    out = S.pp_test(y, lags=[0, 1, 2, 3, 4, 5], model='ar', test_statistic='t1')
    assert set(out) == {'minpValue', 'meanpValue', 'lagmaxp', 'lagminp', 'meanstat', 'minBIC'}
    ps = [S.pp_test(y, lags=q)['pvalue'] for q in range(6)]
    assert out['minpValue'] == pytest.approx(min(ps)) and out['meanpValue'] == pytest.approx(np.mean(ps))
    assert out['lagmaxp'] == int(np.argmax(ps)) and out['lagminp'] == int(np.argmin(ps))
    single = S.pp_test(y, lags=2)
    assert set(single) == {'pvalue', 'stat', 'coeff1', 'loglikelihood', 'AIC', 'BIC', 'HQC', 'rmse'}
    for kw in ({'model': 'xx'}, {'test_statistic': 't3'}):
        with pytest.raises(ValueError):
            S.pp_test(y, **kw)
    with pytest.raises(ValueError):
        S.pp_test(np.r_[y, np.inf])
    with pytest.raises(ValueError):
        S.pp_test(y[:20], lags=25)
    assert S.pp_test(np.r_[y, np.nan], lags=2)['stat'] == pytest.approx(single['stat'], rel=1e-12)
    const = S.pp_test(np.ones(50), lags=1)
    assert np.isnan(const['stat']) and np.isnan(const['pvalue'])


# --------------------------------------------------------------------------- peaks
def test_peak_intervals_known_peaks():
    # sum of equally spaced Gaussian bumps: peaks at known positions, equal intervals
    t = np.arange(600.0)
    centers = np.arange(50, 600, 50)
    y = sum(np.exp(-0.5 * ((t - c) / 4.0) ** 2) for c in centers)
    out = S.peak_intervals(y, min_prom=1)
    assert out['cvInt'] == pytest.approx(0.0, abs=1e-12)
    assert out['acInt1'] == 0.0                                  # equal intervals: defined as 0
    # prominence in sd units equals the bump height / sd(y)
    assert out['meanProm'] == pytest.approx(1.0 / np.std(y, ddof=1), rel=1e-3)


def test_peak_intervals_matches_scipy_find_peaks():
    from scipy.signal import find_peaks
    rng = np.random.default_rng(15)
    y = np.sin(np.arange(1000) / 7.0) + 0.3 * rng.standard_normal(1000)
    z = (y - y.mean()) / y.std(ddof=1)
    locs, props = find_peaks(z, prominence=1)
    out = S.peak_intervals(y, min_prom=1)
    assert out['meanProm'] == pytest.approx(props['prominences'].mean(), rel=1e-12)
    ipi = np.diff(locs)
    assert out['cvInt'] == pytest.approx(np.std(ipi, ddof=1) / np.mean(ipi), rel=1e-12)


def test_peak_intervals_degenerate():
    out = S.peak_intervals(np.ones(100))
    assert all(np.isnan(v) for v in out.values())


# --------------------------------------------------------------------------- stationarity helpers
def test_local_global_unicg_uses_evenly_spaced_rounded_half_up_indices():
    y = np.arange(11, dtype=float)
    # n = 5 over indices 0..10: 0, 2.5, 5, 7.5, 10 -> 0, 3, 5, 8, 10 (exact ties rounded up)
    assert S.local_global(y, 'unicg', 5)['absmean'] == pytest.approx(np.mean([0, 3, 5, 8, 10]))
    y = np.arange(1.0, 21.0)
    idx = np.floor(np.arange(6) * 19 / 5 + 0.5).astype(int)          # n = 6 over 0..19, no ties
    assert S.local_global(y, 'unicg', 6)['absmean'] == pytest.approx(np.mean(y[idx]))


def test_std_nth_deriv_single_difference_convention():
    assert S.std_nth_deriv(np.array([1.0, 3.0]), 1) == 0.0           # std of one value is 0
    y = np.random.default_rng(16).standard_normal(50)
    assert S.std_nth_deriv(y, 3) == pytest.approx(np.std(np.diff(y, 3), ddof=1))


def test_sample_std_convention():
    x = np.random.default_rng(17).standard_normal((6, 4))
    assert _sample_std(x) == pytest.approx(np.std(x, ddof=1))
    assert _sample_std(x, axis=0) == pytest.approx(np.std(x, axis=0, ddof=1))
    assert _sample_std([3.0]) == 0.0 and np.isnan(_sample_std([])) and np.isnan(_sample_std([np.nan]))


def test_cross_prediction_iqr_is_hazen_quantile():
    xperr = np.abs(np.random.default_rng(18).standard_normal((5, 5)))
    out = S._cross_prediction_stats(xperr)
    q = np.quantile(xperr.ravel(), [0.25, 0.75], method='hazen')
    assert out['iqr'] == pytest.approx(q[1] - q[0])


# --------------------------------------------------------------------------- distribution
def test_histogram_mode_do_simple_false_is_deprecated():
    y = np.random.default_rng(19).standard_normal(500)
    ref = D.histogram_mode(y, 10)
    with pytest.warns(DeprecationWarning):
        assert D.histogram_mode(y, 10, do_simple=False) == ref
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert D.histogram_mode(y, 10, do_simple=True) == ref


def test_nan_max_helpers():
    v = np.array([np.nan, 2.0, 5.0, 5.0, np.nan])
    assert D._nanmax_or_nan(v) == 5.0 and D._nanargmax_or_zero(v) == 2
    assert np.isnan(D._nanmax_or_nan(np.full(3, np.nan))) and D._nanargmax_or_zero(np.full(3, np.nan)) == 0
    assert np.isnan(D._nanmax_or_nan(np.array([])))


def test_density_edge_analytic_crossings():
    ratio = 100.0
    # normal: density falls to 1/100 of its peak at mu +- sigma sqrt(2 ln 100)
    pdf = stats.norm(1.0, 2.0).pdf
    thresh = pdf(1.0) / ratio
    half = 2.0 * np.sqrt(2 * np.log(ratio))
    assert D._density_edge(pdf, 1.0, +1, 0.02, thresh) == pytest.approx(1.0 + half, rel=1e-9)
    assert D._density_edge(pdf, 1.0, -1, 0.02, thresh) == pytest.approx(1.0 - half, rel=1e-9)
    # exponential: at scale * ln 100, which is also its 0.99 quantile
    e = stats.expon(scale=3.0)
    assert D._density_edge(e.pdf, 0.0, +1, 0.03, e.pdf(0) / ratio) == pytest.approx(3.0 * np.log(100), rel=1e-9)
    assert e.ppf(0.99) == pytest.approx(3.0 * np.log(100), rel=1e-12)
    # uniform: the jump at the support edges
    u = stats.uniform(2.0, 4.0)
    assert D._density_edge(u.pdf, 4.0, -1, 0.04, u.pdf(4.0) / ratio) == pytest.approx(2.0, abs=1e-9)
    assert D._density_edge(u.pdf, 4.0, +1, 0.04, u.pdf(4.0) / ratio) == pytest.approx(6.0, abs=1e-9)


def test_compare_ks_fit_properties():
    rng = np.random.default_rng(20)
    x = rng.standard_normal(20000)
    out = D.compare_ks_fit(x, 'norm')
    assert out['adiff'] < 0.05 and out['relent'] < 0.01 and abs(out['peaksepy']) < 0.05
    # adiff, olapint and relent do not depend on the scale of the data
    out2 = D.compare_ks_fit(10.0 * x + 3.0, 'norm')
    for k in ('adiff', 'olapint', 'relent'):
        assert out2[k] == pytest.approx(out[k], rel=1e-3, abs=1e-4)
    # a wrong family fits worse
    assert D.compare_ks_fit(x, 'uni')['adiff'] > out['adiff']
    # gamma with shape < 1 (density unbounded at 0) and Weibull with shape < 1 are handled
    g = rng.gamma(0.6, 2.0, 5000)
    for name, data in (('gamma', g), ('wbl', rng.weibull(0.7, 5000) + 1e-3)):
        o = D.compare_ks_fit(data, name)
        assert all(np.isfinite(v) for v in o.values()) and 0 <= o['adiff'] <= 2
    for bad in (np.full(100, 3.0), rng.standard_normal(100)):
        assert all(np.isnan(v) for v in D.compare_ks_fit(bad, 'exp').values())


# --------------------------------------------------------------------------- extreme events
def test_interval_stats_single_interval_has_zero_cv():
    mean, cv = EE._interval_stats(np.array([2, 5]))
    assert mean == 3.0 and cv == 0.0
    assert all(np.isnan(v) for v in EE._interval_stats(np.array([4])))


def test_moving_threshold_iqr_uses_hazen_quantiles():
    y = np.abs(np.random.default_rng(21).standard_normal(300))
    out = EE.moving_threshold(y)
    assert out['iqrq'] >= 0 and out['maxq'] >= out['medianq'] >= out['minq']


# ------------------------------------------------------------------ time axes start at 0
def test_trend_intercepts_are_values_at_t_zero():
    from pyhctsa.operations.stationarity import trend
    N = 200
    t = np.arange(N)
    y = 3.0 + 0.5 * t + np.random.default_rng(40).normal(size=N)
    out = trend(y)
    assert out['gradient'] == pytest.approx(np.polyfit(t, y, 1)[0], rel=1e-10)
    assert out['intercept'] == pytest.approx(np.polyfit(t, y, 1)[1], rel=1e-10)
    yc = np.cumsum(y)
    assert out['interceptYC'] == pytest.approx(np.polyfit(t, yc, 1)[1], rel=1e-10)
    exact = trend(3.0 + 0.5 * t)          # a noiseless line: the intercept is its value at the first sample
    assert exact['intercept'] == pytest.approx(3.0, abs=1e-9)


def test_outlier_include_event_times_are_zero_based():
    from pyhctsa.operations.distribution import outlier_include
    N = 400
    y = 5.0 + np.random.default_rng(41).normal(size=N) * 1e-3      # every sample is an event
    out = outlier_include(y, 'pos', fixed_thresh=1.0)
    # mean and median event time (N-1)/2 sit exactly at the centre, 0, of the [-1, 1] scale
    assert out['meanRelTime'] == pytest.approx(0.0, abs=1e-12)
    assert out['medianRelTime'] == pytest.approx(0.0, abs=1e-12)
