from typing import Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, logsumexp
from scipy.stats import binomtest, chi2, expon, gumbel_l, kstest, norm, wilcoxon

from ..robust import bf_runs_z
from ..utils import ljung_box_pvalue

def _fit_distribution_cdf(x: np.ndarray, the_distn: str) -> tuple:
    """Fit a distribution to data; return its CDF and parameter count.

    Parameters
    ----------
    x : np.ndarray
        Data (already restricted by the caller to the support of the distribution).
    the_distn : str
        One of ``'norm'``, ``'ev'``, ``'uni'``, ``'beta'``, ``'rayleigh'``, ``'exp'``,
        ``'gamma'``, ``'logn'``, ``'wbl'``.

    Returns
    -------
    (callable, int)
        ``cdf(z)`` of the maximum-likelihood fit and the number of fitted parameters.

    Notes
    -----
    Closed-form estimators: normal (mean, unbiased standard deviation), uniform (min, max),
    Rayleigh ``b = sqrt(sum(x^2)/(2n))`` (Johnson, Kotz & Balakrishnan, *Continuous
    Univariate Distributions* vol. 1, ch. 18), exponential (mean), lognormal (mean and
    unbiased standard deviation of ``log x``). Extreme value, beta, gamma and Weibull use
    the maximum-likelihood fits in ``pyhctsa.toolboxes.distribution_fits.distfits``.
    """
    from scipy import stats as _st
    from ..toolboxes.distribution_fits.distfits import betafit, evfit, gamfit, wblfit
    x = np.ravel(np.asarray(x, dtype=float))
    if the_distn == 'norm':
        frozen = _st.norm(loc=float(np.mean(x)), scale=float(np.std(x, ddof=1)))
        return frozen.cdf, 2
    if the_distn == 'ev':
        mu, sigma = evfit(x)
        return _st.gumbel_l(loc=mu, scale=sigma).cdf, 2
    if the_distn == 'uni':
        lo, hi = float(np.min(x)), float(np.max(x))
        return _st.uniform(loc=lo, scale=hi - lo).cdf, 2
    if the_distn == 'beta':
        a, b = betafit(x)
        return _st.beta(a, b).cdf, 2
    if the_distn == 'rayleigh':
        b = float(np.sqrt(np.sum(x ** 2) / (2 * x.size)))
        return _st.rayleigh(scale=b).cdf, 1
    if the_distn == 'exp':
        return _st.expon(scale=float(np.mean(x))).cdf, 1
    if the_distn == 'gamma':
        a, b = gamfit(x)
        return _st.gamma(a, scale=b).cdf, 2
    if the_distn == 'logn':
        lx = np.log(x)
        return _st.lognorm(s=float(np.std(lx, ddof=1)), scale=float(np.exp(np.mean(lx)))).cdf, 2
    if the_distn == 'wbl':
        scale, shape = wblfit(x)
        return _st.weibull_min(shape, scale=scale).cdf, 2
    raise ValueError(f"Unknown distribution '{the_distn}'.")


def _ev_mle(x: np.ndarray) -> tuple:
    """Maximum-likelihood location and scale of the type-I extreme value (minimum) distribution.

    Uses ``scipy.stats.gumbel_l.fit`` (root-finding on the profile likelihood equation for the
    scale). NaN for constant or non-finite data, where the scale is not identified.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)) or not np.ptp(x) > 0:
        return np.nan, np.nan
    mu, sigma = gumbel_l.fit(x)
    return float(mu), float(sigma)


def _lilliefors_statistic(x: np.ndarray, the_distn: str) -> float:
    """
    Lilliefors (KS) statistic against the distribution fitted to the data.

    The two-sided Kolmogorov-Smirnov distance between the empirical CDF of ``x`` and the
    normal ('norm'; sample standard deviation with ``n - 1``), exponential ('exp'; mean
    equal to the sample mean) or type-I extreme value ('ev'; maximum likelihood) CDF whose
    parameters are estimated from ``x``. NaN for fewer than 4 observations or a degenerate fit.

    Notes
    -----
    Lilliefors (1967) JASA 62:399-402; Lilliefors (1969) JASA 64:387-389.
    """
    x = np.asarray(x, dtype=float).ravel()
    if the_distn not in ('norm', 'exp', 'ev'):
        raise ValueError(f"Lilliefors statistic is not defined for distribution '{the_distn}'.")
    if len(x) < 4:
        return np.nan
    if the_distn == 'norm':
        mu, sd = np.mean(x), np.std(x, ddof=1)
        if not sd > 0:
            return np.nan
        cdf = lambda z: norm.cdf(z, mu, sd)
    elif the_distn == 'exp':
        m = np.mean(x)
        if not m > 0:
            return np.nan
        cdf = lambda z: expon.cdf(z, scale=m)
    else:
        mu, sd = _ev_mle(x)
        if not np.isfinite(sd):
            return np.nan
        cdf = lambda z: gumbel_l.cdf(z, mu, sd)
    return float(kstest(x, cdf).statistic)


def _chi2gof_bins(x: np.ndarray, cdf_func, num_bins: int, e_min: float = 5) -> tuple:
    """Pooled bin edges, observed and expected counts used by :func:`_chi2gof_effect`."""
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    nb = int(num_bins)
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        lo, hi = lo - nb // 2 - 0.5, hi + (nb - nb // 2) - 0.5
    edges = lo + np.arange(nb + 1) * (hi - lo) / nb
    edges[-1] = hi
    idx = np.searchsorted(edges[1:-1], x, side='left')       # bins are (e_{k-1}, e_k]
    obs = np.bincount(idx, minlength=nb).astype(float)
    cum = np.concatenate([[0.0], np.asarray(cdf_func(edges[1:-1]), dtype=float), [1.0]])
    exp = N * np.diff(cum)

    edges, obs, exp = list(edges), list(obs), list(exp)
    while len(obs) > 2 and (exp[0] < e_min or exp[1] < e_min or exp[-1] < e_min or exp[-2] < e_min):
        if exp[0] < exp[-1]:         # merge the smaller end bin inwards (ties: the last bin)
            exp[1] += exp[0]; obs[1] += obs[0]
            del exp[0], obs[0], edges[1]
        else:
            exp[-2] += exp[-1]; obs[-2] += obs[-1]
            del exp[-1], obs[-1], edges[-2]
    return np.array(edges), np.array(obs), np.array(exp)


def _chi2gof_effect(x: np.ndarray, cdf_func, num_bins: int, e_min: int = 5) -> float:
    """
    Chi-squared goodness-of-fit statistic divided by the number of observations.

    Pearson's statistic ``sum (O - E)^2 / E`` of ``x`` against a fully specified CDF, with
    ``num_bins`` equal-width bins over the range of the data (tails folded into the end bins
    for the expected counts). Bins at the extremes of either tail with an expected count
    below ``e_min`` (or next to such a bin) are merged inwards, the one with the smaller
    expected count first, until two bins remain or all end bins reach ``e_min``.

    Notes
    -----
    Pearson (1900) Philosophical Magazine 50:157-175.
    """
    x = np.asarray(x, dtype=float).ravel()
    _, obs, exp = _chi2gof_bins(x, cdf_func, num_bins, e_min)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2stat = np.sum((obs - exp) ** 2 / exp)
    return float(chi2stat / len(x))


def _kstest_effect(x: np.ndarray, cdf_func) -> float:
    """One-sample two-sided Kolmogorov-Smirnov statistic of ``x`` against a null CDF.

    The statistic is the maximum distance between the empirical CDF and ``cdf_func``
    (``scipy.stats.kstest``, which handles tied observations exactly). NaN when the null CDF
    is not finite at the data (a degenerate fit).

    Notes
    -----
    Kolmogorov (1933); Smirnov (1948); Massey (1951) JASA 46:68-78.
    """
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(np.asarray(cdf_func(x), dtype=float))):
        logger.warning("The fitted CDF is not finite at the data; the KS statistic is undefined.")
        return np.nan
    return float(kstest(x, cdf_func).statistic)


def distribution_test(x: ArrayLike, the_test: str = 'chi2gof', the_distn: str = 'norm',
                      num_bins: int = 10) -> float:
    """
    Hypothesis test for distributional fits to a data vector.

    Fits a distribution to the data and then performs an appropriate hypothesis
    test to quantify the difference between the two distributions. Returns an
    effect size rather than a p-value (p-values underflow for long series and
    depend on the series length), so larger values indicate a worse fit.

    Parameters
    ----------
    x : array-like
        The input data vector.
    the_test : str, optional
        The hypothesis test to perform:

        - 'chi2gof': chi^2 goodness of fit test (effect size: chi^2 statistic / N)
        - 'ks': Kolmogorov-Smirnov test (effect size: the KS statistic D)
        - 'lillie': Lilliefors test (effect size: the KS statistic D; only defined
          for 'norm', 'ev', and 'exp')

        Default is ``'chi2gof'``.
    the_distn : str, optional
        The distribution to fit:

        - 'norm' (Normal)
        - 'ev' (Extreme value)
        - 'uni' (Uniform)
        - 'beta' (Beta)
        - 'rayleigh' (Rayleigh)
        - 'exp' (Exponential)
        - 'gamma' (Gamma)
        - 'logn' (Log-normal)
        - 'wbl' (Weibull)

        Default is ``'norm'``.
    num_bins : int, optional
        The number of bins to use for the chi^2 goodness of fit test.
        Default is 10.

    Returns
    -------
    float
        Effect size from the hypothesis test (chi^2 statistic / N for 'chi2gof';
        the KS statistic D for 'ks' and 'lillie'). NaN when the fit is not valid
        for the data (e.g., a positive-only distribution fitted to data with
        negative values).
    """
    x = np.asarray(x, dtype=float)
    num_bins = int(num_bins)

    if the_distn == 'beta':
        # hctsa's own (crude) rescaling of the data into the open interval (0,1)
        sd = np.std(x, ddof=1)
        x = (x - np.min(x) + 0.01 * sd) / (np.max(x) - np.min(x) + 0.02 * sd)
    elif the_distn in ('rayleigh', 'exp', 'gamma'):
        if np.any(x < 0):
            return np.nan
    elif the_distn in ('logn', 'wbl'):
        if np.any(x <= 0):
            return np.nan
    elif the_distn not in ('norm', 'ev', 'uni'):
        raise ValueError(f"Unknown distribution '{the_distn}'.")

    if the_test == 'lillie':
        if the_distn in ('norm', 'ev', 'exp'):
            return _lilliefors_statistic(x, the_distn)
        logger.warning("Lilliefors test is only defined for 'norm', 'ev', and 'exp' distributions.")
        return np.nan

    cdf_func, _ = _fit_distribution_cdf(x, the_distn)
    if the_test == 'chi2gof':
        return _chi2gof_effect(x, cdf_func, num_bins)
    elif the_test == 'ks':
        return _kstest_effect(x, cdf_func)
    raise ValueError(f"Unknown test '{the_test}'.")

def _vratiotest(y: np.ndarray, period: int, iid: bool) -> tuple:
    """
    Lo-MacKinlay variance ratio test; returns (p-value, statistic, ratio).

    Compares the variance of ``q``-period increments (overlapping, unbiased estimator) with
    ``q`` times the one-period variance, using all increments of the series
    (``arch.unitroot.VarianceRatio`` with a constant drift). The statistic
    ``sqrt(n)(VR - 1)/sqrt(theta)`` is asymptotically standard normal, with ``theta`` the
    homoskedastic (``iid=True``) or the heteroskedasticity-robust asymptotic variance; the
    p-value is two-sided. NaN for a constant series.

    Notes
    -----
    Lo & MacKinlay (1988) Review of Financial Studies 1:41-66; Campbell, Lo & MacKinlay
    (1997), The Econometrics of Financial Markets, Sec. 2.4.

    Raises
    ------
    ValueError
        If ``period < 2`` or ``len(y) <= 2 * period``.
    """
    from arch.unitroot import VarianceRatio
    y = np.asarray(y, dtype=float).ravel()
    q = int(period)
    if q < 2:
        raise ValueError("The period must be an integer greater than 1.")
    if len(y) <= 2 * q:
        raise ValueError("The series must be longer than twice the period q.")
    if not np.ptp(y) > 0 or not np.ptp(np.diff(y)) > 0:
        return np.nan, np.nan, np.nan
    vr = VarianceRatio(y, lags=q, trend='c', debiased=True, robust=not iid, overlap=True)
    return float(vr.pvalue), float(vr.stat), float(vr.vr)


def variance_ratio_test(y: ArrayLike, periods: Union[int, list[int], float] = 2,
                        iids: Union[int, list[int]] = 0) -> dict:
    """
    Variance ratio test for random walk.

    Implements the Lo-MacKinlay (1988) variance ratio test (via ``arch.unitroot.VarianceRatio``).

    The test assesses the null hypothesis of a random walk in the time series,
    which is rejected for some critical p-value.

    Parameters
    ----------
    y : array-like
        The input time series.
    periods : int or list of int, optional
        A scalar or vector of period(s) to use for the test. Default is 2.
    iids : int or list of int, optional
        A scalar or vector of boolean values (0 or 1) indicating whether to assume
        independent and identically distributed (IID) innovations for each period.
        Default is 0.

    Returns
    -------
    dict
        For a single period: ``pValue``, ``stat`` (the test statistic) and ``ratio``
        (the variance ratio). For several periods: the period and IID flag of the test
        with the largest and smallest p-value (``periodmaxpValue``,
        ``periodminpValue``, ``IIDperiodmaxpValue``, ``IIDperiodminpValue``), the mean,
        max and min test statistic (``meanstat``, ``maxstat``, ``minstat``), and the
        mean, max and min variance ratio (``meanratio``, ``maxratio``, ``minratio``).

    Notes
    -----
    The tests with the largest and smallest p-value are found from the absolute test
    statistic, which orders the tests exactly as the (two-sided) p-value does but, unlike
    it, does not saturate at 0 for strong departures from a random walk (where the
    extremes of the p-values would be decided by the floor of double precision).
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]  # remove missing values
    if not np.all(np.isfinite(y)):
        raise ValueError("The data must be finite.")

    # Single period: return the raw test statistics.
    if isinstance(periods, (int, float, np.number)):
        pvalue, stat, ratio = _vratiotest(y, int(periods), bool(iids))
        return {'pValue': pvalue, 'stat': stat, 'ratio': ratio}

    if not isinstance(periods, list):
        raise ValueError(f"Unknown data type for periods: {type(periods)}, "
                         "select either integer or list of integers.")

    # Multiple periods: iids must be a matching list of logicals (0 or 1).
    if not isinstance(iids, list):
        raise ValueError("Expected iids to be a list of bools, since periods "
                         f"are also a list. Got data type: {type(iids)} instead.")
    if len(iids) != len(periods):
        raise ValueError(f"Length of IIDs list ({len(iids)}) does not match "
                         f"the list of periods ({len(periods)}).")
    if not all(i in (0, 1) for i in iids):
        raise ValueError("List of IIDs must only be logicals (0 or 1).")

    res = np.array([_vratiotest(y, int(p), bool(iid)) for p, iid in zip(periods, iids)])
    pvals, stats, ratios = res[:, 0], res[:, 1], res[:, 2]
    if len(periods) == 1:  # a single test: summarize it directly, as in hctsa
        return {'pValue': pvals[0], 'stat': stats[0], 'ratio': ratios[0]}
    imax, imin = np.argmin(np.abs(stats)), np.argmax(np.abs(stats))  # largest, smallest p-value (first on ties)

    return {
        'periodmaxpValue': periods[imax],
        'periodminpValue': periods[imin],
        'IIDperiodmaxpValue': iids[imax],
        'IIDperiodminpValue': iids[imin],
        'meanstat': np.mean(stats),
        'maxstat': np.max(stats),
        'minstat': np.min(stats),
        # The variance ratio itself, the effect size behind the test. pValue and
        # stat both grow with the series length under any alternative, whereas the
        # ratio converges to a fixed population value.
        'meanratio': np.mean(ratios),
        'maxratio': np.max(ratios),
        'minratio': np.min(ratios),
    }

def hypothesis_test(x: ArrayLike, the_test: str = 'signtest') -> float:
    """
    Perform statistical hypothesis testing on a time series.

    Deprecated in hctsa in favor of :func:`marginal_tests` (tests about the
    distribution of values) and :func:`independence_tests` (tests of serial
    independence), to which this dispatches.

    Parameters
    ----------
    x : array-like
        Input time series.
    the_test : str, optional
        Type of hypothesis test to perform:

        - 'signtest', 'vartest', 'ztest', 'signrank', 'jbtest': see :func:`marginal_tests`
        - 'runsz', 'runstest', 'lbq': see :func:`independence_tests`

        Default is ``'signtest'``.

    Returns
    -------
    float
        P-value from the statistical test (identical to that of the function it
        dispatches to; the z-statistic of the runs test for 'runsz'). A small p-value
        (< 0.05) typically indicates rejection of the null hypothesis.
    """
    if the_test in ('runsz', 'runstest', 'lbq'):
        return independence_tests(x, the_test)
    if the_test in ('signtest', 'vartest', 'ztest', 'signrank', 'jbtest'):
        return marginal_tests(x, the_test)
    raise ValueError(f"Unknown test: {the_test}.")


def _signtest_pvalue(x: np.ndarray) -> float:
    """
    Two-sided sign test p-value for zero median.

    NaNs and zeros are discarded; the p-value is that of the exact two-sided binomial test of
    the number of positive values among the rest (``scipy.stats.binomtest``, probability 1/2).
    Returns 1 if no non-zero values remain.

    Notes
    -----
    Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th ed., Ch. 5.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return 1.0
    return float(binomtest(int(np.sum(x > 0)), n, 0.5).pvalue)


def _signrank_pvalue(x: np.ndarray) -> float:
    """
    Two-sided Wilcoxon signed rank test p-value for zero median.

    NaNs and zeros are discarded and the p-value is that of ``scipy.stats.wilcoxon`` (exact
    distribution when there are no ties, otherwise the normal approximation with the tie
    correction of the variance). Returns 1 if all values are zero and NaN if no values remain.

    Notes
    -----
    Wilcoxon (1945) Biometrics Bulletin 1:80-83; Hollander & Wolfe (1999), Nonparametric
    Statistical Methods, 2nd ed., Sec. 3.1.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    return float(wilcoxon(x, zero_method='wilcox', correction=False, method='auto').pvalue)


def _vartest_pvalue(x: np.ndarray, v: float = 1.0) -> float:
    """
    Two-sided chi-squared test p-value that the variance is v.

    Statistic ``(n - 1) s^2 / v`` (chi-squared with ``n - 1`` degrees of freedom for normal
    data); the p-value is twice the smaller tail, capped at 1. NaNs are ignored. Returns 0 for a
    single value and NaN for no (or non-finite) data.

    Notes
    -----
    Snedecor & Cochran (1989), Statistical Methods, 8th ed.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0 or not np.all(np.isfinite(x)):
        return np.nan
    df = n - 1
    if df == 0:
        return 0.0
    stat = np.sum((x - np.mean(x)) ** 2) / v
    return float(min(1.0, 2.0 * min(chi2.cdf(stat, df), chi2.sf(stat, df))))


_jb_cache = {}

# Sample size from which the asymptotic chi-squared(2) null distribution is used.
_JB_ASYMPTOTIC_N = 2000


def _jb_critical_values(n: int) -> tuple:
    """Critical values (decreasing) at the tabulated significance levels for sample size n.

    The Monte Carlo table is interpolated in ``1/n`` (monotone cubic), anchored at ``1/n = 0`` by
    the chi-squared(2) critical values; returns ``(critical_values, alphas)``.
    """
    if 'interp' not in _jb_cache:
        from scipy.interpolate import PchipInterpolator
        from ..toolboxes.distribution_fits._jbtest_null_table import ALPHAS, CRITICAL_VALUES, N_NODES
        order = np.argsort(-np.asarray(N_NODES))            # increasing 1/n
        u = np.concatenate([[0.0], 1.0 / np.asarray(N_NODES)[order]])
        cv = np.vstack([chi2.isf(ALPHAS, 2)[None, :], CRITICAL_VALUES[order]])
        _jb_cache['interp'] = (PchipInterpolator(u, cv, axis=0), np.asarray(ALPHAS, float))
    interp, alphas = _jb_cache['interp']
    return np.minimum.accumulate(interp(1.0 / n)), alphas


def _jb_statistic(x: np.ndarray) -> float:
    """Jarque-Bera statistic ``n/6 (S^2 + (K - 3)^2/4)`` with the biased sample skewness S and
    kurtosis K (Jarque & Bera 1987); NaN (or inf) for a constant series."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    with np.errstate(all='ignore'):
        d = x - np.mean(x)
        m2 = np.mean(d ** 2)
        return float(n / 6.0 * ((np.mean(d ** 3) / m2 ** 1.5) ** 2 + (np.mean(d ** 4) / m2 ** 2 - 3.0) ** 2 / 4.0))


def _jbtest_pvalue(x: np.ndarray) -> float:
    """
    Jarque-Bera test p-value for normality.

    Statistic ``JB = n/6 (S^2 + (K - 3)^2 / 4)`` with the biased sample skewness S and kurtosis
    K (Jarque & Bera, 1987). Its null distribution converges slowly to chi-squared(2), so for
    ``n < 2000`` the p-value is read from a Monte Carlo table of the distribution of JB for
    normal samples (sizes 3 to 2000, ``scripts/gen_jbtest_null_table.py``), interpolating
    in ``1/n`` and in logit(p) over the whole range ``1e-4 < p < 1 - 1e-4``; beyond the table
    the p-value is extrapolated (log-linearly in the far upper tail, linearly to ``p = 1`` at
    ``JB = 0``), so it is never clamped. For ``n >= 2000`` the chi-squared(2) limit is used.
    NaNs are ignored; returns 1 for two values and 0 for a constant series.

    Raises
    ------
    ValueError
        With fewer than 2 non-NaN values.

    Notes
    -----
    Jarque & Bera (1987) International Statistical Review 55:163-172; finite-sample null
    distribution by simulation, as in Deb & Sefton (1996) Economics Letters 51:123-130.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 non-NaN observations are needed.")
    if n == 2:
        return 1.0
    jb = _jb_statistic(x)
    if not np.isfinite(jb):
        return 0.0
    if n >= _JB_ASYMPTOTIC_N:
        return float(chi2.sf(jb, 2))
    cv, alphas = _jb_critical_values(n)                      # cv decreases as alpha increases
    logit = np.log(alphas) - np.log1p(-alphas)
    if jb > cv[0]:                                            # beyond the table: exponential tail
        k = 10
        slope = (logit[k] - logit[0]) / (cv[k] - cv[0]) if cv[0] > cv[k] else -0.5
        return float(np.exp(logit[0] + slope * (jb - cv[0])))
    if jb < cv[-1]:                                           # below the table: linear to p = 1 at JB = 0
        return float(1.0 - (1.0 - alphas[-1]) * jb / cv[-1])
    lg = np.interp(jb, cv[::-1], logit[::-1])
    return float(1.0 / (1.0 + np.exp(-lg)))


def _log_choose(n, k):
    """Logarithm of the binomial coefficient n choose k (``-inf`` where ``k < 0`` or ``k > n``)."""
    n, k = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    valid = (k >= 0) & (k <= n)
    nn, kk = np.where(valid, n, 1.0), np.where(valid, k, 0.0)
    out = gammaln(nn + 1.0) - gammaln(kk + 1.0) - gammaln(nn - kk + 1.0)
    return np.where(valid, out, -np.inf)


def runstest_pvalue(x: np.ndarray) -> float:
    """
    Two-sided runs test p-value (runs above and below the mean).

    NaNs are ignored and values equal to the mean are discarded. With ``n1`` values above and
    ``n0`` below the mean, the number of runs has an exact distribution under randomness; the
    p-value is twice the smaller tail probability (inclusive of the observed count), capped at 1.
    Returns 1 if no values remain above or below.

    Notes
    -----
    Wald & Wolfowitz (1940); Mood (1940) Ann. Math. Stat. 11:367-392; Swed & Eisenhart (1943)
    Ann. Math. Stat. 14:66-87; Gibbons & Chakraborti (2011), Sec. 3.2.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 1.0
    v = np.mean(x)
    x = x[x != v]
    above = x > v
    n1 = int(np.sum(above))
    n0 = len(above) - n1
    if n1 == 0 or n0 == 0:
        return 1.0
    r = 1 + int(np.sum(above[1:] != above[:-1]))

    N = n1 + n0
    rr = np.arange(2, N + 1)
    m = rr // 2
    even = _log_choose(n1 - 1, m - 1) + _log_choose(n0 - 1, m - 1) + np.log(2.0)
    odd = np.logaddexp(_log_choose(n1 - 1, m) + _log_choose(n0 - 1, m - 1),
                       _log_choose(n1 - 1, m - 1) + _log_choose(n0 - 1, m))
    logp = np.where(rr % 2 == 0, even, odd) - _log_choose(N, n1)
    lower = logsumexp(logp[rr <= r])
    upper = logsumexp(logp[rr >= r])
    return float(min(1.0, 2.0 * np.exp(min(lower, upper))))


def marginal_tests(y: ArrayLike, the_test: str = 'signtest') -> float:
    """
    p-value of a hypothesis test about the distribution of values.

    Tests the distribution of the values of the time series, ignoring their temporal
    order (a shuffled series gives the same p-value); see :func:`independence_tests`
    for tests of serial dependence. This is the part of hctsa's former
    ``HT_HypothesisTest`` that tests marginal properties.

    Parameters
    ----------
    y : array-like
        The input time series.
    the_test : {'signtest', 'vartest', 'ztest', 'signrank', 'jbtest', 'logjb'}, optional
        The test:

        - 'signtest': sign test of zero median (exact binomial test);
        - 'vartest': chi-squared test that the variance is 1 (assuming normality);
        - 'ztest': z-test that the mean is zero (assuming unit variance);
        - 'signrank': Wilcoxon signed rank test of zero median;
        - 'jbtest': Jarque-Bera test of normality;
        - 'logjb': the logarithm of the Jarque-Bera statistic,
          ``JB = n/6 (S^2 + (K - 3)^2/4)`` with the (biased) sample skewness S and kurtosis K
          [1]_: how far the shape of the distribution is from normal, unbounded (unlike a
          tabulated p-value). NaN for a constant series or if JB is exactly 0.

        Default is ``'signtest'``.

    Returns
    -------
    float
        The p-value of the (two-sided) test (small values reject the null hypothesis), or
        ``log(JB)`` for ``'logjb'``.

    References
    ----------
    .. [1] C. M. Jarque and A. K. Bera, "A test for normality of observations and regression
        residuals", International Statistical Review 55(2), 163-172 (1987).
        DOI: 10.2307/1403192
    """
    y = np.asarray(y, dtype=float).ravel()
    if the_test == 'signtest':
        return _signtest_pvalue(y)
    if the_test == 'vartest':
        return _vartest_pvalue(y, 1.0)
    if the_test == 'ztest':
        y = y[~np.isnan(y)]
        z = np.mean(y) / (1.0 / np.sqrt(len(y)))
        return float(2 * norm.cdf(-abs(z)))
    if the_test == 'signrank':
        return _signrank_pvalue(y)
    if the_test == 'jbtest':
        return _jbtest_pvalue(y)
    if the_test == 'logjb':
        jb = _jb_statistic(y[~np.isnan(y)])
        return float(np.log(jb)) if np.isfinite(jb) and jb > 0 else np.nan
    raise ValueError(f"Unknown hypothesis test '{the_test}'.")


def independence_tests(y: ArrayLike, the_test: str = 'runsz') -> float:
    """
    Statistic or p-value of a test of serial independence.

    Tests whether the values of the time series are independent of one another. Unlike
    the tests in :func:`marginal_tests`, the result depends on the temporal order of the
    values. This is the part of hctsa's former ``HT_HypothesisTest`` that tests
    dependence. The runs test is returned as its z-statistic (in closed form), the other
    tests as p-values.

    Parameters
    ----------
    y : array-like
        The input time series.
    the_test : {'runsz', 'runstest', 'lbq'}, optional
        The test:

        - 'runsz': runs test for randomness of the runs of values above and below the
          median; returns the signed z-statistic of the number of runs
          (:func:`pyhctsa.robust.bf_runs_z`): negative for fewer runs than expected
          (positive serial dependence), positive for more (alternation), and
          approximately standard normal under the null hypothesis;
        - 'runstest': the p-value of the same hypothesis (runs above and below the mean,
          exact distribution of the number of runs);
        - 'lbq': Ljung-Box Q-test for autocorrelation up to lag 20.

        Default is ``'runsz'``.

    Returns
    -------
    float
        The z-statistic of the test for 'runsz', otherwise its p-value: the probability,
        under the null hypothesis, of a test statistic at least as extreme as that
        observed. Small values are evidence of serial dependence.
    """
    y = np.asarray(y, dtype=float).ravel()
    if the_test == 'runsz':
        return bf_runs_z(y)
    if the_test == 'runstest':
        return runstest_pvalue(y)
    if the_test == 'lbq':
        return ljung_box_pvalue(y, n_lags=20)
    raise ValueError(f"Unknown hypothesis test '{the_test}'.")
