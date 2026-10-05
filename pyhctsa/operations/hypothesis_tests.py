from typing import Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gammaln, log_ndtr
from scipy.stats import binom, chi2, norm, rankdata, rayleigh, expon, gumbel_l, lognorm, uniform, weibull_min

from ..robust import bf_runs_z
from ..utils import ljung_box_pvalue
from ..toolboxes.distribution_fits.distfits import betafit, evfit, gamfit, wblfit
from ..toolboxes.distribution_fits.jbtest_tables import (ALPHAS as JB_ALPHAS, CRITICAL_VALUES as JB_CRITICAL_VALUES,
                                                         SAMPLE_SIZES as JB_SAMPLE_SIZES)

def _fit_distribution_cdf(x: np.ndarray, the_distn: str) -> tuple:
    """Fit a distribution to data, MATLAB-style; return its CDF and parameter count."""
    n = len(x)
    if the_distn == 'norm':
        mu, sigma = np.mean(x), np.std(x, ddof=1)
        return (lambda z: norm.cdf(z, mu, sigma)), 2
    if the_distn == 'ev':
        loc, scale = evfit(x)
        return (lambda z: gumbel_l.cdf(z, loc=loc, scale=scale)), 2
    if the_distn == 'uni':
        a, b = np.min(x), np.max(x)
        return (lambda z: uniform.cdf(z, loc=a, scale=b - a)), 2
    if the_distn == 'beta':
        a, b = betafit(x)
        return (lambda z: beta_dist.cdf(z, a, b)), 2
    if the_distn == 'rayleigh':
        b = np.sqrt(np.sum(x ** 2) / (2 * n))
        return (lambda z: rayleigh.cdf(z, scale=b)), 1
    if the_distn == 'exp':
        mu = np.mean(x)
        return (lambda z: expon.cdf(z, scale=mu)), 1
    if the_distn == 'gamma':
        a, b = gamfit(x)
        return (lambda z: gamma_dist.cdf(z, a, scale=b)), 2
    if the_distn == 'logn':
        lx = np.log(x)
        mu, sigma = np.mean(lx), np.std(lx, ddof=1)
        return (lambda z: lognorm.cdf(z, s=sigma, scale=np.exp(mu))), 2
    if the_distn == 'wbl':
        a, c = wblfit(x)
        return (lambda z: weibull_min.cdf(z, c, scale=a)), 2
    raise ValueError(f"Unknown distribution '{the_distn}'.")


def _lilliefors_statistic(x: np.ndarray, the_distn: str) -> float:
    """Lilliefors test statistic, matching the KS statistic returned by MATLAB's
    lillietest (the maximum distance between the empirical CDF and the CDF of
    the distribution fitted to the data)."""
    n = len(x)
    if n < 4:
        return np.nan

    ux, counts = np.unique(x, return_counts=True)
    if the_distn == 'norm':
        null_cdf = norm.cdf(ux, np.mean(x), np.std(x, ddof=1))
    elif the_distn == 'exp':
        null_cdf = expon.cdf(ux, scale=np.mean(x))
    elif the_distn == 'ev':
        loc, scale = evfit(x)
        null_cdf = gumbel_l.cdf(ux, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unknown distribution '{the_distn}' for Lilliefors test.")
    return _ks_statistic(counts, null_cdf, n)

def _chi2gof_effect(x: np.ndarray, cdf_func, num_bins: int, e_min: int = 5) -> float:
    """Chi^2 goodness-of-fit effect size, chi^2 statistic / N, where the
    statistic matches MATLAB's chi2gof.

    Data are binned into num_bins equal-width bins spanning the data range;
    tail bins with expected counts below e_min are pooled with neighbors.
    """
    n = len(x)
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        lo -= np.floor(num_bins / 2) + 0.5
        hi += np.ceil(num_bins / 2) - 0.5
    binwidth = (hi - lo) / num_bins
    edges = lo + binwidth * np.arange(num_bins + 1)
    edges[-1] = hi
    edges = edges + np.spacing(edges)  # shift so bins are ( ] intervals
    interior = edges[1:-1]

    obs = np.bincount(np.searchsorted(interior, x, side='right'),
                      minlength=num_bins).astype(float)
    # Tail probability mass is folded into the first and last bins
    exp_counts = n * np.diff(np.concatenate(([0.0], cdf_func(interior), [1.0])))

    if np.any(exp_counts < e_min):
        # Pool the smaller extreme bin into its neighbor each time; interior
        # bins are never pooled together.
        i, j = 0, num_bins - 1
        while i < j - 1 and (exp_counts[i] < e_min or exp_counts[i + 1] < e_min
                             or exp_counts[j] < e_min or exp_counts[j - 1] < e_min):
            if exp_counts[i] < exp_counts[j]:
                exp_counts[i + 1] += exp_counts[i]
                obs[i + 1] += obs[i]
                i += 1
            else:
                exp_counts[j - 1] += exp_counts[j]
                obs[j - 1] += obs[j]
                j -= 1
        exp_counts = exp_counts[i:j + 1]
        obs = obs[i:j + 1]

    chi2_stat = np.sum((obs - exp_counts) ** 2 / exp_counts)
    return float(chi2_stat / n)

def _ks_statistic(counts: np.ndarray, null_cdf: np.ndarray, n: int) -> float:
    """Two-sided KS statistic between an ECDF (from counts at the unique
    sorted data values) and the null CDF evaluated at those values."""
    sample_cdf = np.concatenate(([0.0], np.cumsum(counts) / n))
    delta1 = sample_cdf[:-1] - null_cdf  # jumps approached from the left
    delta2 = sample_cdf[1:] - null_cdf  # jumps approached from the right
    return float(np.max(np.abs(np.concatenate((delta1, delta2)))))

def _kstest_statistic(x: np.ndarray, cdf_func) -> float:
    """Two-sided KS D-statistic against a tabulated null CDF, matching MATLAB's
    kstest called with a two-column CDF argument (null CDF tabulated on the data
    values rounded to 6 decimal places, then linearly interpolated)."""
    n = len(x)
    xmin, xmax = np.min(x), np.max(x)
    grid = np.unique(np.round(x * 1e6) / 1e6)
    if grid[0] > xmin:
        grid = np.concatenate(([xmin], grid))
    if grid[-1] < xmax:
        grid = np.concatenate((grid, [xmax]))
    y_grid = cdf_func(grid)

    ux, counts = np.unique(x, return_counts=True)
    if len(ux) == len(grid) and np.array_equal(ux, grid):
        null_cdf = y_grid
    else:
        null_cdf = np.interp(ux, grid, y_grid)
    return _ks_statistic(counts, null_cdf, n)


def _kstest_effect(x: np.ndarray, cdf_func) -> float:
    """One-sample two-sided KS statistic D (the effect size), matching MATLAB's
    kstest as called by HT_DistributionTest. NaN if the fitted CDF is
    non-finite everywhere (degenerate fit)."""
    if not np.any(np.isfinite(cdf_func(np.unique(x)))):
        logger.warning("Fitted CDF is degenerate for this data; no KS test possible.")
        return np.nan
    return _kstest_statistic(x, cdf_func)

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
        # clumsily scale to the range (0,1), as in MATLAB
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
    """Lo-MacKinlay variance ratio test, a port of MATLAB's vratiotest for one
    period: returns (pValue, stat, ratio).

    The test uses the first N = floor((len(y) - 1) / period) * period increments
    (so that the series divides into whole periods) and the sample drift
    c = (y[N] - y[0]) / N.
    """
    num_obs = len(y)
    if period >= num_obs / 2:
        raise ValueError("Too few observations for the requested period.")
    r = np.diff(y)
    N = ((num_obs - 1) // period) * period  # number of increments used

    c = (y[N] - y[0]) / N
    e1 = r[:N] - c
    sse1 = e1 @ e1
    var1 = sse1 / (N - 1)

    e2 = y[period:N + 1] - y[:N - period + 1] - period * c
    sse2 = e2 @ e2
    var2 = sse2 / (period * (N - period + 1) * (1 - period / N))

    ratio = var2 / var1

    if iid:
        ratio_var = 2 * (2 * period - 1) * (period - 1) / (3 * period)
    else:  # heteroskedasticity-consistent estimator
        summands = np.zeros(period - 1)
        for k in range(1, period):
            delta = N * (e1[k:] ** 2 @ e1[:N - k] ** 2) / sse1 ** 2
            summands[k - 1] = (1 - k / period) ** 2 * delta
        ratio_var = 4 * np.sum(summands)

    stat = np.sqrt(N) * (ratio - 1) / np.sqrt(ratio_var)
    pvalue = 2 * norm.cdf(-abs(stat))  # two-tailed
    if pvalue < 1e-290:
        # scipy's cdf underflows to 0 for |stat| above ~37.5, whereas MATLAB's
        # normcdf continues into the denormal range (to ~38.5); follow it, since
        # which test has the smallest p-value is an output in the multi-test case
        pvalue = 2 * np.exp(log_ndtr(-abs(stat)))
    return pvalue, stat, ratio


def variance_ratio_test(y: ArrayLike, periods: Union[int, list[int], float] = 2,
                        iids: Union[int, list[int]] = 0) -> dict:
    """
    Variance ratio test for random walk.

    Implements the Lo-MacKinlay variance ratio test, as in MATLAB's vratiotest.

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
    p-value of MATLAB's ``signtest(x)`` (median zero, two-sided).

    Zeros are dropped; the exact binomial test is used for fewer than 100 remaining
    values, the normal approximation (with a continuity correction) otherwise.
    """
    d = x[~np.isnan(x)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    npos = int(np.sum(d > 0))
    nneg = n - npos
    if n < 100:
        return float(min(1.0, 2 * binom.cdf(min(nneg, npos), n, 0.5)))
    z = (npos - nneg - np.sign(npos - nneg)) / np.sqrt(n)
    return float(2 * norm.cdf(-abs(z)))


def _signrank_pvalue(x: np.ndarray) -> float:
    """
    p-value of MATLAB's ``signrank(x)`` (Wilcoxon signed rank test of zero median, two-sided).

    Exact (from the permutation distribution of the signed ranks, with ties given their
    average rank) for 15 or fewer non-zero values, the tie-corrected normal
    approximation otherwise.
    """
    d = x[~np.isnan(x)]
    if len(d) == 0:
        raise ValueError('signrank: not enough data.')
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = rankdata(np.abs(d))  # average ranks for ties
    w = np.sum(ranks[d > 0])
    if n > 15:
        _, counts = np.unique(np.abs(d), return_counts=True)
        tieadj = 0.5 * np.sum(counts ** 3 - counts)
        z = (w - n * (n + 1) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1) - tieadj) / 24)
        return float(2 * norm.cdf(-abs(z)))
    # exact: probability that the sum of a random subset of the ranks is at most the
    # smaller of w and n(n+1)/2 - w
    maxw = n * (n + 1) / 2
    if w > maxw / 2:
        w = maxw - w
    v = np.sort(ranks)
    if np.any(v != np.floor(v)):  # half-integer ranks: work in units of 1/2
        v = np.round(2 * v).astype(int)
        w = int(round(2 * w))
    else:
        v = v.astype(int)
        w = int(round(w))
    counts = np.zeros(w + 1)
    counts[0] = 1.0
    for vj in v[v <= w]:
        counts[vj:] = counts[vj:] + counts[:w + 1 - vj].copy()
    return float(min(1.0, 2 * np.sum(counts) / 2.0 ** n))


def _vartest_pvalue(x: np.ndarray, v: float = 1.0) -> float:
    """p-value of MATLAB's ``[~, p] = vartest(x, v)``: chi-squared test that the variance is v (two-sided)."""
    x = x[~np.isnan(x)]
    df = max(len(x) - 1, 0)
    sumsq = np.sum((x - np.mean(x)) ** 2)
    p = chi2.cdf(sumsq / v, df)
    return float(2 * min(p, 1 - p))


def _jbtest_pvalue(x: np.ndarray) -> float:
    """
    p-value of MATLAB's ``[~, p] = jbtest(x)`` (Jarque-Bera test of normality).

    MATLAB interpolates a table of simulated critical values (see
    ``toolboxes/distribution_fits/jbtest_tables.py``), so the p-value is limited to
    [0.001, 0.5].
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError('jbtest: not enough data.')
    if n == 2:
        return 1.0
    z = (x - np.mean(x)) / np.std(x)
    skew_ = np.sum(z ** 3) / n
    kurt_ = np.sum(z ** 4) / n - 3
    jb = n * (skew_ ** 2 / 6 + kurt_ ** 2 / 24)

    # critical values at this sample size: interpolate in 1/n, shape-preserving
    inv_n = 1.0 / JB_SAMPLE_SIZES
    order = np.argsort(inv_n)
    cvs = np.array([PchipInterpolator(inv_n[order], JB_CRITICAL_VALUES[order, j])(1.0 / n)
                    for j in range(len(JB_ALPHAS))])
    if np.isnan(jb):
        return 0.0
    if jb < cvs[-1]:  # smallest critical value at the end
        return float(JB_ALPHAS[-1])
    if cvs[0] <= jb:  # largest critical value at the beginning
        return float(JB_ALPHAS[0])
    pp = PchipInterpolator(JB_ALPHAS, cvs)
    i = int(np.argmax(jb > cvs))  # first index with jb > cvs
    return float(brentq(lambda a: pp(a) - jb, JB_ALPHAS[i - 1], JB_ALPHAS[i],
                        xtol=1e-16, rtol=4 * np.finfo(float).eps))


def _log_choose(n, k):
    """log of the binomial coefficient, -inf where it is zero."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    with np.errstate(invalid='ignore'):
        out = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return np.where((k < 0) | (n - k < 0), -np.inf, out)


def runstest_pvalue(x: np.ndarray) -> float:
    """
    p-value of MATLAB's ``[~, p] = runstest(x)`` (runs above and below the mean; two-sided).

    Values equal to the mean are dropped. The p-value comes from the exact distribution
    of the number of runs (as MATLAB's default for this test),
    ``min(1, 2 (P(R = r) + min(P(R < r), P(R > r))))``.
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 1.0
    v = np.mean(x)
    x = x[x != v]
    n = len(x)
    if n == 0:
        return 1.0
    b = (x > v).astype(int)
    n1 = int(b.sum())
    n0 = n - n1
    nruns = 1 + int(np.sum(b[:-1] != b[1:]))
    if n1 == 0 or n0 == 0:
        plist = np.array([1.0])  # exactly one run
    else:
        r = np.arange(1, 2 * min(n1, n0) + 2)
        plist = np.zeros(len(r))
        logdenom = _log_choose(n, n0)
        even = r % 2 == 0
        k = r[even] // 2
        plist[even] = 2 * np.exp(_log_choose(n1 - 1, k - 1) + _log_choose(n0 - 1, k - 1) - logdenom)
        k = r[~even] // 2
        plist[~even] = (np.exp(_log_choose(n1 - 1, k - 1) + _log_choose(n0 - 1, k) - logdenom)
                        + np.exp(_log_choose(n1 - 1, k) + _log_choose(n0 - 1, k - 1) - logdenom))
    pexact = plist[nruns - 1]
    plo = np.sum(plist[:nruns - 1])
    phi = np.sum(plist[nruns:])
    return float(min(1.0, 2 * (pexact + min(plo, phi))))


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
    the_test : {'signtest', 'vartest', 'ztest', 'signrank', 'jbtest'}, optional
        The test:

        - 'signtest': sign test of zero median (exact binomial test for fewer than
          100 non-zero values, normal approximation otherwise);
        - 'vartest': chi-squared test that the variance is 1 (assuming normality);
        - 'ztest': z-test that the mean is zero (assuming unit variance);
        - 'signrank': Wilcoxon signed rank test of zero median (exact for up to 15
          non-zero values, normal approximation otherwise);
        - 'jbtest': Jarque-Bera test of normality (p-values from MATLAB's table of
          simulated critical values: limited to [0.001, 0.5]).

        Default is ``'signtest'``.

    Returns
    -------
    float
        The p-value of the (two-sided) test. Small values reject the null hypothesis.
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
          exact distribution of the number of runs; as MATLAB's ``runstest``);
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
