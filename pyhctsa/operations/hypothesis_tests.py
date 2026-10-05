from typing import Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.special import log_ndtr
from scipy.stats import jarque_bera, norm, wilcoxon, rayleigh, expon, gumbel_l, lognorm, uniform, weibull_min
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.descriptivestats import sign_test

from ..utils import ljung_box_pvalue
from ..toolboxes.distribution_fits.distfits import betafit, evfit, gamfit, wblfit

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
        (the variance ratio). For several periods: the max, min and mean p-value
        (``maxpValue``, ``minpValue``, ``meanpValue``), the period and IID flag at
        which the max and min p-value occur (``periodmaxpValue``,
        ``periodminpValue``, ``IIDperiodmaxpValue``, ``IIDperiodminpValue``), the mean,
        max and min test statistic (``meanstat``, ``maxstat``, ``minstat``), and the
        mean, max and min variance ratio (``meanratio``, ``maxratio``, ``minratio``).
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
    imax, imin = np.argmax(pvals), np.argmin(pvals)

    return {
        'maxpValue': np.max(pvals),
        'minpValue': np.min(pvals),
        'meanpValue': np.mean(pvals),
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

    Applies a specified statistical test and returns its p-value. Tests are chosen
    to evaluate different null hypotheses about the time series properties.

    Parameters
    ----------
    x : array-like
        Input time series.
    the_test : str, optional
        Type of hypothesis test to perform:

        - 'signtest': Tests if median equals zero
        - 'runstest': Tests for randomness in sequence
        - 'ztest': Tests if mean equals zero (assumes unit variance)
        - 'signrank': Wilcoxon signed rank test for zero median
        - 'jbtest': Jarque-Bera test for normality
        - 'lbq': Ljung-Box Q-test for autocorrelation
        
        Default is ``'signtest'``.

    Returns
    -------
    float
        P-value from the statistical test. A small p-value (< 0.05) typically
        indicates rejection of the null hypothesis.
    """
    x = np.asarray(x)
    p = np.nan
    if the_test == 'signtest':
        _, p = sign_test(x)
    elif the_test == 'runstest':
        _, p = runstest_1samp(x, cutoff='mean', correction=True)
    elif the_test == 'jbtest':
        s = jarque_bera(x)
        p = s.pvalue
    elif the_test == 'ztest':
        x_mean = np.mean(x)
        n = len(x)
        sigma = 1
        zval = (x_mean - 0) / (sigma / np.sqrt(n))
        p = 2 * norm.cdf(-abs(zval))
    elif the_test == 'signrank':
        _, p = wilcoxon(x)
    elif the_test == 'lbq':
        # Ljung-Box Q-test for residual autocorrelation; O(N*n_lags), see
        # utils.ljung_box_pvalue.
        p = ljung_box_pvalue(x, n_lags=20)
    else:
        raise ValueError(f"Unknown test: {the_test}.")
    return p
