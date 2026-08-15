import logging
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from arch.unitroot import VarianceRatio
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, expon
from scipy.stats import gamma as gamma_dist
from scipy.stats import gumbel_l, jarque_bera, kstwo, lognorm, norm, rayleigh, uniform, weibull_min, wilcoxon
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.descriptivestats import sign_test

from ..utils import ljung_box_pvalue
from ..toolboxes.distribution_fits._lilliefors_tables import LILLIE_ALPHAS, LILLIE_TABLES
from ..toolboxes.distribution_fits.distfits import betafit, evfit

logger = logging.getLogger('pyhctsa')

def variance_ratio_test(y: ArrayLike, periods: Union[int, list[int], float] = 2,
                        iids: Union[int, list[int]] = 0) -> dict:
    """
    Variance ratio test for random walk.

    Implements the variance ratio test using the VarianceRatio function from arch.unitroot.

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
        Dictionary of test results.
    """
    y = np.asarray(y)

    # Single period: return the raw test statistics.
    if isinstance(periods, (int, float, np.number)):
        vr = VarianceRatio(y, lags=int(periods), robust=(iids == 0))
        return {'pValue': vr.pvalue, 'stat': vr.stat, 'ratio': vr.vr}

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

    vrs = [VarianceRatio(y, lags=p, robust=(iid == 0))
           for p, iid in zip(periods, iids)]
    pvals = np.array([vr.pvalue for vr in vrs])
    stats = np.array([vr.stat for vr in vrs])
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

def distribution_test(x: ArrayLike, the_test: str = 'chi2gof', the_distn: str = 'norm',
                      num_bins: int = 10) -> float:
    """
    Hypothesis test for distributional fits to a data vector.

    Fits a distribution to the data and then performs an appropriate hypothesis
    test to quantify the difference between the two distributions.

    Parameters
    ----------
    x : array-like
        The input data vector.
    the_test : str, optional
        The hypothesis test to perform:

        - 'chi2gof': chi^2 goodness of fit test
        - 'ks': Kolmogorov-Smirnov test
        - 'lillie': Lilliefors test (only defined for 'norm', 'ev', and 'exp')

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
        P-value from the hypothesis test. NaN when the fit is not valid for
        the data (e.g., a positive-only distribution fitted to data with
        negative values).
    """
    x = np.asarray(x, dtype=float)
    num_bins = int(num_bins)

    if the_distn == 'beta':
        # clumsily scale to the range (0,1), as in MATLAB
        sd = np.std(x, ddof=1)
        x = (x - np.min(x) + 0.01 * sd) / (np.max(x) - np.min(x) + 0.01 * sd)
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
            return _lilliefors_pvalue(x, the_distn)
        logger.warning("Lilliefors test is only defined for 'norm', 'ev', and 'exp' distributions.")
        return np.nan

    cdf_func, n_params = _fit_distribution_cdf(x, the_distn)
    if the_test == 'chi2gof':
        return _chi2gof_pvalue(x, cdf_func, num_bins, n_params)
    elif the_test == 'ks':
        return _kstest_pvalue(x, cdf_func)
    raise ValueError(f"Unknown test '{the_test}'.")

def _fit_distribution_cdf(x: np.ndarray, the_distn: str) -> tuple:
    """Fit a distribution to data, MATLAB-style; return its CDF and parameter count."""
    n = len(x)
    if the_distn == 'norm':
        # MATLAB normfit: sample mean and (n-1)-normalized standard deviation
        mu, sigma = np.mean(x), np.std(x, ddof=1)
        return (lambda z: norm.cdf(z, mu, sigma)), 2
    if the_distn == 'ev':
        # MATLAB 'ev' is the type-I extreme value (Gumbel) distribution for minima
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
        # MATLAB expfit parameterizes by the mean
        mu = np.mean(x)
        return (lambda z: expon.cdf(z, scale=mu)), 1
    if the_distn == 'gamma':
        if np.any(x == 0):
            # MLE is not possible with zeros in the data; MATLAB's gamfit
            # falls back to method-of-moments estimates
            xbar = np.mean(x)
            s2 = np.var(x, ddof=1)
            a, b = xbar ** 2 / s2, s2 / xbar
        else:
            a, _, b = gamma_dist.fit(x, floc=0)
        return (lambda z: gamma_dist.cdf(z, a, scale=b)), 2
    if the_distn == 'logn':
        lx = np.log(x)
        mu, sigma = np.mean(lx), np.std(lx, ddof=1)
        return (lambda z: lognorm.cdf(z, s=sigma, scale=np.exp(mu))), 2
    if the_distn == 'wbl':
        # MATLAB wblfit: X ~ Weibull(a, b) iff log(X) ~ EV(log a, 1/b)
        mu, sigma = evfit(np.log(x))
        a, c = np.exp(mu), 1 / sigma
        return (lambda z: weibull_min.cdf(z, c, scale=a)), 2
    raise ValueError(f"Unknown distribution '{the_distn}'.")

def _chi2gof_pvalue(x: np.ndarray, cdf_func, num_bins: int, n_params: int,
                    e_min: int = 5) -> float:
    """Chi^2 goodness-of-fit p-value, matching MATLAB's chi2gof.

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
    df = len(obs) - 1 - n_params
    if df <= 0:
        return np.nan
    return float(chi2.sf(chi2_stat, df))

def _ks_statistic(counts: np.ndarray, null_cdf: np.ndarray, n: int) -> float:
    """Two-sided KS statistic between an ECDF (from counts at the unique
    sorted data values) and the null CDF evaluated at those values."""
    sample_cdf = np.concatenate(([0.0], np.cumsum(counts) / n))
    delta1 = sample_cdf[:-1] - null_cdf  # jumps approached from the left
    delta2 = sample_cdf[1:] - null_cdf  # jumps approached from the right
    return float(np.max(np.abs(np.concatenate((delta1, delta2)))))

def _kstest_pvalue(x: np.ndarray, cdf_func) -> float:
    """One-sample two-sided KS test p-value, matching MATLAB's kstest as
    called by HT_DistributionTest (null CDF tabulated on the data values
    rounded to 6 decimal places, then linearly interpolated)."""
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
    ks_stat = _ks_statistic(counts, null_cdf, n)

    s = n * ks_stat ** 2
    if (s > 7.24) or ((s > 3.76) and (n > 99)):
        # MATLAB's far-tail approximation (p-values below ~1e-3)
        p = 2 * np.exp(-(2.000071 + 0.331 / np.sqrt(n) + 1.409 / n) * s)
    else:
        # Exact two-sided p-value (Marsaglia, Tsang & Wang), as in MATLAB
        p = kstwo.sf(ks_stat, n)
    return float(min(max(p, 0.0), 1.0))

def _lilliefors_pvalue(x: np.ndarray, the_distn: str) -> float:
    """Lilliefors test p-value, matching MATLAB's lillietest.

    The p-value is found by inverse PCHIP interpolation into the tabulated
    critical values and is clipped to the table range [0.001, 0.5].
    """
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
    ks_stat = _ks_statistic(counts, null_cdf, n)

    table = LILLIE_TABLES[the_distn]
    if n <= 20:
        cvs = table['quantiles'][n - 4]
    else:
        c1, c2, c3 = table['asymptotic']
        cvs = c1 / np.sqrt(n) - c2 / n - c3 / n ** 1.5

    if np.isnan(ks_stat):
        return 0.0  # Inf data; declare highly significant
    if ks_stat < cvs[-1]:  # smallest critical value at end
        return float(LILLIE_ALPHAS[-1])
    if ks_stat >= cvs[0]:  # largest critical value at beginning
        return float(LILLIE_ALPHAS[0])
    # 1-D inverse interpolation into the tabulated quantiles
    pp = PchipInterpolator(LILLIE_ALPHAS, cvs)
    i = int(np.argmax(ks_stat > cvs))  # first index where ks_stat > cvs[i]
    if i == 0:
        return float(LILLIE_ALPHAS[-1])  # ks_stat == cvs[-1] exactly
    return float(brentq(lambda a: pp(a) - ks_stat, LILLIE_ALPHAS[i - 1], LILLIE_ALPHAS[i]))
