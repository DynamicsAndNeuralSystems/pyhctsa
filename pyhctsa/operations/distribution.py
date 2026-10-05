import logging
import warnings
from typing import Dict, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import brentq, least_squares
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import expon, gaussian_kde, gumbel_l, lognorm, norm, rayleigh, uniform, weibull_min, skew, kurtosis

from ..operations.correlation import autocorr, first_crossing
from ..toolboxes.distribution_fits.distfits import betafit, evfit, gamfit, wblfit
from ..robust import bf_exp_fit, bf_fit_density_curve, bf_half_sample_mode, bf_hist_edges, bf_ks_density, bf_residual_stats, bf_runs_z
from ..utils import bin_picker, histc, matlab_quantile, sign_change, simple_binner, x_corr

logger = logging.getLogger('pyhctsa')

def cumulants(x: ArrayLike, cum_what_may: str = 'skew1') -> float:
    """
    Distributional moments of the input data.

    Parameters
    ----------
    x : array-like
        The input time series.
    cum_what_may : str
        The type of higher order moment:
            - 'skew1': skewness
            - 'skew2': skewness correcting for bias
            - 'kurt1': kurtosis
            - 'kurt2': kurtosis correcting for bias

    Returns
    -------
    float
        The specified higher order moment.
    """
    if cum_what_may == 'skew1':
        return skew(x, bias=False)
    if cum_what_may == 'skew2':
        return skew(x, bias=True)
    if cum_what_may == 'kurt1':
        return kurtosis(x, bias=True, fisher=True)
    if cum_what_may == 'kurt2':
        return kurtosis(x, bias=False, fisher=True)
    else:
        return ValueError('Unknown cumulant. Choose either skew1, skew2, kurt1, or kurt2.')

def compare_ks_fit(x: ArrayLike, what_distn: str) -> dict:
    """
    Compares a fitted distribution with the smoothed distribution of the data.

    Fits a standard distribution to the data (by maximum likelihood) and compares it
    with a kernel-smoothed estimate of the distribution of the values. ("KS" here
    means kernel-smoothed, not Kolmogorov-Smirnov.) Both curves are evaluated on a
    common grid of 1000 points that covers the smoothed distribution and the body
    of the fitted distribution (out to where it falls to 1/100 of its peak). They
    are then compared by the area between them, the separation of their peaks,
    their overlap, and the relative entropy.

    The exponential, Rayleigh and gamma distributions require non-negative values,
    and the log-normal and Weibull distributions require positive values; NaN is
    returned if the data do not satisfy this (and for a constant series in the
    Rayleigh and exponential cases). For the beta distribution, the data are first
    rescaled to lie inside (0, 1), and all outputs are then in rescaled units.

    Parameters
    ----------
    x : array-like
        The input data vector.
    what_distn : str
        The type of distribution to fit to the data:
            - 'norm': Gaussian
            - 'ev': extreme value
            - 'uni': uniform
            - 'beta': beta
            - 'rayleigh': Rayleigh
            - 'exp': exponential
            - 'gamma': gamma
            - 'logn': log-normal
            - 'wbl': Weibull

    Returns
    -------
    dict
        adiff: the absolute area between the two distributions (0 for a perfect
        match, at most 2); peaksepy: the maximum of the fitted distribution minus
        that of the smoothed distribution; peaksepx: the position of the peak of
        the fitted distribution minus that of the smoothed distribution; olapint:
        the overlap integral of the two distributions, multiplied by the standard
        deviation of the data so that it does not depend on their scale; relent:
        the relative entropy (Kullback-Leibler divergence), in nats, of the fitted
        distribution from the smoothed distribution.

    Notes
    -----
    adiff, olapint and relent do not depend on the scale of the data, but peaksepy
    and peaksepx do.
    """
    x = np.asarray(x, dtype=float)
    if what_distn not in ('norm', 'ev', 'uni', 'beta', 'rayleigh', 'exp', 'gamma', 'logn', 'wbl'):
        raise ValueError(f"Unknown distribution: {what_distn}.")
    if what_distn == 'beta':
        # clumsily scale to the range (0,1)
        if np.all(x == x[0]):
            logger.warning("Data are a constant; the beta distribution cannot be fitted.")
            return np.nan  # (MATLAB's betafit errors here)
        sd = np.std(x, ddof=1)
        x = (x - np.min(x) + 0.01 * sd) / (np.max(x) - np.min(x) + 0.02 * sd)
    n = len(x)
    x_step = np.std(x, ddof=1) / 100  # set a step size

    # ----------------------------
    # Fit distribution & find the support bounds over which to compare
    # ----------------------------
    # Each branch defines the fitted PDF `pdf_func` and the threshold `thresh` at
    # which to stop searching for the edges of the fitted distribution (1/100 of
    # its peak). The left/right edges are then found either by stepping outwards
    # from a starting point (`left_start`, `right_start`), or, for the positive-only
    # distributions, by pinning the left edge at 0 (`left_start = None`) and
    # walking the right tail from `tail_start`.
    tail_start = None
    if what_distn == 'norm':
        # Normal distribution (normfit uses the unbiased standard deviation)
        loc, scale = np.mean(x), np.std(x, ddof=1)
        pdf_func = lambda z: norm.pdf(z, loc=loc, scale=scale)
        thresh = pdf_func(loc) / 100.0
        left_start = right_start = np.mean(x)

    elif what_distn == 'ev':
        # Extreme value (left Gumbel) distribution
        loc, scale = evfit(x)
        pdf_func = lambda z: gumbel_l.pdf(z, loc=loc, scale=scale)
        thresh = pdf_func(loc) / 100.0
        left_start = right_start = loc

    elif what_distn == 'uni':
        # Uniform distribution (peak of PDF = 1 / (b - a))
        loc, scale = uniform.fit(x)
        pdf_func = lambda z: uniform.pdf(z, loc=loc, scale=scale)
        thresh = pdf_func(np.mean(x)) / 100.0
        left_start = right_start = np.mean(x)

    elif what_distn == 'beta':
        a, b = betafit(x)
        pdf_func = lambda z: beta_dist.pdf(z, a, b)
        thresh = 1e-5  # ok -- consistent since all scaled to the same range
        left_start = right_start = np.mean(x)

    elif what_distn == 'rayleigh':
        if np.any(x < 0):
            logger.warning("The data are not positive, but Rayleigh is a positive-only distribution.")
            return np.nan
        if np.all(x == x[0]):
            logger.warning("Data are a constant.")
            return np.nan
        scale = np.sqrt(np.mean(x ** 2) / 2)  # raylfit
        pdf_func = lambda z: rayleigh.pdf(z, scale=scale)
        thresh = pdf_func(scale) / 100.0  # peak is at the scale parameter
        left_start, tail_start = None, scale

    elif what_distn == 'exp':
        if np.any(x < 0):
            logger.warning("The data contains negative values, but Exponential is a positive-only distribution.")
            return np.nan
        if np.all(x == x[0]):
            logger.warning("Data are a constant.")
            return np.nan
        # Exponential distribution (equivalent to MATLAB's expfit); peak is at 0
        lam = np.mean(x)
        pdf_func = lambda z: expon.pdf(z, loc=0, scale=lam)
        thresh = pdf_func(0) / 100.0
        left_start, tail_start = None, 0.0

    elif what_distn == 'gamma':
        if np.any(x < 0):
            logger.warning("The data contains negative values, but Gamma is a positive-only distribution.")
            return np.nan
        a, b = gamfit(x)
        if not (np.isfinite(a) and np.isfinite(b)):
            logger.warning("No finite gamma fit for this data.")
            return np.nan
        pdf_func = lambda z: gamma_dist.pdf(z, a, scale=b)
        if a < 1:
            thresh = pdf_func(0.0) / 100.0  # unbounded at 0
        else:
            thresh = pdf_func((a - 1) * b) / 100.0
        left_start, tail_start = None, a * b

    elif what_distn == 'logn':
        if np.any(x <= 0):
            logger.warning("The data are not positive, but Log-Normal is a positive-only distribution.")
            return np.nan
        # Log-normal distribution (lognfit uses the unbiased std of log(x)); peak is at the mode
        lx = np.log(x)
        mu, sigma = np.mean(lx), np.std(lx, ddof=1)
        mode = np.exp(mu - sigma ** 2)
        pdf_func = lambda z: lognorm.pdf(z, s=sigma, loc=0, scale=np.exp(mu))
        thresh = pdf_func(mode) / 100.0
        left_start, tail_start = None, mode

    else:  # 'wbl'
        if np.any(x <= 0):
            logger.warning("The data are not positive, but Weibull is a positive-only distribution.")
            return np.nan
        a, c = wblfit(x)  # scale, shape
        if not (np.isfinite(a) and np.isfinite(c)):
            logger.warning("No finite Weibull fit for this data.")
            return np.nan
        pdf_func = lambda z: weibull_min.pdf(z, c, scale=a)
        if c <= 1:
            thresh = pdf_func(0.0)
        else:
            thresh = pdf_func(a * ((c - 1) / c) ** (1 / c)) / 100.0
        left_start, tail_start = None, 0.0

    if tail_start is None:
        xf = _find_bounds(pdf_func, left_start, right_start, x_step, thresh)
    else:
        xf = [0.0, _walk_tail(pdf_func, tail_start, x_step, thresh)]

    # ----------------------------
    # Estimate smoothed empirical distribution
    # ----------------------------
    f, xi, _ = bf_ks_density(x)
    xi = xi[f > 1e-6]  # only keep values greater than 1E-6
    if xi.size == 0:
        return np.nan
    # Round outward
    xi = [np.floor(xi[0] * 10) / 10, np.ceil(xi[-1] * 10) / 10]
    # Find appropriate range [x1 x2] that incorporates the full range of both
    x1 = min(xf[0], xi[0])
    x2 = max(xf[1], xi[1])

    # Rerun both over the same range
    xi = np.linspace(x1, x2, 1000)
    f, _, _ = bf_ks_density(x, xi)
    with np.errstate(all='ignore'):
        ffit = pdf_func(xi)

    # ----------------------------
    # Statistics
    # ----------------------------
    # (as in MATLAB, max/argmax skip NaN, which a degenerate fit, such as one to a
    # constant series, produces)
    dx = xi[1] - xi[0]
    out = {}
    with np.errstate(all='ignore'):
        # ADIFF: returns absolute area between the curves
        out['adiff'] = np.sum(np.abs(f - ffit) * dx)
        # PEAKSEPY: separation (in y) between the maxima of each distribution
        out['peaksepy'] = _matlab_max(ffit) - _matlab_max(f)
        # PEAKSEPX: separation (in x) between the maxima of each distribution
        out['peaksepx'] = xi[_matlab_argmax(ffit)] - xi[_matlab_argmax(f)]
        # OLAPINT: overlap integral between the two curves; multiplying by std(x) makes
        # this scale-invariant
        out['olapint'] = np.sum(f * ffit * dx) * np.std(x, ddof=1)
        # RELENT: relative entropy of the two distributions (points where either
        # density is exactly zero are skipped: 0*log(0) := 0)
        r = (ffit != 0) & (f != 0)
        out['relent'] = np.sum(f[r] * np.log(f[r] / ffit[r]) * dx)

    return out


def _matlab_max(v: np.ndarray) -> float:
    """MATLAB's ``max(v)``: ignores NaN (NaN only if all are NaN)."""
    return np.nan if np.all(np.isnan(v)) else float(np.nanmax(v))


def _matlab_argmax(v: np.ndarray) -> int:
    """Index from MATLAB's ``[~, i] = max(v)``: the first maximum, skipping NaN (1st element if all NaN)."""
    return 0 if np.all(np.isnan(v)) else int(np.nanargmax(v))


def _find_bounds(pdf_func, start_left, start_right, x_step, thresh):
    """Expand left/right until pdf falls below threshold."""
    xf = [start_left, start_right]

    # Left search (only if start_left is not None)
    if start_left is not None:
        ange = 10
        while ange > thresh:
            xf[0] -= x_step
            ange = pdf_func(xf[0])

    # Right search
    ange = 10
    while ange > thresh:
        xf[1] += x_step
        ange = pdf_func(xf[1])

    return xf


def _walk_tail(pdf_func, x_start, x_step, thresh):
    """First grid point x_start + k*x_step (k >= 1) at which a unimodal pdf has fallen to
    <= thresh: what stepping outward from x_start would return, but found by
    bracketing the tail crossing and root-finding, then snapping to the grid. The
    stepping form needs ~100*(scale/std(x)) pdf evaluations, which is effectively
    unbounded for near-constant positive-valued data."""
    if not (10 > thresh):  # replicate the stepping loop's initial ange = 10 sentinel
        return x_start  # (e.g., thresh = inf for a gamma with shape < 1)

    # The pdf may still be rising over the first step (e.g., a Weibull/gamma with
    # shape > 1 walked from 0), in which case the stepping loop stops immediately:
    x_end = x_start + x_step
    if not (pdf_func(x_end) > thresh):
        return x_end

    # Bracket the tail crossing by doubling the distance from x_start
    lo = x_end
    stride = max(x_step, abs(x_start) + x_step)
    hi = lo + stride
    num_doublings = 0
    while pdf_func(hi) > thresh:
        stride *= 2
        hi = lo + stride
        num_doublings += 1
        if num_doublings > 200 or not np.isfinite(hi):
            raise RuntimeError('Could not bracket the tail of the fitted distribution')
    x_cross = brentq(lambda z: pdf_func(z) - thresh, lo, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps)

    # Snap to the stepping grid, then correct for any floating-point boundary
    # ambiguity so the result satisfies the loop's own stopping condition
    # (pdf > thresh at k-1, pdf <= thresh at k):
    k = max(1, int(np.ceil((x_cross - x_start) / x_step)))
    while k > 1 and not (pdf_func(x_start + (k - 1) * x_step) > thresh):
        k -= 1
    while pdf_func(x_start + k * x_step) > thresh:
        k += 1
    return x_start + k * x_step


def withinp(x: ArrayLike, p: float = 1.0, mean_or_median: str = 'mean') -> float:
    """
    Proportion of data points within p standard deviations of the mean or median.

    Parameters
    -----------
    x : array-like
        The input time series.
    p : float
        The number (proportion) of standard deviations. Default is 1.0
    mean_or_median : str 
        Whether to use units of 'mean' and standard deviation, or 'median' 
        and rescaled interquartile range. Default is ``'mean'``.

    Returns
    --------
    float: 
        The proportion of data points within p standard deviations.
    """
    x = np.asarray(x)
    N = len(x)

    if mean_or_median == 'mean':
        mu = np.mean(x)
        sig = np.std(x, ddof=1)
    elif mean_or_median == 'median':
        mu = np.median(x)
        iqr_val = np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')
        sig = iqr_val / 1.35
    else:
        raise ValueError(f"Unknown setting: '{mean_or_median}'")

    # The withinp statistic:
    return np.divide(np.sum((x >= mu - p * sig) & (x <= mu + p * sig)), N)

def unique(y: ArrayLike) -> float:
    """
    The proportion of the time series that are unique values.

    Parameters
    ----------
    y : array-like
        The input time series or data vector.

    Returns
    -------
    float
        The proportion of time series that are unique values.
    """
    y = np.asarray(y)
    return np.divide(len(np.unique(y)), len(y))

def spread(y: ArrayLike, spread_measure: str = 'std') -> float:
    """
    Measure of spread of the input time series.

    Returns the spread of the raw data vector using different statistical measures.

    Parameters
    ----------
    y : array-like
        The input time series.
    spread_measure : str, optional
        The spread measure to use:

        - 'std': standard deviation
        - 'iqr': interquartile range 
        - 'mad': mean absolute deviation
        - 'mead': median absolute deviation

        Default is ``'std'``.

    Returns
    -------
    float
        The calculated spread measure.
    """
    y = np.asarray(y)
    if spread_measure == 'std':
        out = np.std(y, ddof=1)
    elif spread_measure == 'iqr':
        q75 = np.quantile(y, 0.75, method='hazen')
        q25 = np.quantile(y, 0.25, method='hazen')
        out = q75 - q25
    elif spread_measure == 'mad':
        # mean absolute deviation
        out = np.mean(np.absolute(y - np.mean(y, None)), None)
    elif spread_measure == 'mead':
        # median absolute deviation
        out = np.median(np.absolute(y - np.median(y, None)), None)
    else:
        raise ValueError('spread must be one of std, iqr, mad or mead')
    
    return out

def quantile(y: ArrayLike, p: float = 0.5) -> float:
    """ 
    Calculates the quantile value at a specified proportion, p.

    Parameters
    ----------
    y : array-like
        The input data vector.
    p : float 
        The quantile proportion. Default is 0.5, which is the median.

    Returns
    -------
    float: 
        The calculated quantile value.
    """
    y = np.asarray(y)    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in [0,1]")
    
    return float(np.quantile(y, p, method = 'hazen'))

def proportion_values(x: ArrayLike, prop_what: str = 'positive') -> float:
    """
    Calculate the proportion of values meeting specific conditions in a time series.

    Parameters
    ----------
    x : array-like
        Input time series.
    prop_what : str, optional
        Type of values to count:

        - 'zeros': values equal to zero
        - 'positive': values strictly greater than zero
        - 'geq0': values greater than or equal to zero

        Default is ``'positive'``.

    Returns
    -------
    float
        Proportion of values meeting the specified condition.
    """
    x = np.asarray(x)
    N = len(x)

    if prop_what == 'zeros':
        # returns the proportion of zeros in the input vector
        out = sum(x == 0) / N
    elif prop_what == 'positive':
        out = sum(x > 0) / N
    elif prop_what == 'geq0':
        out = sum(x >= 0) / N
    else:
        raise ValueError(f"Unknown condition to measure: {prop_what}")

    return out

def pleft(y: ArrayLike, th: float = 0.1) -> float:
    """
    Distance from the mean at which a given proportion of data are more distant.
    
    Measures the maximum distance from the mean at which a given fixed proportion, `th`, of 
    the time-series data points are further. Normalizes by the standard deviation of the time 
    series.
    
    Parameters
    ----------
    y : array-like
        The input data vector.
    th : float, optional
        The proportion of data further than `th` from the mean. Default is 0.1.
    
    Returns
    -------
    float
        The distance from the mean normalized by the standard deviation.
    """
    y = np.asarray(y)
    p = np.quantile(np.abs(y - np.mean(y)), 1-th, method='hazen')
    # A proportion, th, of the data lie further than p from the mean
    out = np.divide(p, np.std(y, ddof=1))

    return float(out)

def min_max(y: ArrayLike, min_or_max: str = 'max') -> float:
    """
    The maximum and minimum values of the input data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    min_or_max : str, optional
        Return either the minimum or maximum of y:

        - 'min': minimum of y
        - 'max': maximum of y

        Default is ``'max'``.

    Returns
    -------
    float
        The calculated min or max value.
    """
    y = np.asarray(y)
    if min_or_max == 'max':
        out = max(y)
    elif min_or_max == 'min':
        out = min(y)
    else:
        raise ValueError(f"Unknown method '{min_or_max}'")
    
    return out

def mean(y: ArrayLike, mean_type: str = 'arithmetic') -> float:
    """
    A given measure of location of a data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    mean_type : str, optional
        Type of mean to calculate:

        - 'norm' or 'arithmetic': standard arithmetic mean
        - 'median': middle value (50th percentile)
        - 'geom': geometric mean (nth root of product)
        - 'harm': harmonic mean (reciprocal of mean of reciprocals)
        - 'rms': root mean square (quadratic mean)
        - 'iqm': interquartile mean (mean of values between Q1 and Q3)
        - 'midhinge': average of first and third quartiles

        Default is ``'arithmtic'``.

    Returns
    -------
    float
        The calculated mean value.
    """
    y = np.asarray(y)
    N = len(y)

    if mean_type in ['norm', 'arithmetic']:
        out = np.mean(y)
    elif mean_type == 'median': # median
        out = np.median(y)
    elif mean_type == 'geom': # geometric mean
        out = stats.gmean(y)
    elif mean_type == 'harm': # harmonic mean
        out = N/sum(y**(-1))
    elif mean_type == 'rms':
        out = np.sqrt(np.mean(y**2))
    elif mean_type == 'iqm': # interquartile mean
        p = np.percentile(y, [25, 75], method='hazen')
        out = np.mean(y[(y >= p[0]) & (y <= p[1])])
    elif mean_type == 'midhinge':  # average of 1st and third quartiles
        p = np.percentile(y, [25, 75], method='hazen')
        out = np.mean(p)
    else:
        raise ValueError(f"Unknown mean type '{mean_type}'")

    return float(out)

def high_low_mu(y: ArrayLike) -> float:
    """
    The high_low_mu statistic.

    The high_low_mu statistic is the ratio of the mean of the data that is above the
    (global) mean compared to the mean of the data that is below the global mean.

    Parameters
    ----------
    y: array-like
        The input data vector

    Returns
    --------
    float
        The high_low_mu statistic.
    """
    y = np.asarray(y)
    mu = np.mean(y) # mean of data
    mhi = np.mean(y[y > mu]) # mean of data above the mean
    mlo = np.mean(y[y < mu]) # mean of data below the mean
    out = np.divide((mhi-mu), (mu-mlo)) # ratio of the differences

    return out

def fit_mle(y: ArrayLike, fit_what: str = 'gaussian') -> Union[Dict[str, float], float]:
    """
    Maximum likelihood distribution fit to data.

    Fits a specified probability distribution to the data using maximum likelihood 
    estimation (MLE) and returns the fitted parameters.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    fit_what : {'gaussian', 'uniform', 'geometric'}, optional
        Distribution type to fit:

        - 'gaussian': Normal distribution (returns mean and std)
        - 'uniform': Uniform distribution (returns bounds a and b)
        - 'geometric': Geometric distribution (returns p parameter)

        Default is ``'gaussian'``.

    Returns
    -------
    Union[Dict[str, float], float]
        For 'gaussian':
            dict with keys:
                - 'mean': location parameter
                - 'std': scale parameter
        For 'uniform':
            dict with keys:
                - 'a': lower bound
                - 'b': upper bound
        For 'geometric':
            float: success probability p
    """
    y = np.asarray(y)
    out = {}
    if fit_what == 'gaussian':
        loc, scale = stats.norm.fit(y, method="MLE")
        out['mean'] = loc
        out['std'] = scale
    elif fit_what == 'uniform':
        loc, scale = stats.uniform.fit(y, method="MLE")
        out['a'] = loc
        out['b'] = loc + scale 
    elif fit_what == 'geometric':
        samp_mean = np.mean(y)
        p = 1/(1+samp_mean)
        return p
    else:
        raise ValueError(f"Invalid fit specifier, {fit_what}")

    return out

def cv(x: ArrayLike, k: int = 1) -> float:
    """
    Calculate the coefficient of variation of order k.

    The coefficient of variation (CV) of order :math:`k` is defined as

    .. math::

        \\left( \\frac{\\sigma}{\\mu} \\right)^{k},

    where :math:`\\sigma` is the standard deviation and :math:`\\mu` is the
    mean of the input data. It is negative (for odd :math:`k`) when the mean is
    negative, and undefined when the mean is zero, so NaN is returned when the mean
    is at the level of rounding error relative to the spread
    (:math:`|\\mu| < 10^{-10}\\sigma`, as for a centered or z-scored series).

    Parameters
    ----------
    x : array-like
        Input time series or data vector.

    k : int, optional
        Order of the coefficient of variation. Default is 1.

    Returns
    -------
    float
        The coefficient of variation of order :math:`k`, or NaN if the mean is zero
        up to rounding error.
    """
    if not isinstance(k, int) or k < 0:
        logger.warning('k should probably be a positive integer')
        # carry on with just this warning, though

    # Compute the coefficient of variation (of order k) of the data
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if abs(mu) < 1e-10 * sigma:
        # the mean is zero up to rounding error (e.g., a centered or z-scored series), so
        # the ratio is rounding noise of order 1e17
        return np.nan
    return float((sigma / mu) ** k)

def custom_skewness(y: ArrayLike, what_skew: str = 'pearson') -> float:
    """
    Compute custom skewness measures of a time series.

    Calculates the Pearson skewness (using the median or the mode) or the Bowley
    (quartile) skewness coefficient.

    The Pearson skewness (with the median) is defined as

    .. math::

        \\frac{3(\\mu - \\tilde{x})}{\\sigma},

    where :math:`\\mu` is the mean, :math:`\\tilde{x}` is the median,
    and :math:`\\sigma` is the standard deviation. The mode-based version is
    :math:`(\\mu - \\text{mode})/\\sigma`, with the mode estimated by the half-sample
    mode (:func:`pyhctsa.robust.bf_half_sample_mode`), which needs no bins.

    The Bowley skewness is defined as

    .. math::

        \\frac{Q_3 + Q_1 - 2Q_2}{Q_3 - Q_1},

    where :math:`Q_1`, :math:`Q_2`, and :math:`Q_3` are the first,
    second (median), and third quartiles, respectively.

    Parameters
    ----------
    y : array-like
        Input time series.

    what_skew : str, optional
        Skewness measure to compute.

        - ``"pearson"`` (or ``"pearsonMedian"``): Pearson skewness coefficient
          from the median.
        - ``"pearsonMode"``: Pearson skewness coefficient from the half-sample
          mode.
        - ``"bowley"``: Bowley (quartile) skewness coefficient.

        Default is ``"pearson"``.

    Notes
    -----
    The mode of a histogram depends on the number and edges of its bins, so the
    half-sample mode is used instead: closed-form, robust, and without a smoothing
    parameter.

    Returns
    -------
    float
        The calculated skewness value.

        - Positive values indicate right skew.
        - Negative values indicate left skew.
        - Zero indicates symmetry.
    """
    y = np.asarray(y)
    out = 0.0
    if what_skew == 'pearsonMode':
        out = (np.mean(y) - bf_half_sample_mode(y)) / np.std(y, ddof=1)
    elif what_skew in ('pearson', 'pearsonMedian'):
        out = (3 * (np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif what_skew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    else:
        raise ValueError(f"Unknown skewness type '{what_skew}'.")
    
    return float(out)

def burstiness(y: ArrayLike) -> dict:
    """
    Calculate burstiness statistics of a time series.
    
    Implements both the original Goh & Barabasi burstiness and
    the improved Kim & Jo version for finite time series.

    References
    ----------
    .. [1] Goh & Barabasi (2008). Europhys. Lett. 81, 48002
    .. [2] Kim & Jo (2016). http://arxiv.org/pdf/1604.01125v1.pdf
    
    Parameters
    ----------
    y : array-like
        Input time series
    
    Returns
    -------
    dict:

        - 'B': Original burstiness statistic
        - 'B_Kim': Improved burstiness for finite series
    """
    y = np.asarray(y)
    me = np.mean(y)
    std = np.std(y, ddof=1)

    r = np.divide(std,me) # coefficient of variation
    b = np.divide((r - 1), (r + 1)) # Original Goh and Barabasi burstiness statistic, B

    # improved burstiness statistic, accounting for scaling for finite time series
    # Kim and Jo, 2016, http://arxiv.org/pdf/1604.01125v1.pdf
    N = len(y)
    p1 = np.sqrt(N+1)*r - np.sqrt(N-1)
    p2 = (np.sqrt(N+1)-2)*r + np.sqrt(N-1)

    b_kim = np.divide(p1, p2)

    out = {'B': b, 'B_Kim': b_kim}

    return out

def moments(y: ArrayLike, the_mom: int = 0, do_normalize: bool = True) -> float:
    """
    A moment of the distribution of the input time series.
    Returns the standardized central moment: the ``the_mom``-th central moment
    divided by the standard deviation raised to the power ``the_mom`` (or, with
    ``do_normalize=False``, the raw central moment, as hctsa's
    ``DN_Moments(y, theMom, false)``).

    Parameters
    ----------
    y : array-like
        Input time series or data vector.
    the_mom: int, optional
        The moment to calculate. Default is 0.
    do_normalize: bool, optional
        Whether to divide by std(y)**the_mom, giving the scale-invariant
        standardized moment (True, the default), or to return the raw central
        moment (False).

    Returns
    -------
    float
        The calculated moment.
    """
    y = np.asarray(y)

    if not do_normalize:
        return stats.moment(y, the_mom)
    return stats.moment(y, the_mom) / np.std(y, ddof=1) ** the_mom

def _matlab_std(a) -> float:
    """Sample standard deviation as MATLAB's std: 0 (not NaN) for a single value."""
    a = np.asarray(a, dtype=float)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def _fit_lin_gof(x: np.ndarray, y: np.ndarray) -> tuple:
    """Ordinary least-squares line ``a*x + b``, with R^2 and root-mean-square error (n - 2
    degrees of freedom). Returns (a, b, R^2, RMSE), all NaN for fewer than 3 points or a
    constant curve."""
    n = len(y)
    sst = np.sum((y - np.mean(y)) ** 2)
    if n < 3 or not sst > 0:
        return (np.nan,) * 4
    a, b = np.polyfit(x, y, 1)
    sse = np.sum((y - (a * x + b)) ** 2)
    return a, b, 1 - sse / sst, np.sqrt(sse / (n - 2))


def _exp_fit_outputs(x: np.ndarray, y: np.ndarray) -> tuple:
    """Rate b, R^2 and RMSE of the global exponential fit ``a*exp(b*x) + c`` (:func:`pyhctsa.robust.bf_exp_fit`).

    The amplitude a and offset c are not output: they are poorly determined when the
    curve is close to a straight line, as a and c then become large and opposite in sign.
    """
    f = bf_exp_fit(x, y, True)
    return f['b'], f['r2'], f['rmse']


def outlier_include(y: ArrayLike, threshold_how: str = 'abs', inc: float = 0.01,
                    fixed_thresh: Union[float, None] = None) -> dict:
    """
    How the timing and spacing of extreme values change as the threshold rises.

    Raises a threshold th from 0 to the maximum value of the series, in increments of
    ``inc``, and at each threshold takes the "events": the points at or beyond it (for
    'abs', values with abs(y) >= th; for 'pos', y >= th; for 'neg', y <= -th). The
    threshold is applied to y itself, so the series should be z-scored. At each
    threshold it records:

    1. the mean gap (in samples) between successive events, and its standard error
       (std of the gaps / sqrt of their number),
    2. the percentage of points that are events (the number of gaps over the number of
       candidate points, times 100),
    3. the median and mean time of the events, rescaled so that the start of the series
       is -1, the middle is 0 and the end is 1, and std(times)/sqrt(their number)
       (in samples).

    The sweep stops when events are 2% or fewer of the points. The outputs measure how
    these curves change with th, using exponential [f(x) = a*exp(b*x) + c] and linear
    [f(x) = a*x + b] fits, and simple statistics across thresholds. The exponential fits
    are global least-squares fits (:func:`pyhctsa.robust.bf_exp_fit`), which need no
    starting point; only the rate b and the fit quality are returned. If a fit is
    degenerate (a constant curve, too few points), its outputs are NaN.

    If ``fixed_thresh`` is given, the sweep and fits are skipped, and the statistics in
    (1)-(3) are returned for that one threshold.

    Parameters
    ----------
    y : array-like
        The input time series (ideally z-scored).
    threshold_how : {'abs', 'pos', 'neg'}, optional
        The method for determining outliers:

            - 'abs': values furthest from zero in either direction (default).
            - 'pos': the greatest positive values.
            - 'neg': the greatest negative values.

    inc : float, optional
        The increment to move through (in units of the standard deviation if the
        time series is z-scored). Default is 0.01. Unused when ``fixed_thresh`` is given.
    fixed_thresh : float, optional
        A single threshold (in the units of y, e.g., 2 for two standard deviations of a
        z-scored series). If given, the sweep is skipped.

    Returns
    -------
    dict
        From the sweep (``fixed_thresh`` not given):

        - ``mfexpb``, ``mfexpr2``, ``mfexprmse``: the rate b, R^2 and root-mean-square
          error of the exponential fit to the mean gap vs. th;
        - ``nfexpb``, ``nfexpr2``, ``nfexprmse``: the same for an exponential fit to the
          percentage of points that are events vs. th;
        - ``nfla``, ``nflb``, ``nflr2``, ``nflrmse``: slope a, intercept b, R^2 and
          RMSE of a linear fit to the percentage of points that are events vs. th;
        - ``mdtm``, ``mdtmd``, ``mdtstd``: mean, median and standard deviation of the
          mean gap across thresholds;
        - ``mdrm``, ``mdrmd``, ``mdrstd``: mean, median and standard deviation, across
          thresholds, of the median time of the events (-1 to 1);
        - ``mrm``, ``mrmd``, ``mrstd``: the same for the mean time of the events;
        - ``xcmerr1``, ``xcmerrn1``: cross-correlation between the mean gap and its
          standard error across thresholds, at lags +1 and -1;
        - ``stdrfexpb``, ``stdrfexpr2``, ``stdrfexprmse``:
          the rate and fit quality of an exponential fit to
          std(times)/sqrt(their number) vs. th;
        - ``stdrfla``, ``stdrflb``, ``stdrflr2``, ``stdrflrmse``: the same for a linear fit.

        From a single threshold (``fixed_thresh`` given; all NaN except ``propIncluded``
        if events are 2% or fewer of the points): ``meanDt``, ``seDt`` (the mean gap
        between events and its standard error), ``propIncluded`` (the percentage of
        points that are events), ``medianRelTime``, ``meanRelTime`` (the median and mean
        time of the events, -1 to 1) and ``stdRelTime`` (std(times)/sqrt(their number),
        in samples).

        A constant time series returns NaN.
    """
    y = np.asarray(y)

    # Handle constant time series
    if np.all(y[0] == y):
        logger.warning("The time series is a constant!")
        return np.nan

    N = len(y)
    if threshold_how not in ('abs', 'pos', 'neg'):
        raise ValueError(f"Invalid thresholdHow: '{threshold_how}'. Must be 'abs', 'pos', or 'neg'.")

    def _events(idx, th):
        """Indices (of those in idx) of events at threshold th."""
        if threshold_how == 'abs':
            return idx[np.abs(y[idx]) >= th]
        if threshold_how == 'pos':
            return idx[y[idx] >= th]
        return idx[y[idx] <= -th]

    total_points = {'abs': N, 'pos': np.sum(y >= 0), 'neg': np.sum(y <= 0)}[threshold_how]
    trim_threshold = 2  # percent

    # ----------------------------
    # Single threshold: skip the sweep and curve fits
    # ----------------------------
    if fixed_thresh is not None:
        r = _events(np.arange(N), fixed_thresh)
        time_diffs = np.diff(r)
        mean_dt = np.mean(time_diffs) if len(time_diffs) > 0 else np.nan
        prop_included = len(time_diffs) / total_points * 100
        # Same "too few events to say anything meaningful" bar as the sweep
        if np.isnan(mean_dt) or prop_included <= trim_threshold:
            return {'meanDt': np.nan, 'seDt': np.nan, 'propIncluded': prop_included,
                    'medianRelTime': np.nan, 'meanRelTime': np.nan, 'stdRelTime': np.nan}
        r1 = r + 1  # MATLAB's 1-based event times
        return {
            'meanDt': mean_dt,
            'seDt': _matlab_std(time_diffs) / np.sqrt(len(time_diffs)),
            'propIncluded': prop_included,
            'medianRelTime': np.median(r1) / (N / 2) - 1,
            'meanRelTime': np.mean(r1) / (N / 2) - 1,
            'stdRelTime': _matlab_std(r) / np.sqrt(len(r)),
        }

    # Initialize thresholds based on method
    if threshold_how == 'abs':
        thresholds = np.arange(0, max(abs(y)), inc)
    elif threshold_how == 'pos':
        thresholds = np.arange(0, max(y), inc)
    else:
        thresholds = np.arange(0, max(-y), inc)

    if len(thresholds) == 0:
        logger.warning("Error setting increments through the time-series values")
        return np.nan

    # Calculate statistics of over-threshold events, looping over thresholds. Stop as
    # soon as too few events remain to be useful: raising the threshold can only shrink
    # the set of events, so the criteria keep failing for all higher thresholds. This
    # also lets each threshold search only the previous one's events (in time order).
    # Columns: [mean_diff, std_err, percentage, median_pos, mean_pos, std_pos]
    rows = []
    r = np.arange(N)
    for threshold in thresholds:
        r = _events(r, threshold)
        # Intervals between consecutive over-threshold events
        time_diffs = np.diff(r)
        if len(time_diffs) == 0:
            break
        prop_included = len(time_diffs) / total_points * 100  # percentage of events
        if prop_included <= trim_threshold:
            break
        r1 = r + 1  # event times use 1-based indices, as in MATLAB
        rows.append([
            np.mean(time_diffs),  # mean time between events
            _matlab_std(time_diffs) / np.sqrt(len(time_diffs)),  # standard error
            prop_included,
            np.median(r1) / (N / 2) - 1,  # median position (-1 to 1)
            np.mean(r1) / (N / 2) - 1,  # mean position (-1 to 1)
            _matlab_std(r) / np.sqrt(len(r)),  # position std error
        ])
    statistics = np.array(rows).reshape(-1, 6)
    thresholds = thresholds[:len(statistics)]

    results = {}

    # Fit an exponential to the mean inter-event interval as a function of the threshold
    mfexp = _exp_fit_outputs(thresholds, statistics[:, 0])
    results.update(dict(zip(['mfexpb', 'mfexpr2', 'mfexprmse'], mfexp)))

    # Fit an exponential, then a linear trend, to the percentage of points included
    nfexp = _exp_fit_outputs(thresholds, statistics[:, 2])
    results.update(dict(zip(['nfexpb', 'nfexpr2', 'nfexprmse'], nfexp)))
    nfl = _fit_lin_gof(thresholds, statistics[:, 2])
    results.update(dict(zip(['nfla', 'nflb', 'nflr2', 'nflrmse'], nfl)))

    # Basic statistics on mean times
    results.update({
        'mdtm': np.mean(statistics[:, 0]),
        'mdtmd': np.median(statistics[:, 0]),
        'mdtstd': _matlab_std(statistics[:, 0])
    })

    # Statistics on median position deviations
    results.update({
        'mdrm': np.mean(statistics[:, 3]),
        'mdrmd': np.median(statistics[:, 3]),
        'mdrstd': _matlab_std(statistics[:, 3])
    })

    # Statistics on mean position deviations
    results.update({
        'mrm': np.mean(statistics[:, 4]),
        'mrmd': np.median(statistics[:, 4]),
        'mrstd': _matlab_std(statistics[:, 4])
    })

    # Cross-correlation between mean and error
    _, cross_corr = x_corr(statistics[:, 0], statistics[:, 1], max_lags=1)
    results.update({
        'xcmerr1': cross_corr[-1],
        'xcmerrn1': cross_corr[0]
    })

    # Fit an exponential, then a linear trend, to the std of event times
    stdrfexp = _exp_fit_outputs(thresholds, statistics[:, 5])
    results.update(dict(zip(['stdrfexpb', 'stdrfexpr2', 'stdrfexprmse'], stdrfexp)))
    stdrfl = _fit_lin_gof(thresholds, statistics[:, 5])
    results.update(dict(zip(['stdrfla', 'stdrflb', 'stdrflr2', 'stdrflrmse'], stdrfl)))

    return results

def outlier_test(y: ArrayLike, p: float = 2,
                 just_me: Union[str, None] = None) -> Union[dict, float]:
    """
    How distributional statistics depend on distributional outliers.

    Removes the p% of highest and lowest values in the time series (i.e., 2*p% removed in total)
    and returns the ratio of either the mean or the standard deviation of the time series,
    before and after this transformation.

    Parameters
    ----------
    y : array-like
        The input data vector.
    p : float
        The percentage of values to remove beyond upper and lower percentiles. Default is 2.
    just_me : {'mean', 'std'}, optional
        If specified, just returns a number:

        - 'mean': returns the mean of the middle portion of the data
        - 'std': returns the std of the middle portion of the data

        If None (default), returns a dictionary.

    Returns
    -------
    float or dict
        If just_me is specified, returns the mean or std of the middle portion of the data.
        Otherwise, returns a dictionary.
    """

    # mean of the middle (100-2*p)% of the data
    y = np.array(y)
    lower_bound, upper_bound = matlab_quantile(y, [p / 100, (100 - p) / 100])
    
    middle_portion = y[(y > lower_bound) & (y < upper_bound)]
    
    # Mean of the middle (100-2*p)% of the data
    mean_middle = np.mean(middle_portion)
    
    # Std of the middle (100-2*p)% of the data
    std_middle = np.std(middle_portion, ddof=1) / np.std(y, ddof=1)  # [although std(y) should be 1]

    out = {'mean': mean_middle, 'std': std_middle}

    if just_me == 'mean':
        return out['mean']
    elif just_me == 'std':
        return out['std']     
    
    return out

def trimmed_mean(x: ArrayLike, p_exclude: float = 0.0) -> float:
    """
    Mean of the trimmed time series.

    Returns the mean of the time series after removing a specified percentage of 
    the highest and lowest values.

    Parameters
    ----------
    x : array-like
        The input time series or data vector
    p_exclude : float, optional
        The percentage of highest and lowest values to exclude from the mean 
        calculation. Default is 0.0, which gives the standard mean.

    Returns
    -------
    float
        The mean of the trimmed time series.
    """
    if not 0 <= p_exclude < 100:
        raise ValueError("The 'percent' argument must be between 0 and 100.")

    x = np.asarray(x)
    # handle the edge case of an empty array
    if x.size == 0:
        return np.nan

    # sort the array; np.sort conveniently places NaNs at the end
    x_sorted = np.sort(x)

    # count non-NaN values for an accurate trimming calculation
    non_nan_count = np.count_nonzero(~np.isnan(x_sorted))
    if non_nan_count == 0:
        return np.nan

    # calculate the number of elements to trim from each end (k)
    k = non_nan_count * (p_exclude / 100.0) / 2.0

    lowercut = int(np.ceil(k - 0.5))

    # If all data would be trimmed, return NaN
    if (2 * lowercut) >= non_nan_count:
        return np.nan

    # slice the sorted, non-NaN part of the array
    trimmed_x = x_sorted[lowercut : non_nan_count - lowercut]

    out = np.mean(trimmed_x)

    return float(out)

def histogram_asymmetry(y: ArrayLike, num_bins: int = 10, do_simple: bool = True) -> dict:
    """
    Calculate measures of histogram asymmetry for a time series.

    Computes various measures of asymmetry by analyzing the positive and negative 
    values in the histogram distribution separately.

    Parameters
    ----------
    y : array-like
        Input time series
    num_bins : int, optional
        Number of bins to use in histogram calculation. Default is 10.
    do_simple : bool, optional
        If True, uses linearly spaced bins. If False, uses optimized bin edges. Default is `True`.

    Returns
    -------
    dict
        Dictionary containing asymmetry measures.
    """
    y = np.asarray(y)
    # compute the histogram seperately from positive and negative values in the data
    y_pos = y[y > 0]  # filter out the positive vals
    y_neg = y[y < 0]  # filter out the negative vals

    if do_simple:
        counts_pos, bin_edges_pos = simple_binner(y_pos, num_bins)
        counts_neg, bin_edges_neg = simple_binner(y_neg, num_bins)
    else:
        bin_edges_pos = bin_picker(y_pos.min(), y_pos.max(), num_bins)
        counts_pos = histc(y_pos, bin_edges_pos)[:-1]
        bin_edges_neg = bin_picker(y_neg.min(), y_neg.max(), num_bins)
        counts_neg = histc(y_neg, bin_edges_neg)[:-1]
    # normalise by the total counts
    n_non_zero = np.sum(y != 0)
    p_pos = np.divide(counts_pos, n_non_zero)
    p_neg = np.divide(counts_neg, n_non_zero)

    # compute bin centers from bin edges
    bin_centers_pos = np.mean([bin_edges_pos[:-1], bin_edges_pos[1:]], axis=0)
    bin_centers_neg = np.mean([bin_edges_neg[:-1], bin_edges_neg[1:]], axis=0)

    # Histogram counts and overall density differences
    out = {}
    out['densityDiff'] = (np.sum(y > 0) - np.sum(y < 0)) / n_non_zero  # measure of asymmetry about the mean
    out['modeProbPos'] = np.max(p_pos)
    out['modeProbNeg'] = np.max(p_neg)
    out['modeDiff'] = out['modeProbPos'] - out['modeProbNeg']

    # Mean position of maximums (if multiple)
    out['posMode'] = np.mean(bin_centers_pos[p_pos == out['modeProbPos']])
    out['negMode'] = np.mean(bin_centers_neg[p_neg == out['modeProbNeg']])
    out['modeAsymmetry'] = out['posMode'] + out['negMode']

    return out

def histogram_mode(y: ArrayLike, num_bins: Union[int, str] = 10, do_simple: bool = True) -> float:
    """
    Measures the mode of the data vector using histograms with a given number
    of bins.

    The mode is the center of the fullest bin of an equal-width histogram (the mean of
    the centers if several bins are equally full). The bin edges are given explicitly
    (:func:`pyhctsa.robust.bf_hist_edges`), so that the result does not depend on how a
    histogram routine rounds its bin limits and width.

    Parameters
    -----------
    y : array-like
        The input time series.
    num_bins : int or str, optional
        The number of bins to use in the histogram, or the name of a rule for the
        number of bins (``'auto'``, ``'fd'``, ``'sqrt'`` or ``'sturges'``; see
        :func:`pyhctsa.robust.bf_hist_edges`). Default is 10.
    do_simple : bool, optional
        Whether to use equal-width bins between the minimum and maximum with explicit
        edges (`True`, the default), or bins with limits and width rounded to 'nice'
        values (`False`). Ignored if ``num_bins`` is a rule name.

    Returns
    --------
    float
        The mode of the data vector using histograms with num_bins bins. 
    """
    y = np.asarray(y, dtype=float)
    if isinstance(num_bins, str) or do_simple:
        bin_edges = bf_hist_edges(y, num_bins)
        N, _ = np.histogram(y, bins=bin_edges)
    else:
        bin_edges = bin_picker(y.min(), y.max(), num_bins)
        N = histc(y, bin_edges)[:-1]
    # compute bin centers from bin edges
    bin_centres = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)

    # mean position of maximums (if multiple)
    out = np.mean(bin_centres[N == np.max(N)])

    return float(out)

def remove_points(y: ArrayLike, remove_how: str = 'absfar', p: float = 0.1,
                  remove_or_saturate: str = 'remove', random_seed: Union[int, None] = None) -> dict:
    """
    How time-series properties change as points are removed.

    Removes a proportion, p, of points from the time series according to a specified rule,
    and computes a set of statistics before and after the change.

    Parameters
    ----------
    y : array-like
        The input time series.
    remove_how : {'absclose', 'absfar' (default), 'min', 'max', 'random'}, optional
        How to remove points from the time series:

        - 'absclose': those that are the closest to the mean,
        - 'absfar': those that are the furthest from the mean (default),
        - 'min': the lowest values,
        - 'max': the highest values,
        - 'random': at random.

        Default is ``'absfar'``.

    p : float, optional
        The proportion of points to remove. Default is 0.1.
    remove_or_saturate : {'remove', 'saturate'}, optional
        Whether to remove points ('remove') or saturate their values ('saturate').
        Default is ``'remove'``.
    random_seed : int, optional
        Seed for the random ordering used when ``remove_how='random'``, for
        reproducibility. Default is ``None`` (unseeded).

    Returns
    -------
    dict
        Statistics including the change in autocorrelation, time scales, mean, median,
        standard deviation, skewness (``skewnessdiff``, the difference
        skew(y_transform) - skew(y)), and kurtosis (``kurtosisrat``, the ratio
        kurtosis(y_transform) / kurtosis(y)).
    """
    y = np.asarray(y)
    N = len(y)

    is_ = None
    if remove_how == 'absclose':
        is_ = np.argsort(-np.abs(y), kind='stable')   # descending abs, ties stable
    elif remove_how == 'absfar':
        is_ = np.argsort(np.abs(y), kind='stable')     # ascending abs
    elif remove_how == 'min':
        is_ = np.argsort(-y, kind='stable')            # descending y
    elif remove_how == 'max':
        is_ = np.argsort(y, kind='stable')             # ascending y
    elif remove_how == 'random':
        is_ = np.random.default_rng(random_seed).permutation(N)
    else:
        raise ValueError(f"Unknown method '{remove_how}'")
    
    # Indices of points to *keep*:
    # (MATLAB's round: halves go away from zero, unlike Python's round)
    n_keep = N * (1 - p)
    n_keep = int(np.floor(n_keep)) + int(n_keep - np.floor(n_keep) >= 0.5)
    r_keep = np.sort(is_[:n_keep])

    # Indices of points to *transform*:
    r_transform = np.setdiff1d(np.arange(N), r_keep)

    # Do the removing/saturating to convert y -> y_transform
    if remove_or_saturate == 'remove':
        y_transform = y[r_keep]
    elif remove_or_saturate == 'saturate':
        # Saturate out the targeted points
        if remove_how == 'max':
            y_transform = y.copy()
            y_transform[r_transform] = np.max(y[r_keep])
        elif remove_how == 'min':
            y_transform = y.copy()
            y_transform[r_transform] = np.min(y[r_keep])
        elif remove_how == 'absfar':
            y_transform = y.copy()
            y_transform[y_transform > np.max(y[r_keep])] = np.max(y[r_keep])
            y_transform[y_transform < np.min(y[r_keep])] = np.min(y[r_keep])
        else:
            raise ValueError(f"Cannot 'saturate' when using '{remove_how}' method")
    else:
        raise ValueError(f"Unknown removOrSaturate option '{remove_or_saturate}'")
    
    # Compute some autocorrelation properties
    n = 8
    acf_y = autocorr(y, list(range(1, n+1)), 'Fourier')
    acf_y_transform = autocorr(y_transform, list(range(1, n+1)), 'Fourier')
    # Compute output statistics
    out = {}

    # Helper functions
    f_abs_diff = lambda x1, x2: np.abs(x1 - x2) # ignores the sign
    f_ratio = lambda x1, x2: np.divide(x1, x2) # includes the sign

    out['fzcacrat'] = f_ratio(first_crossing(y_transform, 'ac', 0, 'continuous'), 
                              first_crossing(y, 'ac', 0, 'continuous'))
    
    out['ac1rat'] = f_ratio(acf_y_transform[0], acf_y[0])
    out['ac1diff'] = f_abs_diff(acf_y_transform[0], acf_y[0])

    out['ac2rat'] = f_ratio(acf_y_transform[1], acf_y[1])
    out['ac2diff'] = f_abs_diff(acf_y_transform[1], acf_y[1])
    
    out['ac3rat'] = f_ratio(acf_y_transform[2], acf_y[2])
    out['ac3diff'] = f_abs_diff(acf_y_transform[2], acf_y[2])
    
    out['sumabsacfdiff'] = np.sum(np.abs(acf_y_transform - acf_y))
    out['mean'] = np.mean(y_transform)
    out['median'] = np.median(y_transform)
    out['std'] = np.std(y_transform, ddof=1)
    
    # difference rather than ratio: a ratio blows up (and changes sign) when skew(y) is near 0
    out['skewnessdiff'] = stats.skew(y_transform) - stats.skew(y)
    # return kurtosis instead of excess kurtosis
    out['kurtosisrat'] = stats.kurtosis(y_transform, fisher=False) / stats.kurtosis(y, fisher=False)

    return out


def _matlab_round(v: float) -> int:
    """MATLAB's ``round`` (halves away from zero)."""
    return int(np.sign(v) * np.floor(abs(v) + 0.5))


def _hill_estimate(s: np.ndarray, k: int) -> float:
    """Hill estimator from the k largest of the descending-sorted positive values ``s``."""
    if len(s) <= k or s[k] <= 0 or s[k - 1] == s[k]:
        return np.nan  # too few values, or a tie at the threshold
    return float(np.mean(np.log(s[:k])) - np.log(s[k]))


def _moment_estimate(s: np.ndarray, k: int) -> float:
    """Dekkers-Einmahl-de Haan moment estimator of the tail index (as ``_hill_estimate``)."""
    if len(s) <= k or s[k] <= 0 or s[k - 1] == s[k]:
        return np.nan
    log_excess = np.log(s[:k]) - np.log(s[k])
    m1 = np.mean(log_excess)
    m2 = np.mean(log_excess ** 2)
    if m2 <= 0:
        return np.nan
    with np.errstate(all='ignore'):
        xi = m1 + 1 - 0.5 / (1 - m1 ** 2 / m2)
    return float(xi) if np.isfinite(xi) else np.nan


def _gpd_shape(exceed: np.ndarray) -> float:
    """Shape parameter of a generalized Pareto distribution (threshold 0), by probability-weighted moments.

    Hosking and Wallis (1987): with ``a0`` the mean of the exceedances and ``a1`` the
    mean of ``(1 - p_i) z_i`` for the ascending exceedances ``z_i`` with plotting
    positions ``p_i = (i - 0.35)/n``, the shape is ``2 - a0/(a0 - 2 a1)``.
    """
    if np.any(exceed <= 0):
        return np.nan  # ties between the tail values and the threshold
    n = len(exceed)
    z = np.sort(exceed)
    a0 = np.mean(z)
    a1 = np.mean((1 - (np.arange(1, n + 1) - 0.35) / n) * z)
    with np.errstate(all='ignore'):
        xi = 2 - a0 / (a0 - 2 * a1)
    return float(xi) if np.isfinite(xi) else np.nan


def tail_index(y: ArrayLike, tail_frac: float = 0.05) -> dict:
    """
    Tail index of the distribution of values: how heavy its tails are.

    The tail index is estimated in several ways from the ``k = round(tail_frac * N)``
    most extreme values in each tail (taken relative to the median of the data):
    Hill's estimator, a moment estimator, and the shape parameter of a generalized
    Pareto distribution fitted to the exceedances over the (k+1)-th most extreme value
    (by probability-weighted moments, Hosking and Wallis 1987, which has a closed form).
    A larger index means a heavier tail (a power-law tail of exponent alpha has index
    1/alpha; a Gaussian has an index of about zero). NaN is returned for every
    output if there are fewer than 10 tail values or if k is at least N/2.

    Parameters
    ----------
    y : array-like
        The input time series.
    tail_frac : float, optional
        The fraction of the data in each tail used to estimate the index
        (default 0.05).

    Returns
    -------
    dict
        hillUpper, hillLower: Hill estimator for the upper and lower tail (distances
        above and below the median); hillAsym: their difference; momentAbs: the moment
        estimator for the distances from the median; gpdUpper, gpdLower: the generalized
        Pareto shape parameter for the upper and lower exceedances; gpdAsym: their
        difference.
    """
    names = ['hillUpper', 'hillLower', 'hillAsym', 'momentAbs', 'gpdUpper', 'gpdLower', 'gpdAsym']
    out = dict.fromkeys(names, np.nan)

    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = len(y)
    k = _matlab_round(tail_frac * n)  # number of values in each tail
    if k < 10 or k >= n // 2:
        return out  # too few tail values to estimate a tail index

    dev = y - np.median(y)
    tail_up = np.sort(dev[dev > 0])[::-1]  # distances above the median
    tail_lo = np.sort(-dev[dev < 0])[::-1]  # distances below the median
    tail_abs = np.sort(np.abs(dev))[::-1]  # distances from the median

    out['hillUpper'] = _hill_estimate(tail_up, k)
    out['hillLower'] = _hill_estimate(tail_lo, k)
    out['hillAsym'] = out['hillUpper'] - out['hillLower']
    out['momentAbs'] = _moment_estimate(tail_abs, k)

    ys = np.sort(y)[::-1]
    out['gpdUpper'] = _gpd_shape(ys[:k] - ys[k])  # upper exceedances
    ys = np.sort(y)
    out['gpdLower'] = _gpd_shape(ys[k] - ys[:k])  # lower exceedances
    out['gpdAsym'] = out['gpdUpper'] - out['gpdLower']
    return out


def _fmt_threshold(prefix: str, thr: float) -> str:
    """Output name hctsa gives a threshold: sprintf('%s_%.2f') with the dot removed."""
    return f"{prefix}_{thr:.2f}".replace('.', '')


def fit_kernel_smooth(x: ArrayLike, area: Union[None, float, list] = None,
                      numcross: Union[None, float, list] = None,
                      arclength: Union[None, float, list] = None) -> dict:
    """
    Statistics of a kernel-smoothed distribution of the data.

    The data are smoothed with a Gaussian kernel with a normal-reference bandwidth
    (:func:`pyhctsa.robust.bf_ks_density`: 100 points from three bandwidths below the
    minimum to three above the maximum) and the smoothed density, f, is summarized by its number of peaks,
    maximum, entropy, and asymmetry about the mean, plus optional threshold-based
    statistics.

    Parameters
    ----------
    x : array-like
        The input data vector.
    area : float or list of float, optional
        Thresholds for which to compute the integral of f over the region where
        f is below the threshold.
    numcross : float or list of float, optional
        Thresholds for which to count the crossings of f through the threshold.
    arclength : float or list of float, optional
        Half-widths of a window around the mean; the arc length (total absolute change
        in f) within the window is computed.

    Returns
    -------
    dict
        npeaks: the number of 'large enough' maxima of f (those with second difference
        below -0.0002); max: the maximum of f; entropy: the entropy of f; asym: the
        area of f above the mean divided by that below it (NaN if there is essentially
        none below); plsym: the ratio of the total variation of f below the mean to that
        above it (NaN if there is none above); and, for each threshold t,
        ``numcross_t``, ``area_t``, ``arclength_t`` (t formatted to two decimals with the
        dot removed, e.g. ``numcross_005`` for 0.05). NaN (not a dict) for a constant
        series, which has no scale to smooth over.
    """
    x = np.asarray(x, dtype=float).ravel()
    for name, v in (('area', area), ('numcross', numcross), ('arclength', arclength)):
        if v is not None and np.any(np.asarray(v) <= 0):
            raise ValueError(f"'{name}' thresholds must be positive.")
    area, numcross, arclength = (None if v is None else np.atleast_1d(np.asarray(v, dtype=float))
                                 for v in (area, numcross, arclength))

    m = np.mean(x)
    f, xi, _ = bf_ks_density(x)
    if np.any(np.isnan(f)):  # constant data: no scale to smooth over
        return np.nan
    dx = xi[1] - xi[0]
    out = {}

    df = np.diff(f)
    ddf = np.diff(df)
    sdsp = ddf[sign_change(df, 1)]
    out['npeaks'] = int(np.sum(sdsp < -0.0002))  # 'large enough' maxima
    out['max'] = float(np.max(f))  # maximum of the distribution
    fp = f[f > 0]
    out['entropy'] = float(-np.sum(fp * np.log(fp) * dx))
    mass_above = np.sum(f[xi > m] * dx)
    mass_below = np.sum(f[xi < m] * dx)
    # (essentially) no mass below the mean: the ratio is not meaningful
    out['asym'] = np.nan if mass_below < 1e-10 else float(mass_above / mass_below)
    var_below = np.sum(np.abs(np.diff(f[xi < m])) * dx)
    var_above = np.sum(np.abs(np.diff(f[xi > m])) * dx)
    out['plsym'] = np.nan if var_above < 1e-10 else float(var_below / var_above)  # no variation above the mean

    if numcross is not None:  # crossing statistics
        for thr in numcross:
            out[_fmt_threshold('numcross', thr)] = int(np.sum(sign_change(f - thr)))
    if area is not None:  # area statistics
        for thr in area:
            out[_fmt_threshold('area', thr)] = float(np.sum(f[f < thr] * dx))
    if arclength is not None:  # arc length statistics
        for thr in arclength:
            fd = np.abs(np.diff(f[(xi > m - thr) & (xi < m + thr)]))
            out[_fmt_threshold('arclength', thr)] = float(np.sum(fd * dx))
    return out


_SIMPLE_FIT_MODELS = {'gauss1': ('gauss', 3), 'gauss2': ('gauss2', 6), 'exp1': ('exp', 2), 'power1': ('power', 2)}


def simple_fit(x: ArrayLike, dmodel: str, num_bins: Union[int, str] = 'sqrt') -> Union[dict, float]:
    """
    Fits a simple curve to the distribution of the values.

    Fits a simple parametric curve to an estimate of the distribution of values in the
    time series, ignoring their temporal ordering. The distribution is estimated either
    as a histogram, with a specified number of bins, or as a kernel-smoothed density.
    The outputs measure the goodness of fit, and test the residuals (in order of
    increasing value) for remaining structure. The curve is fitted by least squares to
    the density in a deterministic way, with no random starts and no optimizer defaults
    (:func:`pyhctsa.robust.bf_fit_density_curve`).

    Parameters
    ----------
    x : array-like
        The input data vector.
    dmodel : {'gauss1', 'gauss2', 'exp1', 'power1'}
        The curve to fit: a Gaussian, the sum of two Gaussians, an exponential
        ``a*exp(b*x)``, or a power law ``a * x ** b`` (cannot be fit if any bin center is
        not positive; NaN is returned). NaN is also returned if there are no more bins
        than parameters of the model. (hctsa's time-series models, the sinusoids and
        Fourier series, are fitted by ``sinusoid_fit`` in the spectral module; the
        names 'sin1', 'sin2', 'sin3' are passed on to it, as hctsa does.)
    num_bins : int or str, optional
        How to estimate the distribution: the name of a rule for the number of histogram bins
        (default ``'sqrt'``), the number of histogram bins, or 0 for a kernel-smoothed density.

    Returns
    -------
    dict
        r2: R-squared of the fit; adjr2: R-squared adjusted for the number of
        coefficients; rmse: root-mean-square error of the fit, in units of probability
        density of the standardized series (multiplied by the standard deviation of the
        data); resAC1, resAC2: autocorrelations of the residuals, in order of increasing
        value, at lags 1 and 2; resrunsz: the signed z-statistic of a runs test on the
        residuals (:func:`pyhctsa.robust.bf_runs_z`; negative when the residuals have
        fewer runs about their median than expected for a random order). NaN (not a
        dict) if the model cannot be fitted, or for a constant series with ``num_bins = 0``.

    Notes
    -----
    The histogram has equal-width bins spanning the data with explicit edges
    (:func:`pyhctsa.robust.bf_hist_edges`; a number of bins or a rule: ``'auto'``, ``'fd'``,
    ``'sqrt'``, ``'sturges'``), and the kernel-smoothed density is
    :func:`pyhctsa.robust.bf_ks_density`, so neither depends on the rounding of a histogram or
    kernel-density routine's defaults.
    """
    x = np.asarray(x, dtype=float).ravel()
    if dmodel in ('sin1', 'sin2', 'sin3', 'fourier1', 'fourier2', 'fourier3'):
        from .spectral import sinusoid_fit
        return sinusoid_fit(x, dmodel)
    if dmodel not in _SIMPLE_FIT_MODELS:
        raise ValueError(f"Invalid distribution model '{dmodel}' specified")

    if isinstance(num_bins, str) or num_bins != 0:
        # histogram with an explicit number of equal-width bins (a number, or a rule)
        edges = bf_hist_edges(x, num_bins)
        counts, _ = np.histogram(x, bins=edges)
        dnx = (edges[:-1] + edges[1:]) / 2
        dny = counts / (np.sum(counts) * np.mean(np.diff(edges)))  # counts -> probability density
    else:  # kernel-smoothed density instead of a histogram
        dny, dnx, _ = bf_ks_density(x)
        if np.any(np.isnan(dny)):  # constant series: no distribution to fit
            return np.nan

    if dmodel == 'power1' and np.any(dnx <= 0):
        logger.warning(f"The model '{dmodel}' can not be applied to non-positive data")
        return np.nan

    curve, num_params = _SIMPLE_FIT_MODELS[dmodel]
    dfe = len(dny) - num_params  # degrees of freedom of the error
    if dfe < 1:  # no more bins than parameters: the fit is not meaningful
        return np.nan

    # Fit the model by least squares for the density
    dny_fit = bf_fit_density_curve(dnx, dny, curve)

    # Residuals (in order of increasing value) and goodness of fit as in the Curve Fitting
    # Toolbox: R^2, R^2 adjusted for the number of fitted parameters, and the root-mean-square
    # error from the residual sum of squares divided by the degrees of freedom of the error
    res = dny - dny_fit
    sse = np.sum(res ** 2)
    sstot = np.sum((dny - np.mean(dny)) ** 2)
    with np.errstate(all='ignore'):
        r2 = 1 - sse / sstot
        adjr2 = 1 - (1 - r2) * (len(dny) - 1) / dfe
    rmse = np.sqrt(sse / dfe)

    # Remaining structure in the residuals, in order of increasing value:
    res_ac1, res_ac2, res_runsz = bf_residual_stats(res, sstot)
    return {
        'r2': float(r2),
        'adjr2': float(adjr2),
        'rmse': float(rmse * np.std(x, ddof=1)),
        'resAC1': float(res_ac1),
        'resAC2': float(res_ac2),
        'resrunsz': float(res_runsz),
    }
