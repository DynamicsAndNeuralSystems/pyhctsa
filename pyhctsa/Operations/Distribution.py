import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
import logging

from scipy import stats
from scipy.stats import norm, gumbel_l, uniform, expon, lognorm, gaussian_kde

from ..utils import histc, bin_picker, simple_binner, x_corr
from ..operations.correlation import autocorr, first_crossing

def compare_ks_fit(x: ArrayLike, what_distn: str) -> dict:
    """
    Fits a distribution to data.

    Returns simple statistics on the discrepancy between the kernel-smoothed distribution
    of the time-series values and the distribution fitted to it by some model:
    Gaussian, Extreme Value, Uniform, Exponential, and LogNormal.

    Parameters
    ----------
    x : array-like
        The input data vector.
    what_distn : str
        The type of distribution to fit to the data:
            - 'norm': normal
            - 'ev': extreme value
            - 'uni': uniform
            - 'exp': exponential
            - 'logn': Log-Normal

    Returns
    -------
    dict
        Includes the absolute area between the two distributions, the peak separation,
        overlap integral, and relative entropy.
    """
    
    x = np.asarray(x)
    x_step = np.std(x, ddof=1) / 100  # set a step size

    # ----------------------------
    # Fit distribution & thresholds
    # ----------------------------
    if what_distn == 'norm':
        # Fit a normal distribution
        a, b = norm.fit(x)
        peaky = norm.pdf(a, loc=a, scale=b)
        thresh = peaky / 100.0  # stop when gets to 1/100 of peak value
        pdf_func = lambda z: norm.pdf(z, loc=a, scale=b)
        xf = _find_bounds(pdf_func, np.mean(x), np.mean(x), x_step, thresh)
        ffit_func = lambda xi: norm.pdf(xi, loc=a, scale=b)

    elif what_distn == 'ev':
        # Fit an extreme value (Gumbel, left) distribution
        loc, scale = gumbel_l.fit(x)
        peaky = gumbel_l.pdf(loc, loc=loc, scale=scale)
        thresh = peaky / 100.0
        pdf_func = lambda z: gumbel_l.pdf(z, loc=loc, scale=scale)
        xf = _find_bounds(pdf_func, 0.0, 0.0, x_step, thresh)
        ffit_func = lambda xi: gumbel_l.pdf(xi, loc=loc, scale=scale)

    elif what_distn == 'uni':
        # Fit a uniform distribution
        loc, scale = uniform.fit(x)
        a, b = loc, loc + scale
        # Peak of uniform PDF = 1 / (b - a)
        peaky = uniform.pdf(np.mean(x), loc=a, scale=(b - a))
        thresh = peaky / 100.0
        pdf_func = lambda z: uniform.pdf(z, loc=a, scale=(b - a))
        xf = _find_bounds(pdf_func, 0.0, 0.0, x_step, thresh)
        ffit_func = lambda xi: uniform.pdf(xi, loc=a, scale=(b - a))

    elif what_distn == 'exp':
        # Check positivity
        if np.any(x < 0):
            print("The data contains negative values, but Exponential is a positive-only distribution.")
            return np.nan
        # Check constant
        if np.all(x == x[0]):
            print("Data are a constant")
            return np.nan
        # Fit Exponential distribution (equivalent to expfit in MATLAB)
        _, lam = expon.fit(x, floc=0)  # force support at 0
        # Peak of PDF occurs at 0
        peaky = expon.pdf(0, loc=0, scale=lam)
        thresh = peaky / 100.0
        pdf_func = lambda z: expon.pdf(z, loc=0, scale=lam)
        xf = _find_bounds(pdf_func, 0.0, 0.0, x_step, thresh)
        ffit_func = lambda xi: expon.pdf(xi, loc=0, scale=lam)

    elif what_distn == 'logn':
        # Check positivity
        if np.any(x <= 0):
            print("The data are not positive, but Log-Normal is a positive-only distribution.")
            return np.nan
        # Fit log-normal distribution
        shape, loc, scale = lognorm.fit(x, floc=0)  # sigma, 0, exp(mu)
        sigma = shape
        mu = np.log(scale)
        # Mode of log-normal
        mode = np.exp(mu - sigma**2)
        # Peak PDF value at the mode
        peaky = lognorm.pdf(mode, s=sigma, loc=0, scale=np.exp(mu))
        thresh = peaky / 100.0
        pdf_func = lambda z: lognorm.pdf(z, s=sigma, loc=0, scale=np.exp(mu))
        xf = _find_bounds(pdf_func, 0.0, mode, x_step, thresh)
        ffit_func = lambda xi: lognorm.pdf(xi, s=sigma, loc=0, scale=np.exp(mu))

    else:
        raise ValueError(f"Unknown distribution:  {what_distn}.")

    # ----------------------------
    # Estimate smoothed empirical distribution
    # ----------------------------
    xi = np.linspace(np.min(x), np.max(x), 100)
    # Calculate the Kernel Density Estimate (KDE) for the first angle distribution.
    kde = gaussian_kde(x)
    f = kde(xi)
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
    f = kde(xi)
    ffit = ffit_func(xi)

    # ----------------------------
    # Statistics
    # ----------------------------
    dx = xi[1] - xi[0]
    out = {}
    # ADIFF: returns absolute area between the curves
    out['adiff'] = np.sum(np.abs(f - ffit) * dx)
    # PEAKSEPY: separation (in y) between the maxima of each distribution
    out['peaksepy'] = np.max(ffit) - np.max(f)
    # PEAKSEPX: separation (in x) between the maxima of each distribution
    i1 = np.argmax(f)
    i2 = np.argmax(ffit)
    out['peaksepx'] = xi[i2] - xi[i1]
    # OLAPINT: overlap integral between the two curves; normalized by variance
    out['olapint'] = np.sum(f * ffit * dx) * np.std(x, ddof=1)
    # RELENT: relative entropy of the two distributions
    r = ffit > 0
    out['relent'] = np.sum(f[r] * np.log(f[r] / ffit[r]) * dx)

    return out

def _find_bounds(pdf_func, start_left, start_right, xStep, thresh):
    """Expand left/right until pdf falls below threshold."""
    xf = [start_left, start_right]

    # Left search (only if start_left is not None)
    if start_left is not None:
        ange = 10
        while ange > thresh:
            xf[0] -= xStep
            ange = pdf_func(xf[0])

    # Right search
    ange = 10
    while ange > thresh:
        xf[1] += xStep
        ange = pdf_func(xf[1])

    return xf

def withinp(x : ArrayLike, p : float = 1.0, mean_or_median : str = 'mean') -> float:
    """
    Proportion of data points within p standard deviations of the mean or median.

    Parameters
    -----------
    x : array-like
        The input data vector
    p : float
        The number (proportion) of standard deviations
    mean_or_median : str 
        Whether to use units of 'mean' and standard deviation, or 'median' and rescaled interquartile range.

    Returns
    --------
    float: 
        The proportion of data points within p standard deviations
    """
    x = np.asarray(x)
    N = len(x)

    if mean_or_median == 'mean':
        mu = np.mean(x)
        sig = np.std(x, ddof=1)
    elif mean_or_median == 'median':
        mu = np.median(x)
        iqr_val = np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')
        sig = 1.35 * iqr_val
    else:
        raise ValueError(f"Unknown setting: '{mean_or_median}'")

    # The withinp statistic:
    return np.divide(np.sum((x >= mu - p * sig) & (x <= mu + p * sig)), N)

def unique(y : ArrayLike) -> float:
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

def spread(y : ArrayLike, spread_measure : str = 'std') -> float:
    """
    Measure of spread of the input time series.

    Returns the spread of the raw data vector using different statistical measures.

    Parameters
    ----------
    y : array-like
        The input time series or data vector.
    spread_measure : str, optional
        The spread measure to use (default is 'std'):
        - 'std': standard deviation
        - 'iqr': interquartile range 
        - 'mad': mean absolute deviation
        - 'mead': median absolute deviation

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

def quantile(y : ArrayLike, p : float = 0.5) -> float:
    """ 
    Calculates the quantile value at a specified proportion, p.

    Parameters
    ----------
    y : array-like
        The input data vector.
    p : float 
        The quantile proportion (default is 0.5, which is the median).

    Returns
    -------
    float: 
        The calculated quantile value.
    """
    y = np.asarray(y)    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in (0,1)")
    
    return float(np.quantile(y, p, method = 'hazen'))

def proportion_values(x : ArrayLike, prop_what : str = 'positive') -> float:
    """
    Calculate the proportion of values meeting specific conditions in a time series.

    Parameters
    ----------
    x : array-like
        Input time series or data vector
    prop_what : str, optional (default is 'positive')
        Type of values to count:
        - 'zeros': values equal to zero
        - 'positive': values strictly greater than zero
        - 'geq0': values greater than or equal to zero

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

def pleft(y : ArrayLike, th : float = 0.1) -> float:
    """
    Distance from the mean at which a given proportion of data are more distant.
    
    Measures the maximum distance from the mean at which a given fixed proportion, `th`, of the time-series data points are further.
    Normalizes by the standard deviation of the time series.
    
    Parameters
    ----------
    y : array-like
        The input data vector.
    th : float, optional
        The proportion of data further than `th` from the mean (default is 0.1).
    
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

def min_max(y : ArrayLike, min_or_max : str = 'max') -> float:
    """
    The maximum and minimum values of the input data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    minOrMax : str, optional
        Return either the minimum or maximum of y. Default is 'max':
        - 'min': minimum of y
        - 'max': maximum of y

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

def mean(y : ArrayLike, mean_type : str = 'arithmetic') -> float:
    """
    A given measure of location of a data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    mean_type : str, optional
        Type of mean to calculate. Default is 'arithmtic':
        - 'norm' or 'arithmetic': standard arithmetic mean
        - 'median': middle value (50th percentile)
        - 'geom': geometric mean (nth root of product)
        - 'harm': harmonic mean (reciprocal of mean of reciprocals)
        - 'rms': root mean square (quadratic mean)
        - 'iqm': interquartile mean (mean of values between Q1 and Q3)
        - 'midhinge': average of first and third quartiles

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

    Paramters
    ----------
    y : array-like
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

def fit_mle(y : ArrayLike, fit_what : str = 'gaussian') -> Union[Dict[str, float], float]:
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
        Default is 'gaussian'

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

def cv(x : ArrayLike, k : int = 1) -> float:
    """
    Calculate the coefficient of variation of order k.

    The coefficient of variation of order k is (sigma/mu)^k, where sigma is the
    standard deviation and mu is the mean of the input data vector.

    Parameters
    ----------
    x : array-like
        Input time series or data vector.
    k : int, optional
        Order of the coefficient of variation. Default is 1.

    Returns
    -------
    float
        The coefficient of variation of order k.
    """
    if not isinstance(k, int) or k < 0:
        logging.warning('k should probably be a positive integer')
        # carry on with just this warning, though
    
    # Compute the coefficient of variation (of order k) of the data
    return (np.std(x, ddof=1) ** k) / (np.mean(x) ** k)

def custom_skewness(y : ArrayLike, what_skew : str = 'pearson') -> float:
    """
    Calculate custom skewness measures of a time series.

    Computes either the Pearson or Bowley skewness. The Pearson skewness uses mean, 
    median and standard deviation, while the Bowley skewness (also known as quartile 
    skewness) uses quartiles.

    Parameters
    ----------
    y : array-like
        Input time series
    what_skew : {'pearson', 'bowley'}, optional
        The skewness measure to calculate:
        - 'pearson': (3 * mean - median) / std
        - 'bowley': (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
        Default is 'pearson'.

    Returns
    -------
    float
        The calculated skewness measure:
        - Positive values indicate right skew
        - Negative values indicate left skew
        - Zero indicates symmetry
    """
    y = np.asarray(y)
    out = 0.0
    if what_skew == 'pearson':
        out = ((3 * np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif what_skew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    
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
        'B': Original burstiness statistic
        'B_Kim': Improved burstiness for finite series
    """
    y = np.asarray(y)
    mean = np.mean(y)
    std = np.std(y, ddof=1)

    r = np.divide(std,mean) # coefficient of variation
    B = np.divide((r - 1), (r + 1)) # Original Goh and Barabasi burstiness statistic, B

    # improved burstiness statistic, accounting for scaling for finite time series
    # Kim and Jo, 2016, http://arxiv.org/pdf/1604.01125v1.pdf
    N = len(y)
    p1 = np.sqrt(N+1)*r - np.sqrt(N-1)
    p2 = (np.sqrt(N+1)-2)*r + np.sqrt(N-1)

    B_Kim = np.divide(p1, p2)

    out = {'B': B, 'B_Kim': B_Kim}

    return out

def moments(y : ArrayLike, the_mom : int = 0) -> float:
    """
    A moment of the distribution of the input time series.
    Normalizes by the standard deviation.

    Parameters
    ----------
    y : array-like
        Input time series or data vector.
    theMom: int, optional
        The moment to calculate. Default is 0.

    Returns
    -------
    float
        The calculated moment.
    """
    y = np.asarray(y)

    return stats.moment(y, the_mom) / np.std(y, ddof=1)

def outlier_include(y: ArrayLike, threshold_how: str = 'abs', inc: float = 0.01) -> dict:
    """
    How statistics depend on distributional outliers.

    Measures how various statistics of a time series change as more and more outliers are included in the calculation, according to a specified rule for defining outliers.

    At each threshold, the mean, standard error, proportion of included points, median, and standard deviation are calculated. Outputs summarize how these statistics change as more extreme points are included.

    Parameters
    ----------
    y : array-like
        The input time series.
    threshold_how : {'abs', 'pos', 'neg'}, optional
        The method for determining outliers:
            - 'abs': Outliers are furthest from the mean (default).
            - 'pos': Outliers are the greatest positive deviations from the mean.
            - 'neg': Outliers are the greatest negative deviations from the mean.
    inc : float, optional
        The increment to move through thresholds (as a fraction of the standard deviation).
        Default is 0.01.

    Returns
    -------
    dict
        Dictionary containing statistics describing how the statistics change as more outliers are included.
    """
    y = np.asarray(y)
    
    # Handle constant time series
    if np.all(y[0] == y):
        return np.nan
    
    N = len(y)
    results = {}
    
    # Initialize thresholds based on method
    if threshold_how == 'abs':
        thresholds = np.arange(0, max(abs(y)), inc)
        total_points = N
    elif threshold_how == 'pos':
        thresholds = np.arange(0, max(y), inc)
        total_points = np.sum(y >= 0)
    elif threshold_how == 'neg':
        thresholds = np.arange(0, max(-y), inc)
        total_points = np.sum(y <= 0)
    else:
        raise ValueError(f"Invalid thresholdHow: '{threshold_how}'. Must be 'abs', 'pos', or 'neg'.")
    
    if len(thresholds) == 0:
        raise ValueError("Error setting increments through the time-series values")
    
    # Initialize statistics matrix
    # Columns: [mean_diff, std_err, percentage, median_pos, mean_pos, std_pos]
    statistics = np.zeros((len(thresholds), 6))
    
    # Calculate statistics for each threshold
    for i, threshold in enumerate(thresholds):
        # Find indices exceeding threshold
        if threshold_how == 'abs':
            over_threshold_idx = np.argwhere(abs(y) >= threshold).flatten()
        elif threshold_how == 'pos':
            over_threshold_idx = np.argwhere(y >= threshold).flatten()
        elif threshold_how == 'neg':
            over_threshold_idx = np.argwhere(y <= -threshold).flatten()
            
        # Calculate differences between consecutive over-threshold events
        time_diffs = np.diff(over_threshold_idx)
        
        # Store statistics
        statistics[i, 0] = np.mean(time_diffs)  # Mean time between events
        statistics[i, 1] = np.std(time_diffs, ddof=1) / np.sqrt(len(over_threshold_idx))  # Standard error
        statistics[i, 2] = len(time_diffs) / total_points * 100  # Percentage of events
        statistics[i, 3] = (np.median(over_threshold_idx) / (N / 2)) - 1  # Median position deviation
        statistics[i, 4] = np.mean(over_threshold_idx) / (N / 2) - 1  # Mean position deviation
        statistics[i, 5] = np.std(over_threshold_idx, ddof=1) / np.sqrt(len(over_threshold_idx))  # Position std error
    
    # Trim data where statistics become invalid
    first_nan_idx = np.argmax(np.isnan(statistics[:, 0])) if np.any(np.isnan(statistics[:, 0])) else None
    if first_nan_idx and first_nan_idx > 0:
        statistics = statistics[:first_nan_idx, :]
        thresholds = thresholds[:first_nan_idx]
    
    # Further trim based on percentage threshold
    trim_threshold = 2 # percent
    valid_indices = np.argwhere(statistics[:, 2] > trim_threshold).flatten()
    if len(valid_indices) > 0:
        last_valid_idx = valid_indices[-1]
        statistics = statistics[:last_valid_idx + 1, :]
        thresholds = thresholds[:last_valid_idx + 1]
    
    # Basic statistics on mean times
    results.update({
        'mdtm': np.mean(statistics[:, 0]),
        'mdtmd': np.median(statistics[:, 0]),
        'mdtstd': np.std(statistics[:, 0], ddof=1)
    })
    
    # Statistics on median position deviations
    results.update({
        'mdrm': np.mean(statistics[:, 3]),
        'mdrmd': np.median(statistics[:, 3]),
        'mdrstd': np.std(statistics[:, 3], ddof=1)
    })
    
    # Statistics on mean position deviations
    results.update({
        'mrm': np.mean(statistics[:, 4]),
        'mrmd': np.median(statistics[:, 4]),
        'mrstd': np.std(statistics[:, 4], ddof=1)
    })
    
    # Cross-correlation between mean and error
    _, cross_corr = x_corr(statistics[:, 0], statistics[:, 1], max_lags=1)
    results.update({
        'xcmerr1': cross_corr[-1],
        'xcmerrn1': cross_corr[0]
    })
    
    return results

def outlier_test(y: ArrayLike, p: float = 2, just_me: Union[str, None] = None) -> Union[dict, float]:
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
        The percentage of values to remove beyond upper and lower percentiles.
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
    lower_bound = np.percentile(y, p, method='hazen')
    upper_bound = np.percentile(y, (100 - p), method='hazen')
    
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
    y : array-like
        The input time series or data vector
    p_exclude : float, optional
        The percentage of highest and lowest values to exclude from the mean 
        calculation (default is 0.0, which gives the standard mean)

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
        Number of bins to use in histogram calculation. Default is 10
    do_simple : bool, optional
        If True, uses linearly spaced bins. If False, uses optimized bin edges.

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
    out['densityDiff'] = np.sum(y > 0) - np.sum(y < 0)  # measure of asymmetry about the mean
    out['modeProbPos'] = np.max(p_pos)
    out['modeProbNeg'] = np.max(p_neg)
    out['modeDiff'] = out['modeProbPos'] - out['modeProbNeg']

    # Mean position of maximums (if multiple)
    out['posMode'] = np.mean(bin_centers_pos[p_pos == out['modeProbPos']])
    out['negMode'] = np.mean(bin_centers_neg[p_neg == out['modeProbNeg']])
    out['modeAsymmetry'] = out['posMode'] + out['negMode']

    return out

def histogram_mode(y : ArrayLike, num_bins : int = 10, do_simple : bool = True) -> float:
    """
    Measures the mode of the data vector using histograms with a given number
    of bins.

    Parameters
    -----------
    y : array-like
        the input data vector
    num_bins : int, optional
        the number of bins to use in the histogram
    do_simple : bool, optional
        whether to use a simple binning method (linearly spaced bins)

    Returns
    --------
    float
        The mode of the data vector using histograms with num_bins bins. 
    """
    y = np.asarray(y)
    if do_simple:
        N, bin_edges = simple_binner(y, num_bins)
    else:
        bin_edges = bin_picker(y.min(), y.max(), num_bins)
        N = histc(y, bin_edges)[:-1]
    # compute bin centers from bin edges
    bin_centres = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)

    # mean position of maximums (if multiple)
    out = np.mean(bin_centres[N == np.max(N)])

    return float(out)

def remove_points(y : ArrayLike, remove_how : str = 'absfar', p : float = 0.1, 
                  remove_or_saturate : str = 'remove') -> dict:
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
    p : float, optional
        The proportion of points to remove (default: 0.1).
    remove_or_saturate : {'remove', 'saturate'}, optional
        Whether to remove points ('remove') or saturate their values ('saturate').
        Default is 'remove'.

    Returns
    -------
    dict
        Statistics including the change in autocorrelation, time scales, mean.
    """
    y = np.asarray(y)
    N = len(y)

    is_ = None
    if remove_how == 'absclose':
        is_ = np.argsort(np.abs(y))[::-1] # descending
    elif remove_how == 'absfar':
        is_ = np.argsort(np.abs(y)) # ascending
    elif remove_how == 'min':
        is_ = np.argsort(y)[::-1]
    elif remove_how == 'max':
        is_ = np.argsort(y)
    elif remove_how == 'random':
        is_ = np.random.permutation(N)
    else:
        raise ValueError(f"Unknown method '{remove_how}'")
    
    # Indices of points to *keep*:
    rKeep = np.sort(is_[:round(N * (1 - p))])

    # Indices of points to *transform*:
    rTransform = np.setdiff1d(np.arange(N), rKeep)

    # Do the removing/saturating to convert y -> yTransform
    if remove_or_saturate == 'remove':
        yTransform = y[rKeep]
    elif remove_or_saturate == 'saturate':
        # Saturate out the targeted points
        if remove_how == 'max':
            yTransform = y.copy()
            yTransform[rTransform] = np.max(y[rKeep])
        elif remove_how == 'min':
            yTransform = y.copy()
            yTransform[rTransform] = np.min(y[rKeep])
        elif remove_how == 'absfar':
            yTransform = y.copy()
            yTransform[yTransform > np.max(y[rKeep])] = np.max(y[rKeep])
            yTransform[yTransform < np.min(y[rKeep])] = np.min(y[rKeep])
        else:
            raise ValueError(f"Cannot 'saturate' when using '{remove_how}' method")
    else:
        raise ValueError(f"Unknown removOrSaturate option '{remove_or_saturate}'")
    
    # Compute some autocorrelation properties
    n = 8
    acf_y = autocorr(y, list(range(1, n+1)), 'Fourier')
    acf_yTransform = autocorr(yTransform, list(range(1, n+1)), 'Fourier')
    # Compute output statistics
    out = {}

    # Helper functions
    f_absDiff = lambda x1, x2: np.abs(x1 - x2) # ignores the sign
    f_ratio = lambda x1, x2: np.divide(x1, x2) # includes the sign

    out['fzcacrat'] = f_ratio(first_crossing(yTransform, 'ac', 0, 'continuous'), 
                              first_crossing(y, 'ac', 0, 'continuous'))
    
    out['ac1rat'] = f_ratio(acf_yTransform[0], acf_y[0])
    out['ac1diff'] = f_absDiff(acf_yTransform[0], acf_y[0])

    out['ac2rat'] = f_ratio(acf_yTransform[1], acf_y[1])
    out['ac2diff'] = f_absDiff(acf_yTransform[1], acf_y[1])
    
    out['ac3rat'] = f_ratio(acf_yTransform[2], acf_y[2])
    out['ac3diff'] = f_absDiff(acf_yTransform[2], acf_y[2])
    
    out['sumabsacfdiff'] = np.sum(np.abs(acf_yTransform - acf_y))
    out['mean'] = np.mean(yTransform)
    out['median'] = np.median(yTransform)
    out['std'] = np.std(yTransform, ddof=1)
    
    out['skewnessrat'] = stats.skew(yTransform) / stats.skew(y)
    # return kurtosis instead of excess kurtosis
    out['kurtosisrat'] = stats.kurtosis(yTransform, fisher=False) / stats.kurtosis(y, fisher=False)

    return out
