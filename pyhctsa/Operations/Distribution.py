import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
from scipy import stats
from loguru import logger
from ..Utilities.utils import histc, binpicker, simple_binner, xcorr


def Withinp(x : ArrayLike, p : float = 1.0, meanOrMedian : str = 'mean') -> float:
    """
    Proportion of data points within p standard deviations of the mean or median.

    Parameters:
    -----------
    x (array-like): The input data vector
    p (float): The number (proportion) of standard deviations
    meanOrMedian (str): Whether to use units of 'mean' and standard deviation,
                          or 'median' and rescaled interquartile range

    Returns:
    --------
    float: The proportion of data points within p standard deviations

    Raises:
    ValueError: If mean_or_median is not 'mean' or 'median'
    """
    x = np.asarray(x)
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x, ddof=1)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        iqr_val = np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')
        sig = 1.35 * iqr_val
    else:
        raise ValueError(f"Unknown setting: '{meanOrMedian}'")

    # The withinp statistic:
    return np.divide(np.sum((x >= mu - p * sig) & (x <= mu + p * sig)), N)

def Unique(y : ArrayLike) -> float:
    """
    The proportion of the time series that are unique values.

    Parameters
    ----------
    y : array-like
        The input time series or data vector

    Returns
    -------
    float
        the proportion of time series that are unique values
    """
    y = np.asarray(y)
    return np.divide(len(np.unique(y)), len(y))


def Spread(y : ArrayLike, spreadMeasure : str = 'std') -> float:
    """
    Measure of spread of the input time series.

    Returns the spread of the raw data vector using different statistical measures.
    This is part of the Distributional operations from hctsa, implementing DN_Spread.

    Parameters
    ----------
    y : array-like
        The input time series or data vector
    spreadMeasure : str, optional
        The spread measure to use (default is 'std'):
        - 'std': standard deviation
        - 'iqr': interquartile range 
        - 'mad': mean absolute deviation
        - 'mead': median absolute deviation

    Returns
    -------
    float
        The calculated spread measure
    """
    y = np.asarray(y)
    if spreadMeasure == 'std':
        out = np.std(y, ddof=1)
    elif spreadMeasure == 'iqr':
        q75 = np.quantile(y, 0.75, method='hazen')
        q25 = np.quantile(y, 0.25, method='hazen')
        out = q75 - q25
    elif spreadMeasure == 'mad':
        # mean absolute deviation
        out = np.mean(np.absolute(y - np.mean(y, None)), None)
    elif spreadMeasure == 'mead':
        # median absolute deviation
        out = np.median(np.absolute(y - np.median(y, None)), None)
    else:
        raise ValueError('spreadMeasure must be one of std, iqr, mad or mead')
    return out

def Quantile(y : ArrayLike, p : float = 0.5) -> float:
    """ 
    Calculates the quantile value at a specified proportion, p.

    Parameters:
    y (array-like): The input data vector
    p (float): The quantile proportion (default is 0.5, which is the median)

    Returns:
    float: The calculated quantile value

    Raises:
    ValueError: If p is not a number between 0 and 1
    """
    y = np.asarray(y)
    if p == 0.5:
        logger.warning("Using quantile p = 0.5 (median) by default")
    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in (0,1)")
    
    return float(np.quantile(y, p, method = 'hazen'))

def ProportionValues(x : ArrayLike, propWhat : str = 'positive') -> float:
    """
    Calculate the proportion of values meeting specific conditions in a time series.

    Parameters
    ----------
    x : array-like
        Input time series or data vector
    propWhat : str, optional (default is 'positive')
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

    if propWhat == 'zeros':
        # returns the proportion of zeros in the input vector
        out = sum(x == 0) / N
    elif propWhat == 'positive':
        out = sum(x > 0) / N
    elif propWhat == 'geq0':
        out = sum(x >= 0) / N
    else:
        raise ValueError(f"Unknown condition to measure: {propWhat}")

    return out


def PLeft(y : ArrayLike, th : float = 0.1) -> float:
    """
    Distance from the mean at which a given proportion of data are more distant.
    
    Measures the maximum distance from the mean at which a given fixed proportion, `th`, of the time-series data points are further.
    Normalizes by the standard deviation of the time series.
    
    Parameters
    ----------
    y : array_like
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

def MinMax(y : ArrayLike, minOrMax : str = 'max') -> float:
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
    if minOrMax == 'max':
        out = max(y)
    elif minOrMax == 'min':
        out = min(y)
    else:
        raise ValueError(f"Unknown method '{minOrMax}'")
    
    return out

def Mean(y : ArrayLike, meanType : str = 'arithmetic') -> float:
    """
    A given measure of location of a data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    meanType : str, optional
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

    if meanType in ['norm', 'arithmetic']:
        out = np.mean(y)
    elif meanType == 'median': # median
        out = np.median(y)
    elif meanType == 'geom': # geometric mean
        out = stats.gmean(y)
    elif meanType == 'harm': # harmonic mean
        out = N/sum(y**(-1))
    elif meanType == 'rms':
        out = np.sqrt(np.mean(y**2))
    elif meanType == 'iqm': # interquartile mean
        p = np.percentile(y, [25, 75], method='hazen')
        out = np.mean(y[(y >= p[0]) & (y <= p[1])])
    elif meanType == 'midhinge':  # average of 1st and third quartiles
        p = np.percentile(y, [25, 75], method='hazen')
        out = np.mean(p)
    else:
        raise ValueError(f"Unknown mean type '{meanType}'")

    return float(out)

def HighLowMu(y: ArrayLike) -> float:
    """
    The highlowmu statistic.

    The highlowmu statistic is the ratio of the mean of the data that is above the
    (global) mean compared to the mean of the data that is below the global mean.

    Paramters
    ----------
    y (array-like): The input data vector

    Returns
    --------
    float
        The highlowmu statistic.
    """
    y = np.asarray(y)
    mu = np.mean(y) # mean of data
    mhi = np.mean(y[y > mu]) # mean of data above the mean
    mlo = np.mean(y[y < mu]) # mean of data below the mean
    out = np.divide((mhi-mu), (mu-mlo)) # ratio of the differences

    return out


def FitMLE(y : ArrayLike, fitWhat : str = 'gaussian') -> Union[Dict[str, float], float]:
    """
    Maximum likelihood distribution fit to data.

    Fits a specified probability distribution to the data using maximum likelihood 
    estimation (MLE) and returns the fitted parameters.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    fitWhat : {'gaussian', 'uniform', 'geometric'}, optional
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
    if fitWhat == 'gaussian':
        loc, scale = stats.norm.fit(y, method="MLE")
        out['mean'] = loc
        out['std'] = scale
    elif fitWhat == 'uniform':
        loc, scale = stats.uniform.fit(y, method="MLE")
        out['a'] = loc
        out['b'] = loc + scale 
    elif fitWhat == 'geometric':
        sampMean = np.mean(y)
        p = 1/(1+sampMean)
        return p
    else:
        raise ValueError(f"Invalid fit specifier, {fitWhat}")

    return out

def CV(x : ArrayLike, k : int = 1) -> float:
    """
    Calculate the coefficient of variation of order k.

    The coefficient of variation of order k is (sigma/mu)^k, where sigma is the
    standard deviation and mu is the mean of the input data vector.

    Parameters
    ----------
    x : array-like
        Input time series or data vector
    k : int, optional
        Order of the coefficient of variation. Default is 1.

    Returns
    -------
    float
        The coefficient of variation of order k.
    """
    if not isinstance(k, int) or k < 0:
        logger.warn('k should probably be a positive integer')
        # carry on with just this warning, though
    
    # Compute the coefficient of variation (of order k) of the data
    return (np.std(x, ddof=1) ** k) / (np.mean(x) ** k)

def CustomSkewness(y : ArrayLike, whatSkew : str = 'pearson') -> float:
    """
    Calculate custom skewness measures of a time series.

    Computes either the Pearson or Bowley skewness. The Pearson skewness uses mean, 
    median and standard deviation, while the Bowley skewness (also known as quartile 
    skewness) uses quartiles.

    Parameters
    ----------
    y : array-like
        Input time series
    whatSkew : {'pearson', 'bowley'}, optional
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
    if whatSkew == 'pearson':
        out = ((3 * np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif whatSkew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    
    return float(out)

def Burstiness(y: ArrayLike) -> Dict[str, float]:
    """
    Calculate burstiness statistics of a time series.
    
    Implements both the original Goh & Barabasi burstiness and
    the improved Kim & Jo version for finite time series.
    
    Parameters
    ----------
    y : array-like
        Input time series
    
    Returns
    -------
    dict:
        'B': Original burstiness statistic
        'B_Kim': Improved burstiness for finite series
    
    References
    ----------
    - Goh & Barabasi (2008). Europhys. Lett. 81, 48002
    - Kim & Jo (2016). http://arxiv.org/pdf/1604.01125v1.pdf
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

def Moments(y : ArrayLike, theMom : int = 0) -> float:
    """
    A moment of the distribution of the input time series.
    Normalizes by the standard deviation.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    theMom: int, optional
        The moment to calculate. Default is 0.

    Returns
    -------
    float
        The calculated moment.
    """
    y = np.asarray(y)
    return stats.moment(y, theMom) / np.std(y, ddof=1)

def OutlierInclude(y: ArrayLike, thresholdHow: str = 'abs', inc: float = 0.01) -> dict:
    """
    How statistics depend on distributional outliers.

    Measures how various statistics of a time series change as more and more outliers are included in the calculation, according to a specified rule for defining outliers.

    At each threshold, the mean, standard error, proportion of included points, median, and standard deviation are calculated. Outputs summarize how these statistics change as more extreme points are included.

    Parameters
    ----------
    y : array-like
        The input time series.
    thresholdHow : {'abs', 'pos', 'neg'}, optional
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
    if thresholdHow == 'abs':
        thresholds = np.arange(0, max(abs(y)), inc)
        total_points = N
    elif thresholdHow == 'pos':
        thresholds = np.arange(0, max(y), inc)
        total_points = np.sum(y >= 0)
    elif thresholdHow == 'neg':
        thresholds = np.arange(0, max(-y), inc)
        total_points = np.sum(y <= 0)
    else:
        raise ValueError(f"Invalid thresholdHow: '{thresholdHow}'. Must be 'abs', 'pos', or 'neg'.")
    
    if len(thresholds) == 0:
        raise ValueError("Error setting increments through the time-series values")
    
    # Initialize statistics matrix
    # Columns: [mean_diff, std_err, percentage, median_pos, mean_pos, std_pos]
    statistics = np.zeros((len(thresholds), 6))
    
    # Calculate statistics for each threshold
    for i, threshold in enumerate(thresholds):
        # Find indices exceeding threshold
        if thresholdHow == 'abs':
            over_threshold_idx = np.argwhere(abs(y) >= threshold).flatten()
        elif thresholdHow == 'pos':
            over_threshold_idx = np.argwhere(y >= threshold).flatten()
        elif thresholdHow == 'neg':
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
    _, cross_corr = xcorr(statistics[:, 0], statistics[:, 1], maxlags=1)
    results.update({
        'xcmerr1': cross_corr[-1],
        'xcmerrn1': cross_corr[0]
    })
    
    return results


def OutlierTest(y: ArrayLike, p: float = 2, justMe: Union[str, None] = None) -> Union[Dict[str, float], float]:
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
        The percentage (0 < p < 50) of values to remove from both the upper and lower ends of the distribution.
    justMe : {'mean', 'std'}, optional
        If specified, returns only the mean or standard deviation of the middle portion of the data after trimming:
            - 'mean': returns the mean of the trimmed data.
            - 'std': returns the standard deviation of the trimmed data.
        If None (default), returns a dictionary with the ratio of mean and std before and after trimming.

    Returns
    -------
    float or dict
        If justMe is specified, returns the mean or std of the trimmed data.
        Otherwise, returns a dictionary with the ratios:
            - 'mean_ratio': mean(trimmed) / mean(original)
            - 'std_ratio': std(trimmed) / std(original)
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

    if justMe == 'mean':
        return out['mean']
    elif justMe == 'std':
        return out['std']     
    
    return out

def TrimmedMean(x: ArrayLike, pExclude: float = 0.0) -> float:
    """
    Mean of the trimmed time series.

    Returns the mean of the time series after removing a specified percentage of 
    the highest and lowest values.

    Parameters
    ----------
    y : array-like
        The input time series or data vector
    pExclude : float, optional
        The percentage of highest and lowest values to exclude from the mean 
        calculation (default is 0.0, which gives the standard mean)

    Returns
    -------
    float
        The mean of the trimmed time series.
    """
    if not 0 <= pExclude < 100:
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
    k = non_nan_count * (pExclude / 100.0) / 2.0

    lowercut = int(np.ceil(k - 0.5))

    # If all data would be trimmed, return NaN
    if (2 * lowercut) >= non_nan_count:
        return np.nan

    # slice the sorted, non-NaN part of the array
    trimmed_x = x_sorted[lowercut : non_nan_count - lowercut]

    out = np.mean(trimmed_x)
    return float(out)

