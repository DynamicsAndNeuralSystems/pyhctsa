import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
from scipy import stats
from loguru import logger


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
