import numpy as np
from numpy.typing import ArrayLike
from loguru import logger
from ..Operations.Entropy import ApproximateEntropy, SampleEntropy
from ..Operations.Correlation import AutoCorr
from typing import Dict, Union
from scipy.signal import detrend
from scipy.optimize import curve_fit


def FitPolynomial(y : ArrayLike, k : int = 1) -> float:
    """
    Goodness of a polynomial fit to a time series

    Usually kind of a stupid thing to do with a time series, but it's sometimes
    somehow informative for time series with large trends.

    Parameters:
    -----------
    y : array_like
        the time series to analyze.
    k : int, optional
        the order of the polynomial to fit to y.

    Returns:
    --------
    float
        RMS error of the fit
    """
    y = np.asarray(y)
    N = len(y)
    t = np.arange(1, N + 1)

    # Fit a polynomial to the time series
    cf = np.polyfit(t, y, k)
    f = np.polyval(cf, t) # evaluate the fitted poly
    out = np.mean((y - f)**2) # mean RMS error of fit

    return float(out)

def TSLength(y : ArrayLike) -> int:
    """
    Length of an input data vector.

    Parameters
    -----------
    y : array_like
        the time series to analyze.

    Returns
    --------
    int
        The length of the time series
    """
    return len(np.asarray(y))

def StdNthDer(y : ArrayLike, ndr : int = 2) -> float:
    """
    Standard deviation of the nth derivative of the time series.

    Estimates derivatives using successive increments of the time series and computes
    their standard deviation. The process is repeated n times to obtain higher order
    derivatives. This method is particularly popular in heart-rate variability analysis.

    Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    Consultant in a Matlab forum, who stated that "You can measure the standard
    deviation of the nth derivative, if you like".
    cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    This approach is widely used in heart-rate variability literature, see:
    "Do Existing Measures of Long-Term Heart Rate Variability...", Brennan et al. (2001)
    IEEE Trans Biomed Eng 48(11)

    Parameters
    ----------
    y : ArrayLike
        The input time series to analyze
    ndr : int, optional
        The order of derivative to analyze (default=2)
        Uses successive differences to estimate derivatives

    Returns
    -------
    float
        The standard deviation of the nth derivative of the time series
    """
    # crude method of taking a derivative that could be improved upon in future...
    y = np.asarray(y)
    yd = np.diff(y, n=ndr)
    if len(yd) == 0:
        raise ValueError(f"Time series (N = {len(y)}) too short to compute differences at n = {n}")
    out = np.std(yd, ddof=1)

    return float(out)

def Trend(y : ArrayLike) -> dict:
    """
    Quantifies various measures of trend in a time series.

    This function analyzes trends by:
    1. Computing ratio of standard deviations before/after linear detrending
    2. Fitting a linear trend and extracting parameters
    3. Analyzing statistics of the cumulative sum

    For strong linear trends, the standard deviation ratio will be low since
    detrending removes significant variance.

    Parameters
    ----------
    y : ArrayLike
        The input time series

    Returns
    -------
    Dict[str, float]
        Dictionary containing trend measures.
    """
    y = np.asarray(y)
    N = len(y)

    # ratio of std before and after linear detrending
    out = {}
    dt_y = detrend(y)
    out['stdRatio'] = np.std(dt_y, ddof=1) / np.std(y, ddof=1)
    
    # do a linear fit
    # need to use the same xrange as MATLAB with 1 indexing for correct result
    coeffs = np.polyfit(range(1, N+1), y, 1)
    out['gradient'] = coeffs[0]
    out['intercept'] = coeffs[1]

    # Stats on the cumulative sum
    yC = np.cumsum(y)
    out['meanYC'] = np.mean(yC)
    out['stdYC'] = np.std(yC, ddof=1)
    coeffs_yC = np.polyfit(range(1, N+1), yC, 1)
    out['gradientYC'] = coeffs_yC[0]
    out['interceptYC'] = coeffs_yC[1]

    # Mean cumsum in first and second half of the time series
    out['meanYC12'] = np.mean(yC[:int(np.floor(N/2))])
    out['meanYC22'] = np.mean(yC[int(np.floor(N/2)):])

    return out


def StatAv(y: ArrayLike, whatType: str = 'seg', extraParam: int = 5) -> float:
    """
    Simple mean-stationarity metric using the StatAv measure.

    This function divides the time series into non-overlapping subsegments,
    calculates the mean in each segment and returns the standard deviation
    of this set of means. The method provides a simple way to quantify 
    mean-stationarity in time series data.

    For mean-stationary data, the StatAv metric will approach zero, while
    higher values indicate non-stationarity in the mean.

    This implementation is based on:
    "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)

    Parameters
    ----------
    y : ArrayLike
        The input time series
    whatType : str, optional
        Method to segment the time series:
        - 'seg': divide into n segments (default)
        - 'len': divide into segments of length n
    extraParam : int, optional
        Specifies either:
        - Number of segments when whatType='seg' (default=5)
        - Segment length when whatType='len'

    Returns
    -------
    float
        The StatAv statistic. Values closer to zero indicate more 
        stationary means across segments.
    """
    y = np.asarray(y)
    N = len(y)

    if whatType == 'seg':
        # divide time series into n segments
        p = int(np.floor(N / extraParam))  # integer division, lose the last N mod n data points
        M = np.array([np.mean(y[p*j:p*(j+1)]) for j in range(extraParam)])
    elif whatType == 'len':
        if N > 2*extraParam:
            pn = int(np.floor(N / extraParam))
            M = np.array([np.mean(y[j*extraParam:(j+1)*extraParam]) for j in range(pn)])
        else:
            print(f"This time series (N = {N}) is too short for StatAv({whatType},'{extraParam}')")
            return np.nan
    else:
        raise ValueError(f"Error evaluating StatAv of type '{whatType}', please select either 'seg' or 'len'")

    s = np.std(y, ddof=1)  # should be 1 (for a z-scored time-series input)
    sdav = np.std(M, ddof=1)
    out = sdav / s

    return float(out)


def SlidingWindow(y: ArrayLike, windowStat: str = 'mean', acrossWinStat: str = 'std', 
                 numSeg: int = 5, incMove: int = 2) -> Dict:
    """
    Sliding window measures of stationarity.

    This function analyzes time series stationarity by sliding a window along the series,
    calculating specified statistics in each window, and then comparing these local 
    estimates across windows. For each window, it computes a statistic (windowStat) and 
    then summarizes the variation of these statistics across windows (acrossWinStat).

    This implementation is based on:
    "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)

    Note: SlidingWindow(y,'mean','std',X,1) is equivalent to StatAv(y,'seg',X)

    Parameters
    ----------
    y : ArrayLike
        The input time series to analyze
    windowStat : str, optional (default='mean')
        Statistic to calculate in each window:
        - 'mean': arithmetic mean
        - 'std': standard deviation
        - 'ent': distribution entropy (not implemented)
        - 'mom3': skewness (third moment)
        - 'mom4': kurtosis (fourth moment)
        - 'mom5': fifth moment
        - 'lillie': Lilliefors Gaussianity test p-value (not implemented)
        - 'AC1': lag-1 autocorrelation
        - 'apen': Approximate Entropy with m=1, r=0.2
        - 'sampen': Sample Entropy with m=2, r=0.1
    acrossWinStat : str, optional (default='std')
        Method to compare statistics across windows:
        - 'std': standard deviation (normalized by full series std)
        - 'ent': distribution entropy (not implemented)
        - 'apen': Approximate Entropy with m=1, r=0.2
        - 'sampen': Sample Entropy with m=2, r=0.15
    numSeg : int, optional (default=5)
        Number of segments to divide the time series into
        (controls the window length)
    incMove : int, optional (default=2)
        Controls window overlap - window moves by windowLength/incMove at each step
        (e.g., incMove=2 means 50% overlap between windows)

    Returns
    -------
    Dict[str, float]
        A measure of how the local statistics vary across the time series,
        normalized relative to the same measure computed on the full time series.
        Returns np.nan if time series is too short for specified segmentation.
    """
    y = np.asarray(y)
    winLength = np.floor(len(y)/numSeg)
    if winLength == 0:
        logger.warning(f"Time-series of length {len(y)} is too short for {numSeg} windows")
        return np.nan
    inc = np.floor(winLength/incMove) # increment to move at each step
    # if incrment rounded down to zero, prop it up 
    if inc == 0:
        inc = 1
    
    numSteps = int(np.floor((len(y)-winLength)/inc) + 1)
    qs = np.zeros(numSteps)
    
    if windowStat == 'mean':
        for i in range(numSteps):
            qs[i] = np.mean(y[_get_window(i, inc, winLength)])
    elif windowStat == 'std':
        for i in range(numSteps):
            qs[i] = np.std(y[_get_window(i, inc, winLength)], ddof=1)
    elif windowStat == 'ent':
        logger.warning(f"{windowStat} not yet implemented")
    elif windowStat == 'apen':
        for i in range(numSteps):
            qs[i] = ApproximateEntropy(y[_get_window(i, inc, winLength)], 1, 0.2)
    elif windowStat == 'sampen':
        for i in range(numSteps):
            sampen_dict = SampleEntropy(y[_get_window(i, inc, winLength)], 1, 0.1)
            qs[i] = sampen_dict['sampen1']
    # elif windowStat == 'mom3':
    #     for i in range(numSteps):
    #         qs[i] = Moments(y[_get_window(i, inc, winLength)], 3)
    # elif windowStat == 'mom4':
    #     for i in range(numSteps):
    #         qs[i] = Moments(y[_get_window(i, inc, winLength)], 4)
    # elif windowStat == 'mom5':
    #     for i in range(numSteps):
    #         qs[i] = Moments(y[_get_window(i, inc, winLength)], 5)
    elif windowStat == 'AC1':
        for i in range(numSteps):
            qs[i] = AutoCorr(y[_get_window(i, inc, winLength)], 1, 'Fourier')
    else:
        raise ValueError(f"Unknown statistic '{windowStat}'")
    
    if acrossWinStat == 'std':
        out = np.std(qs, ddof=1)/np.std(y, ddof=1)
    elif acrossWinStat == 'apen':
        out = ApproximateEntropy(qs, 1, 0.2)
    elif acrossWinStat == 'sampen':
        print(qs)
        sampen_dict = SampleEntropy(qs, 2, 0.15)
        out = sampen_dict['quadSampEn1']
    elif acrossWinStat == 'ent':
        logger.warning(f"{acrossWinStat} not yet implemented")
        out = np.nan
    else:
        raise ValueError(f"Unknown statistic '{acrossWinStat}'")
    
    return out

def _get_window(stepInd, inc, winLength):
    # helper funtion to convert a step index (stepInd) to a range of indices corresponding to that window
    start_idx = (stepInd) * inc
    end_idx = (stepInd) * inc + winLength
    return np.arange(start_idx, end_idx).astype(int)

