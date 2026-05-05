import logging
import warnings
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import detrend
from scipy.stats import gaussian_kde, kurtosis, skew
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from ..operations.correlation import autocorr, first_crossing
from ..operations.distribution import moments
from ..operations.entropy import approximate_entropy, distribution_entropy, sample_entropy
from ..utils import make_mat_buffer, sign_change, z_score

def local_distributions(y: ArrayLike, num_segs: int = 5, each_or_par: str = 'par',
                        num_points: int = 200) -> dict:
    """
    Compares the distribution in consecutive time-series segments.

    Returns the sum of differences between each kernel-smoothed distribution, either comparing each segment to the parent (full time series)
    distribution or to all other segments.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_segs : int, optional
        The number of segments to break the time series into. Default is 5.
    each_or_par : {'par', 'each'}, optional

        - 'par': compares each local distribution to the parent (full time series) distribution.
        - 'each': compares each local distribution to all other local distributions.

        Default is ``'par'``.

    num_points : int, optional
        Number of points to compute the distribution across in each local segment. Default is 200.

    Returns
    -------
    dict
        Measures of the sum of absolute deviations between distributions across the different pairwise comparisons.
    """
    # preliminaries
    y = np.asarray(y)
    N = len(y)
    lseg = int(np.floor(N / num_segs))
    dns = np.zeros((num_points, num_segs))
    # Make range of ksdensity uniform across all subsegments
    r = np.linspace(np.min(y), np.max(y), num_points)
    # Compute the kernel-smoothed distribution in all num_segs segments of the time series
    for i in range(num_segs):
        start_idx = i * lseg
        end_idx = (i + 1) * lseg
        segment_data = y[start_idx:end_idx]
        #kde = KDEUnivariate(segment_data)
        kde = gaussian_kde(segment_data, bw_method="scott")
        #kde.fit(bw="scott") # tune bw adjustment factor empiricially?
        dns[:, i] = kde.evaluate(r)
    # Compare the local distributions
    if each_or_par in ["par", "parent"]:
        #Compares each subdistribtuion to the parent (full signal) distribution
        #kde = KDEUnivariate(y).fit(bw="scott")
        kde = gaussian_kde(y, bw_method="scott")
        pardn = kde.evaluate(r)
        divs = np.zeros(num_segs)
        for i in range(num_segs):
            divs[i] = np.sum(np.abs(dns[:, i] - pardn))
    elif each_or_par == 'each':
        # Compares each subdistribtuion to the parent (full signal) distribution
        if num_segs == 2:
            out = np.sum(np.abs(dns[:, 0] - dns[:, 1]))
            return out
        # num_segs > 2
        diffmat = np.nan * np.ones((num_segs, num_segs)) 
        for i in range(num_segs):
            for j in range(num_segs):
                if j > i:
                    diffmat[i, j] = np.sum(np.abs(dns[:, i] - dns[:, j]))
        divs = diffmat[~np.isnan(diffmat)] # % (the upper triangle of diffmat)
    else:
        raise ValueError(f"Unknown method: {each_or_par}. Should be 'each' or 'par'. ")

    # Return basic statistics on differences in distributions in different
    # segments of the time series
    out = {}
    out['meandiv'] = np.mean(divs)
    out['mediandiv'] = np.median(divs)
    #out['mindiv'] = np.min(divs)
    out['maxdiv'] = np.max(divs)
    out['stddiv'] = np.std(divs)

    return out

def dyn_win(y: ArrayLike, max_num_segments: int = 10) -> dict:
    """
    How stationarity estimates depend on the number of time-series subsegments.

    Specifically, variation in a range of local measures are implemented: mean,
    standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
    AC(2), and the first zero-crossing of the autocorrelation function.

    The standard deviation of local estimates of these quantities across the time
    series are calculated as an estimate of the stationarity in this quantity as a
    function of the number of splits, n_{seg}, of the time series.

    Parameters
    -----------
    y: array-like
        The time series to analyze.
    max_num_segments: int, optional
        The maximum number of segments to consider. Sweeps from 2 to
        max_num_segments. Defaults to 10.

    Returns
    -------
    out
        The standard deviation of this set of 'stationarity' estimates across these window sizes
    """
    y = np.asarray(y)
    nsegr = np.arange(2, max_num_segments + 1, 1)  # range of nseg to sweep across
    nmov = 1  # controls window overlap
    num_features = 11  # num of features
    fs = np.zeros((len(nsegr), num_features))  # standard deviation of feature values over windows
    tau_g = first_crossing(y, 'ac', 0, 'discrete')  # global tau

    for i, nseg in enumerate(nsegr):
        wlen = int(np.floor(len(y) / nseg))  # window length
        inc = int(np.floor(wlen / nmov))  # increment to move at each step
        # if increment is rounded to zero, prop it up
        if inc == 0:
            inc = 1

        num_steps = int(np.floor((len(y) - wlen) / inc) + 1)
        qs = np.zeros((num_steps, num_features))

        for j in range(num_steps):
            y_sub = y[j * inc:j * inc + wlen]
            tau_l = first_crossing(y_sub, 'ac', 0, 'discrete')

            qs[j, 0] = np.mean(y_sub)
            qs[j, 1] = np.std(y_sub, ddof=1)
            qs[j, 2] = skew(y_sub)
            qs[j, 3] = kurtosis(y_sub)
            sampen_out = sample_entropy(y_sub, 2, 0.15)
            qs[j, 4] = sampen_out['quadSampEn1']  # SampEn_1_015
            #qs[j, 5] = sampen_out['quadSampEn2'] # SampEn_2_015
            qs[j, 6] = autocorr(y_sub, 1, 'Fourier')[0]  # AC1
            qs[j, 7] = autocorr(y_sub, 2, 'Fourier')[0]  # AC2
            # (Sometimes tau_g or taul can be longer than ySub; then these will output NaNs:)
            qs[j, 8] = autocorr(y_sub, tau_g, 'Fourier')[0]  # AC_glob_tau
            qs[j, 9] = autocorr(y_sub, tau_l, 'Fourier')[0]  # AC_loc_tau
            qs[j, 10] = tau_l

        fs[i, :num_features] = np.std(qs, ddof=1, axis=0)

    # fs contains std of quantities at all different 'scales' (segment lengths)
    # how much does the 'std stationarity' vary over different scales?
    fs = np.std(fs, ddof=1, axis=0)

    # Outputs
    out = {}
    out['stdmean'] = fs[0]
    out['stdstd'] = fs[1]
    out['stdskew'] = fs[2]
    out['stdkurt'] = fs[3]
    out['stdsampen1_015'] = fs[4]
    out['stdsampen2_015'] = fs[5]
    out['stdac1'] = fs[6]
    out['stdac2'] = fs[7]
    out['stdactaug'] = fs[8]
    out['stdactaul'] = fs[9]
    out['stdtaul'] = fs[10]

    return out

def moment_corr(x: ArrayLike, window_length: Union[None, float] = None,
                w_overlap: Union[None, float] = None, mom_1: str = 'mean',
                mom_2: str = 'std', what_transform: str = 'none') -> dict:
    """
    Correlations between simple statistics in local windows of a time series.
    The idea to implement this was that of Prof. Nick S. Jones (Imperial College London).

    Parameters
    ---------
    x : array-like
        The input time series.
    window_length : float, optional
        The sliding window length (can be a fraction to specify or a proportion of 
        the time-series length). Default is `None`.
    w_overlap : 
        The overlap between consecutive windows as a fraction of the window length. Default is `None`.
    mom_1, mom_2 : str, optional
        The statistics to investigate correlations between (in each window)

        - 'iqr': interquartile range
        - 'median': median
        - 'std': standard deviation (about the local mean)
        - 'mean': mean

        Default is ``'mean'``. 

    what_transform: str, optional
        The pre-processing what_transform to apply to the time series before analyzing it

        - 'abs': takes absolute values of all data points
        - 'sqrt': takes the square root of absolute values of all data points
        - 'sq': takes the square of every data point
        - 'none': does no what_transform

        Default is ``'none'``.
    
    Returns
    --------
    out
        Dictionary of statistics related to the correlation between simple statistics in local windows of the input time series. 
    """
    x = np.asarray(x)
    N = len(x) # length of the time series

    if window_length is None:
        window_length = 0.02 # 2% of the time-series length
    
    if window_length < 1:
        window_length = int(np.ceil(N * window_length))
    
    # sliding window overlap length
    if w_overlap is None:
        w_overlap = 1/5
    
    if w_overlap < 1:
        w_overlap = int(np.floor(window_length * w_overlap))

    # Apply the specified what_transformation
    if what_transform == 'abs':
        x = np.abs(x)
    elif what_transform == 'sq':
        x = x**2
    elif what_transform == 'sqrt':
        x = np.sqrt(np.abs(x))
    elif what_transform == 'none':
        pass
    else:
        raise ValueError(f"Unknown transformation {what_transform}")
    
    # create the windows
    x_buff = make_mat_buffer(x, window_length, w_overlap)
    num_windows = (N/(window_length - w_overlap)) # number of windows

    if np.size(x_buff, 1) > num_windows:
        x_buff = x_buff[:, :-1] # lose the last point

    points_per_window = np.size(x_buff, 0)
    if points_per_window == 1:
        raise ValueError(f"This time series (N = {N}) is too short to extract {num_windows}")
    
    # okay now we have the sliding window ('buffered') signal, x_buff
    # first calculate the first moment in all the windows
    M1 = _calc_me_moments(x_buff, mom_1)
    M2 = _calc_me_moments(x_buff, mom_2)

    out = {}
    rmat = np.corrcoef(M1, M2)
    R = rmat[0, 1] # correlation coeff
    #out['R'] = R
    out['absR'] = np.abs(rmat[0, 1])
    out['density'] = np.ptp(M1) * np.ptp(M2) / N
    #out['mi'] = MutualInfo(M1, M2, 'gaussian')

    return out

def _calc_me_moments(x_buff: ArrayLike, mom_type: str):
    """Helper function for `moment_corr`"""
    if mom_type == 'mean':
        moms = np.mean(x_buff, axis=0)
    elif mom_type == 'std':
        moms = np.std(x_buff, axis=0, ddof=1)
    elif mom_type == 'median':
        moms = np.median(x_buff, axis=0)
    elif mom_type == 'iqr':
        moms = np.percentile(x_buff, 75, method='hazen', axis=0) - np.percentile(x_buff, 25, method='hazen', axis=0)
    else:
        raise ValueError(f"Unknown statistic {mom_type}")
    
    return moms

def simple_stats(x: ArrayLike, what_stat: str = 'zcross') -> dict:
    """
    Basic statistics about an input time series.

    This function computes various statistical measures about zero-crossings and local 
    extrema in a time series.

    Parameters
    ----------
    x : array-like
        The input time series.
    what_stat : str, optional
        The statistic to return:

        - 'zcross': proportion of zero-crossings (for z-scored input, returns mean-crossings)
        - 'maxima': proportion of points that are local maxima
        - 'minima': proportion of points that are local minima
        - 'pmcross': ratio of crossings above +1σ to crossings below -1σ
        - 'zsczcross': ratio of zero-crossings in raw vs detrended time series

        Default is `zcross`.

    Returns
    -------
    float
        The calculated statistic based on what_stat
    """
    x = np.asarray(x)
    N = len(x)

    out = None
    if what_stat == 'zcross':
        # Proportion of zero-crossings of the time series
        # (% in the case of z-scored input, crosses its mean)
        xch = x[:-1] * x[1:]
        out = np.sum(xch < 0)/N

    elif what_stat == 'maxima':
        # proportion of local maxima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] > 0) & (dx[1:] < 0)) / (N - 1)
    elif what_stat == 'minima':
        # proportion of local minima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] < 0) & (dx[1:] > 0)) / (N-1)
    elif what_stat == 'pmcross':
        # ratio of times cross 1 to -1
        c1sig = np.sum(sign_change(x-1)) # num times cross 1
        c2sig = np.sum(sign_change(x+1)) # num times cross -1
        if c2sig == 0:
            out = np.nan
        else:
            out = c1sig/c2sig
    elif what_stat == 'zsczcross':
        # ratio of zero crossings of raw to detrended time series
        # where the raw has zero mean
        x = z_score(x)
        xch = x[:-1] * x[1:]
        h1 = np.sum(xch < 0) # num of zscross of raw series
        y = detrend(x)
        ych = y[:-1] * y[1:]
        h2 = np.sum(ych < 0) # % of detrended series
        if h1 == 0:
            out = np.nan
        else:
            out = h2/h1
    else:
        raise(ValueError(f"Unknown statistic {what_stat}"))
    
    return out

def local_extrema(y: ArrayLike, how_to_window: str = 'l', n: Union[int, None] = None) -> dict:
    """
    How local maximums and minimums vary across the time series.

    Finds maximums and minimums within given segments of the time series and
    analyzes the results.

    Parameters
    ----------
    y : array-like
        The input time series
    how_to_window : str, optional
        Method to determine window size (default is 'l'):

        - 'l': windows of a given length (n specifies the window length)
        - 'n': specified number of windows to break the time series into (n specifies number of windows)
        - 'tau': sets window length equal to correlation length (first zero-crossing of autocorrelation)

        Default is ``'l'``.

    n : int, optional
        Specifies either:

        - Window length when how_to_window='l' (defaults to 100)
        - Number of windows when how_to_window='n' (defaults to 5)
        - Not used when how_to_window='tau'

        Default is `None`.

    Returns
    -------
    dict
        Statistics about local extrema.
    """
    y = np.asarray(y)
    if n is None:
        if how_to_window == 'l':
            n = 100 # 100 sample windows
        elif how_to_window == 'n':
            n = 5 # 5 windows
    N = len(y)
    # Set the window length
    if how_to_window == 'l':
        window_length = n # window length
    elif how_to_window == 'n':
        window_length = int(np.floor(N/n))
    elif how_to_window == 'tau':
        window_length = first_crossing(y, 'ac', 0, 'discrete')
    else:
        raise ValueError(f"Unknown method {how_to_window}")
    
    if (window_length > N) or (window_length <= 1):
        # This feature is unsuitable if the window length exceeds ts
        out = np.nan
    
    # Buffer the time series
    y_buff = make_mat_buffer(y, window_length) # no overlap
    # each column is a window of samples
    if y_buff[-1, -1] == 0:
        y_buff = y_buff[:, :-1]  # remove last window if zero-padded
    
    num_windows = np.size(y_buff, 1) # number of windows
    # Find local extrema
    loc_max = np.max(y_buff, axis=0) # summary of local maxima
    loc_min = np.min(y_buff, axis=0) # summary of local minima
    abs_loc_min = np.abs(loc_min) # abs val of local minima
    exti = np.where(abs_loc_min > loc_max)
    loc_ext = loc_max.copy()
    loc_ext[exti] = loc_min[exti] # local extrema (furthest from mean; either maxs or mins)
    abs_loc_ext = np.abs(loc_ext) # the magnitude of the most extreme events in each window

    # Return Outputs
    out = {
        'meanrat': np.mean(loc_max) / np.mean(abs_loc_min),
        'medianrat': np.median(loc_max) / np.median(abs_loc_min),
        'minmax': np.min(loc_max),
        'minabsmin': np.min(abs_loc_min),
        'minmaxonminabsmin': np.min(loc_max) / np.min(abs_loc_min),
        'meanmax': np.mean(loc_max),
        'meanabsmin': np.mean(abs_loc_min),
        'meanext': np.mean(loc_ext),
        'medianmax': np.median(loc_max),
        'medianabsmin': np.median(abs_loc_min),
        'medianext': np.median(loc_ext),
        'stdmax': np.std(loc_max, ddof=1),
        'stdmin': np.std(loc_min, ddof=1),
        'stdext': np.std(loc_ext, ddof=1),
        'zcext': np.sum((loc_ext[:-1] * loc_ext[1:]) < 0) / num_windows,
        'meanabsext': np.mean(abs_loc_ext),
        'medianabsext': np.median(abs_loc_ext),
        'diffmaxabsmin': np.sum(np.abs(loc_max - abs_loc_min)) / num_windows,
        'uord': np.sum(np.sign(loc_ext)) / num_windows,
        'maxmaxmed': np.max(loc_max) / np.median(loc_max),
        'minminmed': np.min(loc_min) / np.median(loc_min),
        'maxabsext': np.max(abs_loc_ext) / np.median(abs_loc_ext)
    }

    return out

def kpss_test(y: ArrayLike, lags: Union[int, list] = 0) -> dict:
    """
    Performs the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

    This implementation uses the statsmodels `kpss` function to test whether a time series
    is trend stationary. The null hypothesis is that the time series is trend stationary,
    while the alternative hypothesis is that it is a non-stationary unit-root process.

    The test was introduced in [1].

    The function can be used in two ways:
    1. With a single lag value - returns basic test statistic and p-value
    2. With multiple lag values - returns statistics about how the test results 
       change across different lags
    
    References
    ----------
    .. [1] Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing the null 
        hypothesis of stationarity against the alternative of a unit root: How sure are we 
        that economic time series have a unit root? Journal of Econometrics, 54(1-3), 159-178.

    Parameters
    ----------
    y : ArrayLike
        The input time series to analyze for stationarity
    lags : Union[int, list], optional
        Either:

        - A single lag value (int) to compute the test statistic and p-value
        - A list of lag values to analyze how the test results vary across lags

        Default is 0.

    Returns
    -------
    Dict[str, float]
        The KPSS test statistic and p-value of the test.
    """
    if isinstance(lags, list):
        # evaluate kpss at multiple lags
        p_value = np.zeros(len(lags))
        stat = np.zeros(len(lags))
        for (i, l) in enumerate(lags):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                s, pv, _, _ = kpss(y, nlags=l, regression='ct')
            p_value[i] = pv
            stat[i] = s
        out = {}
        # return stats on outputs
        out['maxpValue'] = np.max(p_value)
        out['minpValue'] = np.min(p_value)
        out['maxstat'] = np.max(stat)
        out['minstat'] = np.min(stat)
        out['lagmaxstat'] = lags[np.argmax(stat)]
        out['lagminstat'] = lags[np.argmin(stat)]
    else:
        if isinstance(lags, int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                stat, p_value, _, _ = kpss(y, nlags=lags, regression='ct')
            # return the statistic and pvalue
            out = {'stat': stat, 'pValue': p_value}
        else:
            raise TypeError("Expected either a single lag (as an int) or list of lags.")
    return out

def range_evolve(y: ArrayLike) -> dict:
    """
    Analyze how the time-series range changes across time.

    This operation measures the range (peak-to-peak) of the time series as a function
    of time by calculating range(x_{1:i}) for i = 1, 2, ..., N, where N is the 
    length of the time series. It provides insights into how new extreme events 
    emerge over time.

    Parameters
    ----------
    y : array-like
        The input time series to analyze.

    Returns
    -------
    Dict[str, float]
        Dictionary containing various metrics about range evolution.
    """
    y = np.asarray(y)
    N = len(y)
    out = {} # initialise storage
    cums = np.zeros(N)
    for i in range(N):
        cums[i] = np.ptp(y[:i+1])  # np.ptp calculates the range (peak to peak)
    
    fullr = np.ptp(y)

    # return number of unqiue entries in a vector, x
    lunique = lambda x : len(np.unique(x))
    out['totnuq'] = lunique(cums)

    # how many of the unique extrema are in the first <proportions> of time series?
    cumtox = lambda x : lunique(cums[:int(np.floor(N*x))])/out['totnuq']
    out['nuqp1'] = cumtox(0.01)
    out['nuqp10'] = cumtox(0.1)
    out['nuqp20'] = cumtox(0.2)
    out['nuqp50'] = cumtox(0.5)

    # how many unique extrema are in the first <length> of time series?
    ns = [10, 50, 100, 1000]
    for n_val in ns:
        if N >= n_val:
            out[f'nuql{n_val}'] = lunique(cums[:n_val])/out['totnuq']
        else:
            out[f'nuql{n_val}'] = np.nan
    # (**2**) Actual proportion of full range captured at different points
    out['p1'] = cums[int(np.ceil(N * 0.01)) - 1]/fullr
    out['p10'] = cums[int(np.ceil(N * 0.1)) - 1]/fullr
    out['p20'] = cums[int(np.ceil(N * 0.2)) - 1]/fullr
    out['p50'] = cums[int(np.ceil(N * 0.5)) - 1]/fullr

    for n_val in ns:
        if N >= n_val:
            out[f'l{n_val}'] = cums[n_val-1]/fullr
        else:
            out[f'l{n_val}'] = np.nan

    return out

def drifting_mean(y: ArrayLike, segment_how: str = 'fix', l: int = 20) -> dict:
    """
    Measures mean drift by analyzing mean and variance in time-series subsegments.

    This operation splits a time series into segments, computes the mean and variance 
    in each segment, and compares the maximum and minimum means to the mean variance. 
    This helps identify if the time series has a drifting mean by comparing local 
    statistics across different segments.

    The method follows this approach:
    1. Splits signal into frames of length N (or num segments)
    2. Computes means of each frame
    3. Computes variance for each frame
    4. Compares ratio of max/min means with mean variance

    Original idea by Rune from Matlab Central forum:
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters
    ----------
    y : array-like
        The input time series
    segment_how : str, optional
        Method to segment the time series:

        - 'fix': fixed-length segments of length l
        - 'num': splits into l number of segments

        Default is ``'fix'``.

    l : int, optional
        Specifies either:

        - The length of segments when segment_how='fix' (default=20)
        - The number of segments when segment_how='num'

        Default is 20.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the measures of mean drift.
    """
    y = np.asarray(y)
    N = len(y)
    
    # Set default segment parameters
    if l is None:
        l = 200 if segment_how == 'fix' else 5
    
    # Calculate segment length
    if segment_how == 'num':
        segment_length = int(np.floor(N/l))
    elif segment_how == 'fix':
        segment_length = l
    else:
        raise ValueError(f"segment_how must be 'fix' or 'num', got {segment_how}")
    
    # Validate segment length
    if segment_length <= 1 or segment_length > N:
        return {
            'max': np.nan,
            'min': np.nan,
            'mean': np.nan,
            'meanmaxmin': np.nan,
            'meanabsmaxmin': np.nan
        }
    
    # Calculate number of complete segments
    num_segments = int(np.floor(N/segment_length))
    
    # More efficient segmentation using array operations
    segments = y[:num_segments * segment_length].reshape(num_segments, segment_length)
    
    # Calculate statistics
    segment_means = np.mean(segments, axis=1)
    segment_vars = np.var(segments, axis=1, ddof=1)
    mean_var = np.mean(segment_vars)
    
    # Prepare output statistics
    out = {
        'max': np.max(segment_means) / mean_var,
        'min': np.min(segment_means) / mean_var,
        'mean': np.mean(segment_means) / mean_var
    }
    out['meanmaxmin'] = (out['max'] + out['min']) / 2
    out['meanabsmaxmin'] = (np.abs(out['max']) + np.abs(out['min'])) / 2

    return out

def local_global(y: ArrayLike, subset_how: str = 'l', n: Union[int, float, None] = None) -> dict:
    """
    Compare local statistics to global statistics of a time series.

    Parameters
    -----------
    y : ArrayLike
        The time series to analyse.
    subset_how : str, optional
        The method to select the local subset of time series:

        - 'l': the first n points in a time series
        - 'p': an initial proportion of the full time series
        - 'unicg': n evenly-spaced points throughout the time series
        - 'randcg': n randomly-chosen points from the time series (chosen with replacement)

        Default is ``'l'``.

    n : int or float, optional
        The parameter for the method specified by subset_how.
        
        Default `None` is 100 samples or 0.1 (10% of time series length) if proportion. 

    Returns
    --------
    dict
        A dictionary containing various statistical measures comparing
        the subset to the full time series.
    """
    # check input time series is z-scored
    y = np.asarray(y)

    if n is None:
        if subset_how in ['l', 'unicg', 'randcg']:
            n = 100 # 100 samples
        elif subset_how == 'p':
            n = 0.1 # 10 % of time series
    N = len(y)

    # Determine subset range to use: r
    if subset_how == 'l':
        # take first n pts of time series
        r = np.arange(min(n, N))
    elif subset_how == 'p':
        # take initial proportion n of time series
        r = np.arange(int(np.floor(N*n)))
    elif subset_how == 'unicg':
        r = np.round(np.linspace(1, N, n)).astype(int) - 1
    else:
        raise ValueError(f"Unknown specifier, {subset_how}. Can be either 'l', 'p', 'unicg', or 'randcg'.")

    if len(r) < 5:
        # It's not really appropriate to compute statistics on less than 5 datapoints
        logging.warning(f"Time series (of length {N}) is too short")
        return np.nan
    
    # Compare statistics of this subset to those obtained from the full time series
    out = {}
    out['absmean'] = np.abs(np.mean(y[r])) # Makes sense without normalization if y is z-scored
    out['std'] = np.std(y[r], ddof=1) # Makes sense without normalization if y is z-scored
    out['median'] = np.median(y[r]) # if median is very small then normalization could be very noisy
    raw_iqr_yr = np.percentile(y[r], 75, method='hazen') - np.percentile(y[r], 25, method='hazen')
    raw_iqr_y = np.percentile(y, 75, method='hazen') - np.percentile(y, 25, method='hazen')
    out['iqr'] = np.abs(1 - (raw_iqr_yr/raw_iqr_y))
    out['skewness'] = np.abs(1 - (skew(y[r])/skew(y)))
    # use Pearson definition (normal ==> 3.0)
    out['kurtosis'] = np.abs(1 - (kurtosis(y[r], fisher=False)/kurtosis(y, fisher=False)))
    out['ac1'] = np.abs(1 - (autocorr(y[r], 1, 'Fourier')[0]/autocorr(y, 1, 'Fourier')[0]))
    out['sampen101'] = sample_entropy(y[r], 1, 0.1)['sampen1']/sample_entropy(y, 1, 0.1)['sampen1']

    return out

def fit_polynomial(y: ArrayLike, k: int = 1) -> float:
    """
    Goodness of a polynomial fit to a time series

    Usually kind of a stupid thing to do with a time series, but it's sometimes
    somehow informative for time series with large trends.

    Parameters
    -----------
    y : ArrayLike
        the time series to analyze.
    k : int, optional
        the order of the polynomial to fit to y. Default is 1.

    Returns
    --------
    float
        RMS error of the fit.
    """
    y = np.asarray(y)
    N = len(y)
    t = np.arange(1, N + 1)

    # Fit a polynomial to the time series
    cf = np.polyfit(t, y, k)
    f = np.polyval(cf, t) # evaluate the fitted poly
    out = np.mean((y - f)**2) # mean RMS error of fit

    return float(out)

def ts_length(y: ArrayLike) -> int:
    """
    Length of an input data vector.

    Parameters
    -----------
    y : array-like
        The time series to analyze.

    Returns
    --------
    int
        The length of the time series.
    """
    return len(np.asarray(y))

def std_nth_deriv(y: ArrayLike, ndr: int = 2) -> float:
    """
    Standard deviation of the nth derivative of the time series.

    Estimates derivatives using successive increments of the time series and computes
    their standard deviation. The process is repeated n times to obtain higher order
    derivatives. This method is particularly popular in heart-rate variability analysis.

    Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    Consultant in a Matlab forum, who stated that "You can measure the standard
    deviation of the nth derivative, if you like".
    cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    This approach is widely used in heart-rate variability literature, see [1].

    References
    ----------
    .. [1] "Do Existing Measures of Long-Term Heart Rate Variability...", Brennan et al. (2001)
        IEEE Trans Biomed Eng 48(11)

    Parameters
    ----------
    y : array-like
        The input time series to analyze.
    ndr : int, optional
        The order of derivative to analyze.
        Uses successive differences to estimate derivatives. Default is 2.

    Returns
    -------
    float
        The standard deviation of the nth derivative of the time series.
    """
    # crude method of taking a derivative that could be improved upon in future...
    y = np.asarray(y)
    yd = np.diff(y, n=ndr)
    if len(yd) == 0:
        raise ValueError(f"Time series (N = {len(y)}) too short to compute differences at n = {n}")
    out = np.std(yd, ddof=1)

    return float(out)

def trend(y: ArrayLike) -> dict:
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
    y : array-like
        The input time series.

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
    yc = np.cumsum(y)
    out['meanYC'] = np.mean(yc)
    out['stdYC'] = np.std(yc, ddof=1)
    coeffs_yc = np.polyfit(range(1, N+1), yc, 1)
    out['gradientYC'] = coeffs_yc[0]
    out['interceptYC'] = coeffs_yc[1]

    # Mean cumsum in first and second half of the time series
    out['meanYC12'] = np.mean(yc[:int(np.floor(N/2))])
    out['meanYC22'] = np.mean(yc[int(np.floor(N/2)):])

    return out

def stat_av(y: ArrayLike, what_type: str = 'seg', extra_param: int = 5) -> float:
    """
    Simple mean-stationarity metric using the StatAv measure.

    This function divides the time series into non-overlapping subsegments,
    calculates the mean in each segment and returns the standard deviation
    of this set of means. The method provides a simple way to quantify 
    mean-stationarity in time series data.

    For mean-stationary data, the StatAv metric will approach zero, while
    higher values indicate non-stationarity in the mean.

    This implementation is based on [1].

    References
    ----------
    .. [1] "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
        Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)

    Parameters
    ----------
    y : array-like
        The input time series
    what_type : str, optional
        Method to segment the time series:

        - 'seg': divide into n segments
        - 'len': divide into segments of length n

        Default is ``'seg'``.

    extra_param : int, optional
        Specifies either:

        - Number of segments when what_type='seg'
        - Segment length when what_type='len'

        Default is 5.

    Returns
    -------
    float
        The stat_av statistic. Values closer to zero indicate more 
        stationary means across segments.
    """
    y = np.asarray(y)
    N = len(y)

    if what_type == 'seg':
        # divide time series into n segments
        p = int(np.floor(N / extra_param))  # integer division, lose the last N mod n data points
        M = np.array([np.mean(y[p*j:p*(j+1)]) for j in range(extra_param)])
    elif what_type == 'len':
        if N > 2*extra_param:
            pn = int(np.floor(N / extra_param))
            M = np.array([np.mean(y[j*extra_param:(j+1)*extra_param]) for j in range(pn)])
        else:
            logging.warning(f"This time series (N = {N}) is too short for stat_av({what_type},'{extra_param}')")
            return np.nan
    else:
        raise ValueError(f"Error evaluating stat_av of type '{what_type}', please select either 'seg' or 'len'")

    s = np.std(y, ddof=1)  # should be 1 (for a z-scored time-series input)
    sdav = np.std(M, ddof=1)
    out = sdav / s

    return float(out)

def sliding_window(y: ArrayLike, window_stat: str = 'mean', across_win_stat: str = 'std',
                 num_seg: int = 5, inc_move: int = 2):
    """
    Sliding window measures of stationarity.

    This function analyzes time series stationarity by sliding a window along the series,
    calculating specified statistics in each window, and then comparing these local 
    estimates across windows. For each window, it computes a statistic (window_stat) and 
    then summarizes the variation of these statistics across windows (across_win_stat).

    This implementation is based on:

    References
    ----------
    .. [1] "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
        Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)

    Note: SlidingWindow(y,'mean','std',X,1) is equivalent to StatAv(y,'seg',X)

    Parameters
    ----------
    y : array-like
        The input time series to analyze.
    window_stat : str, optional (default='mean')
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

        Default is ``'mean'``.

    across_win_stat : str, optional
        Method to compare statistics across windows:

        - 'std': standard deviation (normalized by full series std)
        - 'ent': distribution entropy (not implemented)
        - 'apen': Approximate Entropy with m=1, r=0.2
        - 'sampen': Sample Entropy with m=2, r=0.15

        Default is ``'std'``.
        
    num_seg : int, optional
        Number of segments to divide the time series into. Default is 5.
        (controls the window length)
    inc_move : int, optional
        Controls window overlap - window moves by window_length/inc_move at each step
        (e.g., inc_move=2 means 50% overlap between windows). Default is 2.

    Returns
    -------
    Dict[str, float]
        A measure of how the local statistics vary across the time series,
        normalized relative to the same measure computed on the full time series.
        Returns np.nan if time series is too short for specified segmentation.
    """
    y = np.asarray(y)
    win_length = np.floor(len(y)/num_seg)
    if win_length == 0:
        logging.warning(f"Time-series of length {len(y)} is too short for {num_seg} windows")
        return np.nan
    inc = np.floor(win_length/inc_move) # increment to move at each step
    # if incrment rounded down to zero, prop it up
    if inc == 0:
        inc = 1
    
    num_steps = int(np.floor((len(y)-win_length)/inc) + 1)
    qs = np.zeros(num_steps)
    
    if window_stat == 'mean':
        for i in range(num_steps):
            qs[i] = np.mean(y[_get_window(i, inc, win_length)])
    elif window_stat == 'std':
        for i in range(num_steps):
            qs[i] = np.std(y[_get_window(i, inc, win_length)], ddof=1)
    elif window_stat == 'ent':
        for i in range(num_steps):
            qs[i] = distribution_entropy(y[_get_window(i, inc, win_length)], 'ks','[]')
    elif window_stat == 'apen':
        for i in range(num_steps):
            qs[i] = approximate_entropy(y[_get_window(i, inc, win_length)], 1, 0.2)
    elif window_stat == 'sampen':
        for i in range(num_steps):
            sampen_dict = sample_entropy(y[_get_window(i, inc, win_length)], 1, 0.1)
            qs[i] = sampen_dict['sampen1']
    elif window_stat == 'mom3':
        for i in range(num_steps):
            qs[i] = moments(y[_get_window(i, inc, win_length)], 3)
    elif window_stat == 'mom4':
        for i in range(num_steps):
            qs[i] = moments(y[_get_window(i, inc, win_length)], 4)
    elif window_stat == 'mom5':
        for i in range(num_steps):
            qs[i] = moments(y[_get_window(i, inc, win_length)], 5)
    elif window_stat == 'AC1':
        for i in range(num_steps):
            qs[i] = np.asarray(autocorr(y[_get_window(i, inc, win_length)], 1, 'Fourier')).item()
    else:
        raise ValueError(f"Unknown statistic '{window_stat}'")
    
    if across_win_stat == 'std':
        #% normalized by std of full time series
        out = np.std(qs, ddof=1)/np.std(y, ddof=1)
    elif across_win_stat == 'apen':
        out = approximate_entropy(qs, 1, 0.2)
    elif across_win_stat == 'sampen':
        sampen_dict = sample_entropy(qs, 2, 0.15)
        out = sampen_dict['quadSampEn1']
    elif across_win_stat == 'ent':
        #% get a load of statistics from kernel-smoothed distribution
        kde = gaussian_kde(qs)
        xi = np.linspace(qs.min() - 3 * np.std(qs, ddof=1), qs.max() + 3 * np.std(qs, ddof=1), 100)
        f = kde(xi)
        f_pos = f[f > 0]
        dx = xi[1] - xi[0]
        dist_ent = -np.sum(f_pos * np.log(f_pos) * dx)
        out = dist_ent
    else:
        raise ValueError(f"Unknown statistic '{across_win_stat}'")
    
    return out

def _get_window(step_ind, inc, win_length):
    # helper funtion to convert a step index (stepInd) to a range of indices corresponding to that window
    start_idx = (step_ind) * inc
    end_idx = (step_ind) * inc + win_length
    
    return np.arange(start_idx, end_idx).astype(int)
