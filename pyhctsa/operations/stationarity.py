import logging
logger = logging.getLogger('pyhctsa')
import warnings
from itertools import permutations
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import detrend
from scipy.stats import gaussian_kde, kurtosis, norm, pearsonr, rankdata, skew
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from ..operations.correlation import autocorr, first_crossing
from ..operations.distribution import moments
from ..operations.entropy import approximate_entropy, distribution_entropy, sample_entropy
from ..toolboxes.matlab.matlab_fit import polyfit, robustfit
from ..toolboxes.matlab._pptest_tables import PP_CV_TABLES, PP_SAMP_SIZES, PP_SIG_LEVELS
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
    num_points = int(num_points)
    num_segs = int(num_segs)
    lseg = int(np.floor(N / num_segs))
    dns = np.zeros((num_points, num_segs))
    # Make range of ksdensity uniform across all subsegments
    r = np.linspace(np.min(y), np.max(y), num_points)
    # Compute the kernel-smoothed distribution in all num_segs segments of the time series
    for i in range(num_segs):
        start_idx = i * lseg
        end_idx = (i + 1) * lseg
        segment_data = y[start_idx:end_idx]
        kde = gaussian_kde(segment_data, bw_method='scott')
        dns[:, i] = kde.evaluate(r)
    # Compare the local distributions
    if each_or_par in ['par', 'parent']:
        #Compares each subdistribtuion to the parent (full signal) distribution
        kde = gaussian_kde(y, bw_method='scott')
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
    out['maxdiv'] = np.max(divs)
    out['stddiv'] = np.std(divs, ddof=1)

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
        logger.warning(f"This time series (N = {N}) is too short to extract {num_windows}")
        return np.nan
    
    # okay now we have the sliding window ('buffered') signal, x_buff
    # first calculate the first moment in all the windows
    M1 = _calc_me_moments(x_buff, mom_1)
    M2 = _calc_me_moments(x_buff, mom_2)

    out = {}
    rmat = np.corrcoef(M1, M2)
    R = rmat[0, 1] # correlation coeff
    out['R'] = R
    out['absR'] = np.abs(rmat[0, 1])
    out['density'] = len(M1)/(np.ptp(M1)*np.ptp(M2))

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
        return np.nan
    
    # Buffer the time series
    y_buff = make_mat_buffer(y, int(window_length)) # no overlap
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
    y : array-like
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
        if isinstance(lags, (int, float)):
            lags = int(lags)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                stat, p_value, _, _ = kpss(y, nlags=lags, regression='ct')
            # return the statistic and pvalue
            out = {'stat': stat, 'pValue': p_value}
        else:
            raise TypeError("Expected either a single lag (as an int) or list of lags.")
    return out

def _pp_regression(y: np.ndarray, test_lags: int, model: str) -> dict:
    """
    OLS regression and Newey-West long-run variance underlying the PP test.

    Reproduces the ``runReg`` subfunction of MATLAB's ``pptest``; follows
    Hamilton (1994), p. 514.
    """
    T = len(y) - 1
    y_lag = y[:-1]
    test_y = y[1:]

    if model == 'ar':
        # y(t) = a*y(t-1) + e(t)
        X = y_lag[:, None]
    elif model == 'ard':
        # y(t) = c + a*y(t-1) + e(t)
        X = np.column_stack([np.ones(T), y_lag])
    elif model == 'ts':
        # y(t) = c + d*t + a*y(t-1) + e(t)
        X = np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float), y_lag])
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'ar', 'ard' or 'ts'.")

    # The AR(1) coefficient is always the last column by construction
    Q, R = np.linalg.qr(X)
    coeff = np.linalg.solve(R, Q.T @ test_y)
    num_params = len(coeff)
    res = test_y - X @ coeff
    SSE = res @ res
    dfe = T - num_params
    MSE = SSE / dfe
    S = np.linalg.solve(R, np.eye(num_params))
    cov = S @ S.T * MSE

    # Estimated residual autocovariances:
    gamma = np.array([(res[j:] @ res[:len(res) - j]) / T for j in range(test_lags + 1)])

    # Newey-West estimator:
    lambda_sq = gamma[0] + 2 * np.sum(
        [(1 - j / (test_lags + 1)) * gamma[j] for j in range(1, test_lags + 1)])

    sigma = np.sqrt(MSE)
    # Loglikelihood of the residuals under Gaussian innovations, N(0, sigma)
    LL = -(T / 2) * np.log(2 * np.pi) - T * np.log(sigma) - SSE / (2 * MSE)

    return {
        'coeff': coeff,
        'a': coeff[-1],
        'se_a': np.sqrt(np.diag(cov))[-1],
        'MSE': MSE,
        'RMSE': sigma,
        'gamma0': gamma[0],
        'NWEst': lambda_sq,
        'LL': LL,
        'AIC': 2 * num_params - 2 * LL,
        'BIC': num_params * np.log(T) - 2 * LL,
        'HQC': 2 * num_params * np.log(np.log(T)) - 2 * LL,
    }


def _pp_pvalue(stat: float, T: int, model: str, test_statistic: str) -> float:
    """
    Interpolate a p-value from the tabulated Phillips-Perron critical values.

    Two successive 1-D interpolations, matching MATLAB's ``getPValue``: first
    across sample sizes to get the critical-value row for this T, then across
    that row to get the cumulative probability of the observed statistic.
    """
    samp_sizes = PP_SAMP_SIZES.copy()
    # MATLAB forces max(T, 10000) into the final row rather than extrapolating
    samp_sizes[-1] = max(samp_sizes[-1], T)
    table = PP_CV_TABLES[(model, test_statistic)]

    # interp2(...,'linear') over the sample-size axis at this T
    cv_row = np.array([np.interp(T, samp_sizes, table[:, j], left=np.nan, right=np.nan)
                       for j in range(table.shape[1])])

    if stat <= cv_row[0]:
        return PP_SIG_LEVELS[0]
    if stat >= cv_row[-1]:
        return PP_SIG_LEVELS[-1]
    return float(np.interp(stat, cv_row, PP_SIG_LEVELS))


def pp_test(y: ArrayLike, lags: Union[int, list] = None, model: str = 'ar',
            test_statistic: str = 't1') -> dict:
    """
    Phillips-Perron unit root test.

    The null hypothesis is that the series contains a unit root (i.e., is a random walk, 
    possibly with drift); the alternative is that it is stationary about the specified
    deterministic trend.

    The test statistic is a non-parametric correction of the Dickey-Fuller
    statistic, using a Newey-West estimate of the long-run variance in place of
    the augmenting lagged differences.

    References
    ----------
    .. [1] P. C. B. Phillips and P. Perron, "Testing for a unit root in time series
        regression", *Biometrika*, 75(2), 335 (1988).
    .. [2] J. D. Hamilton, *Time Series Analysis*, Princeton University Press (1994),
        p. 514.

    Parameters
    ----------
    y : array-like
        The input time series.
    lags : int or list of int, optional
        The number of autocovariance lags to include in the Newey-West estimator
        of the long-run variance. A list runs one test per lag and returns
        summary statistics across them. Default is ``range(0, 6)``.
    model : {'ar', 'ard', 'ts'}, optional
        The regression model: 'ar' (autoregressive, no deterministic terms),
        'ard' (autoregressive with drift) or 'ts' (trend stationary).
        Default is 'ar'.
    test_statistic : {'t1', 't2'}, optional
        't1' is the standard t-statistic; 't2' is a lag-adjusted,
        'unStudentized' statistic. Default is 't1'.

    Returns
    -------
    dict
        For a single lag: the p-value, statistic, first regression coefficient
        and regression fit statistics. For multiple lags: summary statistics on
        the p-values, statistics and regression fit statistics across lags.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]  # remove missing values
    if not np.all(np.isfinite(y)):
        raise ValueError("The input time series must be real and finite.")

    model = model.lower()
    test_statistic = test_statistic.lower()
    if test_statistic not in ('t1', 't2'):
        raise ValueError(f"Unknown test statistic '{test_statistic}'. Use 't1' or 't2'.")

    if lags is None:
        lags = list(range(0, 6))  # 5 autoregressive lags
    single = np.isscalar(lags)
    lag_list = [int(lags)] if single else [int(l) for l in lags]

    T = len(y) - 1
    p_values, stats, regs = [], [], []
    for l in lag_list:
        reg = _pp_regression(y, l, model)
        a, se_a, MSE = reg['a'], reg['se_a'], reg['MSE']
        gamma0, lambda_sq = reg['gamma0'], reg['NWEst']

        if test_statistic == 't1':
            stat = (np.sqrt(gamma0 / lambda_sq) * (a - 1) / se_a
                    - 0.5 * (lambda_sq - gamma0) / np.sqrt(lambda_sq) * T * se_a / np.sqrt(MSE))
        else:
            stat = T * (a - 1) - 0.5 * ((T * se_a) ** 2 / MSE) * (lambda_sq - gamma0)

        stats.append(stat)
        p_values.append(_pp_pvalue(stat, T, model, test_statistic))
        regs.append(reg)

    if single:
        reg = regs[0]
        return {
            'pvalue': p_values[0],
            'stat': stats[0],
            'coeff1': reg['coeff'][0],  # could be multiple, depending on the model
            'loglikelihood': reg['LL'],
            'AIC': reg['AIC'],
            'BIC': reg['BIC'],
            'HQC': reg['HQC'],
            'rmse': reg['RMSE'],
        }

    # Return statistics on the set of outputs
    p_values = np.asarray(p_values)
    stats = np.asarray(stats)
    return {
        'maxpValue': np.max(p_values),
        'minpValue': np.min(p_values),
        'meanpValue': np.mean(p_values),
        'stdpValue': np.std(p_values, ddof=1),
        'lagmaxp': lag_list[int(np.argmax(p_values))],
        'lagminp': lag_list[int(np.argmin(p_values))],

        'meanstat': np.mean(stats),
        'maxstat': np.max(stats),
        'minstat': np.min(stats),

        'meanloglikelihood': np.mean([r['LL'] for r in regs]),
        'minAIC': np.min([r['AIC'] for r in regs]),
        'minBIC': np.min([r['BIC'] for r in regs]),
        'minHQC': np.min([r['HQC'] for r in regs]),

        'minrmse': np.min([r['RMSE'] for r in regs]),
        'maxrmse': np.max([r['RMSE'] for r in regs]),
    }

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
    # Running peak-to-peak == cumulative max minus cumulative min: O(N) one pass
    # instead of O(N^2) np.ptp over growing prefixes. Picks the same values.
    cums = (np.maximum.accumulate(y) - np.minimum.accumulate(y)).astype(float)

    fullr = np.ptp(y)

    # return number of unique entries in a vector, x
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
    l = int(l)
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
    y : array-like
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
        logger.warning(f"Time series (of length {N}) is too short")
        return np.nan
    
    # Compare statistics of this subset to those obtained from the full time series
    out = {}
    out['absmean'] = np.abs(np.mean(y[r])) # Makes sense without normalization if y is z-scored
    out['std'] = np.std(y[r], ddof=1) # Makes sense without normalization if y is z-scored
    out['median'] = np.median(y[r]) # if median is very small then normalization could be very noisy
    raw_iqr_yr = np.percentile(y[r], 75, method='hazen') - np.percentile(y[r], 25, method='hazen')
    raw_iqr_y = np.percentile(y, 75, method='hazen') - np.percentile(y, 25, method='hazen')
    out['iqr'] = np.abs(1 - (raw_iqr_yr/raw_iqr_y)) if raw_iqr_y > 0 else np.nan
    out['skewness'] = np.abs(1 - (skew(y[r])/skew(y)))
    # use Pearson definition (normal ==> 3.0)
    out['kurtosis'] = np.abs(1 - (kurtosis(y[r], fisher=False)/kurtosis(y, fisher=False)))
    out['ac1'] = np.abs(1 - (autocorr(y[r], 1, 'Fourier')[0]/autocorr(y, 1, 'Fourier')[0]))

    sampen_full = sample_entropy(y, 1, 0.1)['sampen1']
    sampen_r = sample_entropy(y[r], 1, 0.1)['sampen1']
    out['sampen101'] = sampen_r / sampen_full if sampen_full > 0 else np.nan

    return out

def fit_polynomial(y: ArrayLike, k: int = 1) -> float:
    """
    Goodness of a polynomial fit to a time series

    Usually kind of a stupid thing to do with a time series, but it's sometimes
    somehow informative for time series with large trends.

    Parameters
    -----------
    y : array-like
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
    out = np.sqrt(np.mean((y - f)**2)) # RMS error of fit

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
        logger.warning(f"Time series (N = {len(y)}) too short to compute differences.")
        return np.nan
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
            logger.warning(f"This time series (N = {N}) is too short for stat_av({what_type},'{extra_param}')")
            return np.nan
    else:
        raise ValueError(f"Error evaluating stat_av of type '{what_type}', please select either 'seg' or 'len'")

    s = np.std(y, ddof=1)  # should be 1 (for a z-scored time-series input)
    sdav = np.std(M, ddof=1)
    out = sdav / s

    return float(out)

def sliding_window(y: ArrayLike, window_stat: str = 'mean', across_win_stat: str = 'std',
                 num_seg: int = 5, inc_move: int = 2) -> dict:
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
        logger.warning(f"Time-series of length {len(y)} is too short for {num_seg} windows")
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
    # helper function to convert a step index (stepInd) to a range of indices corresponding to that window
    start_idx = (step_ind) * inc
    end_idx = (step_ind) * inc + win_length
    
    return np.arange(start_idx, end_idx).astype(int)

def _kendall_tie_adj(r: np.ndarray) -> np.ndarray:
    # Tie adjustments accompanying the midranks, as returned by MATLAB's tiedrank:
    # [sum(t*(t-1))/2, sum(t*(t-1)*(t-2)), sum(t*(t-1)*(2*t+5))] over tied-group sizes t
    _, counts = np.unique(r, return_counts=True)
    t = counts[counts > 1].astype(float)
    return np.array([np.sum(t * (t - 1)) / 2,
                     np.sum(t * (t - 1) * (t - 2)),
                     np.sum(t * (t - 1) * (2 * t + 5))])

def _kendall(x: np.ndarray, y: np.ndarray) -> tuple:
    # Kendall's tau-b and its two-tailed p-value, following MATLAB's corr(...,'type','Kendall'):
    # the p-value is exact (permutation distribution of K) for small samples and a
    # continuity-corrected normal approximation otherwise
    n = len(x)
    xrank, yrank = rankdata(x), rankdata(y)
    xadj, yadj = _kendall_tie_adj(xrank), _kendall_tie_adj(yrank)
    n2const = n * (n - 1) // 2

    K = int(round(np.sum(np.sign(xrank[:, None] - xrank[None, :])
                         * np.sign(yrank[:, None] - yrank[None, :])) / 2))

    denom = np.sqrt((n2const - xadj[0]) * (n2const - yadj[0]))
    tau = K / denom if denom > 0 else np.nan

    ties = (xadj[0] > 0) or (yadj[0] > 0)
    if (xadj[0] == n2const) or (yadj[0] == n2const):
        return tau, np.nan

    exact = (n < 10) if ties else (n < 50)
    if exact:
        nfact = 1.0
        for i in range(2, n + 1):
            nfact *= i
        if ties:
            # With ties, take permutations of the midranks
            yperms = np.array(list(permutations(yrank)))
            kperm = np.zeros(yperms.shape[0])
            for w in range(n - 1):
                U = np.sign(xrank[w] - xrank[w+1:])
                V = np.sign(yperms[:, [w]] - yperms[:, w+1:])
                kperm += V @ U
            freq = np.bincount(np.rint(kperm).astype(int) + n2const,
                               minlength=2*n2const+1).astype(float)[:-1]
        else:
            # No ties, use recursion to get the cumulative distribution of the number, C,
            # of positive (xi-xj)*(yi-yj), i<j. K = #pos-#neg = C-Q, and C+Q = n(n-1)/2
            freq = np.array([1.0, 1.0])
            for i in range(3, n + 1):
                freq = np.convolve(freq, np.ones(i))
            interleaved = np.zeros(2 * freq.size)
            interleaved[::2] = freq  # bins at integers, starting at -n2const
            freq = interleaved[:-1]
        # Use twice the smaller of the tail area above and below the observed value
        cum = np.cumsum(freq)
        rcum = nfact - np.concatenate(([0.0], cum[:-1]))
        tail_prob = np.minimum(2 * np.minimum(cum, rcum) / nfact, 1)  # don't count the center bin twice
        pval = tail_prob[K + n2const]
    else:
        if ties:
            std_k = np.sqrt(n2const * (2*n + 5) / 9
                            + xadj[0] * yadj[0] / n2const
                            + xadj[1] * yadj[1] / (18 * n2const * (n - 2))
                            - (xadj[2] + yadj[2]) / 18)
        else:
            std_k = np.sqrt(n * (n - 1) * (2*n + 5) / 18)
        pval = min(1, 2 * norm.cdf(-(abs(K) - 1) / std_k))

    return tau, pval

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    # Pearson's linear correlation and its two-tailed p-value (NaN for constant input,
    # matching MATLAB's corr)
    if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, pval = pearsonr(x, y)

    return r, pval

def ramping_windows(y: ArrayLike, num_seg: int = 10) -> dict:
    """
    Monotonic trend ('ramping') in windowed statistics.

    Splits the time series into ``num_seg`` non-overlapping segments, computes the
    mean, variance, skewness, kurtosis, and lag-1 autocorrelation (AC1) within each
    segment, and quantifies whether each of these quantities trends monotonically
    across the segments (e.g., a variance that ramps up steadily across the series,
    rather than merely fluctuating).

    Existing hctsa stationarity operations (:func:`sliding_window`,
    :func:`drifting_mean`, :func:`stat_av`, :func:`local_distributions`) summarize
    the *spread* of windowed statistics (std, range, entropy) -- order-agnostic
    measures that don't distinguish a monotonic ramp from random fluctuation of the
    same magnitude. This operation targets that gap directly.

    Kendall's rank correlation tau, and Pearson's linear correlation r (each with its
    p-value), are computed between segment index and each windowed statistic.
    Kendall's tau is scale-invariant and detects any monotonic trend, not just a
    linear one -- matching "ramping" more directly than a slope would, and more
    robust to outlying segments. Pearson's r is included alongside it as a
    complementary, effect-size-like measure of specifically *linear* ramping (r^2 is
    the fraction of across-segment variance explained by a linear trend); expect the
    two to agree closely for a clean linear ramp and diverge for a
    monotonic-but-nonlinear one (e.g. a ramp that plateaus). Skewness/kurtosis are
    standardized in the usual sense (normalized by std^3/std^4 respectively, rather
    than :func:`~pyhctsa.operations.distribution.moments`' convention of normalizing
    by std^1 regardless of moment order) so that a shape trend isn't conflated with
    the (separately tracked) scale trend in the segment variance.

    asymAC1 is a nonlinear, asymmetric variant of the lag-1 autocorrelation:
    ``mean(x_t * x_{t+1} * (x_{t+1} - x_t))``, with x z-scored *within* each segment
    (unlike the other statistics above, this one needs the explicit per-segment
    z-scoring, since it's neither scale-invariant like AC1/skewness/kurtosis nor a raw
    moment tracked deliberately like mean/variance). This is the difference between
    the statistic computed forwards (``mean(x_t * x_{t+1}^2)``) and on the
    time-reversed segment (``mean(x_{t+1} * x_t^2)``): swapping x_t <-> x_{t+1}
    negates the expression, so it is manifestly antisymmetric under time reversal and
    hence zero in expectation for any time-reversible process (same spirit as
    :func:`~pyhctsa.operations.correlation.trev`'s third-moment reversibility
    statistic, computed densely at a single lag over the whole series rather than
    trended across segments). A ramp in asymAC1 therefore flags a trend specifically
    in the series' local time-asymmetry/nonlinearity, distinct from a trend in any of
    the other (symmetric) segment statistics above.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_seg : int, optional
        The number of non-overlapping segments to divide the time series into.
        Non-overlapping segments are used deliberately (rather than
        :func:`sliding_window`'s overlapping windows): overlap between windows would
        induce artificial serial correlation between adjacent window-statistics,
        which would inflate the apparent monotonic trend independent of any real
        ramping in the data. Default is 10.

    Returns
    -------
    dict
        Kendall's tau and Pearson's r (each with its p-value) between segment index
        and each windowed statistic. Returns NaN if the time series is too short for
        the requested number of segments.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    num_seg = int(num_seg)

    min_num_seg = 5 # need enough segments for a meaningful trend statistic
    min_seg_length = 20 # heuristic minimum for meaningful skewness/kurtosis/AC1 estimates

    if num_seg < min_num_seg:
        raise ValueError(f"num_seg = {num_seg} is too few segments for a meaningful "
                         f"trend statistic (need >= {min_num_seg})")

    seg_length = N // num_seg
    if seg_length < min_seg_length:
        logger.warning(f"Time series (N = {N}) too short for {num_seg} segments of a meaningful length")
        return np.nan

    # ------------------------------------------------------------------------------
    # Segment the time series (non-overlapping, discarding any remainder)
    # ------------------------------------------------------------------------------
    z = y[:seg_length * num_seg].reshape(num_seg, seg_length) # num_seg x seg_length

    # ------------------------------------------------------------------------------
    # Within-segment statistics
    # ------------------------------------------------------------------------------
    seg_mean = np.mean(z, axis=1)
    seg_var = np.var(z, axis=1, ddof=1)
    # Standardized (not moments-style raw/std) skewness and kurtosis: keeps the
    # shape-trend signal from being conflated with the (separately tracked)
    # scale-trend signal in seg_var, since moments normalizes by std^1 regardless of
    # moment order rather than std^3/std^4.
    seg_skew = skew(z, axis=1)
    seg_kurt = kurtosis(z, axis=1, fisher=False)
    seg_ac1 = np.zeros(num_seg)
    seg_asym_ac1 = np.zeros(num_seg)
    for i in range(num_seg):
        seg_ac1[i] = autocorr(z[i, :], 1, 'Fourier')[0]
        zseg = (z[i, :] - np.mean(z[i, :])) / np.std(z[i, :], ddof=1) # z-scored *within* this segment
        seg_asym_ac1[i] = np.mean(zseg[:-1] * zseg[1:] * (zseg[1:] - zseg[:-1]))

    # ------------------------------------------------------------------------------
    # Kendall's tau and Pearson's r (each with p-value) against segment index
    # ------------------------------------------------------------------------------
    seg_idx = np.arange(1, num_seg + 1, dtype=float)

    out = {}
    out['mean_tau'], out['mean_p'] = _kendall(seg_idx, seg_mean)
    out['var_tau'], out['var_p'] = _kendall(seg_idx, seg_var)
    out['skew_tau'], out['skew_p'] = _kendall(seg_idx, seg_skew)
    out['kurt_tau'], out['kurt_p'] = _kendall(seg_idx, seg_kurt)
    out['ac1_tau'], out['ac1_p'] = _kendall(seg_idx, seg_ac1)
    out['asymac1_tau'], out['asymac1_p'] = _kendall(seg_idx, seg_asym_ac1)

    out['mean_pearson_r'], out['mean_pearson_p'] = _pearson(seg_idx, seg_mean)
    out['var_pearson_r'], out['var_pearson_p'] = _pearson(seg_idx, seg_var)
    out['skew_pearson_r'], out['skew_pearson_p'] = _pearson(seg_idx, seg_skew)
    out['kurt_pearson_r'], out['kurt_pearson_p'] = _pearson(seg_idx, seg_kurt)
    out['ac1_pearson_r'], out['ac1_pearson_p'] = _pearson(seg_idx, seg_ac1)
    out['asymac1_pearson_r'], out['asymac1_pearson_p'] = _pearson(seg_idx, seg_asym_ac1)

    return out

def slow_feature_analysis(y: ArrayLike, num_windows: int = 20) -> dict:
    """
    Slow feature analysis of windowed statistics.

    Splits the time series into ``num_windows`` non-overlapping segments (same
    segmentation and per-segment statistics as :func:`ramping_windows`: mean,
    variance, skewness, and lag-1 autocorrelation, forming a ``num_windows`` x 4
    matrix), then applies Slow Feature Analysis (SFA) to find the linear combination
    of these four statistics that varies as *slowly* as possible across the sequence
    of windows -- i.e., minimizes the variance of its own increments, subject to unit
    variance. This is a fundamentally different criterion to :func:`ramping_windows`'
    per-statistic monotonic-trend tests: SFA is multivariate (can find a slow
    combination spread across mean/variance/skewness/AC1 that no single one of them
    shows individually) and detects any slow, low-frequency evolution, not just a
    monotonic ramp (e.g., a slow rise-then-fall hump across the series, invisible to
    Kendall's tau/Pearson's r, still registers as "slow" here). Validated on
    Empirical1000 to be essentially uncorrelated (max \\|r\\| ~ 0.14) with all
    ``ramping_windows`` (numSeg = 10) trend fields.

    The companion comparison to ordinary PCA addresses a different question: PCA
    finds the combination of the four statistics with *maximal variance* across
    windows, which need not be the slowest one -- a large-amplitude but choppy/noisy
    statistic can dominate variance while a small-amplitude but smooth, genuinely
    slow drift is buried underneath it and missed by PCA. ``pc1VarFrac`` and
    ``slowPCA1corr`` quantify this: how concentrated the variance is in a single PCA
    direction, and whether the slow direction found by SFA coincides with that
    high-variance PCA direction (high overlap) or is a separate, lower-variance mode
    that ordinary variance-based analysis would miss (low overlap).

    Let ``z(t)`` be the ``num_windows`` x 4 matrix of [mean, variance, skewness, AC1]
    computed per window, whitened (centered, then linearly transformed to unit
    covariance). SFA finds the orthogonal directions ``u_i`` that minimize
    ``var(diff(z*u_i))``, i.e. the "slowness" eigenvalues ``eta_i = var(diff(z*u_i))``
    of the covariance of the whitened derivative signal (ascending: ``eta_1`` is the
    slowest direction). For reference, i.i.d. (white-noise) windows give ``eta ~ 2``
    on average; substantially smaller values indicate genuinely slow (smooth,
    low-frequency) structure in some combination of the four per-window statistics.

    References
    ----------
    Wiskott, L. & Sejnowski, T.J. "Slow feature analysis: unsupervised learning of
    invariances." Neural Computation 14(4), 715-770 (2002).

    Parameters
    ----------
    y : array-like
        The input time series.
    num_windows : int, optional
        The number of non-overlapping segments to divide the time series into.
        Non-overlapping segments are used for the same reason as
        :func:`ramping_windows`: overlap would induce artificial serial correlation
        between adjacent window statistics, which would make the derivative-based
        slowness measure spuriously small regardless of any real slow structure in
        the data. 20 was chosen (rather than :func:`ramping_windows`' default of 10)
        because SFA needs enough windows to estimate the underlying 4x4 covariance
        matrices (of the statistics, and of their increments) reasonably reliably --
        at ``num_windows = 10`` the null-distribution spread of the slowness
        eigenvalues is considerably wider, making individual values a noisier signal.
        Default is 20.

    Returns
    -------
    dict
        - ``eta1``: the smallest (slowest) SFA eigenvalue.
        - ``etaEnd``: the largest (fastest/noisiest) SFA eigenvalue.
        - ``etaStd``: the standard deviation of all four SFA eigenvalues (spread of
          the slowness spectrum).
        - ``pc1VarFrac``: the fraction of total variance (across the four statistics)
          explained by the leading PCA component.
        - ``slowPCA1corr``: the absolute correlation between the slowest SFA
          component's scores and the leading PCA component's scores -- near 1 means
          the slow direction is simply the dominant (highest-variance) direction PCA
          would already find; near 0 means SFA has isolated a genuinely separate,
          low-variance slow mode.

        Returns NaN if the time series is too short for the requested number of
        windows, or if fewer than two directions survive the whitening threshold.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    num_windows = int(num_windows)

    min_num_windows = 10 # need enough windows to estimate the 4x4 covariance matrices reliably
    min_window_length = 20 # heuristic minimum for meaningful skewness/AC1 estimates

    if num_windows < min_num_windows:
        raise ValueError(f"num_windows = {num_windows} is too few for a reliable slow "
                         f"feature analysis (need >= {min_num_windows})")

    win_length = N // num_windows
    if win_length < min_window_length:
        logger.warning(f"Time series (N = {N}) too short for {num_windows} windows of a meaningful length")
        return np.nan

    # ------------------------------------------------------------------------------
    # Segment the time series (non-overlapping, discarding any remainder)
    # ------------------------------------------------------------------------------
    z = y[:win_length * num_windows].reshape(num_windows, win_length) # num_windows x win_length

    # ------------------------------------------------------------------------------
    # Per-window statistics: mean, variance, skewness, AC1 (same as ramping_windows)
    # ------------------------------------------------------------------------------
    win_mean = np.mean(z, axis=1)
    win_var = np.var(z, axis=1, ddof=1)
    win_skew = skew(z, axis=1)
    win_ac1 = np.zeros(num_windows)
    for i in range(num_windows):
        win_ac1[i] = autocorr(z[i, :], 1, 'Fourier')[0]
    X = np.column_stack((win_mean, win_var, win_skew, win_ac1)) # num_windows x 4

    # ------------------------------------------------------------------------------
    # PCA (variance-maximizing directions) and SFA (slowness-minimizing directions)
    # ------------------------------------------------------------------------------
    Xc = X - np.mean(X, axis=0)
    Cx = np.cov(Xc, rowvar=False) # 4 x 4
    if not np.all(np.isfinite(Cx)):
        # a degenerate (e.g. constant) window leaves its skewness/AC1 -- and hence
        # the covariance -- undefined
        return np.nan

    eig_vals, Vp = np.linalg.eigh(Cx)
    ord_ = np.argsort(-eig_vals, kind='stable')
    pca_eigs = eig_vals[ord_]
    Vp = Vp[:, ord_]
    pc_scores = Xc @ Vp # num_windows x 4, PC1 = pc_scores[:, 0]

    # Whitening (symmetric/ZCA, avoids an arbitrary rotation among near-degenerate
    # directions). Directions with near-zero variance relative to the leading one
    # (e.g. a per-window statistic that barely varies across windows) are dropped
    # rather than whitened: full whitening would divide by their near-zero std and
    # amplify what is essentially estimation noise into a spuriously enormous
    # "fast" eigenvalue. pca_eigs[0] (the leading, largest eigenvalue) is never
    # itself dropped by this relative threshold.
    rel_floor = 1e-2
    keep = pca_eigs > rel_floor * pca_eigs[0]
    if np.sum(keep) < 2:
        return np.nan
    Vk = Vp[:, keep]
    Zw = Xc @ Vk / np.sqrt(pca_eigs[keep]) # num_windows x sum(keep), approx unit covariance

    dZ = np.diff(Zw, axis=0) # (num_windows-1) x sum(keep), the whitened derivative signal
    Cd = np.cov(dZ, rowvar=False)
    eta_vals, Us = np.linalg.eigh(Cd)
    ord2 = np.argsort(eta_vals, kind='stable') # slowness eigenvalues, ascending = slowest first
    etas = eta_vals[ord2]
    Us = Us[:, ord2]
    slow_scores = Zw @ Us # num_windows x sum(keep), slowest component = slow_scores[:, 0]

    # ------------------------------------------------------------------------------
    # Output statistics
    # ------------------------------------------------------------------------------
    out = {}
    out['eta1'] = etas[0]
    out['etaEnd'] = etas[-1]
    out['etaStd'] = np.std(etas, ddof=1)

    out['pc1VarFrac'] = pca_eigs[0] / np.sum(pca_eigs)
    out['slowPCA1corr'] = abs(_pearson(slow_scores[:, 0], pc_scores[:, 0])[0])

    return out

def _cum_sum_bridge_stats(p: np.ndarray) -> Union[dict, float]:
    p = np.asarray(p, dtype=float).ravel()
    Np = len(p)
    if Np < 20:
        return np.nan

    t = np.arange(1, Np + 1, dtype=float)
    y_c = np.cumsum(p)

    out = {}
    out['meanYC'] = np.mean(y_c)
    coeffs_ols = polyfit(t, y_c, 1)
    out['gradient'] = coeffs_ols[0] # ~ std(y_c) too (r > 0.99 empirically); kept as the interpretable one
    out['intercept'] = coeffs_ols[1]
    resid_ols = y_c - np.polyval(coeffs_ols, t)

    out['meanYC12'] = np.mean(y_c[:Np // 2])
    out['meanYC22'] = np.mean(y_c[Np // 2:])

    bridge = y_c - (t / Np) * y_c[-1]
    scale_factor = np.std(p, ddof=1) * np.sqrt(Np)
    if scale_factor > 0:
        out['maxBridge'] = np.max(np.abs(bridge)) / scale_factor
    else:
        out['maxBridge'] = np.nan
    idx_max = int(np.argmax(np.abs(bridge))) + 1
    out['posMaxBridge'] = idx_max / Np # where the largest deviation from stationarity occurs
    out['stdBridge'] = np.std(bridge, ddof=1)

    rob_coeffs, rob_stats = robustfit(t, y_c)
    robust_gradient = rob_coeffs[1] # not output directly: r > 0.98 with out['gradient']
    rob_resid = y_c - (rob_coeffs[0] + rob_coeffs[1] * t)
    if rob_stats['se'][1] > 0:
        out['gradientDiffSE'] = (out['gradient'] - robust_gradient) / rob_stats['se'][1]
    else:
        out['gradientDiffSE'] = np.nan
    if np.std(rob_resid, ddof=1) > 0:
        out['residStdRatio'] = np.std(resid_ols, ddof=1) / np.std(rob_resid, ddof=1)
    else:
        out['residStdRatio'] = np.nan

    var_p = np.var(p, ddof=1)
    if var_p > 0 and Np > 21:
        t_interior = t[:-1]
        null_var = var_p * t_interior * (Np - t_interior) / Np
        std_resid = bridge[:-1] ** 2 / null_var # ~chi-square(1), mean 1, under the null
        log_std_resid = np.log(std_resid + np.finfo(float).eps) # chi-square(1) is heavy-tailed; log stabilizes the trend estimate
        out['varRatioTrend'] = _kendall(t_interior, log_std_resid)[0]
    else:
        out['varRatioTrend'] = np.nan

    return out

def drifting_mean_cusum(y: ArrayLike) -> dict:
    """
    Parameter-free CUSUM test for a drifting mean.

    :func:`stat_av` and :func:`drifting_mean` both test for a drifting mean by
    splitting the time series into segments -- requiring a choice of segment length
    or number of segments. This is the analogous test with no free parameter: it
    forms ``cumsum(y)`` directly (at full resolution) and computes CUSUM/bridge
    statistics on it, including a comparison between an ordinary least-squares and a
    robust linear fit to flag whether any apparent drift is outlier-driven or a
    genuine trend.

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored).

    Returns
    -------
    dict
        - ``maxBridge``: the largest absolute deviation of the CUSUM 'bridge' (the
          cumsum relative to its endpoint-to-endpoint line), scaled by
          ``std(y)*sqrt(N)``.
        - ``posMaxBridge``: where in the series (as a fraction of its length) that
          largest deviation from stationarity occurs.
        - ``gradientDiffSE``: the difference between the ordinary least-squares and
          robust (bisquare) gradients of the cumsum, in units of the robust fit's
          standard error.
        - ``residStdRatio``: the ratio of the OLS residual standard deviation to the
          robust residual standard deviation.
        - ``varRatioTrend``: Kendall's tau between time and the log of the squared
          bridge, normalized by its variance under the null -- a trend in the local
          variance of the CUSUM.

        Returns NaN if the time series is shorter than 20 samples.
    """
    y = np.asarray(y, dtype=float).ravel()
    out = _cum_sum_bridge_stats(y)
    if isinstance(out, dict):
        for k in ('meanYC', 'gradient', 'intercept', 'meanYC12', 'meanYC22', 'stdBridge'):
            del out[k]

    return out
