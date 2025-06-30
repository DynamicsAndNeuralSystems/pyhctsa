
import numpy as np
from numpy.typing import ArrayLike 
from typing import Union
from scipy.stats import gaussian_kde, kurtosis, skew, expon
from scipy.stats import mode as smode
from ..Utilities.utils import pointOfCrossing, binpicker, ZScore, signChange
from loguru import logger
from statsmodels.tsa.stattools import pacf
from scipy.optimize import curve_fit
from ..Operations.Information import FirstMin
from scipy.linalg import LinAlgError
from ..Toolboxes.c22.periodicity_wang_wrapper import periodicity_wang

def PeriodicityWang(y : ArrayLike) -> dict:
    """
    Periodicity extraction measure of Wang et al. (2007).

    Implements an idea based on the periodicity extraction measure proposed in the paper
    "Structure-based Statistical Features and Multivariate Time Series Clustering"
    by X. Wang, A. Wirth, and L. Wang (2007).

    The function:
    1. Detrends the time series using a three-knot cubic regression spline
    2. Computes autocorrelations up to one third of the length of the time series
    3. Finds the first peak in the autocorrelation function satisfying certain conditions

    While the original paper used a single threshold of 0.01, this implementation tests
    multiple thresholds: 0, 0.01 (original paper threshold), 0.1, 0.2, 1/sqrt(N), 5/sqrt(N), 
    10/sqrt(N), where N is the length of the time series.

    Parameters
    ----------
    y : array-like
        The input time series

    Returns
    -------
    dict
        Dictionary containing periodicity measures for each threshold
    """
    y = np.asarray(y)
    return periodicity_wang(y)

def CompareMinAMI(y : ArrayLike, binMethod : str = 'std1', numBins : int = 10) -> dict:
    """
    Assess the variability in the first minimum of automutual information (AMI) across binning strategies.

    This function computes the first minimum of the automutual information function for a time series
    using various histogram binning strategies and numbers of bins. It summarizes how the location
    of the first minimum varies across these different coarse-grainings.

    Parameters
    ----------
    y : array-like
        The input time series.
    binMethod : str, optional
        The method for estimating mutual information (passed to `HistogramAMI`). Default is 'std1'.
    numBins : int or array-like, optional
        The number of bins (or list of bin counts) to use for AMI estimation. Default is 10.

    Returns
    -------
    dict
        Dictionary containing statistics on the set of first minimums of the automutual information function.
    """
    y = np.asarray(y)
    N = len(y)
    # Range of time lags to consider
    tauRange = np.arange(0, int(np.ceil(N/2))+1)
    numTaus = len(tauRange)

    # range of bin numbers to consider
    if isinstance(numBins, int):
        numBins = [numBins]
    
    numBinsRange = len(numBins)
    amiMins = np.zeros(numBinsRange)

    # Calculate automutual information
    for i in range(numBinsRange):  # vary over number of bins in histogram
        amis = np.zeros(numTaus)
        for j in range(numTaus):  # vary over time lags, tau
            amis[j] = HistogramAMI(y, tauRange[j], binMethod, numBins[i])
            if (j > 1) and ((amis[j] - amis[j-1]) * (amis[j-1] - amis[j-2]) < 0):
                amiMins[i] = tauRange[j-1]
                break
        if amiMins[i] == 0:
            amiMins[i] = tauRange[-1]
    # basic statistics
    out = {}
    out['min'] = np.min(amiMins)
    out['max'] = np.max(amiMins)
    out['range'] = np.ptp(amiMins)
    out['median'] = np.median(amiMins)
    out['mean'] = np.mean(amiMins)
    out['std'] = np.std(amiMins, ddof=1) # will return NaN for single values instead of 0
    out['nunique'] = len(np.unique(amiMins))
    out['mode'], out['modef'] = smode(amiMins)
    out['modef'] = out['modef']/numBinsRange

    # converged value? 
    out['conv4'] = np.mean(amiMins[-5:])

    # look for peaks (local maxima)
    # % local maxima above 1*std from mean
    # inspired by curious result of periodic maxima for periodic signal with
    # bin size... ('quantiles', [2:80])
    diff_ami_mins = np.diff(amiMins[:-1])
    positive_diff_indices = np.where(diff_ami_mins > 0)[0]
    sign_change_indices = signChange(diff_ami_mins, 1)

    # Find the intersection of positive_diff_indices and sign_change_indices
    loc_extr = np.intersect1d(positive_diff_indices, sign_change_indices) + 1
    above_threshold_indices = np.where(amiMins > out['mean'] + out['std'])[0]
    big_loc_extr = np.intersect1d(above_threshold_indices, loc_extr)

    # Count the number of elements in big_loc_extr
    out['nlocmax'] = len(big_loc_extr)

    return out

def HistogramAMI(y : ArrayLike, tau : Union[str, int, ArrayLike] = 1, meth : str = 'even', numBins : int = 10) -> dict:
    """
    The automutual information of the distribution using histograms.

    Computes the automutual information between a time series and its time-delayed version
    using different methods for binning the data.

    Parameters
    ----------
    y : array-like
        The input time series
    tau : int, list, or str, optional
        The time-lag(s) (default: 1)
        Can be an integer time lag, list of time lags, or 'ac'/'tau' to use
        first zero-crossing of autocorrelation function
    meth : str, optional
        The method for binning data (default: 'even'):
        - 'even': evenly-spaced bins through the range
        - 'std1': bins extending to ±1 standard deviation from mean
        - 'std2': bins extending to ±2 standard deviations from mean
        - 'quantiles': equiprobable bins using quantiles
    numBins : int, optional
        The number of bins to use (default: 10)

    Returns
    -------
    Union[float, dict]
        If single tau: The automutual information value
        If multiple taus: Dictionary of automutual information values
    """
    # Use first zero crossing of the ACF as the time lag
    y = np.asarray(y)
    if isinstance(tau, str) and tau in ['ac', 'tau']:
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Bins for the data
    # same for both -- assume same distribution (true for stationary processes, or small lags)
    if meth == 'even':
        b = np.linspace(np.min(y), np.max(y), numBins + 1)
        # Add increment buffer to ensure all points are included
        inc = 0.1
        b[0] -= inc
        b[-1] += inc
    elif meth == 'std1': # bins out to +/- 1 std
        b = np.linspace(-1, 1, numBins + 1)
        if np.min(y) < -1:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 1:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'std2': # bins out to +/- 2 std
        b = np.linspace(-2, 2, numBins + 1)
        if np.min(y) < -2:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 2:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'quantiles': # use quantiles with ~equal number in each bin
        b = np.quantile(y, np.linspace(0, 1, numBins + 1), method='hazen')
        b[0] -= 0.1
        b[-1] += 0.1
    else:
        raise ValueError(f"Unknown method '{meth}'")
    
    # Sometimes bins can be added (e.g., with std1 and std2), so need to redefine numBins
    numBins = len(b) - 1

    # Form the time-delay vectors y1 and y2
    if not isinstance(tau, (list, np.ndarray)):
        # if only single time delay as integer, make into a one element list
        tau = [tau]

    amis = np.zeros(len(tau))
    for i, t in enumerate(tau):
        if t == 0:
            # for tau = 0, y1 and y2 are identical to y
            y1 = y2 = y
        else:
            y1 = y[:-t]
            y2 = y[t:]
        # Joint distribution of y1 and y2
        pij, _, _ = np.histogram2d(y1, y2, bins=(b, b))
        pij = pij[:numBins, :numBins]  # joint
        pij = pij / np.sum(pij)  # normalize
        pi = np.sum(pij, axis=1)  # marginal
        pj = np.sum(pij, axis=0)  # other marginal

        pii = np.tile(pi, (numBins, 1)).T
        pjj = np.tile(pj, (numBins, 1))

        r = pij > 0  # Defining the range in this way, we set log(0) = 0
        amis[i] = np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}

def StickAngles(y : ArrayLike) -> dict:
    """
    Analysis of the line-of-sight angles between time series data pts. 

    Line-of-sight angles between time-series pts treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Parameters:
    -----------
    y : array-like
        The input time series

    Returns:
    --------
    dict
        A dictionary containing various statistics on the obtained sequence of angles.
    """
    y = np.asarray(y)
    # Split the time series into positive and negative parts
    ix = [np.where(y >= 0)[0], np.where(y < 0)[0]]
    n = [len(ix[0]), len(ix[1])]

    # Compute the stick angles
    angles = [[], []]
    for j in range(2):
        if n[j] > 1:
            diff_y = np.diff(y[ix[j]])
            diff_x = np.diff(ix[j])
            angles[j] = np.arctan(diff_y /diff_x)
    allAngles = np.concatenate(angles)

    # Initialise output dictionary
    out = {}
    out['std_p'] = np.nanstd(angles[0], ddof=1) 
    out['mean_p'] = np.nanmean(angles[0]) 
    out['median_p'] = np.nanmedian(angles[0])

    out['std_n'] = np.nanstd(angles[1], ddof=1)
    out['mean_n'] = np.nanmean(angles[1])
    out['median_n'] = np.nanmedian(angles[1])

    out['std'] = np.nanstd(allAngles, ddof=1)
    out['mean'] = np.nanmean(allAngles)
    out['median'] = np.nanmedian(allAngles)

    # difference between positive and negative angles
    # return difference in densities
    
    ksx = np.linspace(np.min(allAngles), np.max(allAngles), 200)
    out['pnsumabsdiff'] = np.nan
    if (len(angles[0]) > 0 and len(angles[1]) > 0 and
        np.var(angles[0]) > 1e-10 and np.var(angles[1]) > 1e-10):
        try:
            ksx = np.linspace(np.min(allAngles), np.max(allAngles), 200)
            # Calculate the Kernel Density Estimate (KDE) for the first angle distribution.
            kde1 = gaussian_kde(angles[0], bw_method='scott')
            ksy1 = kde1(ksx)

            # Calculate the KDE for the second angle distribution.
            kde2 = gaussian_kde(angles[1], bw_method='scott')
            ksy2 = kde2(ksx)

            # If the KDEs are calculated successfully, compute the sum of the absolute
            out['pnsumabsdiff'] = np.sum(np.abs(ksy1 - ksy2))
        except LinAlgError:
            pass
    
    # # how symmetric is the distribution of angles?
    out['symks_p'] = np.nan
    out['ratmean_p'] = np.nan

    if len(angles[0]) > 0 and np.var(angles[0]) > 1e-10:
        try:
            maxdev = np.max(np.abs(angles[0]))
            kde = gaussian_kde(angles[0], bw_method='scott')
            ksy1 = kde(np.linspace(-maxdev, maxdev, 201))
            out['symks_p'] = np.sum(np.abs(ksy1[:100] - ksy1[101:][::-1]))
            out['ratmean_p'] = np.mean(angles[0][angles[0] > 0])/np.mean(angles[0][angles[0] < 0])
        except LinAlgError:
            pass
    
    out['symks_n'] = np.nan
    out['ratmean_n'] = np.nan
    if len(angles[1]) > 0 and np.var(angles[1]) > 1e-10:
        try:
            maxdev = np.max(np.abs(angles[1]))
            kde = gaussian_kde(angles[1], bw_method='scott')
            ksy2 = kde(np.linspace(-maxdev, maxdev, 201))
            out['symks_n'] = np.sum(np.abs(ksy2[:100] - ksy2[101:][::-1]))
            out['ratmean_n'] = np.mean(angles[1][angles[1] > 0])/np.mean(angles[1][angles[1] < 0])
        except LinAlgError:
            pass
    
    # z-score
    zangles = []
    zangles.append(ZScore(angles[0]))
    zangles.append(ZScore(angles[1]))
    zallAngles = ZScore(allAngles)

    # how stationary are the angle sets?

    # there are positive angles
    if len(zangles[0]) > 0:
        # StatAv2
        out['statav2_p_m'], out['statav2_p_s'] = _SUB_statav(zangles[0], 2)
        # StatAv3
        out['statav3_p_m'], out['statav3_p_s'] = _SUB_statav(zangles[0], 3)
        # StatAv4
        out['statav4_p_m'], out['statav4_p_s'] = _SUB_statav(zangles[0], 4)
        # StatAv5
        out['statav5_p_m'], out['statav5_p_s'] = _SUB_statav(zangles[0], 5)
    else:
        out['statav2_p_m'], out['statav2_p_s'] = np.nan, np.nan
        out['statav3_p_m'], out['statav3_p_s'] = np.nan, np.nan
        out['statav4_p_m'], out['statav4_p_s'] = np.nan, np.nan
        out['statav5_p_m'], out['statav5_p_s'] = np.nan, np.nan
    
    # there are negative angles
    if len(zangles[1]) > 0:
        # StatAv2
        out['statav2_n_m'], out['statav2_n_s'] = _SUB_statav(zangles[1], 2)
        # StatAv3
        out['statav3_n_m'], out['statav3_n_s'] = _SUB_statav(zangles[1], 3)
        # StatAv4
        out['statav4_n_m'], out['statav4_n_s'] = _SUB_statav(zangles[1], 4)
        # StatAv5
        out['statav5_n_m'], out['statav5_n_s'] = _SUB_statav(zangles[1], 5)
    else:
        out['statav2_n_m'], out['statav2_n_s'] = np.nan, np.nan
        out['statav3_n_m'], out['statav3_n_s'] = np.nan, np.nan
        out['statav4_n_m'], out['statav4_n_s'] = np.nan, np.nan
        out['statav5_n_m'], out['statav5_n_s'] = np.nan, np.nan
    
    # All angles
    
    # StatAv2
    out['statav2_all_m'], out['statav2_all_s'] = _SUB_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], out['statav3_all_s'] = _SUB_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], out['statav4_all_s'] = _SUB_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = _SUB_statav(zallAngles, 5)
    
    # correlations? 
    if len(zangles[0]) > 0:
        out['tau_p'] = FirstCrossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = AutoCorr(zangles[0], 1, 'Fourier')[0]
        out['ac2_p'] = AutoCorr(zangles[0], 2, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan
        out['ac2_p'] = np.nan
    
    if len(zangles[1]) > 0:
        out['tau_n'] = FirstCrossing(zangles[1], 'ac', 0, 'continuous')
        out['ac1_n'] = AutoCorr(zangles[1], 1, 'Fourier')[0]
        out['ac2_n'] = AutoCorr(zangles[1], 2, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan
        out['ac2_n'] = np.nan
    
    out['tau_all'] = FirstCrossing(zallAngles, 'ac', 0, 'continuous')
    out['ac1_all'] = AutoCorr(zallAngles, 1, 'Fourier')[0]
    out['ac2_all'] = AutoCorr(zallAngles, 2, 'Fourier')[0]


    # What does the distribution look like? 
    
    # Some quantiles and moments
    if len(zangles[0]) > 0:
        out['q1_p'] = np.quantile(zangles[0], 0.01, method='hazen')
        out['q10_p'] = np.quantile(zangles[0], 0.1, method='hazen')
        out['q90_p'] = np.quantile(zangles[0], 0.9, method='hazen')
        out['q99_p'] = np.quantile(zangles[0], 0.99, method='hazen')
        out['skewness_p'] = skew(angles[0])
        out['kurtosis_p'] = kurtosis(angles[0], fisher=False)
    else:
        out['q1_p'], out['q10_p'], out['q90_p'], out['q99_p'], \
            out['skewness_p'], out['kurtosis_p'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    if len(zangles[1]) > 0:
        out['q1_n'] = np.quantile(zangles[1], 0.01, method='hazen')
        out['q10_n'] = np.quantile(zangles[1], 0.1, method='hazen')
        out['q90_n'] = np.quantile(zangles[1], 0.9, method='hazen')
        out['q99_n'] = np.quantile(zangles[1], 0.99, method='hazen')
        out['skewness_n'] = skew(angles[1])
        out['kurtosis_n'] = kurtosis(angles[1], fisher=False)
    else:
        out['q1_n'], out['q10_n'], out['q90_n'], out['q99_n'], \
            out['skewness_n'], out['kurtosis_n'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    F_quantz = lambda x : np.quantile(zallAngles, x, method='hazen')
    out['q1_all'] = F_quantz(0.01)
    out['q10_all'] = F_quantz(0.1)
    out['q90_all'] = F_quantz(0.9)
    out['q99_all'] = F_quantz(0.99)
    out['skewness_all'] = skew(allAngles)
    out['kurtosis_all'] = kurtosis(allAngles, fisher=False)

    return out


def _SUB_statav(x, n):
    # helper function
    NN = len(x)
    if NN < 2 * n: # not long enough
        statavmean = np.nan
        statavstd = np.nan
    x_buff = _buffer(x, int(np.floor(NN/n)))
    if x_buff.shape[1] > n:
        # remove final pt
        x_buff = x_buff[:, :n]
    
    statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
    statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def _buffer(X, n, p=0, opt=None):
    # helper function
    '''Mimic MATLAB routine to generate buffer array
    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html.
    Taken from: https://stackoverflow.com/questions/38453249/does-numpy-have-a-function-equivalent-to-matlabs-buffer 

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from X
    '''

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result

def NonlinearAutoCorr(y : ArrayLike, taus : ArrayLike, doAbs : Union[bool, None] = None) -> float:
    """
    Compute a custom nonlinear autocorrelation of a time series.

    Nonlinear autocorrelations are of the form:
        <x_i x_{i-tau_1} x_{i-tau_2} ...>
    The usual two-point autocorrelation is:
        <x_i x_{i-tau}>

    This function generalizes autocorrelation to higher-order products at multiple lags.

    Parameters
    ----------
    y : array-like
        The z-scored input time series (1D array).
    taus : array-like
        Vector of time delays (lags). For example:
            [2] computes <x_i x_{i-2}>
            [1, 2] computes <x_i x_{i-1} x_{i-2}>
            [1, 1, 3] computes <x_i x_{i-1}^2 x_{i-3}>
            [0, 0, 1] computes <x_i^3 x_{i-1}>
    doAbs : bool or None, optional
        If True, takes the absolute value before the final mean (recommended for even-length taus).
        If None (default), automatically sets doAbs=True for even-length taus and False for odd-length.

    Returns
    -------
    float
        The computed nonlinear autocorrelation.
    """
    y = np.asarray(y)
    taus = np.asarray(taus)
    if doAbs == None:
        if len(taus) % 2 == 1:
            doAbs = False
        else:
            doAbs = True

    N = len(y)
    tmax = np.max(taus)

    nlac = y[tmax:N]

    for i in taus:
        nlac = np.multiply(nlac,y[tmax - i:N - i])

    if doAbs:
        out = np.mean(np.absolute(nlac))

    else:
        out = np.mean(nlac)

    return float(out)

def PartialAutoCorr(y : ArrayLike, maxTau : int = 10, whatMethod : str = 'ols') -> dict:
    """
    Compute the partial autocorrelation of an input time series.
    
    This function calculates the partial autocorrelation function (PACF) up to a specified 
    lag using either ordinary least squares or Yule-Walker equations.

    Parameters
    ----------
    y : array-like
        The input time series as a scalar column vector
    maxTau : int, optional
        The maximum time-delay to compute PACF values for (default=10)
    whatMethod : {'ols', 'Yule-Walker'}, optional
        Method to compute partial autocorrelation (default='ols'):
        - 'ols': Ordinary least squares regression
        - 'Yule-Walker': Yule-Walker equations method

    Returns
    -------
    dict
        Dictionary containing partial autocorrelations for each lag, with keys:
        - 'pac_1': PACF at lag 1
        - 'pac_2': PACF at lag 2
        ...up to maxTau
    """
    y = np.asarray(y)
    N = len(y)
    if maxTau <= 0:
        raise ValueError('Negative or zero time lags not applicable')

    method_map = {'ols': 'ols', 'Yule-Walker': 'ywm'} 
    if whatMethod not in method_map:
        raise ValueError(f"Invalid method: {whatMethod}. Use 'ols' or 'Yule-Walker'.")

    # Compute partial autocorrelation
    pacf_values = pacf(y, nlags=maxTau, method=method_map[whatMethod])

    # Create output dictionary
    out = {}
    for i in range(1, maxTau + 1):
        out[f'pac_{i}'] = pacf_values[i]

    return out


def Embed2Dist(y : ArrayLike, tau : Union[None, str] = None) -> dict:
    """
    Analyzes distances in a 2-dim embedding space of a time series.

    Returns statistics on the sequence of successive Euclidean distances between
    points in a two-dimensional time-delay embedding space with a given
    time-delay, tau.

    Outputs include the autocorrelation of distances, the mean distance, the
    spread of distances, and statistics from an exponential fit to the
    distribution of distances.

    Parameters:
    y (array-like): A z-scored column vector representing the input time series.
    tau (int, optional): The time delay. If None, it's set to the first minimum of the autocorrelation function.

    Returns:
    dict: A dictionary containing various statistics of the embedding.
    """
    y = np.asarray(y)
    N = len(y) # time-series length

    if tau is None:
        tau = 'tau' # set to the first minimum of autocorrelation function
    
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        if tau > N / 10:
            tau = N//10

    # Make sure the time series is a column vector
    y = np.asarray(y).reshape(-1, 1)

    # Construct a 2-dimensional time-delay embedding (delay of tau)
    m = np.hstack((y[:-tau], y[tau:]))

    # Calculate Euclidean distances between successive points in this space, d:
    out = {}
    d = np.sqrt(np.sum(np.diff(m, axis=0)**2, axis=1))
    
    # Calculate autocorrelations
    out['d_ac1'] = AutoCorr(d, 1, 'Fourier')[0] # lag 1 ac
    out['d_ac2'] = AutoCorr(d, 2, 'Fourier')[0] # lag 2 ac
    out['d_ac3'] = AutoCorr(d, 3, 'Fourier')[0] # lag 3 ac

    out['d_mean'] = np.mean(d) # Mean distance
    out['d_median'] = np.median(d) # Median distance
    out['d_std'] = np.std(d, ddof=1) # Standard deviation of distances
    # need to use Hazen method of computing percentiles to get IQR consistent with MATLAB
    q75 = np.percentile(d, 75, method='hazen')
    q25 = np.percentile(d, 25, method='hazen')
    iqr_val = q75 - q25
    out['d_iqr'] = iqr_val # Interquartile range of distances
    out['d_max'] = np.max(d) # Maximum distance
    out['d_min'] = np.min(d) # Minimum distance
    out['d_cv'] = np.mean(d) / np.std(d, ddof=1) # Coefficient of variation of distances

    # Empirical distances distribution often fits Exponential distribution quite well
    # Fit to all values (often some extreme outliers, but oh well)
    l = 1 / np.mean(d)
    nlogL = -np.sum(expon.logpdf(d, scale=1/l))
    out['d_expfit_nlogL'] = nlogL

    # Calculate histogram
    bin_edges = binpicker(d.min(), d.max(), nbins=27)
    N, bin_edges = np.histogram(d, bins=bin_edges, density=True)
    bin_centers = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0)
    #exp_fit = expon.pdf(bin_centers, scale=1/l)
    #out['d_expfit_meandiff'] = np.mean(np.abs(N - exp_fit))

    return out

def Embed2Basic(y : ArrayLike, tau : Union[int, str] = 1) -> dict:
    """
    Point density statistics in a 2-d embedding space.

    Computes a set of point-density statistics in a plot of y_i against y_{i-tau}. The function 
    calculates the density of points near various geometric shapes in the embedding space, 
    including diagonals, parabolas, rings, and circles.

    Parameters
    -----------
    y : array_like
        The input time series.
    tau : int or str, optional
        The time lag (can be set to 'tau' to set the time lag to the first zero
        crossing of the autocorrelation function). Default is 1.

    Returns
    --------
    dict
        Dictionary containing various point density statistics.
    """
    y = np.asarray(y)
    if tau == 'tau':
        # Make tau the first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')

    xt = y[:-tau]  # part of the time series
    xtp = y[tau:]  # time-lagged time series
    N = len(y) - tau  # Length of each time series subsegment

    out = {}

    # Points in a thick bottom-left -- top-right diagonal
    out['updiag01'] = np.divide(np.sum(np.abs(xtp - xt) < 0.1), N)
    out['updiag05'] = np.divide(np.sum(np.abs(xtp - xt) < 0.5), N)

    # Points in a thick bottom-right -- top-left diagonal
    out['downdiag01'] = np.divide(np.sum(np.abs(xtp + xt) < 0.1), N)
    out['downdiag05'] = np.divide(np.sum(np.abs(xtp + xt) < 0.5), N)

    # Ratio of these
    out['ratdiag01'] = np.divide(out['updiag01'], out['downdiag01'])
    out['ratdiag05'] = np.divide(out['updiag05'], out['downdiag05'])

    # In a thick parabola concave up
    out['parabup01'] = np.divide(np.sum(np.abs(xtp - xt**2) < 0.1), N)
    out['parabup05'] = np.divide(np.sum(np.abs(xtp - xt**2) < 0.5), N)

    # In a thick parabola concave down
    out['parabdown01'] = np.divide(np.sum(np.abs(xtp + xt**2) < 0.1), N)
    out['parabdown05'] = np.divide(np.sum(np.abs(xtp + xt**2) < 0.5), N)

    # In a thick parabola concave up, shifted up 1
    out['parabup01_1'] = np.divide(np.sum(np.abs(xtp - (xt**2 + 1)) < 0.1), N)
    out['parabup05_1'] = np.divide(np.sum(np.abs(xtp - (xt**2 + 1)) < 0.5), N)

    # In a thick parabola concave down, shifted up 1 
    out['parabdown01_1'] = np.divide(np.sum(np.abs(xtp + (xt**2 - 1)) < 0.1), N)
    out['parabdown05_1'] = np.divide(np.sum(np.abs(xtp + (xt**2 - 1)) < 0.5), N)

    # In a thick parabola concave up, shifted down 1
    out['parabup01_n1'] = np.divide(np.sum(np.abs(xtp - (xt**2 - 1)) < 0.1), N)
    out['parabup05_n1'] = np.divide(np.sum(np.abs(xtp - (xt**2 - 1)) < 0.5), N)

    # In a thick parabola concave down, shifted down 1
    out['parabdown01_n1'] = np.divide(np.sum(np.abs(xtp + (xt**2 + 1)) < 0.1), N)
    out['parabdown05_n1'] = np.divide(np.sum(np.abs(xtp + (xt**2 + 1)) < 0.5), N)

    # RINGS (points within a radius range)
    out['ring1_01'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.1), N)
    out['ring1_02'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.2), N)
    out['ring1_05'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.5), N)

    # CIRCLES (points inside a given circular boundary)
    out['incircle_01'] = np.divide(np.sum(xtp**2 + xt**2 < 0.1), N)
    out['incircle_02'] = np.divide(np.sum(xtp**2 + xt**2 < 0.2), N)
    out['incircle_05'] = np.divide(np.sum(xtp**2 + xt**2 < 0.5), N)
    out['incircle_1'] = np.divide(np.sum(xtp**2 + xt**2 < 1), N)
    out['incircle_2'] = np.divide(np.sum(xtp**2 + xt**2 < 2), N)
    out['incircle_3'] = np.divide(np.sum(xtp**2 + xt**2 < 3), N)
    
    incircle_values = [out['incircle_01'], out['incircle_02'], out['incircle_05'],
                       out['incircle_1'], out['incircle_2'], out['incircle_3']]
    out['medianincircle'] = np.median(incircle_values)
    out['stdincircle'] = np.std(incircle_values, ddof=1)
    
    return out

def Embed2Shapes(y : ArrayLike, tau : Union[str, int, None] = 'tau', shape : str = 'circle', r : float = 1.0) -> dict:
    """
    Shape-based statistics in a 2-d embedding space.

    Takes a shape and places it on each point in the two-dimensional time-delay
    embedding space sequentially. This function counts the points inside this shape
    as a function of time, and returns statistics on this extracted time series.

    Parameters:
    -----------
    y : array_like
        The input time-series as a (z-scored) column vector.
    tau : int or str, optional
        The time-delay. If 'tau', it's set to the first zero crossing of the autocorrelation function.
    shape : str, optional
        The shape to use. Currently only 'circle' is supported.
    r : float, optional
        The radius of the circle.

    Returns:
    --------
    dict
        A dictionary containing various statistics of the constructed time series.
    """
    y = np.asarray(y)
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        # cannot set time delay > 10% of the length of the time series...
        if tau > len(y)/10:
            tau = int(np.floor(len(y)/10))
        
    # Create the recurrence space, populated by points m
    m = np.column_stack((y[:-tau], y[tau:]))
    N = len(m)

    # Start the analysis
    counts = np.zeros(N)
    if shape == 'circle':
        # Puts a circle around each point in the embedding space in turn
        # counts how many pts are inside this shape, looks at the time series thus formed
        for i in range(N): # across all pts in the time series
            m_c = m - m[i] # pts wrt current pt i
            m_c_d = np.sum(m_c**2, axis=1) # Euclidean distances from pt i
            counts[i] = np.sum(m_c_d <= r**2) # number of pts enclosed in a circle of radius r
    else:
        raise ValueError(f"Unknown shape '{shape}'")
    
    counts -= 1 # ignore self counts

    if np.all(counts == 0):
        print("No counts detected!")
        return np.nan

    # Return basic statistics on the counts
    out = {}
    out['ac1'] = AutoCorr(counts, 1, 'Fourier')[0]
    out['ac2'] = AutoCorr(counts, 2, 'Fourier')[0]
    out['ac3'] = AutoCorr(counts, 3, 'Fourier')[0]
    out['tau'] = FirstCrossing(counts, 'ac', 0, 'continuous')
    out['max'] = np.max(counts)
    out['std'] = np.std(counts, ddof=1)
    out['median'] = np.median(counts)
    out['mean'] = np.mean(counts)
    out['iqr'] = np.percentile(counts, 75, method='hazen') - np.percentile(counts, 25, method='hazen')
    out['iqronrange'] = out['iqr']/np.ptp(counts)

    # distribution - using sqrt binning method
    # numBinsToUse = int(np.ceil(np.sqrt(len(counts)))) # supposed to be what MATLAB uses for 'sqrt' option.
    # binCountsNorm, binEdges = np.histogram(counts, density=True, bins=numBinsToUse)
    # #minX, maxX = np.min(counts), np.max(counts)
    # #binEdges = binpicker(minX, maxX, nbins=numBinsToUse)
    # #binCounts = histc(counts, binEdges)
    # # normalise bin counts
    # #binCountsNorm = np.divide(binCounts, np.sum(binCounts))
    # # get bin centres
    # binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    # out['mode_val'] = np.max(binCountsNorm)
    # out['mode'] = binCentres[np.argmax(binCountsNorm)]
    # # histogram entropy
    # out['hist_ent'] = np.sum(binCountsNorm[binCountsNorm > 0] * np.log(binCountsNorm[binCountsNorm > 0]))

    # Stationarity measure for fifths of the time series
    afifth = int(np.floor(N/5))
    buffer_m = np.array([counts[i*afifth:(i+1)*afifth] for i in range(5)])
    out['statav5_m'] = np.std(np.mean(buffer_m, axis=1), ddof=1) / np.std(counts, ddof=1)
    out['statav5_s'] = np.std(np.std(buffer_m, axis=1, ddof=1), ddof=1) / np.std(counts, ddof=1)

    return out

def FZCGLSCF(y: ArrayLike, alpha: Union[float, int], beta: Union[float, int], maxtau: Union[int, None] = None) -> float:
    """
    The first zero-crossing of the generalized self-correlation function.

    Returns the first zero-crossing of the generalized self-correlation function (GLSCF)
    introduced by Queirós and Moyano (2007). The function calculates the GLSCF at 
    increasing time delays until it finds a zero crossing, and returns this lag value.

    Uses GLSCF to calculate the generalized self-correlations at each lag.

    Parameters
    ----------
    y : array_like
        The input time series
    alpha : float 
        The parameter alpha for GLSCF calculation. Must be non-zero.
    beta : float
        The parameter beta for GLSCF calculation. Must be non-zero.
    maxtau : int, optional
        Maximum time delay to search up to. If None, uses the time-series length.
        Default is None.

    Returns
    -------
    float
        The time lag τ of the first zero-crossing of the GLSCF.

    References
    ----------
    .. [1] Queirós, S.M.D., Moyano, L.G. (2007) "Yet on statistical properties of 
           traded volume: Correlation and mutual information at different value magnitudes"
           Physica A, 383(1), pp. 10-15.
           DOI: 10.1016/j.physa.2007.04.068
    """
    y = np.asarray(y)
    N = len(y)

    if maxtau is None:
        maxtau = N
    
    glscfs = np.zeros(maxtau)

    for i in range(1, maxtau+1):
        tau = i

        glscfs[i-1] = GLSCF(y, alpha, beta, tau)
        if (i > 1) and (glscfs[i-1]*glscfs[i-2] < 0):
            # Draw a straight line between these two and look at where it hits zero
            out = i - 1 + glscfs[i-1]/(glscfs[i-1]-glscfs[i-2])
            return out
    
    return maxtau

def GLSCF(y : ArrayLike, alpha : float, beta : float, tau : Union[int, str] = 'tau') -> float:
    """
    Compute the generalized linear self-correlation function (GLSCF) of a time series.

    This function implements the GLSCF as introduced by Queirós and Moyano (2007) to analyze
    correlations in the magnitude of time series values at different scales. The GLSCF 
    generalizes traditional autocorrelation by applying different exponents to earlier and 
    later time points.

    The function is defined as:
        GLSCF = (E[|x(t)|^α |x(t+τ)|^β] - E[|x(t)|^α]E[|x(t+τ)|^β]) / 
                (σ(|x(t)|^α)σ(|x(t+τ)|^β))
    where E[] denotes expectation and σ() denotes standard deviation.

    Parameters
    ----------
    y : array_like
        The input time series
    alpha : float 
        Exponent applied to the earlier time point x(t). Must be non-zero.
    beta : float
        Exponent applied to the later time point x(t+τ). Must be non-zero.
    tau : Union[int, str], optional
        The time delay (lag) between points. If 'tau', uses first zero-crossing
        of autocorrelation function. Default is 'tau'.

    Returns
    -------
    float
        The GLSCF value at the specified lag τ

    References
    ----------
    .. [1] Queirós, S.M.D., Moyano, L.G. (2007) "Yet on statistical properties of 
           traded volume: Correlation and mutual information at different value magnitudes"
           Physica A, 383(1), pp. 10-15.
           DOI: 10.1016/j.physa.2007.04.068
    """
    # Set tau to first zero-crossing of the autocorrelation function with the input 'tau'
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Take magnitudes of time-delayed versions of the time series
    y1 = np.abs(y[:-tau])
    y2 = np.abs(y[tau:])


    p1 = np.mean(np.multiply((y1 ** alpha), (y2 ** beta)))
    p2 = np.multiply(np.mean(y1 ** alpha), np.mean(y2 ** beta))
    p3 = np.sqrt(np.mean(y1 ** (2*alpha)) - (np.mean(y1 ** alpha))**2)
    p4 = np.sqrt(np.mean(y2 ** (2*beta)) - (np.mean(y2 ** beta))**2)

    glscf = (p1 - p2) / (p3 * p4)

    return glscf

def AutoCorr(y: ArrayLike, tau: Union[int, list] = 1, method: str = 'Fourier') -> Union[float, np.ndarray]:
    """
    Compute the autocorrelation of an input time series.

    Parameters:
    -----------
    y : array_like
        A scalar time series column vector.
    tau : int, list, optional
        The time-delay. If tau is a scalar, returns autocorrelation for y at that
        lag. If tau is a list, returns austocorrelations for y at that set of
        lags. If empty list, returns the full function for the 'Fourier' estimation method.
    method : str, optional
        The method of computing the autocorrelation: 'Fourier',
        'TimeDomainStat', or 'TimeDomain'.

    Returns:
    --------
    float or array
        The autocorrelation at the given time lag(s).

    """
    y = np.array(y)
    N = len(y)  # time-series length

    if tau:
        # if list is not empty
        if np.max(tau) > N - 1:  # -1 because acf(1) is lag 0
            logger.warning(f"Time lag {np.max(tau)} is too long for time-series length {N}.")
        if np.any(np.array(tau) < 0):
            logger.warning('Negative time lags not applicable.')
    
    if method == 'Fourier':
        n_fft = 2 ** (int(np.ceil(np.log2(N))) + 1)
        F = np.fft.fft(y - np.mean(y), n_fft)
        F = F * np.conj(F)
        acf = np.fft.ifft(F)  # Wiener–Khinchin
        acf = acf / acf[0]  # Normalize
        acf = np.real(acf)
        acf = acf[:N]
        
        if not tau:  # list empty, return the full function
            out = acf
        else:  # return a specific set of values
            tau = np.atleast_1d(tau)
            out = np.zeros(len(tau))
            for i, t in enumerate(tau):
                if (t > len(acf) - 1) or (t < 0):
                    out[i] = np.nan
                else:
                    out[i] = acf[t]
    
    elif method == 'TimeDomainStat':
        sigma2 = np.std(y, ddof=1)**2  # time-series variance
        mu = np.mean(y)  # time-series mean
        
        def acf_y(t):
            return np.mean((y[:N-t] - mu) * (y[t:] - mu)) / sigma2
        
        tau = np.atleast_1d(tau)
        out = np.array([acf_y(t) for t in tau])
    
    elif method == 'TimeDomain':
        tau = np.atleast_1d(tau)
        out = np.zeros(len(tau))
        
        for i, t in enumerate(tau):
            if np.any(np.isnan(y)):
                good_r = (~np.isnan(y[:N-t])) & (~np.isnan(y[t:]))
                print(f'NaNs in time series, computing for {np.sum(good_r)}/{len(good_r)} pairs of points')
                y1 = y[:N-t]
                y1n = y1[good_r] - np.mean(y1[good_r])
                y2 = y[t:]
                y2n = y2[good_r] - np.mean(y2[good_r])
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean(y1n * y2n) / np.std(y1[good_r], ddof=0) / np.std(y2[good_r], ddof=0)
            else:
                y1 = y[:N-t]
                y2 = y[t:]
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean((y1 - np.mean(y1)) * (y2 - np.mean(y2))) / np.std(y1, ddof=0) / np.std(y2, ddof=0)
    
    else:
        raise ValueError(f"Unknown autocorrelation estimation method {method}")
    
    return out

def FirstCrossing(y: ArrayLike, corrFun: str = 'ac', threshold: float = 0.0, whatOut: str = 'both') -> Union[dict, float]:
    """
    The first crossing of a given autocorrelation function across a given threshold.

    Parameters
    -----------
    y : array_like
        The input time series
    corrFun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    whatOut : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns
    --------
    dict or float
        The first crossing information, format depends on whatOut
    """
    # Select the self-correlation function
    if corrFun == 'ac':
        # Autocorrelation at all time lags
        corrs = AutoCorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corrFun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = pointOfCrossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if whatOut == 'both':
        out = {
            'firstCrossing': first_crossing_index - 1,
            'pointOfCrossing': point_of_crossing_index - 1
        }
    elif whatOut == 'discrete':
        out = first_crossing_index - 1
    elif whatOut == 'continuous':
        out = point_of_crossing_index - 1
    else:
        raise ValueError(f"Unknown output format '{whatOut}'")

    return out


def TranslateShape(y : ArrayLike, shape : str = 'circle', d : int = 2, howToMove : str = 'pts') -> dict:
    """
    Statistics on datapoints inside geometric shapes across the time series.

    This function moves a specified geometric shape (e.g., a circle or rectangle) of given size
    along the time axis of the input time series and computes statistics on the number of points
    falling within the shape at each position. This is a temporal-domain analogue of similar
    analyses in embedding spaces.

    In the future, this approach could be extended to use soft boundaries, decaying force functions,
    or truncated shapes.

    Parameters
    ----------
    y : array-like
        The input time series (1D array).
    shape : str, optional
        The shape to move along the time series. Supported options: 'circle', 'rectangle'. Default is 'circle'.
    d : int, optional
        Parameter specifying the size of the shape (e.g., radius for 'circle', half-width for 'rectangle'). Default is 2.
    howToMove : str, optional
        Method for moving the shape. Currently, only 'pts' is supported, which places the shape on each point in the time series.

    Returns
    -------
    dict
        Dictionary containing statistics on the number of points inside the shape as it moves through the time series,
        including mean, std, mode, and proportions for various counts.

    """
    y = np.array(y, dtype=float)
    N = len(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.shape[1] > y.shape[0]:
        y = y.T

    # add a time index
    ty = np.column_stack((np.arange(1, N+1), y[:, 0])) # has increasing integers as time in the first column
    if howToMove == 'pts':

        if shape == 'circle':

            r = d # set radius
            w = int(np.floor(r))
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge) # number of admissible points
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = idx - w - 1
                end = idx + w
                win = ty[start:end, :]
                difwin = win - ty[idx - 1, :]
                squared_dists = np.sum(difwin**2, axis=1)
                np_counts[i] = np.sum(squared_dists <= r**2)

        elif shape == 'rectangle':

            w = d
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge)
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = (idx - w) - 1
                end = (idx + w)
                np_counts[i] = np.sum(
                    np.abs(y[start:end, 0]) <= np.abs(y[i, 0])
                )
        else:
            raise ValueError(f"Unknown shape {shape}. Choose either 'circle' or 'rectangle'")
    else:
        raise ValueError(f"Unknown setting for 'howToMove' input: '{howToMove}'. Only option is currently 'pts'.")

    # compute stats on number of hits inside the shape
    out = {}
    out["max"] = np.max(np_counts)
    out["std"] = np.std(np_counts, ddof=1)
    out["mean"] = np.mean(np_counts)
    
    # count the hits
    vals, hits = np.unique_counts(np_counts)
    max_val = np.argmax(hits)
    out["npatmode"] = hits[max_val]/NN
    out["mode"] = vals[max_val]

    count_types = ["ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "elevens"]
    for i in range(1, 12):
        if 2*w + 1 >= i:
            out[f"{count_types[i-1]}"] = np.mean(np_counts == i)
    
    out['statav2_m'] = _stat_av(np_counts, 'mean', 2, 1)
    out['statav2_s'] = _stat_av(np_counts, 'std', 2, 1)
    out['statav3_m'] = _stat_av(np_counts, 'mean', 3, 1)
    out['statav3_s'] = _stat_av(np_counts, 'std', 3, 1)
    out['statav4_m'] = _stat_av(np_counts, 'mean', 4, 1)
    out['statav4_s'] = _stat_av(np_counts, 'std', 4, 1)

    return out

def _stat_av(y: ArrayLike, windowStat: str = 'mean', numSeg: int = 5, incMove: int = 2):
    # helper function to compute sliding winow stats for translate shape
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

     # convert a step index (stepInd) to a range of indices corresponding to that window
    def get_window(stepInd: int):
        start_idx = (stepInd) * inc
        end_idx = (stepInd) * inc + winLength
        return np.arange(start_idx, end_idx).astype(int)
    
    if windowStat == 'mean':
        for i in range(numSteps):
            qs[i] = np.mean(y[get_window(i)])
    elif windowStat == 'std':
        for i in range(numSteps):
            qs[i] = np.std(y[get_window(i)], ddof=1)
    return np.std(qs, ddof=1)/np.std(y, ddof=1)


def AutoCorrShape(y : ArrayLike, stopWhen : Union[int, str] = 'posDrown') -> dict:
    """
    How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters
    -----------
    y : array_like
        The input time series
    stopWhen : str or int, optional
        The criterion for the maximum lag to measure the ACF up to.
        Default is 'posDrown'.

    Returns
    --------
    dict
        A dictionary containing various metrics about the autocorrelation function.
    """
    y = np.asarray(y)
    N = len(y)

    # Only look up to when two consecutive values are under the significance threshold
    th = 2 / np.sqrt(N)  # significance threshold

    # Calculate the autocorrelation function, up to a maximum lag, length of time series (hopefully it's cropped by then)
    acf = []

    # At what lag does the acf drop to zero, Ndrown (by my definition)?
    if isinstance(stopWhen, int):
        taus = list(range(0, stopWhen+1))
        acf = AutoCorr(y, taus, 'Fourier')
        Ndrown = stopWhen
        
    elif stopWhen in ['posDrown', 'drown', 'doubleDrown']:
        # Compute ACF up to a given threshold:
        Ndrown = 0 # the point at which ACF ~ 0
        if stopWhen == 'posDrown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if np.isnan(acf_val):
                    logger.warning("Weird time series (constant?)")
                    out = np.nan
                if acf_val < th:
                    # Ensure ACF is all positive
                    if acf_val > 0:
                        Ndrown = i
                        acf.append(acf_val)
                    else:
                        # stop at the previous point if not positive
                        Ndrown = i-1
                    # ACF has dropped below threshold, break the for loop...
                    break
                # hasn't dropped below thresh, append to list 
                acf.append(acf_val)
            # This should yield the initial, positive portion of the ACF.
            assert all(np.array(acf) > 0)
        elif stopWhen == 'drown':
            # Stop when ACF is very close to 0 (within threshold, th = 2/sqrt(N))
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 0 and abs(acf_val) < th:
                    Ndrown = i
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
        elif stopWhen == 'doubleDrown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if Ndrown > 0 and i == Ndrown * 2:
                    acf.append(acf_val)
                    break
                elif i > 1 and abs(acf_val) < th:
                    Ndrown = i
                acf.append(acf_val)
    else:
        raise ValueError(f"Unknown ACF decay criterion: '{stopWhen}'")

    acf = np.array(acf)
    Nac = len(acf)

    # Check for good behavior
    if np.any(np.isnan(acf)):
        # This is an anomalous time series (e.g., all constant, or conatining NaNs)
        out = np.nan
    
    out = {}
    out['Nac'] = Ndrown

    # Basic stats on the ACF
    out['sumacf'] = np.sum(acf)
    out['meanacf'] = np.mean(acf)
    if stopWhen != 'posDrown':
        out['meanabsacf'] = np.mean(np.abs(acf))
        out['sumabsacf'] = np.sum(np.abs(acf))

    # Autocorrelation of the ACF
    minPointsForACFofACF = 5 # can't take lots of complex stats with fewer than this

    if Nac > minPointsForACFofACF:
        out['ac1'] = AutoCorr(acf, 1, 'Fourier')[0]
        if all(acf > 0):
            out['actau'] = np.nan
        else:
            out['actau'] = AutoCorr(acf, FirstCrossing(acf, 'ac', 0, 'discrete'), 'Fourier')[0]

    else:
        out['ac1'] = np.nan
        out['actau'] = np.nan
    
    # Local extrema
    dacf = np.diff(acf)
    ddacf = np.diff(dacf)
    extrr = signChange(dacf, 1)
    sdsp = ddacf[extrr]

    # Proportion of local minima
    out['nminima'] = np.sum(sdsp > 0)
    out['meanminima'] = np.mean(sdsp[sdsp > 0])

    # Proportion of local maxima
    out['nmaxima'] = np.sum(sdsp < 0)
    out['meanmaxima'] = abs(np.mean(sdsp[sdsp < 0])) # must be negative: make it positive

    # Proportion of extrema
    out['nextrema'] = len(sdsp)
    out['pextrema'] = len(sdsp) / Nac

    # Fit exponential decay (only for 'posDrown', and if there are enough points)
    # Should probably only do this up to the first zero crossing...
    fitSuccess = False
    minPointsToFitExp = 4 # (need at least four points to fit exponential)

    if stopWhen == 'posDrown' and Nac >= minPointsToFitExp:
        # Fit exponential decay to (absolute) ACF:
        # (kind of only makes sense for the first positive period)
        expFunc = lambda x, b : np.exp(-b * x)
        try:
            popt, _ = curve_fit(expFunc, np.arange(Nac), acf, p0=0.5)
            fitSuccess = True
        except:
            fitSuccess = False
        
    if fitSuccess:
        bFit = popt[0] # fitted b
        out['decayTimescale'] = 1 / bFit
        expFit = expFunc(np.arange(Nac), bFit)
        residuals = acf - expFit
        out['fexpacf_r2'] = 1 - (np.sum(residuals**2) / np.sum((acf - np.mean(acf))**2))
        # had to fit a second exponential function with negative b to get same output as MATLAB for std residuals
        expFit2 = expFunc(np.arange(Nac), -bFit)
        residuals2 = acf - expFit2
        out['fexpacf_stdres'] = np.std(residuals2, ddof=1) 

    else:
        # Fit inappropriate (or failed): return nans for the relevant stats
        out['decayTimescale'] = np.nan
        out['fexpacf_r2'] = np.nan
        out['fexpacf_stdres'] = np.nan
    
    return out


def TRev(y : ArrayLike, tau : Union[int, str] = 'ac') -> dict:
    """
    CO_trev: Normalized nonlinear autocorrelation (trev) function of a time series.

    Calculates the trev function, a normalized nonlinear autocorrelation, as described in the TSTOOL nonlinear time-series analysis package. This quantity is often used as a nonlinearity statistic in surrogate data analysis,
    see: "Surrogate time series", T. Schreiber and A. Schmitz, Physica D, 142(3-4), 346 (2000).

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        Time lag. Can be:
            - int: Use the specified lag.
            - 'ac': Use the first zero-crossing of the autocorrelation function.
            - 'mi': Use the first minimum of the automutual information function.
        Default is 'ac'.

    Returns
    -------
    dict
        Dictionary containing:
            - 'raw': The raw trev expression.
            - 'abs': The magnitude of the raw expression.
            - 'num': The numerator.
            - 'absnum': The magnitude of the numerator.
            - 'denom': The denominator.
    """
    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = FirstMin(y, 'mi')
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay. (Is the time series too short?)")

    # Compute trev quantities
    yn = y[:-tau]
    yn1 = y[tau:] # yn, tau steps ahead
    
    out = {}

    # The trev expression used in TSTOOL
    raw = np.mean((yn1 - yn)**3) / (np.mean((yn1 - yn)**2))**(3/2)
    out['raw'] = raw

    # The magnitude
    out['abs'] = np.abs(raw)

    # The numerator
    num = np.mean((yn1-yn)**3)
    out['num'] = num
    out['absnum'] = np.abs(num)

    # the denominator
    out['denom'] = (np.mean((yn1-yn)**2))**(3/2)

    return out


def TC3(y : list, tau : Union[int, str, None] = 'ac'):
    """
    Normalized nonlinear autocorrelation function, tc3.

    Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    given time-delay, tau.
    Statistic is for two time-delays, normalized in terms of a single time delay.
    Used as a test statistic for higher order correlational moments in surrogate
    data analysis.

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        Time lag. Can be:
            - int: Use the specified lag.
            - 'ac': Use the first zero-crossing of the autocorrelation function. Default is 'ac'.
            - 'mi': Use the first minimum of the automutual information function.

    Returns
    -------

    dict
        A dictionary containing:
        - 'raw': The raw tc3 expression
        - 'abs': The magnitude of the raw expression
        - 'num': The numerator
        - 'absnum': The magnitude of the numerator
        - 'denom': The denominator
    """
    # Set the time lag as a measure of the time-series correlation length
    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = FirstMin(y, 'mi')
    
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay (time series too short?)")
    
    # Compute tc3 statistic
    yn = y[:-2*tau]
    yn1 = y[tau:-tau] # yn1, tau steps ahead
    yn2 = y[2*tau:] # yn2, 2*tau steps ahead

    numerator = np.mean(yn * yn1 * yn2)
    denominator = np.abs(np.mean(yn * yn1)) ** (3/2)

    # The expression used in TSTOOL tc3:
    out = {}
    out['raw'] = numerator / denominator

    # The magnitude
    out['abs'] = np.abs(out['raw'])

    # The numerator
    out['num'] = numerator
    out['absnum'] = np.abs(out['num'])

    # The denominator
    out['denom'] = denominator

    return out
