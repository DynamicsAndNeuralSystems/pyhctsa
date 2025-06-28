
import numpy as np
from numpy.typing import ArrayLike 
from typing import Union
from scipy.stats import expon
from ..Utilities.utils import pointOfCrossing, binpicker
from loguru import logger
from statsmodels.tsa.stattools import pacf



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
