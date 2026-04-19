import logging
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import LinAlgError
from scipy.optimize import curve_fit
from scipy.stats import expon, gaussian_kde, kurtosis, skew
from scipy.stats import mode as smode
from statsmodels.tsa.stattools import pacf

from ..operations.information import automutual_info, first_min
from ..toolboxes.c22 import periodicity_wang_wrapper
from ..utils import bin_picker, make_mat_buffer, point_of_crossing, sign_change, z_score, histc

def add_noise(y: ArrayLike, tau: Union[int, str] = 1, ami_method: str = 'even',
              extra_param: Union[int, None] = None, random_seed = None) -> dict:
    """
    Changes in the automutual information with the addition of noise.

    Adds Gaussian-distributed noise to the time series with increasing standard deviation, eta, 
    across the range eta = 0, 0.1, ..., 2, and measures the mutual information at each point. 
    Can be measured using histograms with extra_param bins or using the Information Dynamics 
    Toolkit. The output is a set of statistics on the resulting set of automutual information
    estimates, including a fit to an exponential decay, since the automutual information 
    decreases with the added white noise. This algorithm is quite different, but was based 
    on the idea in [1].

    References
    ----------
    .. [1] "Titration of chaos with added noise", Chi-Sang Poon and Mauricio Barahona 
        P. Natl. Acad. Sci. USA, 98(13) 7107 (2001)

    Parameters
    ----------
    y : array-like
        Input time series (should be z-scored prior to analysis).

    tau : int or str, optional
        Time delay used to compute AMI.

        - If an ``int``, computes AMI at that lag.
        - If ``"ac"`` or ``"tau"``, uses the first zero-crossing of the
        autocorrelation function.

        Default is ``1``.

    ami_method : str, optional
        Estimation method for AMI.

        Histogram-based estimators:

        - ``"std1"``
        - ``"std2"``
        - ``"quantiles"``
        - ``"even"``

        JIDT-based estimators:

        - ``"gaussian"``
        - ``"kernel"``
        - ``"kraskov1"``
        - ``"kraskov2"``

        Default is ``"even"``.

    extra_param : int, optional
        Additional parameter for the AMI estimator.

        - For histogram methods: number of bins.
        - For JIDT methods: estimator-specific parameter.

        Default is ``10``.

    random_seed : int or None, optional
        Seed controlling noise realisations. If ``None``, defaults internally
        to ``0``.

    Returns
    -------
    dict
        Summary statistics of the AMI–noise curve, including exponential
        decay fit parameters and descriptive measures.
    """
    y = np.asarray(y)
    # Set tau to minimum of autocorrelation function if 'ac' or 'tau'
    if tau in ['ac', 'tau']:
        tau = first_crossing(y, 'ac', 0, 'discrete')
    if extra_param is None:
        # JIDT expects empty string for no extra params
        extra_param = ''
    # Generate noise
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(0)
    noise = np.random.randn(len(y)) # generate uncorrelated additive noise

    # Set up noise range
    noise_range = np.linspace(0, 3, 50) # compare properties across this noise range
    num_repeats = len(noise_range)

    # Compute the automutual information across a range of noise levels
    amis = np.zeros(num_repeats)
    if ami_method in ['std1', 'std2', 'quantiles', 'even']:
        # histogram-based methods using my naive implementation in CO_Histogram
        for i in range(num_repeats):
            amis[i] = histogram_ami(y + noise_range[i]*noise, tau, ami_method, extra_param)
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    if ami_method in ['gaussian','kernel','kraskov1','kraskov2']:
        for i in range(num_repeats):
            amis[i] = automutual_info(y + noise_range[i]*noise, tau, ami_method, str(extra_param))
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    # Output statistics
    out = {}
    # Proportion decreases
    out['pdec'] = np.sum(np.diff(amis) < 0) / (num_repeats - 1)

    # Mean change in AMI
    out['meanch'] = np.mean(np.diff(amis))

    # Autocorrelation of AMIs
    out['ac1'] = autocorr(amis, 1, 'Fourier')[0]
    out['ac2'] = autocorr(amis, 2, 'Fourier')[0]

    # Noise level required to reduce ami to proportion x of its initial value
    first_under_vals = [0.75, 0.50, 0.25]
    for val in first_under_vals:
        out[f'firstUnder{int(val*100)}'] = first_under_fn(val * amis[0], noise_range, amis)

    # AMI at actual noise levels: 0.5, 1, 1.5 and 2
    noise_levels = [0.5, 1, 1.5, 2]
    for nlvl in noise_levels:
        out[f'ami_at_{int(nlvl*10)}'] = amis[np.argmax(noise_range >= nlvl)]

    # Count number of times the AMI function crosses its mean
    out['pcrossmean'] = np.sum(np.diff(np.sign(amis - np.mean(amis))) != 0) / (num_repeats - 1)

    # Fit exponential decay model
    exp_func = lambda x, a, b : a * np.exp(b * x)
    popt, pcov = curve_fit(exp_func, noise_range, amis, p0=[amis[0], -1])
    out['fitexpa'], out['fitexpb'] = popt
    residuals = amis - exp_func(noise_range, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((amis - np.mean(amis))**2)
    out['fitexpr2'] = 1 - (ss_res / ss_tot)
    out['fitexpadjr2'] = 1 - (1-out['fitexpr2'])*(len(amis)-1)/(len(amis)-2-1)
    out['fitexprmse'] = np.sqrt(np.mean(residuals**2))

    # Fit linear function
    p = np.polyfit(noise_range, amis, 1)
    out['fitlina'], out['fitlinb'] = p
    lin_fit = np.polyval(p, noise_range)
    out['linfit_mse'] = np.mean((lin_fit - amis)**2)

    return out

def first_under_fn(x : ArrayLike, m : ArrayLike, p : ArrayLike):
    """
    Find the value of m for the first time p goes under the threshold, x. 
    p and m are vectors of the same length
    """
    first_i = next((m_val for m_val, p_val in zip(m, p) if p_val < x), m[-1])

    return first_i


def theiler_q(y: ArrayLike) -> float:
    """
    Computes Theiler's Q statistic which quantifies asymmetry in time. 

    Calculates :math:`Q = \\langle (x_t + x_{t+1})^3 \\rangle / \\langle x^2 \\rangle^{3/2}`
    on a vector :math:`x`, as proposed by James Theiler.

    Copyright (C) 1996, D. Kaplan <kaplan@macalester.edu>

    Parameters
    ----------
    y : array-like
        The input time series.

    Returns
    -------
    float
        Theiler's Q statistic.
    """
    y = np.asarray(y)
    y2 = (np.mean(y**2))**(3/2)
    q = 0.0
    if y2 != 0:
        d2 = y[:-1] + y[1:]
        q = np.mean(d2 **3)/y2

    return float(q)

def crinkle_statistic(y: ArrayLike) -> float:
    """
    Computes Theiler's crinkle statistic.

    The statistic is defined as


    .. math::

        \\frac{\\left\\langle (y_{t-1} - 2 y_t + y_{t+1})^4 \\right\\rangle}
            {\\left\\langle y_t^2 \\right\\rangle^2}

    Copyright (C) 1996, D. Kaplan <kaplan@macalester.edu>
    
    Parameters
    ----------
    y : array-like
        The input time series.

    Returns
    -------
    float
        The crinkle statistic.
    """
    # subtract out the mean
    y = np.asarray(y)
    y = y - np.mean(y)
    y2 = np.mean(y*y)**2
    out = 0
    if y2 != 0:
        d2 = (2 * y[1:-1]) - y[:-2] - y[2:]
        out = np.mean(d2 ** 4)/y2

    return float(out)

def time_rev_kaplan(y: ArrayLike, time_lag: int = 1) -> float:
    """
    Time reversal asymmetry statistic.

    Calculates a time reversal asymmetry statistic as described by D. Kaplan.
    This statistic quantifies the asymmetry of a time series under time reversal.

    Parameters
    ----------
    y : array-like
        The input time series.
    time_lag : int, optional
        The time scale (in samples) to use for the embedding (default is 1).

    Returns
    -------
    float
        The time reversal asymmetry statistic.
    """
    embedded = _lag_embed(np.asarray(y), 3, time_lag)
    a = embedded[:, 0]
    b = embedded[:, 1]
    c = embedded[:, 2]
    res = np.mean(a * a * b - b*c*c)

    return float(res)

def _lag_embed(x: ArrayLike, m: int, lag: int = 1) -> ArrayLike:
    """Constructs a time-delay embedding of a time series."""
    x = np.asarray(x).flatten()
    lx = len(x)
    if lx < lag * (m - 1) + 1:
        raise ValueError("Time series is too short for the given dimension and lag.")
    new_size = lx - lag * (m - 1)
    y = np.zeros((new_size, m))
    for i in range(m):
        # The first column (i=0) should be the most delayed data
        start_index = (m - 1 - i) * lag
        end_index = start_index + new_size
        y[:, i] = x[start_index:end_index]

    return y

def embed2_angle_tau(y: ArrayLike, max_tau: int) -> dict:
    """
    Angle autocorrelation in a 2-dimensional embedding space.

    Investigates how the autocorrelation of angles between successive points in
    the two-dimensional time-series embedding change as tau varies from
    tau = 1, 2, ..., max_tau.

    Parameters
    ----------
    y : array-like
        The input time series (column vector).
    max_tau : int
        The maximum time lag to consider.

    Returns
    -------
    dict
        Dictionary containing statistics of the autocorrelation of angles for each tau,
        including mean, max, min, and autocorrelation at different lags.
    """
    tau_range = np.arange(1, max_tau + 1)
    num_tau = len(tau_range)

    # Ensure y is a column vector
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T

    stats_store = np.zeros((3, num_tau))

    for i, tau in enumerate(tau_range):
        m = np.column_stack((y[:-tau], y[tau:]))
        diff_x = np.diff(m[:, 0])
        diff_y = np.diff(m[:, 1])
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = diff_y / diff_x
        theta = np.arctan(theta)

        if len(theta) == 0:
            raise ValueError(f'Time series (N={len(y)}) too short for embedding')

        stats_store[0, i] = autocorr(theta, 1, 'Fourier')[0]
        stats_store[1, i] = autocorr(theta, 2, 'Fourier')[0]
        stats_store[2, i] = autocorr(theta, 3, 'Fourier')[0]
    # Compute output statistics
    out = {
        'ac1_thetaac1': autocorr(stats_store[0, :], 1, 'Fourier')[0],
        'ac1_thetaac2': autocorr(stats_store[1, :], 1, 'Fourier')[0],
        'ac1_thetaac3': autocorr(stats_store[2, :], 1, 'Fourier')[0],
        'mean_thetaac1': np.mean(stats_store[0, :]),
        'max_thetaac1': np.max(stats_store[0, :]),
        'min_thetaac1': np.min(stats_store[0, :]),
        'mean_thetaac2': np.mean(stats_store[1, :]),
        'max_thetaac2': np.max(stats_store[1, :]),
        'min_thetaac2': np.min(stats_store[1, :]),
        'mean_thetaac3': np.mean(stats_store[2, :]),
        'max_thetaac3': np.max(stats_store[2, :]),
        'min_thetaac3': np.min(stats_store[2, :]),
    }

    out['meanrat_thetaac12'] = out['mean_thetaac1'] / out['mean_thetaac2']
    out['diff_thetaac12'] = np.sum(np.abs(stats_store[1, :] - stats_store[0, :]))

    return out

def embed2(y: ArrayLike, tau: Union[int, str] = 'tau') -> dict:
    """
    Statistics of the time series in a 2-dimensional embedding space.

    Embeds the (z-scored) time series in a two-dimensional time-delay 
    embedding space with a given time-delay, tau, and outputs a set 
    of statistics about the structure in this space, including angular 
    distribution, stationarity, Euclidean distances from the origin, 
    and statistics on outliers.

    Parameters
    ----------
    y : array-like
        The input time series (will be converted to a column vector).
    tau : int or str, optional
        The time-delay. If 'tau', it will be set to the first zero-crossing of 
        the autocorrelation function (ACF).

    Returns
    -------
    dict
        Dictionary containing:
            - Distribution of angles between successive points in the embedding space.
            - Stationarity of the angular distribution (across segments).
            - Euclidean distances from the origin (mean, std, etc.).
            - Statistics on outliers in the embedding space (e.g., area ratios).
    """

    # Set tau to the first zero-crossing of the autocorrelation function with the 'tau' input
    if tau == 'tau':
        tau = first_crossing(y, 'ac', 0, 'discrete')
        if tau > len(y) / 10:
            tau = len(y) // 10
    # Ensure that y is a column vector
    y = np.array(y).reshape(-1, 1)

    # Construct the two-dimensional recurrence space
    m = np.hstack((y[:-tau], y[tau:]))
    N = m.shape[0] # number of points in the recurrence space

    # 1) Distribution of angles time series; angles between successive points in this space
    theta = np.divide(np.diff(m[:, 1]), np.diff(m[:, 0]))
    theta = np.arctan(theta) # measured as deviation from the horizontal

    out = {}

    out['theta_ac1'] = autocorr(theta, 1, 'Fourier')[0]
    out['theta_ac2'] = autocorr(theta, 2, 'Fourier')[0]
    out['theta_ac3'] = autocorr(theta, 3, 'Fourier')[0]

    out['theta_mean'] = np.mean(theta)
    out['theta_std'] = np.std(theta, ddof=1)
    
    bin_edges = np.linspace(-np.pi/2, np.pi/2, 11) # 10 bins in the histogram
    px, _ = _histcounts(theta, bin_edges=bin_edges)
    bin_widths = np.diff(bin_edges)
    out['hist10std'] = np.std(px, ddof=1)
    out['histent'] = -np.sum(px[px>0] * np.log(px[px>0] / bin_widths[px>0]))
    
    # Stationarity in fifths of the time series
    # Use histograms with 4 bins
    x = np.linspace(-np.pi/2, np.pi/2, 5) # 4 bins
    afifth = (N-1) // 5 # -1 because angles are correlations *between* points
    n = np.zeros((len(x)-1, 5))
    for i in range(5):
        n[:, i], _ = np.histogram(theta[afifth*i:afifth*(i+1)], bins=x)
        
    n = n / afifth
    
    for i in range(4):
        out[f'stdb{i+1}'] = np.std(n[:, i], ddof=1)

    # STATIONARITY of points in the space (do they move around in the space)
    # (1) in terms of distance from origin
    afifth = N // 5
    buffer_m = [m[afifth*i:afifth*(i+1), :] for i in range(5)]

    # Mean euclidean distance in each segment
    eucdm = [np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2)) for x in buffer_m]
    for i in range(5):
        out[f'eucdm{i+1}'] = eucdm[i]
    out['std_eucdm'] = np.std(eucdm, ddof=1)
    out['mean_eucdm'] = np.mean(eucdm)

    # Standard deviation of Euclidean distances in each segment
    eucds = [np.std(np.sqrt(x[:, 0]**2 + x[:, 1]**2), ddof=1) for x in buffer_m]
    for i in range(5):
        out[f'eucds{i+1}'] = eucds[i]
    out['std_eucds'] = np.std(eucds, ddof=1)
    out['mean_eucds'] = np.mean(eucds)

    # Maximum volume in each segment (defined as area of rectangle of max span in each direction)
    maxspanx = [np.ptp(x[:, 0]) for x in buffer_m]
    maxspany = [np.ptp(x[:, 1]) for x in buffer_m]
    spanareas = np.multiply(maxspanx, maxspany)
    out['stdspana'] = np.std(spanareas, ddof=1)
    out['meanspana'] = np.mean(spanareas)

    # Outliers in the embedding space
    # area of max span of all points; versus area of max span of 50% of points closest to origin
    d = np.sqrt(m[:, 0]**2 + m[:, 1]**2)
    ix = np.argsort(d)
    
    out['areas_all'] = np.ptp(m[:, 0]) * np.ptp(m[:, 1])
    r50 = ix[:int(np.ceil(len(ix)/2))] # ceil to match MATLAB's round fn output
    
    out['areas_50'] = np.ptp(m[r50, 0]) * np.ptp(m[r50, 1])
    out['arearat'] = out['areas_50'] / out['areas_all']

    return out 

def _histcounts(x: ArrayLike, bins: Union[int, None, str] = None, 
                bin_edges: Union[ArrayLike, None] = None) -> tuple:
    x = np.asarray(x).flatten()

    if bin_edges is not None:
        edges = np.asarray(bin_edges)
    elif bins is None or bins == 'auto':
        # Use Scott's rule for automatic binning
        bin_width = 3.5 * np.std(x, ddof=1) / (len(x) ** (1 / 3))
        edges = np.arange(np.min(x), np.max(x) + bin_width, bin_width)
    elif isinstance(bins, int):
        edges = np.linspace(np.min(x), np.max(x), bins + 1)
    else:
        raise ValueError("Invalid bins parameter")

    n, _ = np.histogram(x, bins=edges)

    n = n / len(x)

    return n, edges

def periodicity_wang(y: ArrayLike) -> dict:
    """
    Periodicity extraction measure of Wang et al. (2007).

    Implements an idea based on the periodicity extraction measure proposed in [1]_.

    The time series is detrended using a three-knot cubic regression spline and
    autocorrelations are computed up to one third of the time-series length. The
    reported frequency corresponds to the first peak in the autocorrelation
    function satisfying a set of conditions.

    The original paper considered a single threshold of ``0.01``. This
    implementation evaluates a range of thresholds:

    - ``0``
    - ``0.01``
    - ``0.1``
    - ``0.2``
    - :math:`1/\\sqrt{N}`
    - :math:`5/\\sqrt{N}`
    - :math:`10/\\sqrt{N}`

    where :math:`N` is the length of the time series.

    References
    ----------
    .. [1] "Structure-based Statistical Features and Multivariate Time Series Clustering"
            by X. Wang, A. Wirth, and L. Wang (2007).

    Parameters
    ----------
    y : array-like
        The input time series

    Returns
    -------
    dict
        Dictionary containing periodicity measures for each threshold.
    """
    y = np.asarray(y)

    return periodicity_wang_wrapper.periodicity_wang(y)

def compare_min_ami(y: ArrayLike, bin_method: str = 'std1', 
                    num_bins: Union[int, ArrayLike] = 10) -> dict:
    """
    Assess the variability in the first minimum of automutual 
    information (AMI) across binning strategies.

    This function computes the first minimum of the automutual 
    information function for a time series using various histogram 
    binning strategies and numbers of bins. It summarizes how the location
    of the first minimum varies across these different coarse-grainings.

    Parameters
    ----------
    y : array-like
        The input time series.
    bin_method : str, optional
        The method for estimating mutual information (passed to `histogram_ami`). Default is 'std1'.
    num_bins : int or array-like, optional
        The number of bins (or list of bin counts) to use for AMI estimation. Default is 10.

    Returns
    -------
    dict
        Dictionary containing statistics on the set of first minimums 
        of the automutual information function.
    """
    y = np.asarray(y)
    n = len(y)
    # Range of time lags to consider
    tau_range = np.arange(0, int(np.ceil(n / 2)) + 1)
    num_taus = len(tau_range)

    # range of bin numbers to consider
    if isinstance(num_bins, int):
        num_bins = [num_bins]

    num_bins_range = len(num_bins)
    ami_mins = np.zeros(num_bins_range)

    # Calculate automutual information
    for i in range(num_bins_range):  # vary over number of bins in histogram
        amis = np.zeros(num_taus)
        for j in range(num_taus):  # vary over time lags, tau
            amis[j] = histogram_ami(y, tau_range[j], bin_method, num_bins[i])
            if (j > 1) and ((amis[j] - amis[j - 1]) * (amis[j - 1] - amis[j - 2]) < 0):
                ami_mins[i] = tau_range[j - 1]
                break
        if ami_mins[i] == 0:
            ami_mins[i] = tau_range[-1]
    # basic statistics
    out = {}
    out['min'] = np.min(ami_mins)
    out['max'] = np.max(ami_mins)
    out['range'] = np.ptp(ami_mins)
    out['median'] = np.median(ami_mins)
    out['mean'] = np.mean(ami_mins)
    out['std'] = np.std(ami_mins, ddof=1) # will return NaN for single values instead of 0
    out['nunique'] = len(np.unique(ami_mins))
    out['mode'], out['modef'] = smode(ami_mins)
    out['modef'] = out['modef'] / num_bins_range

    # converged value? 
    out['conv4'] = np.mean(ami_mins[-5:])

    # look for peaks (local maxima)
    # % local maxima above 1*std from mean
    # inspired by curious result of periodic maxima for periodic signal with
    # bin size... ('quantiles', [2:80])
    diff_ami_mins = np.diff(ami_mins[:-1])
    positive_diff_indices = np.where(diff_ami_mins > 0)[0]
    sign_change_indices = sign_change(diff_ami_mins, 1)

    # Find the intersection of positive_diff_indices and sign_change_indices
    loc_extr = np.intersect1d(positive_diff_indices, sign_change_indices) + 1
    above_threshold_indices = np.where(ami_mins > out['mean'] + out['std'])[0]
    big_loc_extr = np.intersect1d(above_threshold_indices, loc_extr)

    # Count the number of elements in big_loc_extr
    out['nlocmax'] = len(big_loc_extr)

    return out

def histogram_ami(
    y: ArrayLike,
    tau: Union[str, int, ArrayLike] = 1,
    meth: str = 'even',
    num_bins: int = 10
) -> Union[float, dict]:
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
        
    num_bins : int, optional
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
        tau = first_crossing(y, 'ac', 0, 'discrete')

    # Bins for the data
    # same for both -- assume same distribution (true for stationary processes, or small lags)
    if meth == 'even':
        b = np.linspace(np.min(y), np.max(y), num_bins + 1)
        # Add increment buffer to ensure all points are included
        inc = 0.1
        b[0] -= inc
        b[-1] += inc
    elif meth == 'std1':  # bins out to +/- 1 std
        b = np.linspace(-1, 1, num_bins + 1)
        if np.min(y) < -1:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 1:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'std2':  # bins out to +/- 2 std
        b = np.linspace(-2, 2, num_bins + 1)
        if np.min(y) < -2:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 2:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'quantiles':  # use quantiles with ~equal number in each bin
        b = np.quantile(y, np.linspace(0, 1, num_bins + 1), method='hazen')
        b[0] -= 0.1
        b[-1] += 0.1
    else:
        raise ValueError(f"Unknown method '{meth}'")

    # Sometimes bins can be added (e.g., with std1 and std2), so need to redefine numBins
    num_bins = len(b) - 1

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
        pij = pij[:num_bins, :num_bins]  # joint
        pij = pij / np.sum(pij)  # normalize
        pi = np.sum(pij, axis=1)  # marginal
        pj = np.sum(pij, axis=0)  # other marginal

        pii = np.tile(pi, (num_bins, 1)).T
        pjj = np.tile(pj, (num_bins, 1))

        r = pij > 0  # Defining the range in this way, we set log(0) = 0
        amis[i] = np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}

def stick_angles(y: ArrayLike) -> dict:
    """
    Analysis of the line-of-sight angles between time series data pts. 

    Line-of-sight angles between time-series pts treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Parameters
    -----------
    y : array-like
        The input time series

    Returns
    --------
    dict
        A dictionary containing are returned on the obtained sequence of angles, theta, reflecting the
        maximum deviation a stick can rotate before hitting a stick representing
        another time point. Statistics include the mean and spread of theta,
        the different between positive and negative angles, measures of symmetry of
        the angles, stationarity, autocorrelation, and measures of the distribution of
        these stick angles.
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
    all_angles = np.concatenate(angles)

    # Initialise output dictionary
    out = {}
    out['std_p'] = np.nanstd(angles[0], ddof=1) 
    out['mean_p'] = np.nanmean(angles[0]) 
    out['median_p'] = np.nanmedian(angles[0])

    out['std_n'] = np.nanstd(angles[1], ddof=1)
    out['mean_n'] = np.nanmean(angles[1])
    out['median_n'] = np.nanmedian(angles[1])

    out['std'] = np.nanstd(all_angles, ddof=1)
    out['mean'] = np.nanmean(all_angles)
    out['median'] = np.nanmedian(all_angles)

    # difference between positive and negative angles
    # return difference in densities
    
    ksx = np.linspace(np.min(all_angles), np.max(all_angles), 200)
    out['pnsumabsdiff'] = np.nan
    if (len(angles[0]) > 0 and len(angles[1]) > 0 and
        np.var(angles[0]) > 1e-10 and np.var(angles[1]) > 1e-10):
        try:
            ksx = np.linspace(np.min(all_angles), np.max(all_angles), 200)
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
    # handle the case where angles is a constant
    if np.var(angles[0], ddof=1) > 1e-10:
        zangles.append(z_score(angles[0]))
    else:
        zangles.append([])
    if np.var(angles[1], ddof=1) > 1e-10:
        zangles.append(z_score(angles[1]))
    else:
        zangles.append([])
    zallAngles = z_score(all_angles)

    # how stationary are the angle sets?

    # there are positive angles
    if len(zangles[0]) > 0:
        # StatAv2
        out['statav2_p_m'], out['statav2_p_s'] = _sub_statav(zangles[0], 2)
        # StatAv3
        out['statav3_p_m'], out['statav3_p_s'] = _sub_statav(zangles[0], 3)
        # StatAv4
        out['statav4_p_m'], out['statav4_p_s'] = _sub_statav(zangles[0], 4)
        # StatAv5
        out['statav5_p_m'], out['statav5_p_s'] = _sub_statav(zangles[0], 5)
    else:
        out['statav2_p_m'], out['statav2_p_s'] = np.nan, np.nan
        out['statav3_p_m'], out['statav3_p_s'] = np.nan, np.nan
        out['statav4_p_m'], out['statav4_p_s'] = np.nan, np.nan
        out['statav5_p_m'], out['statav5_p_s'] = np.nan, np.nan
    
    # there are negative angles
    if len(zangles[1]) > 0:
        # StatAv2
        out['statav2_n_m'], out['statav2_n_s'] = _sub_statav(zangles[1], 2)
        # StatAv3
        out['statav3_n_m'], out['statav3_n_s'] = _sub_statav(zangles[1], 3)
        # StatAv4
        out['statav4_n_m'], out['statav4_n_s'] = _sub_statav(zangles[1], 4)
        # StatAv5
        out['statav5_n_m'], out['statav5_n_s'] = _sub_statav(zangles[1], 5)
    else:
        out['statav2_n_m'], out['statav2_n_s'] = np.nan, np.nan
        out['statav3_n_m'], out['statav3_n_s'] = np.nan, np.nan
        out['statav4_n_m'], out['statav4_n_s'] = np.nan, np.nan
        out['statav5_n_m'], out['statav5_n_s'] = np.nan, np.nan
    
    # All angles
    
    # StatAv2
    out['statav2_all_m'], out['statav2_all_s'] = _sub_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], out['statav3_all_s'] = _sub_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], out['statav4_all_s'] = _sub_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = _sub_statav(zallAngles, 5)
    
    # correlations? 
    if len(zangles[0]) > 0:
        out['tau_p'] = first_crossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = autocorr(zangles[0], 1, 'Fourier')[0]
        out['ac2_p'] = autocorr(zangles[0], 2, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan
        out['ac2_p'] = np.nan
    
    if len(zangles[1]) > 0:
        out['tau_n'] = first_crossing(zangles[1], 'ac', 0, 'continuous')
        out['ac1_n'] = autocorr(zangles[1], 1, 'Fourier')[0]
        out['ac2_n'] = autocorr(zangles[1], 2, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan
        out['ac2_n'] = np.nan
    
    out['tau_all'] = first_crossing(zallAngles, 'ac', 0, 'continuous')
    out['ac1_all'] = autocorr(zallAngles, 1, 'Fourier')[0]
    out['ac2_all'] = autocorr(zallAngles, 2, 'Fourier')[0]

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
    
    f_quantz = lambda x : np.quantile(zallAngles, x, method='hazen')
    out['q1_all'] = f_quantz(0.01)
    out['q10_all'] = f_quantz(0.1)
    out['q90_all'] = f_quantz(0.9)
    out['q99_all'] = f_quantz(0.99)
    out['skewness_all'] = skew(all_angles)
    out['kurtosis_all'] = kurtosis(all_angles, fisher=False)

    return out

def _sub_statav(x: ArrayLike, n : int) -> tuple:
    # helper function
    nn = len(x)
    if nn < 2 * n: # not long enough
        statavmean = np.nan
        statavstd = np.nan
    x_buff = make_mat_buffer(x, int(np.floor(nn/n)))
    if x_buff.shape[1] > n:
        # remove final pt
        x_buff = x_buff[:, :n]
    statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
    statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def nonlinear_autocorr(y: ArrayLike, taus: ArrayLike, absval: Union[bool, None] = None) -> float:
    """
    Compute a custom nonlinear autocorrelation of a time series.

    Nonlinear autocorrelations generalize the usual (two-point) autocorrelation
    to higher-order products evaluated at multiple lags. In general,

    .. math::

        \\left\\langle \\prod_{k=0}^{m} x_{i-\\tau_k} \\right\\rangle,

    where :math:`\\langle \\cdot \\rangle` denotes the time average. The usual
    two-point autocorrelation is recovered when :math:`m=1` with
    :math:`\\tau_0 = 0` and :math:`\\tau_1 = \\tau`:

    .. math::

        \\langle x_i\\, x_{i-\\tau} \\rangle.

    Parameters
    ----------
    y : array-like
        The z-scored input time series (1D array).

    taus : array-like of int
        Vector of time delays (lags) :math:`\\{\\tau_k\\}` defining the product.

        Examples:

        - ``[2]`` computes :math:`\\langle x_i\\, x_{i-2} \\rangle`.
        - ``[1, 2]`` computes :math:`\\langle x_i\\, x_{i-1}\\, x_{i-2} \\rangle`.
        - ``[1, 1, 3]`` computes :math:`\\langle x_i\\, x_{i-1}^2\\, x_{i-3} \\rangle`.
        - ``[0, 0, 1]`` computes :math:`\\langle x_i^3\\, x_{i-1} \\rangle`.

    absval : bool or None, optional
        Whether to apply an absolute value before the final mean.

        - If ``True``, takes the absolute value before averaging (often useful when
        the product has an even number of terms).
        - If ``None`` (default), sets ``absval=True`` when ``len(taus)`` is even and
        ``absval=False`` when ``len(taus)`` is odd.

    Returns
    -------
    float
        The computed nonlinear autocorrelation.
    """
    y = np.asarray(y)
    taus = np.asarray(taus)
    if absval is None:
        if len(taus) % 2 == 1:
            absval = False
        else:
            absval = True

    n = len(y)
    tmax = np.max(taus)

    nlac = y[tmax:n]

    for i in taus:
        nlac = np.multiply(nlac,y[tmax - i:n - i])

    if absval:
        out = np.mean(np.absolute(nlac))

    else:
        out = np.mean(nlac)

    return float(out)

def partial_autocorr(y: ArrayLike, max_tau: int = 10, what_method: str = 'ols') -> dict:
    """
    Compute the partial autocorrelation of an input time series.
    
    This function calculates the partial autocorrelation function (PACF) up to a specified 
    lag using either ordinary least squares or Yule-Walker equations.

    Parameters
    ----------
    y : array-like
        The input time series as a scalar column vector
    max_tau : int, optional
        The maximum time-delay to compute PACF values for (default=10)
    what_method : {'ols', 'Yule-Walker'}, optional
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
    max_tau = int(max_tau)
    y = np.asarray(y)
    if max_tau <= 0:
        raise ValueError('Negative or zero time lags not applicable')

    method_map = {'ols': 'ols', 'Yule-Walker': 'ywm'} 
    if what_method not in method_map:
        raise ValueError(f"Invalid method: {what_method}. Use 'ols' or 'Yule-Walker'.")

    # Compute partial autocorrelation
    pacf_values = pacf(y, nlags=max_tau, method=method_map[what_method])

    # Create output dictionary
    out = {}
    for i in range(1, max_tau + 1):
        out[f'pac_{i}'] = pacf_values[i]

    return out

def embed2_dist(y: ArrayLike, tau: Union[None, str, int] = None) -> dict:
    """
    Analyzes distances in a 2-dimensional embedding space of a time series.

    Returns statistics on the sequence of successive Euclidean distances between
    points in a two-dimensional time-delay embedding space with a given
    time-delay, tau.

    Outputs include the autocorrelation of distances, the mean distance, the
    spread of distances, and statistics from an exponential fit to the
    distribution of distances.

    Parameters
    ----------
    y : array-like
        The z-scored input time series (1D array).
    tau : (int, optional)
        The time delay. If None, it's set to the first minimum of the autocorrelation function.

    Returns
    -------
    dict: 
        A dictionary containing various statistics of the embedding including the 
        autocorrelation of distances, the mean distance, the spread of distances, 
        and statistics from an exponential fit to the distribution of distances.
    """
    y = np.asarray(y)
    N = len(y) # time-series length

    if tau is None:
        tau = 'tau' # set to the first minimum of autocorrelation function
    
    if tau == 'tau':
        tau = first_crossing(y, 'ac', 0, 'discrete')
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
    out['d_ac1'] = autocorr(d, 1, 'Fourier')[0] # lag 1 ac
    out['d_ac2'] = autocorr(d, 2, 'Fourier')[0] # lag 2 ac
    out['d_ac3'] = autocorr(d, 3, 'Fourier')[0] # lag 3 ac

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
    n_log_l = -np.sum(expon.logpdf(d, scale=1/l))
    out['d_expfit_nlogL'] = n_log_l

    # Calculate histogram
    # % Sum of abs differences between exp fit and observed:
    bin_edges = bin_picker(x_min=d.min(), x_max=d.max(), n_bins=np.floor(np.sqrt(len(d))))
    N, bin_edges = np.histogram(d, bins=bin_edges, density=True)
    bin_centers = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0)
    exp_fit = expon.pdf(bin_centers, scale=1/l)
    out['d_expfit_meandiff'] = np.mean(np.abs(N - exp_fit))

    return out

def embed2_basic(y: ArrayLike, tau: Union[int, str] = 1) -> dict:
    """
    Point-density statistics in a two-dimensional delay embedding.

    Computes a set of point-density statistics in the embedding space formed by
    :math:`(y_i, y_{i-\\tau})`. The method quantifies how points cluster around
    specific geometric structures in this plane, including diagonals,
    parabolas, rings, and circles.

    The embedding corresponds to a standard two-dimensional delay
    reconstruction with lag :math:`\\tau`.

    Parameters
    ----------
    y : array-like
        Input time series (1D array).

    tau : int
        Time delay used to construct the embedding
        :math:`(y_i, y_{i-\\tau})`.

    Returns
    -------
    dict
        Dictionary of point-density statistics associated with different
        geometric regions in the embedding space.
    """
    y = np.asarray(y)
    if tau == 'tau':
        # Make tau the first zero crossing of the autocorrelation function
        tau = first_crossing(y, 'ac', 0, 'discrete')
    tau = int(tau)
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

def embed2_shapes(y: ArrayLike, tau: Union[str, int, None] = 'tau',
                  shape: str = 'circle', r: float = 1.0) -> dict:
    """
    Shape-based statistics in a 2-d embedding space.

    Takes a shape and places it on each point in the two-dimensional time-delay
    embedding space sequentially. This function counts the points inside this shape
    as a function of time, and returns statistics on this extracted time series.

    Parameters
    -----------
    y : array-like
        The input time-series as a (z-scored) column vector.
    tau : int or str, optional
        The time-delay. If 'tau', it's set to the first zero crossing of the 
        autocorrelation function.
    shape : str, optional
        The shape to use. Currently only 'circle' is supported.
    r : float, optional
        The radius of the circle.

    Returns
    --------
    dict
        A dictionary containing various statistics of the constructed time series.
    """
    y = np.asarray(y)
    if tau == 'tau':
        tau = first_crossing(y, 'ac', 0, 'discrete')
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
    out['ac1'] = autocorr(counts, 1, 'Fourier')[0]
    out['ac2'] = autocorr(counts, 2, 'Fourier')[0]
    out['ac3'] = autocorr(counts, 3, 'Fourier')[0]
    out['tau'] = first_crossing(counts, 'ac', 0, 'continuous')
    out['max'] = np.max(counts)
    out['std'] = np.std(counts, ddof=1)
    out['median'] = np.median(counts)
    out['mean'] = np.mean(counts)
    out['iqr'] = np.percentile(counts, 75, method='hazen') - np.percentile(counts,
                                                                           25, method='hazen')
    out['iqronrange'] = out['iqr']/np.ptp(counts)

    # distribution - using sqrt binning method
    num_bins_to_use = int(np.ceil(np.sqrt(len(counts))))
    bin_counts_norm, bin_edges = np.histogram(counts, density=True, bins=num_bins_to_use)
    min_x, max_x = np.min(counts), np.max(counts)
    bin_edges = bin_picker(min_x, max_x, n_bins=num_bins_to_use)
    bin_counts = histc(counts, bin_edges)
    # normalise bin counts
    bin_counts_norm = np.divide(bin_counts, np.sum(bin_counts))
    # get bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    out['mode_val'] = np.max(bin_counts_norm)
    out['mode'] = bin_centres[np.argmax(bin_counts_norm)]
    # histogram entropy
    out['hist_ent'] = np.sum(bin_counts_norm[bin_counts_norm > 0] *
                             np.log(bin_counts_norm[bin_counts_norm > 0]))

    # Stationarity measure for fifths of the time series
    afifth = int(np.floor(N/5))
    buffer_m = np.array([counts[i*afifth:(i+1)*afifth] for i in range(5)])
    out['statav5_m'] = np.std(np.mean(buffer_m, axis=1), ddof=1) / np.std(counts, ddof=1)
    out['statav5_s'] = np.std(np.std(buffer_m, axis=1, ddof=1), ddof=1) / np.std(counts, ddof=1)

    return out

def fzcglscf(y: ArrayLike, alpha: Union[float, int], beta: Union[float, int],
             max_tau: Union[int, None] = None) -> float:
    """
    The first zero-crossing of the generalized self-correlation function.

    Returns the first zero-crossing of the generalized self-correlation function (GLSCF)
    introduced by Queirós and Moyano (2007). The function calculates the GLSCF at 
    increasing time delays until it finds a zero crossing, and returns this lag value.

    Uses glscf to calculate the generalized self-correlations at each lag.

    References
    ----------
    .. [1] Queirós, S.M.D., Moyano, L.G. (2007) "Yet on statistical properties of 
           traded volume: Correlation and mutual information at different value magnitudes"
           Physica A, 383(1), pp. 10-15.
           DOI: 10.1016/j.physa.2007.04.068

    Parameters
    ----------
    y : array-like
        The input time series
    alpha : float 
        The parameter alpha for GLSCF calculation. Must be non-zero.
    beta : float
        The parameter beta for GLSCF calculation. Must be non-zero.
    max_tau : int, optional
        Maximum time delay to search up to. If None, uses the time-series length.
        Default is None.

    Returns
    -------
    float
        The time lag τ of the first zero-crossing of the GLSCF.

    """
    y = np.asarray(y)
    N = len(y)

    if max_tau is None:
        max_tau = N
    
    glscfs = np.zeros(max_tau)

    for i in range(1, max_tau+1):
        tau = i

        glscfs[i-1] = glscf(y, alpha, beta, tau)
        if (i > 1) and (glscfs[i-1]*glscfs[i-2] < 0):
            # Draw a straight line between these two and look at where it hits zero
            out = i - 1 + glscfs[i-1]/(glscfs[i-1]-glscfs[i-2])
            return out
    
    return max_tau

def glscf(y: ArrayLike, alpha: float, beta: float, tau: Union[int, str] = 'tau') -> float:
    """
    Compute the generalized linear self-correlation function (GLSCF)
    of a time series.

    Implements the GLSCF introduced by Queirós and Moyano (2007) to
    analyze correlations in the magnitude of time-series values at
    different scales. The GLSCF generalizes the traditional
    autocorrelation by applying distinct exponents to earlier and later
    time points.

    The GLSCF is defined as

    .. math::

        \\mathrm{GLSCF}(\\tau; \\alpha, \\beta)
        =
        \\frac{
            \\mathbb{E}\\left[ |x(t)|^{\\alpha} |x(t+\\tau)|^{\\beta} \\right]
            -
            \\mathbb{E}\\left[ |x(t)|^{\\alpha} \\right]
            \\mathbb{E}\\left[ |x(t+\\tau)|^{\\beta} \\right]
        }{
            \\sigma\\left(|x(t)|^{\\alpha}\\right)
            \\sigma\\left(|x(t+\\tau)|^{\\beta}\\right)
        },

    where :math:`\\mathbb{E}[\\cdot]` denotes expectation and
    :math:`\\sigma(\\cdot)` denotes the standard deviation.

    References
    ----------
    .. [1] S. M. D. Queirós and L. G. Moyano (2007),
        "Yet on statistical properties of traded volume: Correlation
        and mutual information at different value magnitudes,"
        *Physica A*, 383(1), 10–15.
        DOI: 10.1016/j.physa.2007.04.068

    Parameters
    ----------
    y : array-like
        Input time series.

    alpha : float
        Exponent applied to the earlier time point :math:`x(t)`.
        Must be non-zero.

    beta : float
        Exponent applied to the later time point :math:`x(t+\\tau)`.
        Must be non-zero.

    tau : int or {"tau"}, optional
        Time delay (lag) between points.

        - If an ``int``, computes GLSCF at that lag.
        - If ``"tau"``, uses the first zero-crossing of the
        autocorrelation function.

        Default is ``"tau"``.

    Returns
    -------
    float
        The GLSCF value at the specified lag :math:`\\tau`.
    """
    # Set tau to first zero-crossing of the autocorrelation function with the input 'tau'
    if tau == 'tau':
        tau = first_crossing(y, 'ac', 0, 'discrete')
    
    # Take magnitudes of time-delayed versions of the time series
    y1 = np.abs(y[:-tau])
    y2 = np.abs(y[tau:])

    p1 = np.mean(np.multiply((y1 ** alpha), (y2 ** beta)))
    p2 = np.multiply(np.mean(y1 ** alpha), np.mean(y2 ** beta))
    p3 = np.sqrt(np.mean(y1 ** (2*alpha)) - (np.mean(y1 ** alpha))**2)
    p4 = np.sqrt(np.mean(y2 ** (2*beta)) - (np.mean(y2 ** beta))**2)

    return np.divide((p1-p2), (p3 * p4))

def autocorr(y: ArrayLike, tau: Union[int, list] = 1,
             method: str = 'Fourier') -> Union[float, np.ndarray]:
    """
    Compute the autocorrelation of an input time series.

    Parameters
    ----------
    y : array-like
        A scalar time-series column vector.

    tau : int or list of int, optional
        The time delay(s).

        - If an ``int``, returns the autocorrelation of ``y`` at that lag.
        - If a ``list`` of integers, returns autocorrelations at those lags.
        - If an empty list, returns the full autocorrelation function when
        using the ``"Fourier"`` estimation method.

    method : {"Fourier", "TimeDomainStat", "TimeDomain"}, optional
        Method used to compute the autocorrelation.

        - ``"Fourier"``: Computes autocorrelation via the Wiener–Khinchin
            theorem using the Fourier transform.
        - ``"TimeDomainStat"``: Statistical time-domain estimator.
        - ``"TimeDomain"``: Direct time-domain computation.

    Returns
    --------
    float or array
        The autocorrelation at the given time lag(s).

    """
    y = np.array(y)
    N = len(y)  # time-series length

    if tau:
        # if list is not empty
        if np.max(tau) > N - 1:  # -1 because acf(1) is lag 0
            logging.warning(f"Time lag {np.max(tau)} is too long for time-series length {N}.")
        if np.any(np.array(tau) < 0):
            logging.warning('Negative time lags not applicable.')
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

def first_crossing(y: ArrayLike, corr_fun: str = 'ac', threshold: float = 0.0,
                   what_out: str = 'both') -> Union[dict, float]:
    """
    The first crossing of a given autocorrelation function across a given threshold.

    Parameters
    -----------
    y : array-like
        The input time series
    corr_fun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    what_out : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns
    --------
    dict or float
        The first crossing information, format depends on whatOut.
    """
    # Select the self-correlation function
    if threshold == '1/e':
        threshold = 1/np.e
    if corr_fun == 'ac':
        # Autocorrelation at all time lags
        corrs = autocorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corr_fun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = point_of_crossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if what_out == 'both':
        out = {
            'firstCrossing': first_crossing_index,
            'pointOfCrossing': point_of_crossing_index
        }
    elif what_out == 'discrete':
        out = first_crossing_index
    elif what_out == 'continuous':
        out = point_of_crossing_index
    else:
        raise ValueError(f"Unknown output format '{what_out}'")

    return out

def translate_shape(y: ArrayLike, shape: str = 'circle', d: int = 2,
                    how_to_move: str = 'pts') -> dict:
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
        The shape to move along the time series. Supported options: 'circle', 'rectangle'. 
        Default is 'circle'.
    d : int, optional
        Parameter specifying the size of the shape (e.g., radius for 'circle', 
            half-width for 'rectangle'). Default is 2.
    how_to_move : str, optional
        Method for moving the shape. Currently, only 'pts' is supported, which places 
            the shape on each point in the time series.

    Returns
    -------
    dict
        Dictionary containing statistics on the number of points inside the shape as it 
        moves through the time series, including mean, std, mode, and proportions 
        for various counts.

    """
    y = np.array(y, dtype=float)
    N = len(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.shape[1] > y.shape[0]:
        y = y.T

    # add a time index
    # has increasing integers as time in the first column
    ty = np.column_stack((np.arange(1, N+1), y[:, 0])) 
    if how_to_move == 'pts':

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
                np_counts[i] = np.sum(np.abs(y[start:end, 0]) <= np.abs(y[idx-1, 0]))
        else:
            raise ValueError(f"Unknown shape {shape}. Choose either 'circle' or 'rectangle'")
    else:
        raise ValueError(f"Unknown setting for 'howToMove' input: '{how_to_move}'. Only option is currently 'pts'.")

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

def _stat_av(y: ArrayLike, window_stat: str = 'mean', num_seg: int = 5, inc_move: int = 2) -> float:
    """helper function to compute sliding winow stats for `TranslateShape`"""
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

    # convert a step index (stepInd) to a range of indices corresponding to that window
    def get_window(step_ind: int):
        start_idx = (step_ind) * inc
        end_idx = (step_ind) * inc + win_length

        return np.arange(start_idx, end_idx).astype(int)
    
    if window_stat == 'mean':
        for i in range(num_steps):
            qs[i] = np.mean(y[get_window(i)])
    elif window_stat == 'std':
        for i in range(num_steps):
            qs[i] = np.std(y[get_window(i)], ddof=1)

    return np.std(qs, ddof=1)/np.std(y, ddof=1)

def autocorr_shape(y: ArrayLike, stop_when: Union[int, str] = 'posDrown') -> dict:
    """
    How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters
    -----------
    y : array-like
        The input time series
    stop_when : str or int, optional
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

    # At what lag does the acf drop to zero, n_drown (by my definition)?
    if isinstance(stop_when, int):
        taus = list(range(0, stop_when+1))
        acf = autocorr(y, taus, 'Fourier')
        n_drown = stop_when
        
    elif stop_when in ['posDrown', 'drown', 'doubleDrown']:
        # Compute ACF up to a given threshold:
        n_drown = 0 # the point at which ACF ~ 0
        if stop_when == 'posDrown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = autocorr(y, i-1, 'Fourier')[0]
                if np.isnan(acf_val):
                    logging.warning("Weird time series (constant?)")
                    out = np.nan
                if acf_val < th:
                    # Ensure ACF is all positive
                    if acf_val > 0:
                        n_drown = i
                        acf.append(acf_val)
                    else:
                        # stop at the previous point if not positive
                        n_drown = i-1
                    # ACF has dropped below threshold, break the for loop...
                    break
                # hasn't dropped below thresh, append to list
                acf.append(acf_val)
            # This should yield the initial, positive portion of the ACF.
            assert all(np.array(acf) > 0)
        elif stop_when == 'drown':
            # Stop when ACF is very close to 0 (within threshold, th = 2/sqrt(N))
            for i in range(1, N+1):
                acf_val = autocorr(y, i-1, 'Fourier')[0] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 0 and abs(acf_val) < th:
                    n_drown = i
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
        elif stop_when == 'doubleDrown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = autocorr(y, i-1, 'Fourier')[0]
                if n_drown > 0 and i == n_drown * 2:
                    acf.append(acf_val)
                    break
                elif i > 1 and abs(acf_val) < th:
                    n_drown = i
                acf.append(acf_val)
    else:
        raise ValueError(f"Unknown ACF decay criterion: '{stop_when}'")

    acf = np.array(acf)
    nac = len(acf)

    # Check for good behavior
    if np.any(np.isnan(acf)):
        # This is an anomalous time series (e.g., all constant, or conatining NaNs)
        out = np.nan
    
    out = {}
    out['Nac'] = n_drown

    # Basic stats on the ACF
    out['sumacf'] = np.sum(acf)
    out['meanacf'] = np.mean(acf)
    if stop_when != 'posDrown':
        out['meanabsacf'] = np.mean(np.abs(acf))
        out['sumabsacf'] = np.sum(np.abs(acf))

    # Autocorrelation of the ACF
    min_pts_for_acf_of_acf = 5 # can't take lots of complex stats with fewer than this

    if nac > min_pts_for_acf_of_acf:
        out['ac1'] = autocorr(acf, 1, 'Fourier')[0]
        if all(acf > 0):
            out['actau'] = np.nan
        else:
            out['actau'] = autocorr(acf, first_crossing(acf, 'ac', 0, 'discrete'), 'Fourier')[0]

    else:
        out['ac1'] = np.nan
        out['actau'] = np.nan
    
    # Local extrema
    dacf = np.diff(acf)
    ddacf = np.diff(dacf)
    extrr = sign_change(dacf, 1)
    sdsp = ddacf[extrr]

    # Proportion of local minima
    out['nminima'] = np.sum(sdsp > 0)
    out['meanminima'] = np.mean(sdsp[sdsp > 0])

    # Proportion of local maxima
    out['nmaxima'] = np.sum(sdsp < 0)
    out['meanmaxima'] = abs(np.mean(sdsp[sdsp < 0])) # must be negative: make it positive

    # Proportion of extrema
    out['nextrema'] = len(sdsp)
    out['pextrema'] = len(sdsp) / nac

    # Fit exponential decay (only for 'posDrown', and if there are enough points)
    # Should probably only do this up to the first zero crossing...
    fit_success = False
    min_pts_to_fit_exp = 4 # (need at least four points to fit exponential)

    if stop_when == 'posDrown' and nac >= min_pts_to_fit_exp:
        # Fit exponential decay to (absolute) ACF:
        # (kind of only makes sense for the first positive period)
        exp_func = lambda x, b : np.exp(-b * x)
        try:
            popt, _ = curve_fit(exp_func, np.arange(nac), acf, p0=0.5)
            fit_success = True
        except Exception:
            fit_success = False
    if fit_success:
        b_fit = popt[0] # fitted b
        out['decayTimescale'] = 1 / b_fit
        exp_fit = exp_func(np.arange(nac), b_fit)
        residuals = acf - exp_fit
        out['fexpacf_r2'] = 1 - (np.sum(residuals**2) / np.sum((acf - np.mean(acf))**2))
        exp_fit2 = exp_func(np.arange(nac), -b_fit)
        residuals2 = acf - exp_fit2
        out['fexpacf_stdres'] = np.std(residuals2, ddof=1)

    else:
        # Fit inappropriate (or failed): return nans for the relevant stats
        out['decayTimescale'] = np.nan
        out['fexpacf_r2'] = np.nan
        out['fexpacf_stdres'] = np.nan
    return out

def trev(y: ArrayLike, tau: Union[int, str] = 'ac') -> dict:
    """
    Normalized nonlinear autocorrelation (trev) function of a time series.

    Calculates the trev function, a normalized nonlinear autocorrelation, 
    as described in the TSTOOL nonlinear time-series analysis package. 
    This quantity is often used as a nonlinearity statistic in surrogate data analysis,
    see [1].

    References
    ----------
    .. [1] "Surrogate time series", T. Schreiber and A. Schmitz, Physica D, 142(3-4), 346 (2000).

    Parameters
    ----------
    y : array-like
        Input time series.
    tau : int or str, optional
        Time lag. Can be:
            - int: Use the specified lag.
            - 'ac': Use the first zero-crossing of the autocorrelation function (default).
            - 'mi': Use the first minimum of the automutual information function.

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
        tau = first_crossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = first_min(y, 'mi')
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

def tc3(y: list, tau: Union[int, str, None] = 'ac') -> dict:
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
        tau = first_crossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = first_min(y, 'mi')
    
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
