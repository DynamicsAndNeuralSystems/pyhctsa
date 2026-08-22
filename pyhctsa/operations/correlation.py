import logging
logger = logging.getLogger('pyhctsa')
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import LinAlgError
from scipy.linalg import solve_triangular
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import chi2, expon, gaussian_kde, kurtosis, skew
from scipy.stats import mode as smode
from statsmodels.tsa.stattools import pacf

from ..operations.hypothesis_tests import _kstest_statistic
from ..operations.information import first_min, automutual_info
from ..toolboxes.c22 import periodicity_wang_wrapper
from ..toolboxes.matlab.matlab_fit import fit_exp1, goodness_of_fit
from ..utils import (_first_index_past_threshold, bin_picker, histc,
                     make_mat_buffer, matlab_quantile, point_of_crossing,
                     sign_change, time_delay_embed, z_score)

def _resolve_time_delay(y: ArrayLike, tau: Union[int, str]) -> Union[int, float]:
    """Resolve a string time-delay spec to a lag.

    ``'ac'`` uses the first zero-crossing of the autocorrelation function and
    ``'mi'`` the first minimum of the automutual information. An integer is
    returned unchanged. The resolved value may be NaN (time series too short);
    callers are responsible for handling that.
    """
    if not isinstance(tau, str):
        return tau
    if tau == 'ac':
        return first_crossing(y, 'ac', 0, 'discrete')
    if tau == 'mi':
        return first_min(y, 'mi')
    raise ValueError(f'Invalid time-delay method: {tau}. Choose either mi or ac.')


def add_noise(y: ArrayLike, tau: Union[int, str] = 1, ami_method: str = 'even',
              extra_param: Union[int, None] = None, random_seed = None,
              noise = None) -> dict:
    """
    Changes in the automutual information with the addition of noise.

    Adds Gaussian-distributed noise to the time series with increasing standard deviation, eta, 
    across the range eta = 0, 0.1, ..., 2, and measures the mutual information at each point. 
    Can be measured using histograms with extra_param bins, or Kraskov estimators with k = extra_param.
    The output is a set of statistics on the resulting set of automutual information
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

        Alternative estimators:

        - ``"gaussian"``
        - ``"kraskov1"``
        - ``"kraskov2"``

        Default is ``"even"``.

    extra_param : int, optional
        Additional parameter for the AMI estimator.

        - For histogram methods: number of bins.
        - For alternative methods: estimator-specific parameter.

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
    # Generate noise
    if noise is not None:
        noise = np.asarray(noise)
    else:
        np.random.seed(0 if random_seed is None else random_seed)
        noise = np.random.randn(len(y))  # generate uncorrelated additive noise

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
                logger.warning('Error computing AMI: Time series too short (?)')
                return np.nan
    if ami_method in ['gaussian','kraskov1','kraskov2']:
        for i in range(num_repeats):
            amis[i] = automutual_info(y + noise_range[i]*noise, tau, ami_method, extra_param)
            if np.isnan(amis[i]):
                logger.warning('Error computing AMI: Time series too short (?)')
                return np.nan
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
    c = amis - np.mean(amis)
    out['pcrossmean'] = np.sum(c[:-1] * c[1:] < 0) / (num_repeats - 1)

    # Fit exponential decay model
    a, b = fit_exp1(noise_range, amis, start_point=(amis[0], -1))
    gof = goodness_of_fit(amis, a * np.exp(b * noise_range), num_coeffs=2)
    out['fitexpa'] = a
    out['fitexpb'] = b
    out['fitexpr2'] = gof['rsquare']
    out['fitexpadjr2'] = gof['adjrsquare']
    out['fitexprmse'] = gof['rmse']

    # Fit linear function
    p = np.polyfit(noise_range, amis, 1)
    out['fitlina'], out['fitlinb'] = p
    lin_fit = np.polyval(p, noise_range)
    out['linfit_mse'] = np.mean((lin_fit - amis)**2)

    return out

def first_under_fn(x: ArrayLike, m: ArrayLike, p: ArrayLike) -> float:
    """
    Find the value of m for the first time p goes under the threshold, x.
    p and m are vectors of the same length

    Falls back to the last element of `m` when `p` never goes under `x`.
    """
    idx = _first_index_past_threshold(p, x, 'under')

    return m[-1] if idx is None else m[idx]


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
        The time scale (in samples) to use for the embedding. Default is 1.

    Returns
    -------
    float
        The time reversal asymmetry statistic.
    """
    try:
        # columns ordered most- to least-delayed
        embedded = time_delay_embed(y, 3, time_lag, reverse=True)
    except ValueError:
        logger.warning("Time series is too short for the given dimension and lag.")
        return np.nan
    a = embedded[:, 0]
    b = embedded[:, 1]
    c = embedded[:, 2]
    res = np.mean(a * a * b - b*c*c)

    return float(res)

def embed2_angle_tau(y: ArrayLike, max_tau: int) -> dict:
    """
    Angle autocorrelation in a 2-dimensional embedding space.

    Investigates how the autocorrelation of angles between successive points in
    the two-dimensional time-series embedding change as tau varies from
    tau = 1, 2, ..., max_tau.

    Parameters
    ----------
    y : array-like
        The input time series.
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
            logger.warning(f'Time series (N={len(y)}) too short for embedding')
            return np.nan

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
        The input time series.
    tau : int or str, optional
        The time-delay. If 'tau', it will be set to the first zero-crossing of 
        the autocorrelation function (ACF). Default is ``'tau'``.

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
        The input time series.

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
        idx, valid, nb = _ami_hist_binning(y, bin_method, num_bins[i])
        amis = np.zeros(num_taus)
        for j in range(num_taus):  # vary over time lags, tau
            amis[j] = _ami_from_binning(idx, valid, nb, tau_range[j])
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

def _ami_hist_binning(y: ArrayLike, meth: str, num_bins: int):
    """Per-sample bin index for the histogram-AMI estimators.

    The binning depends only on ``y``, ``meth`` and ``num_bins`` --- not on the time
    lag --- so callers that sweep lags (e.g. ``compare_min_ami``) can compute this once
    and reuse it. Returns ``(idx, valid, num_bins)`` where ``idx[k]`` is the 0-based bin
    of ``y[k]`` and ``valid[k]`` is False if the point falls outside the bin range. The
    assignment reproduces ``np.histogram2d``'s bins exactly (last bin right-inclusive).
    """
    y = np.asarray(y)
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

    # Sometimes bins can be added (e.g., with std1 and std2), so redefine num_bins
    num_bins = len(b) - 1
    idx = np.digitize(y, b) - 1
    idx[y == b[-1]] = num_bins - 1                 # histogram2d's last bin is right-inclusive
    valid = (idx >= 0) & (idx <= num_bins - 1)
    return idx, valid, num_bins


def _ami_from_binning(idx: np.ndarray, valid: np.ndarray, num_bins: int, t: int) -> float:
    """AMI at lag ``t`` from precomputed bin indices.

    Bit-identical to histogram_ami's per-lag value: the joint histogram of the binned
    delay pair is one ``bincount`` of paired indices instead of re-running histogram2d.
    """
    if t == 0:
        # for tau = 0, y1 and y2 are identical to y
        bi = bj = idx
        m = valid
    else:
        bi = idx[:-t]
        bj = idx[t:]
        m = valid[:-t] & valid[t:]
    # Joint distribution of the (binned) delay pair
    pij = np.bincount(bi[m] * num_bins + bj[m],
                      minlength=num_bins * num_bins).reshape(num_bins, num_bins).astype(float)
    pij = pij / np.sum(pij)  # normalize
    pi = np.sum(pij, axis=1)  # marginal
    pj = np.sum(pij, axis=0)  # other marginal

    pii = np.tile(pi, (num_bins, 1)).T
    pjj = np.tile(pj, (num_bins, 1))

    r = pij > 0  # Defining the range in this way, we set log(0) = 0
    return np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))


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
        The input time series.
    tau : int, list, or str, optional
        The time-lag(s). Can be an integer time lag, list of time lags, or 'ac'/'tau' to use
        first zero-crossing of autocorrelation function. Default is 1.
    meth : str, optional
        The method for binning data:

        - 'even': evenly-spaced bins through the range
        - 'std1': bins extending to ±1 standard deviation from mean
        - 'std2': bins extending to ±2 standard deviations from mean
        - 'quantiles': equiprobable bins using quantiles

        Default is ``'even'``.
        
    num_bins : int, optional
        The number of bins to use. Default is 10.

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

    # Bin the data once (the binning is the same for both delay vectors and does not
    # depend on the lag), then evaluate each lag from the precomputed bin indices.
    idx, valid, num_bins = _ami_hist_binning(y, meth, num_bins)

    # Form the time-delay vectors y1 and y2
    if not isinstance(tau, (list, np.ndarray)):
        # if only single time delay as integer, make into a one element list
        tau = [tau]

    amis = np.array([_ami_from_binning(idx, valid, num_bins, t) for t in tau])

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}

def stick_angles(y: ArrayLike) -> dict:
    """
    Analysis of the line-of-sight angles between time series data points. 

    Line-of-sight angles between time-series pts. treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Parameters
    -----------
    y : array-like
        The input time series.

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
    # std_p/std_n dropped: r=0.98 with pos_neg_asymmetry's volPos/volNeg on
    # real EEG data, which measures regime-conditional volatility more directly.
    out['mean_p'] = np.nanmean(angles[0]) 
    out['median_p'] = np.nanmedian(angles[0])

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
    
    # statav2_all_s/statav3_all_s/statav4_all_s dropped: mutually r=0.97-0.98
    # with statav5_all_s (and each other) on real EEG data, so only one
    # representative of the StatAv-spread-of-all-angles family is kept.
    # StatAv2
    out['statav2_all_m'], _ = _sub_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], _ = _sub_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], _ = _sub_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = _sub_statav(zallAngles, 5)
    
    # correlations? 
    # Note: ac2_p/ac2_n/ac2_all dropped (r=0.96-0.99 with the corresponding
    # ac1_*), and tau_all/ac1_all dropped (each r=0.95-0.97 with its own p/n
    # split) -- keeping the p/n split (rather than the pooled 'all') preserves
    # the asymmetry signal that's the actual point of this feature.
    if len(zangles[0]) > 0:
        out['tau_p'] = first_crossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = autocorr(zangles[0], 1, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan
    
    if len(zangles[1]) > 0:
        out['tau_n'] = first_crossing(zangles[1], 'ac', 0, 'continuous')
        out['ac1_n'] = autocorr(zangles[1], 1, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan

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

def _sub_statav(x: ArrayLike, n: int) -> tuple:
    # helper function
    nn = len(x)
    if nn < 2 * n: # not long enough
        statavmean = np.nan
        statavstd = np.nan
    else:
        x_buff = make_mat_buffer(x, int(np.floor(nn/n)))
        if x_buff.shape[1] > n:
            # remove final pt
            x_buff = x_buff[:, :n]
        statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
        statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def joint_non_gaussianity(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 2,
                          theiler_win: int = 1,
                          max_n: Union[int, str] = 10000) -> dict:
    """
    Tests for non-Gaussianity of the joint, time-lagged embedding distribution.

    Embeds the time series in m dimensions at time delay tau (e.g., the pair
    (x_t,x_{t+tau}) for m=2, or the triple (x_t,x_{t+tau},x_{t+2tau}) for
    m=3) and tests whether the resulting point cloud is consistent with a
    multivariate Gaussian.

    cf. existing Gaussianity tests acting on marginal distribution
    (distribution_test, compare_ks_fit), which all
    A linear (e.g., AR(1)) Gaussian process has a Gaussian marginal
    *and* a Gaussian joint embedding distribution;
    a nonlinear or non-reversible process can look
    Gaussian marginally while its lagged joint distribution is visibly
    non-elliptical (curved, multimodal, or heavy/light-tailed along
    directions the marginal alone cannot see).
    cf. also time-irreversibility metrics like trev/tc3
    (which use a single third-moment statistic of lagged
    pairs/triples as a nonlinearity probe) but this is perhaps more general as
    : it tests the whole joint shape rather than one moment combination.

    Two complementary statistics are computed, both based on Mardia's
    (1970) classical multivariate normality measures, chosen because they
    generalize to any embedding dimension m via the same formula (so m=2
    and m=3 are the same code path) and because they reduce, at m=1, to
    ordinary skewness/kurtosis -- the natural multivariate extension of
    what moments already computes:

      (i)  Mardia's multivariate skewness, b1 -- detects asymmetry/curvature
           of the joint distribution (e.g., a banana-shaped point cloud).
           Population value is 0 for any joint Gaussian.
      (ii) Mardia's multivariate kurtosis, b2 -- detects joint tail weight/
           peakedness relative to a Gaussian ellipsoid. Population value is
           m(m+2) for any joint Gaussian (e.g., 8 at m=2, 15 at m=3).

    As a complementary, distribution-shape-sensitive check, the squared
    Mahalanobis distances of each embedded point to the sample mean (which
    are exactly the per-point terms underlying Mardia's kurtosis) are
    compared against their theoretical shape under joint Gaussianity,
    chi^2_m, via a Kolmogorov-Smirnov D-statistic -- this can catch
    departures (e.g., a bimodal or ring-shaped cloud) that the two summary
    moments can miss.

    NOTE ON SIGNIFICANCE: only *raw* statistics are returned, not p-values.
    Mardia's classical asymptotic null distributions assume the N embedded
    points are iid draws, but consecutive embedded vectors overlap in m-1
    coordinates and are therefore strongly autocorrelated, which inflates
    the naive asymptotic test statistics (empirically, up to ~30% false
    positives at a nominal 5% level on a purely linear-Gaussian AR(1)
    process, worse at higher m). This is the same reason trev/tc3
    report raw statistics rather than p-values; for significance
    testing against a null that respects the series' own autocorrelation
    structure, compare these statistics to their distribution over
    surrogates (cf. surrogate_test, make_surrogates).

    The skewness statistic additionally excludes near-diagonal pairs
    (|i-j| <= theiler_win) from its double sum: for correlated (not just
    independent) jointly-Gaussian points, the third moment of their
    Mahalanobis inner product is *not* zero (only the independent case has
    this symmetry), so nearby, strongly-autocorrelated pairs bias the raw
    statistic away from zero even under true joint Gaussianity. (The
    same rationale as rqa's Theiler window, but applied to a third-moment sum
    instead of a distance threshold).
    Empirically this removes most, but not
    all, of the bias (e.g., at m=7 on an AR(1) process, ~0.18 -> ~0.07); the
    residual is the classical small-sample bias of using the *sample*
    covariance to whiten the same points being tested (present even for iid
    data), which widening the window further does not touch.

    References
    ----------
    .. [1] K.V. Mardia, "Measures of multivariate skewness and kurtosis with
        applications", Biometrika 57(3) 519 (1970).

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time delay for the embedding (can be 'ac' or 'mi', or an
        integer). Default: 'ac'.
    m : int, optional
        The embedding dimension. Default: 2, for the pairwise joint
        distribution (x_t,x_{t+tau}); set to 3 for the triple-wise joint
        distribution (x_t,x_{t+tau},x_{t+2tau}).
    theiler_win : int, optional
        The number of temporally-adjacent embedded points excluded from the
        skewness double sum (|i-j| <= theiler_win), to reduce the
        correlated-pair bias described above. Default: 1.
    max_n : int or str, optional
        The maximum number of embedded points used for the skewness
        statistic, whose cost is O(N^2) (it involves all pairwise
        Mahalanobis inner products, unlike the kurtosis and KS statistics,
        which are both O(N)). The mean, covariance, kurtosis, and KS
        statistic always use the full embedded series; only the skewness
        statistic is computed from the first max_n embedded points when the
        series is longer than this (default: 10000, i.e., an 800MB Gram
        matrix, ~1s; a warning is issued whenever this cropping actually
        happens, since the skewness estimate is still visibly noisy even at
        10000 points and keeps improving with more -- this is a memory/time
        cap, not a convergence point. Can be set to 'full' to disable, with
        a second warning above 20000 points where the ~3.2GB+ Gram matrix
        becomes a serious memory cost).

    Returns
    -------
    dict
        Mardia's raw multivariate skewness (Theiler-windowed) and multivariate
        kurtosis, and the Mahalanobis-distance-vs-chi^2 Kolmogorov-Smirnov
        D-statistic. All are unitless departure-from-joint-Gaussianity
        magnitudes with no attached significance level (see note above).
    """
    y = np.asarray(y, dtype=float).ravel()

    # Embed the signal
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        logger.warning('Embedding failed')
        return np.nan
    try:
        Y = time_delay_embed(y, m, int(tau))
    except ValueError:
        logger.warning('Embedding failed')
        return np.nan
    n_emb, d = Y.shape

    # Need enough points to reliably estimate a d x d covariance matrix and
    # for the higher-moment statistics below to be reasonably stable:
    min_n = max(30, 10 * d * (d + 2))
    if n_emb < min_n:
        logger.warning(f'Too few embedded points ({n_emb}) for a meaningful '
                       f'joint-Gaussianity test at m = {d}')
        return np.nan

    # Center and whiten
    mu = np.mean(Y, axis=0)
    Yc = Y - mu
    S = (Yc.T @ Yc) / n_emb # Mardia's convention: divide by N, not N-1

    try:
        L = np.linalg.cholesky(S)
        near_singular = 1.0 / np.linalg.cond(S, 1) < 1e-10
    except np.linalg.LinAlgError:
        near_singular = True
    if near_singular:
        # Near-singular covariance -- typically means tau is too small relative
        # to the series' correlation length, so consecutive embedded
        # coordinates are nearly collinear
        logger.warning('Embedded covariance matrix is near-singular (tau too small?)')
        return np.nan

    X = solve_triangular(L, Yc.T, lower=True) # d x n_emb; column i is the whitened point L^{-1}*(Y[i,:]-mu)'
    D2 = np.sum(X**2, axis=0) # squared Mahalanobis distances, n_emb

    out = {}

    # Mardia's multivariate kurtosis (O(N), uses all embedded points)
    # Raw statistic; population value under joint Gaussianity is d*(d+2)
    # regardless of dimension or sample dependence structure.
    out['mardiaKurt'] = float(np.mean(D2**2))

    # Mahalanobis-distance-vs-chi^2_d Kolmogorov-Smirnov statistic (O(N))
    out['mahalKSstat'] = _kstest_statistic(D2, lambda v: chi2.cdf(v, d))

    # Mardia's multivariate skewness (O(N^2): subsample if needed)
    if isinstance(max_n, str) and max_n == 'full':
        slow_threshold = 20000
        if n_emb > slow_threshold:
            logger.warning(f'{n_emb} embedded points exceeds {slow_threshold} with '
                           f"max_n='full'; skewness Gram matrix may use substantial "
                           f'memory (>{8 * n_emb**2 / 1e9:.1f}GB)')
        X_skew = X
    elif n_emb > max_n:
        logger.warning(f'Cropping to the first {max_n} of {n_emb} embedded points for '
                       'the skewness statistic (memory/time cap, not a convergence '
                       'point -- the estimate is still noisy at this size and would '
                       'keep improving with more data; raise max_n or use \'full\' '
                       'for a more precise, more expensive estimate)')
        X_skew = X[:, :max_n]
    else:
        X_skew = X
    n_skew = X_skew.shape[1]

    n_band = sum(n_skew - abs(k) for k in range(-theiler_win, theiler_win + 1)
                 if abs(k) < n_skew)
    n_off_band = n_skew**2 - n_band
    if n_off_band <= 0:
        logger.warning('theiler_win too large relative to the (possibly subsampled) '
                       'skewness sample size')
        out['mardiaSkew'] = np.nan
    else:
        G = X_skew.T @ X_skew # n_skew x n_skew Gram matrix of Mahalanobis inner products
        G3 = np.power(G, 3, out=G)
        band_sum = sum(np.trace(G3, offset=k)
                       for k in range(-theiler_win, theiler_win + 1)
                       if abs(k) < n_skew)
        out['mardiaSkew'] = float((G3.sum() - band_sum) / n_off_band)

    return out

def _theiler_kth(idx: np.ndarray, dist: np.ndarray, k: int, theiler_win: int,
                 ref_set: np.ndarray, query_set: np.ndarray) -> np.ndarray:
    # For each query point i, returns the k-th nearest-neighbor distance within
    # ref_set, excluding any candidate with |i-j| <= theiler_win. Falls back to an
    # exact brute-force search for any point whose over-fetched candidate list
    # (idx, dist) didn't leave at least k valid (non-excluded) neighbors.
    n = idx.shape[0]
    n_ref = ref_set.shape[0]
    i = np.arange(n)[:, None]
    valid = np.abs(idx - i) > theiler_win
    # position of the k-th valid candidate in each (distance-sorted) row:
    kth_hit = valid & (np.cumsum(valid, axis=1) == k)
    has_kth = kth_hit.any(axis=1)
    pos = np.argmax(kth_hit, axis=1)

    kth = np.full(n, np.nan)
    rows = np.flatnonzero(has_kth)
    kth[rows] = dist[rows, pos[rows]]

    time_idx = np.arange(n_ref)
    for i in np.flatnonzero(~has_kth):
        all_dists = np.sqrt(np.sum((ref_set - query_set[i]) ** 2, axis=1))
        all_dists[np.abs(time_idx - i) <= theiler_win] = np.inf
        kth_dist = np.partition(all_dists, k - 1)[k - 1]
        if np.isfinite(kth_dist):
            kth[i] = kth_dist

    return kth


def _knn_kld(A: np.ndarray, B: np.ndarray, k: int, theiler_win: int) -> float:
    # Wang-Kulkarni-Verdu (2009) k-NN estimator of KL(P_A||P_B), where A and B
    # are equal-sized (n x d) point sets whose rows are in temporal
    # correspondence (row i of A and row i of B share the same underlying time
    # index), so a Theiler window can be applied consistently in both searches.
    n, d = A.shape
    k_fetch = min(n - 1, k + 2 * theiler_win + 5)

    # Self-search within A (for r_k(x_i), the k-th NN of x_i within A\{x_i}):
    dist_a, idx_a = cKDTree(A).query(A, k=k_fetch + 1, workers=-1)
    rk = _theiler_kth(idx_a, dist_a, k, theiler_win, A, A)

    # Cross-search from A into B (for s_k(x_i), the k-th NN of x_i within B):
    dist_b, idx_b = cKDTree(B).query(A, k=k_fetch, workers=-1)
    sk = _theiler_kth(idx_b, dist_b, k, theiler_win, B, A)

    good = np.isfinite(rk) & np.isfinite(sk) & (rk > 0) & (sk > 0)
    n_good = int(np.count_nonzero(good))
    if n_good < 0.5 * n:
        return np.nan

    return float((d / n_good) * np.sum(np.log(sk[good] / rk[good]))
                 + np.log(n / (n - 1)))


def time_rev_kld(y: ArrayLike, tau: Union[int, str] = 'ac', m: int = 2, k: int = 3,
                 theiler_win: int = 1, max_n: Union[int, str] = 'full') -> dict:
    """
    Kullback-Leibler divergence between forward and time-reversed embeddings.

    Embeds the time series in m dimensions at time delay tau (e.g., the pair
    (x_t,x_{t+tau}) for m=2, or the triple (x_t,x_{t+tau},x_{t+2tau}) for
    m=3), and estimates the Kullback-Leibler divergence between the
    distribution of these embedded points and the distribution of the same
    points with their coordinate order reversed (equivalent to embedding
    the time-reversed series). For a (statistically) time-reversible
    process, these two distributions coincide and the divergence is zero in
    the population; departures reflect time-irreversibility.

    cf. trev/tc3, which probe irreversibility via a single third-moment
    statistic of lagged pairs/triples; this is the natural full-density
    generalization (in the same sense that joint_non_gaussianity generalizes
    moments' skewness/kurtosis to the whole joint embedding shape), able to
    detect any asymmetry between the forward and reversed distributions, not
    just a third-moment one.

    cf. also Diks, van Houwelingen, Takens & DeGoede (1995), who compare
    forward and reverse embedding distributions via a different
    (U-statistic-based) route; the approach here instead estimates the
    Kullback-Leibler divergence directly using the k-NN estimator of Wang,
    Kulkarni & Verdu (2009), which needs no binning or KDE grid and remains
    usable at hctsa-scale sample sizes in m = 2 or 3 dimensions.

    The k-NN search follows the same over-fetch-then-Theiler-filter pattern as
    local_density: candidate neighbors are fetched via a KD-tree and any
    within a Theiler window of the query point's time index are discarded (a
    point and its temporal neighbors are strongly autocorrelated, so treating
    them as informative near-neighbors would understate the local spread);
    the rare point left with too few valid candidates falls back to an exact
    brute-force search.

    NOTE ON DIRECTIONALITY: KL(P||Q) and KL(Q||P) are generally different
    quantities, but *not* here: the reversed embedding Q is built by flipping
    the coordinate order of each row of P, an isometry (it preserves all
    pairwise distances, including cross-set ones), applied identically at
    every matching time index. Any purely distance-based two-sample
    divergence estimator -- like the k-NN one used here -- is therefore
    forced to return numerically identical values for KL(P||Q) and KL(Q||P)
    (verified to match to machine precision on synthetic test series); only
    one direction is computed.

    NOTE ON SIGNIFICANCE: only a raw divergence estimate is returned, not a
    p-value -- consecutive embedded points overlap in m-1 coordinates and are
    not independent, the same reason trev/tc3/joint_non_gaussianity report
    raw statistics only. For significance testing against a null that
    respects the series' own autocorrelation structure, compare this
    statistic to its distribution over surrogates (cf. make_surrogates,
    surrogate_test).

    References
    ----------
    .. [1] C. Diks, J.C. van Houwelingen, F. Takens, J. DeGoede, "Reversibility
           as a criterion for discriminating time series", Phys. Lett. A
           201(4-5) 221 (1995).
    .. [2] Q. Wang, S.R. Kulkarni, S. Verdu, "Divergence Estimation for
           Multidimensional Densities via k-Nearest-Neighbor Distances", IEEE
           Trans. Inf. Theory 55(5) 2392 (2009).

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time delay for the embedding (can be 'ac' or 'mi', or an
        integer). Default: 'ac'.
    m : int, optional
        The embedding dimension. Default: 2, for the pairwise joint
        distribution (x_t,x_{t+tau}); set to 3 for the triple-wise joint
        distribution (x_t,x_{t+tau},x_{t+2tau}).
    k : int, optional
        The number of nearest neighbors used by the k-NN divergence estimator.
        Default: 3 (matches local_density's default).
    theiler_win : int, optional
        The number of temporally-adjacent points excluded from both the
        within-set and cross-set neighbor searches (|i-j| <= theiler_win),
        applied at matching time indices in both the forward and reversed
        embeddings. Default: 1.
    max_n : int or str, optional
        The maximum number of embedded points used. The k-NN searches are
        KD-tree-based, not the O(N^2) cost of joint_non_gaussianity's
        skewness statistic (which needs this kind of cap). A warning is issued
        whenever cropping actually happens. Default: 'full' (no cropping); set
        to an integer to cap runtime on unusually long series.

    Returns
    -------
    dict
        The raw k-NN estimate of KL(forward || reversed) and its magnitude. A
        unitless departure-from-reversibility measure, zero up to estimation
        noise for a reversible process (the k-NN estimator can dip slightly
        negative near a true value of zero -- expected behavior of this
        nonparametric estimator, not a bug), with no attached significance
        level (see note above).
    """
    y = np.asarray(y, dtype=float).ravel()

    # Embed the signal
    tau = _resolve_time_delay(y, tau)
    if np.isnan(tau):
        logger.warning('Embedding failed')
        return np.nan
    try:
        Y = time_delay_embed(y, m, int(tau))
    except ValueError:
        logger.warning('Embedding failed')
        return np.nan
    n_emb, d = Y.shape

    min_n = max(50, 10 * (k + theiler_win))
    if n_emb < min_n:
        logger.warning(f'Too few embedded points ({n_emb}) for a meaningful '
                       f'time-reversal KLD estimate at m = {d}')
        return np.nan

    if isinstance(max_n, str) and max_n == 'full':
        pass # no cropping
    elif n_emb > max_n:
        logger.warning(f'Cropping to the first {max_n} of {n_emb} embedded points '
                       "(runtime cap; raise max_n or use 'full' to disable)")
        Y = Y[:max_n, :]
        n_emb = max_n

    # Forward and time-reversed embeddings
    # Reversing the coordinate order of each embedded row is equivalent to
    # embedding the time-reversed series (up to the boundary points, which this
    # avoids re-deriving from scratch):
    P = Y
    Q = Y[:, ::-1]

    # k-NN estimate of KL(P||Q)
    # (KL(Q||P) is numerically identical -- see NOTE ON DIRECTIONALITY above --
    # so only one direction is computed.)
    out = {}
    out['raw'] = _knn_kld(P, np.ascontiguousarray(Q), k, theiler_win)
    out['abs'] = abs(out['raw'])

    return out

def falling_sticks(y: ArrayLike) -> dict:
    """
    Physical falling-sticks model of line-of-sight interaction.

    As in stick_angles, each time-series value is treated as a rigid stick
    standing on the zero baseline, with sticks grouped by sign into a
    'positive' set (protruding up from the zero level) and a 'negative' set
    (protruding down). Here, sticks are toppled: each stick rotates about its
    base towards later same-sign sticks and stops at whichever angle first
    brings it into contact with one -- either its trunk striking the side of
    a taller later stick, or its underside striking the tip of a shorter one
    it topples clean over -- or else it falls flat (angle = pi/2) if no
    later same-sign stick lies within reach (a stick of height h can only
    ever reach as far as horizontal distance h).

    This differs from stick_angles, which only ever compares a stick to
    its immediate same-sign successor via the slope between them.
    falling_sticks instead allows a stick to skip over intervening sticks
    to hit a farther one, so it is sensitive to range-dependent local-
    extremum structure (e.g. a tall stick toppling clean over an
    intervening short one) that the purely local (i, i+1) comparison cannot
    see.

    Adapted from a Python 'FALLstick' reference implementation by Eugene Chon
    <eugenechon04@gmail.com>.

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored: the sign split is around the
        mean, matching stick_angles's convention).

    Returns
    -------
    dict
        Statistics on the resulting fall-angle sequence (location, spread,
        shape, persistence), on the asymmetry between the positive and
        negative branches, and on the three collision types a fall can end in
        -- falling flat, hitting the immediately next stick, or skipping over
        one or more sticks to hit a farther one -- and the two ways a hit can
        occur -- trunk-strike (case 1) vs. tip-strike/topple-over (case 2).
    """
    y = np.asarray(y).flatten()

    ix_pos = np.where(y >= 0)[0]
    ix_neg = np.where(y < 0)[0]

    angles_pos, colour_pos, case_pos = _fall_branch(ix_pos, y)
    angles_neg, colour_neg, case_neg = _fall_branch(ix_neg, y)

    all_angles = np.concatenate((angles_pos, angles_neg))

    out = {}

    # Location and spread of the fall-angle distribution
    out['mean_p'] = _fall_safe_stat(np.mean, angles_pos)
    out['median_p'] = _fall_safe_stat(np.median, angles_pos)
    out['mean_n'] = _fall_safe_stat(np.mean, angles_neg)
    out['median_n'] = _fall_safe_stat(np.median, angles_neg)

    out['mean_all'] = _fall_safe_stat(np.mean, all_angles)
    out['median_all'] = _fall_safe_stat(np.median, all_angles)
    out['std_all'] = _fall_safe_stat(
        lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0, all_angles)

    # Asymmetry between the positive- and negative-branch fall angles:
    if not np.isnan(out['mean_p']) and not np.isnan(out['mean_n']):
        out['diff_pn'] = out['mean_p'] - out['mean_n']
    else:
        out['diff_pn'] = np.nan

    # Collision-type proportions: flat / immediate-hit / skip-hit
    out['propFlat_p'] = _fall_prop_first(colour_pos)
    out['propFlat_n'] = _fall_prop_first(colour_neg)
    out['propFlat_all'] = _fall_prop_first(colour_pos + colour_neg)

    out['propSkip_p'] = _fall_prop_skip(colour_pos)
    out['propSkip_n'] = _fall_prop_skip(colour_neg)
    out['propSkip_all'] = _fall_prop_skip(colour_pos + colour_neg)

    # Hit-type proportions: trunk-strike (case 1) vs. tip-strike/topple-over (case 2)
    out['propCase2_p'] = _fall_prop_case2(case_pos)
    out['propCase2_n'] = _fall_prop_case2(case_neg)
    out['propCase2_all'] = _fall_prop_case2(case_pos + case_neg)

    # Distribution shape
    # (The 90th percentile is omitted: across two independent redundancy-check
    # datasets it landed exactly on the pi/2 flat-fall spike every time, since
    # most series have >=10% flat falls -- it carries no information.)
    if len(all_angles) >= 2:
        out['skewness_all'] = skew(all_angles)
        out['kurtosis_all'] = kurtosis(all_angles, fisher=False)
        out['q10_all'] = np.quantile(all_angles, 0.1, method='hazen')
    else:
        out['skewness_all'] = np.nan
        out['kurtosis_all'] = np.nan
        out['q10_all'] = np.nan

    # Persistence of the fall-angle sequence
    if len(angles_pos) >= 2 and np.std(angles_pos, ddof=1) > 0:
        z_angles_pos = z_score(angles_pos)
        out['tau_p'] = first_crossing(z_angles_pos, 'ac', 0, 'continuous')
        out['ac1_p'] = autocorr(z_angles_pos, 1, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan

    if len(angles_neg) >= 2 and np.std(angles_neg, ddof=1) > 0:
        z_angles_neg = z_score(angles_neg)
        out['tau_n'] = first_crossing(z_angles_neg, 'ac', 0, 'continuous')
        out['ac1_n'] = autocorr(z_angles_neg, 1, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan

    return out

def _fall_branch(ix: ArrayLike, y: ArrayLike) -> tuple:
    # Topples each stick in a same-sign index list ix towards later sticks in
    # the same list, and returns:
    #   angles       -- fall angle for each processed stick (all but the last,
    #                    which trivially always falls flat and so contributes
    #                    no angle of its own)
    #   colour_counts -- [num_flat, num_immediate_hit, num_skip_hit], summing to
    #                    len(ix) (the forced-flat last stick included)
    #   case_counts   -- [num_case1, num_case2], summing to len(ix) (the
    #                    forced-flat last stick counted as case 1)
    nj = len(ix)
    if nj == 0:
        # No sticks at all in this branch (e.g. an all-positive or
        # all-negative series): nothing falls, so there is no phantom
        # 'last bar' to count as a flat fall either.
        return np.zeros(0), np.array([0, 0, 0]), np.array([0, 0])
    if nj == 1:
        # the lone stick always falls flat
        return np.zeros(0), np.array([1, 0, 0]), np.array([1, 0])

    angles = np.zeros(nj - 1)
    colour_flag = np.zeros(nj - 1, dtype=int) # 0 = flat, 1 = immediate hit, 2 = skip hit
    case_flag = np.zeros(nj - 1, dtype=int) # 1 or 2

    for i in range(nj - 1):
        x1 = ix[i]
        y1 = y[x1]
        height1 = abs(y1)

        min_angle = np.pi/2 # default: falls flat
        fall_k = i # lands on itself if flat
        min_case1 = True

        if height1 > 0: # a zero-height stick is already lying flat
            for k in range(i + 1, nj):
                x2 = ix[k]
                y2 = y[x2]
                dx = x2 - x1 # a stick of height1 can reach at most height1
                if dx > height1:
                    break # all later sticks are farther still -- out of reach

                # Case 1 (trunk-strike): stick 1's tip hits the side of stick
                # 2. Case 2 (tip-strike): stick 1 is taller than stick 2 and
                # would clear its top before reaching dx, so instead its
                # underside comes down onto stick 2's tip.
                is_case1 = True
                if abs(y1) > abs(y2) and height1 * np.sin(np.arccos(y2 / y1)) > dx:
                    is_case1 = False

                if is_case1:
                    angle = np.arcsin(dx / height1)
                else:
                    angle = np.arctan(dx / abs(y2))
                angle = min(angle, np.pi - angle)

                if angle < min_angle:
                    min_angle = angle
                    fall_k = k
                    min_case1 = is_case1

        angles[i] = min_angle
        if fall_k == i:
            colour_flag[i] = 0
        elif fall_k == i + 1:
            colour_flag[i] = 1
        else:
            colour_flag[i] = 2
        case_flag[i] = 2 - int(min_case1) # True (case 1) -> 1, False (case 2) -> 2

    colour_counts = np.array([np.sum(colour_flag == 0) + 1, np.sum(colour_flag == 1),
                              np.sum(colour_flag == 2)])
    case_counts = np.array([np.sum(case_flag == 1) + 1, np.sum(case_flag == 2)])

    return angles, colour_counts, case_counts

def _fall_safe_stat(f, x: ArrayLike):
    # helper function
    if len(x) == 0:
        return np.nan
    return f(x)

def _fall_prop_first(colour_counts: ArrayLike) -> float:
    # helper function
    total = np.sum(colour_counts)
    if total == 0:
        return np.nan
    return colour_counts[0] / total

def _fall_prop_skip(colour_counts: ArrayLike) -> float:
    # helper function
    non_flat = colour_counts[1] + colour_counts[2]
    if non_flat == 0:
        return np.nan
    return colour_counts[2] / non_flat

def _fall_prop_case2(case_counts: ArrayLike) -> float:
    # helper function
    total = np.sum(case_counts)
    if total == 0:
        return np.nan
    return case_counts[1] / total

def oversampling(y: ArrayLike) -> dict:
    """
    Detects temporal oversampling relative to a series' own dynamics.

    Implements the oversampling-detection statistic eta (and its downsampling
    correction) from the 'oversampling' stage of the Chaos Decision Tree
    Algorithm [1]:

    .. math::

        \\eta = \\frac{\\mathrm{range}(y)}{\\langle |\\Delta y| \\rangle}

    A large eta means consecutive samples are, on average, tiny relative to
    the full dynamic range the series explores -- i.e., the series is sampled
    much faster than its own dynamics move, so consecutive points are close
    to redundant. Toker et al. flag eta > 10 as 'oversampled': this inflates
    the apparent smoothness/determinism of a series and can bias downstream
    nonlinear statistics (their motivation: left uncorrected, it distorts
    their 0-1 test for chaos -- cf. ``zero_one_test``). Their correction is to
    iteratively halve the sampling rate (keep every second point) until
    eta <= 10 or fewer than 100 points remain.

    cf. ``zero_one_test`` for the chaos-classification stage of the same
    pipeline, and ``permutation_entropy``/``surrogate_test`` for its
    stochasticity-testing stage; this function covers the
    oversampling-diagnosis stage instead. Note eta is scale- and
    location-invariant (a ratio of two amplitude-unit quantities), so
    z-scored or raw y give identical results.

    References
    ----------
    .. [1] Toker, D. et al. "A simple method for detecting chaos in nature",
        Commun. Biol. 3, 11 (2020). DOI: 10.1038/s42003-019-0715-9

    Parameters
    ----------
    y : array-like
        The input time series.

    Returns
    -------
    dict
        Dictionary containing:
            - 'eta': The oversampling statistic, range(y)/mean(|diff(y)|).
            - 'etaRobust': The same ratio with range replaced by a 5th-95th
              percentile range, since eta's numerator (a global max-min) is a
              single-outlier-sensitive statistic; etaRobust asks the same
              oversampling question without letting one extreme point set the
              scale.
            - 'numHalvings': The number of times Toker et al.'s halving
              procedure would downsample y before eta <= 10 (or fewer than 100
              points would remain) -- a direct, interpretable severity measure.
            - 'etaAfterDownsampling': The value of eta after applying
              numHalvings halvings (<=10, unless the series was too short to
              fully correct).
    """
    # Check inputs:
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    if N < 10:
        # Data-dependent (series too short to assess sampling adequacy), not a code error.
        logger.warning('Time series too short to assess oversampling')
        return {'eta': np.nan, 'etaRobust': np.nan,
                'numHalvings': np.nan, 'etaAfterDownsampling': np.nan}

    mean_abs_diff = float(np.mean(np.abs(np.diff(y))))
    if mean_abs_diff == 0:
        # Data-dependent (constant time series), not a code error.
        logger.warning('Zero mean absolute difference (constant time series?)')
        return {'eta': np.nan, 'etaRobust': np.nan,
                'numHalvings': np.nan, 'etaAfterDownsampling': np.nan}

    out = {}

    # eta and its outlier-robust companion:
    out['eta'] = float(np.ptp(y) / mean_abs_diff)

    q05, q95 = matlab_quantile(y, [0.05, 0.95])
    robust_range = q95 - q05
    out['etaRobust'] = float(robust_range / mean_abs_diff)

    # Iterative halving-downsampling correction (Toker et al., 2020):
    eta_threshold = 10
    min_points = 100

    y_ds = y
    num_halvings = 0
    eta_curr = out['eta']
    while eta_curr > eta_threshold and y_ds.size // 2 >= min_points:
        y_ds = y_ds[::2]
        num_halvings += 1
        m_ds = float(np.mean(np.abs(np.diff(y_ds))))
        if m_ds == 0:
            break # degenerate downsampled segment -- keep prior eta_curr
        eta_curr = float(np.ptp(y_ds) / m_ds)
    out['numHalvings'] = float(num_halvings)
    out['etaAfterDownsampling'] = float(eta_curr)

    return out

def _pn_corr(x: np.ndarray, y: np.ndarray) -> float:
    """The off-diagonal entry of MATLAB's ``corrcoef(x, y)``.

    Mirrors the arithmetic of MATLAB's ``correl`` subfunction: build the
    covariance matrix with the (n-1) normalisation, take the square root of
    the diagonal first (to avoid under/overflow) and divide twice, then limit
    the result to [-1, 1].

    A constant input has no defined correlation, and is reported as NaN. The
    regime subsets here are routinely constant on quantised series (a whole
    regime sitting on one level), and centring such a subset leaves only
    rounding dust, so dividing it through -- as MATLAB does -- returns a sign
    determined entirely by summation order rather than by the data.
    """
    if x[0] == x.min() and x[0] == x.max():
        return np.nan
    if y[0] == y.min() and y[0] == y.max():
        return np.nan
    n = x.size
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    cxy = np.dot(xc, yc) / (n - 1)
    cxx = np.dot(xc, xc) / (n - 1)
    cyy = np.dot(yc, yc) / (n - 1)
    r = cxy / np.sqrt(cyy) / np.sqrt(cxx)
    if abs(r) > 1:
        r = np.sign(r)
    return float(r)


def _pn_std(x: np.ndarray) -> float:
    """MATLAB's ``std``: zero for a single observation, NaN for none."""
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    return float(np.std(x, ddof=1))


def pos_neg_asymmetry(y: ArrayLike) -> dict:
    """
    Asymmetry of local dynamics between positive and negative regimes.

    Splits the time series by the sign of each value (assumes y is z-scored, so
    the split is around the mean) and asks whether the one-step-ahead dynamics
    differ between the two regimes: is the series more volatile, or more
    persistent (higher one-step autocorrelation), when the current value is
    above vs. below its mean? This targets a form of distribution-dynamics
    interaction not captured by ``stick_angles``, which instead compares the
    distribution of local slopes *within* each same-sign subsequence.

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored: the regime split threshold
        is 0).

    Returns
    -------
    dict
        The conditional volatility and one-step autocorrelation of the
        positive/negative regimes, and normalized contrasts between them (the
        volatility contrast is a leverage-effect-style statistic; the
        autocorrelation contrast is a threshold-AR(1)-style statistic). Also
        isolates the two zero-crossing transition types (positive-to-negative,
        negative-to-positive) -- posMask/negMask are conditioned on the current
        value only, so they mix crossing and non-crossing steps together; the
        crossing-specific fields ask instead whether the jump *at* a regime
        switch is itself asymmetric (e.g. sharper downward crossings than
        upward ones).
    """
    y = np.asarray(y, dtype=float).ravel()

    # Regime-conditioned one-step-ahead pairs: yLag(t) = y(t), yNext(t) = y(t+1)
    y_lag = y[:-1]
    y_next = y[1:]
    dy = y_next - y_lag

    pos_mask = (y_lag >= 0)
    neg_mask = ~pos_mask
    n_pos = int(np.count_nonzero(pos_mask))
    n_neg = pos_mask.size - n_pos

    out = {}

    m = pos_mask.size
    # proportion of time spent in the positive regime
    out['propPos'] = n_pos / m if m > 0 else np.nan

    # Conditional volatility (leverage-effect-style asymmetry)
    out['volPos'] = _pn_std(dy[pos_mask]) if n_pos >= 2 else np.nan
    out['volNeg'] = _pn_std(dy[neg_mask]) if n_neg >= 2 else np.nan
    vol_all = _pn_std(dy)
    if n_pos >= 2 and n_neg >= 2 and vol_all > 0:
        # Normalized contrast: positive when the positive regime is more volatile.
        out['volAsym'] = (out['volPos'] - out['volNeg']) / vol_all
    else:
        out['volAsym'] = np.nan

    # Conditional one-step autocorrelation (threshold-AR(1)-style asymmetry)
    out['ac1Pos'] = _pn_corr(y_lag[pos_mask], y_next[pos_mask]) if n_pos >= 2 else np.nan
    out['ac1Neg'] = _pn_corr(y_lag[neg_mask], y_next[neg_mask]) if n_neg >= 2 else np.nan
    # Difference on the (already bounded, comparable) correlation scale:
    out['ac1Asym'] = out['ac1Pos'] - out['ac1Neg']

    # Zero-crossing transitions: positive-to-negative (PN) and negative-to-
    # positive (NP). Unlike posMask/negMask (conditioned on the current value
    # only), these isolate the step *at* which the regime actually switches.
    next_neg = y_next < 0
    pn_mask = pos_mask & next_neg # downward crossing
    np_mask = neg_mask & ~next_neg # upward crossing
    n_pn = int(np.count_nonzero(pn_mask))
    n_np = int(np.count_nonzero(np_mask))

    # proportion of steps that are downward / upward crossings
    out['propPN'] = n_pn / m if m > 0 else np.nan
    out['propNP'] = n_np / m if m > 0 else np.nan

    # Crossing-jump volatility:
    out['volPN'] = _pn_std(dy[pn_mask]) if n_pn >= 2 else np.nan
    out['volNP'] = _pn_std(dy[np_mask]) if n_np >= 2 else np.nan
    if n_pn >= 2 and n_np >= 2 and vol_all > 0:
        # Positive when downward crossings are more violent than upward ones.
        out['volAsymCross'] = (out['volPN'] - out['volNP']) / vol_all
    else:
        out['volAsymCross'] = np.nan

    # Does the pre-crossing value predict the post-crossing value (i.e. does a
    # deeper excursion before crossing predict a deeper overshoot after it)?
    out['ac1PN'] = _pn_corr(y_lag[pn_mask], y_next[pn_mask]) if n_pn >= 2 else np.nan
    out['ac1NP'] = _pn_corr(y_lag[np_mask], y_next[np_mask]) if n_np >= 2 else np.nan
    out['ac1AsymCross'] = out['ac1PN'] - out['ac1NP']

    return out

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
        The z-scored input time series.

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
        - If ``None``, sets ``absval=True`` when ``len(taus)`` is even and
            ``absval=False`` when ``len(taus)`` is odd.

        Default is ``None``.

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

def autocorr_x2(y: ArrayLike, taus: ArrayLike = 1,
                what_direction: str = 'forward') -> np.ndarray:
    """
    Asymmetric squared cross-correlation of a time series.

    Computes a lag-resolved generalization of the 'leverage effect' correlation
    used to detect asymmetric volatility feedback [1]_: instead of the usual
    autocorrelation :math:`\\langle x_t\\, x_{t+\\tau} \\rangle`, one term is
    squared:

    .. math::

        \\text{forward:}\\quad \\langle x_t\\, x_{t+\\tau}^2 \\rangle

        \\text{backward:}\\quad \\langle x_t^2\\, x_{t+\\tau} \\rangle

    The forward statistic asks whether the (signed) value now predicts the
    squared (unsigned/energy) value later; the backward statistic asks whether
    the squared value now predicts the (signed) value later. For time-reversible,
    linear processes the two are equal (up to sampling noise); a systematic
    difference between them is a signature of nonlinear, time-irreversible
    structure such as volatility clustering with a leverage (asymmetric)
    feedback.

    Note that ``nonlinear_autocorr(y, [0, tau], False)`` and
    ``nonlinear_autocorr(y, [tau, tau], False)`` already compute the forward and
    backward statistics (respectively) at a single lag; this function computes
    both directions efficiently across a whole vector of lags.

    References
    ----------
    .. [1] J.-P. Bouchaud, A. Matacz and M. Potters, "Leverage effect in financial
        markets: The retarded volatility model", Phys. Rev. Lett. 87, 228701 (2001)

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored: zero mean, unit variance).

    taus : array-like of int, optional
        A vector of (non-negative, integer) time lags to compute the statistic at.
        ``tau = 0`` reduces to the skewness, ``mean(y**3)``, for both directions.
        Default is ``1``.

    what_direction : {"forward", "backward"}, optional
        Which direction to compute.

        - ``"forward"``: :math:`\\langle x_t\\, x_{t+\\tau}^2 \\rangle`
        - ``"backward"``: :math:`\\langle x_t^2\\, x_{t+\\tau} \\rangle`

        Default is ``'forward'``.

    Returns
    -------
    array
        The requested statistic at each lag (same length as ``taus``).
    """
    y = np.asarray(y)
    taus = np.atleast_1d(np.asarray(taus))
    N = len(y)

    if np.max(taus) > N - 1:
        logger.warning(f"Time lag {np.max(taus)} is too long for time-series length {N}.")
    if np.any(taus < 0):
        raise ValueError("taus must be non-negative (this is an asymmetric statistic -- "
                         "use what_direction to choose the direction).")
    if what_direction not in ('forward', 'backward'):
        raise ValueError(f"Unknown what_direction '{what_direction}' "
                         "(should be 'forward' or 'backward').")

    out = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        tau = int(tau)
        y_later = y[tau:N]     # x_{t+tau}
        y_earlier = y[:N-tau]  # x_t
        if what_direction == 'forward':
            # <x_t.x_{t+tau}^2>
            out[i] = np.mean(y_earlier * y_later**2)
        else:
            # <x_t^2.x_{t+tau}>
            out[i] = np.mean(y_earlier**2 * y_later)

    return out

def autocorr_x2_shape(y: ArrayLike, max_lag: Union[int, str] = 'double_drown') -> dict:
    """
    Shape of the time-reversibility profile of a time series.

    :func:`autocorr_x2` computes two asymmetric, 'leverage'-type lag-profiles:

    .. math::

        \\text{forward}(\\tau) = \\langle x_t\\, x_{t+\\tau}^2 \\rangle
        \\quad\\text{(signed value now, energy later)}

        \\text{backward}(\\tau) = \\langle x_t^2\\, x_{t+\\tau} \\rangle
        \\quad\\text{(energy now, signed value later)}

    For a time-reversible process these coincide at every lag (any shared linear
    correlation structure contributes equally to both); a systematic difference,
    :math:`\\text{diff}(\\tau) = \\text{forward}(\\tau) - \\text{backward}(\\tau)`,
    is therefore a lag-resolved time-irreversibility statistic, generalizing the
    single-lag ``trev``/``tc3``-style statistics to a full profile, cf. the
    leverage-effect correlation function of [1]_.

    This function characterizes the *shape* of :math:`\\text{diff}(\\tau)` across
    lags -- its decay, persistence, and extrema -- mirroring how
    :func:`autocorr_shape` characterizes the shape of the ordinary ACF. (An
    earlier version of this function instead characterized the forward and
    backward profiles' shapes separately, but on 300 real time series from
    ``INP_Empirical1000.mat`` their shape descriptors were correlated at
    r = 0.84-0.97 with each other -- i.e., overwhelmingly redundant, since both
    profiles inherit most of their shape from whatever ordinary linear
    correlation the series has. The difference profile cancels that shared
    component and isolates the genuinely asymmetric/nonlinear structure.)

    References
    ----------
    .. [1] J.-P. Bouchaud, A. Matacz and M. Potters, "Leverage effect in financial
        markets: The retarded volatility model", Phys. Rev. Lett. 87, 228701 (2001)

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored: zero mean, unit variance).

    max_lag : int or str, optional
        The maximum lag to compute the profile up to.

        - If an ``int``, a positive maximum lag.
        - If ``"double_drown"``, uses twice the first zero-crossing of the
          ordinary (linear) autocorrelation function (cf. the ``'double_drown'``
          option of :func:`autocorr_shape`), bounded to lie in
          ``[10, floor(N/4)]``.

        Default is ``'double_drown'``.

    Returns
    -------
    dict
        Statistics on the shape of the forward-minus-backward difference profile,
        including its lag-1 value, basic summaries, centroid decay timescale,
        self-autocorrelation, local extrema, first sign change, the correlation
        between the forward and backward profiles, and the maximum lag used.
        All fields are NaN if the profile is too short or ill-defined.
    """
    y = np.asarray(y)
    N = len(y)

    fields = ['diff1', 'sumdiff', 'meandiff', 'meanabsdiff', 'rmsdiff', 'centroiddiff',
              'ac1diff', 'nminima', 'nmaxima', 'pextrema', 'firstsignchangediff',
              'corrfwdbwd', 'maxLag']

    if isinstance(max_lag, str):
        if max_lag == 'double_drown':
            tau0 = first_crossing(y, 'ac', 0, 'discrete')
            if np.isnan(tau0) or tau0 == 0:
                tau0 = 10  # fallback for pathological/near-constant series
            max_lag = min(int(np.floor(N / 4)), max(10, 2 * int(tau0)))
        else:
            raise ValueError(f"Unknown max_lag setting: '{max_lag}'")

    if max_lag < 5:
        # Too short a series/profile to say anything meaningful
        return {f: np.nan for f in fields}

    # Compute the forward and backward lag-profiles, and their difference:
    taus = np.arange(1, max_lag + 1)
    g_fwd = autocorr_x2(y, taus, 'forward')
    g_bwd = autocorr_x2(y, taus, 'backward')

    if np.any(np.isnan(g_fwd)) or np.any(np.isnan(g_bwd)):
        return {f: np.nan for f in fields}

    diff_profile = g_fwd - g_bwd

    out = {}

    # Lag-1 difference: the single-lag time-irreversibility statistic, comparable
    # to trev. (Note: the raw lag-1 and lag-0 values themselves -- g_fwd[0],
    # g_bwd[0], and mean(y**3) -- are not included as outputs here since they
    # duplicate existing operations exactly: ac_nl_0_1, ac_nl_1_1, and the third
    # moment, respectively.)
    out['diff1'] = diff_profile[0]

    # Basic stats on the difference profile
    out['sumdiff'] = np.sum(diff_profile)
    out['meandiff'] = np.mean(diff_profile)
    out['meanabsdiff'] = np.mean(np.abs(diff_profile))
    out['rmsdiff'] = np.sqrt(np.mean(diff_profile**2))

    # Characteristic (centroid) decay timescale of the difference profile -- how
    # many lags the time-irreversibility signature persists over. Centroid-based
    # (rather than an exponential fit) to remain well-defined for non-monotonic
    # profiles.
    sum_abs = np.sum(np.abs(diff_profile))
    if sum_abs > 0:
        out['centroiddiff'] = np.sum(taus * np.abs(diff_profile)) / sum_abs
    else:
        out['centroiddiff'] = np.nan

    # Autocorrelation of the difference profile (smoothness/persistence of the
    # irreversibility signature itself), cf. the ac1 field of autocorr_shape
    out['ac1diff'] = autocorr(diff_profile, 1, 'Fourier')[0]

    # Local extrema of the difference profile, cf. autocorr_shape
    ddiff = np.diff(diff_profile)
    dddiff = np.diff(ddiff)
    extrr = sign_change(ddiff, 1)
    sdsp = dddiff[extrr]
    out['nminima'] = np.sum(sdsp > 0)
    out['nmaxima'] = np.sum(sdsp < 0)
    out['pextrema'] = len(sdsp) / max_lag

    # How quickly the time-irreversibility signature changes sign
    sign_change_idx = np.flatnonzero(np.sign(diff_profile[:-1]) != np.sign(diff_profile[1:]))
    if sign_change_idx.size == 0:
        out['firstsignchangediff'] = max_lag  # no sign change within the window measured
    else:
        # convert from 0-based index space to 1-based lag space
        out['firstsignchangediff'] = int(sign_change_idx[0]) + 1

    # Shape similarity of the two profiles (a different angle from their
    # difference: do they have a similar shape regardless of overall magnitude?)
    out['corrfwdbwd'] = np.corrcoef(g_fwd, g_bwd)[0, 1]

    out['maxLag'] = max_lag  # record how far the profile was measured

    return out

def partial_autocorr(y: ArrayLike, max_tau: int = 10, what_method: str = 'ols') -> dict:
    """
    Compute the partial autocorrelation of an input time series.
    
    This function calculates the partial autocorrelation function (PACF) up to a specified 
    lag using either ordinary least squares or Yule-Walker equations.

    Parameters
    ----------
    y : array-like
        The input time series.
    max_tau : int, optional
        Maximum time-delay to compute PACF values for. Default is 10.
    method : {'ols', 'yule-walker'}, optional
        Method to compute partial autocorrelation:

        - ``'ols'``: Ordinary least squares regression.
        - ``'yule-walker'``: Yule-Walker equations method.

        Default is ``'ols'``.

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

    method_map = {'ols': 'ols-inefficient', 'yule-walker': 'ywm'} 
    if what_method not in method_map:
        raise ValueError(f"Invalid method: {what_method}. Use 'ols' or 'yule-walker'.")

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
        The z-scored input time series.
    tau : (int, optional)
        The time delay. If None, it's set to the first minimum of the autocorrelation function.
        Default is ``None``.

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
        Input time series.

    tau : int
        Time delay used to construct the embedding
        :math:`(y_i, y_{i-\\tau})`.
        Default is 1.

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
        The input time-series (z-scored).
    tau : int or str, optional
        The time-delay. If 'tau', it's set to the first zero crossing of the 
        autocorrelation function. Default is ``'tau'``.
    shape : str, optional
        The shape to use. Currently only 'circle' is supported. Default is ``circle``.
    r : float, optional
        The radius of the circle. Default is 1.0.

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
    if shape == 'circle':
        # Puts a circle around each point in the embedding space in turn
        # counts how many pts are inside this shape, looks at the time series thus formed.
        # Vectorised: one pairwise squared-distance matrix (sqeuclidean == the loop's
        # sum of squared diffs exactly), then count <= r**2 per row. Diagonal is 0, so
        # the self-count subtraction below is preserved. O(N^2) memory.
        from scipy.spatial.distance import cdist
        m_c_d = cdist(m, m, metric='sqeuclidean')
        counts = np.sum(m_c_d <= r**2, axis=1).astype(float)
    else:
        raise ValueError(f"Unknown shape '{shape}'")
    counts -= 1 # ignore self counts

    if np.all(counts == 0):
        logger.warning("embed2_shapes: no counts detected!")
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
        The input time series.
    alpha : float 
        The parameter alpha for GLSCF calculation. Must be non-zero.
    beta : float
        The parameter beta for GLSCF calculation. Must be non-zero.
    max_tau : int, optional
        Maximum time delay to search up to. If None, uses the time-series length.
        Default is ``None``.

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

        Default is ``'tau'``.

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
        Default is 1.

    method : {"Fourier", "TimeDomainStat", "TimeDomain"}, optional
        Method used to compute the autocorrelation.

        - ``"Fourier"``: Computes autocorrelation via the Wiener–Khinchin
            theorem using the Fourier transform.
        - ``"TimeDomainStat"``: Statistical time-domain estimator.
        - ``"TimeDomain"``: Direct time-domain computation.

        Default is ``'Fourier'``.

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
                logger.info(f'NaNs in time series, computing for {np.sum(good_r)}/{len(good_r)} pairs of points.')
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
        The input time series.
    corr_fun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function. Default is ``'ac'``.
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]. Default is 0.
    what_out : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'. Default is ``'both'``.

    Returns
    --------
    dict or float
        The first crossing information, format depends on what_out.
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
        Method for moving the shape. Currently, only ``'pts'`` is supported, which places 
        the shape on each point in the time series. Default is ``'pts'``.

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

            w = int(d)
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
    
    vals, hits = np.unique(np_counts, return_counts=True)
    max_val = np.argmax(hits)
    out["npatmode"] = hits[max_val]/NN
    out["mode"] = vals[max_val]

    count_types = ["ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "elevens"]
    for i in range(1, 12):
        if 2*w + 1 >= i:
            out[f"{count_types[i-1]}"] = np.mean(np_counts == i)
    
    # imported here rather than at module scope: stationarity imports from this
    # module, so a top-level import would close the cycle
    from ..operations.stationarity import sliding_window

    for num_seg in (2, 3, 4):
        out[f'statav{num_seg}_m'] = sliding_window(np_counts, 'mean', 'std', num_seg, 1)
        out[f'statav{num_seg}_s'] = sliding_window(np_counts, 'std', 'std', num_seg, 1)

    return out

def autocorr_shape(y: ArrayLike, stop_when: Union[int, str] = 'pos_drown') -> dict:
    """
    How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters
    -----------
    y : array-like
        The input time series.
    stop_when : str or int, optional
        The criterion for the maximum lag to measure the ACF up to.
        Default is ``'pos_drown'``.

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
        
    elif stop_when in ['pos_drown', 'drown', 'double_drown']:
        # Compute ACF up to a given threshold:
        n_drown = 0 # the point at which ACF ~ 0
        # The Fourier ACF depends only on N, so compute the whole (lag-indexed) curve
        # once and read acf_full[i-1] instead of recomputing a full FFT every lag.
        # acf_full[i-1] is bit-identical to autocorr(y, i-1, 'Fourier')[0].
        acf_full = autocorr(y, [], 'Fourier')
        if stop_when == 'pos_drown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = acf_full[i-1]
                if np.isnan(acf_val):
                    logger.warning("Weird time series (constant?)")
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
                acf_val = acf_full[i-1] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 1 and abs(acf_val) < th:
                    n_drown = i - 1 # convert from index to the corresponding lag
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
            if n_drown == 0:
                # ACF never entered the significance band across available lags
                n_drown = N - 1
        elif stop_when == 'double_drown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = acf_full[i-1]
                if n_drown > 0 and i == 2 * n_drown + 1:
                    acf.append(acf_val)
                    break
                elif i > 1 and abs(acf_val) < th:
                    n_drown = i - 1 # convert from index to the corresponding lag
                acf.append(acf_val)
            if n_drown == 0:
                # ACF never entered the significance band across available lags
                n_drown = N - 1
    else:
        raise ValueError(f"Unknown ACF decay criterion: '{stop_when}'")

    acf = np.array(acf)
    nac = len(acf)

    # Check for good behavior
    if np.any(np.isnan(acf)):
        # This is an anomalous time series (e.g., all constant, or containing NaNs)
        out = np.nan
    
    out = {}
    out['Nac'] = n_drown

    # Basic stats on the ACF
    out['sumacf'] = np.sum(acf)
    out['meanacf'] = np.mean(acf)
    if stop_when != 'pos_drown':
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

    if stop_when == 'pos_drown' and nac >= min_pts_to_fit_exp:
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
            - 'ac': Use the first zero-crossing of the autocorrelation function.
            - 'mi': Use the first minimum of the automutual information function.

        Default is ``'ac'``.

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
        logger.warning("No valid setting for time delay. (Is the time series too short?)")
        return np.nan

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
            - 'ac': Use the first zero-crossing of the autocorrelation function.
            - 'mi': Use the first minimum of the automutual information function.

            Default is 'ac'.

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
        logger.warning("No valid setting for time delay (time series too short?)")
        return np.nan
    
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
