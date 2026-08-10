import logging
logger = logging.getLogger('pyhctsa')
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ..utils import sign_change
from ..toolboxes.infotheory.mutual_info import KraskovMI, GaussianMI

def _get_corr_fn(y: np.ndarray, min_what: str, extra_param: Union[int, float, None]) -> Callable:
    """Helper to return the correct correlation function based on method type."""
    from ..operations.correlation import autocorr, automutual_info

    if min_what in ['ac', 'corr']:
        return lambda x: autocorr(y, tau=x, method='Fourier').item()
    elif min_what == 'mi-hist':
        num_bins = int(extra_param) if extra_param else 10
        return lambda x: _mi_bin(y[:-x], y[x:], 'range', 'range', num_bins)
    elif min_what == 'mi-kraskov2':
        return lambda x: automutual_info(y, x, 'kraskov2', extra_param)
    elif min_what == 'mi-kraskov1':
        return lambda x: automutual_info(y, x, 'kraskov1', extra_param)
    elif min_what in ['mi', 'mi-gaussian']:
        return lambda x: automutual_info(y, x, 'gaussian', extra_param)
    else:
        raise ValueError(f"Unknown correlation type specified: {min_what}")

def _ami_gaussian_curve(y: np.ndarray):
    """Gaussian automutual information at every lag 1..n-1 in one O(n log n) pass.

    Reproduces the windowed Pearson estimate (== ``GaussianMI`` on the 1-D delay
    pair); a degenerate (constant) delay window gives NaN at that lag. Assumes
    ``n >= 3`` (guarded by ``_self_corr_curve``).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    yc = y - y.mean()                      
    # linear autocorrelation  C[tau] = sum_t yc[t]*yc[t+tau]  via zero-padded FFT
    nfft = 1 << (2 * n - 1).bit_length()
    fy = np.fft.rfft(yc, nfft)
    C_all = np.fft.irfft(fy * np.conj(fy), nfft)[:n]
    # per-window sums for the two delayed segments y[:-tau] and y[tau:]
    P = np.concatenate(([0.0], np.cumsum(yc)))         # P[k] = sum_{t<k} yc[t]
    Q = np.concatenate(([0.0], np.cumsum(yc * yc)))    # Q[k] = sum_{t<k} yc[t]^2
    taus = np.arange(1, n)
    m = (n - taus).astype(float)
    S1 = P[n - taus]; S2 = P[n] - P[taus]
    Q1 = Q[n - taus]; Q2 = Q[n] - Q[taus]
    C = C_all[taus]
    num = m * C - S1 * S2
    den = (m * Q1 - S1 * S1) * (m * Q2 - S2 * S2)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.clip(num / np.sqrt(den), -1.0, 1.0)
        auto_corr = -0.5 * np.log(1.0 - r * r)         # AMI(tau), Gaussian estimator
    auto_corr[~(den > 0.0)] = np.nan                   # degenerate window -> NaN
    return auto_corr



_VECTORISED_CORR = ('ac', 'mi', 'mi-gaussian')

def _self_corr_curve(y: np.ndarray, what: str):
    """Full self-correlation curve at lags 1..n-1 for a vectorisable estimator.

    ``'ac'`` -> the FFT autocorrelation, computed once (cf. ``first_crossing``);
    ``'mi'`` / ``'mi-gaussian'`` -> the Gaussian AMI curve. Returns ``None`` when
    the series is too short to have an interior extremum (``n < 3``).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return None
    if what == 'ac':
        from ..operations.correlation import autocorr
        c = np.asarray(autocorr(y, [], 'Fourier'), dtype=float).ravel()
        return c[1:n]                                  # drop lag 0 -> lags 1..n-1
    return _ami_gaussian_curve(y)                       # 'mi' / 'mi-gaussian'


def _first_min_from_curve(c: np.ndarray):
    """First strict local minimum of a precomputed lag-curve (lags 1..len(c))."""
    for i in range(1, len(c) + 1):
        if np.isnan(c[i - 1]):
            logger.warning("No minimum: encountered NaN.")
            return np.nan
        if (i == 2) and (c[1] > c[0]):
            return 1
        elif (i > 2) and c[i - 3] > c[i - 2] < c[i - 1]:
            return i - 1
    return np.nan


def _first_max_from_curve(c: np.ndarray):
    """First strict local maximum of a precomputed lag-curve (lags 1..len(c))."""
    for i in range(1, len(c) + 1):
        if np.isnan(c[i - 1]):
            logger.warning("No maximum: encountered NaN.")
            return np.nan
        if (i > 2) and c[i - 3] < c[i - 2] > c[i - 1]:
            return i - 1
    return np.nan

def first_min(
    y: list,
    min_what: str = 'mi-gaussian',
    extra_param: Union[int, float, None] = None,
) -> int:
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.

    min_what : str, optional
        Correlation measure to minimize.

        Autocorrelation:

        - ``'ac'``: Autocorrelation.

        Automutual information (AMI):

        - ``'mi'``: AMI using the Gaussian estimator (default for AMI).
        - ``'mi-kraskov1'``: AMI using the Kraskov estimator (variant 1).
        - ``'mi-kraskov2'``: AMI using the Kraskov estimator (variant 2).
        - ``'mi-hist'``: AMI using a histogram-based estimator.

        Default is ``'mi-gaussian'``.

    extra_param : any, optional
        Additional parameter required by the chosen ``min_what`` method
        (e.g., a k-nearest-neighbours parameter for Kraskov-based AMI).

    Returns
    -------
    int
        The time of the first minimum.
    """
    y = np.asarray(y)
    n = len(y)
    if min_what in _VECTORISED_CORR:               # vectorised drop-in, identical lag
        c = _self_corr_curve(y, min_what)
        return np.nan if c is None else _first_min_from_curve(c)
    corrfn = _get_corr_fn(y, min_what, extra_param)
    
    auto_corr = np.zeros(n - 1)
    for i in range(1, n):
        auto_corr[i - 1] = corrfn(i)
        
        if np.isnan(auto_corr[i - 1]):
            logger.warning(f"No minimum in {min_what}: encountered NaN.")
            return np.nan
        
        # Check for minimum
        if (i == 2) and (auto_corr[1] > auto_corr[0]):
            return 1
        elif (i > 2) and auto_corr[i - 3] > auto_corr[i - 2] < auto_corr[i - 1]:
            return i - 1
            
    return np.nan

def first_max(
    y: list,
    max_what: str = 'mi-gaussian',
    extra_param: Union[int, float, None] = None,
) -> int:
    """
    Time of first maximum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    max_what : str, optional
        Correlation measure to maximize.

        Autocorrelation:

        - ``'ac'``: Autocorrelation.

        Automutual information (AMI):

        - ``'mi'``: AMI using the Gaussian estimator (default for AMI).
        - ``'mi-kraskov1'``: AMI using the Kraskov estimator (variant 1).
        - ``'mi-kraskov2'``: AMI using the Kraskov estimator (variant 2).
        - ``'mi-hist'``: AMI using a histogram-based estimator.

        Default is ``'mi'``.

    extra_param : any, optional
        An additional parameter required for the specified `max_what` method (e.g., for Kraskov).

    Returns
    -------
    int
        The time of the first maximum.
    """
    y = np.asarray(y)
    n = len(y)
    if max_what in _VECTORISED_CORR:               # vectorised drop-in, identical lag
        c = _self_corr_curve(y, max_what)
        return np.nan if c is None else _first_max_from_curve(c)
    corrfn = _get_corr_fn(y, max_what, extra_param)
    
    auto_corr = np.zeros(n - 1)
    for i in range(1, n):
        auto_corr[i - 1] = corrfn(i)
        
        if np.isnan(auto_corr[i - 1]):
            logger.warning(f"No maximum in {max_what}: encountered NaN.")
            return np.nan

        # Check for maximum
        if i > 2 and auto_corr[i - 3] < auto_corr[i - 2] > auto_corr[i - 1]:
            return i - 1
            
    return np.nan

def _mi_bin(v1: ArrayLike, v2: ArrayLike, r1: Union[str, list] = 'range',
            r2: Union[str, list] = 'range', num_bins: int = 10) -> float:
    """
    Compute mutual information between two data vectors using bin counting.

    Parameters:
    -----------
        v1 (array-like): The first input vector
        v2 (array-like): The second input vector
        r1 (str or list): The bin-partitioning method for v1 ('range', 'quantile', or [min, max])
        r2 (str or list): The bin-partitioning method for v2 ('range', 'quantile', or [min, max])
        num_bins (int): The number of bins to partition each vector into (default : 10)

    Returns:
    --------
        float: The mutual information computed between v1 and v2
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()

    if len(v1) != len(v2):
        raise ValueError("Input vectors must be the same length")

    N = len(v1)

    # Create histograms
    edges_i = _give_me_edges(r1, v1, num_bins)
    edges_j = _give_me_edges(r2, v2, num_bins)

    ni, _ = np.histogram(v1, edges_i)
    nj, _ = np.histogram(v2, edges_j)

    # Create a joint histogram
    hist_xy, _, _ = np.histogram2d(v1, v2, [edges_i, edges_j])

    # Normalize counts to probabilities
    p_i = ni[:num_bins] / N
    p_j = nj[:num_bins] / N
    p_ij = hist_xy / N
    p_ixp_j = np.outer(p_i, p_j)

    # Calculate mutual information
    mask = (p_ixp_j > 0) & (p_ij > 0)
    if np.any(mask):
        mi = np.sum(p_ij[mask] * np.log(p_ij[mask] / p_ixp_j[mask]))
    else:
        logger.warning("The histograms aren't catching any points. Perhaps due to an inappropriate custom range for binning the data.")
        mi = np.nan

    return mi

def _give_me_edges(r, v, n_bins):
    EE = 1E-6 # this small addition gets lost in the last bin
    if n_bins <= 0:
        raise ValueError(f"nbins must be > 0, got {n_bins}")
    if r == 'range':
        return np.linspace(np.min(v), np.max(v) + EE, n_bins + 1)
    elif r == 'quantile': # bin edges based on quantiles
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[-1] += EE
        return edges
    elif isinstance(r, (list, np.ndarray)) and len(r) == 2: # a two-component vector
        return np.linspace(r[0], r[1] + EE, n_bins + 1)
    else:
        raise ValueError(f"Unknown partitioning method '{r}'")

def automutual_info_stats(
    y: ArrayLike,
    max_tau: Optional[int] = None,
    est_method: str = 'gaussian',
    extra_param: Optional[Union[int, str]] = None
) -> Dict[str, float]:
    """
    Calculate statistics on the automutual information (AMI) function of a time series.

    This function computes various statistics on how the automutual information changes
    with increasing time delay, including basic statistics, periodicities, and crossings.

    Parameters
    ----------
    y : array-like
        Input time series.
    max_tau : int, optional
        Maximum time delay to investigate. If None, uses N/4 where N is the length
        of the time series, but won't exceed N/2. Default is `None`.
    est_method : {'gaussian', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information (passed to automutual_info).
        Default is ``'gaussian'``.
    extra_param : int or str, optional
        Extra parameter for the estimator (passed to automutual_info).
        For Kraskov estimators, sets the number of nearest neighbors 'k'. Default is `None`.

    Returns
    -------
    dict
        Dictionary containing AMI statistics.
    """
    from ..operations.correlation import autocorr

    y = np.asarray(y)
    n = len(y)  # length of the time series

    # max_tau: the maximum time delay to investigate
    if max_tau is None:
        max_tau = np.ceil(n / 4)
    max_tau_0 = max_tau

    # Don't go above N/2
    max_tau = min(max_tau, np.ceil(n / 2))

    # Get the AMI data
    max_tau = int(max_tau)
    max_tau_0 = int(max_tau_0)
    time_delay = list(range(1, max_tau + 1))
    ami = automutual_info(
        y,
        time_delay=time_delay,
        est_method=est_method,
        extra_param=extra_param
    )
    ami = np.array(list(ami.values()))

    out = {}  # create dict for storing results

    # Output the raw values
    for i in range(1, max_tau_0 + 1):
        if i <= max_tau:
            out[f'ami{i}'] = ami[i - 1]
        else:
            out[f'ami{i}'] = np.nan

    # Basic statistics
    lami = len(ami)
    out['mami'] = np.mean(ami)
    out['stdami'] = np.std(ami, ddof=1)

    # First minimum of mutual information across range
    dami = np.diff(ami)
    extrema_i = np.where((dami[:-1] * dami[1:]) < 0)[0]
    out['pextrema'] = len(extrema_i) / (lami - 1)
    out['fmmi'] = min(extrema_i) + 1 if len(extrema_i) > 0 else lami

    # Look for periodicities in local maxima
    maxima_i = np.where((dami[:-1] > 0) & (dami[1:] < 0))[0] + 1
    dmaxima_i = np.diff(maxima_i)
    # Is there a big peak in dmaxima? (no need to normalize since a given method 
    # inputs its range; but do it anyway... ;-))
    out['pmaxima'] = len(dmaxima_i) / (lami // 2)
    if len(dmaxima_i) == 0:  # fewer than 2 local maxima
        out['modeperiodmax'] = np.nan
        out['pmodeperiodmax'] = np.nan
    else:
        out['modeperiodmax'] = stats.mode(dmaxima_i, keepdims=True).mode[0]
        out['pmodeperiodmax'] = np.sum(dmaxima_i == out['modeperiodmax']) / len(dmaxima_i)

    # Look for periodicities in local minima
    minima_i = np.where((dami[:-1] < 0) & (dami[1:] > 0))[0] + 1
    dminima_i = np.diff(minima_i)

    out['pminima'] = len(dminima_i) / (lami // 2)

    if len(dminima_i) == 0:  # fewer than 2 local minima
        out['modeperiodmin'] = np.nan
        out['pmodeperiodmin'] = np.nan
    else:
        out['modeperiodmin'] = stats.mode(dminima_i, keepdims=True).mode[0]
        out['pmodeperiodmin'] = np.sum(dminima_i == out['modeperiodmin']) / len(dminima_i)

    # Number of crossings at mean/median level, percentiles
    out['pcrossmean'] = np.mean(np.diff(np.sign(ami - np.mean(ami))) != 0)
    out['pcrossmedian'] = np.mean(np.diff(np.sign(ami - np.median(ami))) != 0)
    out['pcrossq10'] = np.mean(sign_change(ami - np.percentile(ami, 10, method='hazen')))
    out['pcrossq90'] = np.mean(sign_change(ami - np.percentile(ami, 90, method='hazen')))

    # ac1
    out['amiac1'] = autocorr(ami, 1, 'Fourier')[0]

    return out
    
def automutual_info(
        y: ArrayLike,
        time_delay: Union[int, str, List[int]] = 1,
        est_method: str = 'gaussian',
        extra_param: Optional[Union[int, str]] = None) -> Any:
    """
    Compute time-delayed automutual information of a time series.

    Calculates the mutual information between a time series and its time-delayed version
    using various estimation methods.

    References
    ----------
    .. [1] Kraskov, A., Stoegbauer, H., Grassberger, P. (2004).
        Estimating mutual information. Physical Review E, 69(6), 066138.

    Parameters
    ----------
    y : array-like
        Input time series.
    time_delay : int, str, or list of int, optional
        Time lag(s) for automutual information calculation. Can be:

        - int: a fixed lag
        - list of int: multiple lags
        - 'ac': first zero-crossing of autocorrelation
        - 'tau': same as 'ac'
        
        Default is 1.

    est_method : {'gaussian', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information:

        - 'gaussian': Assumes Gaussian variables
        - 'kraskov1': Kraskov estimator 1 (KSG1)
        - 'kraskov2': Kraskov estimator 2 (KSG2)

        Default is `kernel`.

    extra_param : int or str, optional
        Extra parameter for the estimator. For Kraskov estimators,
        this sets the number of nearest neighbors 'k'.
        Default is 4.

    Returns
    -------
    float or dict
        If single time_delay:
            float: The automutual information value
        If multiple time_delay:
            dict: Keys are f"ami{delay}", values are corresponding AMI values
    """
    from ..operations.distribution import first_crossing # zzzz

    if isinstance(time_delay, str) and time_delay in ['ac', 'tau']:
        time_delay = first_crossing(y, corr_fun='ac', threshold=0, what_out='discrete')

    y = np.asarray(y).flatten()
    n = len(y)
    min_samples = 5  # minimum 5 samples to compute mutual information (could make higher?)
    kval = 4 # default 
    if extra_param is not None:
        kval = extra_param

    # Loop over time delays if a vector
    if not isinstance(time_delay, list):
        time_delay = [time_delay]

    num_time_delays = len(time_delay)
    amis = np.full(num_time_delays, np.nan)

    if num_time_delays > 1:
        time_delay = np.sort(time_delay)
    
    if est_method == 'kraskov1':
        mi_calc = KraskovMI(k=kval, algorithm=1, add_noise=False) # no added noise
    elif est_method == 'kraskov2':
        mi_calc = KraskovMI(k=kval, algorithm=2, add_noise=False)
    elif est_method == 'gaussian':
        mi_calc = GaussianMI()
    else:
        raise ValueError(f'Unknown estimator: {est_method}')
    
    for k, delay in enumerate(time_delay):
        if delay > n - min_samples:
            # time series too short - keep the remaining values as NaNs
            break

        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        amis[k] = mi_calc.compute(y1, y2)
        
    if np.isnan(amis).any():
        logger.warning(
            f"Time series (n={n}) is too short for automutual information calculations "
            f"up to lags of {max(time_delay)}"
        )
    
    if num_time_delays == 1:
        # return a scalar if only one time delay
        return amis[0]
    
    else:
        # return a dict for multiple time delays
        return {f"ami{delay}": ami for delay, ami in zip(time_delay, amis)}

def rm_automutual_information(y: ArrayLike, tau: int = 1) -> float:
    """
    Estimates the mutual information of two stationary signals with 
    independent pairs of samples using various approaches.

    Based on a wrapper initially developed by Ben D. Fulcher in MATLAB,
    which is based on rm_information.py initially developed by Rudy Moddemeijer in MATLAB,
    and translated to to python by Tucker Cullen.

    Parameters
    ----------
    y : array-like
        The input time series.
    tau: int
        Time lag for automutual information calculation. Default is 1.

    Returns
    -------
    float:
        Estimate of the auto-mutual information
    """
    if tau >= len(y):
        return np.nan
    elif tau == 0:
        # handle the case when tau = 0 (no lag)
        y1 = y2 = y
    else:
        y1 = y[:-tau]
        y2 = y[tau:]

    out = _rm_info(y1, y2)[0]

    return out

def _rm_info(x: ArrayLike, y: ArrayLike):
    """Unbiased mutual-information estimate of two equal-length signals via a 2D histogram.

    Moddemeijer's histogram estimator (natural-log base, unbiased correction).
    Original MATLAB by R. Moddemeijer; Python translation by Tucker Cullen.

    Parameters
    ----------
    x, y : array-like
        Equal-length 1-D input vectors.

    Returns
    -------
    estimate : float
        The (unbiased) mutual information estimate in nats.
    nbias : float
        The N-bias of the estimate (0 after the unbiased correction).
    sigma : float
        The standard error of the estimate.
    descriptor : np.ndarray
        The histogram descriptor used (see :func:`_rm_histogram_2`).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D vectors")
    if x.size != y.size:
        raise ValueError("Unequal length of x and y")

    h, descriptor = _rm_histogram_2(x, y)
    n_cell_x = int(descriptor[0, 2])
    n_cell_y = int(descriptor[1, 2])

    # marginal (row/column) sums
    h_x = h.sum(axis=1)
    h_y = h.sum(axis=0)

    # log_f = log(h / h_x / h_y) where h != 0, else 0 (vectorised over all cells)
    hf = h.astype(float)
    nz = hf != 0
    log_f = np.zeros_like(hf)
    denom = np.outer(h_x, h_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_f[nz] = np.log(hf[nz] / denom[nz])

    count = hf.sum()
    estimate = np.sum(hf * log_f)
    sigma = np.sum(hf * log_f ** 2)

    # biased estimate, then unbiased correction
    estimate = estimate / count
    sigma = np.sqrt((sigma / count - estimate ** 2) / (count - 1))
    estimate = estimate + np.log(count)
    nbias = (n_cell_x - 1) * (n_cell_y - 1) / (2 * count)
    estimate = estimate - nbias
    nbias = 0

    return estimate, nbias, sigma, descriptor


def _rm_histogram_2(x: ArrayLike, y: ArrayLike):
    """Two-dimensional frequency histogram of two equal-length row vectors.

    Bin bounds and counts are chosen automatically following Moddemeijer's rule
    (``ncell = ceil(n ** (1/3))`` per dimension, bounds padded by half a bin).

    Parameters
    ----------
    x, y : array-like
        Equal-length 1-D input vectors.

    Returns
    -------
    result : np.ndarray
        2D frequency histogram of shape ``(ncell_x, ncell_y)``.
    descriptor : np.ndarray
        Histogram descriptor ``[[lower_x, upper_x, ncell_x], [lower_y, upper_y, ncell_y]]``.

    Notes
    -----
    Original MATLAB by R. Moddemeijer; Python translation by Tucker Cullen.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D vectors")
    if x.size != y.size:
        raise ValueError("Unequal length of x and y")
    n = x.size

    minx, maxx = np.amin(x), np.amax(x)
    miny, maxy = np.amin(y), np.amax(y)
    deltax = (maxx - minx) / (n - 1)
    deltay = (maxy - miny) / (n - 1)
    ncell = np.ceil(n ** (1 / 3))
    descriptor = np.array([
        [minx - deltax / 2, maxx + deltax / 2, ncell],
        [miny - deltay / 2, maxy + deltay / 2, ncell],
    ])

    lowerx, upperx, ncellx = descriptor[0]
    lowery, uppery, ncelly = descriptor[1]
    ncellx, ncelly = int(ncellx), int(ncelly)

    # cell indices (1-based rounding as in the original), then vectorised scatter-add
    xx = np.around((x - lowerx) / (upperx - lowerx) * ncellx + 0.5).astype(int) - 1
    yy = np.around((y - lowery) / (uppery - lowery) * ncelly + 0.5).astype(int) - 1

    result = np.zeros((ncellx, ncelly), dtype=int)
    in_bounds = (xx >= 0) & (xx < ncellx) & (yy >= 0) & (yy < ncelly)
    np.add.at(result, (xx[in_bounds], yy[in_bounds]), 1)

    return result, descriptor

