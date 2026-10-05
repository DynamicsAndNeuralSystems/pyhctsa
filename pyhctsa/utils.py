import csv
import hashlib
import os
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from importlib import resources
from typing import Union, Callable
import yaml
from pyhctsa import __version__
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.stats import chi2


def _check_optional_deps(dep: str) -> bool:
    """Check whether an optional dependency exists.
    Returns True if available, else False."""
    try:
        version(dep)
        return True
    except PackageNotFoundError:
        return False

def _validate_data(ts: np.ndarray) -> bool:
    """validate a time series before computing features"""
    if len(ts) < 100:
        logger.warning("Time series is too short!")
        return False
    if np.all(ts == ts[0]):
        # constant time series
        # maybe do a tolerance instead?
        logger.warning("Time series is constant.")
        return False
    if np.any(np.isnan(ts)):
        # data contains nans
        logger.warning("Time series contains NaNs.")
        return False
    if np.any(np.isinf(ts)):
        logger.warning("Time series contains Inf.")
        return False

    return True

def _ml_rng(seed: int) -> np.random.RandomState:
    """
    ``rng(seed, 'twister')``, as a numpy ``RandomState``.
    """
    return np.random.RandomState(5489 if seed == 0 else seed)

def _ml_randperm(n: int, rng: np.random.RandomState) -> np.ndarray:
    """
    MATLAB's ``randperm(n)``: the 1-based ordering that sorts ``rand(1, n)``.
    """
    return np.argsort(rng.random_sample(n), kind='stable') + 1

def _linspace(d1: float, d2: float, n: int) -> np.ndarray:
    """
    MATLAB's ``linspace(d1, d2, n)``.
    """
    d1 = float(d1)
    d2 = float(d2)
    n1 = n - 1
    if np.isinf((d2 - d1) * (n1 - 1)):
        i = np.arange(n1 + 1, dtype=float)
        y = d1 + (d2 / n1) * i - (d1 / n1) * i
    else:
        y = d1 + np.arange(n1 + 1, dtype=float) * (d2 - d1) / n1
    if y.size:
        if d1 == d2:
            y[:] = d1
        else:
            y[n - 1] = d2
    return y

def _load_csv(path: str) -> list:
    """Helper function to load CSV formatted datasets."""
    dataset = [] # list of np.ndarray
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                time_series = np.array([float(value) for value in row if value != ''])
                dataset.append(time_series)
            except ValueError:
                continue
    return dataset

def get_dataset(which: str = "e1000") -> list:
    """
    Load predefined datasets for testing and validation.

    Parameters
    ----------
    which : str, default="e1000"
        Dataset identifier.

        Options are:

        - ``"e1000"``: Empirical 1000 dataset.
        - ``"sinusoid"``: Sinusoidal test data.
        - ``"noise"``: Gaussian noise data (T = 1000 sample length time series).

    Returns
    -------
    list
        List of time series data, where each element is a time series instance as a numpy array.
    """
    dataset = []
    utils_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = {
        "e1000": {
            "path": "./data/e1000.csv",
            "loader": lambda p: _load_csv(p),
            "desc": "empirical1000"
        },
        "sinusoid": {
            "path": "./data/sinusoid.txt",
            "loader": lambda p: [np.loadtxt(p)],
            "desc": "sinusoid"
        },
        "noise": {
            "path": "./data/noise_gaussian.txt",
            "loader": lambda p: [np.loadtxt(p)],
            "desc": "gaussian noise"
        }
    }

    if which not in datasets:
        raise NotImplementedError(f"Dataset '{which}' not found. Available options: {list(datasets.keys())}")

    logger.info(f"Loading {datasets[which]['desc']} dataset...")
    data_path = os.path.normpath(os.path.join(utils_dir, datasets[which]['path']))
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    dataset = datasets[which]['loader'](data_path)
    logger.info(f"Loaded dataset of {len(dataset)} time series.")
    return dataset
    
# config `preprocess:` values and the label suffix each adds
_PREPROCESS_LABELS = {'decimate_ac1e': '_dec'}

def _preprocess_decorator(zscore: bool = False, absval: bool = False,
                          preprocess: Union[str, None] = None) -> Callable:
    """
    Decorator to preprocess time series data before feature computation.
    
    Applies optional z-score normalization, an optional hctsa ``BF_PreProcess`` step
    and/or absolute value transformation to the input time series before passing it to
    the decorated function.

    The order is: z-score, then ``preprocess``, then absolute value. A
    ``preprocess`` step is followed by a second z-score if ``zscore`` is True, matching
    hctsa's ``zscore(BF_PreProcess(x_z, ...))``.

    Parameters
    ----------
    zscore : bool, optional
        If True, z-score normalize the input time series to have mean 0 and 
        standard deviation 1. Default is False.
    absval : bool, optional
        If True, take the absolute value of all data points in the input time series.
        Default is False.
    preprocess : {None, 'decimate_ac1e'}, optional
        A named pre-processing step applied to the (z-scored) series. Currently
        ``'decimate_ac1e'``: keep one sample per floored 1/e autocorrelation time
        (see :func:`decimate_ac1e`). If the delay cannot be determined the wrapped
        function is not called and ``nan`` is returned. Default is None.
    
    Returns
    -------
    decorator : function
        A decorator function that wraps a feature computation function and applies
        the specified preprocessing operations to the input time series before
        passing it to the wrapped function.
    """
    if preprocess is not None and preprocess not in _PREPROCESS_LABELS:
        raise ValueError(f"Unknown preprocess setting '{preprocess}'; "
                         f"supported: {sorted(_PREPROCESS_LABELS)}")

    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            if zscore:
                x = z_score(x)
            if preprocess == 'decimate_ac1e':
                x = decimate_ac1e(x, rezscore=zscore)
                if not isinstance(x, np.ndarray):
                    return np.nan  # no 1/e time: undefined, like hctsa's NaN
            if absval:
                x = np.abs(x)
            return func(x, *args, **kwargs)
        return wrapper
    return decorator

def z_score(x: ArrayLike) -> np.ndarray:
    """
    Z-score the input data vector.

    This function standardizes the input array by removing the mean and scaling to unit variance.
    It performs the z-scoring operation twice to reduce numerical error, as recommended for 
    high-precision applications.

    Parameters
    ----------
    x : array-like
        Input data vector (1D array or list of numbers).

    Returns
    -------
    np.ndarray
        The z-scored version of the input data.
    """
    # Convert input to numpy array
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input cannot be converted to numeric array: {e}")
    
    if x.size == 0:
        raise ValueError("Input array is empty.")

    # Check for NaNs, infs, etc. i.e., check data is finite
    if not np.isfinite(x).all():
        raise ValueError(f"data contains non-finite values (NaN/inf) at "
                         f"idxs: {np.argwhere(np.isfinite(x) == False)}")
    
    # robust checks for constant values
    var_x = np.var(x, ddof=1)
    if var_x < 1e-10:
        raise ValueError(f"Data has sample variance {var_x:.2e} < 1e-10. "
                         "Values appear to be constant.")
    data_range = np.ptp(x)  # peak-to-peak (max - min)
    if data_range < 1e-10:
        raise ValueError(f"Data range {data_range:.2e} < 1e-10. Values appear to be constant.")

    # Z-score twice to reduce numerical error
    zscored_data = np.divide((x - np.mean(x)), np.std(x, ddof=1))
    zscored_data = np.divide((zscored_data - np.mean(zscored_data)), np.std(zscored_data, ddof=1))

    return zscored_data

def ljung_box_pvalue(x: ArrayLike, n_lags: int = 20, model_df: int = 0) -> float:
    """
    P-value of the Ljung-Box Q-test for autocorrelation up to ``n_lags``.

    Equivalent to ``statsmodels.stats.diagnostic.acorr_ljungbox(x,
    lags=[n_lags], model_df=model_df)`` but O(N * n_lags) rather than O(N^2):
    it forms only the ``n_lags + 1`` autocovariances actually needed instead of
    statsmodels' full ``acf(fft=False)`` (which computes the entire N-lag
    autocovariance via ``np.correlate(..., 'full')``). Matches statsmodels to
    ~1e-15; not bit-identical, because ``np.dot`` (BLAS) rounds differently than
    numpy's full-array correlation.

    Parameters
    ----------
    x : array-like
        Input series, e.g. model residuals.
    n_lags : int, optional
        Maximum lag tested. Default is 20.
    model_df : int, optional
        Degrees of freedom consumed by a fitted model; the chi-squared
        reference distribution uses ``n_lags - model_df`` degrees of freedom.
        Default is 0.

    Returns
    -------
    float
        The Ljung-Box p-value.
    """
    x = np.asarray(x)
    t = np.sum(~np.isnan(x)) # effective sample size
    n_lags = min(n_lags, t - 1)
    nobs = x.shape[0]
    xo = x - x.mean()
    acov = np.array([np.dot(xo[:nobs-k], xo[k:]) for k in range(n_lags+1)]) / nobs
    sacf = acov / acov[0]
    sacf2 = sacf[1:n_lags+1]**2 / (nobs - np.arange(1, n_lags+1))
    q = nobs * (nobs+2) * np.cumsum(sacf2)[n_lags-1]
    return chi2.sf(q, n_lags - model_df)

def matlab_quantile(x: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Quantiles of `x` at the proportions `p`, bit-for-bit as MATLAB's ``quantile``.

    MATLAB's linear-interpolation scheme is the one NumPy calls ``method='hazen'``,
    but the two evaluate it with different floating-point arithmetic and can disagree
    in the last bit. That is enough to move a point across a bin edge when a quantile
    lands on an order statistic, so this reproduces the arithmetic of MATLAB's
    ``prctile`` (which ``quantile`` calls as ``prctile(x, 100*p)``) exactly.

    Parameters
    ----------
    x : array-like
        The input data.
    p : array-like
        The quantile proportions, in [0, 1].

    Returns
    -------
    numpy.ndarray
        The quantiles of `x`.
    """
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    # quantile() defers to prctile() with percentages; the round trip through 100 is
    # part of the arithmetic being reproduced
    r = (100.0 * np.atleast_1d(np.asarray(p, dtype=float)) / 100.0) * n
    k = np.floor(r + 0.5)  # index of the row just before r
    kp1 = k + 1            # index of the row just after r
    r = r - k              # the ratio between the two rows

    # Cap indices that fall outside the range 1 to n
    k = np.where((k < 1) | np.isnan(k), 1, k).astype(int)
    kp1 = np.minimum(kp1, n).astype(int)
    xk, xkp1 = xs[k - 1], xs[kp1 - 1]

    y = (0.5 + r) * xkp1 + (0.5 - r) * xk
    y = np.where(r == -0.5, xk, y)   # values hit exactly are copied, not interpolated
    return np.where(xk == xkp1, xk, y)  # as are identical values


def histc(x: ArrayLike, bins: ArrayLike) -> np.ndarray:
    """Counts the number of values in x that are within each specified bin."""
    # Get indices of the bins to which each value in input array belongs.
    map_to_bins = np.digitize(x, bins)
    # Vectorised count. (idx - 1) % nbins reproduces the original loop's res[el-1]
    # wrap-around exactly: below-range (el==0) -> -1 -> last bin; above-range
    # (el==nbins) -> nbins-1 -> last bin.
    nbins = bins.shape[0]
    res = np.bincount((map_to_bins - 1) % nbins, minlength=nbins).astype(float)
    return res

def bin_picker(x_min: float, x_max: float, n_bins: Union[None, int],
               bin_width_est: Union[None, float] = None) -> np.ndarray:
    """
    Choose histogram bins. 

    Parameters
    -----------
    x_min : float
        Minimum value of the data range.
    x_max : float
        Maximum value of the data range.
    n_bins : int or None
        Number of bins. If None, an automatic rule is used.
    bin_width_est : float or None
        Estimate of the bin width.

    Returns
    --------
    edges : numpy.ndarray
        Array of bin edges.
    """
    if bin_width_est is None:
        raw_bin_width = abs(x_max - x_min)/n_bins
    else:
        raw_bin_width = bin_width_est

    if x_min is not None:
        if not np.issubdtype(type(x_min), np.floating):
            raise ValueError("Input must be float type when number of bins is specified.")

        xscale = max(abs(x_min), abs(x_max))
        xrange = x_max - x_min

        # Make sure the bin width is not effectively zero
        raw_bin_width = max(raw_bin_width, np.spacing(xscale))

        # If the data are not constant, place the bins at "nice" locations
        if xrange > max(np.sqrt(np.spacing(xscale)), np.finfo(xscale).tiny):
            # Choose the bin width as a "nice" value
            pow_of_ten = 10 ** np.floor(np.log10(raw_bin_width))
            rel_size = raw_bin_width / pow_of_ten  # guaranteed in [1, 10)

            # Automatic rule specified
            if n_bins is None:
                if rel_size < 1.5:
                    bin_width = 1 * pow_of_ten
                elif rel_size < 2.5:
                    bin_width = 2 * pow_of_ten
                elif rel_size < 4:
                    bin_width = 3 * pow_of_ten
                elif rel_size < 7.5:
                    bin_width = 5 * pow_of_ten
                else:
                    bin_width = 10 * pow_of_ten

                left_edge = max(min(bin_width * np.floor(x_min / bin_width), x_min), -np.finfo(x_max).max)
                n_bins_actual = max(1, np.ceil((x_max - left_edge) / bin_width))
                right_edge = min(max(left_edge + n_bins_actual * bin_width, x_max), np.finfo(x_max).max)

            # Number of bins specified
            else:
                bin_width = pow_of_ten * np.floor(rel_size)
                left_edge = max(min(bin_width * np.floor(x_min / bin_width), x_min), -np.finfo(x_min).max)
                if n_bins > 1:
                    ll = (x_max - left_edge) / n_bins
                    ul = (x_max - left_edge) / (n_bins - 1)
                    p10 = 10 ** np.floor(np.log10(ul - ll))
                    bin_width = p10 * np.ceil(ll / p10)

                n_bins_actual = n_bins
                right_edge = min(max(left_edge + n_bins_actual * bin_width, x_max), np.finfo(x_max).max)

        else:  # the data are nearly constant
            if n_bins is None:
                n_bins = 1

            bin_range = max(1, np.ceil(n_bins * np.spacing(xscale)))
            left_edge = np.floor(2 * (x_min - bin_range / 4)) / 2
            right_edge = np.ceil(2 * (x_max + bin_range / 4)) / 2

            bin_width = (right_edge - left_edge) / n_bins
            n_bins_actual = n_bins

        if not np.isfinite(bin_width):
            edges = np.linspace(left_edge, right_edge, n_bins_actual + 1)
        else:
            edges = np.concatenate([
                [left_edge],
                left_edge + np.arange(1, n_bins_actual) * bin_width,
                [right_edge]
            ])
    else:
        # empty input
        if n_bins is not None:
            edges = np.arange(n_bins + 1, dtype=float)
        else:
            edges = np.array([0.0, 1.0])

    return edges

def simple_binner(x_data: ArrayLike, num_bins: int) -> tuple:
    """
    Generate a histogram from equally spaced bins.
   
    Parameters
    ----------
    x_data : array-like 
        A data vector.
    num_bins : int 
        The number of bins.

    Returns
    -------
    tuple: (N, binEdges)
       The counts and extremities of the bins.
    """
    min_x = np.min(x_data)
    max_x = np.max(x_data)
    
    # Linearly spaced bins:
    bin_edges = np.linspace(min_x, max_x, num_bins + 1)
    # Vectorised: searchsorted against the interior edges gives each point's bin in
    # one pass (half-open interior bins; the max value equals the last edge so it
    # lands in the final inclusive bin, matching the loop).
    idx = np.searchsorted(bin_edges[1:-1], x_data, side='right')
    N = np.bincount(idx, minlength=num_bins).astype(int)

    return N, bin_edges

def point_of_crossing(x: ArrayLike, threshold: float) -> tuple:
    """
    Linearly interpolate to the point of crossing a threshold

    Parameters
    ----------
    x : array-like)
        a vector
    threshold : float
        a threshold x crosses

    Returns
    -------
        tuple: (firstCrossing, pointOfCrossing)
        firstCrossing (int): the first discrete value after which a crossing event has occurred
        pointOfCrossing (float): the (linearly) interpolated point of crossing
        Both are NaN if every element of x is NaN.
    """
    x = np.asarray(x)

    # An entirely undefined input (e.g. the autocorrelation of a constant series is
    # 0/0 at every lag) has no answer; this differs from a well-defined sequence that
    # simply never crosses the threshold, which saturates at the last element below.
    if np.all(np.isnan(x)):
        return np.nan, np.nan

    if x[0] > threshold:
        crossings = np.where((x - threshold) < 0)[0]
    else:
        crossings = np.where((x - threshold) > 0)[0]

    if crossings.size == 0:
        # Never crosses: report the last element. BF_PointOfCrossing returns
        # N here, an index its caller converts to the lag N-1, so the lags
        # returned from this branch stop one short of the array length.
        n = len(x)
        fc = n - 1
        poc = n - 1
    else:
        fc = crossings[0]
        # continuous version
        value_before = x[fc - 1]
        value_after = x[fc]
        poc = (
            fc - 1
            + (threshold - value_before) / (value_after - value_before)
        )

    return fc, poc

def sign_change(y: Union[list, np.ndarray], do_find: int = 0) -> ArrayLike:
    """
    Where a data vector changes sign.

    Parameters
    ----------
    y : array-like
        The input time series.
    do_find : int
        - If 0, returns a logical vector with 1s where the input changes sign.
        - If 1, returns a logical vector of indices where the input vector changes sign.
    """
    if do_find == 0:
        return np.multiply(y[1:],y[0:len(y)-1]) < 0
    indexs = np.where((np.multiply(y[1:],y[0:len(y)-1]) < 0))[0]

    return indexs

def make_buffer(y: ArrayLike, buffer_size: int) -> np.ndarray:
    """
    Make a buffered version of a time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    buffer_size : int
        The length of each buffer segment.

    Returns
    -------
    y_buffer : ndarray
        2D array where each row is a segment of length `buffer_size` 
        corresponding to consecutive, non-overlapping segments of the input time series.
    """
    y = np.asarray(y)
    N = len(y)

    num_buffers = int(np.floor(N/buffer_size))

    # may need trimming
    y_buffer = y[:num_buffers*buffer_size]
    # then reshape
    y_buffer = y_buffer.reshape((num_buffers,buffer_size))

    return y_buffer

def make_mat_buffer(x: ArrayLike, n: int, p: int = 0,
                    opt: Union[str, None] = None) -> np.ndarray:
    '''
    Create a buffer array.

    Taken from: https://stackoverflow.com/questions/38453249/does-numpy-have-
        a-function-equivalent-to-matlabs-buffer 

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
        Buffer array created from x
    '''
    
    if opt not in [None, 'nodelay']:
        raise ValueError(f'{opt} not implemented')

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result

# ------------------------------------------------------------------------------
# Adaptive time delays, Theiler windows and decimation (hctsa BF_GetTau,
# BF_TheilerWindow, BF_PreProcess)
# ------------------------------------------------------------------------------
_TAU_RULES = ('ac', 'ac1e', 'mi', 'mi-gaussian')
_TAU_CACHE: list = []  # (rule, N, digest, tau); small FIFO cache, like hctsa's persistent cache
_TAU_CACHE_SIZE = 4


def _round_half_away(x: float) -> float:
    """MATLAB's ``round``: halves are rounded away from zero (NumPy rounds to even)."""
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


def _ml_std(x: ArrayLike) -> float:
    """MATLAB's ``std``: the sample standard deviation (N - 1), which is 0 (not NaN) for a single value."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def _acf_fourier(y: np.ndarray) -> np.ndarray:
    """ACF at lags 0..N-1 (CO_AutoCorr(y, [], 'Fourier')); all-NaN for a constant series."""
    from .operations.correlation import autocorr
    with np.errstate(all='ignore'):
        return np.asarray(autocorr(y, [], 'Fourier'), dtype=float).ravel()


def _tau_ac(y: np.ndarray) -> float:
    """First zero crossing of the ACF (CO_FirstCrossing(y,'ac',0,'discrete'))."""
    if y.size < 2:
        return np.nan
    fc, _ = point_of_crossing(_acf_fourier(y), 0.0)
    return np.nan if np.isnan(fc) else int(fc)


def _tau_ac1e(y: np.ndarray) -> float:
    """Floor of the first 1/e crossing of the ACF, at least 1 (BF_GetTau's TauAC1e)."""
    if y.size < 2:
        return np.nan
    acf = _acf_fourier(y)
    threshold = 1.0 / np.e
    if np.any(np.isnan(acf)) or not np.any(acf < threshold):
        # degenerate series, or the ACF never decays to 1/e
        return np.nan
    _, poc = point_of_crossing(acf, threshold)  # already in lag units
    return int(max(1, np.floor(poc)))


def _tau_mi_gaussian(y: np.ndarray) -> float:
    """First local minimum of the Gaussian AMI (CO_FirstMin(y,'mi-gaussian')).

    The AMI at each lag is computed from the Pearson correlation of the two delayed windows,
    as IN_AutoMutualInfo does, stopping at the first minimum. (A vectorized FFT/cumulative-sum
    curve is not used: it loses precision at long lags of smooth series, where r is close to 1.)
    """
    n = y.size
    prev2 = prev1 = np.nan  # AMI at lags i-2, i-1
    # IN_AutoMutualInfo gives NaN for lags > N - 5, and CO_FirstMin gives up on a NaN
    for i in range(1, n - 4):
        y1, y2 = y[:-i], y[i:]
        d1, d2 = y1 - y1.mean(), y2 - y2.mean()
        den = np.sqrt(np.dot(d1, d1) * np.dot(d2, d2))
        if not den > 0:
            return np.nan
        r = min(1.0, max(-1.0, np.dot(d1, d2) / den))
        with np.errstate(divide='ignore'):
            cur = -0.5 * np.log(1.0 - r * r)
        if np.isnan(cur):
            return np.nan
        if i == 2 and cur > prev1:
            return 1  # already increases at lag 2 from lag 1
        if i > 2 and prev2 > prev1 < cur:
            return i - 1
        prev2, prev1 = prev1, cur
    return np.nan


def _tau_mi(y: np.ndarray) -> float:
    """min(first minimum of the Kraskov (k=4) AMI, 'ac1e' delay), at least 1 (BF_GetTau's TauMI)."""
    from .operations.information import automutual_info
    n = y.size
    tau_ac = _tau_ac1e(y)
    if np.isnan(tau_ac):
        max_lag = n // 10
    else:
        # only lags up to tau_ac can matter; one extra lag to detect a minimum at tau_ac
        max_lag = min(int(tau_ac) + 1, n // 10)
    if tau_ac == 1:
        return 1  # can't go below 1
    if max_lag < 2:
        return tau_ac  # series too short to locate an AMI minimum
    lags = list(range(1, max_lag + 1))
    ami = automutual_info(y, lags, 'kraskov1', 4)
    amis = np.array([ami[f'ami{l}'] for l in lags], dtype=float)

    # first local minimum (at lag 1 if the AMI already increases from lag 1 to 2)
    tau_min = np.nan
    for l in range(1, max_lag):  # 1-based lag l; amis[l-1] is AMI(l)
        if np.isnan(amis[l]):
            break
        if amis[l] > amis[l - 1] and (l == 1 or amis[l - 2] > amis[l - 1]):
            tau_min = l
            break

    if not np.isnan(tau_min):
        return min(tau_min, tau_ac) if not np.isnan(tau_ac) else tau_min
    if not np.isnan(tau_ac):
        return tau_ac
    # no 1/e crossing of the ACF and no AMI minimum: first drop of the AMI below the
    # Gaussian AMI at a correlation of 1/e
    mi_threshold = -0.5 * np.log(1.0 - np.exp(-2.0))
    below = np.flatnonzero(amis < mi_threshold)
    return int(below[0] + 1) if below.size else np.nan


def get_tau(y: ArrayLike, rule: Union[int, str] = 'ac1e') -> Union[int, float]:
    """
    A time delay (in samples) set by the time series' own timescale.

    Port of hctsa's ``BF_GetTau``. Adaptive delays let a feature measure structure
    relative to the series' own correlation time rather than the sampling interval.
    Both adaptive rules scale linearly with the sampling rate and sit on the low side
    of the correlation time: iterated maps, which decorrelate within one step, keep
    ``tau = 1``.

    Parameters
    ----------
    y : array-like
        The input time series.
    rule : int or {'ac1e', 'mi', 'mi-gaussian', 'ac'}
        How to set the delay:

        - an integer: returned unchanged (so a fixed delay and a rule can share an argument);
        - ``'ac1e'``: the largest integer lag at which the ACF is still at least 1/e (the
          floor of the linearly interpolated first 1/e crossing), and at least 1;
        - ``'mi'``: ``max(1, min(first local minimum of the Kraskov k=4 AMI, ac1e))``. If the
          ACF never falls to 1/e, the AMI is searched up to ``N // 10`` lags for its first local
          minimum, or else its first drop below ``-0.5 * log(1 - exp(-2))`` (the Gaussian AMI
          at a correlation of 1/e);
        - ``'mi-gaussian'``: the first local minimum of the Gaussian AMI (the rule 'mi' meant
          in earlier versions of hctsa; a monotonic function of |ACF|, so not a nonlinear
          timescale);
        - ``'ac'``: the first zero crossing of the ACF (a series whose ACF is defined but
          never crosses zero gives ``N - 1``, as in hctsa).

    Returns
    -------
    int or float
        The delay as an ``int``, or ``nan`` if it cannot be determined (constant series,
        ACF that never falls to 1/e for 'ac1e', series too short, ...). Callers should
        propagate a NaN delay to a NaN feature value.

    Notes
    -----
    The last few results are cached (keyed on the rule and the series' values), since
    many operations resolve the same delay for the same series and the Kraskov AMI behind
    ``'mi'`` is the expensive part.

    Raises
    ------
    ValueError
        For an unknown rule, or a non-integral numeric ``rule``.
    """
    if not isinstance(rule, str):
        if rule is None or not np.isfinite(rule) or float(rule) != int(rule):
            raise ValueError(f"A numeric time delay must be an integer, got {rule!r}.")
        return int(rule)
    if rule not in _TAU_RULES:
        raise ValueError(f"Unknown time-delay rule '{rule}'; expected an integer or one of {_TAU_RULES}.")

    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    digest = hashlib.blake2b(y.tobytes(), digest_size=16).digest()
    for c_rule, c_n, c_digest, c_tau in _TAU_CACHE:
        if c_rule == rule and c_n == y.size and c_digest == digest:
            return c_tau

    if rule == 'ac1e':
        tau = _tau_ac1e(y)
    elif rule == 'mi':
        tau = _tau_mi(y)
    elif rule == 'mi-gaussian':
        tau = _tau_mi_gaussian(y)
    else:  # 'ac'
        tau = _tau_ac(y)
    if not isinstance(tau, (int, np.integer)):
        tau = np.nan if np.isnan(tau) else int(tau)

    _TAU_CACHE.append((rule, y.size, digest, tau))
    del _TAU_CACHE[:-_TAU_CACHE_SIZE]
    return tau


def theiler_window(y: Union[ArrayLike, None], spec: Union[int, float, list, tuple],
                   N: Union[int, None] = None) -> Union[int, float]:
    """
    Resolve a Theiler-window specification to a number of samples.

    Port of hctsa's ``BF_TheilerWindow``. Neighbor-based methods exclude candidate
    neighbors j of a reference point i with ``|i - j| <= W``, since those are close in
    state space only because successive values are correlated (Theiler, Phys. Rev. A 34,
    2427, 1986). The window should span the time over which values stay correlated, which
    differs from series to series.

    Parameters
    ----------
    y : array-like or None
        The time series (used to compute its autocorrelation time). May be ``None`` for a
        numeric ``spec``.
    spec : [str, number] or number
        The Theiler window:

        - ``['ac', k]`` (or tuple): ``ceil(k * first zero crossing of the ACF)`` (recommended);
        - ``['ac1e', k]``: ``ceil(k * get_tau(y, 'ac1e'))``; shorter and more stable than the
          zero crossing, NaN if the ACF never falls to 1/e;
        - an integer >= 0: a fixed number of samples;
        - a number in (0, 1): a proportion of ``N`` (legacy; scales with series length),
          rounded half away from zero as in MATLAB. (A value of 1 or more is a number of samples.)
    N : int, optional
        The length a proportional window refers to. Default ``len(y)``.

    Returns
    -------
    int or float
        The window in samples, or ``nan`` when the ACF-based delay cannot be set (constant
        series, ...). Callers should propagate NaN to a NaN feature value.

    Raises
    ------
    ValueError
        For a malformed ``spec``.
    """
    if isinstance(spec, (list, tuple)):
        ok = (len(spec) == 2 and isinstance(spec[0], str) and spec[0] in ('ac', 'ac1e')
              and isinstance(spec[1], (int, float, np.integer, np.floating))
              and not isinstance(spec[1], bool) and spec[1] >= 0)
        if not ok:
            raise ValueError("Theiler window must be specified as ['ac', k] or ['ac1e', k], with k >= 0")
        tau = get_tau(y, spec[0])
        if np.isnan(tau):
            return np.nan
        return int(np.ceil(spec[1] * tau))
    if isinstance(spec, (int, float, np.integer, np.floating)) and not isinstance(spec, bool) and spec >= 0:
        if 0 < spec < 1:  # a proportion of the series length
            if N is None:
                if y is None:
                    raise ValueError("N is required for a proportional window when y is None.")
                N = len(y)
            return int(_round_half_away(spec * N))
        return int(_round_half_away(spec))
    raise ValueError("Unrecognized Theiler window specification")


def pre_process(y: ArrayLike, how: Union[str, None]) -> Union[np.ndarray, float]:
    """
    Pre-process a time series (hctsa's ``BF_PreProcess``).

    Parameters
    ----------
    y : array-like
        The input time series.
    how : {'diff1', 'rescale_tau', 'decimate_ac1e'} or None
        - ``'diff1'``: incremental differences;
        - ``'rescale_tau'``: coarse-grain by averaging non-overlapping windows whose length is the
          first zero crossing of the ACF;
        - ``'decimate_ac1e'``: keep one sample per (floored) 1/e autocorrelation time
          (``y[::get_tau(y, 'ac1e')]``), so features of the result do not change trivially with
          the sampling rate. Iterated maps (ACF below 1/e at lag 1) are unchanged. In hctsa it is
          used as ``zscore(BF_PreProcess(x_z, 'decimate_ac1e'))``, see :func:`decimate_ac1e`.
        - ``None`` or ``''``: no change.

    Returns
    -------
    numpy.ndarray or float
        The processed series, or ``nan`` (scalar) when the required time delay cannot be set
        (as hctsa, which returns the scalar NaN).
    """
    y = np.asarray(y, dtype=float).ravel()
    if how is None or how == '':
        return y
    if how == 'diff1':
        return np.diff(y)
    if how == 'rescale_tau':
        tau = get_tau(y, 'ac')
        if np.isnan(tau) or tau < 1 or tau > y.size:
            return np.nan
        return np.mean(make_buffer(y, int(tau)), axis=1)
    if how == 'decimate_ac1e':
        tau = get_tau(y, 'ac1e')
        if np.isnan(tau):
            return np.nan
        return y[::int(tau)]
    raise ValueError(f"Unknown preprocessing setting: '{how}'")


def _zscore_matlab(x: np.ndarray) -> np.ndarray:
    """MATLAB's ``zscore``: (x - mean)/std with the N-1 std, and a zero std treated as 1."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x, ddof=1) if x.size > 1 else 0.0
    if not sd > 0:
        sd = 1.0
    return (x - mu) / sd


def decimate_ac1e(x_z: ArrayLike, rezscore: bool = True) -> Union[np.ndarray, float]:
    """
    ``zscore(BF_PreProcess(x_z, 'decimate_ac1e'))``: decimate by the 1/e autocorrelation time.

    Keeps every ``tau``-th sample, ``tau = get_tau(x_z, 'ac1e')``, then z-scores the result
    (MATLAB ``zscore``: N-1 std, a constant result becomes zeros). hctsa uses this for the
    ``*_dec`` variants of rate-dependent features. A config requests it with
    ``preprocess: decimate_ac1e`` (see :class:`~pyhctsa.calculator.FeatureCalculator`).

    Parameters
    ----------
    x_z : array-like
        The (already z-scored) time series.
    rezscore : bool, optional
        Z-score the decimated series (default True, as hctsa).

    Returns
    -------
    numpy.ndarray or float
        The decimated series, or scalar ``nan`` if the delay cannot be set (hctsa's behavior:
        the operation then returns NaN).
    """
    y = pre_process(x_z, 'decimate_ac1e')
    if not isinstance(y, np.ndarray):
        return np.nan
    return _zscore_matlab(y) if rezscore else y


def time_delay_embed(y: ArrayLike, m: int, tau: Union[int, str] = 1,
                     reverse: bool = False) -> np.ndarray:
    """
    Time-delay embedding of a univariate time series into an `m`-dimensional space.

    Row ``k`` of the result is ``[y[k], y[k + tau], ..., y[k + (m-1)*tau]]``, so the
    columns run from the least- to the most-delayed copy of the series.

    Parameters
    ----------
    y : array-like
        The input time series.
    m : int
        The embedding dimension. Must be at least 1.
    tau : int or str, optional
        The time delay between successive coordinates, or a rule understood by
        :func:`get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``, ``'mi-gaussian'``). Default is 1.
    reverse : bool, optional
        If True, order the columns from the most- to the least-delayed copy
        (i.e. reverse the column order). Default is False.

    Returns
    -------
    numpy.ndarray
        The embedded time series, of shape ``(len(y) - (m-1)*tau, m)``.

    Raises
    ------
    ValueError
        If `m` is less than 1, the time series is too short to embed with the
        given parameters, or a ``tau`` rule gives no delay for this series (NaN).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    m = int(m)
    if isinstance(tau, str):
        tau = get_tau(y, tau)
        if np.isnan(tau):
            raise ValueError('Time delay could not be determined for this time series.')
    tau = int(tau)

    if m < 1:
        raise ValueError(f'Embedding dimension must be at least 1, got {m}.')

    n_embed = n - (m - 1) * tau
    if n_embed <= 0:
        raise ValueError(f'Time series (N = {n}) too short to embed with these '
                         'embedding parameters.')

    # one broadcast gather rather than a per-column copy
    idx = np.arange(n_embed)[:, None] + tau * np.arange(m)[None, :]
    embedded = y[idx]

    return embedded[:, ::-1] if reverse else embedded

def binarize(y: ArrayLike, binarize_how: str = 'diff') -> ArrayLike:
    """
    Converts an input vector into a binarized version.

    Parameters
    -----------
    y : array-like
        The input time series
    binarize_how : str, optional
        Method to binarize the time series: 'diff', 'mean', 'median', 'iqr'.
    
    Returns
    --------
    y_bin : array-like
        The binarized time series
    """
    if binarize_how == 'diff':
        # Binary signal: 1 for stepwise increases, 0 for stepwise decreases
        y_bin = _step_binary(np.diff(y))
    
    elif binarize_how == 'mean':
        # Binary signal: 1 for above mean, 0 for below mean
        y_bin = _step_binary(y - np.mean(y))
    
    elif binarize_how == 'median':
        # Binary signal: 1 for above median, 0 for below median
        y_bin = _step_binary(y - np.median(y))
    
    elif binarize_how == 'iqr':
        # Binary signal: 1 if inside interquartile range, 0 otherwise
        iqr = np.quantile(y,[.25,.75], method='hazen')
        iniqr = np.logical_and(y > iqr[0], y<iqr[1])
        y_bin = np.zeros(len(y))
        y_bin[iniqr] = 1
    else:
        raise ValueError(f"Unknown binary transformation setting '{binarize_how}'")

    return y_bin

def _step_binary(x : ArrayLike) -> ArrayLike:
    # Transform real values to 0 if <=0 and 1 if >0:
    y = np.zeros(len(x))
    y[x > 0] = 1

    return y

def x_corr(x: ArrayLike, y: ArrayLike, normed: bool = True, max_lags: int = 10) -> tuple:
    """
    Calculates the cross-correlation coefficients or inner products between two
    signals at various lags. The function computes correlations up to a specified
    maximum lag in both positive and negative directions.

    Taken from https://github.com/colizoli/xcorr_python 

    Parameters
    ----------
    x : array-like
        First input signal. Must be equal length to y.
    y : array-like
        Second input signal. Must be equal length to x.
    normed : bool, optional
        If True, returns normalized correlation coefficients (default).
        If False, returns raw inner products.
        Default is True.
    max_lags : int, optional
        Maximum lag to compute in both directions. Must be positive and less than
        the signal length. Default is 10.

    Returns
    -------
    tuple
        A tuple containing:
        - lags : np.ndarray
            Array of lag values ranging from -max_lags to +max_lags.
        - c : np.ndarray
            Cross-correlation values at each lag. Normalized correlation coefficients
            if normed=True, otherwise raw inner products.
    """

    nx = len(x)
    if nx != len(y):
        raise ValueError('x and y must be equal length')
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if max_lags is None:
        max_lags = nx - 1

    if max_lags >= nx or max_lags < 1:
        raise ValueError('max_lags must be None or strictly '
                         f'positive <{nx}')

    lags = np.arange(-max_lags, max_lags + 1)
    c = c[nx - 1 - max_lags:nx + max_lags]
    return lags, c

def make_function_name_mappings(
    yaml_file: Union[str, None] = None, csv_out_fpath: Union[None, str] = None) -> pd.DataFrame:
    """
    Map pyhctsa function names to their legacy counterparts in the MATLAB HCTSA.
    """
    # only the names are needed, so parse with a private loader that ignores
    # !range rather than overriding the constructor on the shared SafeLoader
    class _MappingLoader(yaml.SafeLoader):
        pass
    _MappingLoader.add_constructor("!range", lambda loader, node: None)

    if yaml_file is None:
        yaml_file = resources.files("pyhctsa.configurations").joinpath("hctsa.yaml")

    with open(yaml_file, "r", encoding="utf-8") as f:
        yam = yaml.load(f, Loader=_MappingLoader)
    module_dfs = []
    for module in yam:
        corr_mod = yam[module]
        python_funcs = []
        ml_funcs = []

        for pyfunc in corr_mod:
            python_funcs.append(pyfunc)

            # If legacy_name missing, use NaN
            meta = corr_mod[pyfunc] or {}
            ml_funcs.append(meta.get("legacy_name", np.nan))

        df = pd.DataFrame(
            {"pyhctsa name": python_funcs, "hctsa legacy name": ml_funcs}
        )
        df["pyhctsa module"] = module
        module_dfs.append(df)

    df_all_modules = pd.concat(module_dfs, ignore_index=True)

    # append version number
    df_all_modules.attrs["feature_set_version"] = __version__

    if csv_out_fpath:
        # Write metadata as comment lines, then the CSV
        with open(csv_out_fpath, "w", encoding="utf-8", newline="") as f:
            df_all_modules.to_csv(f, index=False)

    return df_all_modules
