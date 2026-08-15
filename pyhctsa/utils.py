import csv
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

    print(f"Loading {datasets[which]['desc']} dataset...")
    data_path = os.path.normpath(os.path.join(utils_dir, datasets[which]['path']))
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    dataset = datasets[which]['loader'](data_path)
    print(f"Loaded dataset of {len(dataset)} time series.")
    return dataset
    
def _preprocess_decorator(zscore: bool = False, absval: bool = False) -> Callable:
    """
    Decorator to preprocess time series data before feature computation.
    
    Applies optional z-score normalization and/or absolute value transformation
    to the input time series before passing it to the decorated function.

    Note that by default, the zscore operation will always be applied **before** the absolute 
    value operation if both are enabled.

    Parameters
    ----------
    zscore : bool, optional
        If True, z-score normalize the input time series to have mean 0 and 
        standard deviation 1. Default is False.
    absval : bool, optional
        If True, take the absolute value of all data points in the input time series.
        Default is False.
    
    Returns
    -------
    decorator : function
        A decorator function that wraps a feature computation function and applies
        the specified preprocessing operations to the input time series before
        passing it to the wrapped function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            if zscore:
                x = z_score(x)
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


def _first_index_past_threshold(p: ArrayLike, threshold: float,
                                side: str = 'under') -> Union[int, None]:
    """
    Index of the first element of `p` strictly past `threshold`, or None.

    The shared predicate behind the several "first time this curve drops below
    (or rises above) a level" features. Callers differ only in what they report
    for that index and what they fall back to when there is no crossing, so
    those policies stay with the callers rather than being folded in here.

    NaNs never satisfy either comparison, so a curve that is all-NaN returns
    None rather than an arbitrary index.

    Parameters
    ----------
    p : array-like
        The curve to scan.
    threshold : float
        The level to compare against.
    side : {'under', 'over'}, optional
        Whether to look for the first element below or above `threshold`.
        Default is ``'under'``.

    Returns
    -------
    int or None
        The 0-based index of the first qualifying element, or None if there
        is none.
    """
    p = np.asarray(p)
    if side == 'under':
        hits = np.flatnonzero(p < threshold)
    elif side == 'over':
        hits = np.flatnonzero(p > threshold)
    else:
        raise ValueError(f'Unknown setting: {side}')

    return int(hits[0]) if hits.size > 0 else None


def _matlab_linspace(d1: float, d2: float, n: int) -> np.ndarray:
    """
    MATLAB's ``linspace``.

    Differs from ``numpy.linspace`` in the endpoint arithmetic: MATLAB forms
    ``d1 + (0:n-1)*(d2-d1)/(n-1)`` and then *overwrites* the last element with
    ``d2``, rather than computing it. The two agree to within a bit almost
    everywhere, but the difference is visible once the result is passed through
    ``floor`` (as every caller here does), where a last-bit discrepancy moves an
    index by one.

    Parameters
    ----------
    d1, d2 : float
        First and last values of the generated vector.
    n : int
        Number of points. Must be at least 2.

    Returns
    -------
    numpy.ndarray
        ``n`` evenly spaced values from `d1` to `d2` inclusive.
    """
    d1 = float(d1)
    d2 = float(d2)
    n1 = n - 1
    if np.isinf((d2 - d1) * (n1 - 1)):
        # the naive product overflows; distribute the division instead
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


def histc(x: ArrayLike, bins: ArrayLike) -> int:
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
    """
    x = np.asarray(x)

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

def time_delay_embed(y: ArrayLike, m: int, tau: int = 1,
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
    tau : int, optional
        The time delay between successive coordinates. Default is 1.
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
        If `m` is less than 1, or the time series is too short to embed with the
        given parameters.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    m = int(m)
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
    yaml.SafeLoader.add_constructor("!range", lambda loader, node: None)

    if yaml_file is None:
        yaml_file = resources.files("pyhctsa.configurations").joinpath("hctsa.yaml")

    with open(yaml_file, "r", encoding="utf-8") as f:
        yam = yaml.safe_load(f)
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
