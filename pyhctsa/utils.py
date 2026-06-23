import csv
import os
from functools import wraps
from itertools import product
from importlib.metadata import PackageNotFoundError, version
from importlib import resources
from typing import Union, Callable, Any
import yaml
from pyhctsa import __version__
from pathlib import Path
import logging
logger = logging.getLogger('pyhctsa')
import warnings

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.stats import chi2

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

def _build_repr_html(feature_funcs: dict[str, Callable],
    skipped_functions: list[tuple[str, list[str]]],
    config: dict[str, Any],
    config_path: str) -> str:
    """
    Construct an sklearn-like information box when instantiating a FeatureCalculator object.

    Contains information about the modules and feature functions loaded for a given
    configuration YAML file.

    """

    config_name = Path(config_path).name # extract the name of the module
    n_feature_funcs = len(feature_funcs)
    n_skipped = len(skipped_functions)
    module_counts = {}
    for module_key, module_cfg in config.items():
        count = 0
        for feature_name, feature_config in module_cfg.items():
            configs = feature_config.get("configs", [{}])
            for config_item in configs:
                clean = {k: v for k, v in config_item.items()
                         if k not in ("zscore", "abs", "select", "exclude")}
                keys = list(clean.keys())
                values = [v if isinstance(v, list) else [v] for v in clean.values()]
                combos = list(product(*values)) if keys else [()]
                count += len(combos)
        if count > 0:
            module_counts[module_key] = count
    module_counts = dict(sorted(module_counts.items(), key=lambda x: -x[1]))
    total = sum(module_counts.values())
    max_count = max(module_counts.values(), default=1)
    rows = ""
    skipped_color = "#A32D2D" if n_skipped > 0 else "#412402"
    for mod, count in module_counts.items():
        pct = count / max_count * 100
        pct_of_total = count / total * 100
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="font-size:12px;color:#633806;width:140px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{mod}</span>'
            f'<div style="flex:1;background:#FAC775;border-radius:2px;height:6px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:#BA7517;"></div>'
            f'</div>'
            f'<span style="font-size:12px;color:#412402;min-width:32px;text-align:right;">{count}</span>'
            f'<span style="font-size:11px;color:#633806;min-width:36px;text-align:right;">({pct_of_total:.1f}%)</span>'
            f'</div>'
        )
    return (
        f'<div style="display:inline-block;border:0.5px solid #EF9F27;border-left:4px solid #BA7517;'
        f'border-radius:8px;padding:1rem 1.25rem;background:#FAEEDA;'
        f'font-family:monospace;min-width:340px;max-width:560px;">'
        f'<div style="font-size:13px;font-weight:500;color:#412402;margin-bottom:12px;">FeatureCalculator</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;">'
        f'<div style="background:#FAC775;border-radius:8px;padding:10px 12px;">'
        f'<div style="font-size:11px;color:#633806;margin-bottom:2px;">operations loaded</div>'
        f'<div style="font-size:20px;font-weight:500;color:#412402;">{n_feature_funcs}</div>'
        f'</div>'
        f'<div style="background:#FAC775;border-radius:8px;padding:10px 12px;">'
        f'<div style="font-size:11px;color:#633806;margin-bottom:2px;">operations skipped</div>'
        f'<div style="font-size:20px;font-weight:500;color:{skipped_color};">{n_skipped}</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:11px;color:#633806;margin-bottom:6px;">operations per module</div>'
        f'{rows}'
        f'<div style="font-size:11px;color:#633806;margin-top:10px;border-top:0.5px solid #EF9F27;padding-top:8px;word-break:break-all;">'
        f'config: {config_name}'
        f'</div>'
        f'</div>'
    )

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
        # Never crosses
        n = len(x)
        fc = n
        poc = n
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
