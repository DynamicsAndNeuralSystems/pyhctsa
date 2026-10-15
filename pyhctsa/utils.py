import csv
import hashlib
import math
import os
from fractions import Fraction
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
from scipy.signal import correlate
from scipy.stats import chi2


def _check_optional_deps(dep: str) -> bool:
    """Check whether an optional dependency exists.
    Returns True if available, else False."""
    try:
        version(dep)
        return True
    except PackageNotFoundError:
        return False

def _reference_series() -> list:
    """Well-behaved test series (a Gaussian AR(1), a positive series, a noisy sine, a random walk) from
    which :func:`nan_outputs` learns the field names of a function's output."""
    rng = np.random.RandomState(0)
    e = rng.randn(1000)
    ar = np.zeros(1000)
    for i in range(1, 1000):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    return [z(ar), np.exp(0.5 * z(ar)), z(np.sin(0.2 * np.arange(1000)) + 0.3 * rng.randn(1000)), z(np.cumsum(e))]

_FIELD_CACHE: dict = {}

def nan_outputs(func: Callable, *args, **kwargs) -> Union[dict, float]:
    """The output of ``func(y, *args, **kwargs)`` when every one of its outputs is undefined: a dict
    with all of its fields NaN (``nan`` if the function does not return a dict).

    The field names are those of the function's own output, with the same arguments, on a
    well-behaved reference series (found the first time they are needed, then remembered).
    """
    key = (func, repr(args), repr(sorted(kwargs.items())))
    if key not in _FIELD_CACHE:
        fields = None
        for ref in _reference_series():
            try:
                with np.errstate(all='ignore'):
                    out = func(ref, *args, **kwargs)
            except Exception:
                continue
            if isinstance(out, dict):
                fields = list(out)
                break
        _FIELD_CACHE[key] = fields
    fields = _FIELD_CACHE[key]
    return np.nan if fields is None else dict.fromkeys(fields, np.nan)

def dict_output(func: Callable) -> Callable:
    """Decorator for a feature function that returns a dict of outputs (one feature per field).

    When the function cannot compute anything (hctsa: ``out = NaN``, every output undefined) and
    would return a bare ``nan``, the decorated function returns the dict with every field NaN
    instead (see :func:`nan_outputs`), so that the feature names stay ``<label>.<field>``
    whatever the data.
    """
    @wraps(func)
    def wrapper(y, *args, **kwargs):
        out = func(y, *args, **kwargs)
        if isinstance(out, (float, np.floating)) and np.isnan(out):
            return nan_outputs(func, *args, **kwargs)
        return out
    return wrapper

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

def _even_split_points(a, b, n: int, rounding: str = 'round') -> np.ndarray:
    """``n`` integer positions evenly spread from ``a`` to ``b`` (inclusive), by exact arithmetic.

    Position ``k`` (``k = 0..n-1``) is the exact rational ``a + k (b - a)/(n - 1)`` rounded half
    up (``rounding='round'``) or floored (``rounding='floor'``), so that a position falling
    exactly halfway between two integers (or exactly on an integer) is never moved by
    floating-point error. hctsa uses such grids for split points, window lengths and reference
    indices: for example ``n = 3`` points over ``[1, N]`` with even ``N`` has a midpoint exactly
    halfway between two integers, and this function puts it on the upper one.

    Parameters
    ----------
    a, b : int, float or fractions.Fraction
        End points (a float is taken at its exact binary value).
    n : int
        Number of points. ``n <= 0`` gives an empty array and ``n == 1`` gives the first
        point only.
    rounding : {'round', 'floor'}
        How the exact positions are converted to integers.

    Returns
    -------
    numpy.ndarray
        int64 array of length ``max(n, 0)``, non-decreasing when ``b >= a``, from the rounded
        ``a`` to the rounded ``b``.
    """
    if rounding not in ('round', 'floor'):
        raise ValueError(f"Unknown rounding '{rounding}'")
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    fa, fb = Fraction(a), Fraction(b)
    d = math.lcm(fa.denominator, fb.denominator)
    A, B = int(fa * d), int(fb * d)                      # integer end points in units of 1/d
    den = d * max(n - 1, 1)
    num = A * max(n - 1, 1) + np.arange(n, dtype=np.int64) * (B - A)   # den * (exact positions)
    if rounding == 'floor':
        return num // den
    return (2 * num + den) // (2 * den)


def _arange_inclusive(start, step, stop):
    """Regularly spaced vector ``start, start + step, ..., stop`` with ``stop`` included.

    Parameters
    ----------
    start, step, stop : float
        First element, increment and bound. ``stop`` is the last element when the increment
        lands on it up to floating-point rounding (a relative tolerance of 1e-10 on the number
        of steps), so that e.g. ``(-5, 0.1, 5)`` has 101 elements; otherwise the last element
        is the largest ``start + k step`` not beyond ``stop``.

    Returns
    -------
    numpy.ndarray
        float64 vector (``start + step * arange(n)``, the last element clipped to ``stop`` if
        rounding took it past); empty if ``step == 0``, if the step points away from ``stop``
        or if any argument is NaN.
    """
    a = float(start)
    st = float(step)
    b = float(stop)
    if np.isnan(a) or np.isnan(st) or np.isnan(b) or st == 0 or (st > 0 and a > b) or (st < 0 and a < b):
        return np.zeros(0)
    if np.isinf(a) or np.isinf(b) or np.isinf(st):
        return np.array([a]) if a == b else np.zeros(0)
    ratio = (b - a) / st
    n = int(np.floor(ratio + 1e-10 * max(1.0, abs(ratio)))) + 1
    y = a + st * np.arange(n, dtype=float)
    y = np.minimum(y, b) if st > 0 else np.maximum(y, b)
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

# The Empirical1000 dataset (Fulcher, "1000 Empirical Time series", figshare,
# doi:10.6084/m9.figshare.5436136, version 11, CC BY 4.0): one time series per row of
# ``hctsa_timeseries-data.csv``. It is not shipped with the package; get_dataset('e1000')
# downloads it once into the user cache directory (see :func:`_data_cache_dir`).
_E1000 = {
    "url": "https://ndownloader.figshare.com/files/68104411",
    "filename": "e1000_hctsa_timeseries-data.csv",
    "size": 33059330,
    "sha256": "1dcd0184c080ffb2f90279304c6cd0cbee20ab9f04f65ad7956cae5f60485fec",
    "doi": "10.6084/m9.figshare.5436136",
}


def _data_cache_dir() -> str:
    """Directory for downloaded datasets.

    ``$PYHCTSA_DATA_DIR`` if set; otherwise ``%LOCALAPPDATA%\\pyhctsa`` on Windows and
    ``$XDG_CACHE_HOME/pyhctsa`` (default ``~/.cache/pyhctsa``) elsewhere.
    """
    env = os.environ.get("PYHCTSA_DATA_DIR")
    if env:
        return os.path.expanduser(env)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "pyhctsa")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "pyhctsa")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_verified(url: str, dest: str, size: int, sha256: str, timeout: float = 60.0) -> str:
    """Download ``url`` to ``dest`` (atomically) unless a verified copy is already there.

    The file is accepted only if its size and SHA-256 digest match. Raises ``ConnectionError``
    (with instructions for a manual download) if the download fails, and ``OSError`` if the
    downloaded file does not verify.
    """
    if os.path.exists(dest) and os.path.getsize(dest) == size and _sha256(dest) == sha256:
        return dest
    import urllib.request
    import urllib.error
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.part{os.getpid()}"
    logger.info(f"Downloading {url} to {dest} ...")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
    except (urllib.error.URLError, OSError, ValueError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ConnectionError(
            f"Could not download {url} ({e}). Check the network connection, or download the file "
            f"manually and save it as {dest}.") from e
    got_size, got_sha = os.path.getsize(tmp), _sha256(tmp)
    if got_size != size or got_sha != sha256:
        os.remove(tmp)
        raise OSError(f"Downloaded file from {url} failed verification "
                      f"(size {got_size}, expected {size}; sha256 {got_sha}, expected {sha256}).")
    os.replace(tmp, dest)
    return dest


def get_dataset(which: str = "e1000") -> list:
    """
    Load predefined datasets for testing and validation.

    Parameters
    ----------
    which : str, default="e1000"
        Dataset identifier.

        Options are:

        - ``"e1000"``: the Empirical1000 dataset of 1000 empirical time series (Fulcher,
          figshare, doi:10.6084/m9.figshare.5436136). It is not bundled with pyhctsa: the first
          call downloads it (33 MB) into the user cache directory (``~/.cache/pyhctsa``; set
          ``PYHCTSA_DATA_DIR`` to use another directory) and verifies its SHA-256 digest.
        - ``"sinusoid"``: Sinusoidal test data (bundled).
        - ``"noise"``: Gaussian noise data (T = 1000 sample length time series; bundled).

    Returns
    -------
    list
        List of time series data, where each element is a time series instance as a numpy array.

    Raises
    ------
    ConnectionError
        If ``"e1000"`` is not cached and cannot be downloaded.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = {
        "e1000": "empirical1000",
        "sinusoid": "sinusoid",
        "noise": "gaussian noise",
    }
    if which not in datasets:
        raise NotImplementedError(f"Dataset '{which}' not found. Available options: {list(datasets.keys())}")

    logger.info(f"Loading {datasets[which]} dataset...")
    if which == "e1000":
        path = _fetch_verified(_E1000["url"], os.path.join(_data_cache_dir(), _E1000["filename"]),
                               _E1000["size"], _E1000["sha256"])
        dataset = _load_csv(path)
    else:
        fname = {"sinusoid": "sinusoid.txt", "noise": "noise_gaussian.txt"}[which]
        data_path = os.path.join(utils_dir, "data", fname)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        dataset = [np.loadtxt(data_path)]
    logger.info(f"Loaded dataset of {len(dataset)} time series.")
    return dataset

def z_score(x: ArrayLike) -> np.ndarray:
    """
    Z-score the input data vector.

    This function standardizes the input array by removing the mean and scaling to unit variance
    (sample standard deviation, ``n - 1`` in the denominator). It performs the z-scoring operation
    twice to reduce numerical error. The check for a constant series is scale-free: there is no
    threshold on the variance or range, so data of any scale (however small its units) is
    standardized.

    Parameters
    ----------
    x : array-like
        Input data vector (1D array or list of numbers).

    Returns
    -------
    np.ndarray
        The z-scored version of the input data.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numeric array.
    ValueError
        If the input is empty, contains NaN or Inf, or is constant: all values equal, up to
        floating-point rounding at their own magnitude (a range of at most a few units in the
        last place of the largest absolute value), so that there is no variation to standardize.
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

    # A constant series (relative to the precision of its own values: scale-free)
    data_range = np.ptp(x)  # peak-to-peak (max - min)
    if data_range <= 4 * np.spacing(np.max(np.abs(x))):
        raise ValueError(f"Data range {data_range:.2e} is zero up to rounding. "
                         "Values appear to be constant.")

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
    autocovariance via ``np.correlate(..., 'full')``). Agrees with statsmodels
    to ~1e-15 (``np.dot`` (BLAS) rounds differently from numpy's full-array correlation).

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

def simple_binner(x_data: ArrayLike, num_bins: int) -> tuple:
    """
    Generate a histogram from equally spaced bins.
   
    Port of hctsa's ``BF_SimpleBinner``, Copyright (C) Ben D. Fulcher, relicensed by the author
    under GPL-3.0-or-later for pyhctsa.

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

def frame_signal(x: ArrayLike, n: int, p: int = 0) -> np.ndarray:
    """Split a signal into frames (columns) of length ``n``, optionally overlapping.

    Parameters
    ----------
    x : array-like
        Signal (treated as a vector), length L.
    n : int
        Frame length (>= 1).
    p : int, optional
        Overlap between successive frames, ``0 <= p < n``. Default 0. Frames start at ``x[0]``.

    Returns
    -------
    numpy.ndarray
        float64 array of shape ``(n, ncols)``; column j holds ``x[j*h : j*h + n]`` with
        ``h = n - p``. The last frame is zero-padded. ``ncols = max(1, ceil((L - p)/h))``
        (zero columns for an empty signal).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(n)
    p = int(p)
    if n < 1 or p < 0 or p >= n:
        raise ValueError("require n >= 1 and 0 <= p < n")
    L = x.size
    h = n - p
    if L == 0:
        return np.zeros((n, 0))
    z = x
    ncols = max(1, int(np.ceil((L - p) / h)))
    need = (ncols - 1) * h + n
    if z.size < need:
        z = np.concatenate([z, np.zeros(need - z.size)])
    win = np.lib.stride_tricks.sliding_window_view(z, n)[::h][:ncols]
    return win.T.copy()

# ------------------------------------------------------------------------------
# Adaptive time delays, Theiler windows and decimation (hctsa BF_GetTau,
# BF_TheilerWindow, BF_PreProcess)
# ------------------------------------------------------------------------------
_TAU_RULES = ('ac', 'ac1e', 'mi', 'mi-gaussian')
_TAU_CACHE: list = []  # (rule, N, digest, tau); small FIFO cache, like hctsa's persistent cache
_TAU_CACHE_SIZE = 4


def _round_half_away(x: float) -> float:
    """Round to the nearest integer, with ties rounded away from zero.

    This is "school" rounding, which hctsa uses to turn proportions of a length into integer
    counts and window sizes (ties are common, for example half of an odd length). Unlike
    ``round``/``numpy.round``, which round ties to the nearest even integer, a tie always moves
    away from zero.

    Parameters
    ----------
    x : float
        Value to round.

    Returns
    -------
    float
        The rounded value as a Python float. NaN and +/-Inf are returned unchanged.

    Notes
    -----
    The fractional part ``x - trunc(x)`` is exact in binary floating point, so a tie is detected
    without error: the value is truncated toward zero and moved one unit away from zero when the
    discarded fraction is at least one half. (The textbook ``sign(x)*floor(|x| + 0.5)`` is wrong
    for 0.49999999999999994 and for odd integers beyond 2**52, where the added 0.5 rounds.)
    """
    frac, whole = math.modf(float(x))
    if abs(frac) >= 0.5:
        whole += math.copysign(1.0, whole if whole != 0 else frac)
    return whole


def _round_half_away_array(x: ArrayLike) -> np.ndarray:
    """Vectorized :func:`_round_half_away`; returns a float64 array (0-d for scalar input)."""
    a = np.asarray(x, dtype=float)
    with np.errstate(invalid='ignore'):
        whole = np.trunc(a)
        up = np.abs(a - whole) >= 0.5  # NaN/Inf compare False and fall through unchanged
        out = np.where(up, whole + np.sign(a), whole)
    return np.asarray(out, dtype=float)


def _sample_std(x: ArrayLike, axis: Union[int, None] = None):
    """Sample standard deviation with the N-1 normalization; 0 for a single observation.

    This is ``numpy.std(ddof=1)`` with one pyhctsa convention: a single finite observation gives
    0 (not NaN), so that features built on a spread do not drop out for degenerate short
    inputs. An empty input gives NaN without a warning. NaN or Inf in the data propagate to NaN. For complex data the result is real
    (the absolute value is squared).

    Parameters
    ----------
    x : array_like
        Data.
    axis : int or None
        Axis to operate along; ``None`` flattens.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Standard deviation(s); the input shape with ``axis`` removed.

    Notes
    -----
    S = sqrt( sum |A_i - mean(A)|^2 / (N-1) ).
    """
    a = np.asarray(x)
    if a.dtype.kind not in 'fc':
        a = a.astype(float)
    if axis is None:
        a = a.reshape(-1)
        axis = 0
        scalar = True
    else:
        scalar = False
    n = a.shape[axis]
    out_shape = a.shape[:axis] + a.shape[axis + 1:]
    if n == 0:
        res = np.full(out_shape, np.nan)
    elif n == 1:
        # one observation: 0, unless that value is itself NaN/Inf
        res = np.where(np.isfinite(np.squeeze(a, axis=axis)), 0.0, np.nan)
    else:
        with np.errstate(all='ignore'):
            res = np.std(a, axis=axis, ddof=1)
    if scalar:
        return np.float64(res)
    return np.asarray(res, dtype=float)




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


_TAU_MI_JITTER_SEED = 0   # fixed seed of the always-applied 1e-10 jitter in _tau_mi (deterministic)
_TAU_MI_JITTER = 1e-10    # jitter standard deviation, relative to the standard deviation of the ranks
_TIE_TOL = 1e-5           # values whose sorted gaps are at most this (times std) form a cluster of near-identical values
_TIE_ISO = 10.0           # ... which is a tie group if the gaps on both sides exceed this many times _TIE_TOL


def _tie_aware_ranks(y: ArrayLike, tol: float = _TIE_TOL, iso: float = _TIE_ISO) -> np.ndarray:
    """Average ranks (1..N) in which repeated values, even if blurred by float noise, are ties.

    Plain average ranks only tie *exactly* equal values, so a tiny perturbation of a quantized
    series (which splits each repeated value into many distinct ones) would scramble the ranks
    within every tie group. Here a maximal run of sorted values whose consecutive gaps are all
    at most ``tol * std(y)`` is treated as one tied value if it stands alone, i.e. the gaps to
    the neighbouring values on both sides exceed ``iso * tol * std(y)``. Runs inside a dense
    region of a continuous sample (no such isolating gaps) are left alone, and exactly equal values
    are always ties. The ranks are therefore a function of the values, unchanged by a
    perturbation much smaller than ``tol * std(y)`` (and, for quantized data, by one smaller than
    about ``iso * tol * std(y)``), invariant under any strictly increasing transformation of the
    data, and equal to ``scipy.stats.rankdata(y)`` for data with no repeats or clusters.

    Parameters
    ----------
    y : array-like
        Finite one-dimensional data.
    tol : float, optional
        Cluster gap threshold, as a fraction of the sample standard deviation (default 1e-5).
    iso : float, optional
        Isolation factor (default 10).

    Returns
    -------
    numpy.ndarray
        Average ranks, same length as ``y``; tied values share the mean of their ranks.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        return np.empty(0)
    order = np.argsort(y, kind='stable')
    ys = y[order]
    gap = np.diff(ys)
    t = tol * (np.std(y, ddof=1) if n > 1 else 0.0)
    run_start = np.r_[True, gap > t]                  # runs of gaps <= t
    run_id = np.cumsum(run_start) - 1
    size = np.bincount(run_id)
    first = np.cumsum(np.r_[0, size[:-1]])
    last = first + size - 1
    gap_before = np.r_[np.inf, gap][first]
    gap_after = np.r_[gap, np.inf][last]
    isolated = (gap_before > iso * t) & (gap_after > iso * t)
    # one group per isolated run; elsewhere one group per exactly-equal value
    new_group = np.where(isolated[run_id], run_start, np.r_[True, gap > 0])
    gid = np.cumsum(new_group) - 1
    count = np.bincount(gid)
    start = np.cumsum(np.r_[0, count[:-1]])
    mean_rank = start + (count + 1) / 2.0             # mean of ranks start+1 .. start+count
    ranks = np.empty(n)
    ranks[order] = mean_rank[gid]
    return ranks


def _smooth3(a: np.ndarray) -> np.ndarray:
    """Centred 3-point moving average; the two end points average the two values available."""
    out = np.empty_like(a)
    if a.size == 1:
        out[:] = a
        return out
    out[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
    out[0] = (a[0] + a[1]) / 2.0
    out[-1] = (a[-2] + a[-1]) / 2.0
    return out


def _tau_mi(y: np.ndarray) -> float:
    """min(first minimum of the smoothed Kraskov AMI of tie-aware ranks, 'ac1e' delay), at least 1.

    See :func:`get_tau` (rule ``'mi'``) for the definition. Deterministic: the jitter that breaks
    exact ties among the (tie-aware) ranks is always applied, with normal draws from
    ``bf_random`` at a fixed seed (``_TAU_MI_JITTER_SEED``), as in hctsa's ``BF_GetTau``.
    """
    from .ksg import ksg_mi
    n = y.size
    tau_ac = _tau_ac1e(y)
    if tau_ac == 1:
        return 1  # can't go below 1
    if np.isnan(tau_ac):
        max_lag = n // 10
    else:
        # lags 1..ac1e+2: a (3-point smoothed) minimum at ac1e needs the AMI two lags beyond it
        max_lag = min(int(tau_ac) + 2, n // 10)
    if max_lag < 2:
        return tau_ac  # series too short to locate an AMI minimum

    # Kraskov (k=4, algorithm 1) AMI of the z-scored tie-aware ranks; the same fixed-seed
    # jitter (1e-10 sd) is always added to break the exact ties
    r = _tie_aware_ranks(y)
    sd = np.std(r, ddof=1)
    if not sd > 0:
        return np.nan  # constant series
    r = (r - r.mean()) / sd
    from pyhctsa.robust import bf_random  # (robust imports utils)
    r = r + _TAU_MI_JITTER * bf_random(n, _TAU_MI_JITTER_SEED, 'normal')
    amis = np.full(max_lag, np.nan)
    for lag in range(1, max_lag + 1):
        if lag > n - 5:  # as IN_AutoMutualInfo, no AMI for too few samples
            break
        amis[lag - 1] = ksg_mi(r[:-lag], r[lag:], 4, 1)
    valid = amis[~np.isnan(amis)]  # NaNs only trail
    if valid.size < 2:
        return tau_ac

    # first local minimum of the 3-point-smoothed AMI (at lag 1 if it already rises from lag 1 to 2)
    sm = _smooth3(valid)
    tau_min = np.nan
    for l in range(1, sm.size):  # 1-based lag l; sm[l-1] is the smoothed AMI at lag l
        if sm[l] > sm[l - 1] and (l == 1 or sm[l - 2] > sm[l - 1]):
            tau_min = l
            break

    if not np.isnan(tau_min):
        return min(tau_min, tau_ac) if not np.isnan(tau_ac) else tau_min
    if not np.isnan(tau_ac):
        return tau_ac
    # no 1/e crossing of the ACF and no AMI minimum: first drop of the AMI below the
    # Gaussian AMI at a correlation of 1/e
    mi_threshold = -0.5 * np.log(1.0 - np.exp(-2.0))
    below = np.flatnonzero(valid < mi_threshold)
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
        - ``'mi'``: ``max(1, min(tau_mi, ac1e))``, where ``tau_mi`` is the first minimum of the
          3-point-smoothed Kraskov AMI of tie-aware ranks. Precisely: the series is replaced by
          its ranks (average ranks, in which repeated values, even when blurred by float noise
          below about ``1e-5`` of the standard deviation, are ties; see ``_tie_aware_ranks``) and
          z-scored; fixed-seed Gaussian jitter of ``1e-10`` sd is always added (so the result is
          deterministic and does not depend on a random stream); the Kraskov (k=4, algorithm 1)
          AMI is computed at lags ``1..ac1e+2``; it is smoothed with a centred 3-point moving
          average; and ``tau_mi`` is the lag of the first local minimum of the smoothed curve
          (lag 1 if it already rises from lag 1 to 2). The ranks make the delay invariant under
          monotone transformations of the data and robust to tiny perturbations; the smoothing
          suppresses minima that are only estimator noise. If the ACF never falls to 1/e, the
          AMI is searched up to ``N // 10`` lags for its first local minimum, or else its first
          drop below ``-0.5 * log(1 - exp(-2))`` (the Gaussian AMI at a correlation of 1/e).
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
          rounded half away from zero. (A value of 1 or more is a number of samples.)
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


def _zscore_or_zeros(x: np.ndarray) -> np.ndarray:
    """z-score with the N-1 standard deviation; a constant input gives zeros.

    Parameters
    ----------
    x : numpy.ndarray
        Data (callers pass a 1-D series); all elements are standardized together.

    Returns
    -------
    numpy.ndarray
        float64 array of the same shape: ``(x - mean)/sd``. If ``sd`` is exactly 0 (a constant series,
        or a single value) it is replaced by 1, so the result is all zeros. A NaN ``sd`` (data containing
        NaN or Inf) propagates, giving all NaN. Empty input gives an empty array.

    Notes
    -----
    pyhctsa convention: the sample standard deviation (n-1 in the denominator) is used, and a
    standard deviation of 0 is replaced by 1, so that a constant series maps to zeros rather
    than NaN (features that standardize their input then stay defined).
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a.copy()
    sd = float(_sample_std(a))
    if sd == 0:
        sd = 1.0
    with np.errstate(invalid='ignore'):
        return (a - np.mean(a)) / sd


def decimate_ac1e(x_z: ArrayLike, rezscore: bool = True) -> Union[np.ndarray, float]:
    """
    ``zscore(BF_PreProcess(x_z, 'decimate_ac1e'))``: decimate by the 1/e autocorrelation time.

    Keeps every ``tau``-th sample, ``tau = get_tau(x_z, 'ac1e')``, then z-scores the result
    (N-1 standard deviation; a constant result becomes zeros). hctsa uses this for the
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
    return _zscore_or_zeros(y) if rezscore else y


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
    Sliding inner product (cross-correlation) of two signals for a symmetric range of lags.

    For lag :math:`k` the value is :math:`c_k = \\sum_n x_{n+k} y_n`, summed over the indices
    where both terms exist; the means are not removed. With ``normed`` the values are divided by
    :math:`\\sqrt{\\sum x^2 \\sum y^2}`, which makes a signal's zero-lag autocorrelation 1.

    Parameters
    ----------
    x, y : array-like
        Signals with the same number of samples (ValueError otherwise).
    normed : bool, optional
        Divide by the geometric mean of the signals' energies (default), or return the raw sums.
    max_lags : int or None, optional
        Largest lag, with ``1 <= max_lags <= len(x) - 1`` (ValueError otherwise); None means
        ``len(x) - 1``. Default is 10.

    Returns
    -------
    tuple
        ``(lags, c)``: the integer lags from ``-max_lags`` to ``max_lags`` and ``c`` at each.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if y.size != n:
        raise ValueError(f'x and y must have the same number of samples (got {n} and {y.size})')
    max_lags = n - 1 if max_lags is None else int(max_lags)
    if not 1 <= max_lags <= n - 1:
        raise ValueError(f'max_lags must be between 1 and len(x) - 1 = {n - 1}, or None (got {max_lags})')
    lags = np.arange(-max_lags, max_lags + 1)
    everything = correlate(x, y, mode='full', method='auto')  # lag 0 sits at position n - 1
    c = everything[lags + (n - 1)]
    if normed:
        c = c / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
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
