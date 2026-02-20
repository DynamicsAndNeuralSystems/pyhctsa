import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter, resample_poly
from statsmodels.tsa.tsatools import detrend

from ..operations.distribution import outlier_test
from ..operations.stationarity import sliding_window, stat_av
from ..utils import z_score

def _med_filt_1d(x: ArrayLike, k: int) -> ArrayLike:
    """Apply a length-k median filter to a 1D array x.
    
    For odd k, y(i) is the median of x[i-(k-1)//2 : i+(k-1)//2+1]
    For even k, y(i) is the median of x[i-k//2 : i+k//2]
    
    Based on: https://gist.github.com/bhawkins/3535131.
    """
    assert k > 0, "Median filter length must be positive."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    if k % 2 == 1:
        # Odd case: symmetric window
        k2 = (k - 1) // 2
        left_pad = k2
        right_pad = k2
    else:
        # Even case: asymmetric window (more on the left)
        left_pad = k // 2
        right_pad = k // 2 - 1
    
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, left_pad] = x
    
    # Fill left side of the window
    for i in range(left_pad):
        j = left_pad - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
    
    # Fill right side of the window
    for i in range(right_pad):
        j = i + 1
        y[:-j, left_pad + j] = x[j:]
        y[-j:, left_pad + j] = x[-1]
    
    return np.median(y, axis=1)

def _safe_divide(num, denom):
    """Return num/denom, or np.nan if denom is zero."""
    return num / denom if denom != 0 else np.nan

def preproc_compare(y : ArrayLike, detrend_meth : str = 'medianf') -> dict:
    """
    Compare time-series properties before and after pre-processing.

    Applies a specified pre-processing transformation to the input time series
    and evaluates how selected statistical properties change relative to the
    original series.

    Parameters
    ----------
    y : array-like
        Input time series.

    detrend_meth : str, optional
        Pre-processing method to apply.

        Polynomial detrending:

        - ``"poly<n>"``: Polynomial detrending of order ``n`` (``1 ≤ n ≤ 9``),
        e.g., ``"poly1"`` for linear detrending.

        Differencing:

        - ``"diff<n>"``: Apply ``n`` successive differences,
        e.g., ``"diff1"``.

        Median filtering:

        - ``"medianf<n>"``: Median filter with window length ``n`` (must be odd),
        e.g., ``"medianf3"``.

        Running average:

        - ``"rav<n>"``: Moving average with window size ``n``,
        e.g., ``"rav4"``.

        Resampling:

        - ``"resample_<P>_<Q>"``: Resample by a factor of ``P/Q``,
        where ``P`` is the upsampling factor and ``Q`` is the
        downsampling factor (e.g., ``"resample_1_2"`` for half rate).

        Default is ``"medianf"``.

    Returns
    -------
    dict
        Comparison of stationarity and distributional measures between the
        original and transformed time series.
    """
    y = np.asarray(y)
    N = len(y)
    r = np.arange(N)
    # Apply preproc....
    # 1) Polynomial detrend
    y_d = None
    if 'poly' in detrend_meth:
        # extract the order
        order = detrend_meth.strip("poly")
        if not order:
            raise ValueError(f"Could not detect an order for polynomial: {detrend_meth}."
                             f"Choose poly<o> where o is an integer between 1 and 9, e.g., poly1.")
        try:
            order = int(order)
        except ValueError:
            print(f"Could not convert order: `{order}' to integer.")
        y_d = detrend(y, order=order, axis=0)

    # 2) Differencing
    elif 'diff' in detrend_meth:
        ndiff = detrend_meth.strip("diff")
        if not ndiff:
            raise ValueError(f"Could not detect num diffs for diff: {detrend_meth}."
                             "Choose diff<n> where n is an integer > 0, e.g., diff1.")
        try:
            ndiff = int(ndiff)
        except ValueError:
            print(f"Could not convert ndiff: `{ndiff}' to integer.")
        y_d = np.diff(y, n=ndiff, axis=0)
    
    # 3) Median filter
    elif 'medianf' in detrend_meth:
        med_ord = detrend_meth.strip("medianf")
        if not med_ord:
            raise ValueError(f"Could not detect median filter order for median filter: {detrend_meth}. "
                             "Choose medianf<n> where n is an integer >= 3, e.g., medianf3.")
        try:
            med_ord = int(med_ord)
        except ValueError:
            print(f"Could not convert median order: `{med_ord}' to integer.")
        y_d = _med_filt_1d(y, med_ord)
    
    # 4) Running average
    elif 'rav' in detrend_meth:
        rav_wsize = detrend_meth.strip("rav")
        if not rav_wsize:
            raise ValueError(f"Could not detect running average window size for wsize: {detrend_meth}."
                             "Choose rav<n> where n is an integer > 1, e.g., rav4.")
        try:
            rav_wsize = int(rav_wsize)
        except ValueError:
            print(f"Could not running average window size: `{rav_wsize}' to integer.")
        y_d = lfilter(np.ones(rav_wsize)/rav_wsize, [1], y)
    
    elif 'resample' in detrend_meth:
        rs_params = detrend_meth.strip("resample_")
        if not rs_params:
            raise ValueError(f"Could not detect resample parameters P_Q: {detrend_meth}."
                             "Choose resample_<P>_<Q> where P is the upsampling factor and Q "
                             "is the downsampling factor.")
        P, Q = rs_params.split("_")
        try:
            P = int(P)
            Q = int(Q)
        except ValueError as e:
            raise(e)
        y_d = resample_poly(y, P, Q)

    # Check that the outputs are meaningful...
    if np.all(y_d == 0):
        out = np.nan
        return out
    
    # Statistical tests on original and processed time series
    # zscore both
    y = z_score(y)
    y_d = z_score(y_d)

    out = {}

    # 1) Stationarity
    for seg in [2, 4, 6, 8, 10]:
        num = stat_av(y_d, 'seg', seg)
        denom = stat_av(y, 'seg', seg)
        out[f'statv{seg}'] = _safe_divide(num, denom)

    # Sliding window mean 
    for win, step in [(2,2), (5,1), (5,2), (10,1), (10,2)]:
        num = sliding_window(y_d, 'mean', 'std', win, step)
        denom = sliding_window(y, 'mean', 'std', win, step)
        out[f'swms{win}_{step}'] = _safe_divide(num, denom)

    # Sliding window std 
    for win, step in [(2,1), (2,2), (5,1), (5,2), (10,1), (10,2)]:
        num = sliding_window(y_d, 'std', 'std', win, step)
        denom = sliding_window(y, 'std', 'std', win, step)
        out[f'swss{win}_{step}'] = _safe_divide(num, denom)

    # 3) Outliers
    for thresh, method in [(2, 'mean'), (5, 'mean'), (2, 'std'), (5, 'std')]:
        num = outlier_test(y_d, thresh, method)
        denom = outlier_test(y, thresh, method)
        key = f'olbt_{"m" if method=="mean" else "s"}{thresh}'
        out[key] = _safe_divide(num, denom)

    return out
