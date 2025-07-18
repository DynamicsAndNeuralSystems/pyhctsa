import numpy as np
from statsmodels.tsa.tsatools import detrend
from pyhctsa.Operations.Stationarity import StatAv, SlidingWindow
from scipy.signal import lfilter
from numpy.typing import ArrayLike
from scipy.signal import resample_poly
from pyhctsa.Utilities.utils import ZScore
from pyhctsa.Operations.Distribution import OutlierTest

def _medfilt1d(x, k):
    """Apply a length-k median filter to a 1D array x.
    Taken from https://gist.github.com/bhawkins/3535131.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median(y, axis=1)

def _safe_divide(num, denom):
    """Return num/denom, or np.nan if denom is zero."""
    return num / denom if denom != 0 else np.nan

def PreProcCompare(y : ArrayLike, detrendMeth : str = 'medianf') -> dict:
    y = np.asarray(y)
    N = len(y)
    r = np.arange(N)
    # Apply preproc....
    # 1) Polynomial detrend
    y_d = None
    if 'poly' in detrendMeth:
        # extract the order
        order = detrendMeth.strip("poly")
        if not order:
            raise ValueError(f"Could not detect an order for polynomial: {detrendMeth}. Choose poly<o> where o is an integer between 1 and 9, e.g., poly1.")
        try:
            order = int(order)
        except ValueError as e:
            print(f"Could not convert order: `{order}' to integer.")
        y_d = detrend(y, order=order, axis=0)

    # 2) Differencing
    elif 'diff' in detrendMeth:
        ndiff = detrendMeth.strip("diff")
        if not ndiff:
            raise ValueError(f"Could not detect num diffs for diff: {detrendMeth}. Choose diff<n> where n is an integer > 0, e.g., diff1.")
        try:
            ndiff = int(ndiff)
        except ValueError as e:
            print(f"Could not convert ndiff: `{ndiff}' to integer.")
        y_d = np.diff(y, n=ndiff, axis=0)
    
    # 3) Median filter
    elif 'medianf' in detrendMeth:
        med_ord = detrendMeth.strip("medianf")
        if not med_ord:
            raise ValueError(f"Could not detect median filter order for median filter: {detrendMeth}. Choose medianf<n> where n is an integer >= 3, e.g., medianf3.")
        try:
            med_ord = int(med_ord)
        except ValueError as e:
            print(f"Could not convert median order: `{med_ord}' to integer.")
        y_d = _medfilt1d(y, med_ord)
    
    # 4) Running average
    elif 'rav' in detrendMeth:
        rav_wsize = detrendMeth.strip("rav")
        if not rav_wsize:
            raise ValueError(f"Could not detect running average window size for wsize: {detrendMeth}. Choose rav<n> where n is an integer > 1, e.g., rav4.")
        try:
            rav_wsize = int(rav_wsize)
        except ValueError as e:
            print(f"Could not running average window size: `{rav_wsize}' to integer.")
        y_d = lfilter(np.ones(rav_wsize)/rav_wsize, [1], y)
    
    elif 'resample' in detrendMeth:
        rs_params = detrendMeth.strip("resample_")
        if not rs_params:
            raise ValueError(f"Could not detect resample parameters P_Q: {detrendMeth}. Choose resample_<P>_<Q> where P is the upsampling factor and Q is the downsampling factor.")
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
    y = ZScore(y)
    y_d = ZScore(y_d)

    out = {}

    # 1) Stationarity
    for seg in [2, 4, 6, 8, 10]:
        num = StatAv(y_d, 'seg', seg)
        denom = StatAv(y, 'seg', seg)
        out[f'statv{seg}'] = _safe_divide(num, denom)

    # Sliding window mean 
    for win, step in [(2,2), (5,1), (5,2), (10,1), (10,2)]:
        num = SlidingWindow(y_d, 'mean', 'std', win, step)
        denom = SlidingWindow(y, 'mean', 'std', win, step)
        out[f'swms{win}_{step}'] = _safe_divide(num, denom)

    # Sliding window std 
    for win, step in [(2,1), (2,2), (5,1), (5,2), (10,1), (10,2)]:
        num = SlidingWindow(y_d, 'std', 'std', win, step)
        denom = SlidingWindow(y, 'std', 'std', win, step)
        out[f'swss{win}_{step}'] = _safe_divide(num, denom)


    #TODO:Gaussianity

    # TODO: Compare distribution to fitted normal distribution

    # 3) Outliers
    for thresh, method in [(2, 'mean'), (5, 'mean'), (2, 'std'), (5, 'std')]:
        num = OutlierTest(y_d, thresh, method)
        denom = OutlierTest(y, thresh, method)
        key = f'olbt_{"m" if method=="mean" else "s"}{thresh}'
        out[key] = _safe_divide(num, denom)

    return out
