import numpy as np
from typing import Union
from scipy import signal
from ..Utilities.utils import binpicker, histc
from numpy.typing import ArrayLike


def RawHRVMeas(x: ArrayLike) -> dict:
    """
    Compute Poincaré plot-based HRV (Heart Rate Variability) measures from RR interval time series.

    This function computes the triangular histogram indices and Poincaré plot measures commonly used 
    in HRV analysis. It is specifically designed for time series consisting of consecutive RR intervals 
    measured in milliseconds. It is not suitable for other types of time series.

    The computed features are widely used in clinical and physiological studies of autonomic nervous 
    system activity. The Poincaré plot measures (SD1 and SD2) are standard metrics for short- and 
    long-term variability, while the triangular indices provide geometric summaries of the RR 
    distribution.

    References
    ----------
    - M. Brennan, M. Palaniswami, and P. Kamen, 
    "Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability?", 
    IEEE Transactions on Biomedical Engineering, 48(11), pp. 1342–1347, 2001.
    - Original MATLAB implementation adapted from: Max Little's `hrv_classic.m`
    (http://www.maxlittle.net/)

    Parameters
    ----------
    x : array_like
        Time series of RR intervals in milliseconds.

    Returns
    -------
    out : dict
        Dictionary containing the following HRV features   
        - 'tri10'   : Triangular histogram index using 10 bins.
        - 'tri20'   : Triangular histogram index using 20 bins.
        - 'trisqrt' : Triangular histogram index using a number of bins determined by the square root rule.
        - 'SD1'     : Standard deviation of the Poincaré plot’s minor axis (short-term variability).
        - 'SD2'     : Standard deviation of the Poincaré plot’s major axis (long-term variability).
    """

    x = np.asarray(x)
    N = len(x)
    
    out = {}

    # triangular histogram index  
    # 10 bins  
    edges10 = binpicker(x.min(), x.max(), 10)
    hist_counts10 = histc(x, edges10)
    out['tri10'] = N/np.max(hist_counts10)

    # 20 bins
    edges20 = binpicker(x.min(), x.max(), 20)
    hist_counts20 = histc(x, edges20)
    out['tri20'] = N/np.max(hist_counts20)

    # (sqrt samples) bins
    edges_sqrt = binpicker(x.min(), x.max(), int(np.ceil(np.sqrt(N))))
    hist_counts_sqrt = histc(x, edges_sqrt)
    out['trisqrt'] = N/np.max(hist_counts_sqrt)

    # Poincare plot measures
    diffx = np.diff(x)
    out['SD1'] = 1/np.sqrt(2) * np.std(diffx, ddof=1) * 1000
    out['SD2'] = np.sqrt(2 * np.var(x, ddof=1) - (1/2) * np.std(diffx, ddof=1)**2) * 1000

    return out
