import numpy as np
from numpy.typing import ArrayLike
from ..Toolboxes.Max_Little import fastdfa
from ..Utilities.utils import make_buffer

def FastDFA(y: ArrayLike) -> float:
    """
    Measures the scaling exponent of the time series using a fast implementation
    of detrended fluctuation analysis (DFA).

    This is a Python wrapper for Max Little's ML_fastdfa code.
    The original fastdfa code is by Max A. Little and publicly available at:
    http://www.maxlittle.net/software/index.php

    Parameters
    ----------
    y : array-like
        Input time series (1D array), fed straight into the fastdfa script.

    Returns
    -------
    float
        Estimated scaling exponent from log-log linear fit of fluctuation vs interval.
    """
    y = np.asarray(y)
    intervals, flucts = fastdfa.fastdfa(y)
    idx = np.argsort(intervals)
    intervals_sorted = intervals[idx]
    flucts_sorted = flucts[idx]

    # Log-log linear fit
    coeffs = np.polyfit(np.log10(intervals_sorted), np.log10(flucts_sorted), 1)
    alpha = coeffs[0]
    return alpha
