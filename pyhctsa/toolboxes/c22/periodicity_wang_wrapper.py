# Wrapper for the catch22 C implementation of PD_PeriodicityWang (see PD_PeriodicityWang.c).
# catch22: Copyright (C) 2018- Carl H. Lubba, Ben D. Fulcher and contributors, GPL-3.0; see
# LICENSE and NOTICE in this directory. Distributed with pyhctsa under GPL-3.0-or-later.
import numpy as np

try:
    from . import PD_PeriodicityWang as _pd_module
except ImportError:
    # Fallback for development/testing
    import PD_PeriodicityWang as _pd_module

# output names, in threshold order (the C extension returns th1..th7)
_TH_NAMES = ('th0', 'th0_01', 'th0_1', 'th0_2', 'th1_sqrtN', 'th5_sqrtN', 'th10_sqrtN')


def periodicity_wang(x):
    """
    Python wrapper for PD_PeriodicityWang C function that returns all threshold results.
    
    Parameters
    ----------
    x : array-like
        Input time series (1D array)
    
    Returns
    -------
    dict
        Dictionary containing results for all thresholds:
        - 'th0': threshold 0
        - 'th0_01': threshold 0.01
        - 'th0_1': threshold 0.1
        - 'th0_2': threshold 0.2
        - 'th1_sqrtN': threshold 1/sqrt(N)
        - 'th5_sqrtN': threshold 5/sqrt(N)
        - 'th10_sqrtN': threshold 10/sqrt(N)

        Each value is the first peak that meets the specified periodicity criteria,
        or 0 if none is found (no periodicity at that threshold).
    """
    # Ensure input is a contiguous numpy array
    x = np.ascontiguousarray(x, dtype=np.float64)
    
    # Validate input
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
    
    # Call the C extension function
    try:
        result = _pd_module.periodicity_wang_wrapper(x)
    except Exception as e:
        raise RuntimeError(f"Error in C function: {e}")
    return {name: result[f'th{i}'] for i, name in enumerate(_TH_NAMES, start=1)}
    