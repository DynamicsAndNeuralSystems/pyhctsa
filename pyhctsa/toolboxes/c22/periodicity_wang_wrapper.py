import numpy as np

try:
    from . import PD_PeriodicityWang as _pd_module
except ImportError:
    # Fallback for development/testing
    import PD_PeriodicityWang as _pd_module

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
        - 'th1': threshold 0.0
        - 'th2': threshold 0.01
        - 'th3': threshold 0.1
        - 'th4': threshold 0.2
        - 'th5': threshold 1/sqrt(N)
        - 'th6': threshold 5/sqrt(N)
        - 'th7': threshold 10/sqrt(N)
        
        Each value is the first peak that meets the specified periodicity criteria,
        or 1 if none found (matching MATLAB behavior).
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
        return result
    except Exception as e:
        raise RuntimeError(f"Error in C function: {e}")
    