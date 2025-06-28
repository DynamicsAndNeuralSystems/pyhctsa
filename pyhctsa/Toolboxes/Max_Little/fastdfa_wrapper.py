import os
import sys
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Determine the correct library extension for the platform
if sys.platform.startswith("darwin"):
    lib_ext = ".dylib"
elif sys.platform.startswith("win"):
    lib_ext = ".dll"
else:
    lib_ext = ".so"

_lib_path = os.path.join(os.path.dirname(__file__), f"ML_fastdfa_core{lib_ext}")
_lib = ctypes.CDLL(_lib_path)

_lib.fastdfa_core.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # x (input array)
    ctypes.c_ulong,                                         # elements (length of x)
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),         # intervals (pointer-to-pointer)
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # flucts (output)
    ctypes.POINTER(ctypes.c_ulong)                          # N_scales (in/out)
]
_lib.fastdfa_core.restype = None

def fastdfa(x):
    """
    Python wrapper for the fastdfa_core C function.
    Always lets the C code compute and allocate the intervals array.

    Parameters
    ----------
    x : array-like
        Input time series (1D array).

    Returns
    -------
    intervals : np.ndarray
        Interval widths at each scale.
    flucts : np.ndarray
        Fluctuation amplitudes at each scale.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    elements = x.shape[0]
    max_scales = int(np.log2(elements))
    flucts = np.zeros(max_scales, dtype=np.float64)
    N_scales_c = ctypes.c_ulong(0)

    # Always pass a pointer to a NULL pointer for intervals
    intervals_pp = ctypes.POINTER(ctypes.c_ulong)()

    _lib.fastdfa_core(
        x,
        elements,
        ctypes.byref(intervals_pp),
        flucts,
        ctypes.byref(N_scales_c)
    )

    size = N_scales_c.value

    # Check for NULL pointer
    if not bool(intervals_pp):
        raise ValueError("C function did not allocate intervals (possibly due to invalid input or too short signal).")

    # Convert the returned C pointer to a numpy array
    intervals_arr = np.ctypeslib.as_array(intervals_pp, shape=(size,))
    intervals_arr = np.array(intervals_arr, copy=True)  # Copy to Python memory

    flucts = flucts[:size]
    return intervals_arr, flucts
