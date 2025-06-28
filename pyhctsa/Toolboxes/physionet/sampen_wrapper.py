import numpy as np
from . import sampen as _sampen_c

def calculate_sampen(data: np.ndarray, M: int, r: float) -> np.ndarray:
    """Low-level wrapper for C sampen function."""
    y = np.asarray(data, dtype=np.float64)
    result = _sampen_c.calculate(y, M, r)
    return result
    