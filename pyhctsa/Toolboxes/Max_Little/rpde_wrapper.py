import numpy as np
from . import close_returns as _close_returns_c

def close_returns_analysis(x, embed_dims, embed_delay, eta):
    """
    Python wrapper for close returns analysis.
    
    Parameters
    ----------
    x : array-like
        Input time series (1D array).
    embed_dims : int
        Embedding dimension (must be >= 1).
    embed_delay : int
        Embedding delay (must be >= 1).
    eta : float
        Close return distance threshold (must be > 0).
    
    Returns
    -------
    close_returns : np.ndarray
        Close return time histogram (length = embed_elements)
        where embed_elements = len(x) - (embed_dims - 1) * embed_delay
    """
    x = np.asarray(x, dtype=np.float64)
    return _close_returns_c.close_returns(
        x=x,
        embed_dims=embed_dims,
        embed_delay=embed_delay,
        eta=eta
    )
