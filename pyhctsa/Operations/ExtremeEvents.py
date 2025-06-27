import numpy as np
from numpy.typing import ArrayLike
from typing import Dict

def MovingThreshold(y : ArrayLike, a : float = 1.0, b : float = 0.1) -> Dict[str, float]:
    """
    Moving threshold model for extreme events in a time series.

    Inspired by an idea contained in:
    "Reactions to extreme events: Moving threshold model"
    Altmann et al., Physica A 364, 435--444 (2006)

    This algorithm uses the occurrence of extreme events to modify a hypothetical 
    'barrier' that classifies new points as 'extreme' or not. The barrier begins 
    at sigma (standard deviation), and:
    - If |next_point| > barrier: barrier increases by proportion 'a'
    - Otherwise: barrier decreases by proportion 'b'

    Parameters
    ----------
    y : ArrayLike
        The input time series (should be z-scored)
    a : float, optional (default=1.0)
        The barrier jump parameter - how much to increase barrier after extreme event
    b : float, optional (default=0.1)
        The barrier decay proportion (0-1) - how much to decrease barrier otherwise

    Returns
    -------
    Dict[str, float]
        Dictionary containing barrier and kick statistics.
    """
    y = np.asarray(y)

    if b < 0 or b > 1:
        raise ValueError('The decay proportion, b, should be between 0 and 1')
    
    N = len(y)
    y = np.abs(y)  # extreme events defined in terms of absolute deviation from mean
    q = np.zeros(N) # the barrier
    kicks = np.zeros(N)

    # Treat the barrier as knowing nothing about the time series, until it encounters it
    # (except for the std! -- starts at 1)

    # Initial condition of barrier q:
    # The barrier will get smarter about the distribution but will decay to simulate 'forgetfulness' in the original model(!)

    q[0] = 1  # begin at sigma

    for i in range(1, N):
        if y[i] > q[i-1]:  # Extreme event -- time series value more extreme than the barrier
            q[i] = (1 + a) * y[i]  # increase barrier above the new observation by a factor a
            kicks[i] = q[i] - q[i-1]  # The size of the increase
        else:
            q[i] = (1 - b) * q[i-1]  # Decrease barrier by proportion b

    # Basic statistics on the barrier dynamics, q
    out = {
        'meanq': np.mean(q),
        'medianq': np.median(q),
        'iqrq': np.percentile(q, 75, method='hazen') - np.percentile(q, 25, method='hazen'),
        'maxq': np.max(q),
        'minq': np.min(q),
        'stdq': np.std(q, ddof=1),
        'meanqover': np.mean(q - y),
        'pkick': np.sum(kicks) / (N - 1),  # probability of a kick
    }

    # Kicks (when the barrier is changed due to extreme event)
    fkicks = np.argwhere(kicks > 0).flatten()
    Ikicks = np.diff(fkicks)
    out['stdkicks'] = np.std(Ikicks, ddof=1)
    out['meankickf'] = np.mean(Ikicks)
    out['mediankicksf'] = np.median(Ikicks)

    return out
