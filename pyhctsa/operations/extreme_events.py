import numpy as np
from numpy.typing import ArrayLike

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

def _barrier_loop_py(y: np.ndarray, a: float, b: float):
    """Pure-Python fallback, used when numba is unavailable. Iterating a Python
    list of floats avoids the per-element numpy scalar boxing that dominates the
    naive loop; kicks are collected sparsely and scattered into a zero array so
    that np.sum() sees the exact same summation order as the original."""
    N = y.shape[0]
    a1 = 1.0 + a
    b1 = 1.0 - b
    q = [0.0] * N
    q[0] = 1.0
    prev = 1.0
    kick_i, kick_v = [], []
    push_i, push_v = kick_i.append, kick_v.append
    yl = y.tolist()
    for i in range(1, N):
        yi = yl[i]
        if yi > prev:
            cur = a1 * yi
            push_i(i)
            push_v(cur - prev)
        else:
            cur = b1 * prev
        q[i] = cur
        prev = cur
    kicks = np.zeros(N, dtype=np.float64)
    if kick_i:
        kicks[kick_i] = kick_v
    return np.asarray(q), kicks


def _barrier_loop_impl(y, a, b):
    N = y.shape[0]
    a1 = 1.0 + a
    b1 = 1.0 - b
    q = np.empty(N, dtype=np.float64)
    kicks = np.zeros(N, dtype=np.float64)
    q[0] = 1.0
    prev = 1.0
    for i in range(1, N):
        yi = y[i]
        if yi > prev:
            cur = a1 * yi
            kicks[i] = cur - prev
        else:
            cur = b1 * prev
        q[i] = cur
        prev = cur
    return q, kicks


if _HAVE_NUMBA:
    _barrier_loop_nb = njit(cache=True)(_barrier_loop_impl)


def _hazen(sq: np.ndarray, p: float) -> float:
    """Hazen (alpha=beta=0.5) quantile from an already-sorted array."""
    n = sq.size
    idx = p * n - 0.5
    if idx <= 0.0:
        return float(sq[0])
    if idx >= n - 1:
        return float(sq[-1])
    lo = int(idx)
    return float(sq[lo] + (idx - lo) * (sq[lo + 1] - sq[lo]))


def moving_threshold(y: ArrayLike, a: float = 1.0, b: float = 0.1) -> dict:
    """
    Moving threshold model for extreme events in a time series.

    Inspired by an idea contained in Altmann et al. (2006) [1].

    This algorithm uses the occurrence of extreme events to modify a hypothetical
    'barrier' that classifies new points as 'extreme' or not. The barrier begins
    at sigma (standard deviation), and if the absolute value of the next data point
    is greater than the barrier, the barrier is increased by a proportion 'a',
    otherwise the position of the barrier is decreased by a proportion 'b'.

    References
    ----------
    .. [1] "Reactions to extreme events: Moving threshold model"
        Altmann et al., Physica A 364, 435--444 (2006)

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored).
    a : float, optional
        The barrier jump parameter - how much to increase barrier after extreme event. Default is 1.0.
    b : float, optional
        The barrier decay proportion (0-1) - how much to decrease barrier otherwise. Default is 0.1.

    Returns
    -------
    dict
        Dictionary containing barrier and kick statistics.
    """
    if b < 0 or b > 1:
        raise ValueError('The decay proportion, b, should be between 0 and 1')

    y = np.asarray(y, dtype=np.float64)
    N = y.shape[0]
    y = np.abs(y)  # extreme events defined in terms of absolute deviation from mean

    # Treat the barrier as knowing nothing about the time series, until it
    # encounters it (except for the std! -- starts at 1). The barrier gets
    # smarter about the distribution but decays to simulate 'forgetfulness'.
    if _HAVE_NUMBA:
        q, kicks = _barrier_loop_nb(np.ascontiguousarray(y), float(a), float(b))
    else:
        q, kicks = _barrier_loop_py(y, float(a), float(b))

    # Basic statistics on the barrier dynamics, q.
    # One sort gives median/IQR/min/max, replacing three partitions + two scans.
    sq = np.sort(q)
    n = sq.size
    mid = n >> 1
    medianq = float(sq[mid]) if n & 1 else 0.5 * (sq[mid - 1] + sq[mid])

    out = {
        'meanq': np.mean(q),
        'medianq': medianq,
        'iqrq': _hazen(sq, 0.75) - _hazen(sq, 0.25),
        'maxq': sq[-1],
        'minq': sq[0],
        'stdq': np.std(q, ddof=1),
        'meanqover': np.mean(q - y),
        'pkick': np.sum(kicks) / (N - 1),  # probability of a kick
    }

    # Kicks (when the barrier is changed due to extreme event)
    f_kicks = np.flatnonzero(kicks > 0)
    i_kicks = np.diff(f_kicks)  # time intervals between successive kicks
    if i_kicks.size > 0:
        out.update({
            'stdkickf': np.std(i_kicks, ddof=1) if i_kicks.size > 1 else np.nan,
            'meankickf': np.mean(i_kicks),
            'mediankickf': np.median(i_kicks),
        })
    else:
        out.update({'stdkickf': np.nan, 'meankickf': np.nan, 'mediankickf': np.nan})
    return out