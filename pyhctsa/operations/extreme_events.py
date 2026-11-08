import numpy as np
from ..utils import _iqr, _quantile
from numpy.typing import ArrayLike

from ._symbolic_complexity import _normalized_lz_complexity
from ..utils import _sample_std, dict_output

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


def moving_threshold(y: ArrayLike, a: float = 1.0, b: float = 0.1) -> dict:
    """
    Moving threshold model for extreme events in a time series.

    Inspired by an idea contained in Altmann et al. (2006) [1].

    This algorithm uses the occurrence of extreme events to modify a hypothetical
    'barrier' that classifies new points as 'extreme' or not. The barrier begins
    at sigma (standard deviation), and if the absolute value of the next data point
    is greater than the barrier, the barrier is moved to (1+a) times that absolute
    value (above the new observation), otherwise the position of the barrier is
    decreased by a proportion 'b'.

    References
    ----------
    .. [1] "Reactions to extreme events: Moving threshold model"
        Altmann et al., Physica A 364, 435--444 (2006)

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored).
    a : float, optional
        The barrier jump parameter: after an extreme event the barrier is set to (1+a) times
        the absolute value of the data point. Default is 1.0.
    b : float, optional
        The barrier decay proportion (0-1) - how much to decrease barrier otherwise. Default is 0.1.

    Returns
    -------
    dict
        Dictionary containing barrier and kick statistics, including `pkick` (the probability of a
        kick, number of kicks / (N-1)) and `meankicksize` (the mean size of the barrier jump when a
        kick occurs; NaN if there are none).
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
        'iqrq': _iqr(sq),
        'maxq': sq[-1],
        'minq': sq[0],
        'stdq': _sample_std(q),
        'meanqover': np.mean(q - y),
        'pkick': np.count_nonzero(kicks) / (N - 1),  # probability of a kick: number of kicks / (N-1)
    }

    # Kicks (when the barrier is changed due to extreme event)
    f_kicks = np.flatnonzero(kicks)  # indices of kicks (steps where an extreme event increased the barrier)
    out['meankicksize'] = np.mean(kicks[f_kicks]) if f_kicks.size else np.nan  # mean size of the barrier jump
    i_kicks = np.diff(f_kicks)  # time intervals between successive kicks
    if i_kicks.size > 0:
        out.update({
            'stdkickf': _sample_std(i_kicks),
            'meankickf': np.mean(i_kicks),
            'mediankickf': np.median(i_kicks),
        })
    else:
        out.update({'stdkickf': np.nan, 'meankickf': np.nan, 'mediankickf': np.nan})
    return out

def _interval_stats(idx: np.ndarray) -> tuple:
    """Mean and coefficient of variation of the gaps between consecutive (sorted) indices."""
    if idx.size >= 2:
        intervals = np.diff(idx).astype(np.float64)
        mean_i = np.mean(intervals)
        if mean_i > 0:
            sd = _sample_std(intervals)
            return mean_i, sd / mean_i
        return mean_i, np.nan
    return np.nan, np.nan


def extreme_event_order(y: ArrayLike, extreme_thresh: float = 0.05) -> dict:
    """
    Temporal patterning of positive vs. negative extreme events.

    Labels each point in the top/bottom `extreme_thresh` fraction of the distribution as a
    positive- or negative-direction 'extreme event' (raw threshold exceedances, not
    declustered peaks), then asks not just how often each type occurs but how the two types
    are *ordered* in time relative to each other: does a positive extreme tend to be followed
    by another positive one (clustering by sign), or does the sequence alternate?

    Parameters
    ----------
    y : array-like
        The input time series (assumed z-scored).
    extreme_thresh : float, optional
        The proportion of points (in each direction) to count as 'extreme'; must be in (0, 0.5)
        so that the two tails cannot overlap. Default is 0.05.

    Returns
    -------
    dict
        Dictionary containing:

        - `propPosEvents`: proportion of extreme events that are positive-direction.
        - `meanInterval`, `cvInterval`: mean (in samples) and coefficient of variation of the
          inter-event intervals of the combined (both-direction) event sequence.
        - `meanIntervalPos`, `cvIntervalPos`, `meanIntervalNeg`, `cvIntervalNeg`: the same,
          computed within each direction's own event sub-sequence.
        - `alternationRate`: proportion of consecutive event pairs whose direction differs.
        - `propPN`, `propNP`: proportion of consecutive event pairs that switch
          positive-to-negative and negative-to-positive (they sum to `alternationRate`).
        - `meanIntervalPN`, `cvIntervalPN`, `meanIntervalNP`, `cvIntervalNP`: mean and
          coefficient of variation of the gaps between successive PN (NP) switches,
          timestamped at the later (post-switch) event.
        - `lzComplexity`: normalized Lempel-Ziv complexity (Michael Small's MS_complexitybs) of
          the 0/1 event-direction sequence. NaN with fewer than 10 events, or if all events
          have the same direction.

        Outputs that need at least two events (the intervals and alternation measures) are NaN
        otherwise.
    """
    if extreme_thresh is None:
        extreme_thresh = 0.05
    if extreme_thresh <= 0 or extreme_thresh >= 0.5:
        raise ValueError('extreme_thresh must be in (0,0.5) so the two tails cannot overlap.')

    y = np.asarray(y, dtype=np.float64).ravel()
    yq = y[~np.isnan(y)]
    upper_thresh, lower_thresh = _quantile(yq, [1 - extreme_thresh, extreme_thresh])

    pos_idx = np.flatnonzero(y > upper_thresh) + 1  # (only differences matter)
    neg_idx = np.flatnonzero(y < lower_thresh) + 1

    event_idx = np.concatenate([pos_idx, neg_idx])
    event_type = np.concatenate([np.ones(pos_idx.size), np.zeros(neg_idx.size)])
    order = np.argsort(event_idx, kind='stable')
    event_idx = event_idx[order]
    event_type = event_type[order]
    n_events = event_idx.size

    out = {'propPosEvents': np.mean(event_type) if n_events else np.nan}

    out['meanInterval'], out['cvInterval'] = _interval_stats(event_idx)
    out['meanIntervalPos'], out['cvIntervalPos'] = _interval_stats(pos_idx)
    out['meanIntervalNeg'], out['cvIntervalNeg'] = _interval_stats(neg_idx)

    if n_events >= 2:
        type_change = np.diff(event_type)  # +1 = N-to-P switch, -1 = P-to-N switch
        out['alternationRate'] = np.mean(type_change != 0)
        out['propPN'] = np.mean(type_change == -1)
        out['propNP'] = np.mean(type_change == 1)
        pn_switch_idx = event_idx[np.flatnonzero(type_change == -1) + 1]  # timestamp at post-switch event
        np_switch_idx = event_idx[np.flatnonzero(type_change == 1) + 1]
    else:
        out['alternationRate'] = out['propPN'] = out['propNP'] = np.nan
        pn_switch_idx = np_switch_idx = np.array([], dtype=int)
    out['meanIntervalPN'], out['cvIntervalPN'] = _interval_stats(pn_switch_idx)
    out['meanIntervalNP'], out['cvIntervalNP'] = _interval_stats(np_switch_idx)

    # Normalized LZ complexity of the direction sequence (unstable below ~10 symbols; undefined
    # when every event has the same direction, as then log(#symbols) = 0)
    if n_events >= 10 and 0 < out['propPosEvents'] < 1:
        out['lzComplexity'] = _normalized_lz_complexity(event_type)
    else:
        out['lzComplexity'] = np.nan
    return out


@dict_output
def extremogram(y: ArrayLike, quantile: float = 0.95, max_lag: int = 100) -> dict:
    """
    Serial dependence of extremes: the largest value of the extremogram over a range of lags.

    The extremogram of Davis and Mikosch [1]_ is the analogue of the autocorrelation function for
    extreme events: for the event that :math:`|x_t|` exceeds a high threshold :math:`u`, it is
    the conditional probability that the event recurs :math:`k` samples later,

    .. math::

        \\chi(k) = P(|x_{t+k}| > u \\mid |x_t| > u),

    estimated by the fraction of exceedances that are followed by another at lag :math:`k`. With
    :math:`u` the ``quantile`` of :math:`|x_t - \\bar{x}|`, :math:`\\chi(k)` is the empirical
    lag-:math:`k` tail-dependence coefficient at that level (Ledford and Tawn [2]_ give
    diagnostics of extremal dependence of this kind). Under independence of the extremes it is
    the exceedance rate :math:`\\hat{p} \\approx 1 -` ``quantile`` at every lag, whatever the
    dependence of the bulk of the distribution, and it detects extremes that recur in time even
    when the autocorrelation of :math:`|x|` is zero.

    The function scans :math:`k = 1, ..., K`, with :math:`K = \\min(` ``max_lag``
    :math:`, \\lfloor N/10 \\rfloor)`. At each lag, with :math:`n_k` exceedances that have a sample
    :math:`k` steps later and :math:`c_k` of them followed by another exceedance, the count's
    excess over its expectation under independence is expressed as a Poisson z-score,
    :math:`z_k = (c_k - n_k \\hat{p})/\\sqrt{n_k \\hat{p}}`. Davis, Mikosch and Cribben [3]_
    study the sampling variability of the extremogram and give bootstrap confidence bands for it.

    Parameters
    ----------
    y : array-like
        Input time series.
    quantile : float, optional
        Quantile of :math:`|x_t - \\bar{x}|` that sets the threshold, in (0, 1). Default is 0.95.
    max_lag : int, optional
        Largest lag scanned, further limited to a tenth of the series length. Default is 100.

    Returns
    -------
    dict
        - 'max_chi': the largest :math:`\\chi(k)`.
        - 'max_z': the largest :math:`z_k`.
        - 'peak_lag': the lag at which :math:`z_k` is largest.

        All NaN if the series is shorter than 10 samples or has fewer than two exceedances.

    References
    ----------
    .. [1] R. A. Davis and T. Mikosch, "The extremogram: a correlogram for extreme events",
       Bernoulli 15, 977-1009 (2009).
    .. [2] A. W. Ledford and J. A. Tawn, "Diagnostics for dependence within time series
       extremes", J. R. Stat. Soc. B 65, 521-543 (2003).
    .. [3] R. A. Davis, T. Mikosch and I. Cribben, "Towards estimating extremal serial
       dependence via the bootstrapped extremogram", J. Econometrics 170, 142-152 (2012).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), not {quantile}")
    K = min(int(max_lag), n // 10)
    if K < 1 or not np.all(np.isfinite(y)):
        return np.nan
    a = np.abs(y - y.mean())
    ex = a > _quantile(a, quantile, method='linear')
    n_ex = int(ex.sum())
    if n_ex < 2:
        return np.nan
    p_hat = n_ex / n
    idx = np.flatnonzero(ex)
    lags = np.arange(1, K + 1)
    n_k = np.array([np.count_nonzero(idx < n - k) for k in lags])
    c_k = np.array([np.count_nonzero(ex[idx[idx < n - k] + k]) for k in lags])
    with np.errstate(invalid='ignore', divide='ignore'):
        chi = c_k / n_k
        z = (c_k - n_k * p_hat) / np.sqrt(n_k * p_hat)
    if not np.any(np.isfinite(z)):
        return np.nan
    k_best = int(np.nanargmax(z))
    return {"max_chi": float(np.nanmax(chi)), "max_z": float(z[k_best]), "peak_lag": k_best + 1}
