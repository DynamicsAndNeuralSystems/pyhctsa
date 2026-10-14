"""Nearest-neighbour (locally constant) one-step prediction in a delay embedding."""
import logging

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger('pyhctsa')

_BLOCK_BYTES = 32 * 2 ** 20

# A prediction error is reported only if enough states could be predicted for it to describe
# the series: at least this many predictions, and at least this fraction of all states.
_MIN_PREDICTIONS = 10
_MIN_PREDICTED_FRACTION = 0.5


def _nn_prediction_residuals(y: ArrayLike, de: int, tau: int, theiler_win: int = 0):
    """Residuals of the drop-one-out nearest-neighbour (zeroth-order) one-step predictor.

    The series is embedded in ``de`` dimensions with delay ``tau``: the state at time
    ``t >= (de - 1) * tau`` is ``(y[t], y[t - tau], ..., y[t - (de - 1) * tau])``. The value
    following each state (``y[t + 1]``) is predicted by the value that followed its nearest
    neighbour (Euclidean distance) among the other states, excluding states within
    ``theiler_win`` samples of it (``theiler_win = 0`` excludes only the state itself).
    Exact distance ties go to the earliest neighbour.

    A state with no admissible neighbour (every other state lies inside its Theiler window)
    cannot be predicted and is left out of the residuals. The error is still meaningful if
    most states are predicted, so NaN is returned only when fewer than 10 states, or fewer
    than half of all states, could be predicted: below that the residuals describe a small and
    possibly unrepresentative part of the series (for example only its ends).

    Parameters
    ----------
    y : array-like
        Input time series.
    de, tau : int
        Embedding dimension and delay (positive).
    theiler_win : int, optional
        Theiler window: neighbours with ``|j - i| <= theiler_win`` are excluded.

    Returns
    -------
    numpy.ndarray or float
        Residuals (prediction minus data), one per predictable state; at most
        ``len(y) - 1 - (de - 1) * tau`` of them, in time order. NaN if too few states
        could be predicted (see above).

    Notes
    -----
    The method of analogues / locally constant predictor: E. N. Lorenz, J. Atmos. Sci.
    26:636 (1969); J. D. Farmer and J. J. Sidorowich, Phys. Rev. Lett. 59:845 (1987);
    G. Sugihara and R. M. May, Nature 344:734 (1990). Theiler window: J. Theiler, Phys. Rev.
    A 34:2427 (1986). The search is brute force in blocks of rows (bounded memory).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    de, tau, theiler_win = int(de), int(tau), int(theiler_win)
    if de < 1 or tau < 1 or theiler_win < 0:
        raise ValueError("de and tau must be positive and theiler_win non-negative")
    span = (de - 1) * tau
    m = y.size - 1 - span  # states that have a successor
    if m < _MIN_PREDICTIONS:
        logger.warning('Time series is too short to embed for nearest-neighbour prediction')
        return np.nan

    # coordinates of the states: column k holds y[t - k*tau], t = span..N-2
    t = np.arange(span, span + m)
    states = np.column_stack([y[t - k * tau] for k in range(de)])
    successors = y[t + 1]

    nearest = np.empty(m, dtype=np.intp)
    predictable = np.ones(m, dtype=bool)
    cols = np.arange(m)
    block = max(1, _BLOCK_BYTES // (8 * m))
    for start in range(0, m, block):
        rows = np.arange(start, min(start + block, m))
        d2 = np.zeros((rows.size, m))
        for k in range(de):
            diff = states[rows, k][:, None] - states[None, :, k]
            d2 += diff * diff
        d2[np.abs(cols[None, :] - rows[:, None]) <= theiler_win] = np.inf
        j = np.argmin(d2, axis=1)  # first minimum: smallest index on exact ties
        nearest[rows] = j
        predictable[rows] = np.isfinite(d2[np.arange(rows.size), j])

    n_predicted = int(predictable.sum())
    if n_predicted < max(_MIN_PREDICTIONS, _MIN_PREDICTED_FRACTION * m):
        logger.warning(f'Only {n_predicted} of {m} states have a neighbour outside the Theiler window')
        return np.nan
    return (successors[nearest] - successors)[predictable]
