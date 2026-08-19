from typing import Optional

import numpy as np
from scipy.linalg import qr, solve_triangular

from .matlab_fit import lsqcurvefit_trr

# Default fit options for every library model in this file. Only the gauss/sin
# models constrain anything: gauss widths c_i >= 0 and sin frequencies b_i >= 0.
_FIT_OPTS = dict(tol_fun=1e-6, tol_x=1e-6, max_iter=400, max_fun_evals=600,
                 diff_min_change=1e-8, diff_max_change=0.1)


def _two_points_required(x):
    if len(x) < 2:
        raise ValueError("At least two data points are required to fit a curve.")


# ------------------------------------------------------------------------------
# Start points
# ------------------------------------------------------------------------------

def exp1start(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Start point for the ``exp1`` model, ``a*exp(b*x)``.

    The idea is that for uniformly spaced x, y should form a geometric
    sequence; summing the first and second halves and taking the ratio gives b
    directly. Non-uniform x is first resampled onto a uniform grid by
    log-linear interpolation.
    """
    x = np.asarray(x, dtype=float).ravel().copy()
    y = np.asarray(y, dtype=float).ravel().copy()
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x, kind='stable')
        x, y = x[idx], y[idx]

    n = len(x)
    q = n // 2
    if q < 1:
        _two_points_required(x)

    if n > 2 and np.any(np.diff(np.diff(x)) > 1 / n ** 2):
        # Non-uniform x: build a uniform grid x1 and interpolate y onto it,
        # exponentially (i.e. linearly in log|y|).
        drop = np.diff(x) < np.finfo(float).eps ** 0.7
        keep = np.ones(n, dtype=bool)
        keep[:-1] = ~drop
        x, y = x[keep], y[keep]
        n = len(x)
        if n < 2:
            return np.random.rand(2)
        q = n // 2
        sid = np.sign(y)
        x1 = np.linspace(np.min(x), np.max(x), n)
        x1[-1] = x1[-1] - np.spacing(x1[-1])
        # bin index of each x1 within edges x
        idc = np.searchsorted(x, x1, side='right') - 1
        idc = np.clip(idc, 0, n - 2)
        ly = np.log(np.abs(y) + np.finfo(float).eps)
        b = (ly[idc + 1] - ly[idc]) / (x[idc + 1] - x[idc])
        a = ly[idc] - b * x[idc]
        y = sid * np.exp(a + b * x1)
        x = x1

    S1 = np.sum(y[:q])
    while S1 == 0:
        q -= 1
        if q < 1:
            return np.array([0.0, 0.0])
        S1 = np.sum(y[:q])
    S2 = np.sum(y[q:2 * q])

    mx = np.mean(np.diff(x))
    if mx <= 0:
        # For constant x no sensible guess exists; fall back to b = 1.
        b = 1.0
    else:
        with np.errstate(all='ignore'):
            b = np.real(np.power(complex(S2 / S1), 1.0 / q))
            b = np.real(np.log(complex(np.power(complex(S2 / S1), 1.0 / q)))) / mx

    with np.errstate(all='ignore'):
        a = np.sum(y) / np.sum(np.exp(b * x))
    if not np.isfinite(a):
        # exp(b*x) underflowed to zero, so any 'a' fits equally badly
        a = 1.0

    return np.array([a, b])


def _ensure_positive(u: np.ndarray, v: np.ndarray):
    """Drop points where ``u`` is non-positive."""
    keep = u > 0
    return u[keep], v[keep]


def _make_min_x_first(x: np.ndarray, y: np.ndarray):
    """
    Move the minimum-x point to the front, coalescing ties by averaging y.
    """
    if len(x) == 0:
        return x, np.nan
    min_x = np.min(x)
    idx = x == min_x
    min_y = np.mean(y[idx])
    x, y = x[~idx], y[~idx]
    return np.concatenate([[min_x], x]), np.concatenate([[min_y], y])


def power2start(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Start point for ``power2``, ``c + a*x^b`` (``power2start``).

    Assumes every point comes from the same power law, estimates b at each
    point relative to the leftmost one, and takes the arithmetic mean.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x, y = _ensure_positive(x, y)
    y, x = _ensure_positive(y, x)
    x, y = _make_min_x_first(x, y)

    if np.size(x) < 2:
        return np.random.rand(3)

    with np.errstate(all='ignore'):
        b = np.mean(np.log(y[0] / y[1:]) / np.log(x[0] / x[1:]))
        a = np.mean(y / x ** b)
        c = np.mean(y - a * x ** b)
    return np.array([a, b, c])


def power1start(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Start point for ``power1``, ``a*x^b``: ``power2start`` without the offset."""
    return power2start(x, y)[:2]


def gaussnstart(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Start point for ``gaussn``, ``sum(a_i*exp(-((x-b_i)/c_i)^2))`` (``gaussnstart``).

    Peels off one peak at a time: take the largest remaining y as the height,
    its x as the centre, estimate the width from the points below it, subtract
    that Gaussian, and repeat.
    """
    x = np.asarray(x, dtype=float).ravel().copy()
    y = np.asarray(y, dtype=float).ravel().copy()
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x, kind='stable')
        x, y = x[idx], y[idx]

    p, q, r = [], [], []
    while len(p) < n:
        k = int(np.flatnonzero(y == np.max(y))[-1])  # 'last' max
        a = y[k]
        b = x[k]
        idsel = (y > 0) & (y < a)
        if not np.any(idsel):
            break
        c = np.mean(np.abs(x[idsel] - b) / np.sqrt(np.log(a / y[idsel]))) / (2 * n - len(p))
        y = y - a * np.exp(-((x - b) / c) ** 2)
        p.append(b)
        q.append(a)
        r.append(c)

    if len(p) < n:
        # Not enough significant peaks: space the rest evenly over the domain
        num_found = len(p)
        num_to_add = n - num_found
        b = np.linspace(np.min(x), np.max(x), num_to_add + 2)
        p = p + list(b[1:-1])
        q = q + [np.max(y)] * num_to_add
        r = r + [b[1] - b[0]] * num_to_add

    start = np.zeros(3 * n)
    start[0::3] = q
    start[1::3] = p
    start[2::3] = r
    return start


def getxygrid(x: np.ndarray, y: np.ndarray):
    """
    Put (x, y) on an evenly spaced lattice so an FFT can be taken (``getxygrid``).

    Sorts, merges near-duplicate x by averaging, and -- when the spacing is
    already uniform -- resamples onto ``linspace`` by linear interpolation.
    """
    x = np.asarray(x, dtype=float).ravel().copy()
    y = np.asarray(y, dtype=float).ravel().copy()
    _two_points_required(x)

    diffx = np.diff(x)
    if np.any(diffx < 0):
        idx = np.argsort(x, kind='stable')
        x, y = x[idx], y[idx]
        diffx = np.diff(x)

    # Merge repeated x entries to avoid dividing by zero
    tol = np.finfo(float).eps ** 0.7
    idx = np.concatenate([diffx < tol, [False]])
    idx2 = np.concatenate([[False], idx[:-1]])
    x[idx] = (x[idx] + x[idx2]) / 2
    y[idx] = (y[idx] + y[idx2]) / 2
    x, y = x[~idx2], y[~idx2]

    if len(x) > 2:
        diffx = np.diff(x)
        if np.all(np.abs(diffx - diffx[0]) < tol * np.max(diffx)):
            newx = np.linspace(np.min(x), np.max(x), len(x))
            y = np.interp(newx, x, y)
            x = newx

    return x, y


def sinnstart(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Start point for ``sinn``, ``sum(a_i*sin(b_i*x+c_i))`` (``sinnstart``).

    Finds each frequency from a peak of the FFT of the running residual, then
    recovers amplitude and phase by linear least squares on the sine/cosine
    pair at every frequency found so far.
    """
    x, y = getxygrid(x, y)
    if len(x) < 2:
        return np.random.rand(3 * n)

    start = np.zeros(3 * n)
    oldpeaks = []
    lengthx = len(x)
    freqs = np.zeros(n)
    res = y.copy()  # residuals from fit so far
    ab = None

    for j in range(1, n + 1):
        fy = np.fft.fft(res)  # don't subtract mean, no constant term
        if oldpeaks:
            fy[oldpeaks] = 0  # omit frequencies already used

        half = fy[:lengthx // 2]
        # max() on complex data orders by magnitude, then by phase
        maxloc = int(np.lexsort((np.angle(half), np.abs(half)))[-1])
        oldpeaks.append(maxloc)
        w = 2 * np.pi * max(0.5, maxloc) / (x[-1] - x[0])
        freqs[j - 1] = w

        X = np.zeros((lengthx, 2 * j))
        for k in range(j):
            X[:, 2 * k] = np.sin(freqs[k] * x)
            X[:, 2 * k + 1] = np.cos(freqs[k] * x)

        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        if j < n:
            res = y - X @ ab

    for k in range(n):
        start[3 * k] = np.sqrt(ab[2 * k] ** 2 + ab[2 * k + 1] ** 2)
        start[3 * k + 1] = freqs[k]
        start[3 * k + 2] = np.arctan2(ab[2 * k + 1], ab[2 * k])
    return start


# ------------------------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------------------------

def _gaussn_basis(p_nl: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Design matrix of ``exp(-((x-b_i)/c_i)^2)`` columns for the separable fit."""
    A = np.empty((len(x), n))
    for i in range(n):
        b, c = p_nl[2 * i], p_nl[2 * i + 1]
        # c == 0 is singular; fill a column of ones so any 'a' is valid
        A[:, i] = np.ones(len(x)) if c == 0 else np.exp(-((x - b) / c) ** 2)
    return A


def _exp1_basis(p_nl: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    return np.exp(p_nl[0] * x)[:, None]


def _power1_basis(p_nl: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(all='ignore'):
        return (x ** p_nl[0])[:, None]


_SEPARABLE_BASIS = {'gauss': _gaussn_basis, 'exp': _exp1_basis, 'power': _power1_basis}


def _lin_solve(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve ``A x = y`` for the separable linear coefficients.

    For an overdetermined system this uses a column-pivoted QR and returns a
    *basic* solution -- at most ``rank(A)`` nonzeros, with the coefficients of
    the dropped columns set to zero. NumPy's ``lstsq`` returns the
    minimum-norm solution instead, which differs whenever ``A`` is rank
    deficient. The separable Gaussian design matrix does degenerate that way
    (two peaks collapsing onto each other), and the two conventions then send
    the optimiser to different places, so the basic solution is reproduced here.
    """
    with np.errstate(all='ignore'):
        if not np.all(np.isfinite(A)):
            return np.zeros(A.shape[1])

        Q, R, piv = qr(A, mode='economic', pivoting=True)
        absdiag = np.abs(np.diag(R))
        tol = max(A.shape) * np.finfo(float).eps * (absdiag[0] if absdiag.size else 0.0)
        rank = int(np.sum(absdiag > tol))

        coeffs = np.zeros(A.shape[1])
        if rank == 0:
            return coeffs
        z = solve_triangular(R[:rank, :rank], (Q.T @ y)[:rank])
        coeffs[piv[:rank]] = z
        return coeffs


def sinn_eval(p: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Evaluate ``sum(a_i*sin(b_i*x+c_i))``."""
    f = np.zeros(len(x))
    for i in range(n):
        a, b, c = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        f = f + a * np.sin(b * x + c)
    return f


def sinn_jac(p: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Analytic Jacobian of :func:`sinn_eval` with respect to all 3n coefficients."""
    J = np.empty((len(x), 3 * n))
    for i in range(n):
        a, b, c = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        arg = b * x + c
        J[:, 3 * i] = np.sin(arg)
        J[:, 3 * i + 1] = a * x * np.cos(arg)
        J[:, 3 * i + 2] = a * np.cos(arg)
    return J


# ------------------------------------------------------------------------------
# The fit
# ------------------------------------------------------------------------------

def _check_start_point(start: np.ndarray, rng=None) -> np.ndarray:
    """
    Validate a start point.

    ---NOTE:
    When the heuristic produces a non-finite or complex start point it is
    discarded entirely and replaced with ``rand(size(start))``. Any fit that
    takes this path is therefore *not reproducible* -- it depends on the RNG
    state when the fit is called. It arises for, e.g.,
    ``exp1`` on a histogram whose upper half is all-empty bins, which drives
    the estimate of ``b`` to -Inf.
    """
    start = np.asarray(start, dtype=float)
    if not np.all(np.isfinite(start)):
        rng = np.random.default_rng() if rng is None else rng
        return rng.random(start.shape)
    return start


def fit_library_model(x, y, model: str):
    """
    Fit one of the Curve Fitting Toolbox library models.

    Parameters
    ----------
    x, y : array-like
        The data to fit. Must be the same length.
    model : {'exp1', 'power1', 'gauss1', 'gauss2', 'sin1', 'sin2', 'sin3'}
        The library model name.

    Returns
    -------
    dict
        ``coeffs`` (in the library coefficient order), ``yhat`` (fitted values)
        and ``residuals`` (``y - yhat``).

    Raises
    ------
    ValueError
        If the model name is unknown, or if ``power1`` is given non-positive
        x-data.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if model.startswith('sin'):
        # Non-separable: optimise all 3n coefficients with an analytic Jacobian,
        # subject to the frequencies b_i >= 0.
        n = int(model[3:])
        start = _check_start_point(sinnstart(x, y, n))
        lower = np.full(3 * n, -np.inf)
        lower[1::3] = 0.0
        coeffs = lsqcurvefit_trr(
            lambda p, xd: sinn_eval(p, xd, n), start, x, y,
            jac=lambda p, xd: sinn_jac(p, xd, n), lower=lower, **_FIT_OPTS)
        yhat = sinn_eval(coeffs, x, n)
        return {'coeffs': coeffs, 'yhat': yhat, 'residuals': y - yhat}

    if model.startswith('gauss'):
        n = int(model[5:])
        full_start = _check_start_point(gaussnstart(x, y, n))
        # Separable: drop the linear a_i, keep (b_i, c_i)
        start = np.concatenate([full_start[3 * i + 1:3 * i + 3] for i in range(n)])
        lower = np.tile([-np.inf, 0.0], n)  # widths c_i >= 0
        basis = _gaussn_basis
    elif model == 'exp1':
        n = 1
        start = _check_start_point(exp1start(x, y))[1:]  # keep b only
        lower = None
        basis = _exp1_basis
    elif model == 'power1':
        n = 1
        if np.any(x <= 0):
            raise ValueError("Power functions cannot be fit to non-positive xdata.")
        start = _check_start_point(power1start(x, y))[1:]  # keep b only
        lower = None
        basis = _power1_basis
    else:
        raise ValueError(f"Unknown library model '{model}'.")

    # Separable residual: the linear coefficients are recovered by least squares
    # at every evaluation, so the solver only sees the nonlinear ones. A
    # finite-difference Jacobian is used here (Jacobian='off').
    def sep_eval(p_nl, xd):
        A = basis(p_nl, xd, n)
        return A @ _lin_solve(A, y)

    p_nl = lsqcurvefit_trr(sep_eval, start, x, y, lower=lower, **_FIT_OPTS)

    A = basis(p_nl, x, n)
    lin = _lin_solve(A, y)
    yhat = A @ lin

    if model.startswith('gauss'):
        coeffs = np.empty(3 * n)
        coeffs[0::3] = lin
        coeffs[1::3] = p_nl[0::2]
        coeffs[2::3] = p_nl[1::2]
    else:
        coeffs = np.array([lin[0], p_nl[0]])

    return {'coeffs': coeffs, 'yhat': yhat, 'residuals': y - yhat}
