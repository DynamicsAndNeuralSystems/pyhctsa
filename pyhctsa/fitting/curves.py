"""Curve fitting: nonlinear least squares, polynomial fits and goodness-of-fit statistics.

Robust (bisquare) linear regression is in :mod:`pyhctsa.fitting.robust_regression`.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares as _least_squares


def fit_curve(fun, p0, x, y):
    """Nonlinear least-squares fit of the coefficients of ``fun(p, x)`` to ``y``.

    Minimizes ``sum((fun(p, x) - y)**2)`` over ``p``, starting from ``p0``, with the
    trust-region-reflective algorithm of :func:`scipy.optimize.least_squares` (Branch, Coleman
    & Li 1999, SIAM J. Sci. Comput. 21(1):1-23), a forward-difference Jacobian and the solver's
    default tolerances.

    Parameters
    ----------
    fun : callable
        ``fun(p, x)`` returns the model values (array-like, one per data point).
    p0 : array-like
        Start point (k coefficients).
    x, y : array-like
        Data; ``y`` is flattened, ``x`` is passed to ``fun`` as a float array.

    Returns
    -------
    numpy.ndarray
        Fitted coefficients (float64, length k), in the order of ``p0``.

    Raises
    ------
    ValueError
        If the data contain NaN or Inf, there are fewer data points than coefficients, or the
        residuals are not finite at the start point.
    """
    x = np.asarray(x, dtype=float)
    y = np.ravel(np.asarray(y, dtype=float))
    p0 = np.ravel(np.asarray(p0, dtype=float))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y cannot contain NaN or Inf values.")
    if y.size < p0.size:
        raise ValueError("Not enough data points to fit the model coefficients.")

    def resid(p):
        return np.ravel(np.asarray(fun(p, x), dtype=float)) - y

    with np.errstate(all="ignore"):
        sol = _least_squares(resid, p0, method="trf", jac="2-point")
    return np.array(sol.x, dtype=float)


def fit_exp(x, y, start_point=(1.0, -0.2)):
    """Fit the exponential model ``a*exp(b*x)`` by nonlinear least squares.

    Parameters
    ----------
    x, y : array-like
        Data.
    start_point : sequence of two floats
        Start values for ``(a, b)``.

    Returns
    -------
    (float, float)
        The fitted ``(a, b)``.
    """
    p = fit_curve(lambda c, t: c[0] * np.exp(c[1] * t), start_point, x, y)
    return float(p[0]), float(p[1])


def polyfit(x, y, deg):
    """Least-squares polynomial fit of degree ``deg``; coefficients in descending powers.

    Parameters
    ----------
    x, y : array-like
        Data points (flattened, equal length m).
    deg : int
        Polynomial degree (non-negative).

    Returns
    -------
    numpy.ndarray
        Coefficients, shape ``(deg + 1,)``, compatible with ``numpy.polyval``. All-NaN if any
        input value is NaN or infinite, or if there are fewer points than coefficients (the
        fit is not identified; a minimum-norm solution would be an arbitrary choice).

    Notes
    -----
    Thin wrapper of :func:`numpy.polyfit`.
    """
    x = np.ravel(np.asarray(x, dtype=float))
    y = np.ravel(np.asarray(y, dtype=float))
    n = int(deg) + 1
    if x.size < n or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.full(n, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.polyfit(x, y, deg)
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)


def goodness_of_fit(y, y_fit, num_coeffs):
    """Goodness-of-fit statistics of a fitted curve.

    Parameters
    ----------
    y : array-like
        Observed values (n points).
    y_fit : array-like
        Fitted values.
    num_coeffs : int
        Number of fitted coefficients m.

    Returns
    -------
    dict
        ``'sse'`` (sum of squared errors), ``'rsquare'``, ``'adjrsquare'``, ``'rmse'`` and
        ``'dfe'`` (n - m). ``adjrsquare`` and ``rmse`` are NaN when ``n - m <= 0``;
        ``rsquare`` and ``adjrsquare`` are NaN when ``y`` is constant (the total sum of squares is
        zero) or contains NaN.

    Notes
    -----
    SSE = sum (y - yhat)^2, SST = sum (y - mean(y))^2, R^2 = 1 - SSE/SST, adjusted
    R^2 = 1 - SSE (n-1) / (SST (n-m)) and RMSE = sqrt(SSE/(n-m)) (Draper & Smith, *Applied
    Regression Analysis*, 3rd ed., ch. 2).
    """
    y = np.ravel(np.asarray(y, dtype=float))
    yh = np.ravel(np.asarray(y_fit, dtype=float))
    n = y.size
    v = n - int(num_coeffs)
    with np.errstate(all="ignore"):
        sse = float(np.sum((y - yh) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2)) if n > 0 else np.nan
        defined = bool(np.isfinite(sse) and np.isfinite(sst) and sst > 0)
        rsq = 1.0 - sse / sst if defined else np.nan
        adj = 1.0 - sse * (n - 1) / (sst * v) if (defined and v > 0) else np.nan
        rmse = float(np.sqrt(sse / v)) if v > 0 else np.nan
    return {
        "sse": sse,
        "rsquare": float(rsq),
        "adjrsquare": float(adj),
        "rmse": rmse,
        "dfe": v,
    }
