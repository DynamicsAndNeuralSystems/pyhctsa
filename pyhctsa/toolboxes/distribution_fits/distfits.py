"""Maximum-likelihood fits of one-sample parametric distributions.

Thin wrappers of the maximum-likelihood fitters of :mod:`scipy.stats` with the location (and,
for the beta distribution, the scale) fixed, so that each function returns just the shape and
scale parameters of the usual two-parameter family. Input that has no maximum-likelihood fit
(empty or non-finite data, constant data, values outside the support of the family) returns NaN
parameters rather than raising.
"""
import warnings

import numpy as np
from scipy import optimize, stats as _st

_FIT_ERRORS = (ValueError, RuntimeError, OverflowError, FloatingPointError, ZeroDivisionError,
               np.linalg.LinAlgError)


def _as_vector(x) -> np.ndarray:
    return np.ravel(np.asarray(x, dtype=float))


def _usable(x: np.ndarray) -> bool:
    """True if ``x`` has at least two finite values that are not all equal."""
    return x.size >= 2 and bool(np.all(np.isfinite(x))) and bool(np.ptp(x) > 0)


def _fit(dist, x, *, shape_count, **fixed):
    """Run ``dist.fit``; returns the free parameters or NaNs if the fit fails or is degenerate."""
    nan = (float("nan"),) * shape_count
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            params = dist.fit(x, **fixed)
    except _FIT_ERRORS:
        return nan
    return tuple(float(p) for p in params)


def betafit(x: np.ndarray) -> np.ndarray:
    """Maximum-likelihood fit of a beta distribution on (0, 1); returns ``[a, b]``.

    Parameters
    ----------
    x : array-like
        Data strictly inside (0, 1), not all equal.

    Returns
    -------
    numpy.ndarray
        ``[a, b]`` shape parameters (float64, shape (2,)); ``[nan, nan]`` if the data are empty,
        non-finite, constant or not strictly inside (0, 1).

    Notes
    -----
    ``scipy.stats.beta.fit(x, floc=0, fscale=1)``: maximizes the beta log-likelihood
    (Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* vol. 2, 2nd ed., ch. 25).
    """
    x = _as_vector(x)
    if not _usable(x) or np.any(x <= 0) or np.any(x >= 1):
        return np.full(2, np.nan)
    a, b = _fit(_st.beta, x, shape_count=2, floc=0, fscale=1)[:2]
    return np.array([a, b], dtype=float)


def evfit(x: np.ndarray) -> tuple:
    """Maximum-likelihood fit of the type-I extreme value (minimum, left Gumbel) distribution.

    Parameters
    ----------
    x : array-like
        Data (finite, not all equal).

    Returns
    -------
    (float, float)
        ``(mu, sigma)`` location and scale; ``(nan, nan)`` if the data are empty, non-finite or
        constant.

    Notes
    -----
    ``scipy.stats.gumbel_l.fit``; the density is ``exp(z - exp(z)) / sigma`` with
    ``z = (x - mu) / sigma``.
    """
    x = _as_vector(x)
    nan = (float("nan"), float("nan"))
    if not _usable(x):
        return nan
    mu, sigma = _fit(_st.gumbel_l, x, shape_count=2)
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        return nan
    return mu, sigma


def gamfit(x: np.ndarray) -> tuple:
    """Maximum-likelihood fit of a gamma distribution; returns ``(shape, scale)``.

    Parameters
    ----------
    x : array-like
        Strictly positive data (finite, not all equal).

    Returns
    -------
    (float, float)
        ``(a, b)`` shape and scale; ``(nan, nan)`` if the data are empty, non-finite, constant or
        not strictly positive (the likelihood of a gamma law is unbounded or zero when the data
        contain zeros).

    Notes
    -----
    ``scipy.stats.gamma.fit(x, floc=0)``: the MLE solves ``log a - digamma(a) = log(mean(x)) -
    mean(log(x))`` with ``b = mean(x) / a`` (Johnson, Kotz & Balakrishnan, vol. 1, ch. 17).
    """
    x = _as_vector(x)
    nan = (float("nan"), float("nan"))
    if not _usable(x) or np.any(x <= 0):
        return nan
    a, _, b = _fit(_st.gamma, x, shape_count=3, floc=0)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return nan
    return a, b


def wblfit(x: np.ndarray) -> tuple:
    """Maximum-likelihood fit of a Weibull distribution; returns ``(scale, shape)``.

    Parameters
    ----------
    x : array-like
        Strictly positive data (finite, not all equal).

    Returns
    -------
    (float, float)
        ``(a, b)`` scale and shape; ``(nan, nan)`` if the data are empty, non-finite, constant or
        not strictly positive.

    Notes
    -----
    The profile score equation of the shape ``c``,
    ``sum(x^c log x) / sum(x^c) - 1/c - mean(log x) = 0``, is strictly increasing in ``c`` and is
    solved by Brent's method; the scale is then ``mean(x^c)^(1/c)`` (Johnson, Kotz &
    Balakrishnan, *Continuous Univariate Distributions* vol. 1, ch. 21). (The generic
    ``scipy.stats.weibull_min.fit`` stops at a looser tolerance.) The data are divided by their
    maximum so that ``x^c`` cannot overflow.
    """
    x = _as_vector(x)
    nan = (float("nan"), float("nan"))
    if not _usable(x) or np.any(x <= 0):
        return nan
    xmax = float(np.max(x))
    xs = x / xmax
    lx = np.log(xs)
    mean_lx = float(np.mean(lx))

    def score(c):
        w = np.exp(c * lx)                       # xs**c, in (0, 1]
        return float(np.sum(w * lx) / np.sum(w)) - 1.0 / c - mean_lx

    lo, hi = 1.0, 1.0
    for _ in range(200):                         # bracket the root: score < 0 at lo, > 0 at hi
        if score(lo) < 0:
            break
        lo /= 2.0
    for _ in range(200):
        if score(hi) > 0:
            break
        hi *= 2.0
    if not (score(lo) < 0 < score(hi)):
        return nan
    c = float(optimize.brentq(score, lo, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))
    scale = xmax * float(np.mean(np.exp(c * lx))) ** (1.0 / c)
    if not (np.isfinite(c) and np.isfinite(scale) and c > 0 and scale > 0):
        return nan
    return scale, c
