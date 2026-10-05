"""Shared helpers for the robust, portable hctsa features (ports of hctsa ``BF_*`` functions).

hctsa redefined a number of fragile features so that they do not depend on the random
stream, the optimizer, the kernel-density defaults or the histogram bin rule of the
platform. The ``BF_*`` helpers written for this live in ``PeripheryFunctions`` in hctsa;
this module holds their Python ports. The operation modules import from here::

    from ..robust import bf_random, bf_ks_density, bf_hist_edges, ...

Every function follows the hctsa one step for step (same arithmetic, same order of
operations where it matters), so that the two implementations agree to rounding. The
random generator (:func:`bf_random`) is bit-identical to MATLAB's ``BF_Random`` for
uniform draws, and normal draws agree to about 1e-16.

Conventions: the Python name is the snake_case of the hctsa name; vectors are 1-D
arrays; hctsa functions that return several outputs return a tuple, and ``BF_ExpFit``
returns a dict with the same field names as the MATLAB struct. Permutations from
:func:`bf_random` are 1-based, exactly as in MATLAB (subtract one to index an array).
"""
from typing import Optional, Tuple, Union

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from .utils import _linspace, _round_half_away, matlab_quantile

__all__ = [
    'bf_random', 'bf_random_seed', 'bf_tie_break_noise', 'bf_runs_z', 'bf_residual_stats', 'bf_theil_sen',
    'bf_exp_fit', 'bf_fit_density_curve', 'bf_gauss_mix2', 'bf_fit_sinusoids',
    'bf_ks_density', 'bf_hist_edges', 'bf_quantile_edges', 'bf_half_sample_mode', 'bf_remove_points',
]


# ------------------------------------------------------------------------------
# BF_Random
# ------------------------------------------------------------------------------
_M1 = 4294967087
_M2 = 4294944443
_NORM = 2.328306549295727688e-10


@njit(cache=True)
def _mrg32k3a(k, state, num_skip):
    """``k`` uniforms in (0,1) from the MRG32k3a state (six int64 words), after
    discarding ``num_skip`` draws. Integer products stay below 2^63 (and below 2^53,
    where MATLAB's doubles are exact), so the stream is identical to MATLAB's."""
    s10, s11, s12, s20, s21, s22 = state[0], state[1], state[2], state[3], state[4], state[5]
    u = np.empty(k)
    for i in range(k + num_skip):
        p1 = (1403580 * s11 - 810728 * s10) % _M1  # (% is non-negative: as MATLAB mod)
        s10 = s11
        s11 = s12
        s12 = p1
        p2 = (527612 * s22 - 1370589 * s20) % _M2
        s20 = s21
        s21 = s22
        s22 = p2
        if i >= num_skip:
            if p1 > p2:
                u[i - num_skip] = (p1 - p2) * _NORM
            else:
                u[i - num_skip] = (p1 - p2 + _M1) * _NORM
    return u


def bf_random(n: int, seed: Union[int, float, ArrayLike] = 0, kind: str = 'uniform') -> np.ndarray:
    """Portable pseudo-random numbers: the same stream in MATLAB and Python (hctsa ``BF_Random``).

    L'Ecuyer's combined multiple recursive generator MRG32k3a (L'Ecuyer 1999,
    doi:10.1287/opre.47.1.159), in integer arithmetic. It leaves NumPy's global random
    state untouched. Uniform draws are bit-identical to hctsa's ``BF_Random``; normal
    draws agree to the rounding of ``log``, ``cos`` and ``sin`` (about 1e-16).

    Parameters
    ----------
    n : int
        The number of values to return.
    seed : int or sequence of 6 ints, optional
        A scalar seed (integer in [0, 4e9)) sets all six state words to ``12345 + seed``
        and discards the first 8 draws, so that streams from neighboring seeds are
        unrelated (default 0). A vector of six state words ``[s10 s11 s12 s20 s21 s22]``
        starts the raw generator with no draws discarded; the state
        ``[12345]*6`` reproduces L'Ecuyer's reference implementation.
    kind : {'uniform', 'normal', 'perm'}, optional
        ``'uniform'``: n numbers in the open interval (0,1) (default).
        ``'normal'``: n standard normal numbers by Box-Muller: the uniforms (u1, u2) =
        (2k-1, 2k) give ``sqrt(-2 log u1) cos(2 pi u2)`` and ``sqrt(-2 log u1) sin(2 pi u2)``
        as values 2k-1 and 2k.
        ``'perm'``: a random permutation of 1..n (**1-based**, as in MATLAB): the ranks of n
        uniform numbers, i.e. the stable argsort of the n uniforms (plus 1).

    Returns
    -------
    numpy.ndarray
        A 1-D array of length n (float for 'uniform' and 'normal', int for 'perm').
    """
    n = int(n)
    if np.ndim(seed) == 0:
        state = np.full(6, 12345 + int(seed), dtype=np.int64)
        num_skip = 8
    else:
        state = np.asarray(seed, dtype=np.int64).ravel().copy()
        if state.size != 6:
            raise ValueError('a vector seed must hold the six state words')
        num_skip = 0

    if kind == 'uniform':
        return _mrg32k3a(n, state, num_skip)
    if kind == 'normal':
        m = -(-n // 2)  # ceil(n/2)
        u = _mrg32k3a(2 * m, state, num_skip)
        r = np.sqrt(-2.0 * np.log(u[0::2]))
        theta = 2.0 * np.pi * u[1::2]
        x = np.empty(2 * m)
        x[0::2] = r * np.cos(theta)
        x[1::2] = r * np.sin(theta)
        return x[:n]
    if kind == 'perm':
        return np.argsort(_mrg32k3a(n, state, num_skip), kind='stable') + 1
    raise ValueError(f"Unknown kind '{kind}'")


def bf_random_seed(random_seed=None) -> int:
    """The integer seed (for :func:`bf_random`) that a ``random_seed`` input stands for (hctsa ``BF_RandomSeed``).

    ``'default'`` or ``None`` give the fixed seed 0; a number gives that seed (rounded and made
    non-negative: ``mod(round(abs(s)), 4e9)``); ``'none'`` gives a seed drawn from NumPy's global
    random stream (so repeated calls differ).
    """
    if random_seed is None:
        return 0
    if isinstance(random_seed, str):
        if random_seed == 'default':
            return 0
        if random_seed == 'none':
            return int(np.floor(4e9 * np.random.random_sample()))
        raise ValueError(f"Not sure how to interpret the random seed '{random_seed}'")
    s = float(np.asarray(random_seed).ravel()[0]) if np.size(random_seed) else None
    if s is None:
        return 0
    # MATLAB round: half away from zero
    return int(np.mod(np.floor(abs(s) + 0.5), 4e9))


# ------------------------------------------------------------------------------
# BF_TieBreakNoise
# ------------------------------------------------------------------------------
def bf_tie_break_noise(y: ArrayLike, seed: int = 0) -> np.ndarray:
    """Add tiny, reproducible jitter to break exact ties in ``y`` (hctsa ``BF_TieBreakNoise``).

    ``y`` is returned unchanged unless it has a high proportion of repeated values
    (fewer than 90% of its values unique), in which case Gaussian noise of standard
    deviation ``1e-10 * std(y)`` is added: small enough to leave a well-behaved continuous
    series untouched, but enough to break the exact ties that make nearest-neighbor
    (Kraskov/KSG) mutual-information estimators degenerate on quantized or
    periodic-orbit data.

    The noise is ``bf_random(numel(y), seed, 'normal')``, so the same input gives the
    same output as hctsa in MATLAB (to ~1e-16 relative to the noise) and nothing about
    NumPy's global random state is consumed.

    Parameters
    ----------
    y : array-like
        The input vector (or matrix; the repeat test and noise scale use all elements;
        the noise fills a matrix in column-major order, as MATLAB's ``reshape``).
    seed : int, optional
        Seed of the ``bf_random`` stream (default 0). Use different seeds for different
        variables that will be jittered and then compared to one another.

    Returns
    -------
    numpy.ndarray
        The (possibly) jittered input, same shape.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y
    unique_frac = np.unique(y).size / y.size
    sigma = np.std(y, ddof=1)
    if unique_frac < 0.9 and sigma > 0:
        noise = bf_random(y.size, seed, 'normal').reshape(y.shape, order='F')
        y = y + 1e-10 * sigma * noise
    return y


# ------------------------------------------------------------------------------
# BF_RunsZ, BF_ResidualStats
# ------------------------------------------------------------------------------
def bf_runs_z(y: ArrayLike) -> float:
    """Signed z-statistic of a runs test for randomness about the median (hctsa ``BF_RunsZ``).

    Splits the values of ``y`` into those above the median and those at or below it,
    counts the runs and standardizes the count with its exact mean and variance under
    random order (Wald-Wolfowitz): ``z = (R - mu)/sigma``, ``mu = 1 + 2 n1 n2/n``,
    ``sigma^2 = 2 n1 n2 (2 n1 n2 - n) / (n^2 (n - 1))``. ``z < 0`` (too few runs) means
    positive serial dependence, ``z > 0`` negative. If no value lies above the median
    the groups are instead those at the median and those below it.

    Parameters
    ----------
    y : array-like
        A vector (NaN values are ignored).

    Returns
    -------
    float
        The standardized number of runs; NaN for a constant series, or when the null
        distribution of the number of runs has zero variance.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.nan
    m = np.median(y)
    is_up = y > m
    if not is_up.any():
        is_up = y >= m
    n1 = int(is_up.sum())
    n2 = y.size - n1
    if n1 == 0 or n2 == 0:
        return np.nan
    n = n1 + n2
    r = 1 + int(np.sum(is_up[1:] != is_up[:-1]))
    mu = 1 + 2 * n1 * n2 / n
    v = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    if v == 0:
        return np.nan
    return float((r - mu) / np.sqrt(v))


def bf_residual_stats(res: ArrayLike, sstot: float) -> Tuple[float, float, float]:
    """Statistics of the residuals of a fit, for remaining structure (hctsa ``BF_ResidualStats``).

    The autocorrelation of the residuals at lags 1 and 2 (the Fourier method of
    ``CO_AutoCorr``) and a runs test (:func:`bf_runs_z`). The residuals are taken in the
    order given. If the fit is exact (residual sum of squares at most ``1e-12 * sstot``)
    all three outputs are NaN: only numerical error is left.

    Parameters
    ----------
    res : array-like
        The residuals.
    sstot : float
        The total sum of squares of the fitted data about its mean.

    Returns
    -------
    tuple of float
        ``(ac1, ac2, runsz)``.
    """
    from .operations.correlation import autocorr
    res = np.asarray(res, dtype=float).ravel()
    if np.sum(res ** 2) <= 1e-12 * sstot:
        return np.nan, np.nan, np.nan
    with np.errstate(all='ignore'):
        ac = np.asarray(autocorr(res, [1, 2], 'Fourier'), dtype=float).ravel()
    return float(ac[0]), float(ac[1]), bf_runs_z(res)


# ------------------------------------------------------------------------------
# BF_TheilSen
# ------------------------------------------------------------------------------
def bf_theil_sen(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Theil-Sen robust straight-line fit (hctsa ``BF_TheilSen``).

    The slope is the median of the slopes of the lines through all pairs of points with
    different ``x``; the intercept is the median of ``y - slope*x``. (This is not
    ``scipy.stats.theilslopes``, whose default intercept differs.)

    Returns
    -------
    numpy.ndarray
        ``[slope, intercept]``, in the order of ``np.polyfit(x, y, 1)``; both NaN if
        there are fewer than two distinct ``x`` values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ii, jj = np.triu_indices(x.size, 1)
    dx = x[jj] - x[ii]
    good = dx != 0
    if not good.any():
        return np.array([np.nan, np.nan])
    slope = np.median((y[jj[good]] - y[ii[good]]) / dx[good])
    return np.array([slope, np.median(y - slope * x)])


# ------------------------------------------------------------------------------
# BF_ExpFit
# ------------------------------------------------------------------------------
def _sse_of_rate(bs: np.ndarray, x: np.ndarray, y: np.ndarray, with_offset: bool) -> np.ndarray:
    """Sum of squared errors of the best-fitting a (and c) at each rate in ``bs``."""
    with np.errstate(all='ignore'):
        e = np.exp(np.outer(x, bs))  # each column is exp(b*x) for one rate
        if with_offset:
            ec = e - e.mean(axis=0)
            yc = y - y.mean()
            den = np.sum(ec ** 2, axis=0)
            sse = np.sum(yc ** 2) - (ec.T @ yc) ** 2 / den
        else:
            den = np.sum(e ** 2, axis=0)
            sse = np.sum(y ** 2) - (e.T @ y) ** 2 / den
        sse = np.where(~(den > 0) | ~np.isfinite(sse), np.inf, sse)
    return sse


def bf_exp_fit(x: ArrayLike, y: ArrayLike, with_offset: bool = True, max_rate: float = 20) -> dict:
    """Global least-squares fit of ``a*exp(b*x) + c`` by variable projection (hctsa ``BF_ExpFit``).

    For a given rate ``b`` the best ``a`` (and ``c``) follow in closed form, so only
    ``b`` is searched: on a grid of 401 rates between ``-max_rate/range(x)`` and
    ``max_rate/range(x)`` (including 0, a constant), then refined by 40 golden-section
    steps around the best grid point. The result is the global optimum within the
    allowed range of rates and needs no starting point. If the best rate is at the edge
    of the range, ``b`` is that limiting value.

    Parameters
    ----------
    x, y : array-like
        Predictor and data (same length).
    with_offset : bool, optional
        Fit ``a*exp(b*x) + c`` (default) or ``a*exp(b*x)``.
    max_rate : float, optional
        Largest allowed ``|b|`` in units of ``1/range(x)`` (default 20).

    Returns
    -------
    dict
        ``a, b, c`` (``c = 0`` without offset), ``r2`` (in [0,1]), ``adjr2``, ``rmse``
        (``sqrt(SSE/(n-p))``, p = 3 or 2). All NaN if ``y`` is constant or not finite, or
        if there are not more points than parameters.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    num_params = 2 + int(with_offset)
    yc = y - y.mean()
    sst = np.sum(yc ** 2)
    x_range = np.max(x) - np.min(x) if n else 0.0
    if n <= num_params or not (sst > 0) or not np.all(np.isfinite(y)) or x_range == 0:
        return dict(a=np.nan, b=np.nan, c=np.nan, r2=np.nan, adjr2=np.nan, rmse=np.nan)

    b_max = max_rate / x_range
    b_grid = _linspace(-b_max, b_max, 401)
    k = int(np.argmin(_sse_of_rate(b_grid, x, y, with_offset)))
    lo = b_grid[max(k - 1, 0)]
    hi = b_grid[min(k + 1, b_grid.size - 1)]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):  # golden-section search
        b1 = hi - phi * (hi - lo)
        b2 = lo + phi * (hi - lo)
        if _sse_of_rate(np.array([b1]), x, y, with_offset)[0] <= _sse_of_rate(np.array([b2]), x, y, with_offset)[0]:
            hi = b2
        else:
            lo = b1
    b = (lo + hi) / 2

    e = np.exp(b * x)
    if with_offset:
        a = np.sum((e - e.mean()) * yc) / np.sum((e - e.mean()) ** 2)
        c = y.mean() - a * e.mean()
    else:
        a = np.sum(e * y) / np.sum(e ** 2)
        c = 0.0
    sse = np.sum((y - a * e - c) ** 2)
    r2 = min(max(1 - sse / sst, 0.0), 1.0)
    return dict(a=a, b=b, c=c, r2=r2,
                adjr2=1 - (1 - r2) * (n - 1) / (n - num_params),
                rmse=np.sqrt(sse / (n - num_params)))


# ------------------------------------------------------------------------------
# BF_GaussMix2
# ------------------------------------------------------------------------------
def bf_gauss_mix2(xc: ArrayLike, p: ArrayLike, delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixture of two Gaussians fitted to a binned distribution by EM (hctsa ``BF_GaussMix2``).

    Fits ``w1 N(mu1, s1^2) + w2 N(mu2, s2^2)`` to a histogram/density ``p`` on the
    equally spaced grid ``xc``, by EM on the grid points weighted by the mass ``p``
    (maximum likelihood for the binned data). Each component's variance includes
    ``delta^2/12``, the variance within a bin. The start is deterministic: means at the
    lower and upper quartiles, weights 1/2, standard deviations that of the
    distribution. Runs until the log-likelihood changes by less than 1e-10 (at most 1000
    iterations).

    Parameters
    ----------
    xc : array-like
        Grid points (e.g. bin centers).
    p : array-like
        Density or counts at each point (non-negative; rescaled to sum to 1).
    delta : float
        Grid spacing.

    Returns
    -------
    w, mu, sigma : numpy.ndarray
        Length-2 arrays, components ordered by increasing mean.
    """
    xc = np.asarray(xc, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    p = p / np.sum(p)
    v0 = delta ** 2 / 12

    cp = np.cumsum(p)
    mu = np.array([xc[np.argmax(cp >= 0.25)], xc[np.argmax(cp >= 0.75)]])
    m_bar = np.sum(p * xc)
    s_bar = np.sqrt(np.sum(p * (xc - m_bar) ** 2) + v0)
    if mu[0] == mu[1]:  # both quartiles in one bin
        mu = m_bar + s_bar * np.array([-0.5, 0.5])
    w = np.array([0.5, 0.5])
    sigma = s_bar * np.array([1.0, 1.0])

    ll_old = -np.inf
    with np.errstate(all='ignore'):
        for _ in range(1000):
            # E step: responsibility of component 1 at each grid point (and the log-likelihood)
            l1 = np.log(w[0]) - np.log(sigma[0]) - (xc - mu[0]) ** 2 / (2 * sigma[0] ** 2)
            l2 = np.log(w[1]) - np.log(sigma[1]) - (xc - mu[1]) ** 2 / (2 * sigma[1] ** 2)
            lmax = np.maximum(l1, l2)
            ll = np.sum(p * (lmax + np.log(np.exp(l1 - lmax) + np.exp(l2 - lmax))))
            r1 = 1 / (1 + np.exp(l2 - l1))
            if abs(ll - ll_old) < 1e-10:
                break
            ll_old = ll

            # M step
            n1 = np.sum(p * r1)
            n2 = 1 - n1
            if n1 < 1e-8 or n2 < 1e-8:  # one component has vanished
                break
            w = np.array([n1, n2])
            mu = np.array([np.sum(p * r1 * xc) / n1, np.sum(p * (1 - r1) * xc) / n2])
            sigma = np.sqrt(np.array([np.sum(p * r1 * (xc - mu[0]) ** 2) / n1,
                                      np.sum(p * (1 - r1) * (xc - mu[1]) ** 2) / n2]) + v0)

    ix = np.argsort(mu, kind='stable')
    return w[ix], mu[ix], sigma[ix]


# ------------------------------------------------------------------------------
# BF_FitDensityCurve
# ------------------------------------------------------------------------------
def _res_exp(th, t, p):
    """Residual (curve minus p) and Jacobian for ``a*exp(b*t)``, ``th = [a, b]``."""
    e = np.exp(th[1] * t)
    return th[0] * e - p, np.column_stack([e, th[0] * e * t])


def _res_gauss(th, x, p):
    """Residual and Jacobian for ``a*exp(-(x-m)^2/(2 s^2))``, ``th = [a, m, log s]``."""
    s = np.exp(th[2])
    g = np.exp(-(x - th[1]) ** 2 / (2 * s ** 2))
    f = th[0] * g
    return f - p, np.column_stack([g, f * (x - th[1]) / s ** 2, f * (x - th[1]) ** 2 / s ** 2])


def _res_gauss2(th, x, p):
    """Residual and Jacobian for a sum of two Gaussians, ``th = [a1 m1 log(s1) a2 m2 log(s2)]``."""
    r1, j1 = _res_gauss(th[:3], x, 0 * p)
    r2, j2 = _res_gauss(th[3:], x, 0 * p)
    return r1 + r2 - p, np.column_stack([j1, j2])


def _levenberg_marquardt(res_fun, th):
    """Minimize the sum of squares of ``res_fun(th) -> (r, J)`` from the start ``th``.

    Damping starts at 1e-3, divided by 3 after a step that lowers the sum of squares
    and multiplied by 3 after one that does not; stops when the sum of squares changes
    by a relative 1e-14 or after 200 iterations. Returns ``(th, cost)``.
    """
    th = np.asarray(th, dtype=float)
    lam = 1e-3
    with np.errstate(all='ignore'):
        r, jac = res_fun(th)
        cost = r @ r
        for _ in range(200):
            a = jac.T @ jac
            g = jac.T @ r
            try:
                step = -np.linalg.solve(a + lam * np.diag(np.diag(a)) + 1e-300 * np.eye(th.size), g)
            except np.linalg.LinAlgError:
                step = np.full(th.size, np.nan)  # singular: treated as a failed step
            r_new, j_new = res_fun(th + step)
            cost_new = r_new @ r_new
            if np.isfinite(cost_new) and cost_new < cost:
                th = th + step
                change = cost - cost_new
                r, jac, cost = r_new, j_new, cost_new
                lam = max(lam / 3, 1e-12)
                if change <= 1e-14 * cost:
                    break
            else:
                lam = 3 * lam
                if lam > 1e12:
                    break
    return th, cost


def bf_fit_density_curve(x: ArrayLike, p: ArrayLike, model: str) -> np.ndarray:
    """Deterministic least-squares fit of a simple curve to a density (hctsa ``BF_FitDensityCurve``).

    Fits a Gaussian, a sum of two Gaussians, an exponential or a power law to a density
    (or histogram) ``p`` at the points ``x`` by minimizing the sum of squared differences.
    No random starts and no toolbox optimizer: each model starts at a fixed point
    computed from the data, and a Levenberg-Marquardt iteration with analytic Jacobians
    descends to the nearest minimum.

    Models: ``'exp'`` (``a*exp(b*t)``, ``t = (x - mean(x))/std(x)``, started from the
    ``p^2``-weighted line of ``log p`` against ``t``), ``'power'`` (``t = log(x/mean(x))``,
    positive ``x``), ``'gauss'`` (the better of two starts: the distribution's mean and
    standard deviation; its peak and half the standard deviation) and ``'gauss2'`` (started
    from :func:`bf_gauss_mix2`; equally spaced ``x``).

    Parameters
    ----------
    x : array-like
        The positions (e.g. bin centers).
    p : array-like
        The density at each position.
    model : {'gauss', 'gauss2', 'exp', 'power'}

    Returns
    -------
    numpy.ndarray
        The fitted curve at each ``x``.
    """
    x = np.asarray(x, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    m_p = np.sum(p * x) / np.sum(p)  # mean and standard deviation of the distribution
    s_p = np.sqrt(np.sum(p * (x - m_p) ** 2) / np.sum(p))
    if s_p == 0:
        s_p = 1.0  # all mass in one bin

    if model in ('exp', 'power'):
        if model == 'power':
            t = np.log(x / np.mean(x))  # a power law is an exponential in log(x)
        else:
            sx = np.std(x, ddof=1)
            t = (x - np.mean(x)) / (sx + (sx == 0))
        # start: weighted least squares line for log(p) over the non-empty bins
        ok = p > 0
        w = p[ok]
        z = np.column_stack([np.ones(ok.sum()), t[ok]])
        th0 = None
        if ok.sum() >= 2:
            with np.errstate(all='ignore'):
                th0 = np.linalg.lstsq(z * w[:, None], np.log(p[ok]) * w, rcond=None)[0]  # [log(a), b]
        if th0 is None or not np.all(np.isfinite(th0)):
            th0 = np.array([np.log(np.max(p)), 0.0])
        th, _ = _levenberg_marquardt(lambda th: _res_exp(th, t, p), [np.exp(th0[0]), th0[1]])
        return p + _res_exp(th, t, p)[0]

    if model == 'gauss':
        starts = [[np.max(p), m_p, np.log(s_p)],
                  [np.max(p), x[np.argmax(p)], np.log(s_p / 2)]]
        cost = np.inf
        th = None
        for s0 in starts:
            th_i, c_i = _levenberg_marquardt(lambda th: _res_gauss(th, x, p), s0)
            if c_i < cost:
                th, cost = th_i, c_i
        if th is None:  # (all costs non-finite: keep the first start, as MATLAB would error)
            th = np.asarray(starts[0])
        return p + _res_gauss(th, x, p)[0]

    if model == 'gauss2':
        mix_w, mix_mu, mix_sig = bf_gauss_mix2(x, p, x[1] - x[0])
        th0 = [mix_w[0] / (mix_sig[0] * np.sqrt(2 * np.pi)), mix_mu[0], np.log(mix_sig[0]),
               mix_w[1] / (mix_sig[1] * np.sqrt(2 * np.pi)), mix_mu[1], np.log(mix_sig[1])]
        th, _ = _levenberg_marquardt(lambda th: _res_gauss2(th, x, p), th0)
        return p + _res_gauss2(th, x, p)[0]

    raise ValueError(f"Unknown model '{model}'")


# ------------------------------------------------------------------------------
# BF_FitSinusoids
# ------------------------------------------------------------------------------
def _sin_design(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Sine and cosine columns at each frequency in ``f``."""
    x = np.empty((t.size, 2 * len(f)))
    for i, fi in enumerate(f):
        x[:, 2 * i] = np.sin(2 * np.pi * fi * t)
        x[:, 2 * i + 1] = np.cos(2 * np.pi * fi * t)
    return x


def _sin_basis(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the sinusoids at frequencies ``f``."""
    if len(f) == 0:
        return np.zeros((t.size, 0))
    return np.linalg.qr(_sin_design(f, t), mode='reduced')[0]


def _sin_rss(y: np.ndarray, q: np.ndarray, fc: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Residual sum of squares after fitting ``y`` with the span of ``q`` and a sine and
    cosine pair at each candidate frequency in ``fc`` (vectorized over blocks)."""
    n = t.size
    r = y - q @ (q.T @ y)
    rss = np.full(fc.size, r @ r)
    block = max(1, int(np.floor(2e6 / n)))
    for b0 in range(0, fc.size, block):
        sl = slice(b0, min(b0 + block, fc.size))
        arg = 2 * np.pi * np.outer(t, fc[sl])
        s = np.sin(arg)
        c = np.cos(arg)
        s = s - q @ (q.T @ s)  # remove what the existing sinusoids already span
        c = c - q @ (q.T @ c)
        ss = np.sum(s ** 2, axis=0)
        cc = np.sum(c ** 2, axis=0)
        sc = np.sum(s * c, axis=0)
        rs = s.T @ r
        rc = c.T @ r
        dt = ss * cc - sc ** 2
        ok = (ss > 1e-6 * n) & (cc > 1e-6 * n) & (dt > 1e-8 * ss * cc)  # skip candidates in the span of q
        gain = np.zeros(ss.size)
        gain[ok] = (rs[ok] ** 2 * cc[ok] - 2 * rs[ok] * rc[ok] * sc[ok] + rc[ok] ** 2 * ss[ok]) / dt[ok]
        rss[sl] = rss[sl] - gain
    return rss


def bf_fit_sinusoids(y: ArrayLike, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of a sum of ``k`` sinusoids to a time series (hctsa ``BF_FitSinusoids``).

    Fits ``y(t) = sum_i a_i sin(2 pi f_i t) + b_i cos(2 pi f_i t)``, ``t = 1..N``. The
    amplitudes and phases are linear, found by least squares for given frequencies, so
    only the ``k`` frequencies are searched (variable projection), deterministically:
    each frequency is added at the grid frequency (``2N`` points on
    ``[1/(2N), 1/2 - 1/(2N)]``) that most reduces the residual sum of squares given those
    already chosen, then each is refined twice in turn by 8 zooming local grids.

    Parameters
    ----------
    y : array-like
        The time series.
    k : int
        The number of sinusoids.

    Returns
    -------
    yfit : numpy.ndarray
        The fitted values.
    freqs : numpy.ndarray
        The ``k`` fitted frequencies in cycles per sample, in increasing order.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    k = int(k)
    t = np.arange(1, n + 1, dtype=float)
    f_lims = np.array([1, n - 1]) / (2 * n)  # allowed frequency range
    f_grid = _linspace(f_lims[0], f_lims[1], 2 * n)  # search grid, spacing ~ 1/(4N)

    # Greedy search: add the grid frequency that most reduces the residual sum of squares
    f = np.zeros(k)
    for j in range(k):
        q = _sin_basis(f[:j], t)
        f[j] = f_grid[int(np.argmin(_sin_rss(y, q, f_grid, t)))]

    # Refine each frequency in turn (the others held fixed) by zooming local grids
    h0 = f_grid[1] - f_grid[0]
    for _ in range(2):
        for j in range(k):
            q = _sin_basis(np.delete(f, j), t)
            h = h0
            for _zoom in range(8):
                fc = np.minimum(np.maximum(f[j] + h * _linspace(-1, 1, 21), f_lims[0]), f_lims[1])
                f[j] = fc[int(np.argmin(_sin_rss(y, q, fc, t)))]
                h = h / 5
    freqs = np.sort(f)

    # Amplitudes and phases by least squares at the final frequencies
    x = _sin_design(freqs, t)
    yfit = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return yfit, freqs


# ------------------------------------------------------------------------------
# BF_KSDensity
# ------------------------------------------------------------------------------
def bf_ks_density(y: ArrayLike, xi: Optional[ArrayLike] = None,
                  h: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Gaussian kernel density estimate with an explicit bandwidth (hctsa ``BF_KSDensity``).

    Evaluates ``f(x) = mean_i N(x; y_i, h^2)`` exactly (no binning, no truncation of the
    kernel). The default bandwidth is Silverman's rule with a robust scale,
    ``h = s*(4/(3n))^(1/5)``, ``s = median(|y - median(y)|)/0.6745`` (``s = std(y)`` if the
    median absolute deviation is zero; if that too is zero, ``h``, ``xi`` and ``f`` are NaN).

    Parameters
    ----------
    y : array-like
        The data (NaNs are ignored).
    xi : array-like, optional
        Evaluation points (default: 100 equally spaced points from ``min(y) - 3h`` to
        ``max(y) + 3h``).
    h : float, optional
        Kernel bandwidth (standard deviation of the Gaussian kernel).

    Returns
    -------
    f : numpy.ndarray
        The density estimate at ``xi``.
    xi : numpy.ndarray
        The evaluation points.
    h : float
        The bandwidth used.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    n = y.size

    if h is None:
        s = np.median(np.abs(y - np.median(y))) / 0.6745  # robust estimate of the standard deviation
        if s <= 0:
            s = np.std(y, ddof=1) if n > 1 else 0.0
        h = np.nan if s <= 0 else s * (4 / (3 * n)) ** (1 / 5)

    if xi is None:
        xi = _linspace(np.min(y) - 3 * h, np.max(y) + 3 * h, 100)
    xi = np.asarray(xi, dtype=float).ravel()

    # Sum the Gaussian kernels (in blocks of evaluation points, to limit memory)
    f = np.zeros(xi.size)
    block = max(1, int(np.floor(2e6 / n)))
    with np.errstate(all='ignore'):
        for i in range(0, xi.size, block):
            ix = slice(i, min(i + block, xi.size))
            f[ix] = np.sum(np.exp(-0.5 * ((xi[ix][None, :] - y[:, None]) / h) ** 2), axis=0) / (n * h * np.sqrt(2 * np.pi))
    return f, xi, float(h)


# ------------------------------------------------------------------------------
# BF_HistEdges, BF_QuantileEdges
# ------------------------------------------------------------------------------
def bf_hist_edges(y: ArrayLike, bin_rule: Union[int, str] = 'auto',
                  limits: Optional[ArrayLike] = None) -> np.ndarray:
    """Equal-width histogram bin edges from an explicit bin-count rule (hctsa ``BF_HistEdges``).

    The bins span ``[min(y), max(y)]`` (or ``limits``) with width ``(max - min)/num_bins``.
    The interior edges are lowered, and the end edges widened, by ``1e-6`` of a bin
    width, so that values exactly on an edge of the ideal grid (lattice-valued data)
    always fall just above it, in the upper bin. Pass the result to ``np.histogram`` /
    ``np.searchsorted`` (right-open bins).

    Parameters
    ----------
    y : array-like
        The data (NaNs are ignored).
    bin_rule : int or str, optional
        The number of bins, or a rule for ``n`` values: ``'sqrt'`` (``ceil(sqrt(n))``),
        ``'sturges'`` (``ceil(log2(n) + 1)``), ``'fd'`` (Freedman-Diaconis,
        ``ceil(range/(2 IQR n^(-1/3)))``, Sturges if IQR = 0) or ``'auto'`` (the larger of
        the two; default). A rule gives at most ``n`` and at least 1 bins.
    limits : array-like, optional
        ``[lower, upper]``, the interval to span instead of the range of the data.

    Returns
    -------
    numpy.ndarray
        ``num_bins + 1`` increasing edges.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    n = y.size
    data_range = np.max(y) - np.min(y)
    if limits is not None and np.size(limits) > 0:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        lo, hi = float(np.min(y)), float(np.max(y))

    if isinstance(bin_rule, str):
        num_sturges = np.ceil(np.log2(n) + 1)
        if data_range > 0:
            q = matlab_quantile(y, [0.25, 0.75])
            fd_width = 2 * (q[1] - q[0]) * n ** (-1 / 3)
        else:
            fd_width = 0.0
        num_fd = np.ceil(data_range / fd_width) if fd_width > 0 else num_sturges
        if bin_rule == 'sqrt':
            num_bins = np.ceil(np.sqrt(n))
        elif bin_rule == 'sturges':
            num_bins = num_sturges
        elif bin_rule == 'fd':
            num_bins = num_fd
        elif bin_rule == 'auto':
            num_bins = max(num_sturges, num_fd)
        else:
            raise ValueError(f"Unknown bin rule '{bin_rule}'")
        num_bins = max(1, min(num_bins, n))
    else:
        num_bins = bin_rule
    num_bins = int(num_bins)

    if hi == lo:  # constant data: one bin of unit width
        return lo + np.array([-0.5, 0.5])
    bin_width = (hi - lo) / num_bins
    tol = 1e-6 * bin_width
    edges = lo + np.arange(num_bins + 1) * bin_width - tol  # interior edges, lowered by tol
    edges[0] = lo - tol
    edges[-1] = hi + tol
    return edges


def bf_quantile_edges(y: ArrayLike, num_bins: int) -> np.ndarray:
    """Histogram bin edges at the quantiles of the data: equiprobable bins (hctsa ``BF_QuantileEdges``).

    The quantiles of ``y`` (MATLAB ``quantile`` convention) at ``0, 1/num_bins, ..., 1``,
    with repeated quantiles merged (fewer bins for tied values). All but the last edge
    are lowered, and the last widened, by ``1e-9`` of the range of the data, so that a
    value on an edge falls just above it, in the upper bin.

    Parameters
    ----------
    y : array-like
        The data (NaNs are ignored).
    num_bins : int
        The number of bins.

    Returns
    -------
    numpy.ndarray
        Increasing edges (a single value +/- 0.5 for constant data).
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    edges = np.unique(matlab_quantile(y, _linspace(0, 1, int(num_bins) + 1)))
    if edges.size == 1:  # constant data: one bin of unit width
        return edges + np.array([-0.5, 0.5])
    tol = 1e-9 * (edges[-1] - edges[0])
    edges[:-1] = edges[:-1] - tol  # lower all but the last edge,
    edges[-1] = edges[-1] + tol  # and widen the last
    return edges


# ------------------------------------------------------------------------------
# BF_HalfSampleMode
# ------------------------------------------------------------------------------
def bf_half_sample_mode(y: ArrayLike) -> float:
    """Half-sample mode: a robust, bin-free estimate of the mode (hctsa ``BF_HalfSampleMode``).

    Repeatedly keeps the ``ceil(n/2)`` consecutive sorted values that span the shortest
    interval (the first, lowest, on ties) until three or fewer values remain; the mode
    is then the mean of the two closest of them (the middle one if equally close).

    Reference: Bickel and Fruhwirth, Comput. Stat. Data Anal. 50(12), 3500 (2006).

    Parameters
    ----------
    y : array-like
        The data (NaNs are ignored).

    Returns
    -------
    float
        The estimated mode.
    """
    y = np.sort(np.asarray(y, dtype=float).ravel())
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.nan
    while y.size > 3:
        h = -(-y.size // 2)  # ceil(n/2): the number of values in a half-sample
        i = int(np.argmin(y[h - 1:] - y[:y.size - h + 1]))  # the shortest window of h consecutive values
        y = y[i:i + h]
    if y.size == 3:
        d1, d2 = y[1] - y[0], y[2] - y[1]
        if d1 < d2:
            return float(np.mean(y[:2]))
        if d1 > d2:
            return float(np.mean(y[1:]))
        return float(y[1])
    return float(np.mean(y))


# ------------------------------------------------------------------------------
# BF_RemovePoints
# ------------------------------------------------------------------------------
def bf_remove_points(y: ArrayLike, remove_how: str = 'absfar', p: float = 0.1,
                     remove_or_saturate: str = 'remove',
                     random_seed: Union[int, str, None] = None) -> np.ndarray:
    """
    Remove or saturate a proportion of the points of a time series (hctsa's ``BF_RemovePoints``).

    Chooses a proportion, ``p``, of the points of the (z-scored) series according to a rule, and
    either deletes them or clips their values. Removing deletes the chosen points and closes up
    the rest into a shorter series (in the original order). Saturating keeps them in place but
    clips their values to the most extreme value among the points kept. Used by hctsa's
    ``DN_RemovePoints`` (order-free statistics of the changed series) and ``CO_RemovePoints``
    (autocorrelation statistics of the changed series).

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored).
    remove_how : {'absclose', 'absfar', 'min', 'max', 'random'}, optional
        How to choose the points to remove:

        - 'absclose': those closest to the mean,
        - 'absfar': those furthest from the mean (default),
        - 'min': the lowest values,
        - 'max': the highest values,
        - 'random': at random.
    p : float, optional
        The proportion of points to remove. Default 0.1.
    remove_or_saturate : {'remove', 'saturate'}, optional
        Whether to remove the points (default) or to saturate their values ('saturate'; possible
        for 'absfar', 'min' and 'max' only).
    random_seed : int, 'default' or 'none', optional
        Only relevant for ``remove_how='random'``: the seed of the random ordering (see
        :func:`bf_random_seed`; ``None`` or ``'default'`` is seed 0, so the result is reproducible).
        The ordering is ``bf_random(N, seed, 'perm')``, the portable generator, so it matches hctsa's.

    Returns
    -------
    numpy.ndarray
        The series after removing (a shorter series) or saturating (the original length) the
        chosen points.

    Raises
    ------
    ValueError
        For an unknown ``remove_how`` or ``remove_or_saturate``, or saturating with a method
        that cannot be saturated.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    # Order the values by the criterion, so that the points to *keep* come first
    if remove_how == 'absclose':
        order = np.argsort(-np.abs(y), kind='stable')   # (MATLAB sort(...,'descend') is stable)
    elif remove_how == 'absfar':
        order = np.argsort(np.abs(y), kind='stable')
    elif remove_how == 'min':
        order = np.argsort(-y, kind='stable')
    elif remove_how == 'max':
        order = np.argsort(y, kind='stable')
    elif remove_how == 'random':
        order = bf_random(n, bf_random_seed(random_seed), 'perm') - 1
    else:
        raise ValueError(f"Unknown method '{remove_how}'")

    # Points to keep: round(N*(1 - p)) of them, in the original order
    n_keep = int(_round_half_away(n * (1 - p)))
    keep = np.sort(order[:n_keep])

    if remove_or_saturate == 'remove':
        return y[keep]
    if remove_or_saturate == 'saturate':
        y_t = y.copy()
        if remove_how in ('max', 'min', 'absfar'):
            kept = y[keep]
            if kept.size == 0:
                return y_t  # (MATLAB would error on max([]) assignment; nothing to clip to)
            if remove_how == 'max':
                y_t[np.setdiff1d(np.arange(n), keep)] = np.max(kept)
            elif remove_how == 'min':
                y_t[np.setdiff1d(np.arange(n), keep)] = np.min(kept)
            else:
                hi, lo = np.max(kept), np.min(kept)
                y_t[y_t > hi] = hi
                y_t[y_t < lo] = lo
            return y_t
        raise ValueError(f"Cannot 'saturate' when using '{remove_how}' method")
    raise ValueError(f"Unknown remove_or_saturate option: '{remove_or_saturate}'")
