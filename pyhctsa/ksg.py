"""
Nonparametric (KSG) mutual-information estimator.

:func:`ksg_mi` is the k-nearest-neighbor estimator of Kraskov, Stoegbauer and Grassberger
(KSG), algorithms 1 and 2 [1]_.

It is written directly from the published definitions; the Java Information Dynamics
Toolkit (JIDT; Lizier, 2014 [2]_), which hctsa uses for these estimators, served only as a
reference for the method conventions (which neighbors are counted, how the marginals are
scaled), not as a source of code.

The KSG estimates, for ``N`` samples of (possibly multivariate) ``x`` and ``y``, and with
``eps(i)`` the max-norm distance from sample ``i`` to its ``k``-th nearest neighbor in the
joint ``(x, y)`` space:

* algorithm 1: ``I = psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>``, where ``n_x`` (``n_y``)
  counts the samples whose ``x`` (``y``) coordinates are strictly closer than ``eps(i)`` to those
  of sample ``i``;
* algorithm 2: ``I = psi(k) - 1/k + psi(N) - <psi(n_x) + psi(n_y)>``, where ``eps_x(i)`` and
  ``eps_y(i)`` are the largest ``x`` and ``y`` separations among the ``k`` joint neighbors, and
  ``n_x`` (``n_y``) counts the samples within (at most) ``eps_x(i)`` (``eps_y(i)``).

The estimators assume continuous data. On data with repeated values, algorithm 2 is ill-defined
twice over: the ``k`` joint neighbors are not unique where several samples tie at the k-th
distance (which ones a k-d tree returns depends on its traversal, so on the platform), and every
sample sharing the boundary value counts towards ``n_x`` (``n_y``), which biases the estimate
down, often below zero. So, in algorithm 2, each column of ``x`` or ``y`` that has a repeated
value has its repeated entries made distinct first, with tiny Gaussian jitter (standard
deviation ``1e-10`` of the standardized column) from the portable fixed-seed stream
:func:`pyhctsa.robust.bf_random`: the ties are broken as the continuous data they approximate
would break them, and the result is deterministic and the same on every platform. Values that
occur once are not touched (distances can still tie without repeated values, e.g. on a regular
grid; those ties are left to the tree).

References
----------
.. [1] A. Kraskov, H. Stoegbauer, P. Grassberger, "Estimating mutual information",
   Phys. Rev. E 69, 066138 (2004).
.. [2] J. T. Lizier, "JIDT: an information-theoretic toolkit for studying the dynamics of
   complex systems", Front. Robot. AI 1, 11 (2014).
"""
import numpy as np
from .robust import bf_random
from .utils import _sample_std
from numba import njit
from scipy.special import digamma

from .utils import _knn_theiler


# ----------------------------------------------------------------------------------------------
# Neighbor search and counting
# ----------------------------------------------------------------------------------------------

def _max_separation(a, neighbors):
    """For each row i of ``a``: the largest max-norm separation from row i of its neighbors."""
    return np.abs(a[neighbors] - a[:, None, :]).max(axis=(1, 2))


@njit(cache=True)
def _count_within(m, radius, w, strict):
    """For each row i of ``m``: the number of other rows within max-norm distance
    ``radius[i]`` (``< radius`` if ``strict``, else ``<= radius``), ignoring rows whose index
    differs from i by at most ``w`` (which includes row i itself).

    The rows are sorted by their first column and searched outward from each row until the
    separation in that column reaches the radius: exact (the inequality is applied to the very
    differences the neighbor search used), and much faster than a tree for the one- or
    two-dimensional blocks used here.
    """
    n, d = m.shape
    order = np.argsort(m[:, 0])
    ms = m[order]
    counts = np.empty(n, np.int64)
    for p in range(n):
        i = order[p]
        r = radius[i]
        total = 0
        for direction in (1, -1):
            q = p + direction
            while 0 <= q < n:
                gap = (ms[q, 0] - ms[p, 0]) * direction
                if gap >= r if strict else gap > r:
                    break
                if abs(order[q] - i) > w:
                    inside = True
                    for c in range(1, d):
                        a = abs(ms[q, c] - ms[p, c])
                        if a >= r if strict else a > r:
                            inside = False
                            break
                    if inside:
                        total += 1
                q += direction
        counts[i] = total
    return counts


# ----------------------------------------------------------------------------------------------
# Public estimators
# ----------------------------------------------------------------------------------------------

def _as_columns(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be a vector or an (n_samples, n_dims) array")
    return a


def _standardize(a):
    """Zero mean and unit (sample) variance in each column; constant columns are only centered."""
    sd = _sample_std(a, axis=0)
    sd[sd == 0.0] = 1.0
    return (a - a.mean(axis=0)) / sd


# Standard deviation of the tie-breaking jitter of algorithm 2, relative to the standardized column
_TIE_JITTER = 1e-10


def _break_ties(a, first_seed):
    """``a`` (standardized columns) with its repeated values made distinct: in each non-constant
    column j, every entry whose value occurs more than once gets Gaussian jitter of standard
    deviation ``_TIE_JITTER``, entry i the i-th value of ``bf_random(n, first_seed + j,
    'normal')``, so the result is deterministic and portable. Jitter this small can land two
    entries on the same float again (a birthday collision at round-off resolution, in large tie
    blocks); the entries still repeated are then jittered again, with the stream of seed
    ``first_seed + j + 1000 * pass``, until none is. Entries whose values are distinct are not
    touched.
    """
    out = a
    n = a.shape[0]
    for j in range(a.shape[1]):
        if _sample_std(a[:, j]) == 0:
            continue
        col = a[:, j]
        for p in range(100):
            _, inverse, counts = np.unique(col, return_inverse=True, return_counts=True)
            repeated = counts[inverse.ravel()] > 1
            if not repeated.any():
                break
            if out is a:
                out = a.copy()
            col = out[:, j]
            col[repeated] += _TIE_JITTER * bf_random(n, first_seed + j + 1000 * p, 'normal')[repeated]
    return np.ascontiguousarray(out)


def ksg_mi(x, y, k=4, algorithm=1, theiler=0):
    """Kraskov-Stoegbauer-Grassberger estimate of the mutual information I(x; y), in nats.

    Each column of ``x`` and of ``y`` is first standardized to zero mean and unit variance, so
    that the max norm weighs all dimensions equally. The estimators assume continuous data. In
    algorithm 2, the repeated values of each column then get tiny, fixed-seed Gaussian jitter
    (standard deviation 1e-10, from :func:`pyhctsa.robust.bf_random`; seeds 0, 1, ... over the
    columns of ``x`` and then of ``y``), which breaks the ties deterministically and identically
    on every platform; see the module docstring. Values that occur once are not touched.
    Algorithm 1 is used as given: it depends on the joint neighbors only through the k-th
    distance, so it is deterministic on tied data, though biased by them.

    Parameters
    ----------
    x, y : array-like, shape (n_samples,) or (n_samples, n_dims)
        The two (possibly multivariate) variables; one row per joint observation.
    k : int, optional
        Number of nearest neighbors in the joint space. Default 4.
    algorithm : {1, 2}, optional
        KSG algorithm 1 (default) or 2.
    theiler : int, optional
        Theiler window ``w`` for serially correlated samples: samples whose row index differs
        by at most ``w`` from that of the sample being considered are neither its neighbors
        nor counted in its marginal neighborhoods. Default 0 (only the sample itself).

    Returns
    -------
    float
        The estimated mutual information, in nats.
    """
    if algorithm not in (1, 2):
        raise ValueError("algorithm must be 1 or 2")
    k = int(k)
    theiler = int(theiler)
    if k < 1:
        raise ValueError("k must be at least 1")
    if theiler < 0:
        raise ValueError("theiler must be non-negative")
    x = _as_columns(x, "x")
    y = _as_columns(y, "y")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of samples")
    if n <= k + 2 * theiler:
        raise ValueError(f"Too few samples ({n}) for k={k} and theiler={theiler}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite")

    x = np.ascontiguousarray(_standardize(x))
    y = np.ascontiguousarray(_standardize(y))
    if algorithm == 2:
        # break exact ties (repeated values) with fixed-seed jitter, a seed per column
        x = _break_ties(x, 0)
        y = _break_ties(y, x.shape[1])
    # the k nearest max-norm neighbors of each joint point outside the Theiler window
    dist, neighbors = _knn_theiler(np.hstack([x, y]), k, theiler, metric='chebyshev')

    if algorithm == 1:
        eps = dist[:, -1]
        n_x = _count_within(x, eps, theiler, True)
        n_y = _count_within(y, eps, theiler, True)
        mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    else:
        n_x = _count_within(x, _max_separation(x, neighbors), theiler, False)
        n_y = _count_within(y, _max_separation(y, neighbors), theiler, False)
        mi = digamma(k) - 1.0 / k + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return float(mi)
