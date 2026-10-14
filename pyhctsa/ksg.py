"""
Nonparametric and Gaussian mutual-information estimators.

:func:`ksg_mi` is the k-nearest-neighbor estimator of Kraskov, Stoegbauer and Grassberger
(KSG), algorithms 1 and 2 [1]_. :func:`gaussian_mi` is the closed-form estimator for jointly
Gaussian variables.

Both are written directly from the published definitions; the Java Information Dynamics
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

References
----------
.. [1] A. Kraskov, H. Stoegbauer, P. Grassberger, "Estimating mutual information",
   Phys. Rev. E 69, 066138 (2004).
.. [2] J. T. Lizier, "JIDT: an information-theoretic toolkit for studying the dynamics of
   complex systems", Front. Robot. AI 1, 11 (2014).
"""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.special import digamma


# ----------------------------------------------------------------------------------------------
# Neighbor search and counting
# ----------------------------------------------------------------------------------------------

def _joint_neighbors(z, k, w):
    """Distances and indices of the k nearest max-norm neighbors of each row of ``z``.

    Rows whose index differs by at most ``w`` from the query row (the row itself included) are
    not candidates. Rather than searching with exclusions, this asks the k-d tree for the
    ``k + 2w + 1`` nearest rows, of which at most ``2w + 1`` fall in the window, and keeps the
    first k that do not.
    """
    n = z.shape[0]
    dist, idx = cKDTree(z).query(z, k=k + 2 * w + 1, p=np.inf)
    keep = np.abs(idx - np.arange(n)[:, None]) > w
    keep &= np.cumsum(keep, axis=1) <= k
    return dist[keep].reshape(n, k), idx[keep].reshape(n, k)


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
    sd = a.std(axis=0, ddof=1)
    sd[sd == 0.0] = 1.0
    return (a - a.mean(axis=0)) / sd


def ksg_mi(x, y, k=4, algorithm=1, theiler=0):
    """Kraskov-Stoegbauer-Grassberger estimate of the mutual information I(x; y), in nats.

    Each column of ``x`` and of ``y`` is first standardized to zero mean and unit variance, so
    that the max norm weighs all dimensions equally. The caller is responsible for breaking
    exact ties (repeated values) in the data, e.g. with tiny noise, as the estimator is
    ill-defined for coincident points.

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
    dist, neighbors = _joint_neighbors(np.hstack([x, y]), k, theiler)

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


def gaussian_mi(x, y):
    """Mutual information I(x; y) in nats under a joint Gaussian model.

    ``I = (ln det C_x + ln det C_y - ln det C_xy) / 2``, with ``C`` the sample covariance
    matrices of ``x``, ``y`` and of the stacked ``(x, y)``. For scalars this reduces to
    ``-ln(1 - r**2) / 2`` for the Pearson correlation ``r``. Only linear dependence
    contributes.

    Parameters
    ----------
    x, y : array-like, shape (n_samples,) or (n_samples, n_dims)

    Returns
    -------
    float
        The mutual information, in nats. It is unbounded as ``x`` and ``y`` approach exact linear
        dependence (a very large, rounding-limited value, or ``inf`` if the joint covariance
        is numerically singular), and ``nan`` if ``x`` or ``y`` alone has linearly dependent
        columns (e.g. is constant).
    """
    x = _as_columns(x, "x")
    y = _as_columns(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    def logdet(a):
        sign, value = np.linalg.slogdet(np.atleast_2d(np.cov(a, rowvar=False)))
        return value if sign > 0 else None

    ld_x, ld_y, ld_xy = logdet(x), logdet(y), logdet(np.hstack([x, y]))
    if ld_x is None or ld_y is None:
        return float("nan")
    if ld_xy is None:
        return float("inf")
    return float(0.5 * (ld_x + ld_y - ld_xy))
