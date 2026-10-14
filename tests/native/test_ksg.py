"""Accuracy tests for the KSG and Gaussian mutual-information estimators (pyhctsa.ksg).

References are analytic mutual informations of Gaussians and a brute-force O(N^2) implementation
of the KSG definitions; none depends on JIDT or MATLAB.
"""
import numpy as np
import pytest
from scipy.special import digamma

from pyhctsa.ksg import gaussian_mi, ksg_mi


def _reference_ksg(x, y, k, algorithm, w=0):
    """KSG estimate by exhaustive search, straight from the definitions (after standardizing)."""
    def standardize(a):
        a = np.asarray(a, float)
        a = a[:, None] if a.ndim == 1 else a
        return (a - a.mean(0)) / a.std(0, ddof=1)

    x, y = standardize(x), standardize(y)
    n = len(x)
    total = 0.0
    for i in range(n):
        dx = np.abs(x - x[i]).max(axis=1)
        dy = np.abs(y - y[i]).max(axis=1)
        allowed = np.abs(np.arange(n) - i) > w
        joint = np.maximum(dx, dy)
        cand = np.flatnonzero(allowed)
        nb = cand[np.argsort(joint[cand], kind="stable")[:k]]
        if algorithm == 1:
            eps = joint[nb[-1]]
            n_x = np.sum(allowed & (dx < eps))
            n_y = np.sum(allowed & (dy < eps))
            total += digamma(n_x + 1) + digamma(n_y + 1)
        else:
            n_x = np.sum(allowed & (dx <= dx[nb].max()))
            n_y = np.sum(allowed & (dy <= dy[nb].max()))
            total += digamma(n_x) + digamma(n_y)
    shift = digamma(k) + digamma(n) - (0.0 if algorithm == 1 else 1.0 / k)
    return shift - total / n


def _gaussian_pair(n, rho, rng):
    x = rng.standard_normal(n)
    return x, rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)


# ------------------------------------------------------------------------------------ KSG


@pytest.mark.parametrize("algorithm", [1, 2])
@pytest.mark.parametrize("theiler", [0, 3])
@pytest.mark.parametrize("dims", [(1, 1), (2, 1), (1, 2), (2, 3)])
def test_ksg_matches_exhaustive_search(algorithm, theiler, dims):
    rng = np.random.default_rng(7)
    n = 150
    x = rng.standard_normal((n, dims[0]))
    y = 0.5 * x[:, [0] * dims[1]] + rng.standard_normal((n, dims[1]))
    for k in (1, 4):
        assert ksg_mi(x, y, k, algorithm, theiler) == pytest.approx(
            _reference_ksg(x, y, k, algorithm, theiler), abs=1e-12)


def test_ksg_algorithm1_strict_inequality_on_lattice_data():
    # many exact ties in the marginals: the algorithm-1 counts must be strict (< eps)
    rng = np.random.default_rng(3)
    x = rng.integers(0, 6, 120).astype(float)
    y = (x + rng.integers(0, 3, 120)).astype(float)
    assert ksg_mi(x, y, 3, 1) == pytest.approx(_reference_ksg(x, y, 3, 1), abs=1e-12)


@pytest.mark.parametrize("algorithm", [1, 2])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("k", [3, 4, 10])
def test_ksg_recovers_gaussian_mi(algorithm, rho, k):
    exact = -0.5 * np.log(1 - rho**2)
    est = np.mean([ksg_mi(*_gaussian_pair(3000, rho, np.random.default_rng(s)), k, algorithm)
                   for s in range(4)])
    assert est == pytest.approx(exact, abs=0.02)


@pytest.mark.parametrize("algorithm", [1, 2])
def test_ksg_error_shrinks_with_sample_size(algorithm):
    rho = 0.8
    exact = -0.5 * np.log(1 - rho**2)

    def mean_abs_error(n):
        return np.mean([abs(ksg_mi(*_gaussian_pair(n, rho, np.random.default_rng(s)),
                                   4, algorithm) - exact) for s in range(8)])

    assert mean_abs_error(4000) < 0.5 * mean_abs_error(100)


@pytest.mark.parametrize("algorithm", [1, 2])
def test_ksg_independent_variables_give_zero(algorithm):
    est = [ksg_mi(np.random.default_rng(s).standard_normal(2000),
                  np.random.default_rng(100 + s).standard_normal(2000), 4, algorithm)
           for s in range(6)]
    assert abs(np.mean(est)) < 0.01


@pytest.mark.parametrize("algorithm", [1, 2])
def test_ksg_multivariate_gaussian(algorithm):
    # x is 2-D with correlated components; y depends on both
    rng = np.random.default_rng(11)
    n = 4000
    cov = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]])  # (x1, x2, y)
    z = rng.multivariate_normal(np.zeros(3), cov, size=n)
    exact = 0.5 * np.log(np.linalg.det(cov[:2, :2]) * cov[2, 2] / np.linalg.det(cov))
    assert ksg_mi(z[:, :2], z[:, 2], 4, algorithm) == pytest.approx(exact, abs=0.03)


def test_ksg_captures_nonlinear_dependence():
    # y = x^2 is uncorrelated with x, so the Gaussian estimate is ~0 but KSG is clearly positive
    x = np.random.default_rng(5).standard_normal(2000)
    y = x**2 + 0.05 * np.random.default_rng(6).standard_normal(2000)
    assert gaussian_mi(x, y) < 0.01
    assert ksg_mi(x, y) > 0.5


@pytest.mark.parametrize("algorithm", [1, 2])
def test_ksg_invariant_to_affine_rescaling_and_symmetric(algorithm):
    x, y = _gaussian_pair(500, 0.7, np.random.default_rng(2))
    base = ksg_mi(x, y, 4, algorithm)
    assert ksg_mi(100 * x + 3, 0.01 * y - 5, 4, algorithm) == pytest.approx(base, abs=1e-12)
    assert ksg_mi(y, x, 4, algorithm) == pytest.approx(base, abs=1e-12)


def test_ksg_theiler_window_removes_serial_dependence_inflation():
    # For a slowly varying series, neighbors in time are neighbors in space; with no window the
    # lag-1 AMI of an AR(1) is therefore estimated from neighbors that are nearly the same
    # point. Excluding them must change the estimate.
    rng = np.random.default_rng(4)
    e = rng.standard_normal(1500)
    y = np.empty(1500)
    y[0] = e[0]
    for i in range(1, 1500):
        y[i] = 0.98 * y[i - 1] + e[i]
    plain = ksg_mi(y[:-1], y[1:], 4, 1, 0)
    windowed = ksg_mi(y[:-1], y[1:], 4, 1, 25)
    assert plain != pytest.approx(windowed, abs=1e-3)


def test_ksg_validates_input():
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="algorithm"):
        ksg_mi(x, x, algorithm=3)
    with pytest.raises(ValueError, match="k must"):
        ksg_mi(x, x, k=0)
    with pytest.raises(ValueError, match="theiler"):
        ksg_mi(x, x, theiler=-1)
    with pytest.raises(ValueError, match="same number"):
        ksg_mi(x, x[:-1])
    with pytest.raises(ValueError, match="Too few"):
        ksg_mi(x, x, k=4, theiler=3)
    with pytest.raises(ValueError, match="finite"):
        ksg_mi(np.append(x[:-1], np.nan), x)


def test_ksg_smallest_admissible_sample():
    # n = k + 2*theiler + 1: exactly k candidate neighbors remain for every sample
    rng = np.random.default_rng(8)
    x, y = rng.standard_normal(9), rng.standard_normal(9)
    assert ksg_mi(x, y, 4, 1, 2) == pytest.approx(_reference_ksg(x, y, 4, 1, 2), abs=1e-12)


# ------------------------------------------------------------------------------- Gaussian


def test_gaussian_mi_scalar_is_minus_half_log_one_minus_r_squared():
    x, y = _gaussian_pair(400, 0.6, np.random.default_rng(1))
    r = np.corrcoef(x, y)[0, 1]
    assert gaussian_mi(x, y) == pytest.approx(-0.5 * np.log(1 - r**2), rel=1e-10)


def test_gaussian_mi_multivariate_matches_population_value():
    cov = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]])
    z = np.random.default_rng(0).multivariate_normal(np.zeros(3), cov, size=200000)
    exact = 0.5 * np.log(np.linalg.det(cov[:2, :2]) * cov[2, 2] / np.linalg.det(cov))
    assert gaussian_mi(z[:, :2], z[:, 2]) == pytest.approx(exact, abs=5e-3)


def test_gaussian_mi_degenerate_cases():
    t = np.linspace(0, 1, 50)
    assert gaussian_mi(t, 3 * t + 1) > 10              # exactly linearly dependent: unbounded
    assert np.isnan(gaussian_mi(np.ones(50), t))        # constant variable
    assert gaussian_mi(t, np.random.default_rng(0).standard_normal(50)) >= 0
