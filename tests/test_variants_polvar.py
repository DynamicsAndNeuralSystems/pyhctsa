"""pyhctsa-only variant pol_var_resid (of pol_var).

Exact values on small cases, NaN (never an error) on degenerate or short input, and fast specificity checks: each
variant is null on a nuisance that the original responds to, and still responds to its target.
"""
import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.operations.medical import pol_var, pol_var_resid
from pyhctsa.utils import z_score


def _auc(a, b):
    """P(B > A) + P(B = A)/2, the Mann-Whitney AUC."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean((b[:, None] > a[None, :]) + 0.5 * (b[:, None] == a[None, :])))


def _ar1(phi, n, rng, innov=None, burn=300):
    e = rng.standard_normal(n + burn) if innov is None else innov(n + burn)
    return lfilter([1.0], [1.0, -phi], e)[burn:]


def _arch_ar(n, rng, phi=0.4, alpha=0.9, burn=200):
    """AR(1) driven by ARCH(1) innovations: volatility clustering with no linear correlation in the innovations."""
    z = rng.standard_normal(n + burn)
    e = np.empty(n + burn)
    e[0] = z[0]
    for i in range(1, n + burn):
        e[i] = np.sqrt(1 - alpha + alpha * e[i - 1] ** 2) * z[i]
    return lfilter([1.0], [1.0, -phi], e)[burn:]


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


# ---------------------------------------------------------------- pol_var_resid: exact values

def test_pol_var_resid_hand_case():
    # ar_order = 0: the residuals are x - mean(x) = x; |x| has median (0.3 + 2)/2, so the symbols are
    # 000111000111: four runs of three
    a = np.array([0.1, 0.2, 0.3, 2.0, 3.0, 4.0])
    x = np.concatenate([a, -a])
    assert pol_var_resid(x, D=3, ar_order=0) == pytest.approx(4 / 12, abs=1e-15)
    assert pol_var_resid(x, D=2, ar_order=0) == pytest.approx(4 / 12, abs=1e-15)  # floor(3/2) per run
    assert pol_var_resid(x, D=4, ar_order=0) == 0.0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_pol_var_resid_matches_an_independent_ar_fit(p):
    x = np.random.default_rng(5).standard_normal(400).cumsum() * 0.05 + np.random.default_rng(6).standard_normal(400)
    # normal equations for x_t ~ c + sum_j a_j x_{t-j}
    X = np.column_stack([np.ones(len(x) - p)] + [x[p - j:len(x) - j] for j in range(1, p + 1)])
    beta = np.linalg.solve(X.T @ X, X.T @ x[p:])
    r = np.abs(x[p:] - X @ beta)
    s = (r >= np.median(r)).astype(int)
    count, i = 0, 0
    while i <= len(s) - 3:  # hctsa MD_PolVar's scan: count a word of 3 equal symbols and jump past it
        if len(set(s[i:i + 3])) == 1:
            count, i = count + 1, i + 3
        else:
            i += 1
    assert pol_var_resid(x, D=3, ar_order=p) == pytest.approx(count / len(s), abs=1e-15)


def test_pol_var_resid_regression_values():
    x = np.random.default_rng(42).standard_normal(300)
    assert pol_var_resid(x, 3) == pytest.approx(38 / 298, abs=1e-15)
    assert pol_var_resid(x, 6) == pytest.approx(4 / 298, abs=1e-15)
    assert pol_var_resid(x, 3, ar_order=1) == pytest.approx(39 / 299, abs=1e-15)


def test_pol_var_resid_is_affine_invariant():
    x = np.random.default_rng(1).standard_normal(500)
    assert pol_var_resid(x, 3) == pol_var_resid(3.7 * x - 12.0, 3) == pol_var_resid(z_score(x), 3)


def test_pol_var_resid_null_value_for_iid_noise():
    # iid symbols with P(1) = 1/2: E = 1 / (2^D - 1), whatever the noise distribution
    rng = np.random.default_rng(11)
    for x in (rng.standard_normal(40000), rng.standard_t(3, 40000), rng.exponential(1.0, 40000)):
        assert pol_var_resid(x, 3) == pytest.approx(1 / 7, abs=0.006)
        assert pol_var_resid(x, 6) == pytest.approx(1 / 63, abs=0.003)


# ---------------------------------------------------------------- pol_var_resid: degenerate input

def test_pol_var_resid_degenerate_inputs_give_nan():
    t = np.arange(200)
    assert np.isnan(pol_var_resid(np.ones(50), 3))                     # constant
    assert np.isnan(pol_var_resid(np.sin(0.3 * t), 3))                 # exact AR(2): no innovations
    assert np.isnan(pol_var_resid(np.r_[np.ones(20), np.nan], 3))      # non-finite
    x = np.random.default_rng(0).standard_normal(30)
    assert np.isnan(pol_var_resid(x[:7], 3))                           # 5 residuals < 2 D
    assert np.isfinite(pol_var_resid(x[:8], 3))                        # 6 residuals = 2 D
    for n in range(0, 15):
        v = _quiet(pol_var_resid, x[:n], 6)                            # never raises
        assert np.isnan(v) or 0 <= v <= 1


# ---------------------------------------------------------------- pol_var_resid: specificity

def test_pol_var_resid_null_on_marginal_and_responsive_to_volatility_clustering():
    rng = np.random.default_rng(2026)
    n, reps = 1000, 40
    gauss = [rng.standard_normal(n) for _ in range(reps)]
    t3 = [rng.standard_t(3, n) for _ in range(reps)]
    ar = [_ar1(0.4, n, rng) for _ in range(reps)]
    arch = [_arch_ar(n, rng) for _ in range(reps)]

    def f(series, fn):
        return [fn(z_score(y)) for y in series]

    # nuisance: iid Gaussian vs iid t(3) noise (no clustering in either); the original separates them
    assert _auc(f(gauss, lambda y: pol_var(y, 1, 3)), f(t3, lambda y: pol_var(y, 1, 3))) > 0.9
    for D in (3, 6):
        assert abs(_auc(f(gauss, lambda y: pol_var_resid(y, D)), f(t3, lambda y: pol_var_resid(y, D))) - 0.5) < 0.2
    # target: ARCH(1) innovations (volatility clustering) vs AR(1) with the same linear filter
    for D in (3, 6):
        assert _auc(f(ar, lambda y: pol_var_resid(y, D)), f(arch, lambda y: pol_var_resid(y, D))) > 0.9


# ---------------------------------------------------------------- registration in hctsa.yaml

def test_variants_are_registered_as_pyhctsa_only():
    fc = FeatureCalculator()
    cfg = fc.config
    assert cfg["medical"]["pol_var_resid"]["variant_of"] == "pol_var"
    assert "legacy_name" not in cfg["medical"]["pol_var_resid"]
    assert {"pol_var_resid_3", "pol_var_resid_6"} <= set(fc.feature_funcs)
    assert fc.min_lengths["pol_var_resid_3"] == 8 and fc.min_lengths["pol_var_resid_6"] == 14
    x = np.random.default_rng(9).standard_normal(300)
    assert fc.feature_funcs["pol_var_resid_3"](x) == pytest.approx(pol_var_resid(z_score(x), 3), abs=1e-15)
