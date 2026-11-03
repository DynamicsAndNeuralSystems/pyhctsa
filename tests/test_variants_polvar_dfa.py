"""pyhctsa-only variants pol_var_resid (of pol_var) and dfa_largescale (of fluctuation_analysis's DFA).

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
from pyhctsa.operations.scaling import dfa_largescale, fluctuation_analysis
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


def _fgn(H, n, rng):
    """Exact fractional Gaussian noise by Davies-Harte circulant embedding."""
    k = np.arange(n + 1)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    s = np.sqrt(np.maximum(np.fft.fft(row).real, 0) / len(row))
    z = rng.standard_normal(len(row)) + 1j * rng.standard_normal(len(row))
    return np.fft.fft(s * z).real[:n]


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


# ---------------------------------------------------------------- dfa_largescale: exact values

def _dfa_reference(x, k=1, n_tau=25):
    """Loop-based DFA over tau in [max(10, sqrt(N)), N/4], with numpy.polyfit and scipy's Theil-Sen slope."""
    n = len(x)
    y = np.cumsum(x - np.mean(x))
    lo, hi = max(10.0, np.sqrt(n)), n / 4
    taus = sorted({int(np.floor(v + 0.5)) for v in np.exp(np.linspace(np.log(lo), np.log(hi), n_tau))})
    F = []
    for tau in taus:
        tt = np.arange(tau)
        res = [y[w * tau:(w + 1) * tau] - np.polyval(np.polyfit(tt, y[w * tau:(w + 1) * tau], k), tt)
               for w in range(n // tau)]
        F.append(np.sqrt(np.mean(np.concatenate(res) ** 2)))
    return stats.theilslopes(np.log(F), np.log(taus))[0], taus


@pytest.mark.parametrize("n", [80, 400, 1000, 3001])
@pytest.mark.parametrize("k", [1, 2])
def test_dfa_largescale_matches_a_loop_reference(n, k):
    rng = np.random.default_rng(n)
    x = 0.1 * rng.standard_normal(n).cumsum() + rng.standard_normal(n)
    assert dfa_largescale(x, k=k) == pytest.approx(_dfa_reference(x, k)[0], abs=1e-12)


def test_dfa_largescale_timescales():
    assert _dfa_reference(np.zeros(80) + np.arange(80) % 3)[1] == list(range(10, 21))   # [10, 20]: 11 scales
    taus = _dfa_reference(np.arange(400) % 7)[1]
    assert taus[0] == 20 and taus[-1] == 100 and len(taus) == 25                       # [sqrt(400), 400/4]


def test_dfa_largescale_regression_values():
    y = np.random.default_rng(42).standard_normal(1000)
    assert dfa_largescale(y) == pytest.approx(0.43528786367569783, abs=1e-12)
    assert dfa_largescale(np.cumsum(y)) == pytest.approx(1.4156677937909592, abs=1e-12)
    assert dfa_largescale(y, k=2) == pytest.approx(0.4754113147616863, abs=1e-12)


def test_dfa_largescale_known_exponents():
    # means over 8 series of N = 20000 (the SD of a single estimate is about 0.035-0.06 here)
    rng = np.random.default_rng(3)
    white = [rng.standard_normal(20000) for _ in range(8)]
    assert np.mean([dfa_largescale(x) for x in white]) == pytest.approx(0.5, abs=0.05)
    assert np.mean([dfa_largescale(np.cumsum(x)) for x in white]) == pytest.approx(1.5, abs=0.05)  # random walk
    assert np.mean([dfa_largescale(_fgn(0.8, 20000, rng)) for _ in range(8)]) == pytest.approx(0.8, abs=0.05)
    assert dfa_largescale(3 * white[0] + 5) == pytest.approx(dfa_largescale(white[0]), abs=1e-12)


# ---------------------------------------------------------------- dfa_largescale: degenerate input

def test_dfa_largescale_degenerate_inputs_give_nan():
    x = np.random.default_rng(0).standard_normal(200)
    assert np.isnan(dfa_largescale(x[:79]))         # N/4 < 2 * 10: range under a factor of 2
    assert np.isfinite(dfa_largescale(x[:80]))
    assert np.isnan(dfa_largescale(np.ones(200)))   # constant
    assert np.isnan(dfa_largescale(np.r_[x[:100], np.nan]))
    assert np.isnan(dfa_largescale(x[:80], k=9))    # a polynomial of order 9 fills the smallest window (10)
    for n in range(0, 81):
        v = _quiet(dfa_largescale, x[:n])            # never raises
        assert np.isnan(v) == (n < 80)


# ---------------------------------------------------------------- dfa_largescale: specificity

def test_dfa_largescale_null_on_short_range_correlation_and_responsive_to_scaling():
    rng = np.random.default_rng(77)
    n, reps = 5000, 30
    white = [rng.standard_normal(n) for _ in range(reps)]
    ar5 = [_ar1(0.5, n, rng) for _ in range(reps)]
    rho1 = 0.5 * (2 ** 1.5 - 2)   # lag-1 autocorrelation of fGn with H = 0.75
    ar_m = [_ar1(rho1, n, rng) for _ in range(reps)]
    fgn = [_fgn(0.75, n, rng) for _ in range(reps)]

    def classical(series):
        return [fluctuation_analysis(z_score(y), 2, 'dfa', 50, 1, None, True)['alpha'] for y in series]

    def large(series):
        return [dfa_largescale(z_score(y)) for y in series]

    # nuisance: white noise vs AR(1), phi = 0.5 (both have exponent 0.5); classical DFA alpha separates them
    assert _auc(classical(white), classical(ar5)) > 0.9
    assert abs(_auc(large(white), large(ar5)) - 0.5) < 0.25
    # target: fGn H = 0.75 vs AR(1) with the same lag-1 autocorrelation
    assert _auc(large(ar_m), large(fgn)) > 0.9


# ---------------------------------------------------------------- registration in hctsa.yaml

def test_variants_are_registered_as_pyhctsa_only():
    fc = FeatureCalculator()
    cfg = fc.config
    assert cfg["medical"]["pol_var_resid"]["variant_of"] == "pol_var"
    assert cfg["scaling"]["dfa_largescale"]["variant_of"] == "fluctuation_analysis"
    assert "legacy_name" not in cfg["medical"]["pol_var_resid"] and "legacy_name" not in cfg["scaling"]["dfa_largescale"]
    assert {"pol_var_resid_3", "pol_var_resid_6", "dfa_largescale_1"} <= set(fc.feature_funcs)
    assert fc.min_lengths["pol_var_resid_3"] == 8 and fc.min_lengths["pol_var_resid_6"] == 14
    x = np.random.default_rng(9).standard_normal(300)
    assert fc.feature_funcs["pol_var_resid_3"](x) == pytest.approx(pol_var_resid(z_score(x), 3), abs=1e-15)
    assert fc.feature_funcs["dfa_largescale_1"](x) == pytest.approx(dfa_largescale(z_score(x), 1), abs=1e-12)
