"""pyhctsa-only variants of correlation features: crinkle_statistic_kurtosis, nonlinear_autocorr_excess,
nonlinear_autocorr_abs_surrogate_ratio and trev_studentized.

Exact values on small cases, degenerate inputs, and fast specificity checks: each variant should be null on the
nuisance its original responds to (linear autocorrelation, the marginal distribution) and still respond to its
target (nonlinear or irreversible dynamics)."""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.signal import lfilter

from pyhctsa.operations import correlation as co


def _z(y):
    y = np.asarray(y, dtype=float)
    return (y - y.mean()) / y.std(ddof=1)


def _ar1(n, phi, rng, innov=None, burn=200):
    e = rng.standard_normal(n + burn) if innov is None else innov(n + burn)
    return lfilter([1.0], [1.0, -phi], e)[burn:]


def _rank_series(n, phi, rng):
    # distinct, evenly spaced values (0..n-1) with the copula of a Gaussian AR(1); strictly increasing
    # transforms of it (exp(r / 50), log1p(r)) keep every gap far above the tie tolerance of the ranks
    return np.argsort(np.argsort(_ar1(n, phi, rng))).astype(float)


def _monotone_versions(r):
    return [np.exp(r / 50), np.log1p(r), 3 * r - 7]


def _ma1(n, theta, rng):
    e = rng.standard_normal(n + 1)
    return e[1:] + theta * e[:-1]


def _ma1_theta(rho1):   # MA(1) coefficient with lag-1 autocorrelation rho1
    return (1 - np.sqrt(1 - 4 * rho1 ** 2)) / (2 * rho1)


def _arch_ar(n, rng, phi=0.4, alpha=0.9, burn=200):
    zz = rng.standard_normal(n + burn)
    e = np.empty(n + burn)
    e[0] = zz[0]
    for i in range(1, n + burn):
        e[i] = np.sqrt(1 - alpha + alpha * e[i - 1] ** 2) * zz[i]
    return lfilter([1.0], [1.0, -phi], e)[burn:]


def _auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] < b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (a.size * b.size)


def _scan(f, sim_a, sim_b, n=1000, reps=40, seed=0):
    rng = np.random.default_rng(seed)
    fa = [f(_z(sim_a(n, rng))) for _ in range(reps)]
    fb = [f(_z(sim_b(n, rng))) for _ in range(reps)]
    return _auc(fa, fb)


AR04 = lambda n, rng: _ar1(n, 0.4, rng)                                     # noqa: E731
MA04 = lambda n, rng: _ma1(n, _ma1_theta(0.4), rng)                         # noqa: E731
EXP_AR04 = lambda n, rng: np.exp(_ar1(n, 0.4, rng) * np.sqrt(1 - 0.16))     # noqa: E731  same copula as AR04
ARCH = lambda n, rng: _arch_ar(n, rng)                                      # noqa: E731


# ------------------------------------------------------------------------------
# crinkle_statistic_kurtosis
# ------------------------------------------------------------------------------
def test_crinkle_kurtosis_exact_alternating():
    # x - mean = +-1/2 alternating: second differences are -+2 alternating, kurtosis 1, excess -2
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    assert co.crinkle_statistic_kurtosis(y, gaussian_rank=False) == pytest.approx(-2.0)
    assert co.crinkle_statistic_kurtosis(y) == pytest.approx(-2.0)   # two tied values: ranks keep the pattern


def test_crinkle_kurtosis_is_the_shape_factor_of_crinkle():
    # crinkle = kurt(D2 x) * (<(D2 x)^2> / <x^2>)^2 (uncentred D2 moments, exactly)
    rng = np.random.default_rng(1)
    x = _z(_ar1(500, 0.6, rng))
    d2 = np.diff(x - x.mean(), 2)
    kurt_uncentred = np.mean(d2 ** 4) / np.mean(d2 ** 2) ** 2
    factor = (np.mean(d2 ** 2) / np.mean((x - x.mean()) ** 2)) ** 2
    assert co.crinkle_statistic(x) == pytest.approx(kurt_uncentred * factor, rel=1e-12)
    # the variant centres D2 (whose mean is O(1/N)), so it is close to kurt - 3
    assert co.crinkle_statistic_kurtosis(x, gaussian_rank=False) == pytest.approx(kurt_uncentred - 3, abs=1e-3)


def test_crinkle_kurtosis_gaussian_and_iid_limits():
    rng = np.random.default_rng(2)
    # Gaussian AR(1): K -> 0 whatever the autocorrelation, while crinkle -> 3 (6 - 8 rho1 + 2 rho2)^2
    for phi in (0.0, 0.5, 0.9):
        x = _z(_ar1(200_000, phi, rng))
        assert abs(co.crinkle_statistic_kurtosis(x)) < 0.05
        assert co.crinkle_statistic(x) == pytest.approx(3 * (6 - 8 * phi + 2 * phi ** 2) ** 2, rel=0.03)
    # iid uniform (excess kurtosis -1.2): without ranks K = kappa / 2 = -0.6; with ranks K = 0
    u = _z(rng.uniform(size=200_000))
    assert co.crinkle_statistic_kurtosis(u, gaussian_rank=False) == pytest.approx(-0.6, abs=0.02)
    assert abs(co.crinkle_statistic_kurtosis(u)) < 0.05


def test_crinkle_kurtosis_invariant_to_monotone_transforms():
    r = _rank_series(300, 0.5, np.random.default_rng(3))
    for y in _monotone_versions(r):
        assert co.crinkle_statistic_kurtosis(y) == co.crinkle_statistic_kurtosis(r)


@pytest.mark.parametrize("y", [[], [1.0], [1, 2, 3, 4], [5.0] * 20, np.arange(20.0), np.arange(20.0) ** 2])
def test_crinkle_kurtosis_degenerate_is_nan(y):
    # fewer than 5 samples, constant, and (without ranks) constant second difference
    assert np.isnan(co.crinkle_statistic_kurtosis(y, gaussian_rank=False))
    if len(y) < 5 or np.ptp(y) == 0:
        assert np.isnan(co.crinkle_statistic_kurtosis(y))


def test_crinkle_kurtosis_specificity():
    # crinkle_statistic separates AR(1) from an MA(1) with the same rho1 (a second-order difference) and a
    # monotone transform of the AR(1) (the marginal); the variant does neither and keeps the ARCH target
    f = co.crinkle_statistic_kurtosis
    assert _scan(co.crinkle_statistic, AR04, MA04) < 0.25
    assert _scan(co.crinkle_statistic, AR04, EXP_AR04) > 0.9
    assert abs(_scan(f, AR04, MA04) - 0.5) < 0.2
    assert abs(_scan(f, AR04, EXP_AR04) - 0.5) < 0.2
    assert _scan(f, AR04, ARCH) > 0.9


# ------------------------------------------------------------------------------
# nonlinear_autocorr_excess
# ------------------------------------------------------------------------------
def _acov(x, k):
    return np.mean(x[k:] * x[:len(x) - k])


def _pairings(items):
    # all perfect matchings of a list (independent of the implementation's recursion)
    if not items:
        yield []
        return
    a, rest = items[0], items[1:]
    for j in range(len(rest)):
        for p in _pairings(rest[:j] + rest[j + 1:]):
            yield [(a, rest[j])] + p


@pytest.mark.parametrize("taus", [[1, 2, 3], [0, 1, 1], [1, 1, 2], [0, 0, 3], [3, 3, 3], [1, 3, 5, 7, 8], [1, 2, 3, 4, 5]])
def test_excess_equals_moment_minus_isserlis(taus):
    rng = np.random.default_rng(4)
    x = _z(_ar1(400, 0.5, rng))
    lags = [0, *taus]
    gauss = sum(np.prod([_acov(x, abs(lags[i] - lags[j])) for i, j in p])
                for p in _pairings(list(range(len(lags))))) if len(lags) % 2 == 0 else 0.0
    want = co.nonlinear_autocorr(x, taus, absval=False) - gauss
    assert co.nonlinear_autocorr_excess(x, taus, gaussian_rank=False) == pytest.approx(want, abs=1e-12)


def test_excess_hand_values():
    rng = np.random.default_rng(5)
    x = _z(rng.standard_normal(50))
    c = [_acov(x, k) for k in range(4)]
    m = co.nonlinear_autocorr(x, [1, 2, 3], absval=False)
    assert co.nonlinear_autocorr_excess(x, [1, 2, 3], False) == pytest.approx(m - (c[1] ** 2 + c[2] ** 2 + c[1] * c[3]))
    m = co.nonlinear_autocorr(x, [1, 1, 2], absval=False)
    assert co.nonlinear_autocorr_excess(x, [1, 1, 2], False) == pytest.approx(m - (2 * c[1] ** 2 + c[0] * c[2]))
    # odd number of factors: no Gaussian part
    assert co.nonlinear_autocorr_excess(x, [1, 3], False) == co.nonlinear_autocorr(x, [1, 3], absval=False)
    # a single lag is a plain autocovariance, whose excess is zero by construction
    assert co.nonlinear_autocorr_excess(x, [2], False) == pytest.approx(0.0, abs=1e-15)


def test_excess_gaussian_null_and_rank_invariance():
    rng = np.random.default_rng(6)
    x = _z(_ar1(100_000, 0.7, rng))
    for taus, tol in (([1, 2, 3], 0.03), ([0, 1, 1], 0.03), ([1, 1, 2], 0.03), ([1, 3], 0.03),
                      ([1, 2, 3, 4, 5], 0.15)):   # six factors: a much noisier estimate
        assert abs(co.nonlinear_autocorr_excess(x, taus)) < tol
        # invariant under strictly increasing transforms (identical ranks)
        r = _rank_series(300, 0.5, rng)
        for y in _monotone_versions(r):
            assert co.nonlinear_autocorr_excess(y, taus) == co.nonlinear_autocorr_excess(r, taus)
    # the original is dominated by the Gaussian part, e.g. <x_t x_{t-1}^2 x_{t-2}> ~ 2 rho1^2 + rho2
    assert co.nonlinear_autocorr(x, [1, 1, 2], absval=False) == pytest.approx(3 * 0.49, abs=0.05)


def test_excess_mirror_fourth_order_lag():
    # [k, k, k] = <x_t x_{t-k}^3> is the time mirror of [0, 0, k] = <x_t^3 x_{t-k}>. On a Gaussian AR(1) both
    # are 3 rho^k (Isserlis) and their excess is zero. A fourth-order echo x_t = e_t + a g(e_{t-k}), with g odd,
    # g^2 = 1 and E[e g(e)] = 0, is white with white squares and no third-order lag moment; its lowest lag
    # moment is <x_t x_{t-k}^3> = a E[e^3 g(e)], seen by [k, k, k] at its own lag and by nothing else here
    rng = np.random.default_rng(14)
    x = _z(_ar1(100_000, 0.6, rng))
    for k in (1, 2):
        assert co.nonlinear_autocorr(x, [k, k, k], absval=False) == pytest.approx(3 * 0.6 ** k, abs=0.06)
        assert abs(co.nonlinear_autocorr_excess(x, [k, k, k])) < 0.03
    theta = np.sqrt(2 * np.log(2))   # |e|-weighted median of |e| for a Gaussian, so E[e g(e)] = 0
    e = rng.standard_normal(50_000)
    y = e.copy()
    y[3:] += 0.8 * np.sign(e[:-3]) * np.where(np.abs(e[:-3]) < theta, 1.0, -1.0)
    assert co.nonlinear_autocorr_excess(y, [3, 3, 3]) < -0.25
    for taus in ([0, 0, 3], [2, 2, 2], [4, 4, 4], [3]):
        assert abs(co.nonlinear_autocorr_excess(y, taus)) < 0.05


def test_excess_degenerate():
    assert np.isnan(co.nonlinear_autocorr_excess([1.0, 2.0, 3.0], [3]))
    assert np.isnan(co.nonlinear_autocorr_excess([1.0, -1.0], [1, 2]))
    assert np.isnan(co.nonlinear_autocorr_excess(np.ones(10), [1, 2]))   # all values tied
    assert co.nonlinear_autocorr_excess(np.zeros(10), [1, 2, 3], gaussian_rank=False) == 0.0
    for bad in ([], [-1, 2], [1.5]):
        with pytest.raises(ValueError):
            co.nonlinear_autocorr_excess(np.arange(10.0), bad)


def test_excess_specificity():
    # ac_nl_1_2_3 responds to rho2 (AR vs MA at matched rho1) and the odd-order ac_nl_1_3 to the marginal
    # (a monotone transform of the AR); the excess versions do not, and still see ARCH dependence
    orig = lambda taus: (lambda x: co.nonlinear_autocorr(x, taus, absval=False))   # noqa: E731
    exc = lambda taus: (lambda x: co.nonlinear_autocorr_excess(x, taus))            # noqa: E731
    assert _scan(orig([1, 2, 3]), AR04, MA04, n=3000) < 0.25
    assert _scan(orig([1, 3]), AR04, EXP_AR04, n=3000) > 0.75
    assert abs(_scan(exc([1, 2, 3]), AR04, MA04, n=3000) - 0.5) < 0.2
    assert abs(_scan(exc([1, 3]), AR04, EXP_AR04, n=3000) - 0.5) < 0.2
    assert abs(_scan(exc([0, 1, 1]), AR04, EXP_AR04) - 0.5) < 0.2
    assert _scan(exc([0, 1, 1]), AR04, ARCH) > 0.9


# ------------------------------------------------------------------------------
# nonlinear_autocorr_abs_surrogate_ratio
# ------------------------------------------------------------------------------
def test_abs_surrogate_ratio_deterministic_and_rank_invariant():
    rng = np.random.default_rng(7)
    x = _z(_ar1(500, 0.5, rng))
    f = co.nonlinear_autocorr_abs_surrogate_ratio
    v = f(x, [1, 2])
    assert v == f(x, [1, 2])
    r = _rank_series(300, 0.5, rng)
    assert all(f(y, [1, 2]) == f(r, [1, 2]) for y in _monotone_versions(r))   # depends on the ranks only
    assert v != f(x, [1, 2], random_seed=1)
    # value over the Gaussian ranks divided by the surrogate mean
    from pyhctsa.operations.surrogates import _make_surrogates
    g = co._gaussian_ranks(x)
    num = np.mean(np.abs(g[2:] * g[1:-1] * g[:-2]))
    surrs = _make_surrogates(g, 'RP', 10, 0)
    den = np.mean([np.mean(np.abs(s[2:] * s[1:-1] * s[:-2])) for s in (_z(c) for c in surrs.T)])
    assert v == pytest.approx(num / den, rel=1e-12)


def test_abs_surrogate_ratio_gaussian_null():
    rng = np.random.default_rng(8)
    vals = [co.nonlinear_autocorr_abs_surrogate_ratio(_z(_ar1(2000, 0.6, rng)), [1, 2]) for _ in range(20)]
    assert np.mean(vals) == pytest.approx(1.0, abs=0.03)


def test_abs_surrogate_ratio_degenerate():
    f = co.nonlinear_autocorr_abs_surrogate_ratio
    assert np.isnan(f([1.0, 2.0, 3.0], [1]))
    assert np.isnan(f(np.arange(5.0), [5]))
    assert np.isnan(f(np.ones(20), [1, 2]))


def test_abs_surrogate_ratio_specificity():
    # the absval original is a marginal moment (iid Gaussian vs iid t3) and sees rho1 (AR vs MA); the ratio is
    # null on both and keeps the ARCH target
    orig = lambda x: co.nonlinear_autocorr(x, [1, 2], absval=True)               # noqa: E731
    rat = lambda x: co.nonlinear_autocorr_abs_surrogate_ratio(x, [1, 2])           # noqa: E731
    iid_g = lambda n, rng: rng.standard_normal(n)                                  # noqa: E731
    iid_t = lambda n, rng: rng.standard_t(3, n)                                    # noqa: E731
    assert _scan(orig, iid_g, iid_t) < 0.1
    assert abs(_scan(rat, iid_g, iid_t) - 0.5) < 0.2
    assert abs(_scan(rat, AR04, MA04) - 0.5) < 0.2
    assert _scan(rat, AR04, ARCH) > 0.9


# ------------------------------------------------------------------------------
# trev_studentized
# ------------------------------------------------------------------------------
def test_trev_studentized_matches_explicit_hac():
    rng = np.random.default_rng(9)
    y = _z(_ar1(300, 0.3, rng))
    for tau in (1, 3):
        s = (y[tau:] - y[:-tau]) ** 3
        n = s.size
        m = max(tau, int(np.floor(4 * (n / 100) ** (2 / 9))) // 2 + 1,
                2 * int(co.get_tau(y, 'ac1e')))
        sc = s - s.mean()
        g = [np.sum(sc[k:] * sc[:n - k]) / n for k in range(2 * m + 1)]
        w = [min(1.0, max(0.0, 2 - k / m)) for k in range(2 * m + 1)]
        lrv = g[0] + 2 * sum(w[k] * g[k] for k in range(1, 2 * m + 1))
        bart = g[0] + 2 * sum((1 - k / (2 * m + 1)) * g[k] for k in range(1, 2 * m + 1))
        lrv = max(lrv, 0.05 * bart)
        assert co.trev_studentized(y, tau) == pytest.approx(s.mean() / np.sqrt(lrv / n), rel=1e-12)


def test_trev_studentized_bartlett_part_matches_statsmodels_hac():
    # the floor is 0.05 x the Bartlett (Newey-West) long-run variance: check that estimator against statsmodels
    rng = np.random.default_rng(10)
    s = rng.standard_normal(200) ** 3
    L = 6
    res = sm.OLS(s, np.ones_like(s)).fit(cov_type='HAC', cov_kwds={'maxlags': L, 'use_correction': False})
    sc = s - s.mean()
    n = s.size
    bart = np.mean(sc * sc) + 2 * sum((1 - k / (L + 1)) * np.sum(sc[k:] * sc[:n - k]) / n for k in range(1, L + 1))
    assert res.bse[0] == pytest.approx(np.sqrt(bart / n), rel=1e-10)


def test_trev_studentized_symmetries():
    rng = np.random.default_rng(11)
    y = _ar1(400, 0.5, rng, innov=lambda n: rng.exponential(1.0, n) - 1.0)
    for tau in (1, 2, 5):
        t = co.trev_studentized(y, tau)
        assert t > 0 or t < 0
        assert co.trev_studentized(y[::-1], tau) == pytest.approx(-t, rel=1e-10)   # time reversal flips the sign
        assert co.trev_studentized(3.0 * y + 2.0, tau) == pytest.approx(t, rel=1e-10)
        # same sign as trev's raw value
        assert np.sign(co.trev(y, tau)['raw']) == np.sign(t)


def test_trev_studentized_degenerate():
    assert np.isnan(co.trev_studentized([1.0, 2.0, 4.0], 1))     # two increments
    assert np.isnan(co.trev_studentized(np.arange(10.0), 8))
    assert np.isnan(co.trev_studentized(np.ones(30), 1))         # constant
    assert np.isnan(co.trev_studentized(np.arange(30.0), 2))     # constant increments
    assert np.isfinite(co.trev_studentized([1.0, 2.0, 4.0, 3.0], 1))


def test_trev_studentized_null_calibration_and_specificity():
    # on reversible processes T is ~N(0, 1) for Gaussian and lognormal marginals alike, so |T| no longer
    # separates them (|trev| does); irreversible exponential-innovation AR(1) is still detected
    rng = np.random.default_rng(12)
    gauss = [co.trev_studentized(_z(_ar1(1000, 0.9, rng)), 1) for _ in range(60)]
    logn = [co.trev_studentized(_z(np.exp(_ar1(1000, 0.9, rng) * np.sqrt(1 - 0.81))), 1) for _ in range(60)]
    assert 0.75 < np.std(gauss) < 1.3 and 0.75 < np.std(logn) < 1.3
    assert abs(_auc(np.abs(gauss), np.abs(logn)) - 0.5) < 0.2
    absraw = lambda x: co.trev(x, 1)['abs']                                         # noqa: E731
    ar09 = lambda n, rng: _ar1(n, 0.9, rng)                                         # noqa: E731
    exp_ar09 = lambda n, rng: np.exp(_ar1(n, 0.9, rng) * np.sqrt(1 - 0.81))         # noqa: E731
    assert _scan(absraw, ar09, exp_ar09) > 0.75
    irrev = lambda n, rng: _ar1(n, 0.9, rng, innov=lambda m: rng.exponential(1.0, m) - 1.0)   # noqa: E731
    assert _scan(lambda x: co.trev_studentized(x, 1), ar09, irrev) > 0.95


def test_new_variants_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    rng = np.random.default_rng(13)
    y = _ar1(300, 0.5, rng)
    calc = FeatureCalculator()
    labels = ['crinkle_statistic_kurtosis', 'crinkle_statistic_kurtosis_dec', 'ac_nl_excess_1_to_3',
              'ac_nl_excess_1_1_2_dec', 'ac_nl_abs_surr_ratio_1_to_2', 'ac_nl_abs_surr_ratio_1_to_4_dec',
              'trev_studentized_1', 'trev_studentized_ac', 'trev_studentized_mi']
    for lab in labels:
        assert lab in calc.feature_funcs
        assert np.isfinite(calc.feature_funcs[lab](y))
    assert calc.feature_funcs['ac_nl_excess_1_to_3'](y) == co.nonlinear_autocorr_excess(_z(y), [1, 2, 3])
    assert len([k for k in calc.feature_funcs if k.startswith('ac_nl_excess_')]) == 67
    assert len([k for k in calc.feature_funcs if k.startswith('ac_nl_abs_surr_ratio_')]) == 22
    assert len([k for k in calc.feature_funcs if k.startswith('trev_studentized_')]) == 7
