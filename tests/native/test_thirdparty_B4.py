"""Accuracy tests for the library swaps of work package B4: Gaussian HMM (hmmlearn) and simple
exponential smoothing (statsmodels). Oracles are direct numpy implementations of the published
algorithms (Rabiner 1989 scaled forward-backward / Baum-Welch with a tied variance; the SES recursion),
and simulations with known parameters."""
import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf


# ------------------------------------------------------------------------------
# Gaussian HMM
# ------------------------------------------------------------------------------
def _emission(x, mu, cov):
    return np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / cov) / np.sqrt(2 * np.pi * cov)


def _forward_backward(x, mu, cov, P, pi):
    """Rabiner (1989) scaled forward-backward: gamma, xi sums, log-likelihood."""
    T, K = len(x), len(mu)
    B = _emission(x, mu, cov)
    alpha = np.zeros((T, K))
    beta = np.ones((T, K))
    c = np.zeros(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ P) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    xi = np.zeros((K, K))
    for t in range(T - 1):
        xi += alpha[t][:, None] * P * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
    return gamma, xi, np.sum(np.log(c))


def _baum_welch_step(x, mu, cov, P, pi):
    gamma, xi, ll = _forward_backward(x, mu, cov, P, pi)
    new_pi = gamma[0]
    new_P = xi / xi.sum(axis=1, keepdims=True)
    new_mu = (gamma * x[:, None]).sum(axis=0) / gamma.sum(axis=0)
    new_cov = np.sum(gamma * (x[:, None] - new_mu[None, :]) ** 2) / len(x)
    return new_mu, new_cov, new_P, new_pi, ll


def _sample_hmm(rng, T, mu, sigma, P, pi):
    K = len(mu)
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        s[t] = rng.choice(K, p=P[s[t - 1]])
    return np.asarray(mu)[s] + sigma * rng.randn(T), s


def _random_problem(rng, K, T=200):
    x = rng.randn(T) * 1.3 + rng.randn(T).cumsum() * 0.05
    mu = np.sort(rng.randn(K))
    P = rng.dirichlet(np.ones(K) * 3, size=K)
    pi = rng.dirichlet(np.ones(K) * 3)
    return x, mu, 0.5 + rng.rand(), P, pi


@pytest.mark.parametrize('K', [2, 3])
@pytest.mark.parametrize('seed', range(4))
def test_one_em_cycle_matches_baum_welch_oracle(K, seed):
    rng = np.random.RandomState(seed)
    x, mu, cov, P, pi = _random_problem(rng, K)
    mu_o, cov_o, P_o, pi_o, ll_o = _baum_welch_step(x, mu, cov, P, pi)
    mu_n, cov_n, P_n, pi_n, LL = mf._gaussian_hmm_em(x, mu, cov, P, pi, 1, 1e-4, 0.0)
    np.testing.assert_allclose(mu_n, mu_o, atol=1e-8, rtol=1e-8)
    assert abs(cov_n - cov_o) < 1e-8
    np.testing.assert_allclose(P_n, P_o, atol=1e-8)
    np.testing.assert_allclose(pi_n, pi_o, atol=1e-8)
    assert len(LL) == 1 and abs(LL[0] - ll_o) < 1e-8


@pytest.mark.parametrize('K', [2, 3])
def test_several_cycles_match_oracle_and_do_not_modify_inputs(K):
    rng = np.random.RandomState(10 + K)
    x, mu, cov, P, pi = _random_problem(rng, K)
    saved = [np.copy(v) for v in (x, mu, P, pi)]
    out = mf._gaussian_hmm_em(x, mu, cov, P, pi, 5, 0.0, 0.0)
    for v, v0 in zip((x, mu, P, pi), saved):
        np.testing.assert_array_equal(v, v0)
    m, c, Pm, p = mu, cov, P, pi
    lls = []
    for _ in range(5):
        m, c, Pm, p, ll = _baum_welch_step(x, m, c, Pm, p)
        lls.append(ll)
    np.testing.assert_allclose(out[0], m, atol=1e-7)
    assert abs(out[1] - c) < 1e-7
    np.testing.assert_allclose(out[2], Pm, atol=1e-7)
    np.testing.assert_allclose(out[4], lls, atol=1e-7)


@pytest.mark.parametrize('K', [1, 2, 3])
def test_loglik_matches_forward_algorithm(K):
    rng = np.random.RandomState(20 + K)
    x, mu, cov, P, pi = _random_problem(rng, K, T=300)
    P = P / P.sum(axis=1, keepdims=True)
    ll = mf._gaussian_hmm_loglik(x, mu, cov, P, pi)
    assert abs(ll - _forward_backward(x, mu, cov, P, pi)[2]) < 1e-9


def test_loglik_is_nan_for_non_finite_parameters():
    x = np.random.RandomState(0).randn(50)
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert np.isnan(mf._gaussian_hmm_loglik(x, np.array([0, np.nan]), 1.0, P, np.ones(2) / 2))
    assert np.isnan(mf._gaussian_hmm_loglik(x, np.array([0, 1.0]), np.nan, P, np.ones(2) / 2))


def test_em_log_likelihood_is_monotone_without_floor():
    rng = np.random.RandomState(5)
    x, s = _sample_hmm(rng, 600, [-1.5, 1.5], 1.0, np.array([[0.9, 0.1], [0.2, 0.8]]), np.ones(2) / 2)
    _, _, _, _, LL = mf._gaussian_hmm_em(x, np.array([-0.5, 0.5]), np.var(x), np.full((2, 2), 0.5),
                                         np.ones(2) / 2, 30, 0.0, 0.0)
    assert len(LL) == 30
    assert np.all(np.diff(LL) > -1e-8)


def test_em_k1_reduces_to_moments():
    rng = np.random.RandomState(6)
    x = 2.0 + 0.5 * rng.randn(400)
    mu, cov, P, pi, LL = mf._gaussian_hmm_em(x, np.array([0.0]), 1.0, np.ones((1, 1)), np.ones(1), 30, 1e-4, 0.0)
    assert abs(mu[0] - x.mean()) < 1e-6 and abs(cov - x.var()) < 1e-6
    np.testing.assert_allclose(P, 1.0)
    np.testing.assert_allclose(LL[1:], LL[1], rtol=1e-12)   # converged after one cycle


def test_em_stops_on_relative_gain():
    rng = np.random.RandomState(6)
    x = np.r_[rng.randn(300) - 2, rng.randn(300) + 2]
    mu, cov, P, pi, LL = mf._gaussian_hmm_em(x, np.array([-0.5, 0.5]), 1.0, np.full((2, 2), 0.5), np.ones(2) / 2,
                                             30, 1e-4, 0.0)
    assert len(LL) < 30
    # stops at the first cycle (from the third) whose gain since the second cycle is less than
    # 1 + tol times the gain up to the previous cycle
    gains = LL - LL[1]
    assert gains[-1] < (1 + 1e-4) * gains[-2]
    assert all(not (gains[c] < (1 + 1e-4) * gains[c - 1]) for c in range(2, len(LL) - 1))


def test_variance_floor_is_respected():
    rng = np.random.RandomState(7)
    x = rng.choice([-1.0, 0.0, 1.0], size=300)
    floor = 0.2
    mu, cov, *_ = mf._gaussian_hmm_em(x, np.array([-1.0, 0.0, 1.0]), 0.5, np.full((3, 3), 1 / 3),
                                      np.ones(3) / 3, 30, 1e-4, floor)
    assert cov >= floor - 1e-12


@pytest.mark.parametrize('mu_true,sigma,P_true,seed', [
    ([-2.0, 2.0], 1.0, [[0.95, 0.05], [0.10, 0.90]], 1),
    ([-3.0, 0.0, 3.0], 0.7, [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]], 2),
])
def test_recovers_simulated_hmm_parameters(mu_true, sigma, P_true, seed):
    K = len(mu_true)
    P_true = np.array(P_true)
    rng = np.random.RandomState(seed)
    x, s = _sample_hmm(rng, 6000, mu_true, sigma, P_true, np.ones(K) / K)
    mu, cov, P, pi, LL = mf._gaussian_hmm_fit(x, K)
    order = np.argsort(mu)
    np.testing.assert_allclose(mu[order], mu_true, atol=0.1)
    assert abs(cov - sigma ** 2) < 0.06 * sigma ** 2 + 0.02
    np.testing.assert_allclose(P[np.ix_(order, order)], P_true, atol=0.04)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    # the fit is at least as good as the true parameters
    assert LL >= mf._gaussian_hmm_loglik(x, mu_true, sigma ** 2, P_true, np.ones(K) / K) - 5.0


def test_fit_is_deterministic():
    rng = np.random.RandomState(3)
    x, _ = _sample_hmm(rng, 800, [-1.0, 1.0], 1.0, np.array([[0.9, 0.1], [0.1, 0.9]]), np.ones(2) / 2)
    a, b = mf._gaussian_hmm_fit(x, 2), mf._gaussian_hmm_fit(x, 2)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


# ------------------------------------------------------------------------------
# Exponential smoothing
# ------------------------------------------------------------------------------
def _ses_forecasts(x, a):
    level = np.mean(x[:10])
    f = np.empty(len(x))
    for t in range(len(x)):
        f[t] = level
        level = a * x[t] + (1 - a) * level
    return f


@pytest.mark.parametrize('a', [0.01, 0.2, 0.5, 0.99, 1.0])
def test_fit_exp_smooth_matches_recursion(a):
    x = np.random.RandomState(0).randn(300).cumsum()
    f = mf._fit_exp_smooth(x, a)
    assert f.shape == x.shape and f.dtype == np.float64
    np.testing.assert_allclose(f, _ses_forecasts(x, a), atol=1e-12, rtol=1e-12)


def test_fit_exp_smooth_special_cases():
    x = np.random.RandomState(1).randn(50)
    f = mf._fit_exp_smooth(x, 1.0)
    np.testing.assert_allclose(f[1:], x[:-1], atol=1e-14)
    np.testing.assert_allclose(mf._fit_exp_smooth(np.full(40, 3.5), 0.4), 3.5)
    assert mf._fit_exp_smooth(x, 0.3)[0] == pytest.approx(x[:10].mean())
    assert len(mf._fit_exp_smooth(x[:4], 0.3)) == 4  # fewer than 10 samples


@pytest.mark.parametrize('alpha_true,seed', [(0.3, 0), (0.6, 1), (0.85, 2)])
def test_best_alpha_recovers_local_level_model(alpha_true, seed):
    # local level model: x_t = l_{t-1} + e_t, l_t = l_{t-1} + alpha e_t
    rng = np.random.RandomState(seed)
    e = rng.randn(1000)
    level = np.cumsum(np.r_[0.0, alpha_true * e[:-1]])
    x = level + e
    alpha, diag = mf._exp_smooth_best_alpha(x)
    assert abs(alpha - alpha_true) < 0.08
    assert set(diag) == {'slope05_1', 'p1_1', 'cup_1'}
    assert diag['cup_1'] == 1 and diag['p1_1'] > 0
    assert 0.01 <= alpha <= 1


def test_best_alpha_ar1_and_random_walk():
    rng = np.random.RandomState(4)
    x = np.zeros(1000)
    for t in range(1, 1000):
        x[t] = 0.9 * x[t - 1] + rng.randn()
    alpha, _ = mf._exp_smooth_best_alpha(x)
    assert 0.5 < alpha <= 1
    walk = np.cumsum(rng.randn(1000))
    assert abs(mf._exp_smooth_best_alpha(walk)[0] - 1.0) < 0.05


def test_best_alpha_white_noise_is_small_and_nonfinite_gives_nan():
    noise = np.random.RandomState(5).randn(1000)
    assert mf._exp_smooth_best_alpha(noise)[0] < 0.1
    alpha, diag = mf._exp_smooth_best_alpha(np.full(200, np.nan))
    assert np.isnan(alpha) and all(np.isnan(v) for v in diag.values())


def test_exp_smoothing_runs_end_to_end():
    rng = np.random.RandomState(6)
    x = np.cumsum(rng.randn(500)) * 0.2 + rng.randn(500)
    try:
        out = mf.exp_smoothing(x, 0.5, 'best')
    except NotImplementedError:  # residual_analysis('full') needs _arfit (work package B1)
        pytest.skip('residual_analysis needs the ARFIT rewrite')
    assert 0.01 <= out['alphamin'] <= 1
    assert all(k in out for k in ('slope05_1', 'p1_1', 'cup_1', 'meane', 'stde', 'ac1', 'minsbc'))
    assert out == mf.exp_smoothing(x, 0.5, 'best')
