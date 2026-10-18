"""GARCH fits (garch_fit, garch_compare) and the estimator they share (pyhctsa.operations._garch):
the likelihood convention, the gradient, convergence to a global optimum, parameter recovery, the
test for conditional heteroskedasticity, and the three model types."""
import numpy as np
import pytest

from pyhctsa.operations import _garch as G
from pyhctsa.operations import model_fit as mf


def _simulate(n, omega, alpha, beta, seed, gamma=0.0, nu=None):
    """GARCH(1,1) / GJR(1,1) with Gaussian or unit-variance Student-t innovations."""
    rng = np.random.default_rng(seed)
    m = n + 500
    z = rng.standard_normal(m) if nu is None else rng.standard_t(nu, m) * np.sqrt((nu - 2) / nu)
    y = np.empty(m)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    e_prev = 0.0
    for t in range(m):
        s2 = omega + (alpha + (gamma if e_prev < 0 else 0.0)) * e_prev ** 2 + beta * s2
        y[t] = np.sqrt(s2) * z[t]
        e_prev = y[t]
    y = y[500:]
    return (y - y.mean()) / y.std(ddof=1)


def _loglik(y, w, a, b):
    """Gaussian zero-mean GARCH(1,1) log-likelihood with presample variance and squared
    innovation both equal to the mean of the squared series (written out as a plain loop)."""
    v0 = np.mean(y ** 2)
    s2_prev, e2_prev, llf = v0, v0, 0.0
    for e in y:
        s2 = w + b * s2_prev + a * e2_prev
        llf += -0.5 * (np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)
        s2_prev, e2_prev = s2, e ** 2
    return llf


def test_likelihood_convention():
    y = _simulate(1500, 0.1, 0.1, 0.8, seed=3)
    f = G.mle(G.Data(y), 1, 1)
    w, b, a = f['theta']
    assert f['llf'] == pytest.approx(_loglik(y, w, a, b), abs=1e-8)
    # the conditional variances follow the same recursion from the presample
    assert f['h'][0] == pytest.approx(w + (a + b) * np.mean(y ** 2), rel=1e-12)


@pytest.mark.parametrize('P, Q, gjr, t', [(1, 1, False, False), (2, 3, False, False),
                                          (1, 1, True, False), (1, 1, False, True), (2, 2, True, True)])
def test_gradient(P, Q, gjr, t):
    y = _simulate(600, 0.1, 0.1, 0.8, seed=2)
    D = G.Data(y)
    S = G.Spec(P, Q, gjr, t)
    th = np.zeros(S.k)
    th[0] = 0.1
    th[S.ib] = 0.5 / P
    th[S.ia] = 0.2 / Q
    if gjr:
        th[S.ig] = 0.05
    if t:
        th[S.ik] = 0.1
    g = G._nll(th, D, S)[1]
    for i in range(S.k):
        e = np.zeros(S.k)
        e[i] = 1e-6
        num = (G._nll(th + e, D, S, False) - G._nll(th - e, D, S, False)) / 2e-6
        assert g[i] == pytest.approx(num, abs=1e-7)


def test_global_optimum_of_a_two_mode_likelihood():
    # a deterministic chaotic map (Spence-like) has an ARCH(1) mode with a much higher likelihood
    # than the GARCH mode the best grid point leads to; all grid points are run, so the better is found
    x = np.empty(2000)
    x[0] = 0.3
    for i in range(1999):
        x[i + 1] = 3.9 * x[i] * (1 - x[i])
    y = (x - x.mean()) / x.std(ddof=1)
    D = G.Data(y)
    f = G.mle(D, 1, 1)
    S = f['spec']
    for st in G._start_grid(D, S):
        th, _ = G._newton(G._interior(st, S), D, S, G.MU_COARSE + G.MU_FINE)
        assert -G._nll(th, D, S, False) * D.N <= f['llf'] + 1e-6


def test_parameter_recovery():
    y = _simulate(6000, 0.1, 0.1, 0.8, seed=1)
    f = G.mle(G.Data(y), 1, 1)
    w, b, a = f['theta']
    assert a == pytest.approx(0.1, abs=0.04)
    assert b == pytest.approx(0.8, abs=0.08)
    assert f['dec'] < 1e-8
    se = G.opg_errors(f['theta'], G.Data(y), f['spec'])
    assert np.all(np.isfinite(se)) and np.all(se[1:3] < 0.1)


def test_leverage_and_tails_recovered():
    y = _simulate(8000, 0.03, 0.03, 0.88, seed=4, gamma=0.12)
    out = mf.garch_fit(y, 'none', model_type='gjr')
    assert out['hasStructure'] == 1 and out['leverage'] == pytest.approx(0.12, abs=0.06)
    y = _simulate(8000, 0.02, 0.08, 0.9, seed=5, nu=5)
    out = mf.garch_fit(y, 'none', innovation_dist='t')
    assert out['distDoF'] == pytest.approx(5, abs=1.5)


def test_white_noise_is_constant_variance():
    # Without conditional heteroskedasticity the GARCH coefficient is not identified: the
    # constant-variance model is reported, as numbers
    for seed in range(4):
        y = np.random.default_rng(seed).standard_normal(1000)
        out = mf.garch_fit(y, 'none')
        out_gjr = mf.garch_fit(y, 'none', model_type='gjr')
        out_t = mf.garch_fit(y, 'none', innovation_dist='t')
        assert out['hasStructure'] == 0
        assert out['ARCH_1'] == 0 and out['GARCH_1'] == 0 and out['persistence'] == 0
        assert out['stdsigma'] == 0 and np.isnan(out['ARCHerr_1'])
        assert out_gjr['leverage'] == 0
        assert out_t['distDoF'] == 100
        assert np.isfinite(out['uncondVar'])


def test_structure_found_and_outputs():
    y = _simulate(1500, 0.1, 0.15, 0.75, seed=5)
    out = mf.garch_fit(y, 'none')
    assert out['hasStructure'] == 1
    assert 0.4 < out['persistence'] < 1 and out['ARCH_1'] > 0.05
    assert out['persistence'] == pytest.approx(out['GARCH_1'] + out['ARCH_1'])
    assert out['LLF'] < 0 and np.isfinite(out['ARCHerr_1'])
    out_t = mf.garch_fit(y, 'none', innovation_dist='t')
    out_gjr = mf.garch_fit(y, 'none', model_type='gjr')
    assert out_t['LLF'] >= out['LLF'] - 1e-3   # (t nests the Gaussian as DoF -> inf, bounded here at 100)
    assert out_gjr['LLF'] >= out['LLF'] - 1e-6
    with pytest.raises(ValueError):
        mf.garch_fit(y, 'none', model_type='egarch')


def test_reproducible_under_tiny_perturbation():
    y = _simulate(1000, 0.1, 0.12, 0.8, seed=7)
    a = mf.garch_fit(y, 'none')
    b = mf.garch_fit(y + 1e-9 * np.random.default_rng(0).standard_normal(len(y)), 'none')
    for k in ('ARCH_1', 'GARCH_1', 'persistence', 'LLF', 'maxsigma'):
        assert a[k] == pytest.approx(b[k], abs=1e-5)


def test_garch_compare():
    y = _simulate(1200, 0.1, 0.15, 0.75, seed=6)
    out = mf.garch_compare(y, 'none', (1, 2), (1, 2))
    assert out['hasStructure'] == 1
    assert out['maxLLF'] >= out['minLLF']   # (nested models: the likelihood never decreases with the orders)
    assert out['bestpBIC'] in (0, 1, 2) and out['bestqBIC'] in (0, 1, 2)
    assert np.isfinite(out['Ks_vary_p']) and np.isfinite(out['Ks_vary_q'])
    # white noise: the constant and the residual statistics are those of constant variance, which is chosen
    w = mf.garch_compare(np.random.default_rng(3).standard_normal(1000), 'none', (1, 2), (1, 2))
    assert w['hasStructure'] == 0 and w['Ks_vary_p'] == 0 and w['minK'] == w['maxK']
    assert w['maxLLF'] >= w['minLLF']
    assert (w['bestpAIC'], w['bestqAIC'], w['bestpBIC'], w['bestqBIC']) == (0, 0, 0, 0)


def _rossler_z(n=5000, dt=0.2):
    from scipy.integrate import solve_ivp
    t = np.arange(0, n * dt + 300, dt)
    sol = solve_ivp(lambda _, s: [-s[1] - s[2], s[0] + 0.2 * s[1], 0.2 + s[2] * (s[0] - 5.5)],
                    (0, t[-1]), [1, 1, 0], t_eval=t, rtol=1e-9, atol=1e-9)
    return sol.y[2][-n:]


def test_at_floor():
    assert G.at_floor(np.array([1e-6, 0.1, 0.8])) and G.at_floor(np.array([1.005e-6]))
    assert not G.at_floor(np.array([1.02e-6, 0.1, 0.8]))


def test_floor_gives_nan_for_likelihood_and_variance_path():
    # a flow with long near-zero stretches: the Gaussian likelihood has no maximum, the constant sits at its floor
    z = _rossler_z()
    out = mf.garch_fit(z, 'ar')
    assert out['hasStructure'] == 1
    for key in ('LLF', 'aic', 'bic', 'constant', 'maxsigma', 'minsigma', 'uncondVar', 'zres_meanabs',
                'engle_pval_stde_5', 'ARCHerr_1'):
        assert np.isnan(out[key]), key
    for key in ('ARCH_1', 'GARCH_1', 'persistence', 'offset'):
        assert np.isfinite(out[key]), key
    cmp_ = mf.garch_compare(z, 'ar')
    for key in ('maxLLF', 'bestpBIC', 'bestqAIC', 'minK', 'max_meanarchps', 'Ks_vary_p'):
        assert np.isnan(cmp_[key]), key
    # ordinary series are not affected
    y = _simulate(1000, 0.1, 0.12, 0.8, seed=7)
    assert np.isfinite(mf.garch_fit(y, 'none')['LLF']) and np.isfinite(mf.garch_compare(y, 'none')['maxLLF'])
