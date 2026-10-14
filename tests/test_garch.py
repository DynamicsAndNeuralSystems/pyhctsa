"""GARCH fits (garch_fit, garch_compare): the likelihood convention, parameter recovery, errors at a bound of
0, and the three model types."""
import warnings

import numpy as np
import pytest

from pyhctsa.operations import model_fit as mf

pytest.importorskip('arch')


def _garch11_llf(y, omega, beta, alpha):
    """Gaussian zero-mean GARCH(1,1) log-likelihood with presample variance and squared innovation
    both equal to the mean of the squared series (written out independently of arch)."""
    v0 = np.mean(y ** 2)
    s2_prev, e2_prev, llf = v0, v0, 0.0
    for e in y:
        s2 = omega + beta * s2_prev + alpha * e2_prev
        llf += -0.5 * (np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)
        s2_prev, e2_prev = s2, e ** 2
    return llf


def _simulate_garch11(n, omega, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500)
    y = np.empty(n + 500)
    s2 = omega / (1 - alpha - beta)
    e_prev = 0.0
    for t in range(n + 500):
        s2 = omega + alpha * e_prev ** 2 + beta * s2
        y[t] = np.sqrt(s2) * z[t]
        e_prev = y[t]
    return y[500:]


def _fit(y, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return mf._garch_estimate(y, *args, **kwargs)


def test_likelihood_convention():
    y = _simulate_garch11(1500, 0.1, 0.1, 0.8, seed=3)
    y = (y - y.mean()) / y.std(ddof=1)
    f = _fit(y, 1, 1)
    assert f['llf'] == pytest.approx(_garch11_llf(y, f['constant'], f['garch'][0], f['arch'][0]), abs=1e-8)
    # the conditional variances follow the same recursion
    s2 = f['constant'] + f['garch'][0] * np.mean(y ** 2) + f['arch'][0] * np.mean(y ** 2)
    assert f['sigma2'][0] == pytest.approx(s2, rel=1e-10)


def test_parameter_recovery():
    y = _simulate_garch11(4000, 0.1, 0.1, 0.8, seed=1)
    sd = y.std(ddof=1)
    f = _fit(y / sd, 1, 1)
    assert f['arch'][0] == pytest.approx(0.1, abs=0.04)
    assert f['garch'][0] == pytest.approx(0.8, abs=0.08)
    assert f['constant'] * sd ** 2 == pytest.approx(0.1, rel=0.6)
    se = np.sqrt(np.diag(f['cov']))
    assert np.all(np.isfinite(se)) and np.all(se[1:3] < 0.1)


def test_errors_at_zero_bound():
    # Gaussian noise has no volatility clustering, so a coefficient often sits at its lower bound of 0;
    # its error is still reported (MATLAB's effective bound is 2e-12, where hctsa reports the error).
    for seed in range(4):
        y = np.random.default_rng(seed).standard_normal(1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = mf.garch_fit(y, 'none')
        assert np.isfinite(out['ARCHerr_1']) and np.isfinite(out['GARCHerr_1'])


def test_garch_fit_outputs():
    y = _simulate_garch11(1000, 0.1, 0.15, 0.75, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = mf.garch_fit(y, 'none')
        out_t = mf.garch_fit(y, 'none', innovation_dist='t')
        out_gjr = mf.garch_fit(y, 'none', model_type='gjr')
        out_e = mf.garch_fit(y, 'none', model_type='egarch')   # (the EGARCH backcast is passed as a variance)
    assert 0.4 < out['persistence'] < 1 and out['ARCH_1'] > 0.05
    assert out['persistence'] == pytest.approx(out['GARCH_1'] + out['ARCH_1'])
    assert out['LLF'] < 0 and np.isfinite(out['ARCHerr_1'])
    assert out_t['distDoF'] > 2 and out_t['LLF'] >= out['LLF'] - 1e-3   # (t nests the Gaussian as DoF -> inf)
    assert np.isfinite(out_gjr['leverage']) and out_gjr['LLF'] >= out['LLF'] - 1e-3
    assert np.isfinite(out_e['LLF']) and np.isnan(out_e['persistence'])


def test_garch_compare():
    y = _simulate_garch11(800, 0.1, 0.15, 0.75, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = mf.garch_compare(y, 'none', (1, 2), (1, 2))
    assert out['maxLLF'] >= out['minLLF'] and out['bestpAIC'] in (1, 2) and out['bestqBIC'] in (1, 2)
    assert np.isfinite(out['Ks_vary_p']) and np.isfinite(out['Ks_vary_q'])
