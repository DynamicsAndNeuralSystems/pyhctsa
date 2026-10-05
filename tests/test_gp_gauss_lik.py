"""Exact Gaussian-likelihood inference (gpml infGaussLik) in the gpml port."""
import numpy as np

from pyhctsa.toolboxes.matlab.gpml.gpml import CovSEisoNoise, gp_predict, gp_train
from pyhctsa.operations.model_fit import gp_fit_across, gp_local_prediction


def _data():
    rng = np.random.RandomState(1)
    t = np.floor(np.linspace(1, 200, 20))
    return t, rng.randn(20)


def test_nlz_and_prediction_match_direct_formulae():
    t, y = _data()
    hyp = {'cov': np.array([np.log(10.3), 0.2, -1.1]), 'lik': np.log(0.3), 'mean': np.zeros(0)}
    nlZ, _ = gp_train(hyp, CovSEisoNoise, t, y)
    A = CovSEisoNoise.K(hyp['cov'], t) + np.exp(2 * hyp['lik']) * np.eye(20)
    ref = 0.5 * y @ np.linalg.solve(A, y) + 0.5 * np.linalg.slogdet(A)[1] + 10 * np.log(2 * np.pi)
    assert abs(nlZ - ref) < 1e-10

    ts = np.arange(1, 201.)
    mu, s2, _, fs2 = gp_predict(hyp, CovSEisoNoise, t, y, ts)
    Ks = CovSEisoNoise.K(hyp['cov'], t, ts)
    kss = np.exp(2 * hyp['cov'][1]) + np.exp(2 * hyp['cov'][2])
    assert np.allclose(mu, Ks.T @ np.linalg.solve(A, y), atol=1e-10)
    assert np.allclose(fs2, np.maximum(kss - np.sum(Ks * np.linalg.solve(A, Ks), axis=0), 0), atol=1e-9)
    assert np.allclose(s2, fs2 + np.exp(2 * hyp['lik']))


def test_gradient_matches_finite_differences():
    t, y = _data()
    theta = np.array([np.log(10.3), 0.2, -1.1, np.log(0.3)])

    def f(th):
        return gp_train({'cov': th[:3], 'lik': th[3], 'mean': np.zeros(0)}, CovSEisoNoise, t, y)

    _, d = f(theta)
    g = np.concatenate([d['cov'], d['lik']])
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1e-6
        fd = (f(theta + e)[0] - f(theta - e)[0]) / 2e-6
        assert abs(fd - g[i]) < 1e-6 * max(1, abs(g[i]))


def test_outputs_names_and_determinism():
    rng = np.random.RandomState(0)
    y = np.cumsum(rng.randn(300))
    y = (y - y.mean()) / y.std(ddof=1)
    a = gp_fit_across(y)
    assert set(a) == {'stde', 'meanabs_std', 'stdmu', 'meanS', 'stdS', 'nlml',
                      'logh1', 'logh2', 'logh3', 'h_lonN'}
    assert a == gp_fit_across(y)          # no state carried between calls
    b = gp_local_prediction(y, num_train=10, num_test=3, num_preds=5, pmode='frombefore')
    assert {'maxabs_std', 'meanabs', 'minabs_run', 'maxnlml', 'minnlml', 'stdnlml'} <= set(b)
    assert not any(k.endswith(('stderr', 'abserr', 'mlik')) for k in b)
