"""
Tests for the outputs of the GP operations on a series without correlation structure
(pyhctsa#63).

When the fitted model does not beat its white-noise special case (``_gp_beats_white_noise``), only
the total variance is identified, not its split between the kernel components: the signal
amplitudes are reported as NaN, the noise standard deviation as that of the white-noise model (the
root-mean-square of the fitted values) and ``nlml`` as that of the white-noise model. Series with
structure must give exactly the outputs they gave before the change.
"""
import numpy as np
import pytest

import pyhctsa.operations.model_fit as mf
from pyhctsa.toolboxes.gpml.cov import parse_cov

RTOL = 1e-12
# Pins of iterative GP fits, generated on main 0fa9013 (before the change) on Ben's Mac (one BLAS
# thread); the change reproduced them bit for bit there. The tolerance is loosened to 1e-8 so the
# pins are portable across platforms (the optimizer path differs in the last digits between
# runners).
RTOL_FIT = 1e-8


def _series():
    # (as in tests/test_refactor_gp_layout.py)
    rng = np.random.default_rng(7)
    wn = rng.standard_normal(300)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    return {k: (y - y.mean()) / y.std(ddof=1) for k, y in (('wn', wn), ('ar1', ar))}


SERIES = _series()


def _white_noise(y):
    """Log root-mean-square and per-point white-noise nlml of y, computed independently."""
    ms = np.mean(np.asarray(y) ** 2)
    return 0.5 * np.log(ms), 0.5 * np.log(2 * np.pi * ms) + 0.5


def test_issue_63_reproduction():
    # The reproduction of pyhctsa#63 (seed 7, 300 standard-normal points, z-scored)
    rng = np.random.default_rng(7)
    y = rng.standard_normal(300)
    y = (y - y.mean()) / y.std(ddof=1)
    out = mf.gp_hyperparameters(y, 'covSEiso_covPeriodic_covNoise', 1, 100, 'first')
    log_sd, nlml = _white_noise(y[:100])
    assert out['logh1'] == 0.0                     # SE length scale: the sampling interval (a bound)
    assert np.isnan(out['logh2'])                  # SE amplitude: not identified
    assert np.isnan(out['logh3'])                  # periodic length scale
    assert np.isnan(out['logh4'])                  # period
    assert np.isnan(out['logh5'])                  # periodic amplitude: not identified
    np.testing.assert_allclose(out['logh6'], log_sd, rtol=RTOL, atol=0)   # total noise sd
    np.testing.assert_allclose(out['nlml'], nlml, rtol=RTOL, atol=0)      # white-noise optimum
    for k in ('stde', 'meanabs_std', 'std_mu_data', 'std_S_data', 'maxS', 'minS', 'meanS'):
        assert np.isfinite(out[k]), k             # prediction outputs: still reported


@pytest.mark.parametrize('cov_func', ['covSEiso_covNoise', 'covRQiso_covNoise', 'covMaterniso3_covNoise'])
def test_gp_hyperparameters_white_noise_other_kernels(cov_func):
    y = SERIES['wn']
    out = mf.gp_hyperparameters(y, cov_func, 1, 200, 'first')
    log_sd, nlml = _white_noise(y[:200])
    components = parse_cov(cov_func)[1]
    for i in mf._gp_amp_pos(components):
        assert np.isnan(out[f'logh{i + 1}'])
    for i in mf._gp_noise_pos(components):
        np.testing.assert_allclose(out[f'logh{i + 1}'], log_sd, rtol=RTOL, atol=0)
    assert out['logh1'] == 0.0
    np.testing.assert_allclose(out['nlml'], nlml, rtol=RTOL, atol=0)


def test_gp_fit_across_white_noise():
    # 20 training points of white noise: no structure. The noise is the root-mean-square of the
    # training points; nlml is that of this white-noise model on the whole series.
    y = SERIES['wn']
    out = mf.gp_fit_across(y, 'covSEiso_covNoise')
    tt = mf._even_split_points(0, len(y) - 1, 20, 'floor').astype(int)
    log_sd, _ = _white_noise(y[tt])
    nlml = 0.5 * np.log(2 * np.pi) + log_sd + 0.5 * np.mean(y ** 2) / np.exp(2 * log_sd)
    np.testing.assert_allclose(out['logh1'], np.log(np.mean(np.diff(tt))), rtol=RTOL, atol=0)
    assert np.isnan(out['logh2'])
    np.testing.assert_allclose(out['logh3'], log_sd, rtol=RTOL, atol=0)
    np.testing.assert_allclose(out['nlml'], nlml, rtol=RTOL, atol=0)


# Series with structure: every output as on main 0fa9013 (Ben's Mac, one BLAS thread)
STRUCTURED = [
    (lambda y: mf.gp_hyperparameters(y, 'covSEiso_covNoise', 1, 200, 'first'),
     {'logh1': 0.6171109040249246, 'logh2': -0.16161561102675429, 'logh3': -0.8616060751511386,
      'nlml': 1.029397269591294, 'stde': 0.3282653257772964, 'meanabs_std': 0.5330431288631408,
      'std_mu_data': 0.8394383419213436, 'std_S_data': 0.003712274968268119,
      'maxS': 0.535998370527123, 'minS': 0.4988072493061797, 'meanS': 0.4989946447392767,
      'oosmedabserr': np.nan}),
    (lambda y: mf.gp_hyperparameters(y, 'covSEiso_covPeriodic_covNoise', 1, 100, 'first'),
     {'logh1': 0.3908235127329746, 'logh2': -0.2387798671492472, 'logh3': -0.15538473403730213,
      'logh4': 2.2211549998698223, 'logh5': -0.9829938874908694, 'logh6': -1.1836353141449412,
      'nlml': 0.9393031835549714, 'stde': 0.2065106209812229, 'meanabs_std': 0.4390471736963014,
      'std_mu_data': 0.7899034962905629, 'std_S_data': 0.0034444607798271437,
      'maxS': 0.4044709294729455, 'minS': 0.3799750078019432, 'meanS': 0.380183431346412,
      'oosmedabserr': np.nan}),
    # (60 training points: with the default 20, spaced 15.7 apart, AR(1) with phi 0.8 has no
    # detectable structure)
    (lambda y: mf.gp_fit_across(y, 'covSEiso_covNoise', 60),
     {'stde': 0.8815896046182313, 'meanabs_std': 0.7742250757193035, 'stdmu': 0.5084668466279626,
      'meanS': 0.9121470883915328, 'stdS': 0.006074436771743998, 'nlml': 1.3117309502518246,
      'logh1': 3.969836437534253, 'logh2': -0.49022908538706833, 'logh3': -0.12613370097295276,
      'h_lonN': 0.1765862175595535}),
]


@pytest.mark.parametrize('fn, expected', STRUCTURED)
def test_structured_outputs_unchanged(fn, expected):
    out = fn(SERIES['ar1'])
    assert set(out) == set(expected)
    for k, v in expected.items():
        np.testing.assert_allclose(out[k], v, rtol=RTOL_FIT, atol=0, equal_nan=True, err_msg=k)
