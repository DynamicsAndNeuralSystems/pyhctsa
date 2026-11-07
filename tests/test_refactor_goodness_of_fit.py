"""Equivalence and pinned outputs for the operations whose R^2 / adjusted R^2 / RMSE now come
from :func:`pyhctsa.fitting.goodness_of_fit` (duplication-audit theme 7).

Two kinds of check:

- Same-runner reference (the equivalence check): verbatim copies of the inline goodness-of-fit
  code that the refactor replaced (``_ref_*``, from main 0fa9013) are compared with the new code
  on seeded inputs, including a constant y (SST = 0) and dof <= 0, at rtol 1e-12, atol 0, on
  whatever machine runs the test.
- Operation-level pins: values generated from the main checkout at 0fa9013, before the refactor,
  with a tolerance per operation set by how reproducible that operation is across platforms
  (BLAS / LAPACK summation order and optimizer paths differ between runners).
"""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.fitting import goodness_of_fit
from pyhctsa.operations import distribution, graph, scaling, spectral, symbolic


# --- Verbatim copies of the replaced inline code (main 0fa9013) --------------------------------

def _ref_fit_lin_gof(x: np.ndarray, y: np.ndarray) -> tuple:
    # pyhctsa/operations/distribution.py:865-875 (main 0fa9013), the whole function
    n = len(y)
    if n < 3 or not np.ptp(y) > 1e-12 * np.max(np.abs(y)):  # (constant to within rounding)
        return (np.nan,) * 4
    sst = np.sum((y - np.mean(y)) ** 2)
    a, b = np.polyfit(x, y, 1)
    sse = np.sum((y - (a * x + b)) ** 2)
    return a, b, 1 - sse / sst, np.sqrt(sse / (n - 2))


def _ref_polysat_fit(w: np.ndarray, y: np.ndarray) -> dict:
    # pyhctsa/operations/spectral.py:325-362 (main 0fa9013), the whole function
    w = np.asarray(w, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 3 or not np.all(np.isfinite(y)) or not np.ptp(y) > 1e-12 * np.max(np.abs(y)):
        return dict(a=np.nan, b=np.nan, r2=np.nan, rmse=np.nan)
    w2 = w ** 2
    scale = np.max(w2)

    def sse(lb):
        g = w2[:, None] / (np.exp(np.atleast_1d(lb))[None, :] + w2[:, None])
        return np.sum(y ** 2) - (g.T @ y) ** 2 / np.sum(g ** 2, axis=0)

    u = np.linspace(np.log(1e-6 * scale), np.log(1e2 * scale), 200)
    k = int(np.argmin(sse(u)))
    lo, hi = u[max(k - 1, 0)], u[min(k + 1, u.size - 1)]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):  # golden-section search in log(b)
        u1 = hi - phi * (hi - lo)
        u2 = lo + phi * (hi - lo)
        if sse(u1)[0] <= sse(u2)[0]:
            hi = u2
        else:
            lo = u1
    b = np.exp((lo + hi) / 2)
    g = w2 / (b + w2)
    a = np.dot(g, y) / np.dot(g, g)
    s = np.sum((y - a * g) ** 2)
    return dict(a=a, b=b, r2=1 - s / np.sum((y - np.mean(y)) ** 2), rmse=np.sqrt(s / (n - 2)))


def _ref_linear_adjr2(x: np.ndarray, y: np.ndarray) -> float:
    # pyhctsa/operations/symbolic.py:819-824 (main 0fa9013), the whole function
    n = len(y)
    p = np.polyfit(x, y, 1)
    rsq = 1 - np.sum((y - np.polyval(p, x)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return 1 - (1 - rsq) * (n - 1) / (n - 2)


def _ref_simple_fit_gof(dny, dny_fit, num_params):
    # pyhctsa/operations/distribution.py:1638, 1648-1654 (main 0fa9013); returns r2, adjr2 and
    # rmse before the site's own * std(x) scaling, which the refactor kept at the site
    dfe = len(dny) - num_params  # degrees of freedom of the error
    res = dny - dny_fit
    sse = np.sum(res ** 2)
    sstot = np.sum((dny - np.mean(dny)) ** 2)
    with np.errstate(all='ignore'):
        r2 = 1 - sse / sstot
        adjr2 = 1 - (1 - r2) * (len(dny) - 1) / dfe
    rmse = np.sqrt(sse / dfe)
    return float(r2), float(adjr2), float(rmse)


def _ref_visibility_graph_gof(k_prob, k_fit, num_params):
    # pyhctsa/operations/graph.py:296-303 (main 0fa9013), with len(k_vals) == len(k_prob); returns
    # rmse before the site's own * stdk scaling, which the refactor kept at the site
    res = k_prob - k_fit  # residuals, in order of increasing degree
    sse = np.sum(res ** 2)
    sstot = np.sum((k_prob - np.mean(k_prob)) ** 2)
    dfe = len(k_prob) - num_params  # degrees of freedom of the error
    with np.errstate(all='ignore'):
        r2 = 1 - sse / sstot
        adjr2 = 1 - (1 - r2) * (len(k_prob) - 1) / dfe
    rmse = np.sqrt(sse / dfe)
    return r2, adjr2, rmse


def _ref_sinusoid_fit_gof(y, fitted, n_coeff):
    # pyhctsa/operations/spectral.py:1452-1459 (main 0fa9013)
    N = len(y)
    res = y - fitted
    sse = np.sum(res ** 2)
    sstot = np.sum((y - np.mean(y)) ** 2)
    dfe = N - n_coeff  # degrees of freedom of the error
    with np.errstate(all='ignore'):
        r2 = 1 - sse / sstot
        adjr2 = 1 - (1 - r2) * (N - 1) / dfe if dfe > 0 else np.nan
        rmse = np.sqrt(sse / dfe) if dfe > 0 else np.nan
    return float(r2), float(adjr2), float(rmse)


def _ref_mfdfa_r2(log_s, log_fq_q):
    # pyhctsa/operations/scaling.py:648-654 (main 0fa9013), one q column
    r2 = np.nan
    p = np.polyfit(log_s, log_fq_q, 1)
    fitted = np.polyval(p, log_s)
    ss_res = np.sum((log_fq_q - fitted) ** 2)
    ss_tot = np.sum((log_fq_q - np.mean(log_fq_q)) ** 2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    return r2


def _ref_inputs():
    """Seeded (x, y, y_fit) triples: noisy line, noise, near-exact fit, constant y with a
    nonzero fit (SST = 0, SSE > 0), and constant y fitted exactly (SST = SSE = 0)."""
    r = np.random.default_rng(11)
    out = []
    for n in (5, 12, 40, 300):
        x = np.linspace(0.5, 3.0, n)
        y = 0.7 * x - 0.2 + r.standard_normal(n)
        out.append((x, y, y + 0.3 * r.standard_normal(n)))
        out.append((x, np.abs(y) + 0.1, np.abs(y) + 0.1 + 1e-9 * r.standard_normal(n)))
        c = np.full(n, 2.5)
        out.append((x, c, c + 0.1 * r.standard_normal(n)))
        out.append((x, c, c.copy()))
    return out


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-12, atol=0)


def test_reference_whole_functions():
    # _fit_lin_gof, _polysat_fit and _linear_adjr2: old and new function on the same inputs
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        for x, y, _ in _ref_inputs():
            _close(distribution._fit_lin_gof(x, y), _ref_fit_lin_gof(x, y))
            new, old = spectral._polysat_fit(x, y), _ref_polysat_fit(x, y)
            _close([new[k] for k in ('a', 'b', 'r2', 'rmse')], [old[k] for k in ('a', 'b', 'r2', 'rmse')])
            if len(y) >= 3:  # the caller of _linear_adjr2 requires more than 2 points
                _close(symbolic._linear_adjr2(x, y), _ref_linear_adjr2(x, y))


def test_reference_inline_blocks():
    # The inline blocks of simple_fit, visibility_graph, sinusoid_fit and mfdfa against the
    # goodness_of_fit call that replaced each, with the arguments the site now passes
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        for x, y, yf in _ref_inputs():
            n = len(y)
            for m in (2, 3, 5, 6):
                if n - m >= 1:  # simple_fit (dfe < 1 returns early) and visibility_graph
                    # (more distinct degrees than parameters) never reach dof <= 0
                    g = goodness_of_fit(y, yf, m, guard_sst=False, adj_from_rsquare=True)
                    new = (g['rsquare'], g['adjrsquare'], g['rmse'])
                    _close(new, _ref_simple_fit_gof(y, yf, m))
                    _close(new, _ref_visibility_graph_gof(y, yf, m))
            for m in (2, 3, 4, 5, 6, 12, 13):  # sinusoid_fit, including dof = N - m <= 0
                g = goodness_of_fit(y, yf, m, guard_sst=False, adj_from_rsquare=True)
                _close((g['rsquare'], g['adjrsquare'], g['rmse']), _ref_sinusoid_fit_gof(y, yf, m))
            if n >= 8:  # mfdfa fits at least 8 scales
                p = np.polyfit(x, y, 1)
                _close(goodness_of_fit(y, np.polyval(p, x), 2)['rsquare'], _ref_mfdfa_r2(x, y))


def _series():
    r = np.random.default_rng(7)
    wn = r.standard_normal(600)
    e = r.standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    return {'wn': wn, 'ar1': ar}


CASES = {
    'spectral_summaries': lambda y: spectral.spectral_summaries(y),
    'sinusoid_fit_sin1': lambda y: spectral.sinusoid_fit(y, 'sin1'),
    'sinusoid_fit_fourier1': lambda y: spectral.sinusoid_fit(y, 'fourier1'),
    'simple_fit_gauss1': lambda y: distribution.simple_fit(y, 'gauss1'),
    'simple_fit_exp1_ks': lambda y: distribution.simple_fit(y, 'exp1', 0),
    'outlier_include': lambda y: distribution.outlier_include(y),
    'visibility_graph': lambda y: graph.visibility_graph(y, 'horiz'),
    'transition_p_alphabet': lambda y: symbolic.transition_p_alphabet(y),
    'mfdfa': lambda y: scaling.mfdfa(y),
}

# Generated from the main checkout (0fa9013), before the refactor.
EXPECTED = {
    ('spectral_summaries', 'wn'): {'fpolysat_r2': 0.9810690730312851, 'fpolysat_rmse': 0.03413046016877067, 'fpoly2_r2': 0.9987726868073356, 'fpoly2_rmse': 0.008698815144143513},
    ('sinusoid_fit_sin1', 'wn'): {'r2': 0.0014937324451347322, 'adjr2': -0.0018513471781647084, 'rmse': 0.922775328752735},
    ('sinusoid_fit_fourier1', 'wn'): {'r2': 0.022396254165364282, 'adjr2': 0.017475429941364418, 'rmse': 0.9138313255143474},
    ('simple_fit_gauss1', 'wn'): {'r2': 0.9168221256096379, 'adjr2': 0.9092605006650596, 'rmse': 0.04494544570221991},
    ('simple_fit_exp1_ks', 'wn'): {'r2': 0.004903534590019687, 'adjr2': -0.005250510975388201, 'rmse': 0.13638393981312666},
    ('outlier_include', 'wn'): {'nflr2': 0.9485279796940483, 'nflrmse': 6.354357228287337, 'stdrflr2': 0.8334423855477887, 'stdrflrmse': 3.7441908156958306},
    ('visibility_graph', 'wn'): {'dgaussk_r2': 0.7687129256831513, 'dgaussk_adjr2': 0.7398020413935452, 'dgaussk_rmse': 0.10742332995895273, 'dexpk_r2': 0.45992427266315505, 'dexpk_adjr2': 0.428155112231576, 'dexpk_rmse': 0.15925232799392475, 'dpowerk_r2': 0.2385123324434938, 'dpowerk_adjr2': 0.1937189402342876, 'dpowerk_rmse': 0.1890992895590311},
    ('transition_p_alphabet', 'wn'): {'trflin5_adjr2': 0.7844188036710243, 'trflin10adjr2': 0.7844188036710243},
    ('mfdfa', 'wn'): {'meanR2': 0.981515248346813},
    ('spectral_summaries', 'ar1'): {'fpolysat_r2': 0.9806546281820142, 'fpolysat_rmse': 0.08676843718559242, 'fpoly2_r2': 0.7759450219267804, 'fpoly2_rmse': 0.2955811471653132},
    ('sinusoid_fit_sin1', 'ar1'): {'r2': 0.15206778735488746, 'adjr2': 0.14922714342642818, 'rmse': 1.670930286807655},
    ('sinusoid_fit_fourier1', 'ar1'): {'r2': 0.15363678086667654, 'adjr2': 0.14937656332070348, 'rmse': 1.6707835489397702},
    ('simple_fit_gauss1', 'ar1'): {'r2': 0.9484996958868672, 'adjr2': 0.9438178500584006, 'rmse': 0.032014166331249774},
    ('simple_fit_exp1_ks', 'ar1'): {'r2': 0.007916881614282167, 'adjr2': -0.002206415512102655, 'rmse': 0.13324873476439322},
    ('outlier_include', 'ar1'): {'nflr2': 0.9428431376251895, 'nflrmse': 7.073868383570861, 'stdrflr2': 0.8817177567577196, 'stdrflrmse': 3.3010726377579376},
    ('visibility_graph', 'ar1'): {'dgaussk_r2': 0.8415462965525552, 'dgaussk_adjr2': 0.8127365322893834, 'dgaussk_rmse': 0.07998391658633684, 'dexpk_r2': 0.39600470321992254, 'dexpk_adjr2': 0.34567176182158277, 'dexpk_rmse': 0.14951128268332914, 'dpowerk_r2': 0.17434714204398594, 'dpowerk_adjr2': 0.10554273721431817, 'dpowerk_rmse': 0.17480583821464138},
    ('transition_p_alphabet', 'ar1'): {'trflin5_adjr2': 0.9217186292026659, 'trflin10adjr2': 0.9217186292026659},
    ('mfdfa', 'ar1'): {'meanR2': 0.9546859292192903},
}


# Tolerance per operation and output (main-checkout values above, compared across platforms).
# Tier 2, rtol 1e-8: linear least squares (np.polyfit / lstsq), whose result depends on the
# LAPACK build's summation order. Tier 3, rtol 1e-5: iterative optimizers, which stop at a
# convergence tolerance and can take a different path on another platform (the golden-section
# search of _polysat_fit, bf_fit_sinusoids' grid + Gauss-Newton refinement, scipy's
# least_squares, and bf_fit_density_curve's Levenberg-Marquardt). R^2 and adjusted R^2 are
# ratios of order 1 that can be near 0 for a poor fit (adjusted R^2 of -0.002), so they also get
# atol = 1e-9 * 1 (the scale of R^2); RMSEs here are not near zero.
_TIER2 = 1e-8
_TIER3 = 1e-5
RTOL = {
    'spectral_summaries': {'fpolysat_r2': _TIER3, 'fpolysat_rmse': _TIER3,  # golden-section search
                           'fpoly2_r2': _TIER2, 'fpoly2_rmse': _TIER2},      # quadratic lstsq
    'sinusoid_fit_sin1': _TIER3,      # bf_fit_sinusoids
    'sinusoid_fit_fourier1': _TIER3,  # scipy.optimize.least_squares
    'simple_fit_gauss1': _TIER3,      # bf_fit_density_curve (Levenberg-Marquardt)
    'simple_fit_exp1_ks': _TIER3,     # bf_fit_density_curve on a kernel density
    'outlier_include': _TIER2,        # np.polyfit lines through closed-form statistics
    'visibility_graph': _TIER3,       # bf_fit_density_curve on the degree distribution
    'transition_p_alphabet': _TIER2,  # np.polyfit line (_linear_adjr2)
    'mfdfa': _TIER2,                  # np.polyfit of the DFA fluctuation functions
}


def _tol(name, key):
    rtol = RTOL[name][key] if isinstance(RTOL[name], dict) else RTOL[name]
    atol = 1e-9 if 'r2' in key else 0.0  # R^2-type outputs: scale 1 (see above)
    return rtol, atol


@pytest.fixture(scope='module')
def outputs():
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {(name, s): f(y) for s, y in _series().items() for name, f in CASES.items()}
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize('case', list(EXPECTED))
def test_pinned_outputs(outputs, case):
    out = outputs[case]
    for key, val in EXPECTED[case].items():
        rtol, atol = _tol(case[0], key)
        np.testing.assert_allclose(out[key], val, rtol=rtol, atol=atol, err_msg=f'{case} {key}')


def test_zero_total_sum_of_squares():
    # Equal histogram counts (SST = 0) and a constant series keep the plain floating-point
    # result -inf, not the NaN of goodness_of_fit's default guard (values from main, 0fa9013).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = np.arange(400.0)
        o = distribution.simple_fit(x, 'gauss2')
        assert o['r2'] == -np.inf and o['adjr2'] == -np.inf
        # The RMSE of a near-exact fit is near zero (rounding error of the Levenberg-Marquardt
        # fit), so it is pinned with an absolute tolerance: 1e-9 times its scale, the uniform
        # density 1/ptp(x) times the std(x) the site multiplies by (about 0.29).
        scale = np.std(x, ddof=1) / np.ptp(x)
        np.testing.assert_allclose(o['rmse'], 2.440727683641595e-06, rtol=0, atol=1e-9 * scale)
        o = spectral.sinusoid_fit(np.ones(50), 'sin1')
        assert o['r2'] == -np.inf and o['adjr2'] == -np.inf
        # bf_fit_sinusoids is an iterative optimizer: tier 3
        np.testing.assert_allclose(o['rmse'], 0.9831736241435396, rtol=_TIER3, atol=0)


def test_goodness_of_fit_conventions():
    y = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    yh = np.array([1.2, 2.1, 3.3, 3.9, 4.6])
    sse = np.sum((y - yh) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    g = goodness_of_fit(y, yh, 2)
    assert g['adjrsquare'] == 1 - sse * 4 / (sst * 3)
    assert goodness_of_fit(y, yh, 2, adj_from_rsquare=True)['adjrsquare'] == 1 - (1 - r2) * 4 / 3
    assert g['rsquare'] == r2 and g['rmse'] == np.sqrt(sse / 3)
    c = np.ones(5)
    assert np.isnan(goodness_of_fit(c, yh, 2)['rsquare'])
    with np.errstate(all='ignore'):
        assert goodness_of_fit(c, yh, 2, guard_sst=False)['rsquare'] == -np.inf
        assert np.isnan(goodness_of_fit(c, c, 2, guard_sst=False)['rsquare'])
    assert np.isnan(goodness_of_fit(y, yh, 5, guard_sst=False)['adjrsquare'])
