"""Accuracy tests for the clean-room short statistics (work package B3): Kaplan's statistics,
nsAMDF curve helpers, fractal-dimension error, approximate entropy, x_corr."""
import json
from pathlib import Path

import numpy as np
import pytest

from pyhctsa.operations.correlation import crinkle_statistic, theiler_q, time_rev_kaplan
from pyhctsa.operations.distribution import outlier_include  # noqa: F401  (x_corr caller)
from pyhctsa.operations.entropy import approximate_entropy, permutation_entropy
from pyhctsa.operations.nonlinearity import (
    _fractal_dim_error, _increment_pnorm_curve, fractal_dimensions, nsamdf)
from pyhctsa.utils import x_corr

SP = Path('/Users/benfulcher/ClaudeCode/pyhctsa-cleanroom-work')
SPEC = SP / 'cleanroom/specs/thirdparty'
FIX = SPEC / 'thirdparty_blackbox.json'
INPUTS = SPEC / 'gen/thirdparty_inputs.json'

pytestmark = pytest.mark.skipif(not FIX.exists(), reason='clean-room fixtures not available')


def _load():
    return json.load(open(FIX)), json.load(open(INPUTS))


def _series(inputs, name):
    return np.asarray(inputs[name], dtype=float).ravel()


def _close(a, b, tol=1e-12):
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.all(np.abs(a[m] - b[m]) <= tol * np.maximum(1, np.abs(b[m])))


# ---- Kaplan --------------------------------------------------------------------------
def test_kaplan_hand_case():
    x = np.array([1, 2, 4, 8, 3, 1, 0, 5], float)
    assert theiler_q(x) == pytest.approx(8.586957361877015, abs=1e-12)
    assert crinkle_statistic(x) == pytest.approx(36.833333333333336, abs=1e-12)
    tr = [time_rev_kaplan(x, L) for L in (1, 2, 3, 4)]
    _close(tr, [2.6666666666666665, -19.75, 27.5, np.nan])


def test_kaplan_undefined_is_nan():
    z = np.full(20, 3.0)
    assert np.isnan(crinkle_statistic(z))
    assert np.isnan(theiler_q(np.zeros(20)))
    assert np.isfinite(time_rev_kaplan(np.arange(9.0), 4))


def test_kaplan_fixture():
    fx, inputs = _load()
    for rec in fx['kaplan']:
        x = np.asarray(rec['x'] if 'x' in rec else _series(inputs, rec['series']), float).ravel()
        _close(theiler_q(x), rec['theilerQ'])
        _close(crinkle_statistic(x), rec['crinkle'])
        _close([time_rev_kaplan(x, L) for L in (1, 2, 3, 4)], rec['timerev'])


# ---- nsAMDF ---------------------------------------------------------------------------
def test_nsamdf_unit():
    c = _increment_pnorm_curve(np.array([0, 1, 0, 1.0]), 2, 2)
    _close(c, [0, np.sqrt(3), 0])


def test_nsamdf_fixture():
    fx, inputs = _load()
    for rec in fx['nsamdf']:
        y = _series(inputs, rec['series'])
        out = nsamdf(y, 2, 10, 0.5, 7)
        _close(out['L'] if isinstance(out, dict) else out, rec['L'])
        out = nsamdf(y, 1, 5, 0.25, 4)
        _close(out['L'] if isinstance(out, dict) else out, rec['L_b'])


# ---- fractal dimension error -----------------------------------------------------------
@pytest.mark.parametrize('g', [0.0, 1.0, 3.0])
def test_fractal_dim_error_minimum_at_true_d(g):
    from scipy.special import digamma, gammaln
    d0, kmin, kmax = 2.5, 3, 10
    k = np.arange(1, kmax + 1, dtype=float)
    z = np.exp((digamma(k) - digamma(1)) / d0) if g == 0 else \
        np.exp((gammaln(k + g / d0) - gammaln(k)) / g)
    mom = 0.3 * z
    e_true = _fractal_dim_error(d0, g, kmin, kmax, mom)
    assert e_true < 1e-6
    assert _fractal_dim_error(1.2, g, kmin, kmax, mom) > e_true
    assert _fractal_dim_error(6.0, g, kmin, kmax, mom) > e_true


def test_fractal_dimensions_fixture():
    fx, inputs = _load()
    for rec in fx['NL_FractalDimensions']:
        y = _series(inputs, rec['series'])
        out = fractal_dimensions(y, 2, 10, -1, 1, 5, ('ac', 1), 32, (1, 5))
        for key, ref in rec['fd_a'].items():
            _close(out[key], ref, 1e-7)


# ---- ApEn and permutation entropy --------------------------------------------------------
def test_apen_fixture():
    fx, inputs = _load()
    for rec in fx['EN_ApEn']:
        y = _series(inputs, rec['series'])
        _close(approximate_entropy(y, 1, 0.1), rec['apen_1_01'])
        _close(approximate_entropy(y, 2, 0.1), rec['apen_2_01'])
        _close(approximate_entropy(y, 1, 0.2), rec['apen_1_02'])
        _close(approximate_entropy(y, 2, 0.2), rec['apen_2_02'])
        _close(approximate_entropy(y, 2, 0.2, 3), rec['apen_2_02_tau3'])
        _close(approximate_entropy(y, 2, 0.2, 'ac1e'), rec['apen_2_02_ac1e'])


def test_apen_matches_bruteforce_pincus():
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(120), 1)  # ties, so boundary matches at exactly r
    m, r, tau = 2, 0.2 * x.std(ddof=1), 2

    def phi(mm):
        n = len(x) - (mm - 1) * tau
        u = np.array([x[i:i + (mm - 1) * tau + 1:tau] for i in range(n)])
        c = [(np.max(np.abs(u - u[i]), axis=1) <= r).mean() for i in range(n)]
        return np.mean(np.log(c))
    assert approximate_entropy(x, m, 0.2, tau) == pytest.approx(phi(m) - phi(m + 1), abs=1e-12)


def test_apen_antropy_public_crosscheck():
    antropy = pytest.importorskip('antropy')
    x = np.random.default_rng(2).standard_normal(300)
    assert approximate_entropy(x, 2, 0.2) == pytest.approx(
        antropy.app_entropy(x, order=2, tolerance=0.2 * x.std(ddof=1)), abs=1e-12)


def test_permutation_entropy_bits():
    x = np.random.default_rng(3).standard_normal(500)
    out = permutation_entropy(x, 3, 1)
    assert np.isfinite(out['normPermEn']) and 0 < out['normPermEn'] <= 1


# ---- x_corr -------------------------------------------------------------------------------
def test_xcorr_fixture():
    fx, _ = _load()
    for rec in fx['xcorr']:
        x, y = np.asarray(rec['x'], float).ravel(), np.asarray(rec['y'], float).ravel()
        lags, c = x_corr(x, y, True, 1)
        _close(lags, rec['lags1'], 0); _close(c, rec['coeff1'])
        lags, c = x_corr(x, y, True, 5)
        _close(c, rec['coeff5'])
        lags, c = x_corr(x, y, False, 5)
        _close(c, rec['none5'], 1e-12)


def test_xcorr_definition():
    rng = np.random.default_rng(4)
    x, y = rng.standard_normal(30) + 2, rng.standard_normal(30) - 1
    lags, c = x_corr(x, y, normed=False, max_lags=3)
    for k, ck in zip(lags, c):
        n = np.arange(30)
        ok = (n + k >= 0) & (n + k < 30)
        assert ck == pytest.approx(np.sum(x[n[ok] + k] * y[n[ok]]))
