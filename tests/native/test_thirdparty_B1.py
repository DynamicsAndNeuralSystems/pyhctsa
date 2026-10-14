"""Accuracy tests of the stepwise AR fit (``pyhctsa.operations._ar_stepwise``): fixtures from the
reference implementation (black-box recordings), simulation and analytic properties."""
import json
import os

import numpy as np
import pytest
from scipy import stats

from pyhctsa.operations._ar_stepwise import _arfit, _arfit_modes, _arfit_residuals
from pyhctsa.operations.model_fit import ar_fit

SPEC = os.environ.get(
    'THIRDPARTY_SPEC_DIR', '/Users/benfulcher/ClaudeCode/pyhctsa-cleanroom-work/cleanroom/specs/thirdparty')
_BB = os.path.join(SPEC, 'thirdparty_blackbox.json')
_IN = os.path.join(SPEC, 'gen', 'thirdparty_inputs.json')
have_fixtures = os.path.exists(_BB) and os.path.exists(_IN)
pytestmark = pytest.mark.skipif(not have_fixtures, reason='black-box fixtures not available')

if have_fixtures:
    BB = json.load(open(_BB))
    INPUTS = json.load(open(_IN))
    ROWS = [r for r in BB['arfit'] if 'error' not in r and r['series'] != 'const50']


def _id(r):
    return f"{r['series']}-{r['pmin']}-{r['pmax']}-{r['selector']}"


def _sorted(x):  # modes sorted by (tau, per)
    order = np.lexsort((x[0][1], x[1][1]))
    return order


@pytest.mark.parametrize('row', ROWS if have_fixtures else [], ids=_id)
def test_arfit_fixture(row):
    v = np.array(INPUTS[row['series']], dtype=float)
    w, A, C, sbc, fpe, (dof, Uinv) = _arfit(v, row['pmin'], row['pmax'], row['selector'])
    np.testing.assert_allclose(A, np.atleast_1d(row['A']), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(C, row['C'], rtol=1e-9)
    np.testing.assert_allclose(sbc, np.atleast_1d(row['sbc']), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fpe, np.atleast_1d(row['fpe']), rtol=1e-9, atol=1e-10)
    assert dof == row['dof']
    np.testing.assert_allclose(Uinv, np.atleast_2d(row['Uinv']), rtol=1e-8, atol=1e-11)
    if 'res' not in row:
        return
    siglev, res = _arfit_residuals(w, A, v)
    np.testing.assert_allclose(res, row['res'], atol=1e-10)
    np.testing.assert_allclose(siglev, row['siglev'], rtol=1e-8, atol=1e-14)
    per, tau, exctn, lam = _arfit_modes(A, C, (dof, Uinv))
    ref_per = np.array(row['per'], dtype=float).reshape(2, -1)
    ref_tau = np.array(row['tau'], dtype=float).reshape(2, -1)
    ref_ex = np.atleast_1d(np.array(row['exctn'], dtype=float))
    ref_lam = np.atleast_1d(row['lambda_re']) + 1j * np.atleast_1d(row['lambda_im'])
    mine = np.lexsort((per[0], tau[0]))
    ref = np.lexsort((ref_per[0], ref_tau[0]))
    np.testing.assert_allclose(per[:, mine], ref_per[:, ref], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(tau[:, mine], ref_tau[:, ref], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(exctn[mine], ref_ex[ref], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(lam[mine], ref_lam[ref], rtol=1e-8, atol=1e-10)
    a_err = stats.t.ppf(0.975, dof) * np.sqrt(np.diag(Uinv) * C)
    np.testing.assert_allclose(a_err, np.atleast_1d(row['Aerr']), rtol=1e-8)


def test_too_short_series_raises():
    with pytest.raises(ValueError, match=r'12 samples gives 4 equations, too few'):
        _arfit(np.random.default_rng(0).standard_normal(12), 1, 8)


def test_constant_series_is_finite_and_ar_fit_nan():
    v = np.full(50, 3.0)
    _, A, C, sbc, fpe, _ = _arfit(v, 1, 8)
    assert np.all(np.isfinite(sbc)) and np.isfinite(C)
    np.testing.assert_allclose(A, [1 / 3] * 3, atol=1e-6)
    assert all(np.isnan(x) for x in ar_fit(v).values())


def test_invalid_arguments():
    v = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError):
        _arfit(v, 0, 3)
    with pytest.raises(ValueError):
        _arfit(v, 4, 3)
    with pytest.raises(ValueError):
        _arfit(v, 1, 3, 'aic')


@pytest.mark.skipif(not have_fixtures, reason='fixtures')
def test_MF_arfit_fixture():
    for row in BB['MF_arfit']:
        got = ar_fit(np.array(INPUTS[row['series']], dtype=float), 1, 8, 'sbc')
        assert isinstance(got, dict)
        for key, ref in row['out'].items():
            if key not in got:
                continue
            np.testing.assert_allclose(got[key], ref, rtol=1e-8, atol=1e-10, equal_nan=True,
                                       err_msg=f"{row['series']} {key}")


def _simulate_ar(phi, n, rng, burn=500):
    from scipy.signal import lfilter
    e = rng.standard_normal(n + burn)
    return lfilter([1.0], np.r_[1.0, -np.asarray(phi)], e)[burn:]


def test_ar2_recovery_within_three_standard_errors():
    phi = np.array([1.2, -0.5])
    v = _simulate_ar(phi, 5000, np.random.default_rng(1))
    _, A, C, sbc, fpe, (dof, Uinv) = _arfit(v, 1, 6)
    assert A.size == 2 and sbc.size == fpe.size == 6
    se = np.sqrt(np.diag(Uinv) * C)
    assert np.all(np.abs(A - phi) < 3 * se)


def test_intercept_model_recovers_mean_term():
    rng = np.random.default_rng(2)
    n = 4000
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = 2.0 + 0.6 * v[i - 1] + rng.standard_normal()
    w, A, C, *_ , (dof, Uinv) = _arfit(v, 1, 3, zero=False)
    assert Uinv.shape == (A.size + 1,) * 2
    se = np.sqrt(np.diag(Uinv) * C)
    assert abs(w - 2.0) < 3 * se[0] and abs(A[0] - 0.6) < 3 * se[1]
    _, res = _arfit_residuals(w, A, v)
    assert abs(res.mean()) < 0.05


def test_residuals_definition_and_whiteness():
    rng = np.random.default_rng(3)
    v = _simulate_ar([0.7], 2000, rng)
    w, A, *_ = _arfit(v, 1, 5)
    siglev, res = _arfit_residuals(w, A, v)
    p = A.size
    np.testing.assert_allclose(res, v[p:] - sum(A[j] * v[p - 1 - j:len(v) - 1 - j] for j in range(p)))
    assert 0 <= siglev <= 1
    # A deliberately wrong (zero) model leaves strongly autocorrelated residuals
    s2, _ = _arfit_residuals(0.0, np.array([0.0]), v)
    assert s2 < 1e-10
    with pytest.raises(ValueError):
        _arfit_residuals(0.0, np.array([0.5]), v, k=1)
    with pytest.raises(ValueError):
        _arfit_residuals(0.0, np.array([0.5]), v, k=len(v))


def test_modes_complex_pair_period_and_damping():
    phi = np.array([1.2, -0.5])
    v = _simulate_ar(phi, 5000, np.random.default_rng(4))
    _, A, C, _, _, th = _arfit(v, 2, 2)
    per, tau, exctn, lam = _arfit_modes(A, C, th)
    assert per.shape == tau.shape == (2, 2) and np.isclose(exctn.sum(), 1)
    np.testing.assert_allclose(per[0], 2 * np.pi / np.abs(np.angle(lam)))
    np.testing.assert_allclose(tau[0], -1 / np.log(np.abs(lam)))
    # eigenvalues are the roots of z^2 - A1 z - A2
    np.testing.assert_allclose(np.sort_complex(lam), np.sort_complex(np.roots([1, -A[0], -A[1]])), atol=1e-10)
    assert np.all(per[1] > 0) and np.all(tau[1] > 0)


def test_modes_real_eigenvalues_period_conventions():
    # lambda = +0.5 (infinite period) and -0.3 (period 2)
    A = np.array([0.2, 0.15])
    C, th = 1.0, (500, 0.002 * np.eye(2))
    per, tau, exctn, lam = _arfit_modes(A, C, th)
    pos = lam.real > 0
    assert np.isinf(per[0][pos]).all() and per[1][pos][0] == 0
    np.testing.assert_allclose(per[0][~pos], 2.0)
    np.testing.assert_allclose(per[1][~pos], 0.0, atol=1e-6)


def test_first_minimizer_selected():
    v = np.random.default_rng(5).standard_normal(300)
    _, A, _, sbc, _, _ = _arfit(v, 1, 6, 'sbc')
    assert A.size == 1 + int(np.argmin(sbc))
