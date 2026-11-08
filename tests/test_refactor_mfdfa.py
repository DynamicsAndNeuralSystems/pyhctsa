"""Pinned outputs of `mfdfa` and the DFA path of `fluctuation_analysis`, which now share
`scaling._dfa_fluctuation` (the q-th order fluctuation function, with mfdfa's eps floor and
log-domain evaluation as explicit arguments) and `scaling._log_tau_grid` (duplication-audit theme 15).

Two kinds of check:

- Same-runner reference (the equivalence check): verbatim copies of the code the refactor replaced
  (``_ref_*``, from main 0fa9013) are compared with the shared helpers on seeded inputs, including
  windows of exactly zero variance and q <= 0, at rtol 1e-12, atol 0, on whatever machine runs the test.
- Operation-level pins: values generated before the refactor, from origin/dup-goodness-of-fit at
  4a495c2 (output-identical to main at 0fa9013), with a per-operation tolerance (see `RTOL`).
"""
import warnings

import numpy as np
import pytest

from pyhctsa.operations import scaling
from pyhctsa.utils import _arange_inclusive, _round_half_away_array, z_score


# --- Verbatim copies of the replaced code (main 0fa9013) ----------------------------------------

def _ref_dfa_fluctuation(resid: np.ndarray, q: float) -> float:
    # pyhctsa/operations/scaling.py:30-47 (main 0fa9013), the whole function (docstring dropped);
    # used by the DFA path of fluctuation_analysis
    var = np.mean(resid ** 2, axis=0)
    if q == 2:
        return float(np.sqrt(np.mean(var)))
    if q <= 0 and np.any(var <= 0):
        return np.nan
    if q == 0:
        return float(np.exp(0.5 * np.mean(np.log(var))))
    return float(np.mean(var ** (q / 2)) ** (1 / q))


def _ref_mfdfa_scales(min_scale, max_scale, num_scales):
    # pyhctsa/operations/scaling.py:608 (main 0fa9013)
    scales = np.unique(_round_half_away_array(np.exp(np.linspace(np.log(min_scale), np.log(max_scale), num_scales)))).astype(int)
    return scales


def _ref_mfdfa_log_fq_row(detrended_segs, q_true):
    # pyhctsa/operations/scaling.py:613-616, 623, 632-638 (main 0fa9013): ln F_q(s) at one scale,
    # from the detrended segments (segs @ resid_op.T, one segment per row)
    q_list = q_true.copy()
    q_zero = np.flatnonzero(q_true == 0)
    q_list[q_list == 0] = 1e-4  # q = 0 handled separately via log-averaging below
    q_zero_idx = q_zero[0] if q_zero.size else None
    log_fq = np.full(q_list.size, np.nan)
    eps = np.finfo(float).eps
    F2 = np.mean((detrended_segs) ** 2, axis=1)
    F2[F2 < eps] = eps  # floor to avoid log(0) / 0^(negative q)
    for qi, q in enumerate(q_list):
        log_fq[qi] = (1 / q) * np.log(np.mean(F2 ** (q / 2)))
    if q_zero_idx is not None:
        # q = 0 limit: F_0(s) = exp{ (1/(4Ns)) * sum( log F^2(s,v) ) }
        log_fq[q_zero_idx] = np.mean(np.log(F2)) / 2
    return log_fq


def _ref_windows():
    """Seeded detrended-window matrices (segments x window length), some with windows of exactly
    zero variance and some with variances below machine epsilon."""
    r = np.random.default_rng(150)
    out = []
    for nseg, s in ((4, 10), (16, 25), (60, 7)):
        x = r.standard_normal((nseg, s))
        out.append(x)
        z = x.copy()
        z[0] = 0.0
        z[1] *= 1e-9  # variance about 1e-18, below eps
        out.append(z)
        out.append(np.zeros((nseg, s)))
    return out


def test_reference_dfa_fluctuation():
    # fluctuation_analysis's DFA path: _dfa_fluctuation(resid, q) with its defaults (windows as
    # columns), for orders including 2, 0 and negative q on windows of zero variance
    with np.errstate(all='ignore'):
        for w in _ref_windows():
            for q in (-3, -0.5, 0, 0.5, 1, 2, 2.5, 4):
                np.testing.assert_allclose(scaling._dfa_fluctuation(w.T, q), _ref_dfa_fluctuation(w.T, q),
                                           rtol=1e-12, atol=0, err_msg=f'q={q}')


def test_reference_mfdfa_log_fq():
    # mfdfa's per-scale ln F_q(s), now _dfa_fluctuation(..., floor=eps, log=True), with and without q = 0
    eps = np.finfo(float).eps
    with np.errstate(all='ignore'):
        for w in _ref_windows():
            for q_min, q_max in ((-5, 5), (1, 4), (-2, -0.5)):
                q_true = _arange_inclusive(q_min, 0.5, q_max)
                np.testing.assert_allclose(scaling._dfa_fluctuation(w.T, q_true, floor=eps, log=True),
                                           _ref_mfdfa_log_fq_row(w, q_true), rtol=1e-12, atol=0)


def test_reference_scale_grid():
    for lo, hi, n in ((10, 250, 20), (10, 40, 20), (4, 1000, 20), (6, 13, 20)):
        np.testing.assert_array_equal(scaling._log_tau_grid(lo, hi, n), _ref_mfdfa_scales(lo, hi, n))


def _series():
    r = np.random.default_rng(15)
    n = 1000
    wn = r.standard_normal(n)
    e = r.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    # constant runs: segments of exactly zero variance (mfdfa's eps floor; NaN for q <= 0 without it)
    plateau = np.r_[np.zeros(300), r.standard_normal(400), np.ones(300)]
    return {'wn': z_score(wn), 'ar1': z_score(ar), 'plateau': z_score(plateau)}


SERIES = _series()
nan = np.nan

MFDFA = {  # mfdfa(x, q_range=[-5, 5]) (the registered configuration)
    'wn': {'meanR2': 0.9886298710294548, 'h2': 0.4838232324478439, 'alphaMin': 0.4018662397225041,
           'alphaMax': 0.5458344859384807, 'alphaWidth': 0.1439682462159766, 'fAlphaMax': 1.0,
           'alpha0': 0.498706317711614, 'spectrumAsymmetry': 2.05482372077223},
    'ar1': {'meanR2': 0.9876569898085494, 'h2': 0.7982808739921264, 'alphaMin': 0.5780188785190141,
            'alphaMax': 1.066961775063266, 'alphaWidth': 0.48894289654425194, 'fAlphaMax': 1.0,
            'alpha0': 0.8632119610500378, 'spectrumAsymmetry': 1.3997219281510995},
    # alphaMin (about -6e-10, rounding noise around 0) is left out
    'plateau': {'meanR2': 0.6992687707823303, 'h2': 0.8195256726947718, 'alphaMax': 0.9225087835950334,
                'alphaWidth': 0.9225087841982171, 'fAlphaMax': 1.2246616008046711,
                'alpha0': 0.9225087835950334, 'spectrumAsymmetry': nan},
}

MFDFA_Q14 = {  # mfdfa(x, q_range=[1, 4]) (no q = 0)
    'wn': {'meanR2': 0.9853684449040114, 'h2': 0.4838232324478439, 'alphaMin': 0.4265313152995134,
           'alphaMax': 0.48024066517466224, 'alphaWidth': 0.05370934987514886,
           'fAlphaMax': 0.9882855846761263, 'alpha0': 0.48024066517466224, 'spectrumAsymmetry': nan},
    'plateau': {'meanR2': 0.9390530169250716, 'h2': 0.8195256726947718, 'alphaMin': 0.8026572084899419,
                'alphaMax': 0.9233354706713497, 'alphaWidth': 0.12067826218140776,
                'fAlphaMax': 1.2275550055717779, 'alpha0': 0.9233354706713497, 'spectrumAsymmetry': nan},
}

FA_DFA = {  # fluctuation_analysis(x, q=q, wtf='dfa', k=1): (q, series) -> outputs
    (2, 'wn'): {'alpha': 0.4851171084771426, 'ssr': 0.002165427667313735},
    (2, 'ar1'): {'alpha': 0.6715386113380976, 'ssr': 0.004471433845142926,
                 'r1_alpha': 1.1190181093881373, 'r2_alpha': 0.6576952180679665},
    (2, 'plateau'): {'alpha': 0.8769644559307357, 'ssr': 0.10074770985217904,
                     'r1_alpha': 0.6574498947507215, 'r2_alpha': 1.5392654776899903},
    (0, 'wn'): {'alpha': 0.5026075003642442, 'ssr': 0.0015451497722403752},
    (0, 'ar1'): {'alpha': 0.7611622320362383, 'ssr': 0.005612812492862905,
                 'r1_alpha': 1.1066393587779875, 'r2_alpha': 0.7272405525687656},
    # no floor here: zero-variance windows make the q = 0 average undefined
    (0, 'plateau'): {'alpha': nan, 'ssr': nan},
}


# Tolerance of the pins, tier 2 (linear algebra and least squares), rtol 1e-8: every output here is
# computed from DFA fluctuation functions whose detrending is a pinv (mfdfa) or QR (fluctuation_analysis)
# projection, followed by np.polyfit / least-squares line and broken-line fits, so it depends on the
# LAPACK build's summation order. None of the pinned values is near zero (mfdfa's plateau alphaMin,
# rounding noise around 0, is already left out), so no atol is needed.
RTOL = 1e-8


def _check(out, expected):
    for key, val in expected.items():
        np.testing.assert_allclose(out[key], val, rtol=RTOL, atol=0, err_msg=key)


@pytest.mark.parametrize('name', sorted(MFDFA))
def test_mfdfa_pinned(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _check(scaling.mfdfa(SERIES[name], q_range=[-5, 5]), MFDFA[name])


@pytest.mark.parametrize('name', sorted(MFDFA_Q14))
def test_mfdfa_no_q0_pinned(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _check(scaling.mfdfa(SERIES[name], q_range=[1, 4]), MFDFA_Q14[name])


@pytest.mark.parametrize('q, name', sorted(FA_DFA))
def test_fluctuation_analysis_dfa_pinned(q, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _check(scaling.fluctuation_analysis(SERIES[name], q=q, wtf='dfa', k=1), FA_DFA[(q, name)])


def test_dfa_fluctuation_floor_and_log():
    resid = np.array([[0.0, 1.0, -2.0], [0.0, -1.0, 2.0]])  # window variances 0, 1, 4
    eps = np.finfo(float).eps
    assert np.isnan(scaling._dfa_fluctuation(resid, 0))
    assert np.isnan(scaling._dfa_fluctuation(resid, -2))
    np.testing.assert_allclose(scaling._dfa_fluctuation(resid, 2), np.sqrt(5 / 3), rtol=1e-15)
    q = np.array([-2.0, 0.0, 2.0])
    out = scaling._dfa_fluctuation(resid, q, floor=eps, log=True)
    var = np.array([eps, 1.0, 4.0])
    np.testing.assert_allclose(out, [-0.5 * np.log(np.mean(var ** -1)), np.mean(np.log(var)) / 2,
                                     0.5 * np.log(np.mean(var))], rtol=1e-15)
