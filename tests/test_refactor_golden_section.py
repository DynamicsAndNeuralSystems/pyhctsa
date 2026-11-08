"""Equivalence and pinned outputs for the shared golden-section kernel.

``robust._golden_section_min`` is the golden-section search shared by ``robust.bf_exp_fit`` and
``operations.nonlinearity._saturating_exp_fit``; factoring it out must not change any output.

Two checks:

1. Same-runner reference (the equivalence check that matters): the two golden-section loops
   that the kernel replaced are copied verbatim from main 0fa9013 below (``_ref_*``) and must
   give the same minimizer as ``_golden_section_min`` at rtol 1e-12 on several objectives and
   brackets, on whatever machine runs the test.
2. Operation-level pins of every operation that calls the kernel. The values were generated from
   the main checkout before the refactor (main 0fa9013, on Ben's Mac); the tolerance is set per
   output (see ``_rtol``), since other platforms reproduce the fits only to about 1e-6.
"""
import warnings

import numpy as np
import pytest

from pyhctsa.operations.distribution import outlier_include
from pyhctsa.operations.model_fit import loop_local_simple
from pyhctsa.operations.nonlinearity import _saturating_exp_fit, largest_lyap
from pyhctsa.operations.stationarity import std_nth_deriv_change
from pyhctsa.operations.symbolic import transition_p_alphabet
from pyhctsa.robust import _golden_section_min, _sse_of_rate, bf_exp_fit


# --- verbatim copies of the replaced loops (main 0fa9013) -------------------------------------

def _ref_bf_exp_fit_golden(x, y, with_offset, lo, hi):
    # copied from pyhctsa/robust.py lines 390-398 (inside bf_exp_fit) on main 0fa9013
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):  # golden-section search
        b1 = hi - phi * (hi - lo)
        b2 = lo + phi * (hi - lo)
        if _sse_of_rate(np.array([b1]), x, y, with_offset)[0] <= _sse_of_rate(np.array([b2]), x, y, with_offset)[0]:
            hi = b2
        else:
            lo = b1
    b = (lo + hi) / 2
    return b


def _ref_saturating_golden(sse, sgn, lo, hi):
    # copied from pyhctsa/operations/nonlinearity.py lines 2649-2657 (inside _saturating_exp_fit)
    # on main 0fa9013; ``sse`` maps an array of rates to an array of objective values
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):  # golden-section search in log|b|
        u1 = hi - phi * (hi - lo)
        u2 = lo + phi * (hi - lo)
        if sse(sgn * np.exp(u1))[0] <= sse(sgn * np.exp(u2))[0]:
            hi = u2
        else:
            lo = u1
    b = sgn * np.exp((lo + hi) / 2)
    return b


def _saturating_sse(t, p):
    # the objective of _saturating_exp_fit (nonlinearity.py, main 0fa9013, lines 2634-2641)
    def sse(b):
        b = np.atleast_1d(np.asarray(b, dtype=float))
        with np.errstate(all='ignore'):
            g = 1.0 - np.exp(np.outer(t, b))
            den = np.sum(g ** 2, axis=0)
            out = np.sum(p ** 2) - (g.T @ p) ** 2 / den
        out[~(den > 0) | ~np.isfinite(out)] = np.inf
        return out
    return sse


def _series(kind):
    n = 500
    if kind == 'ar1':
        rng = np.random.default_rng(2); e = rng.standard_normal(n); y = np.zeros(n)
        for t in range(1, n): y[t] = 0.8 * y[t - 1] + e[t]
    else:
        rng = np.random.default_rng(3); y = np.sin(2 * np.pi * np.arange(n) / 37) + 0.3 * rng.standard_normal(n)
    return (y - y.mean()) / y.std(ddof=1)


OPS = {
    'outlier_include': lambda y: outlier_include(y, 'abs', 0.01),
    'transition_p_alphabet': lambda y: transition_p_alphabet(y, [2, 3, 4, 5, 6, 7, 8], 1),
    'largest_lyap': lambda y: largest_lyap(y, embed_params=[1, 4]),
    'std_nth_deriv_change': lambda y: std_nth_deriv_change(y),
    'loop_local_simple': lambda y: loop_local_simple(y, 'mean'),
    'bf_exp_fit_offset': lambda y: bf_exp_fit(np.arange(1, 41.), np.abs(y[:40]).cumsum(), True),
    'bf_exp_fit_nooffset': lambda y: bf_exp_fit(np.arange(1, 41.), np.abs(y[:40]).cumsum(), False, 40),
    'saturating_exp_fit': lambda y: _saturating_exp_fit(np.arange(1, 61.), np.abs(y[:60]).cumsum()),
}

EXPECTED = {
    ('ar1', 'outlier_include'): {'mfexpb': 2.1634567328140175, 'mfexpr2': 0.9911909611311535, 'mfexprmse': 0.7171442873618252, 'nfexpb': -1.0080706511701287, 'nfexpr2': 0.9984041450973923, 'nfexprmse': 1.146651044904612, 'stdrfexpb': 1.384716719678781, 'stdrfexpr2': 0.994622981966881, 'stdrfexprmse': 0.5699139149010414},
    ('ar1', 'transition_p_alphabet'): {'meandiagfexp_a': 0.9518112045608399, 'meandiagfexp_b': -0.48341516403030604, 'meandiagfexp_r2': 0.9852357670383904, 'meandiagfexp_adjr2': 0.9822829204460686, 'meandiagfexp_rmse': 0.0159855777998404, 'maxdiagfexp_a': 0.6627644466935889, 'maxdiagfexp_b': -0.3149659921360167, 'maxdiagfexp_r2': 0.9671612239628987, 'maxdiagfexp_adjr2': 0.9605934687554785, 'maxdiagfexp_rmse': 0.02090393261517731, 'trfexp_a': 1.0082756055751136, 'trfexp_b': -0.16230343156568988, 'trfexp_r2': 0.984384226403196, 'trfexp_adjr2': 0.9812610716838353, 'trfexp_rmse': 0.02229656615014708, 'trcovfexp_a': 0.293030648430324, 'trcovfexp_b': -0.7949434129706214, 'trcovfexp_r2': 0.9954647302910079, 'trcovfexp_adjr2': 0.9945576763492094, 'trcovfexp_rmse': 0.001555052634436466, 'stdeigfexp_a': 0.2712747753002735, 'stdeigfexp_b': -0.2251375880682429, 'stdeigfexp_r2': 0.9803837195770687, 'stdeigfexp_adjr2': 0.9764604634924825, 'stdeigfexp_rmse': 0.007024392233588439},
    ('ar1', 'largest_lyap'): {'expfit_a': 1.7490235932201303, 'expfit_b': -0.5837230780759545, 'expfit_r2': 0.9683657465320821, 'expfit_adjr2': 0.9680103054818807, 'expfit_rmse': 0.03996097856526671},
    ('ar1', 'std_nth_deriv_change'): {'fexp_a': 0.2118839880588739, 'fexp_b': 0.6646838923808013, 'fexp_r2': 0.9999845663288156, 'fexp_rmse': 0.21909040472072494, 'minOrder': 1.0, 'minRatio': 1.0, 'overDiffRatio': 240.71698570311383, 'isInterior': 0.0, 'minOrderInterp': 1.0},
    ('ar1', 'loop_local_simple'): {'sws_fexp_b': -0.2851744878047869, 'sws_fexp_r2': 0.9914136402641577, 'sws_fexp_adjr2': 0.9889603946253456, 'sws_fexp_rmse': 0.0038409517999407933},
    ('ar1', 'bf_exp_fit_offset'): {'a': 12.969690451794357, 'b': 0.02360981753328741, 'c': -11.852816441988855, 'r2': 0.9823832576815348, 'adjr2': 0.9814310013399962, 'rmse': 0.8224717614502375},
    ('ar1', 'bf_exp_fit_nooffset'): {'a': 3.0467869552190643, 'b': 0.05038927765029011, 'c': 0.0, 'r2': 0.965693695222583, 'adjr2': 0.9647908977284404, 'rmse': 1.132541772580041},
    ('ar1', 'saturating_exp_fit'): {'a': -41.18772072860475, 'b': 0.010020378065216487, 'r2': 0.9863792774182357, 'adjr2': 0.9861444373737226, 'rmse': 1.1567232948975534},
    ('sin', 'outlier_include'): {'mfexpb': 5.7603541055074325, 'mfexpr2': 0.9950454226778205, 'mfexprmse': 0.5503197200957725, 'nfexpb': -0.09295868693877789, 'nfexpr2': 0.9956318275427171, 'nfexprmse': 2.0614459552156226, 'stdrfexpb': 3.620341483420356, 'stdrfexpr2': 0.9939670331479992, 'stdrfexprmse': 0.7156118867230568},
    ('sin', 'transition_p_alphabet'): {'meandiagfexp_a': 1.1078202637047667, 'meandiagfexp_b': -0.4899688805925899, 'meandiagfexp_r2': 0.9894895581243229, 'meandiagfexp_adjr2': 0.9873874697491876, 'meandiagfexp_rmse': 0.015580048827624322, 'maxdiagfexp_a': 0.8987668567096528, 'maxdiagfexp_b': -0.38717533407006843, 'maxdiagfexp_r2': 0.9858592401120183, 'maxdiagfexp_adjr2': 0.983031088134422, 'maxdiagfexp_rmse': 0.017123514234549653, 'trfexp_a': 1.1954721641468349, 'trfexp_b': -0.17444476661341268, 'trfexp_r2': 0.9899327517815025, 'trfexp_adjr2': 0.987919302137803, 'trfexp_rmse': 0.021563641600966403, 'trcovfexp_a': 0.9316047798137568, 'trcovfexp_b': -1.0120490439898053, 'trcovfexp_r2': 0.9965054899377485, 'trcovfexp_adjr2': 0.9958065879252982, 'trcovfexp_rmse': 0.0028489798883608976, 'stdeigfexp_a': 0.14448166873217663, 'stdeigfexp_b': -0.1153979721630088, 'stdeigfexp_r2': 0.8891459836998379, 'stdeigfexp_adjr2': 0.8669751804398055, 'stdeigfexp_rmse': 0.008223353693886128},
    ('sin', 'largest_lyap'): {'expfit_a': 1.346059264741789, 'expfit_b': -0.6816755822723262, 'expfit_r2': 0.6082017074214228, 'expfit_adjr2': 0.6060128901444475, 'expfit_rmse': 0.09528677603961304},
    ('sin', 'std_nth_deriv_change'): {'fexp_a': 0.21721168024180207, 'fexp_b': 0.6664390290972432, 'fexp_r2': 0.9999934267452425, 'fexp_rmse': 0.14924117800118378, 'minOrder': 1.0, 'minRatio': 1.0, 'overDiffRatio': 295.5752990316364, 'isInterior': 0.0, 'minOrderInterp': 1.0},
    ('sin', 'loop_local_simple'): {'sws_fexp_b': -0.9670774658523957, 'sws_fexp_r2': 0.8511169394157636, 'sws_fexp_adjr2': 0.8085789221059818, 'sws_fexp_rmse': 0.005948922309670753},
    ('sin', 'bf_exp_fit_offset'): {'a': 465.5902802779816, 'b': 0.0016626345112370573, 'c': -465.67102130569714, 'r2': 0.9873575031538504, 'adjr2': 0.9866741249459504, 'rmse': 1.0879503187174921},
    ('sin', 'bf_exp_fit_nooffset'): {'a': 5.493536652770192, 'b': 0.04649545026077186, 'c': 0.0, 'r2': 0.936853324834461, 'adjr2': 0.9351915702248416, 'rmse': 2.399257962993339},
    ('sin', 'saturating_exp_fit'): {'a': -250.87626707363822, 'b': 0.0030088021976658586, 'r2': 0.994299143250739, 'adjr2': 0.994200852617131, 'rmse': 1.1039034615437537},
}


def _rtol(op, key):
    # Tier 3 (iterative optimizer): every output of an exponential fit (bf_exp_fit,
    # _saturating_exp_fit, and the ``*fexp*``/``expfit_*`` keys of the larger operations) comes out
    # of the golden-section search, which amplifies ulp-level platform differences in the SSE;
    # CI drift was up to 2e-6 (ubuntu, bf_exp_fit with offset on ``sin``), so rtol 1e-5.
    if op in ('bf_exp_fit_offset', 'bf_exp_fit_nooffset', 'saturating_exp_fit') or 'exp' in key:
        return 1e-5
    # Tier 1 (closed-form arithmetic): the remaining std_nth_deriv_change keys are ratios of
    # standard deviations of differenced series and orders picked from them (CI drift <= 1e-11).
    return 1e-9


@pytest.mark.parametrize('kind, op', sorted(EXPECTED))
def test_pinned_outputs(kind, op):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = OPS[op](_series(kind))
    for key, want in EXPECTED[(kind, op)].items():
        np.testing.assert_allclose(out[key], want, rtol=_rtol(op, key), atol=0,
                                   err_msg=f'{kind} {op} {key}')


def _fit_data():
    # (x, y) pairs for the bf_exp_fit objective: cumulative sums of the two series, a clean
    # decaying exponential, a noisy growing exponential with offset, and a linear ramp
    rng = np.random.default_rng(11)
    x = np.arange(1, 41.)
    out = [(x, np.abs(_series(k)[:40]).cumsum()) for k in ('ar1', 'sin')]
    out.append((x, 3.0 * np.exp(-0.2 * x)))
    out.append((x, 0.5 * np.exp(0.05 * x) + 2.0 + 0.1 * rng.standard_normal(40)))
    out.append((x, 0.3 * x + 0.05 * rng.standard_normal(40)))
    return out


@pytest.mark.parametrize('with_offset', [True, False])
def test_bf_exp_fit_loop_matches_reference(with_offset):
    # the replaced loop of bf_exp_fit against the kernel, on several objectives, each on the
    # bracket bf_exp_fit itself would use (around the grid minimum) and on wider/edge brackets
    for x, y in _fit_data():
        b_max = 20 / (x.max() - x.min())
        b_grid = np.linspace(-b_max, b_max, 401)
        k = int(np.argmin(_sse_of_rate(b_grid, x, y, with_offset)))
        brackets = [(b_grid[max(k - 1, 0)], b_grid[min(k + 1, b_grid.size - 1)]),
                    (-b_max, b_max), (b_grid[0], b_grid[2]), (0.0, 0.1)]
        for lo, hi in brackets:
            want = _ref_bf_exp_fit_golden(x, y, with_offset, lo, hi)
            got = _golden_section_min(lambda b: _sse_of_rate(np.array([b]), x, y, with_offset)[0], lo, hi)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=f'{lo}, {hi}')


def test_saturating_loop_matches_reference():
    # the replaced loop of _saturating_exp_fit against the kernel, as _saturating_exp_fit calls
    # it (in log|b|, with a sign), on its own objective and on generic vectorized objectives
    t = np.arange(1, 61.)
    objectives = [_saturating_sse(t, np.abs(_series(k)[:60]).cumsum()) for k in ('ar1', 'sin')]
    objectives.append(_saturating_sse(t, 4.0 * (1 - np.exp(-0.1 * t))))
    objectives += [lambda b: np.atleast_1d((np.log(np.abs(b)) + 2.0) ** 2),
                   lambda b: np.atleast_1d(np.abs(b - 0.01)),
                   lambda b: np.atleast_1d(np.cos(3 * b) + 0.1 * b ** 2),
                   lambda b: np.atleast_1d(np.zeros_like(b, dtype=float))]  # all ties
    u = np.linspace(np.log(1e-5), np.log(5.0), 200)
    brackets = [(u[0], u[2]), (u[97], u[99]), (u[-3], u[-1]), (u[0], u[-1])]
    for sse in objectives:
        for sgn in (-1.0, 1.0):
            for lo, hi in brackets:
                want = _ref_saturating_golden(sse, sgn, lo, hi)
                got = sgn * np.exp(_golden_section_min(lambda v: sse(sgn * np.exp(v))[0], lo, hi))
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_golden_section_min():
    # a fixed number of steps on a parabola, and ties go to the left part of the bracket
    assert abs(_golden_section_min(lambda v: (v - 0.3) ** 2, 0.0, 1.0) - 0.3) < 1e-8
    phi = (np.sqrt(5) - 1) / 2
    assert _golden_section_min(lambda v: 0.0, 0.0, 1.0, n_iter=1) == phi / 2
