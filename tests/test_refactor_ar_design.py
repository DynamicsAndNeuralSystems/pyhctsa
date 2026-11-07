"""Pinned outputs of the operations whose AR lag-design matrix now comes from ``utils._ar_design``.

The refactor shares only the design (the solvers stay with each method), so no output may change.

Two kinds of check:

1. Same-runner reference (the equivalence check): verbatim copies of the pre-refactor inline
   lag designs (from main 0fa9013), including the hand-built ``_lag_matrix`` and the full
   ``_arcov`` and ``_n4_arx_order_compute``, are compared with ``_ar_design`` and the refactored
   code at rtol 1e-12, atol 0, on whatever machine runs the test.
2. Operation-level pins: the expected values were generated from the main checkout (0fa9013)
   before the refactor, on one machine (arm64 macOS); other BLAS builds reproduce the least-squares
   outputs only to about 1e-12, so each operation has a tolerance set by the kind of computation
   (see TOLERANCES).
"""
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pyhctsa.operations import _ar_stepwise as ars
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import pre_process as pp
from pyhctsa.utils import _ar_design


def _series():
    rng = np.random.default_rng(7)
    n = 400
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
    s = np.sin(2 * np.pi * np.arange(n) / 23) + 0.4 * rng.standard_normal(n)
    return {'ar2': x, 'sine': s}


def _z(y):
    return (y - y.mean()) / y.std(ddof=1)


def _arfit_intercept(y):
    w, A = ars._arfit(y, 2, 2, 'sbc', zero=False)[:2]
    return dict(zip(['w', 'A1', 'A2'], np.r_[w, A]))


CASES = {
    'ar_cov': lambda y: mf.ar_cov(_z(y), 3),
    'ar_fit': lambda y: mf.ar_fit(_z(y), 1, 8, 'sbc'),
    'compare_ar': lambda y: mf.compare_ar(_z(y), np.arange(1, 11), 0.5),
    'fit_subsegments_ar': lambda y: mf.fit_subsegments(_z(y), 'ar', 2, 'uniform', (10, 0.1)),
    'fit_subsegments_arcrosspred': lambda y: mf.fit_subsegments(_z(y), 'arcrosspred', 2, 'uniform', 4),
    'state_space_n4sid': lambda y: mf.state_space_n4sid(_z(y), 2),
    'armax': lambda y: mf.armax(_z(y), (2, 2), 0.5),
    'ar_rms_error': lambda y: {'rms': pp._ar_rms_error(y, 3)},
    'arfit_intercept': _arfit_intercept,
    'arfit_residuals': lambda y: {'siglev': ars._arfit_residuals(0.1, np.array([0.5, -0.2]), y)[0]},
}

EXPECTED = {
    ('ar2', 'ar_cov'): {'a1': 1.0, 'a3': 0.3361505635773714, 'ac1': -0.00010890642926765943, 'ac3': 0.026122120803897343, 'meanabs': 0.6551419633614199, 'noisevar': 0.6985016098092028},
    ('ar2', 'ar_fit'): {'A1': 0.6213043443036719, 'aroundmin_sbc': 1.21427989119967, 'maxA': 0.6213043443036719, 'meane': 0.0008246655826442661, 'minsbc': -0.32614027616043195, 'sbc_4': -0.2981068169852348},
    ('ar2', 'compare_ar'): {'aic_n': 2.0, 'bestaic': -0.23609763385913482, 'maxv': 0.8876669982268439, 'meanv': 0.7921560222104703, 'medianv': 0.7780644199835608, 'minstdfromi': 0.0022845008697757736},
    ('ar2', 'fit_subsegments_ar'): {'a_1_max': -0.3859718446332069, 'a_1_min': -0.7824609228598686, 'a_2_max': 0.5444024858846911, 'a_2_min': 0.11685915743663376, 'fpe_max': 1.088856192687056, 'fpe_range': 0.49313727874691893},
    ('ar2', 'fit_subsegments_arcrosspred'): {'excess_ratio': 1.3529854016707128, 'iqroffdiag': 0.08264638274426062, 'range': 0.16302125488057717, 'rangemedian': 0.12863819383731534, 'rangerange': 0.023493531288477087, 'std': 0.0540051335857315},
    ('ar2', 'state_space_n4sid'): {'A_1': 0.17529890388035158, 'ac1': 0.17563294746074318, 'c_2': 7.822351146347693, 'lossfn': 0.6959028148874552, 'noisevar': 0.7029321362499548, 'stde': 0.8935871998835235},
    ('ar2', 'armax'): {'AR_1': -0.7239825052317683, 'ac1': 0.17730426015467501, 'ftbth': 1.0, 'maxonstd': 3.115637743355882, 'noisevar': 0.702599945095397, 'stde': 0.8939332896720091},
    ('ar2', 'ar_rms_error'): {'rms': 0.9279907385696496},
    ('ar2', 'arfit_intercept'): {'A1': 0.6241178721214122, 'A2': -0.33288667168052544, 'w': -0.10821521537925634},
    ('ar2', 'arfit_residuals'): {'siglev': 0.3246428670656602},
    ('sine', 'ar_cov'): {'a1': 1.0, 'a3': -0.2803295797329705, 'ac1': -0.015799851193964657, 'ac3': 0.15349160593333927, 'meanabs': 0.5139025052466873, 'noisevar': 0.4034154222662808},
    ('sine', 'ar_fit'): {'A1': 0.34453504587277756, 'aerr_mean': 0.1027370432253851, 'fpe_8': -1.1737682750715752, 'meanabs': 0.44619320399623064, 'minsbc': -1.0927277830654942, 'sbc_5': -0.9427152180640226},
    ('sine', 'compare_ar'): {'aic_n': 10.0, 'bestaic': -1.0760295130231041, 'maxv': 0.42346324310753647, 'meanv': 0.3709769895700876, 'medianv': 0.36420882820749423, 'minstdfromi': 0.007097107574999143},
    ('sine', 'fit_subsegments_ar'): {'a_1_max': -0.4059555658245152, 'a_1_min': -0.7259044721290966, 'a_2_max': -0.07973861856367521, 'a_2_min': -0.38319354003893413, 'fpe_max': 0.5675963187624438, 'fpe_range': 0.30038860535875855},
    ('sine', 'fit_subsegments_arcrosspred'): {'excess_ratio': 1.101608431858118, 'iqroffdiag': 0.07492303365433917, 'range': 0.15262386860831434, 'rangemedian': 0.1296308397950814, 'rangerange': 0.030158498960165425, 'std': 0.04944747605727809},
    ('sine', 'state_space_n4sid'): {'A_1': 0.9595888493773679, 'ac1': 0.10035241891631096, 'c_2': 4.387769445898068, 'lossfn': 0.2438822394925024, 'noisevar': 0.24634569645707313, 'stde': 0.5199775441624336},
    ('sine', 'armax'): {'AR_1': -1.924715844076748, 'ac1': 0.11049825508812416, 'ftbth': 1.0, 'maxonstd': 3.097165447561861, 'noisevar': 0.2396680429047516, 'stde': 0.5204749815384425},
    ('sine', 'ar_rms_error'): {'rms': 0.5155428876700379},
    ('sine', 'arfit_intercept'): {'A1': 0.5868978481959851, 'A2': 0.2244504030312026, 'w': -0.005858168610528645},
    ('sine', 'arfit_residuals'): {'siglev': 8.209356725504051e-214},
}


# ------------------------------------------------------------------------------
# 1. Same-runner reference: verbatim copies of the pre-refactor lag designs
# ------------------------------------------------------------------------------

def _ref_lag_matrix(y, na, start):
    # The hand-built design: verbatim body of model_fit._lag_matrix, main 0fa9013,
    # model_fit.py lines 4571-4574.
    n_obs = len(y)
    idx = np.arange(start, n_obs)
    x = np.column_stack([y[idx - k] for k in range(1, na + 1)])
    return x, y[idx]


def _ref_arcov(y, p):
    # Verbatim body of model_fit._arcov, main 0fa9013, model_fit.py lines 3350-3362.
    y = np.asarray(y, dtype=float).ravel()
    p = int(p)
    n_obs = len(y)
    if n_obs <= p:
        raise ValueError("The series is too short for the requested AR order")
    if not np.all(np.isfinite(y)):
        return np.full(p, np.nan), np.nan
    win = sliding_window_view(y, p + 1)
    design = win[:, :p][:, ::-1]
    target = win[:, p]
    phi = np.linalg.lstsq(design, target, rcond=None)[0]
    resid = target - design @ phi
    return phi, float(resid @ resid) / (n_obs - p)


def _ref_n4_arx_order_compute(y):
    # Verbatim body of model_fit._n4_arx_order_compute, main 0fa9013, model_fit.py lines 1313-1330.
    _pinv = mf._pinv
    n_obs = len(y)
    lmax = max(min(20, n_obs // 10), 1)
    if n_obs - lmax < lmax + 2:
        return 0
    target = y[lmax:]
    n = len(target)
    lags = sliding_window_view(y, lmax + 1)[:, :lmax][:, ::-1]   # column k-1 = y[t-k]
    floor_s2 = 1e-10 * np.mean(target ** 2)
    aic = np.empty(lmax + 1)
    with np.errstate(divide='ignore'):
        for k in range(lmax + 1):
            if k == 0:
                res = target
            else:
                x = lags[:, :k]
                res = target - x @ (_pinv(x) @ target)
            aic[k] = np.log(max(np.mean(res ** 2), floor_s2)) + 2.0 * k / n
    return int(np.argmin(aic))


def _ref_inline_designs(y, p):
    # The other inline designs, verbatim from main 0fa9013 (each as (regressors, targets)):
    designs = {}
    # model_fit._armax_fit stage 1, model_fit.py lines 368-370:
    m = p
    win = sliding_window_view(y, m + 1)
    x1 = win[:, :m][:, ::-1]
    t1 = win[:, m]
    designs['armax_stage1'] = (x1, t1)
    # model_fit._fb_lsq forward design, model_fit.py lines 2286-2289:
    win = sliding_window_view(y, p + 1)  # row j = y[j:j+p+1]
    x_fwd = win[:, :p][:, ::-1]
    t_fwd = win[:, p]
    designs['fb_lsq_fwd'] = (x_fwd, t_fwd)
    # model_fit.ar_cov residual regressors, model_fit.py line 3408:
    past = sliding_window_view(y, p + 1)[:, :p][:, ::-1]  # y[t-1], ..., y[t-p] for t = p..N-1
    designs['ar_cov_past'] = (past, y[p:])
    # _ar_stepwise._arfit, _ar_stepwise.py lines 90-91 (lag block and target column):
    windows = sliding_window_view(y, p + 1)[:, ::-1]  # row t: v_t, v_{t-1}, ..., v_{t-pmax}
    blocks = [windows[:, 1:], windows[:, :1]]
    designs['arfit'] = (blocks[0], blocks[1][:, 0])
    # _ar_stepwise._arfit_residuals, _ar_stepwise.py lines 169-170 (regressors and target):
    past = sliding_window_view(y, p + 1)  # row: v_{t-p}, ..., v_t
    designs['arfit_residuals'] = (past[:, :p][:, ::-1], past[:, p])
    return designs


@pytest.mark.parametrize('name', ['ar2', 'sine'])
@pytest.mark.parametrize('p', [1, 3, 8])
def test_ar_design_matches_reference(name, p):
    y = _z(_series()[name])
    x, t = _ar_design(y, p)
    for site, (x_ref, t_ref) in _ref_inline_designs(y, p).items():
        np.testing.assert_allclose(x, x_ref, rtol=1e-12, atol=0, err_msg=site)
        np.testing.assert_allclose(t, t_ref, rtol=1e-12, atol=0, err_msg=site)
    for start in (p, p + 5):
        x_new, t_new = mf._lag_matrix(y, p, start)
        x_ref, t_ref = _ref_lag_matrix(y, p, start)
        np.testing.assert_allclose(x_new, x_ref, rtol=1e-12, atol=0)
        np.testing.assert_allclose(t_new, t_ref, rtol=1e-12, atol=0)
        # The solver result on the hand-built design and on the helper's copy:
        np.testing.assert_allclose(mf._lag_regression(y, p, start), mf._pinv(x_ref) @ t_ref,
                                   rtol=1e-12, atol=0)
    phi, s2 = mf._arcov(y, p)
    phi_ref, s2_ref = _ref_arcov(y, p)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(s2, s2_ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize('name', ['ar2', 'sine'])
def test_n4_order_and_ar_cov_match_reference(name, monkeypatch):
    y = _z(_series()[name])
    assert mf._n4_arx_order_compute(y) == _ref_n4_arx_order_compute(y)
    # ar_cov (the near-zero ac1 among its outputs) with the old _arcov patched in:
    new = mf.ar_cov(y, 3)
    monkeypatch.setattr(mf, '_arcov', _ref_arcov)
    ref = mf.ar_cov(y, 3)
    for key in new:
        np.testing.assert_allclose(new[key], ref[key], rtol=1e-12, atol=0, err_msg=key)


# ------------------------------------------------------------------------------
# 2. Operation-level pins (main-checkout values) with per-operation tolerances
# ------------------------------------------------------------------------------

# Every case here is a least-squares AR or state-space fit (lstsq, pinv, SVD-based subspace
# identification), so all use the linear-algebra tier, rtol 1e-8: CI drift was at most 2.5e-12
# (state_space_n4sid c_2). armax's ARMAX fit is a Hannan-Rissanen start refined by
# Levenberg-Marquardt iterations, which is the iterative-optimizer tier, rtol 1e-5.
TOLERANCES = {case: 1e-8 for case in CASES}
TOLERANCES['armax'] = 1e-5
# Outputs near zero get an absolute tolerance of 1e-9 times their scale: the residual ACF values
# (ac1, ac3 of ar_cov, state_space_n4sid and armax) and the mean residual (meane) are computed on
# a z-scored series, so scale 1 (ar2 ar_cov ac1 is -1.09e-4 and drifted 2.5e-16 absolute).
# arfit_residuals' siglev is a p-value that reaches 8e-214 on the sine; its scale is 1 (a
# probability). Other outputs keep atol 0.
NEAR_ZERO = {'ac1', 'ac3', 'meane', 'siglev'}
ATOL = 1e-9 * 1.0


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_pinned_outputs(key):
    series, case = key
    out = CASES[case](_series()[series])
    for name, value in EXPECTED[key].items():
        np.testing.assert_allclose(float(out[name]), value, rtol=TOLERANCES[case],
                                   atol=ATOL if name in NEAR_ZERO else 0,
                                   err_msg=f'{key} {name}')


def test_ar_design_layout():
    y = np.arange(10.0)
    x, t = _ar_design(y, 3)
    np.testing.assert_array_equal(t, y[3:])
    np.testing.assert_array_equal(x, np.column_stack([y[2:-1], y[1:-2], y[:-3]]))
    x, t = _ar_design(y, 2, start=5)
    np.testing.assert_array_equal(t, y[5:])
    np.testing.assert_array_equal(x, np.column_stack([y[4:-1], y[3:-2]]))
