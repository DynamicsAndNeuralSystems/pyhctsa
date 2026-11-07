"""Pinned outputs of the operations whose AR lag-design matrix now comes from ``utils._ar_design``.

The expected values were generated from the main checkout (0fa9013) before the refactor; the
refactor shares only the design (the solvers stay with each method), so they must not change.
"""
import numpy as np
import pytest

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


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_pinned_outputs(key):
    series, case = key
    out = CASES[case](_series()[series])
    for name, value in EXPECTED[key].items():
        np.testing.assert_allclose(float(out[name]), value, rtol=1e-12, atol=0, err_msg=f'{key} {name}')


def test_ar_design_layout():
    y = np.arange(10.0)
    x, t = _ar_design(y, 3)
    np.testing.assert_array_equal(t, y[3:])
    np.testing.assert_array_equal(x, np.column_stack([y[2:-1], y[1:-2], y[:-3]]))
    x, t = _ar_design(y, 2, start=5)
    np.testing.assert_array_equal(t, y[5:])
    np.testing.assert_array_equal(x, np.column_stack([y[4:-1], y[3:-2]]))
