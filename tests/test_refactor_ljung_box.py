"""Pinned outputs for the shared Ljung-Box helper (``utils.ljung_box``, duplication-audit theme 13).

The expected values were generated from the main checkout (0fa9013) before the three
Ljung-Box / portmanteau computations were merged into ``utils.ljung_box`` and
``utils._lagged_products``; the refactor reproduced every output bit for bit.
"""
import warnings

import numpy as np
import pytest

from pyhctsa import utils as U
from pyhctsa.operations import _ar_stepwise as AS
from pyhctsa.operations import hypothesis_tests as H
from pyhctsa.operations import model_fit as MF

RTOL = 1e-12


def _series():
    rng = np.random.default_rng(13)
    e = rng.standard_normal(400)
    a = np.zeros(400)
    for t in range(1, 400):
        a[t] = 0.8 * a[t - 1] + e[t]
    return {'ar1': (a - a.mean()) / a.std(ddof=1), 'heavy': rng.standard_t(3, 300),
            'short': rng.standard_normal(60)}


SERIES = _series()

# Values from the main checkout (0fa9013), before the refactor.
EXPECTED = {
    'ar1': {'indep': 8.01654818828701e-122, 'lbp': 2.3534308113317144e-118,
            'lbq': [4.64988959138848e-32, 1.0082598189464637e-37, 6.385299629647729e-35,
                    1.517611236570662e-33],
            'arres': 2.913466648362787e-71, 'arfit': 0.468153369255064},
    'heavy': {'indep': 0.19251996153247075, 'lbp': 0.2071743017599773,
              'lbq': [0.6745965607307194, 0.8263588083095549, 0.7355286098174318,
                      0.2537088106524087],
              'arres': 3.7800882409511463e-16, 'arfit': 0.12985801334896604},
    'short': {'indep': 0.3884461774658021, 'lbp': 0.5572478485418042,
              'lbq': [0.9862626812732126, 0.9853808757850935, 0.5353551729388424,
                      0.7474046980400187],
              'arres': 0.07972843044805492, 'arfit': 0.40947516925893396},
}


@pytest.mark.parametrize('name', list(SERIES))
def test_pinned_outputs(name):
    x, exp = SERIES[name], EXPECTED[name]
    np.testing.assert_allclose(H.independence_tests(x, 'lbq'), exp['indep'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(U.ljung_box_pvalue(x, 5, 2), exp['lbp'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(MF._lbq_test_pvalues(x ** 2)[[0, 4, 9, 19]], exp['lbq'],
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(AS._arfit_residuals(0.1, np.array([0.5, -0.2]), x)[0],
                               exp['arres'], rtol=RTOL, atol=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        siglev = MF.ar_fit(x, 1, 8, 'sbc')['res_siglev']
    np.testing.assert_allclose(siglev, exp['arfit'], rtol=RTOL, atol=0)


def test_ljung_box_lags_and_model_df():
    x = SERIES['heavy']
    q_all, p_all = U.ljung_box(x, 10, model_df=3)
    q_sel, p_sel = U.ljung_box(x, [2, 7, 10], model_df=3)
    np.testing.assert_array_equal(q_sel, q_all[[1, 6, 9]])
    np.testing.assert_array_equal(p_sel, p_all[[1, 6, 9]])
    assert np.isnan(p_all[:3]).all() and np.isfinite(p_all[3:]).all()  # no positive dof at lags <= 3


def test_ljung_box_matches_statsmodels():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = SERIES['heavy'] ** 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = acorr_ljungbox(x, lags=20, model_df=2, return_df=True)
    q, p = U.ljung_box(x, 20, model_df=2, later_first=True)
    np.testing.assert_allclose(q, ref['lb_stat'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(p, ref['lb_pvalue'], rtol=RTOL, atol=0)
