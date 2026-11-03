"""kpss_test works the same across statsmodels' change of kpss return type (result_object)."""
import warnings

import numpy as np
import pytest
from statsmodels.tsa.stattools import kpss as sm_kpss

import pyhctsa.operations.stationarity as st

HAS_RESULT_OBJECT = st._accepts_result_object(sm_kpss)


def _series():
    rng = np.random.default_rng(1)
    return {"noise": rng.standard_normal(200), "walk": np.cumsum(rng.standard_normal(200))}


def _legacy_stat_pvalue(y, nlags):
    """The statistic and p-value from statsmodels' original tuple return."""
    kwargs = {"result_object": False} if HAS_RESULT_OBJECT else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm_kpss(y, regression="ct", nlags=nlags, **kwargs)
    return res[0], res[1]


def _check_matches_legacy(lags=(0, 3)):
    for y in _series().values():
        for lag in lags:
            out = st.kpss_test(y, lag)
            stat, pvalue = _legacy_stat_pvalue(y, lag)
            assert out["stat"] == stat
            assert out["pValue"] == pvalue
        out = st.kpss_test(y, list(lags))
        stats = [_legacy_stat_pvalue(y, lag)[0] for lag in lags]
        assert out["maxstat"] == max(stats)
        assert out["minstat"] == min(stats)


def test_kpss_test_matches_tuple_values_and_raises_no_future_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _check_matches_legacy()


@pytest.mark.skipif(not HAS_RESULT_OBJECT, reason="needs statsmodels >= 0.15 (KPSSResult)")
def test_kpss_test_works_once_kpss_returns_a_result_object(monkeypatch):
    # statsmodels 0.16 / after July 2027: kpss returns a KPSSResult, which unpacks to only two values

    def future_kpss(x, regression="c", nlags="auto", store=False, *, result_object=None):
        return sm_kpss(x, regression=regression, nlags=nlags, store=store, result_object=True)

    monkeypatch.setattr(st, "kpss", future_kpss)
    _check_matches_legacy()


def test_kpss_test_works_with_statsmodels_before_result_object(monkeypatch):
    # statsmodels < 0.15: no result_object argument, a 4-tuple is returned

    def old_kpss(x, regression="c", nlags="auto", store=False):
        return _legacy_stat_pvalue(x, nlags) + (nlags, {})

    monkeypatch.setattr(st, "kpss", old_kpss)
    assert not st._accepts_result_object(old_kpss)
    _check_matches_legacy()
