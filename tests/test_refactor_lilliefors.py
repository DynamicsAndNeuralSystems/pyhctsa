"""Pinned outputs of the Lilliefors path (duplication audit theme #22).

``hypothesis_tests._lilliefors_statistic`` now takes its fitted CDF from
``_fit_distribution_cdf`` (normal, exponential and ``evfit`` extreme-value fits) and its
statistic from ``_kstest_effect``, instead of refitting with its own copies (including the
removed ``_ev_mle``). The expected values below were generated from the main checkout before
the refactor (commit 0fa9013), on seeded series, and are checked at rtol 1e-12.
"""
import numpy as np
import pytest

from pyhctsa.operations import hypothesis_tests as H


def _series():
    rng = np.random.default_rng(20261107)
    w = rng.standard_normal(1000)
    e = rng.standard_normal(1000)
    a = np.zeros(1000)
    for t in range(1, 1000):
        a[t] = 0.8 * a[t - 1] + e[t]
    g = rng.gamma(2.0, 1.5, 300)
    return {'white': w, 'ar1': a, 'gamma': g}


CALLS = [('lillie', 'norm'), ('lillie', 'exp'), ('lillie', 'ev'),
         ('ks', 'ev'), ('ks', 'norm'), ('ks', 'exp'), ('chi2gof', 'ev')]

EXPECTED = {'ar1': [0.025228978384886636, np.nan, 0.0742640586927566, 0.0742640586927566,
                    0.025228978384886636, np.nan, 0.13671394055976097],
            'gamma': [0.11942076695461207, 0.13841732089877584, 0.22358082125982726, 0.22358082125982726,
                      0.11942076695461207, 0.13841732089877584, 0.2997818841586364],
            'white': [0.019106636820861334, np.nan, 0.0682473409453169, 0.0682473409453169,
                      0.019106636820861334, np.nan, 0.09773158732094402]}


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_distribution_test_pinned(name):
    x = _series()[name]
    got = [H.distribution_test(x, t, d) for t, d in CALLS]
    np.testing.assert_allclose(got, EXPECTED[name], rtol=1e-12, atol=0)


def test_lilliefors_degenerate_cases():
    # Values on main: fewer than 4 points and a constant series give NaN for norm and ev;
    # the exponential fit of a constant positive series is well defined.
    assert all(np.isnan(H._lilliefors_statistic(np.array([1.0, 2.0, 3.0]), d)) for d in ('norm', 'exp', 'ev'))
    const = np.ones(20)
    assert np.isnan(H._lilliefors_statistic(const, 'norm'))
    assert np.isnan(H._lilliefors_statistic(const, 'ev'))
    np.testing.assert_allclose(H._lilliefors_statistic(const, 'exp'), 0.6321205588285577, rtol=1e-12, atol=0)
    assert np.isnan(H._lilliefors_statistic(-np.abs(np.arange(1.0, 31.0)), 'exp'))  # mean <= 0
    with pytest.raises(ValueError):
        H._lilliefors_statistic(np.arange(10.0), 'gamma')
