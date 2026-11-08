"""Pins the first-extremum scans after they were merged into ``utils._first_extremum``.

The expected values were generated from the main checkout (0fa9013) before the refactor;
the outputs are lags (integers), NaN, or the series length when no extremum exists.
"""
import numpy as np
import pytest

from pyhctsa.operations.information import first_max, first_min
from pyhctsa.utils import _first_extremum, get_tau, z_score


def _series():
    rng = np.random.default_rng(42)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    sin = np.sin(2 * np.pi * np.arange(300) / 23) + 0.3 * rng.standard_normal(300)
    return {'ar1': z_score(ar), 'sin': z_score(sin), 'trend': np.arange(60.0), 'const': np.ones(30)}


# the registered configurations of first_min / first_max in hctsa.yaml
_CONFIGS = [('ac', None), ('mi-gaussian', None), ('mi-kraskov1', 4), ('mi-hist', 5),
            ('mi-hist', 10), ('mi-kraskov2', 4)]
nan = np.nan
_EXPECTED = {  # name: (first_min per config, first_max per config, get_tau 'mi-gaussian')
    'ar1': ([11, 9, 5, 7, 5, 5], [17, 11, 8, 8, 6, 8], 9),
    'sin': ([11, 6, 6, 7, 6, 6], [23, 11, 11, 11, 11, 11], 6),
    'trend': ([42, nan, 1, 1, 1, nan], [60, nan, 2, 5, 10, nan], nan),
    'const': ([nan, nan, nan, nan, nan, 1], [nan] * 6, nan),
}


@pytest.mark.parametrize('name', list(_EXPECTED))
def test_first_extremum_outputs_pinned(name):
    y = _series()[name]
    mins, maxs, tau = _EXPECTED[name]
    got_min = [first_min(y, w, e) for w, e in _CONFIGS]
    got_max = [first_max(y, w, e) for w, e in _CONFIGS]
    np.testing.assert_array_equal(np.array(got_min, float), np.array(mins, float))
    np.testing.assert_array_equal(np.array(got_max, float), np.array(maxs, float))
    np.testing.assert_array_equal(float(get_tau(y, 'mi-gaussian')), float(tau))


def test_first_extremum_is_lazy_and_keeps_conventions():
    seen = []

    def curve():
        for lag, v in enumerate([3.0, 2.0, 1.0, 2.0, 0.0, 5.0], start=1):
            seen.append(lag)
            yield v

    assert _first_extremum(curve(), 'min', not_found=99) == 3
    assert seen == [1, 2, 3, 4]  # stops right after the minimum is confirmed
    assert _first_extremum([1.0, 2.0, 3.0], 'min', not_found=7) == 1  # rises at lag 2
    assert _first_extremum([1.0, 2.0, 3.0], 'max', not_found=7) == 7  # lag 1 never a maximum
    assert np.isnan(_first_extremum([3.0, 2.0, 1.0], 'min'))  # default fall-through is NaN
    assert np.isnan(_first_extremum([3.0, nan, 1.0, 2.0], 'min', not_found=7))
