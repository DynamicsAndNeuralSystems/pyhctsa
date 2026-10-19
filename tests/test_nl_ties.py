"""Exact repeats and tied distances in the nonlinear operations that search for neighbors.

Distances within 1e-12 of the range of the data are equal, a pair closer than that is a repeated
point, and equally near neighbors are treated alike. So on series with exact repeats (a periodic
orbit reached after a transient, an integer-valued series) the outputs are identical, up to
rounding, for ``y``, ``3.7 * y - 12`` and ``-y`` (which z-score to the same series up to rounding
and sign); quantities that cannot be estimated from such series are NaN rather than numbers set by
rounding; and on continuous data the outputs are unchanged.
"""
import numpy as np
import pytest

import pyhctsa.operations.nonlinearity as nl
from pyhctsa.robust import bf_random
from pyhctsa.utils import z_score


# ------------------------------------------------------------------------------
# Deterministic test series (the same numbers as in hctsa's Robust_tiesTests)
# ------------------------------------------------------------------------------

def _logistic(r, x0, burn, n):
    v = x0
    for _ in range(burn):
        v = r * v * (1 - v)
    x = np.zeros(n)
    for i in range(n):
        x[i] = v
        v = r * v * (1 - v)
    return x


def _henon_x(n):
    a, b, u, v = 1.4, 0.3, 0.1, 0.1
    for _ in range(100):
        u, v = 1 - a * u * u + v, b * u
    x = np.zeros(n)
    for i in range(n):
        x[i] = u
        u, v = 1 - a * u * u + v, b * u
    return x


def _rk4(f, s, h):
    k1 = f(s)
    k2 = f(s + h / 2 * k1)
    k3 = f(s + h / 2 * k2)
    k4 = f(s + h * k3)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _lorenz_x(n):
    s = np.array([1.0, 1.0, 1.0])
    h = 0.01

    def f(s):
        return np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])

    for _ in range(5000):
        s = _rk4(f, s, h)
    x = np.zeros(n)
    for i in range(n):
        s = _rk4(f, _rk4(f, s, h), h)
        x[i] = s[0]
    return x


def _ar(n):
    e = bf_random(n + 100, 7, 'normal')
    x = np.zeros(n + 100)
    for i in range(1, n + 100):
        x[i] = 0.8 * x[i - 1] + e[i]
    return x[100:]


def _sin_pair(n):
    t7 = np.sin(2 * np.pi * np.arange(7) / 7 + 0.3)
    t20 = 0.5 * np.sin(2 * np.pi * np.arange(20) / 20 + 0.9)
    i = np.arange(n)
    return t7[i % 7] + t20[i % 20]


SERIES = {
    'orbit': _logistic(3.83, 0.3, 0, 1200),  # a period-3 orbit after a short transient
    'integer': np.floor(8 * _logistic(3.9, 0.4, 100, 1500)),  # 9 integer levels
    'sin7': np.tile(np.sin(2 * np.pi * np.arange(7) / 7 + 0.3), 150),  # exactly periodic
    'sin7sin20': _sin_pair(1400),  # exactly periodic, period 140
    'henon': _henon_x(1500),
    'henon_long': _henon_x(5000),
    'logistic4': _logistic(4.0, 0.3141, 200, 5000),
    'lorenz': _lorenz_x(1500),
    'ar': _ar(1500),
}

CALLS = {
    'fractal_dimensions': lambda x: nl.fractal_dimensions(x, 2, 10, -1, 1, 5, ['ac', 1], 32, [1, 5]),
    'largest_lyap': lambda x: nl.largest_lyap(x, -1, ['ac1e', 30], ['ac1e', 1], 3, [1, 4]),
    'return_time': lambda x: nl.return_time(x, 0.02, 100, ['ac', 1], 500, [1, 8]),
    'dvv': lambda x: nl.dvv(x, 3, 100, 2, 50, 10, 0),
    'evt_local_dim': lambda x: nl.evt_local_dim(x, 'ac1e', 3, 0.98, ['ac', 1], 200, 5, 'full'),
    'lyap_spec': lambda x: nl.lyap_spec(x, 1, 3, 30, 'full', ['ac', 1]),
    'tisean_c1': lambda x: nl.tisean_c1(x, 1, [2, 6], 25, 0.1),
}


def _run(name, x):
    out = CALLS[name](z_score(x))
    return {k: float(v) for k, v in out.items()} if isinstance(out, dict) else {'': float(out)}


def _assert_same(a, b, rel):
    assert sorted(a) == sorted(b)
    for k in a:
        assert np.isnan(a[k]) == np.isnan(b[k]), k
        if not np.isnan(a[k]):
            assert a[k] == pytest.approx(b[k], rel=rel, abs=1e-9), k


@pytest.mark.parametrize('series', ['orbit', 'integer', 'sin7sin20'])
@pytest.mark.parametrize('name', list(CALLS))
def test_ties_do_not_depend_on_rounding(name, series):
    # y, 3.7*y - 12 and -y z-score to the same series up to rounding (and sign), and distances
    # do not change sign; lyap_spec adds a dither of fixed sign pattern, so only the affine map
    # (the fits of fractal_dimensions and the exponential fit of largest_lyap are found to the
    # optimizers' tolerance, so those agree to 1e-4)
    x = SERIES[series]
    ref = _run(name, x)
    rel = 1e-4 if name in ('fractal_dimensions', 'largest_lyap') else 1e-6
    _assert_same(ref, _run(name, 3.7 * x - 12), rel)
    if name != 'lyap_spec':
        _assert_same(ref, _run(name, -x), rel)


def test_fractal_dimensions_not_identified_on_a_periodic_orbit():
    # the k-th neighbor distances are all equal or all zero: the moments do not grow with k (the
    # error has its minimum at the upper end of the search interval) or are zero (the error is
    # flat), and D is not identified
    for s in ('sin7', 'sin7sin20'):
        assert all(np.isnan(v) for v in _run('fractal_dimensions', SERIES[s]).values())


def test_return_time_short_lag_shares_are_bounded():
    # the profile of an exactly periodic series is zero over the second half of the lags: all of
    # its mass and variation are at short lags, a share of 1 (never infinite)
    out = _run('return_time', SERIES['sin7sin20'])
    assert out['fracmeanrt'] == 1.0 and out['fracsdrt'] == 1.0
    assert all(not np.isinf(v) for v in out.values())


def test_dvv_target_variance_is_never_negative():
    for s in ('sin7', 'orbit', 'integer', 'sin7sin20'):
        out = _run('dvv', SERIES[s])
        assert out['min'] >= 0
    assert _run('dvv', SERIES['sin7'])['min'] == 0  # targets of equal delay vectors are equal


def test_evt_local_dim_is_nan_for_discrete_distances():
    # the distances to a pole of an integer-valued series take a few values: no continuous tail
    out = _run('evt_local_dim', SERIES['integer'])
    assert np.isnan(out['meanLocalDim'])
    assert out['propValidPoles'] < 0.5


def test_c1_is_nan_on_periodic_orbits():
    # at every mass some reference point coincides with its k-th neighbor
    for s in ('sin7', 'orbit'):
        assert all(np.isnan(v) for v in _run('tisean_c1', SERIES[s]).values())


def test_largest_lyap_on_a_periodic_orbit_is_bounded():
    # repeated points are not neighbors: the nearest neighbor is another state of the orbit, and
    # the divergence curve is bounded rather than built from coincident pairs
    out = _run('largest_lyap', SERIES['sin7sin20'])
    assert np.isfinite(out['maxp']) and abs(out['maxp']) < 1


def test_known_values_on_quantized_chaotic_maps():
    # recording a chaotic map to two decimals creates many ties and repeated points, but must not
    # spoil the estimates: the largest exponent of the logistic map at r = 4 is ln 2 and that of
    # the Henon map about 0.42, and the information dimension of the Henon attractor about 1.2
    x = SERIES['henon_long']
    xq = np.round(x, 2)
    assert _run('largest_lyap', SERIES['logistic4'])['vse_gradient'] == pytest.approx(np.log(2), abs=0.03)
    assert _run('largest_lyap', xq)['vse_gradient'] == pytest.approx(0.42, abs=0.05)
    assert _run('tisean_c1', xq)['mediand'] == pytest.approx(1.22, abs=0.1)
    assert _run('lyap_spec', xq)['LE1'] == pytest.approx(0.42, abs=0.05)
    assert _run('fractal_dimensions', x)['meanDq'] == pytest.approx(_run('fractal_dimensions', xq)['meanDq'], abs=0.1)


# values on continuous series, which have no repeated points or ties
CONTROLS = {
    ('henon', 'fractal_dimensions'): {'maxDq': 1.50163751973429, 'meanDq': 1.38521900426553, 'linfit_a': -0.0758464254148477},
    ('henon', 'largest_lyap'): {'p2': 0.233488691101789, 'maxp': 5.16672404832131, 'vse_gradient': 0.422918769331889},
    ('henon', 'return_time'): {'std': 0.471509540881739, 'fracmeanrt': 0.4590929132545169, 'hhist': 4.4894830130248},
    ('henon', 'dvv'): {'min': 0.00745023096111694, 'mean': 0.712964878798353, 'meanNormCDF': 0.0712135111771347},
    ('henon', 'evt_local_dim'): {'meanLocalDim': 1.41606998272907, 'meanTheta': 0.977744966442953},
    ('henon', 'lyap_spec'): {'LE1': 0.427378712740251, 'LE2': -0.29918177251794, 'KYdim': 2.07526011297902},
    ('henon', 'tisean_c1'): {'bestestd': 1.22310798962911, 'mediand': 1.24000078981573},
    ('lorenz', 'fractal_dimensions'): {'maxDq': 1.97517036771539, 'meanDq': 1.89145847468339, 'linfit_a': 0.0826055941557108},
    ('lorenz', 'largest_lyap'): {'p2': 0.105109627738154, 'maxp': 3.94511600723218, 'vse_gradient': 0.0149060369039576},
    ('lorenz', 'return_time'): {'std': 1.14002587757889, 'fracmeanrt': 0.49613360516532407, 'hhist': 3.7700599732039},
    ('lorenz', 'dvv'): {'min': 0.00072812580103939, 'mean': 0.451756458674478, 'meanNormCDF': 0.170153772311657},
    ('lorenz', 'evt_local_dim'): {'meanLocalDim': 2.20771651829273, 'meanTheta': 0.302932913651838},
    ('lorenz', 'lyap_spec'): {'LE1': 0.0180333937251304, 'LE2': -0.0522034353540037, 'KYdim': 1.34544457855775},
    ('lorenz', 'tisean_c1'): {'bestestd': 2.06240421533585, 'mediand': 2.00544669230779},
    ('ar', 'fractal_dimensions'): {'maxDq': 5.03606326138446, 'meanDq': 4.97704712809368, 'linfit_a': 0.110370161627988},
    ('ar', 'largest_lyap'): {'p2': 0.875545119874215, 'maxp': 2.16537730930472, 'vse_gradient': 0.268767922552235},
    ('ar', 'return_time'): {'std': 0.346663014385052, 'fracmeanrt': 0.5353737761818226, 'hhist': 4.55342960769032},
    ('ar', 'dvv'): {'min': 0.371304904635249, 'mean': 0.697908079471809, 'meanNormCDF': 0.817785250649402},
    ('ar', 'evt_local_dim'): {'meanLocalDim': 3.32596973866923, 'meanTheta': 0.737900979387082},
    ('ar', 'lyap_spec'): {'LE1': 0.0115368355200486, 'LE2': -0.354730457448597, 'KYdim': 1.03252282198441},
    ('ar', 'tisean_c1'): {'bestestd': 2.05444839000702, 'mediand': 3.94965887069702},
}


@pytest.mark.parametrize('series,name', list(CONTROLS))
def test_continuous_series_unchanged(series, name):
    out = _run(name, SERIES[series])
    for k, v in CONTROLS[(series, name)].items():
        assert out[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
