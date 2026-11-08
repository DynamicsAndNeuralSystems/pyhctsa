"""The spread of sliding_window statistics relative to its stationary sampling spread ('stdratio', 'entnorm'):
the decorrelation time that sets the surrogate blocks; ratios near their stationary values at every length,
for smooth, oscillatory and nonlinear processes; the response to a drifting mean, variance and autocorrelation;
invariance to the scale and offset of the series; reproducibility; and the edge cases."""
import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import stationarity as st
from pyhctsa.utils import z_score


def _ar(coefs, n, seed, burn=500):
    e = np.random.default_rng(seed).standard_normal(n + burn)
    return lfilter([1.0], np.r_[1.0, -np.asarray(coefs)], e)[burn:]


def _osc(period, r, n, seed):
    return _ar([2 * r * np.cos(2 * np.pi / period), -r ** 2], n, seed)


def _median_ratio(gen, window_stat, num_seg, inc_move, N, reps=40, seed0=0):
    return np.median([st.sliding_window(z_score(gen(N, seed0 + i)), window_stat, 'stdratio', num_seg, inc_move)
                      for i in range(reps)])


# ------------------------------------------------------------------------------
# the decorrelation time
# ------------------------------------------------------------------------------
def test_decorrelation_time():
    assert st._decorrelation_time(np.random.default_rng(0).standard_normal(5000)) == 1
    # AR(1): |rho_k| = 0.9^k falls below 1/e beyond k = 9.5
    assert 8 <= st._decorrelation_time(_ar([0.9], 20000, 1)) <= 12
    # a lightly damped oscillation: the envelope (0.97^k) decays far more slowly than the first 1/e crossing
    y = _osc(40, 0.97, 20000, 2)
    assert st._decorrelation_time(y) > 3 * st.get_tau(y, 'ac1e')
    assert st._decorrelation_time(np.ones(50)) == 1


# ------------------------------------------------------------------------------
# near the stationary value at every length
# ------------------------------------------------------------------------------
@pytest.mark.parametrize('window_stat, num_seg, inc_move', [('mean', 10, 1), ('AC1', 10, 1), ('std', 10, 1),
                                                            ('specen', 5, 1), ('asymAC1', 10, 1)])
@pytest.mark.parametrize('process', ['ar05', 'ar09', 'osc12'])
def test_stationary_ratio_flat_in_length(window_stat, num_seg, inc_move, process):
    gen = {'ar05': lambda N, s: _ar([0.5], N, s),
           'ar09': lambda N, s: _ar([0.9], N, s),
           'osc12': lambda N, s: _osc(12, 0.9, N, s)}[process]
    short = _median_ratio(gen, window_stat, num_seg, inc_move, 300)
    long = _median_ratio(gen, window_stat, num_seg, inc_move, 3000)
    # the raw spread falls by about sqrt(10) over this range; the ratio stays near 1 (a median over 40
    # series of a ratio with 4-9 degrees of freedom varies by about 10%)
    assert 0.6 < short < 1.5 and 0.6 < long < 1.5
    assert abs(np.log(long / short)) < np.log(1.5)


def test_raw_spread_falls_with_length():
    # the nuisance the ratio removes: the plain 'std' summary is sampling noise that shrinks as 1/sqrt(N)
    raw = [np.median([st.sliding_window(z_score(_ar([0.5], N, i)), 'mean', 'std', 10, 1) for i in range(20)])
           for N in (300, 3000)]
    assert raw[1] < 0.45 * raw[0]


def test_overlapping_windows():
    # windows that overlap (inc_move = 10) have correlated statistics: the null allows for it
    r = _median_ratio(lambda N, s: _ar([0.5], N, s), 'permen', 5, 10, 2000)
    assert 0.7 < r < 1.4


# ------------------------------------------------------------------------------
# the response to change over time
# ------------------------------------------------------------------------------
def test_drift_responses():
    N, rng = 2000, np.random.default_rng(3)
    t = np.linspace(0, 1, N)
    ratios = {'mean': [], 'std': [], 'AC1': []}
    for i in range(10):
        e = rng.standard_normal(N)
        ratios['mean'].append(st.sliding_window(z_score(lfilter([1], [1, -0.5], e) + t), 'mean', 'stdratio', 10, 1))
        ratios['std'].append(st.sliding_window(z_score(e * np.linspace(0.5, 2, N)), 'std', 'stdratio', 10, 1))
        a, x = np.linspace(0.1, 0.8, N), np.zeros(N)
        for k in range(1, N):
            x[k] = a[k] * x[k - 1] + e[k] * np.sqrt(1 - a[k] ** 2)
        ratios['AC1'].append(st.sliding_window(z_score(x), 'AC1', 'stdratio', 10, 1))
    for v in ratios.values():
        assert np.median(v) > 2


def test_steeper_trend_larger_ratio():
    # a trend within each window must not lengthen the decorrelation time and so inflate the null
    N, rng = 1000, np.random.default_rng(4)
    t = np.linspace(-1, 1, N)
    med = []
    for slope in (1, 4, 16):
        r = []
        for i in range(10):
            e = lfilter([1], [1, -0.5], rng.standard_normal(N))
            r.append(st.sliding_window(z_score(e + slope * e.std() * t), 'mean', 'stdratio', 2, 2))
        med.append(np.median(r))
    assert med[0] > 3 and med[1] > 2 * med[0] and med[2] > 2 * med[1]


def test_entnorm_tracks_spread_not_length():
    # the entropy of the window statistics, in units of their stationary spread: flat in length on a
    # stationary series (the plain entropy falls with the log of the shrinking spread)
    def med(how, N):
        return np.median([st.sliding_window(z_score(_ar([0.5], N, i)), 'specen', how, 5, 1) for i in range(20)])
    assert abs(med('entnorm', 3000) - med('entnorm', 300)) < 0.5
    assert med('ent', 300) - med('ent', 3000) > 0.8


# ------------------------------------------------------------------------------
# invariances, reproducibility and edge cases
# ------------------------------------------------------------------------------
@pytest.mark.parametrize('window_stat', ['mean', 'std', 'AC1', 'specen'])
def test_scale_and_offset_invariant(window_stat):
    y = _ar([0.7], 600, 4)
    a = st.sliding_window(y, window_stat, 'stdratio', 10, 1)
    b = st.sliding_window(3.7 * y - 12.0, window_stat, 'stdratio', 10, 1)
    assert a == pytest.approx(b, rel=1e-9)


def test_reproducible():
    y = _ar([0.5], 800, 5)
    assert st.sliding_window(y, 'AC1', 'stdratio', 2, 1) == st.sliding_window(y, 'AC1', 'stdratio', 2, 1)


def test_edge_cases():
    # a constant series: no spread is expected or observed
    assert np.isnan(st.sliding_window(np.zeros(200), 'mean', 'stdratio', 10, 1))
    # every window statistic undefined (asymAC1 lag too long for the windows)
    assert np.isnan(st.sliding_window(_ar([0.5], 40, 6), 'asymAC1', 'stdratio', 10, 1, asym_tau=5))
