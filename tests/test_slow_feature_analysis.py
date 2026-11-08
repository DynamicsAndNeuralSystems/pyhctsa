"""slow_feature_analysis: multi-scale SFA of windowed statistics with a calibrated min-p drift score."""
import numpy as np
import pytest
from scipy.signal import lfilter

from pyhctsa.operations import stationarity as st
from pyhctsa.operations._sfa_null_table import PROBS, QUANTILES, W_NODES

FIELDS = {'driftScore', 'driftWinLength', 'slowPeriod', 'etaEnd', 'etaStd', 'pc1VarFrac', 'slowPCA1corr',
          'slowCorrMean', 'slowCorrVar', 'slowCorrSkew', 'slowCorrAC1', 'slowCorrTrev'}
SCORE_01 = 2.0  # driftScore at p = 0.01
PERIOD_LO, PERIOD_HI = 0.9, 1.1


def _ar1(n, phi, rng):
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 500))[500:]


def _tv_ar1(phi, rng):
    e = rng.standard_normal(len(phi))
    y = np.zeros(len(phi))
    for t in range(1, len(phi)):
        y[t] = phi[t] * y[t - 1] + e[t]
    return y


def _am(n, cycles, rng):
    t = np.arange(n)
    return (1 + 0.6 * np.cos(2 * np.pi * cycles * t / n + rng.uniform(0, 2 * np.pi))) * rng.standard_normal(n)


def _score(y):
    return st.slow_feature_analysis((y - y.mean()) / y.std())['driftScore']


def test_null_table_shape():
    assert list(W_NODES) == [10 * 2 ** j for j in range(8)]
    assert QUANTILES.shape == (len(W_NODES), len(PROBS))
    assert np.all(np.diff(QUANTILES, axis=1) >= 0)
    # under the null, eta1 concentrates around 2 (random order) as the number of windows grows
    med = QUANTILES[:, np.searchsorted(PROBS, 0.5)]
    assert np.all(np.diff(med) > 0) and 0.5 < med[0] < 0.8 and 1.7 < med[-1] < 2.0


@pytest.mark.parametrize("gen", ["white", "ar0.8", "t3"])
@pytest.mark.parametrize("n", [200, 400, 800, 1600, 4000])
def test_null_calibration(gen, n):
    # false-positive rate at p = 0.05 and 0.01 at most nominal (up to sampling error, 3 SE)
    rng = np.random.default_rng([11, n, len(gen)])
    make = {"white": lambda: rng.standard_normal(n),
            "ar0.8": lambda: _ar1(n, 0.8, rng),
            "t3": lambda: rng.standard_t(3, n)}[gen]
    reps = 200
    s = np.array([_score(make()) for _ in range(reps)])
    assert np.all(s >= 0)
    for alpha in (0.05, 0.01):
        bound = alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)
        assert np.mean(s >= -np.log10(alpha)) <= bound


@pytest.mark.parametrize("kind", ["ar_coef", "variance", "mean"])
def test_detects_linear_drift(kind):
    n = 1600
    rng = np.random.default_rng([12, len(kind)])
    make = {"ar_coef": lambda: _tv_ar1(np.linspace(0.2, 0.7, n), rng),
            "variance": lambda: np.linspace(1.0, 1.6, n) * rng.standard_normal(n),
            "mean": lambda: np.linspace(0.0, 0.8, n) + rng.standard_normal(n)}[kind]
    s = np.array([_score(make()) for _ in range(30)])
    assert np.mean(s >= SCORE_01) >= 0.5
    # a strong drift is unmistakable (p < 1e-4)
    big = {"ar_coef": lambda: _tv_ar1(np.linspace(0.0, 0.9, n), rng),
           "variance": lambda: np.linspace(1.0, 3.0, n) * rng.standard_normal(n),
           "mean": lambda: np.linspace(0.0, 3.0, n) + rng.standard_normal(n)}[kind]
    assert _score(big()) > 4


@pytest.mark.parametrize("cycles, n", [(3, 800), (8, 1600)])
def test_detects_slow_amplitude_modulation(cycles, n):
    # about eight windows per cycle are needed: W >= 40 for 3 cycles, W >= 80 for 8
    rng = np.random.default_rng([13, cycles])
    outs = [st.slow_feature_analysis(_am(n, cycles, rng)) for _ in range(20)]
    s = np.array([o['driftScore'] for o in outs])
    assert np.mean(s >= SCORE_01) >= 0.9
    W_best = np.array([n // o['driftWinLength'] for o in outs])
    assert np.median(W_best) >= 8 * cycles / 1.6
    # the slow mode is a variance mode
    assert np.median([o['slowCorrVar'] for o in outs]) > 0.8


@pytest.mark.parametrize("kind, n, period", [("am", 1600, 1600 / 3), ("am", 1600, 200),
                                             ("mean", 1600, 400), ("mean", 4000, 500)])
def test_slow_period(kind, n, period):
    # a slow sinusoidal component of known period: amplitude modulation, or a sinusoidal mean
    rng = np.random.default_rng([20, n, int(period)])
    t = np.arange(n)
    est = []
    for _ in range(20):
        phase = rng.uniform(0, 2 * np.pi)
        if kind == "am":
            y = (1 + 0.6 * np.cos(2 * np.pi * t / period + phase)) * rng.standard_normal(n)
        else:
            y = 0.8 * np.sin(2 * np.pi * t / period + phase) + rng.standard_normal(n)
        est.append(st.slow_feature_analysis(y)['slowPeriod'])
    est = np.array(est)
    assert np.mean(np.isfinite(est)) >= 0.9
    assert PERIOD_LO * period <= np.nanmedian(est) <= PERIOD_HI * period


def test_slow_period_nan_without_drift():
    # NaN unless the drift score is significant at p < 0.05
    rng = np.random.default_rng(21)
    outs = [st.slow_feature_analysis(rng.standard_normal(800)) for _ in range(40)]
    for o in outs:
        assert np.isnan(o['slowPeriod']) == (o['driftScore'] < -np.log10(0.05))
    assert np.mean([np.isnan(o['slowPeriod']) for o in outs]) >= 0.8


def test_drift_win_length_is_a_timescale():
    # a slow mean drift is best seen with few, long windows; a fast modulation with many short ones
    rng = np.random.default_rng(14)
    n = 4000
    slow = st.slow_feature_analysis(np.linspace(0, 3, n) + rng.standard_normal(n))
    fast = st.slow_feature_analysis(_am(n, 20, rng))
    assert slow['driftWinLength'] >= 200 and fast['driftWinLength'] <= 25
    # a monotone drift's "period" is the record length; the modulation's is its own (200)
    assert slow['slowPeriod'] == pytest.approx(n, rel=0.05)
    assert fast['slowPeriod'] == pytest.approx(200, rel=0.05)


def test_outputs_and_ranges():
    rng = np.random.default_rng(15)
    out = st.slow_feature_analysis(np.linspace(0, 2, 1000) + rng.standard_normal(1000))
    assert set(out) == FIELDS and all(np.isfinite(v) for v in out.values())
    assert 0 <= out['etaStd'] and out['etaEnd'] > 0
    assert 0.2 <= out['pc1VarFrac'] <= 1
    for k in FIELDS - {'driftScore', 'driftWinLength', 'slowPeriod', 'etaEnd', 'etaStd', 'pc1VarFrac'}:
        assert 0 <= out[k] <= 1
    assert out['slowCorrMean'] > 0.8


def test_p_value_lookup():
    for i, W in enumerate(W_NODES):
        q = QUANTILES[i]
        assert st._sfa_log10_p(q[-1] + 1, W) == 0.0
        j = np.searchsorted(PROBS, 0.01)
        assert st._sfa_log10_p(q[j], W) == pytest.approx(np.log10(PROBS[j]))
        # power-law tail beyond the table: continuous at its edge, monotone, finite at eta1 = 0
        assert st._sfa_log10_p(q[0], W) == pytest.approx(-5)
        tail = [st._sfa_log10_p(q[0] * f, W) for f in (0.9, 0.5, 0.1, 0.0)]
        assert np.all(np.diff(tail) < 0) and tail[0] < -5 and np.isfinite(tail[-1])


def test_adjacent_window_correction():
    assert st._sfa_adjacent_corr(0.0, 20) == 0.0 and st._sfa_adjacent_corr(-0.5, 20) == 0.0
    # matches the simulated correlation of adjacent window means of an AR(1) process
    rng = np.random.default_rng(16)
    y = _ar1(400_000, 0.8, rng).reshape(-1, 20).mean(axis=1)
    assert st._sfa_adjacent_corr(0.8, 20) == pytest.approx(np.corrcoef(y[1:], y[:-1])[0, 1], abs=0.01)


def test_degenerate_inputs():
    rng = np.random.default_rng(17)
    for y in (np.ones(1000), rng.standard_normal(199)):
        out = st.slow_feature_analysis(y)
        assert set(out) == FIELDS and all(np.isnan(v) for v in out.values())
    # N = 200 is the shortest length (one scale, W = 10)
    out = st.slow_feature_analysis(rng.standard_normal(200))
    assert out['driftWinLength'] == 20 and np.isfinite(out['driftScore'])
    # constant stretches (undefined window skewness, AC1, trev) do not break it; a flat second half
    # is an unmistakable change
    y = rng.standard_normal(1000)
    y[500:] = 0.0
    out = st.slow_feature_analysis(y)
    assert np.isfinite(out['driftScore']) and out['driftScore'] > 4
    # coarsely discretized data (tied window statistics)
    out = st.slow_feature_analysis(np.round(rng.standard_normal(1000)))
    assert np.isfinite(out['driftScore'])
    with pytest.raises(ValueError):
        st.slow_feature_analysis(rng.standard_normal(1000), min_window_length=5)


def test_min_window_length():
    rng = np.random.default_rng(18)
    y = rng.standard_normal(1000)
    assert st.slow_feature_analysis(y, min_window_length=50)['driftWinLength'] >= 50
    # 10 windows of 31 samples need N >= 310
    assert np.isfinite(st.slow_feature_analysis(y[:310], min_window_length=31)['driftScore'])
    assert all(np.isnan(v) for v in st.slow_feature_analysis(y[:309], min_window_length=31).values())


def test_in_default_feature_set():
    from pyhctsa.calculator import FeatureCalculator
    calc = FeatureCalculator()
    labels = [k for k in calc.feature_funcs if 'slow_feature_analysis' in k]
    assert len(labels) == 1
    out = calc.feature_funcs[labels[0]](np.random.default_rng(19).standard_normal(500))
    assert set(out) == FIELDS
