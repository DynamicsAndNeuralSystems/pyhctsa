"""Tests for outputs whose definitions were ill-posed on ordinary inputs (unbounded ratios,
values set by the series length, statistics used outside their domain)."""
import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import distribution as di
from pyhctsa.operations import entropy as en
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import symbolic as sy


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def _all_nan(out):
    """Every output undefined (a dict of NaNs, or a bare NaN when no series gives outputs with these arguments)."""
    return all(np.isnan(v) for v in out.values()) if isinstance(out, dict) else bool(np.isnan(out))


def _ar1(phi, n, seed, innovations='gauss'):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 500) if innovations == 'gauss' else rng.exponential(size=n + 500) - 1
    x = np.zeros(n + 500)
    for t in range(1, n + 500):
        x[t] = phi * x[t - 1] + e[t]
    return _zscore(x[500:])


# ------------------------------------------------------------------------------
# binary_stats: pupstat2
# ------------------------------------------------------------------------------
def test_pupstat2_is_a_bounded_contrast_of_the_two_halves():
    rng = np.random.default_rng(0)
    up = _zscore(np.r_[-np.ones(250), np.ones(250)] + 0.1 * rng.standard_normal(500))
    assert sy.binary_stats(up, 'mean')['pupstat2'] == 1.0  # every 1 in the second half (was inf)
    assert sy.binary_stats(up[::-1], 'mean')['pupstat2'] == -1.0
    for seed in range(20):
        y = _zscore(np.random.default_rng(seed).standard_normal(501))  # odd length
        for how in ('mean', 'diff', 'iqr'):
            assert -1 <= sy.binary_stats(y, how)['pupstat2'] <= 1


def test_pupstat2_compares_proportions_and_is_antisymmetric():
    # 3 ones among the first 5 symbols and 3 among the last 6: proportions 0.6 and 0.5
    y = np.array([1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1.0])
    y_bin = (y > np.mean(y)).astype(int)
    assert y_bin[:5].sum() == 3 and y_bin[5:].sum() == 3
    assert sy.binary_stats(y, 'mean')['pupstat2'] == pytest.approx((0.5 - 0.6) / (0.5 + 0.6))
    # reversing an even-length series swaps the halves
    y = _zscore(np.random.default_rng(3).standard_normal(400))
    assert sy.binary_stats(y[::-1], 'mean')['pupstat2'] == pytest.approx(-sy.binary_stats(y, 'mean')['pupstat2'])


# ------------------------------------------------------------------------------
# burstiness and the harmonic mean: statistics of non-negative data
# ------------------------------------------------------------------------------
def test_burstiness_is_bounded_on_its_domain_and_nan_outside_it():
    rng = np.random.default_rng(0)
    for x in (rng.exponential(size=2000), rng.lognormal(size=2000), rng.poisson(3, 2000).astype(float),
              np.r_[0.0, rng.gamma(0.2, size=1999)], np.full(50, 2.0) + 1e-3 * rng.random(50)):
        out = di.burstiness(x)
        r = x.std(ddof=1) / x.mean()
        assert out['B'] == pytest.approx((r - 1) / (r + 1), abs=1e-15)
        assert -1 <= out['B'] <= 1 and -1.01 <= out['B_Kim'] <= 1.01
    assert abs(di.burstiness(rng.exponential(size=20000))['B']) < 0.02  # a Poisson process: 0
    # the pole at mean = -std (was 39, -1.8e16 and -41 here), and any other signed series
    z = _zscore(rng.standard_normal(1000))
    for mu in (-0.95, -1.0, -1.05, 0.0, 0.5):
        out = di.burstiness(z + mu)
        assert np.isnan(out['B']) and np.isnan(out['B_Kim'])
    assert all(np.isnan(v) for v in di.burstiness(np.zeros(10)).values())


def test_harmonic_mean_domain():
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=500)
    hm = di.mean(x, 'harm')
    assert hm == pytest.approx(1 / np.mean(1 / x), rel=1e-13)
    assert hm <= di.mean(x, 'geom') <= di.mean(x, 'norm')
    assert di.mean(np.r_[0.0, x], 'harm') == 0.0
    assert di.mean(np.array([1, 2, 4]), 'harm') == pytest.approx(12 / 7)  # integer input
    # signed data: the sum of reciprocals has poles (was between -184 and 2.0 over four samples of mean 0.1)
    z = _zscore(rng.standard_normal(1000))
    assert np.isnan(di.mean(z + 0.1, 'harm')) and np.isnan(di.mean(z - 1.05, 'harm'))
    assert np.isnan(di.mean(-x, 'harm'))


# ------------------------------------------------------------------------------
# time_rev_kld: needs ten Theiler windows
# ------------------------------------------------------------------------------
def test_time_rev_kld_is_nan_when_the_series_spans_too_few_theiler_windows():
    rng = np.random.default_rng(0)
    walk = _zscore(np.cumsum(rng.standard_normal(5000)))  # reversible, but one correlation time long
    out = co.time_rev_kld(walk, 'ac1e', 2)
    assert np.isnan(out['raw']) and np.isnan(out['abs'])  # (was between -2.4 and 2.4 over eight such walks)
    # an explicit window: 10 * (w + k) embedded points are needed
    y = _ar1(0.5, 500, 1)
    assert np.isfinite(co.time_rev_kld(y, 1, 2, k=3, theiler_win=46)['raw'])   # 499 >= 490
    assert np.isnan(co.time_rev_kld(y, 1, 2, k=3, theiler_win=47)['raw'])      # 499 < 500
    # the cap on the number of embedded points counts
    assert _all_nan(co.time_rev_kld(_ar1(0.5, 2000, 2), 1, 2, k=3, theiler_win=40, max_n=400))


def test_time_rev_kld_separates_reversible_from_irreversible_series():
    reversible = [co.time_rev_kld(_ar1(phi, 2000, seed), 'ac1e', 2)['raw']
                  for seed, phi in enumerate((0.0, 0.3, 0.6, 0.8, 0.9, 0.95))]
    assert np.all(np.isfinite(reversible)) and np.max(np.abs(reversible)) < 0.12
    irreversible = [co.time_rev_kld(_ar1(0.5, 2000, seed, 'exp'), 'ac1e', 2)['raw'] for seed in range(4)]
    assert np.min(irreversible) > 0.25


# ------------------------------------------------------------------------------
# ar_fit: aroundmin_* is the depth of the order-selection minimum
# ------------------------------------------------------------------------------
@pytest.mark.parametrize('crit', ['sbc', 'fpe'])
def test_ar_fit_aroundmin_is_the_depth_of_the_minimum(crit):
    for seed, phi in enumerate((0.0, 0.5, 0.9)):
        y = _ar1(phi, 1000, seed)
        out = mf.ar_fit(y, 1, 8, 'sbc')
        vals = np.array([out[f'{crit}_{p}'] for p in range(1, 9)])
        pos = int(np.argmin(vals))
        assert out[f'popt_{crit}'] == pos + 1
        neighbors = [vals[j] for j in (pos - 1, pos + 1) if 0 <= j < 8]
        assert out[f'aroundmin_{crit}'] == pytest.approx(np.mean(neighbors) - vals[pos], abs=1e-14)
        assert out[f'aroundmin_{crit}'] >= 0
        # differences of a log criterion do not depend on the scale of the series
        # (the ratio it replaces went from 1.72 to 0.98 on rescaling a white-noise series by 1.05)
        scaled = mf.ar_fit(1.05 * y, 1, 8, 'sbc')
        assert scaled[f'aroundmin_{crit}'] == pytest.approx(out[f'aroundmin_{crit}'], abs=1e-9)
    assert np.isnan(mf.ar_fit(_ar1(0.5, 300, 5), 2, 2, 'sbc')[f'aroundmin_{crit}'])  # a single order


def test_ar_fit_aroundmin_is_deep_for_a_clear_order_and_shallow_for_white_noise():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(3000)
    x = np.zeros(3000)
    for t in range(2, 3000):
        x[t] = 1.2 * x[t - 1] - 0.7 * x[t - 2] + e[t]
    out = mf.ar_fit(_zscore(x[500:]), 1, 8, 'sbc')
    assert out['popt_sbc'] == 2 and out['aroundmin_sbc'] > 0.1
    white = [mf.ar_fit(_zscore(np.random.default_rng(s).standard_normal(1000)), 1, 8, 'sbc')['aroundmin_fpe']
             for s in range(10)]
    assert 0 <= np.min(white) and np.max(white) < 0.02  # (the ratio ranged from 0.006 to 3.7)


# ------------------------------------------------------------------------------
# loop_local_simple: the window length that forecasts best
# ------------------------------------------------------------------------------
def test_loop_local_simple_reports_the_best_window_length():
    rng = np.random.default_rng(0)
    n = 1000
    walk = _zscore(np.cumsum(rng.standard_normal(n)))
    white = _zscore(rng.standard_normal(n))
    smooth = _zscore(np.sin(2 * np.pi * np.arange(n) / 150) + rng.standard_normal(n))
    out = {name: mf.loop_local_simple(y, 'mean') for name, y in (('walk', walk), ('white', white), ('smooth', smooth))}
    assert out['walk']['stde_minpos'] == 1         # the last value is the best predictor
    assert out['white']['stde_minpos'] >= 9        # the longest averages are
    assert 2 <= out['smooth']['stde_minpos'] <= 10
    for name, y in (('walk', walk), ('white', white), ('smooth', smooth)):
        stde = np.array([mf.local_simple(y, 'mean', length)['stde'] for length in range(1, 11)])
        assert out[name]['stde_minpos'] == np.argmin(stde) + 1
        assert out[name]['stde_minsize'] == pytest.approx(stde.min() / stde.mean())
        assert 0 < out[name]['stde_minsize'] <= 1
        assert 'stde_peakpos' not in out[name] and 'stde_peaksize' not in out[name]


def test_loop_local_simple_best_window_varies_across_series():
    # the output it replaces (the position of the curve's extreme value) was 1 on 92% of a panel like this
    positions = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(-0.9, 0.95)
        positions.append(mf.loop_local_simple(_ar1(phi, 500, seed), 'mean')['stde_minpos'])
    values, counts = np.unique(positions, return_counts=True)
    assert len(values) >= 3 and counts.max() < 0.8 * len(positions)


# ------------------------------------------------------------------------------
# rpde: a fixed maximum recurrence time
# ------------------------------------------------------------------------------
def _logistic(n, x0=0.3):
    x = np.empty(n + 100)
    x[0] = x0
    for i in range(1, n + 100):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return _zscore(x[100:])


def test_rpde_does_not_depend_on_length_for_deterministic_dynamics():
    lengths = (500, 1000, 2000, 5000)
    sine = [en.rpde(_zscore(np.sin(2 * np.pi * np.arange(n) / 37)), 3, 1) for n in lengths]
    chaos = [en.rpde(_logistic(n), 3, 1) for n in lengths]
    assert np.ptp([o['H_norm'] for o in sine]) < 0.02 and np.ptp([o['H_norm'] for o in chaos]) < 0.08
    # (the share of occupied bins of a broad RPD, as for chaos, still grows as more points return)
    assert np.ptp([o['propNonZero'] for o in sine]) < 0.02
    # the modal recurrence time of a sinusoid holds a fixed share of the returns (with one bin per
    # embedded point, maxRPD was proportional to the length: 162, 637 and 1113 for 500, 2000 and 4000 points)
    assert np.ptp([o['maxRPD'] for o in sine]) < 0.02 * sine[0]['maxRPD']
    assert max(o['H_norm'] for o in sine) < 0.2 < 0.8 < min(o['H_norm'] for o in chaos)
    # without the limit the same dynamics give outputs that drift with the length
    unlimited = [en.rpde(_logistic(n), 3, 1, t_max=-1)['H_norm'] for n in lengths]
    assert np.ptp(unlimited) > 0.1


def test_rpde_is_high_for_noise_once_enough_points_return():
    out = [en.rpde(_zscore(np.random.default_rng(s).standard_normal(2000)), 3, 1, 0.3) for s in range(3)]
    assert min(o['H_norm'] for o in out) > 0.9
    assert all(0 < o['propNonZero'] <= 1 and o['maxRPD'] >= 1 for o in out)


def test_rpde_period_is_the_modal_recurrence_time():
    y = _zscore(np.sin(2 * np.pi * np.arange(1000) / 20))
    hist = en._close_returns_hist(np.ascontiguousarray(np.c_[y[:-5], y[5:]]), 0.12, 100)
    assert np.argmax(hist) == 20 and hist[:2].sum() == 0
    assert hist.sum() <= 1000 - 5 - 100  # only the points that can be followed for 100 steps
