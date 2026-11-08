"""The random-order expectation that local_distributions subtracts from its mean distance: exact
against a Monte Carlo of random permutations and against the untruncated hypergeometric sums, an
excess that averages 0 over permutations, and the edge cases; and the effective-sample-size
output."""
import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import hypergeom

from pyhctsa.operations import stationarity as ST
from pyhctsa.utils import _coarse_grain


def _bins(y, num_bins=10):
    return _coarse_grain(y, num_bins, 'quantile'), num_bins


def _null(y, num_segs, mode, num_bins=10):
    b, B = _bins(y, num_bins)
    counts = np.bincount(b, minlength=B)
    return np.sum(ST._random_order_tv(counts, len(y), len(y) // num_segs, mode))


def _permuted_distances(y, num_segs, mode, reps, seed, num_bins=10):
    """Mean total variation distance of each of ``reps`` random permutations of ``y``."""
    b, B = _bins(y, num_bins)
    N, n = len(y), len(y) // num_segs
    par = np.bincount(b, minlength=B) / N
    perm = np.argsort(np.random.default_rng(seed).random((reps, N)), axis=1)
    bp = b[perm]
    seg = np.stack([np.apply_along_axis(np.bincount, 1, bp[:, i * n:(i + 1) * n], minlength=B) / n
                    for i in range(num_segs)], axis=1)  # reps x num_segs x B
    if mode == 'par':
        d = 0.5 * np.abs(seg - par).sum(axis=2)
    else:
        d = np.stack([0.5 * np.abs(seg[:, i] - seg[:, j]).sum(axis=1)
                      for i in range(num_segs) for j in range(i + 1, num_segs)], axis=1)
    return d.mean(axis=1)


_SERIES = {
    'gauss_100': np.random.default_rng(1).standard_normal(100),
    'gauss_203': np.random.default_rng(2).standard_normal(203),  # 3 samples left over with 5 segments
    'gauss_1000': np.random.default_rng(3).standard_normal(1000),
    'tied_150': np.random.default_rng(4).integers(0, 4, 150).astype(float),  # fewer, unequal bins
    'skewed_ties_400': np.floor(np.random.default_rng(5).exponential(1.0, 400) * 2),
}


@pytest.mark.parametrize('name', list(_SERIES))
@pytest.mark.parametrize('num_segs,mode', [(5, 'par'), (3, 'par'), (5, 'each'), (2, 'each')])
def test_null_matches_permutations(name, num_segs, mode):
    y = _SERIES[name]
    d = _permuted_distances(y, num_segs, mode, reps=4000, seed=10)
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert abs(d.mean() - _null(y, num_segs, mode)) < 4 * se


@pytest.mark.parametrize('name', ['gauss_203', 'tied_150'])
@pytest.mark.parametrize('num_segs,mode', [(5, 'par'), (3, 'each')])
def test_excess_averages_zero_over_permutations(name, num_segs, mode):
    y = _SERIES[name]
    rng = np.random.default_rng(20)
    ex = np.array([ST.local_distributions(rng.permutation(y), num_segs, mode)['ex_meandiv']
                   for _ in range(1500)])
    assert abs(ex.mean()) < 4 * ex.std(ddof=1) / np.sqrt(len(ex))


@pytest.mark.parametrize('N,K,n', [(2000, 200, 400), (2000, 37, 666), (6000, 1500, 1200), (500, 499, 100)])
def test_truncation_par(N, K, n):
    x = np.arange(max(0, n - (N - K)), min(n, K) + 1)  # full support
    full = 0.5 * np.sum(hypergeom.pmf(x, N, K, n) * np.abs(x / n - K / N))
    assert abs(ST._random_order_tv(np.array([K, N - K]), N, n, 'par')[0] - full) < 1e-12


@pytest.mark.parametrize('N,K,n', [(600, 60, 200), (900, 13, 300), (1500, 700, 300)])
def test_truncation_each(N, K, n):
    x = np.arange(max(0, n - (N - K)), min(n, K) + 1)
    px = hypergeom.pmf(x, N, K, n)
    full = 0.0
    for xi, pxi in zip(x, px):
        yv = np.arange(max(0, n - (N - n - (K - xi))), min(n, K - xi) + 1)
        full += pxi * np.sum(hypergeom.pmf(yv, N - n, K - xi, n) * np.abs(xi - yv)) / n
    assert abs(ST._random_order_tv(np.array([K, N - K]), N, n, 'each')[0] - 0.5 * full) < 1e-12


def test_null_equal_bins_values():
    # 10 equal bins, 5 segments, comparing to the parent
    for N, expected in ((100, 0.229), (300, 0.136), (1000, 0.075), (2000, 0.053), (6000, 0.031)):
        assert ST._random_order_tv(np.full(10, N // 10), N, N // 5, 'par').sum() == pytest.approx(expected, abs=6e-4)


def test_raw_distance_recovered():
    y = _SERIES['gauss_1000']
    b, B = _bins(y)
    seg = [np.bincount(b[i * 200:(i + 1) * 200], minlength=B) / 200 for i in range(5)]
    raw = np.mean([0.5 * np.abs(s - np.bincount(b, minlength=B) / 1000).sum() for s in seg])
    out = ST.local_distributions(y, 5, 'par')
    assert out['ex_meandiv'] == pytest.approx(raw - _null(y, 5, 'par'), abs=1e-14)


def test_mean_shift_detected():
    rng = np.random.default_rng(30)
    y = rng.standard_normal(6000)
    shifted = y + 0.5 * (np.arange(6000) >= 3000)
    assert ST.local_distributions(shifted, 5, 'par')['ex_meandiv'] > 0.03
    assert abs(ST.local_distributions(y, 5, 'par')['ex_meandiv']) < 0.01


def test_edge_cases():
    # fewer values than segments: empty segments
    out = ST.local_distributions(np.arange(4.0), 5, 'par')
    assert np.isnan(out['ex_meandiv']) and np.isnan(out['stddiv'])
    assert np.isnan(ST.local_distributions(np.array([1.0]), 2, 'each'))
    # constant series: one bin, no difference and no expected difference
    for mode in ('par', 'each'):
        out = ST.local_distributions(np.ones(50), 3, mode)
        assert out['ex_meandiv'] == 0 and out['stddiv'] == 0
    # two segments with 'each': a float; 'parent' is 'par'
    y = _SERIES['gauss_203']
    assert isinstance(ST.local_distributions(y, 2, 'each'), float)
    assert ST.local_distributions(y, 3, 'parent') == ST.local_distributions(y, 3, 'par')
    with pytest.raises(ValueError):
        ST.local_distributions(y, 3, 'both')


# ------------------------------------------------------------------------------
# effective sample size
# ------------------------------------------------------------------------------
def test_shifted_segment_acf_is_zero_in_random_order():
    # the pooled within-segment autocorrelation plus (n - k) / (n (n - 1)) averages 0 over random permutations
    z = (np.random.default_rng(40).random(300) < 0.3).astype(float)
    rng = np.random.default_rng(41)
    n, lags = 60, np.array([1, 2, 5, 20])
    r = np.array([ST._segment_acf(rng.permutation(z), 5, n)[lags] for _ in range(4000)]) + (n - lags) / (n * (n - 1))
    assert np.all(np.abs(r.mean(axis=0)) < 4 * r.std(axis=0, ddof=1) / np.sqrt(len(r)))


def test_variance_inflation_ar1():
    # AR(1) with phi = 0.5: 1 + 2 sum of (1 - k/n) rho(k) approaches (1 + phi) / (1 - phi) = 3 for long segments
    y = lfilter([1], [1, -0.5], np.random.default_rng(42).standard_normal(200000))
    assert ST._variance_inflation(ST._segment_acf(y, 5, 40000), 40000) == pytest.approx(3, rel=0.05)
    assert ST._variance_inflation(np.full(10, np.nan), 10) == 1.0
    assert np.all(np.isnan(ST._segment_acf(np.ones(30), 3, 10)))


def test_ess_near_zero_in_random_order():
    y = np.random.default_rng(44).standard_normal(1000)
    rng = np.random.default_rng(45)
    ex = np.array([ST.local_distributions(rng.permutation(y), 5, 'par')['ex_meandiv_ess'] for _ in range(300)])
    assert abs(ex.mean()) < 0.002


def test_ess():
    rng = np.random.default_rng(43)
    ars = [lfilter([1], [1, -0.9], rng.standard_normal(9000))[3000:] for _ in range(20)]
    for mode in ('par', 'each'):
        outs = [ST.local_distributions(ar, 5, mode) for ar in ars]
        assert np.median([o['ex_meandiv'] for o in outs]) > 0.02  # correlated but stationary
        assert abs(np.median([o['ex_meandiv_ess'] for o in outs])) < 0.005
    y = rng.standard_normal(6000) + 1.0 * (np.arange(6000) >= 3000)
    out = ST.local_distributions(y, 5, 'each')
    assert out['ex_meandiv_ess'] == pytest.approx(out['ex_meandiv'], abs=0.01)  # a mean shift is kept
    for mode in ('par', 'each'):
        out = ST.local_distributions(np.ones(50), 3, mode)
        assert out['ex_meandiv_ess'] == 0
    assert all(np.isnan(v) for v in ST.local_distributions(np.arange(4.0), 5, 'par').values())
