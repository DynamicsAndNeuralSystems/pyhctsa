"""One coarse-graining convention (utils._coarse_grain): equiprobable symbols from Hazen quantile
cuts with bins closed on the left and tied values sharing the symbol of their mid-rank, and
equal-width symbols closed on the left; and pinned outputs of every operation routed through it.

The pinned values were generated from branch cons-coarse-grain (the convention this file fixes),
not from main: the operations' outputs changed with the convention."""
import numpy as np
import pytest

from pyhctsa.utils import _coarse_grain, z_score
from pyhctsa.operations import (correlation as CO, entropy as EN, information as IN, medical as MD,
                                physics as PH, stationarity as ST, symbolic as SB)


def _hazen_partition(x, k):
    """Symbols of untied data from explicit Hazen quantile cuts, bins closed on the left."""
    cuts = np.quantile(x, np.arange(1, k) / k, method='hazen')
    return np.searchsorted(cuts, x, side='right')


@pytest.mark.parametrize('n', [7, 10, 99, 100, 999, 1000])
@pytest.mark.parametrize('k', [2, 3, 4, 5, 7, 16])
def test_quantile_is_hazen_partition_without_ties(n, k):
    x = np.random.default_rng(n * 31 + k).standard_normal(n)
    s = _coarse_grain(x, k)
    if k <= 2 * n:  # (beyond the outermost plotting positions, np.quantile clamps to the extremes)
        np.testing.assert_array_equal(s, _hazen_partition(x, k))
    counts = np.bincount(s, minlength=k)
    assert counts.max() - counts.min() <= 1  # equiprobable up to one value
    np.testing.assert_array_equal(_coarse_grain(np.exp(x), k), s)  # rank invariance


def test_value_on_a_cut_takes_the_upper_symbol():
    # N = 5, k = 2: the Hazen median is the third value itself
    np.testing.assert_array_equal(_coarse_grain([5., 1., 3., 2., 4.], 2), [1, 0, 1, 0, 1])
    # N = 4, k = 4: no value on a cut, one per symbol
    np.testing.assert_array_equal(_coarse_grain([0.3, 0.1, 0.4, 0.2], 4), [2, 0, 3, 1])


def test_ties_share_the_symbol_of_their_mid_rank():
    # a tied group goes to the bin that holds the center of its rank span (ranks 1-4 -> 2.5,
    # position 2/6 < 1/2), not to the bin above the cut that falls inside it
    np.testing.assert_array_equal(_coarse_grain([1., 1., 1., 1., 2., 3.], 2), [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(_coarse_grain([3., 2., 1., 1., 1., 1.], 2), [1, 1, 0, 0, 0, 0])
    # mid-rank exactly on the cut (ranks 2-5 -> 3.5, position 1/2): the upper symbol
    np.testing.assert_array_equal(_coarse_grain([0., 1., 1., 1., 1., 2.], 2), [0, 1, 1, 1, 1, 1])
    # symmetric under negation when no group sits on a cut
    x = np.round(np.random.default_rng(0).standard_normal(400), 1)
    s = _coarse_grain(x, 3)
    for v in np.unique(x):
        assert np.unique(s[x == v]).size == 1
    np.testing.assert_array_equal(_coarse_grain(-x, 3), 2 - s)


def test_constant_and_degenerate_inputs():
    np.testing.assert_array_equal(_coarse_grain(np.full(9, 2.5), 4), np.full(9, 2))
    np.testing.assert_array_equal(_coarse_grain(np.full(9, 2.5), 3, 'width'), np.full(9, 1))
    assert _coarse_grain([], 3).size == 0
    with pytest.raises(ValueError):
        _coarse_grain([1., np.nan, 2.], 2)
    with pytest.raises(ValueError):
        _coarse_grain([1., 2.], 2, 'nope')


def test_width_bins():
    x = np.array([0., 0.25, 0.5, 0.75, 1.0, 0.1, 0.9])
    np.testing.assert_array_equal(_coarse_grain(x, 4, 'width'), [0, 1, 2, 3, 3, 0, 3])  # left-closed, max in top bin
    np.testing.assert_array_equal(_coarse_grain([-5., 0., 0.49, 0.5, 9.], 2, 'width', limits=(0, 1)),
                                  [0, 0, 0, 1, 1])  # outside the limits: the end bins


def _series():
    e = np.random.default_rng(11).standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.7 * ar[t - 1] + e[t]
    ties = np.round(np.random.default_rng(12).standard_normal(600), 1)
    return {'ar': z_score(ar), 'ties': z_score(ties)}


PINNED = {
    'ar': {'tm': 0.5676126878130218, 'm3': 2.0327581625980127, 'm3d': 3.280824003958632,
           'sur': 1.1998811283597748, 'tpa': -0.5360530214902377, 'sh': 1.0147096322172033,
           'lz': 0.8767377755971087, 'km': 0.02463529893220595, 'ld': 0.02466382055727509,
           'ami': 0.07590359760152048, 'cma': 5.2, 'mi': 0.17973268815247131,
           'porta': 48.49498327759197, 'disp': 0.931600090036146},
    'ties': {'tm': 0.3572621035058431, 'm3': 2.1959683320437957, 'm3d': 3.1390241995035084,
             'sur': 1.1342828461122842, 'tpa': -0.6722236896918674, 'sh': 1.0963147169400949,
             'lz': 0.9844073269862273, 'km': -0.05071187757161752, 'ld': 0.01542586157366363,
             'ami': -0.0018700400995858672, 'cma': 1.9, 'mi': 0.003682013988504173,
             'porta': 35.953177257525084, 'disp': 0.9903793851696793},
}


@pytest.mark.parametrize('name', ['ar', 'ties'])
def test_pinned_operation_outputs(name):
    y = _series()[name]
    got = {
        'tm': SB.transition_matrix(y, 'quantile', 3, 1)['ondiag'],
        'm3': SB.motif_three(y, 'quantile')['hh'],
        'm3d': SB.motif_three(y, 'diffquant')['hhh'],
        'sur': SB.surprise(y, 'dist', 20, 3, 'quantile', 100)['mean'],
        'tpa': SB.transition_p_alphabet(y, list(range(2, 8)), 1)['meandiagfexp_b'],
        'sh': EN.shannon_entropy(y, 3, 2),
        'lz': EN.lempel_ziv_complexity(y, 4),
        'km': PH.kramers_moyal(y)['driftCubic'],
        'ld': ST.local_distributions(y, 5, 'par')['ex_meandiv'],
        'ami': CO.histogram_ami(y, [1, 2], 'quantiles', 5)['ami2'],
        'cma': CO.compare_min_ami(y, 'quantiles', list(range(2, 12)))['mean'],
        'mi': IN._mi_bin(y[:-1], y[1:], 'quantile', 'quantile', 4),
        'porta': MD.porta(y)['pV1'],  # (equal-width; unchanged from main)
        'disp': EN.dispersion_entropy(y, 2, 6, 1)['normDispEn'],  # (equal-width; unchanged from main)
    }
    for k, v in PINNED[name].items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=0), k


def test_coarse_grain_symbols_are_one_based():
    y = _series()['ties']
    s = SB.coarse_grain(y, 'quantile', 4)
    np.testing.assert_array_equal(s, _coarse_grain(y, 4) + 1)
    d = SB.coarse_grain(y, 'diff', 3)
    np.testing.assert_array_equal(d, _coarse_grain(np.diff(y), 3) + 1)
