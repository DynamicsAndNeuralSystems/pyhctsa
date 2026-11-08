"""Pinned outputs of the operations whose inline Shannon entropies now call utils._shannon_entropy.

The expected values were generated from the main checkout (commit 0fa9013, before the refactor), so this
test checks that the refactor left every pinned output unchanged (rtol 1e-12).
"""
import numpy as np
import pytest

from pyhctsa.operations import (correlation, entropy, nonlinearity, spectral, stationarity, symbolic,
                                wavelet)
from pyhctsa.utils import _shannon_entropy


def _series():
    n = 500
    r = np.random.default_rng(11); wn = r.standard_normal(n)
    r = np.random.default_rng(12); e = r.standard_normal(n); ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    r = np.random.default_rng(13); sine = np.sin(2 * np.pi * np.arange(n) / 23.7) + 0.3 * r.standard_normal(n)
    z = lambda x: (x - x.mean()) / x.std(ddof=1)
    return {'wn': z(wn), 'ar1': z(ar), 'sine': z(sine)}


_SERIES = _series()

_CALLS = [
  ('entropy', 'permutation_entropy', dict(m=3, tau=1), ['permEnLE', 'normWPE']),
  ('entropy', 'rpde', dict(m=2, tau=1), ['H', 'H_norm']),
  ('entropy', 'permutation_entropy_complexity', dict(m=4, tau=1), ['hNorm', 'jsComplexity']),
  ('entropy', 'wavelet_entropy', dict(), None),
  ('entropy', 'dispersion_entropy', dict(m=3), ['dispEn', 'fDispEn']),
  ('entropy', 'shannon_entropy', dict(num_bins=3, depth=3), None),
  ('nonlinearity', 'persistent_homology', dict(), ['persistenceEntropyH1']),
  ('nonlinearity', 'poincare_section', dict(), ['hboxcounts5', 'hboxcounts10']),
  ('nonlinearity', 'rqa', dict(), ['L_entr']),
  ('nonlinearity', 'return_time', dict(), ['hhist', 'hcgdist', 'hhisthist']),
  ('spectral', 'cross_frequency_coupling', dict(), ['maxMI', 'entropyMI']),
  ('spectral', 'spectral_summaries_phase', dict(), ['phEnt']),
  ('spectral', 'bicoherence', dict(), ['entropy']),
  ('spectral', 'spectral_time_freq', dict(), ['se_mean', 'se_std']),
  ('wavelet', 'cwt', dict(), ['SC_h', 'dd_SC_h']),
  ('wavelet', 'wpd_best_tree', dict(), ['entropy']),
  ('symbolic', 'motif_two', dict(), ['h', 'hhh']),
  ('symbolic', 'motif_three', dict(), ['hh', 'hhh']),
  ('symbolic', 'transition_matrix', dict(num_groups=3), ['transEntropy']),
  ('stationarity', 'sliding_window', dict(window_stat='specen', across_win_stat='std', num_seg=5, inc_move=2), None),
  ('correlation', 'embed2_shapes', dict(), ['hist_ent']),
]

# (module, operation, series, output key or None for a scalar output) -> value on main
# Re-pinned on this branch (cons-coarse-grain, #60): the following values change with the convention
# this branch fixes (see its PR body); every other value is the main-checkout value.
#   entropy.shannon_entropy|ar1.None
#   entropy.shannon_entropy|sine.None
#   entropy.shannon_entropy|wn.None
_EXPECTED = {
    ('entropy', 'permutation_entropy', 'wn', 'permEnLE'): 0.8951686285063656,
    ('entropy', 'permutation_entropy', 'wn', 'normWPE'): 0.9989359783759333,
    ('entropy', 'permutation_entropy', 'ar1', 'permEnLE'): 0.8574065835488686,
    ('entropy', 'permutation_entropy', 'ar1', 'normWPE'): 0.8998093352292206,
    ('entropy', 'permutation_entropy', 'sine', 'permEnLE'): 0.8897718124330334,
    ('entropy', 'permutation_entropy', 'sine', 'normWPE'): 0.9744601143299252,
    ('entropy', 'rpde', 'wn', 'H'): 4.888480923135362,
    ('entropy', 'rpde', 'wn', 'H_norm'): 0.7868647790946126,
    ('entropy', 'rpde', 'ar1', 'H'): 4.930017637975115,
    ('entropy', 'rpde', 'ar1', 'H_norm'): 0.7935506552309022,
    ('entropy', 'rpde', 'sine', 'H'): 4.71626164002619,
    ('entropy', 'rpde', 'sine', 'H_norm'): 0.7591438387267782,
    ('entropy', 'permutation_entropy_complexity', 'wn', 'hNorm'): 0.9949453328290863,
    ('entropy', 'permutation_entropy_complexity', 'wn', 'jsComplexity'): 0.006633386640429113,
    ('entropy', 'permutation_entropy_complexity', 'ar1', 'hNorm'): 0.9280524475736934,
    ('entropy', 'permutation_entropy_complexity', 'ar1', 'jsComplexity'): 0.08020747865396677,
    ('entropy', 'permutation_entropy_complexity', 'sine', 'hNorm'): 0.9545217776180919,
    ('entropy', 'permutation_entropy_complexity', 'sine', 'jsComplexity'): 0.05623535718780492,
    ('entropy', 'wavelet_entropy', 'wn', None): 0.7125752950298527,
    ('entropy', 'wavelet_entropy', 'ar1', None): 0.9465474722450863,
    ('entropy', 'wavelet_entropy', 'sine', None): 0.5363290849534131,
    ('entropy', 'dispersion_entropy', 'wn', 'dispEn'): 5.12611713304794,
    ('entropy', 'dispersion_entropy', 'wn', 'fDispEn'): 4.26535211391932,
    ('entropy', 'dispersion_entropy', 'ar1', 'dispEn'): 4.503189672231559,
    ('entropy', 'dispersion_entropy', 'ar1', 'fDispEn'): 3.2306244420279695,
    ('entropy', 'dispersion_entropy', 'sine', 'dispEn'): 4.164579615543262,
    ('entropy', 'dispersion_entropy', 'sine', 'fDispEn'): 2.937993914120066,
    ('entropy', 'shannon_entropy', 'wn', None): 1.088066797770798,
    ('entropy', 'shannon_entropy', 'ar1', None): 0.9349343261626059,
    ('entropy', 'shannon_entropy', 'sine', None): 0.8447257424865436,
    ('nonlinearity', 'persistent_homology', 'wn', 'persistenceEntropyH1'): 5.045602488533601,
    ('nonlinearity', 'persistent_homology', 'ar1', 'persistenceEntropyH1'): 4.975308596394514,
    ('nonlinearity', 'persistent_homology', 'sine', 'persistenceEntropyH1'): 4.8371356563554535,
    ('nonlinearity', 'poincare_section', 'wn', 'hboxcounts5'): 3.2205384639152936,
    ('nonlinearity', 'poincare_section', 'wn', 'hboxcounts10'): 4.461237741885455,
    ('nonlinearity', 'poincare_section', 'ar1', 'hboxcounts5'): 3.1903034673452337,
    ('nonlinearity', 'poincare_section', 'ar1', 'hboxcounts10'): 4.140543814494026,
    ('nonlinearity', 'poincare_section', 'sine', 'hboxcounts5'): 2.9386680288690097,
    ('nonlinearity', 'poincare_section', 'sine', 'hboxcounts10'): 3.7143241076597393,
    ('nonlinearity', 'rqa', 'wn', 'L_entr'): 1.2387092680446776,
    ('nonlinearity', 'rqa', 'ar1', 'L_entr'): 1.3509777207706024,
    ('nonlinearity', 'rqa', 'sine', 'L_entr'): 1.6470490506191524,
    ('nonlinearity', 'return_time', 'wn', 'hhist'): 4.472308468172633,
    ('nonlinearity', 'return_time', 'wn', 'hcgdist'): 2.975713148022678,
    ('nonlinearity', 'return_time', 'wn', 'hhisthist'): 2.15123913969404,
    ('nonlinearity', 'return_time', 'ar1', 'hhist'): 4.405839443085681,
    ('nonlinearity', 'return_time', 'ar1', 'hcgdist'): 2.9027167496870727,
    ('nonlinearity', 'return_time', 'ar1', 'hhisthist'): 1.916120823475707,
    ('nonlinearity', 'return_time', 'sine', 'hhist'): 2.8628422301593286,
    ('nonlinearity', 'return_time', 'sine', 'hcgdist'): 1.7747330908716947,
    ('nonlinearity', 'return_time', 'sine', 'hhisthist'): 0.804375154146435,
    ('spectral', 'cross_frequency_coupling', 'wn', 'maxMI'): 0.0027669992565727973,
    ('spectral', 'cross_frequency_coupling', 'wn', 'entropyMI'): 0.9779073404282601,
    ('spectral', 'cross_frequency_coupling', 'ar1', 'maxMI'): 0.008002689391028026,
    ('spectral', 'cross_frequency_coupling', 'ar1', 'entropyMI'): 0.885562936566551,
    ('spectral', 'cross_frequency_coupling', 'sine', 'maxMI'): 0.0023984490349583396,
    ('spectral', 'cross_frequency_coupling', 'sine', 'entropyMI'): 0.9500031723597337,
    ('spectral', 'spectral_summaries_phase', 'wn', 'phEnt'): 0.9824885423791163,
    ('spectral', 'spectral_summaries_phase', 'ar1', 'phEnt'): 0.9595208668351107,
    ('spectral', 'spectral_summaries_phase', 'sine', 'phEnt'): 0.9195289363999657,
    ('spectral', 'bicoherence', 'wn', 'entropy'): 0.9256585472248551,
    ('spectral', 'bicoherence', 'ar1', 'entropy'): 0.9258297541122679,
    ('spectral', 'bicoherence', 'sine', 'entropy'): 0.9254243470205629,
    ('spectral', 'spectral_time_freq', 'wn', 'se_mean'): 0.8484850820247939,
    ('spectral', 'spectral_time_freq', 'wn', 'se_std'): 0.05383432797209206,
    ('spectral', 'spectral_time_freq', 'ar1', 'se_mean'): 0.6706420946908878,
    ('spectral', 'spectral_time_freq', 'ar1', 'se_std'): 0.12806716866708387,
    ('spectral', 'spectral_time_freq', 'sine', 'se_mean'): 0.5072185077451921,
    ('spectral', 'spectral_time_freq', 'sine', 'se_std'): 0.07941899776972719,
    ('wavelet', 'cwt', 'wn', 'SC_h'): -0.7431444342858313,
    ('wavelet', 'cwt', 'wn', 'dd_SC_h'): 5.631226632000506,
    ('wavelet', 'cwt', 'ar1', 'SC_h'): -0.8730600311601187,
    ('wavelet', 'cwt', 'ar1', 'dd_SC_h'): 5.486907924229928,
    ('wavelet', 'cwt', 'sine', 'SC_h'): -0.686906427888653,
    ('wavelet', 'cwt', 'sine', 'dd_SC_h'): 5.463399608084321,
    ('wavelet', 'wpd_best_tree', 'wn', 'entropy'): 2.9743846310738706,
    ('wavelet', 'wpd_best_tree', 'ar1', 'entropy'): 2.1894176517888306,
    ('wavelet', 'wpd_best_tree', 'sine', 'entropy'): 1.8267676433237336,
    ('symbolic', 'motif_two', 'wn', 'h'): 0.6939061939740839,
    ('symbolic', 'motif_two', 'wn', 'hhh'): 1.9458785822460631,
    ('symbolic', 'motif_two', 'ar1', 'h'): 0.6940989831262901,
    ('symbolic', 'motif_two', 'ar1', 'hhh'): 2.0829070647477783,
    ('symbolic', 'motif_two', 'sine', 'h'): 0.6940989831262901,
    ('symbolic', 'motif_two', 'sine', 'hhh'): 2.041934370095439,
    ('symbolic', 'motif_three', 'wn', 'hh'): 2.1934906052405316,
    ('symbolic', 'motif_three', 'wn', 'hhh'): 3.2842807145975343,
    ('symbolic', 'motif_three', 'ar1', 'hh'): 1.9639407775368931,
    ('symbolic', 'motif_three', 'ar1', 'hhh'): 2.823879283708701,
    ('symbolic', 'motif_three', 'sine', 'hh'): 1.8405672509322415,
    ('symbolic', 'motif_three', 'sine', 'hhh'): 2.5502414844877435,
    ('symbolic', 'transition_matrix', 'wn', 'transEntropy'): 1.0968984103723638,
    ('symbolic', 'transition_matrix', 'ar1', 'transEntropy'): 0.8673485826687254,
    ('symbolic', 'transition_matrix', 'sine', 'transEntropy'): 0.7439629836536096,
    ('stationarity', 'sliding_window', 'wn', None): 0.0140515794974415,
    ('stationarity', 'sliding_window', 'ar1', None): 0.04426522926976249,
    ('stationarity', 'sliding_window', 'sine', None): 0.030052489347364913,
    ('correlation', 'embed2_shapes', 'wn', 'hist_ent'): -3.08490506696435,
    ('correlation', 'embed2_shapes', 'ar1', 'hist_ent'): -3.0872094968566044,
    ('correlation', 'embed2_shapes', 'sine', 'hist_ent'): -2.7003921223466354,
}

_MODULES = dict(entropy=entropy, nonlinearity=nonlinearity, spectral=spectral, wavelet=wavelet,
                symbolic=symbolic, stationarity=stationarity, correlation=correlation)


@pytest.mark.parametrize("mod, fn, kw, keys", _CALLS, ids=[f"{c[0]}.{c[1]}" for c in _CALLS])
def test_pinned_outputs(mod, fn, kw, keys):
    if fn == 'persistent_homology':
        pytest.importorskip('ripser')
    for name, y in _SERIES.items():
        out = getattr(_MODULES[mod], fn)(y, **kw)
        for k in (keys if keys is not None else [None]):
            got = float(out[k]) if k is not None else float(out)
            want = _EXPECTED[(mod, fn, name, k)]
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=f"{mod}.{fn} {name} {k}")


def test_shannon_entropy_conventions():
    p = np.array([0.5, 0.0, 0.25, 0.25])
    assert _shannon_entropy(p) == pytest.approx(1.5 * np.log(2), rel=1e-15)
    assert _shannon_entropy(p, base=2) == pytest.approx(1.5, rel=1e-15)
    assert _shannon_entropy(p * 4, normalize=True) == _shannon_entropy(p)
    assert _shannon_entropy(np.zeros(3)) == 0.0  # p = 0 terms contribute 0
    assert _shannon_entropy(np.array([])) == 0.0
    assert _shannon_entropy(np.array([np.nan, 1.0])) == 0.0  # NaN terms are dropped
    P = np.column_stack([p, [1.0, 0, 0, 0]])
    np.testing.assert_array_equal(_shannon_entropy(P, base=2, axis=0), [1.5, 0.0])
