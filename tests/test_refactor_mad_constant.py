"""Pinned outputs of the operations that use the MAD and IQR normal-consistency constants.

Every robust scale now uses the exact factors ``utils.MAD_TO_SIGMA = 1/Phi^{-1}(3/4)`` and
``utils.IQR_TO_SIGMA = 1/(2 Phi^{-1}(3/4))`` (multiplied), replacing the rounded 1.4826, 1/0.6745,
1/1.349 and 1/1.35, and ``symbolic.surprise`` takes its quartiles with ``np.quantile(method='hazen')``
instead of ``scipy.stats.mstats.mquantiles(alphap=0.5, betap=0.5)``. These outputs changed by about
1e-5 (relative) or less.

Two kinds of check:

1. Same-runner reference checks (``test_ref_*``): independent, explicit computations of the chosen
   convention (the exact constants from ``scipy.stats.norm.ppf``, and the MAD and IQR scales built
   with them) compared with the operations at ``rtol=1e-12, atol=0`` on whatever machine runs the
   test.
2. Operation-level pins (``test_pinned_outputs``): the expected values were generated from branch
   ``cons-mad-constant`` (this change, on Ben's arm64 Mac), on two seeded series. Other platforms
   reproduce them only to the accuracy of the operation's numerics, so each operation has a
   tolerance set by what it computes (``TOL`` below).
"""
import numpy as np
import pytest
from scipy.stats import norm

from pyhctsa.fitting import robust_linear_fit
from pyhctsa.operations import distribution as di
from pyhctsa.operations import entropy as en
from pyhctsa.operations import medical as md
from pyhctsa.operations import spectral as sp
from pyhctsa.operations import symbolic as sy
from pyhctsa.operations import wavelet as wl
from pyhctsa.robust import bf_ks_density
from pyhctsa.utils import IQR_TO_SIGMA, MAD_TO_SIGMA, get_tau, z_score

NAN = float('nan')


def test_constants():
    np.testing.assert_allclose(MAD_TO_SIGMA, 1.482602218505602, rtol=1e-15)
    np.testing.assert_allclose(1 / MAD_TO_SIGMA, norm.ppf(0.75), rtol=1e-15)
    np.testing.assert_allclose(1 / IQR_TO_SIGMA, norm.ppf(0.75) - norm.ppf(0.25), rtol=1e-15)
    # consistency on a large Gaussian sample
    x = np.random.default_rng(0).standard_normal(200_000)
    assert abs(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))) - 1) < 0.01
    assert abs(IQR_TO_SIGMA * np.subtract(*np.quantile(x, [0.75, 0.25])) - 1) < 0.01


def _series():
    r = np.random.default_rng(11)
    e = r.standard_normal(300)
    a = np.zeros(300)
    for t in range(1, 300):
        a[t] = 0.8 * a[t - 1] + e[t]
    r = np.random.default_rng(12)
    t = np.arange(300)
    s = np.sin(2 * np.pi * t / 23.7) + 0.4 * r.standard_normal(300)
    return {'ar1': z_score(a), 'sine': z_score(s)}


SERIES = _series()

CALLS = {
    'hrv_classic': lambda y: {k: v for k, v in md.hrv_classic(y).items() if k.startswith('pnnrel')},
    'porta': lambda y: md.porta(y, 6, 1),
    'withinp_median': lambda y: [di.withinp(y, 1.0, 'median'), di.withinp(y, 2.0, 'median')],
    'scal_2_freq': lambda y: wl.scal_2_freq(y, 'db3', 5, 1),
    'dwt_coeff': lambda y: {k: v for k, v in wl.dwt_coeff(y, 'db3', 3).items() if k.startswith('noisestd')},
    'specparam': lambda y: sp.specparam(y),
    'envelope_stats_ifspread': lambda y: [sp.envelope_stats(y)['dom_ifspread']],
    'rhythmicity': lambda y: {k: v for k, v in sp.rhythmicity(y).items() if k in ('lavi_fracSustained', 'lavi_fracTransient', 'lavi_numPeaks')},
    'bf_ks_density': lambda y: bf_ks_density(y)[0][::10],
    'distribution_entropy_ks': lambda y: [en.distribution_entropy(y, 'ks')],
    'robust_linear_fit': lambda y: np.concatenate([robust_linear_fit(y[:-1], y[1:])[0], [robust_linear_fit(y[:-1], y[1:])[1]['se'][0]]]),
    'get_tau_mi': lambda y: [get_tau(y, 'mi')],
    'surprise_quantiles': lambda y: {k: v for k, v in sy.surprise(y, 'dist', 0.2, 3, 'quantile', 100, 0).items() if k in ('lq', 'uq', 'median')},
}

EXPECTED = {
    'hrv_classic|ar1': {'pnnrel025': 0.8260869565217391, 'pnnrel05': 0.6521739130434783, 'pnnrel1': 0.32441471571906355, 'pnnrel2': 0.05351170568561873, 'pnnrel3': 0.0},
    'porta|ar1': {'pV0': 26.51006711409396, 'pV1': 46.644295302013425, 'pV2LV': 9.731543624161073, 'pV2UV': 17.114093959731544},
    'withinp_median|ar1': [0.67, 0.9333333333333333],
    'scal_2_freq|ar1': {'lmax': 4.0, 'period': 20.0, 'pf': 0.05},
    'dwt_coeff|ar1': {'noisestd_l1': 0.39208797063599604, 'noisestd_l2': 0.5790762915042326, 'noisestd_l3': 0.8945585680950359},
    'specparam|ar1': {'apExponent': 1.543703364988981, 'apOffset': 0.30624955469873316, 'numPeaks': 0.0, 'maxPeakFreq': NAN, 'maxPeakPower': NAN, 'maxPeakBW': NAN, 'totalPeakPower': 0.0, 'periodicFraction': 0.0, 'modelR2': 0.8156041254963851, 'modelMAE': 0.08488111348753054},
    'envelope_stats_ifspread|ar1': [0.010889757981706613],
    'rhythmicity|ar1': {'lavi_fracSustained': 0.07692307692307693, 'lavi_fracTransient': 0.0, 'lavi_numPeaks': 0.0},
    'bf_ks_density|ar1': [9.2554714669606e-05, 0.011069317959441334, 0.03431802489416914, 0.12807417321199072, 0.2573609480082557, 0.40879076673725856, 0.28770886753227026, 0.10207649212071881, 0.024579167156587176, 0.0036474466497897874],
    'distribution_entropy_ks|ar1': [1.970222296574162],
    'robust_linear_fit|ar1': [0.0017207338035153486, 0.7967247512701789, 0.03665806460041107],
    'get_tau_mi|ar1': [4.0],
    'surprise_quantiles|ar1': {'median': 1.0747982641734826, 'lq': 0.9203640572617908, 'uq': 1.2568362938830038},
    'hrv_classic|sine': {'pnnrel025': 0.8060200668896321, 'pnnrel05': 0.6053511705685619, 'pnnrel1': 0.3110367892976589, 'pnnrel2': 0.05016722408026756, 'pnnrel3': 0.0},
    'porta|sine': {'pV0': 32.5503355704698, 'pV1': 47.31543624161074, 'pV2LV': 1.342281879194631, 'pV2UV': 18.79194630872483},
    'withinp_median|sine': [0.7433333333333333, 0.9966666666666667],
    'scal_2_freq|sine': {'lmax': 4.0, 'period': 20.0, 'pf': 0.05},
    'dwt_coeff|sine': {'noisestd_l1': 0.4189670619434174, 'noisestd_l2': 0.6168279326097911, 'noisestd_l3': 1.0933392613497954},
    'specparam|sine': {'apExponent': 0.3801370141442019, 'apOffset': -0.17623758380489524, 'numPeaks': 0.0, 'maxPeakFreq': NAN, 'maxPeakPower': NAN, 'maxPeakBW': NAN, 'totalPeakPower': 0.0, 'periodicFraction': 0.0, 'modelR2': 0.33708740196054676, 'modelMAE': 0.06966832414773523},
    'envelope_stats_ifspread|sine': [0.0011500295724338975],
    'rhythmicity|sine': {'lavi_fracSustained': 0.0, 'lavi_fracTransient': 0.0, 'lavi_numPeaks': 0.0},
    'bf_ks_density|sine': [0.00011168361294214723, 0.008646355037337778, 0.07931404818181119, 0.23825097505161977, 0.31327066005436494, 0.28794065584687806, 0.29409989198439024, 0.15441954712117034, 0.030886148198756107, 0.0019506651691027865],
    'distribution_entropy_ks|sine': [1.7106396193583342],
    'robust_linear_fit|sine': [0.006902372401146078, 0.7578790399136967, 0.0392827690356562],
    'get_tau_mi|sine': [4.0],
    'surprise_quantiles|sine': {'median': 1.0986122886681098, 'lq': 1.0055218656020977, 'uq': 1.201266442728193},
}


# Tolerances of the operation-level pins, by what the operation computes:
#   tier 1, closed-form arithmetic (counts, quantiles, histograms, FFT and wavelet transforms,
#     kernel densities): rtol 1e-9 (CI drift from BLAS/FFT summation order is at most about 1e-11);
#   tier 2, linear algebra and least squares: rtol 1e-8;
#   tier 3, iterative optimizers: rtol 1e-5 (the optimizer's path depends on the platform's arithmetic).
TIER1 = dict(rtol=1e-9, atol=0)
TIER2 = dict(rtol=1e-8, atol=0)
TIER3 = dict(rtol=1e-5, atol=0)
TOL = {
    'hrv_classic': TIER1,               # proportions of increments beyond multiples of a MAD scale
    'porta': TIER1,                     # symbol counts on IQR-scaled levels
    'withinp_median': TIER1,            # proportion within an IQR scale of the median
    'scal_2_freq': TIER1,               # wavelet decomposition, MAD scales, argmax
    'dwt_coeff': TIER1,                 # wavelet decomposition, MAD noise scales
    'specparam': TIER3,                 # aperiodic fit by scipy.optimize.least_squares (trf)
    'envelope_stats_ifspread': TIER1,   # Hilbert transform (FFT), scaled MAD
    'rhythmicity': TIER1,               # wavelet (FFT) spectrum, scaled MAD, counts
    'bf_ks_density': TIER1,             # kernel density estimate
    'distribution_entropy_ks': TIER1,   # kernel density estimate and its entropy
    'robust_linear_fit': TIER2,         # IRLS: weighted least squares, stopped at a 1e-10 step
    'get_tau_mi': TIER1,                # histogram mutual information (an integer lag)
    'surprise_quantiles': TIER1,        # seeded resampling, counts, hazen quantiles
}


@pytest.mark.parametrize('key', list(EXPECTED))
def test_pinned_outputs(key):
    name, sname = key.split('|')
    got = CALLS[name](SERIES[sname])
    want = EXPECTED[key]
    if isinstance(want, dict):
        got_v = np.array([float(got[k]) for k in want])
        want_v = np.array(list(want.values()))
    else:
        got_v = np.ravel(np.asarray(got, dtype=float))[:len(want)]
        want_v = np.array(want)
    np.testing.assert_allclose(got_v, want_v, equal_nan=True, **TOL[name])


# ---------------------------------------------------------------------------------------------
# Same-runner reference checks: explicit computations of the chosen convention (not the old code)
# ---------------------------------------------------------------------------------------------

_Q3 = float(norm.ppf(0.75))   # Phi^{-1}(3/4): MAD of a unit Gaussian
_IQR1 = float(norm.ppf(0.75) - norm.ppf(0.25))   # IQR of a unit Gaussian


def _ref_mad_sigma(x):
    return np.median(np.abs(x - np.median(x))) / _Q3


def _ref_iqr_sigma(x):
    return (np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')) / _IQR1


def _ref_corpus():
    r = np.random.default_rng(5)
    return list(SERIES.values()) + [r.standard_normal(500), r.standard_t(3, 400)]


def test_ref_bf_ks_density():
    # Silverman bandwidth with the exact MAD scale, and the Gaussian KDE evaluated directly
    for y in _ref_corpus():
        n = y.size
        h_ref = _ref_mad_sigma(y) * (4 / (3 * n)) ** (1 / 5)
        f, xi, h = bf_ks_density(y)
        np.testing.assert_allclose(h, h_ref, rtol=1e-12, atol=0)
        np.testing.assert_allclose(xi, np.linspace(y.min() - 3 * h_ref, y.max() + 3 * h_ref, 100),
                                   rtol=1e-12, atol=1e-12 * np.ptp(y))  # atol: xi crosses zero
        f_ref = norm.pdf(xi[None, :], loc=y[:, None], scale=h_ref).mean(axis=0)
        np.testing.assert_allclose(f, f_ref, rtol=1e-12, atol=0)


def test_ref_withinp_median():
    for y in _ref_corpus():
        mu, sig = np.median(y), _ref_iqr_sigma(y)
        for p in (0.5, 1.0, 2.0):
            want = np.mean(np.abs(y - mu) <= p * sig)
            np.testing.assert_allclose(di.withinp(y, p, 'median'), want, rtol=1e-12, atol=0)


def test_ref_hrv_pnnrel():
    for y in _ref_corpus():
        d = np.diff(y)
        sig = _ref_mad_sigma(d)
        got = md.hrv_classic(y)
        for key, c in (('pnnrel025', 0.25), ('pnnrel05', 0.5), ('pnnrel1', 1), ('pnnrel2', 2), ('pnnrel3', 3)):
            np.testing.assert_allclose(got[key], np.mean(np.abs(d) > c * sig), rtol=1e-12, atol=0)
