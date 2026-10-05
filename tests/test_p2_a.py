"""Tests for the ports of the hctsa robust/all (and robust/finish) changes to the correlation,
information, entropy, surrogates, symbolic and nonlinearity operations (pyhctsa branch p2-a).

MATLAB values (tests/data/p2_a.json, built by tests/data/p2_a/build_fixture.py) are from hctsa
robust/all + robust/finish on four test series; the remaining tests are properties of the ports.
"""
import importlib
import json
import logging
import os
import warnings

import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import entropy as en
from pyhctsa.operations import information as inf
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import surrogates as su
from pyhctsa.operations import symbolic as sy
from pyhctsa.robust import bf_random
from pyhctsa.toolboxes.Tisean_3_0_1 import tisean

warnings.filterwarnings('ignore')
logging.getLogger('pyhctsa').setLevel(logging.ERROR)

with open(os.path.join(os.path.dirname(__file__), 'data', 'p2_a.json')) as f:
    FIX = json.load(f)

NS = {}
for _m in ('correlation', 'information', 'entropy', 'surrogates', 'symbolic', 'nonlinearity'):
    for _k, _v in vars(importlib.import_module('pyhctsa.operations.' + _m)).items():
        if not _k.startswith('__'):
            NS.setdefault(_k, _v)

# hctsa's BF_Random 'perm' is the argsort of n uniforms (robust/finish ef1a176b); older pyhctsa
# robust.py had the Fisher-Yates shuffle
PERM_OK = bool(np.array_equal(bf_random(8, 3, 'perm'), [8, 6, 3, 7, 2, 5, 1, 4]))
NEEDS_PERM = {'fd2', 'fd3', 'rqa', 'rqa2', 'rect', 'evt1', 'evt2'}
try:
    import ripser  # noqa: F401
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

# relative/absolute tolerance per call (default 1e-8); the fits are 1e-7-accurate searches or float32 ports
TOL = {'tpa_c': 1e-5, 'tpa_def': 1e-5, 'fd1': 1e-6, 'fd2': 1e-6, 'fd3': 1e-6, 'lyap': 1e-5, 'lyap7': 1e-5,
       'pha': 1e-3, 'phb': 1e-3, 'an_1_gaussian_x': 1e-6, 'mami_ac_gaussian': 1e-4, 'mami_ac1e_gaussian': 1e-4}
# output fields to leave out (MATLAB's optimizer vs ours: the exponential-fit parameters of randomize)
SKIP_FIELDS = {'rand_dyndist': ('fexp',), 'rand_permute': ('fexp',), 'rand_statdist': ('fexp',), 'rand_s5': ('fexp',)}
# (call, series) pairs that are degenerate at rounding level or differ by a known upstream effect
KNOWN_DIFF = {('evt1', 'quant_ar'), ('evt2', 'quant_ar'), ('e2d_tau', 'quant_ar'), ('rect', 'lattice300'),
              ('evt1', 'lattice300'), ('evt2', 'lattice300')}

CASES = [(k, s) for k, c in FIX['calls'].items() for s in c['expected']]


def _flat(v):
    if isinstance(v, dict):
        return {k: (np.nan if x is None else float(x)) for k, x in v.items()}
    return {'': np.nan if v is None else float(v)}


@pytest.mark.parametrize('key,series', CASES, ids=[f'{k}-{s}' for k, s in CASES])
def test_matches_matlab(key, series):
    call = FIX['calls'][key]
    if key in NEEDS_PERM and not PERM_OK:
        pytest.skip("needs the argsort 'perm' of bf_random (hctsa ef1a176b)")
    if call['py'].startswith('persistent_homology') and not HAS_RIPSER:
        pytest.skip('ripser not installed')
    if (key, series) in KNOWN_DIFF:
        pytest.skip('degenerate / tied input: known difference from MATLAB')
    y = np.array(FIX['series'][series])
    got = eval(call['py'], NS, {'y': y})
    exp = _flat(call['expected'][series])
    got = _flat(got) if not (isinstance(got, float) and np.isnan(got)) else {'': np.nan}
    tol = TOL.get(key, 1e-8)
    skip = SKIP_FIELDS.get(key, ())
    for name, e in exp.items():
        if any(s in name for s in skip):
            continue
        g = got.get(name, np.nan)
        if np.isnan(e):
            assert np.isnan(g), f'{key}.{name}: expected NaN, got {g}'
        else:
            assert g == pytest.approx(e, rel=tol, abs=tol), f'{key}.{name}'


# ------------------------------------------------------------------------------
# surrogates: MATLAB's SD_MakeSurrogates / SD_SurrogateTest / SD_Surrogates, same seed
# ------------------------------------------------------------------------------
SD_CASES = list(FIX['sd'])


@pytest.mark.parametrize('name', SD_CASES)
def test_make_surrogates_equal_matlab(name):
    d = FIX['sd'][name]
    x = np.array(d['y'])
    np.testing.assert_allclose(su._make_surrogates(x, 'RP', 20, 42), np.array(d['z']), atol=1e-13)
    np.testing.assert_array_equal(su._make_surrogates(x, 'RandPerm', 20, 42), np.array(d['zr']))


@pytest.mark.parametrize('name', SD_CASES)
def test_surrogate_test_equal_matlab(name):
    d = FIX['sd'][name]
    x = np.array(d['y'])
    out = su.surrogate_test(x, 'RP', 20, ['o3', 'tc3', 'amigaussian1', 'fmmigaussian', 'nlpe'], 42)
    assert set(out) == set(d['st'])  # zscore, mediqr, prank only
    for k, v in d['st'].items():
        assert out[k] == pytest.approx(np.nan if v is None else v, rel=1e-8, abs=1e-8, nan_ok=True), k


@pytest.mark.parametrize('name', SD_CASES)
def test_surrogates_equal_matlab(name):
    d = FIX['sd'][name]
    x = np.array(d['y'])
    for out, ref in ((su.surrogates(x, 1, 20, 1, 'tc3', 42), d['s1']), (su.surrogates(x, 3, 20, 3, 'trev', 42), d['s3'])):
        for k, v in ref.items():
            assert out[k] == pytest.approx(np.nan if v is None else v, rel=1e-7, abs=1e-8, nan_ok=True), k


def test_sd_give_me_stats_outputs_and_constant_surrogates():
    out = su._sd_give_me_stats(1.0, np.arange(10.0), 'both')
    assert set(out) == {'zscore', 'mediqr', 'prank'}
    out = su._sd_give_me_stats(1.0, np.ones(10), 'right')
    assert np.isnan(out['zscore']) and np.isnan(out['mediqr'])
    assert out['prank'] == pytest.approx(1.0)


def test_surrogates_equal_surrogates_give_nan_kernel_outputs(monkeypatch):
    # a statistic that is the same for every surrogate
    monkeypatch.setattr(su, 'tc3', lambda x, tau: {'raw': 0.5})
    out = su.surrogates(np.random.RandomState(0).randn(60), 1, 10, 3, 'tc3', 0)
    assert isinstance(out, dict)
    for k in ('normpatponmax', 'stdfrommean', 'ztestp', 'kspminfromext', 'ksphereonmax'):
        assert np.isnan(out[k]), k


def test_make_surrogates_leave_numpy_state_alone():
    np.random.seed(5)
    before = np.random.get_state()[1].copy()
    su._make_surrogates(np.random.RandomState(1).randn(100), 'AAFT', 3, 0)
    np.random.seed(5)
    assert np.array_equal(np.random.get_state()[1], before)


# ------------------------------------------------------------------------------
# properties of the individual ports
# ------------------------------------------------------------------------------
def test_add_noise_deterministic_and_leaves_numpy_state_alone():
    y = np.random.RandomState(0).randn(300)
    np.random.seed(11)
    s0 = np.random.get_state()[1].copy()
    a = co.add_noise(y, 1, 'gaussian')
    np.random.seed(11)
    assert np.array_equal(np.random.get_state()[1], s0)
    b = co.add_noise(y, 1, 'gaussian', random_seed='default')
    assert a == b
    assert co.add_noise(y, 1, 'gaussian', random_seed=1) != a
    assert 0 <= a['fitexpr2'] <= 1


def test_add_noise_nan_fit_when_ami_constant_or_not_positive(monkeypatch):
    y = np.random.RandomState(0).randn(100)
    monkeypatch.setattr(co, 'automutual_info', lambda *a, **k: 0.5)  # a constant AMI curve
    out = co.add_noise(y, 1, 'gaussian')
    assert all(np.isnan(out[k]) for k in ('fitexpa', 'fitexpb', 'fitexpr2', 'fitexpadjr2', 'fitexprmse'))
    assert np.isfinite(out['fitlina'])
    # no positive AMI at all: nothing to fit
    calls = []

    def one_positive(*a, **k):
        calls.append(1)
        return 1.0 if len(calls) == 1 else -1.0  # only the noise-free level is positive
    monkeypatch.setattr(co, 'automutual_info', one_positive)
    out = co.add_noise(y, 1, 'gaussian')
    assert np.isnan(out['fitexpa']) and np.isfinite(out['fitlina'])


def test_num_prominent_peaks_matches_scipy_prominence():
    from scipy.signal import find_peaks, peak_prominences
    rng = np.random.RandomState(3)
    for _ in range(20):
        x = rng.randn(60)
        peaks, _ = find_peaks(x)
        prom = peak_prominences(x, peaks)[0]
        for thr in (0.1, 0.5, 1.0):
            assert co._num_prominent_peaks(x, thr) == int(np.sum(prom >= thr))


def test_num_prominent_peaks_plateau_counts_once():
    assert co._num_prominent_peaks([0, 3, 3, 3, 0], 1) == 1
    assert co._num_prominent_peaks([0, 3, 2.5, 3.2, 0], 1) == 1  # the lower of two close peaks is not prominent


def test_compare_min_ami_outputs():
    out = co.compare_min_ami(np.random.RandomState(0).randn(300), 'even', list(range(2, 21)))
    assert 'nprompeaks' in out and 'nlocmax' not in out


def test_histogram_ami_tied_quantiles_and_constant():
    y = np.repeat(np.arange(4.0), 25)  # heavily tied: fewer quantile bins
    assert np.isfinite(co.histogram_ami(y, 1, 'quantiles', 10))
    assert np.isfinite(co.histogram_ami(np.ones(50), 1, 'even', 10))  # one unit bin


def test_partial_autocorr_burg_ar2_and_constant():
    rng = np.random.RandomState(0)
    e = rng.randn(5000)
    y = np.zeros(5000)
    for i in range(2, 5000):
        y[i] = 0.5 * y[i - 1] - 0.3 * y[i - 2] + e[i]
    out = co.partial_autocorr(y, 5)
    assert out['pac_2'] == pytest.approx(-0.3, abs=0.05)
    assert abs(out['pac_4']) < 0.05
    assert all(v == 0 for v in co.partial_autocorr(np.ones(50), 4).values())
    short = co.partial_autocorr(rng.randn(6), 10)  # lags beyond N - 1 are undefined
    assert np.isnan(short['pac_7']) and np.isfinite(short['pac_5'])


def test_partial_autocorr_burg_sinusoid_gives_zeros_beyond_order():
    y = np.sin(2 * np.pi * np.arange(400) / 37.0)
    out = co.partial_autocorr(y, 6)
    assert all(abs(out[f'pac_{i}']) < 0.05 for i in range(4, 7))  # (a sinusoid is predictable from its two past values)


def test_embed2_dist_equal_distances_nan():
    out = co.embed2_dist(np.arange(100, dtype=float), 1)
    assert np.isnan(out['d_expfit_meandiff'])


def test_embed2_all_nan_angles_do_not_raise():
    out = co.embed2(np.ones(100), 1)
    assert out['hist10std'] == 0 and out['histent'] == 0


def test_stick_angles_constant_series_does_not_raise():
    out = co.stick_angles(np.zeros(100))
    assert out['symks_p'] == 1 and np.isnan(out['symks_n'])


def test_gaussian_ami_floor():
    ramp = np.arange(1.0, 301.0)
    v = inf.automutual_info(ramp, 3, 'gaussian')
    assert v == pytest.approx(-0.5 * np.log(1e-12))
    # MATLAB's max ignores NaN: a constant window takes the floor too
    step = np.r_[np.zeros(100), np.ones(100)]
    assert inf.automutual_info(step, 150, 'gaussian') == pytest.approx(-0.5 * np.log(1e-12))


def test_first_min_nan_on_trend_and_curve_matches_pointwise():
    ramp = np.arange(1.0, 301.0)
    assert np.isnan(inf.first_min(ramp, 'mi-gaussian'))  # flat floor, then NaN past N - 5
    rng = np.random.RandomState(2)
    y = np.convolve(rng.randn(400), np.ones(8) / 8, 'valid')
    c = inf._ami_gaussian_curve(y)
    for lag in (1, 7, 30, 120):
        assert c[lag - 1] == pytest.approx(inf.automutual_info(y, lag, 'gaussian'), abs=1e-9)


def test_multivariate_ami_collinear_lags_nan():
    out = inf.multivariate_ami(np.tile([1.0, -1.0], 100) + 0.0, 'ac', 'gaussian')
    assert isinstance(out, dict) and np.isnan(out['multiAMI']) and np.isnan(out['synergy'])
    assert np.isfinite(out['ami_tau'])


def test_mi_bin_constant_vector_is_nan():
    assert np.isnan(inf._mi_bin(np.ones(50), np.random.RandomState(0).randn(50)))
    v = np.random.RandomState(0).randn(200)
    assert np.isfinite(inf._mi_bin(v[:-1], v[1:], 'range', 'range', 10))
    assert np.isfinite(inf._mi_bin(v[:-1], v[1:], 'quantile', 'quantile', 10))


def test_distribution_entropy_constant_nan_and_explicit_edges():
    assert np.isnan(en.distribution_entropy(np.ones(100), 'hist', 10))
    assert np.isnan(en.distribution_entropy(np.ones(100), 'ks', None))
    y = np.random.RandomState(0).randn(500)
    assert np.isfinite(en.distribution_entropy(y, 'hist', 'auto', 0.01))
    # the bandwidth is the normal-reference rule when none is given
    assert en.distribution_entropy(y, 'ks', None) != en.distribution_entropy(y, 'ks', 1.0)


def test_surprise_deterministic_and_golden_ratio_points():
    y = np.random.RandomState(0).randn(2000)
    np.random.seed(1)
    state = np.random.get_state()[1].copy()
    a = sy.surprise(y, 'dist', 20, 3, 'quantile', 100, 0)
    b = sy.surprise(y, 'dist', 20, 3, 'quantile', 100, 99)  # random_seed is ignored
    assert a == b
    np.random.seed(1)
    assert np.array_equal(np.random.get_state()[1], state)
    # all points when there are at most 2*num_iters of them
    c = sy.surprise(y[:100], 'dist', 5, 2, 'quantile', 500)
    assert c['sum'] == pytest.approx(c['mean'] * 95)


def test_surprise_periodic_series_nan_effect_size_and_short_series():
    out = sy.surprise(np.tile([0.0, 1.0, 2.0], 100), 'dist', 6, 3, 'quantile', 500)
    assert np.isnan(out['effectSize']) and np.isnan(out['tstat'])
    short = sy.surprise(np.random.RandomState(0).randn(10), 'dist', 20, 3, 'quantile')
    assert all(np.isnan(v) for v in short.values())


def test_transition_p_alphabet_r2_bounded():
    y = np.random.RandomState(0).randn(1000)
    out = sy.transition_p_alphabet(y, np.arange(2, 11), 1)
    for k, v in out.items():
        if k.endswith('r2'):
            assert 0 <= v <= 1


def test_c2g_c2t_double_precision_and_no_stale_point():
    # block 2 is cut short by a zero correlation sum: the zero row must not be counted
    e = np.exp(np.linspace(-3, 0, 12))
    c1 = np.column_stack([e, np.linspace(0.01, 1, 12) ** 2])
    c2 = np.column_stack([e, np.r_[np.linspace(0.05, 0.5, 6) ** 2, np.zeros(6)]])
    out = tisean.c2g([c1, c2])
    assert out[1].shape == (6, 3)
    ref = tisean.c2g([c1, c2[:6]])
    np.testing.assert_array_equal(out[1], ref[1])
    t = tisean.c2t([c1, c2])
    assert t[1].shape == (5, 2) and np.all(np.isfinite(t[0]))
    # steep local slopes overflowed the single-precision prefactor exp((e1 c0 - e0 c1)/(e1 - e0))
    e2 = np.exp([-1.0, -0.9, 0.0])
    cc = np.exp([-20.0, -5.0, 0.0])  # exponent 130 > 88.7
    big = tisean.c2g([np.column_stack([e2, cc])])
    assert np.all(np.isfinite(big[0]))


def test_tisean_d2_no_overflow_emulation():
    assert not hasattr(nl, '_c2g_overflows')


def test_dvv_iaaft_ranks_the_real_part():
    # the surrogate has the amplitude distribution of the data and (almost) its spectrum
    x = np.cumsum(np.random.RandomState(0).randn(300))
    perm = np.random.RandomState(1).permutation(300)
    s = nl._dvv_iaaft(x, perm)
    np.testing.assert_allclose(np.sort(s), np.sort(x))
    a, b = np.abs(np.fft.fft(x)), np.abs(np.fft.fft(s))
    assert np.mean(np.abs(a - b)) / np.mean(a) < 0.05


def test_ml_randsample_unique_and_in_range():
    rng = nl._ml_rng(0)
    v = nl._ml_randsample(997, 100, rng)
    assert len(np.unique(v)) == 100 and v.min() >= 1 and v.max() <= 997


def test_bf_random_seed_semantics():
    assert nl._bf_random_seed('default') == 0
    assert nl._bf_random_seed(3) == 3
    assert nl._bf_random_seed(-3.4) == 3
    assert nl._bf_random_seed(5e9) == 1e9
    with pytest.raises(ValueError):
        nl._bf_random_seed('weird')


@pytest.mark.skipif(not HAS_RIPSER, reason='ripser not installed')
def test_persistent_homology_h0_does_not_include_h1_intervals():
    t = np.arange(400)
    y = np.sin(2 * np.pi * t / 40) + 0.05 * np.random.RandomState(0).randn(400)
    a = nl.persistent_homology(y, 5, 3, max_dim=1)
    b = nl.persistent_homology(y, 5, 3, max_dim=0)
    assert a['totalPersistenceH0'] == pytest.approx(b['totalPersistenceH0'])
    assert a['maxPersistenceH1'] > 0 and b['maxPersistenceH1'] == 0


def test_randomize_reproducible_and_seeded():
    y = np.random.RandomState(0).randn(200)
    a = en.randomize(y, 'permute', 3)
    assert a == en.randomize(y, 'permute', 3)
    assert a != en.randomize(y, 'permute', 4)


def test_return_time_histogram_edges():
    y = np.sin(np.arange(600) / 5.0) + 0.1 * np.random.RandomState(0).randn(600)
    out = nl.return_time(y, 0.05, 50, ('ac', 1), -1, (1, 3))
    assert np.isfinite(out['hhisthist'])
