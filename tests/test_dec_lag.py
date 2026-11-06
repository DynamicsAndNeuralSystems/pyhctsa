"""The `_dec` redesign: lag-tau forms over the whole series, and `_dec` as an average over all
decimation phases.

Each lag form is checked against a direct (brute-force) implementation of its definition at
tau > 1, and against the original function at tau = 1. Phase averaging is checked against an
explicit loop over the phases.
"""
import logging

import numpy as np
import pytest

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.operations.correlation import (crinkle_statistic, crinkle_statistic_kurtosis,
                                            nonlinear_autocorr, translate_shape)
from pyhctsa.operations.entropy import sample_entropy, sample_entropy_ranked
from pyhctsa.operations.extreme_events import moving_threshold
from pyhctsa.operations.medical import pol_var, porta
from pyhctsa.operations.model_fit import local_simple
from pyhctsa.operations.scaling import higuchi_fd
from pyhctsa.operations.symbolic import binary_stats
from pyhctsa.toolboxes.physionet import sampen as _sampen_c
from pyhctsa.utils import _zscore_or_zeros, decimation_phases, get_tau, z_score


def _smooth(n, k, seed):
    rng = np.random.default_rng(seed)
    y = np.convolve(rng.standard_normal(n + k), np.ones(k) / k, 'valid')[:n]
    return z_score(y)


@pytest.fixture(scope='module')
def calc():
    logging.disable(logging.CRITICAL)
    yield FeatureCalculator()
    logging.disable(logging.NOTSET)


# ----------------------------------------------------------------------------- SampEn

def _sampen_brute(y, M, r, tau, W):
    """SampEn(m), m = 0..M-1, by direct comparison of delay templates (|i - j| > W)."""
    n = y.size
    out = np.full(M, np.nan)
    for m in range(M):
        def count(length):
            # templates of `length` points whose (m+1)-extension also fits, as PhysioNet/Lake
            starts = [i for i in range(n) if i + m * tau < n - (0 if length == m + 1 else 0)]
            c = 0
            for a in range(len(starts)):
                for b in range(a + 1, len(starts)):
                    i, j = starts[a], starts[b]
                    if j - i <= W:
                        continue
                    if length == m and (j + m * tau >= n):  # B: must be extendable
                        continue
                    if i + (length - 1) * tau >= n or j + (length - 1) * tau >= n:
                        continue
                    if all(abs(y[i + k * tau] - y[j + k * tau]) < r for k in range(length)):
                        c += 1
            return c
        A = count(m + 1)
        if m == 0:
            k = n - (W + 1)
            B = k * (k + 1) / 2
        else:
            B = count(m)
        if A > 0 and B > 0:
            out[m] = -np.log(A / B)
    return out


@pytest.mark.parametrize('tau,W', [(1, 0), (1, 3), (3, 0), (3, 3), (5, 7)])
def test_sampen_core_matches_direct_count(tau, W):
    y = _smooth(90, 6, 0)
    ref = _sampen_brute(y, 4, 0.3, tau, W)
    got = _sampen_c.calculate(np.ascontiguousarray(y), 4, 0.3, False, tau, W + 1)[:4]
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, equal_nan=True)


def test_sample_entropy_defaults_unchanged():
    y = _smooth(400, 10, 1)
    a = sample_entropy(y, 5, 0.2)
    b = sample_entropy(y, 5, 0.2, tau=1, theiler_win=0)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_equal(a[k], b[k])


def test_sample_entropy_theiler_removes_trivial_matches():
    """On an oversampled series, delayed SampEn without a window is pulled down by matches between
    templates closer than the delay; the window removes them."""
    y = _smooth(2000, 40, 2)
    plain = sample_entropy(y, 4, 0.2, tau='ac1e')
    windowed = sample_entropy(y, 4, 0.2, tau='ac1e', theiler_win=['ac1e', 1])
    assert windowed['sampen4'] > plain['sampen4'] + 0.1


def test_sample_entropy_ranked_tau_on_ranks():
    y = _smooth(600, 12, 3)
    a = sample_entropy_ranked(y, 4, 0.2, tau='ac1e', theiler_win=['ac1e', 1])
    b = sample_entropy_ranked(np.exp(2 * y), 4, 0.2, tau='ac1e', theiler_win=['ac1e', 1])
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- lag forms

TAU = 7


def test_nonlinear_autocorr_lag():
    y = _smooth(500, 15, 4)
    for taus, absval in (([0, 3, 6], False), ([1, 1, 2], False), ([1, 2, 3, 4], True)):
        lags = np.array(taus) * TAU
        t0 = lags.max()
        prod = np.array([y[t] * np.prod([y[t - l] for l in lags]) for t in range(t0, y.size)])
        ref = np.mean(np.abs(prod)) if absval else np.mean(prod)
        np.testing.assert_allclose(nonlinear_autocorr(y, taus, absval, tau=TAU), ref, rtol=1e-12)


def test_crinkle_lag():
    y = _smooth(400, 15, 5)
    x = y - y.mean()
    d2 = np.array([x[t - TAU] - 2 * x[t] + x[t + TAU] for t in range(TAU, y.size - TAU)])
    np.testing.assert_allclose(crinkle_statistic(y, tau=TAU), np.mean(d2 ** 4) / np.mean(x ** 2) ** 2,
                               rtol=1e-12)
    assert np.isfinite(crinkle_statistic_kurtosis(y, tau=TAU))


def test_porta_lag_counts_words_over_all_t():
    y = _smooth(400, 15, 6)
    out = porta(y, 6, tau=TAU)
    base = porta(y, 6)
    # levels from the whole series, as in the tau = 1 call; words (s_t, s_{t+tau}, s_{t+2tau})
    from pyhctsa.robust import bf_hist_edges
    edges = bf_hist_edges(y, 6)
    sym = np.clip(np.searchsorted(edges, y, side='right'), 1, len(edges) - 1)
    n0 = sum(sym[t] == sym[t + TAU] == sym[t + 2 * TAU] for t in range(y.size - 2 * TAU))
    np.testing.assert_allclose(out['pV0'], 100 * n0 / (y.size - 2 * TAU))
    np.testing.assert_allclose(sum(out.values()), 100)
    assert out['pV0'] < base['pV0']  # steps tau apart are rarely flat


def test_pol_var_lag_counts_runs_along_chains():
    y = _smooth(500, 15, 7)
    D = 3
    dx = np.abs(y[TAU:] - y[:-TAU]) >= 1
    pc = 0
    for p in range(TAU):
        run = 0
        for v in list(dx[p::TAU]) + [None]:
            if run and (v is None or v != prev):
                pc += run // D
                run = 0
            if v is not None:
                run += 1
                prev = v
    np.testing.assert_allclose(pol_var(y, 1, D, tau=TAU), pc / dx.size)


def test_translate_shape_lag():
    y = _smooth(300, 15, 8)
    out = translate_shape(y, 'circle', 2.5, 'pts', tau=TAU)
    w = 2
    counts = [sum(j ** 2 + (y[t + j * TAU] - y[t]) ** 2 <= 2.5 ** 2 for j in range(-w, w + 1))
              for t in range(w * TAU, y.size - w * TAU)]
    np.testing.assert_allclose(out['mean'], np.mean(counts))
    np.testing.assert_allclose(out['std'], np.std(counts, ddof=1))


def test_local_simple_lag():
    y = _smooth(400, 15, 9)
    out = local_simple(y, 'mean', 1, tau=TAU)
    e = y[:-TAU] - y[TAU:]
    np.testing.assert_allclose(out['meane'], e.mean())
    np.testing.assert_allclose(out['meanabs'], np.mean(np.abs(e)))
    ez = _zscore_or_zeros(e)
    ac1 = np.sum(ez[:-TAU] * ez[TAU:]) / (np.sum(ez ** 2))
    np.testing.assert_allclose(out['ac1'], ac1, atol=1e-10)


def test_higuchi_lag_scales():
    y = _smooth(800, 15, 10)
    out = higuchi_fd(y, 8, tau=TAU)
    assert np.isfinite(out['HFD'])
    # tau = 1 is Higuchi's method
    a, b = higuchi_fd(y, 8), higuchi_fd(y, 8, tau=1)
    for k in a:
        np.testing.assert_equal(a[k], b[k])


@pytest.mark.parametrize('f', [
    lambda y, t: nonlinear_autocorr(y, [0, 3, 6], False, tau=t),
    lambda y, t: crinkle_statistic(y, tau=t),
    lambda y, t: pol_var(y, 1, 3, tau=t),
])
def test_lag_forms_nan_without_delay(f):
    assert np.isnan(f(np.ones(50), 'ac1e'))


# ----------------------------------------------------------------------------- phase averaging

def test_decimation_phases():
    y = _smooth(500, 12, 11)
    tau = int(get_tau(y, 'ac1e'))
    phases = decimation_phases(y, 'ac1e')
    assert len(phases) == tau
    for p, yp in enumerate(phases):
        np.testing.assert_allclose(yp, _zscore_or_zeros(y[p::tau]))


def test_dec_is_phase_average(calc):
    y = _smooth(1500, 12, 12)
    x = z_score(y)
    tau = int(get_tau(x, 'ac1e'))
    f = calc.feature_funcs['moving_threshold_0p1_0p1_dec']
    got = f(y)
    outs = [moving_threshold(_zscore_or_zeros(x[p::tau]), 0.1, 0.1) for p in range(tau)]
    for k, v in got.items():
        vals = np.array([o[k] for o in outs], dtype=float)
        np.testing.assert_allclose(v, np.nanmean(vals), rtol=1e-12)


def test_dec_rank_uses_rank_delay(calc):
    y = _smooth(1500, 12, 13)
    x = z_score(y)
    tau = int(get_tau(x, 'ac1e-rank'))
    got = calc.feature_funcs['binary_stats_diff_dec-rank'](y)
    outs = [binary_stats(_zscore_or_zeros(x[p::tau]), 'diff') for p in range(tau)]
    np.testing.assert_allclose(got['pupstat2'], np.nanmean([o['pupstat2'] for o in outs]), rtol=1e-12)


def test_moving_threshold_1_0p02_dec(calc):
    f = calc.feature_funcs['moving_threshold_1_0p02_dec']
    out = f(_smooth(3000, 4, 14))
    assert not {'pkick', 'meankickf', 'mediankickf', 'stdkickf', 'meankicksize'} & set(out)
    # decimated phases shorter than the barrier's memory (1/b = 50): undefined
    short = f(_smooth(1000, 60, 15))
    assert all(np.isnan(v) for v in short.values())


def test_registered_labels(calc):
    labels = set(calc.feature_funcs)
    for new in ['sample_entropy_5_0p2_ac1e', 'higuchi_fd_8_ac1e', 'local_simple_mean_1_ac1e',
                'ac_nl_0_3_6_ac1e', 'ac_nl_1_1_2_ac1e', 'ac_nl_1_to_4_absval_ac',
                'translate_shape_2p5_ac1e', 'crinkle_statistic_ac1e', 'pol_var_3_ac1e',
                'moving_threshold_0p1_0p1_dec', 'moving_threshold_1_0p02_dec',
                'binary_stats_diff_dec-rank']:
        assert new in labels, new
    for gone in ['lempel_ziv_complexity_2_dec', 'lempel_ziv_complexity_5_dec',
                 'sample_entropy_5_0p2_dec', 'higuchi_fd_8_dec', 'binary_stats_diff_dec']:
        assert gone not in labels, gone
