"""Accuracy tests for fluctuation analysis (mostly detrended fluctuation analysis, wtf='dfa')."""
import numpy as np
import pytest

from scipy import stats

from pyhctsa.operations.scaling import (
    _broken_line_fit, _detrend_windows, _fluctuations, _log_tau_grid, fluctuation_analysis)
from pyhctsa.utils import get_tau, z_score


def fgn(n, hurst, seed):
    """Fractional Gaussian noise by the Davies-Harte circulant embedding method."""
    rng = np.random.default_rng(seed)
    k = np.arange(n + 1, dtype=float)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                   + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([gamma, gamma[-2:0:-1]])  # first row of the circulant, length 2n
    lam = np.fft.fft(row).real
    assert lam.min() > -1e-10
    lam = np.clip(lam, 0, None)
    m = row.size
    w = np.zeros(m, dtype=complex)
    w[0] = np.sqrt(lam[0] / m) * rng.standard_normal()
    w[m // 2] = np.sqrt(lam[m // 2] / m) * rng.standard_normal()
    j = np.arange(1, m // 2)
    a, b = rng.standard_normal(j.size), rng.standard_normal(j.size)
    w[j] = np.sqrt(lam[j] / (2 * m)) * (a + 1j * b)
    w[m - j] = np.conj(w[j])
    return np.fft.fft(w).real[:n]


def alpha(x, **kw):
    return fluctuation_analysis(x, wtf='dfa', **kw)['alpha']


def ar1(phi, n, rng, burn=300):
    e = rng.standard_normal(n + burn)
    y = np.empty_like(e)
    y[0] = e[0]
    for t in range(1, len(e)):
        y[t] = phi * y[t - 1] + e[t]
    return y[burn:]


def auc(a, b):
    """P(B > A) + P(B = A)/2, the Mann-Whitney AUC."""
    a, b = np.asarray(a), np.asarray(b)
    return np.mean(b[:, None] > a[None, :]) + 0.5 * np.mean(b[:, None] == a[None, :])


def reference(x, k=1, n_tau=50):
    """Loop-based DFA-k with windows from both ends, numpy.polyfit detrending, scipy's Theil-Sen slope
    over tau in [max(3(k+1), 5, 8 tau_c), N/2], and a brute-force broken line over [max(3(k+1), 5), N/2]."""
    y = np.cumsum(x - np.mean(x))
    n = len(y)

    def grid(lo, hi):
        return sorted({int(np.floor(v + 0.5)) for v in np.exp(np.linspace(np.log(lo), np.log(hi), n_tau))})

    def F(tau):
        tt = np.arange(tau)
        nw = n // tau
        starts = [w * tau for w in range(nw)] + ([n - (w + 1) * tau for w in range(nw)] if n % tau else [])
        res = [y[s:s + tau] - np.polyval(np.polyfit(tt, y[s:s + tau], k), tt) for s in starts]
        return np.sqrt(np.mean(np.concatenate(res) ** 2))

    tau_0 = max(5, 3 * (k + 1))
    tau_c = get_tau(x, 'ac1e')
    taus = grid(max(tau_0, 8 * tau_c), n // 2)
    a = stats.theilslopes(np.log([F(t) for t in taus]), np.log(taus))[0]
    taus = np.array(grid(tau_0, n // 2))
    u, v = np.log(taus), np.log([F(t) for t in taus])
    dec = (u - u[0]) / np.log(10)
    cand = [i for i in range(len(u)) if dec[i] >= 0.3 and dec[-1] - dec[i] >= 0.3 and i >= 5 and len(u) - i >= 6]
    # the same broken line in another basis: intercept, slope, and the change of slope after the breakpoint
    fits = [np.linalg.lstsq(np.column_stack([np.ones_like(u), u, np.maximum(u - u[i], 0)]), v, rcond=None)
            for i in cand]
    sse = [np.sum((v - np.column_stack([np.ones_like(u), u, np.maximum(u - u[i], 0)]) @ f[0]) ** 2)
           for i, f in zip(cand, fits)]
    j = int(np.argmin(sse))
    c = fits[j][0]
    return a, c[1], c[1] + c[2], j in (0, len(cand) - 1)


FIELDS = {'alpha', 'ssr', 'r1_alpha', 'r2_alpha', 'alphadiff', 'r1_ssr', 'r2_ssr'}


@pytest.mark.parametrize("n,k,seed", [(400, 1, 0), (1000, 1, 1), (1000, 2, 2), (3001, 3, 3)])
def test_matches_a_loop_reference(n, k, seed):
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.standard_normal(n).cumsum() + ar1(0.6, n, rng)
    out = fluctuation_analysis(x, wtf='dfa', k=k)
    a, r1, r2, edge = reference(x, k)
    assert set(out) == FIELDS
    assert out['alpha'] == pytest.approx(a, abs=1e-10)
    if edge:
        assert np.isnan(out['r1_alpha']) and np.isnan(out['r2_alpha'])
    else:
        assert out['r1_alpha'] == pytest.approx(r1, abs=1e-9)
        assert out['r2_alpha'] == pytest.approx(r2, abs=1e-9)
        assert out['alphadiff'] == pytest.approx(r1 - r2, abs=1e-9)


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7, 0.9])
def test_fgn_exponent_matches_hurst(hurst):
    est = [alpha(fgn(2000, hurst, seed), k=1) for seed in range(10)]
    assert abs(np.mean(est) - hurst) < 0.05


def test_white_noise_exponent_is_half():
    est = [alpha(np.random.default_rng(s).standard_normal(2000)) for s in range(10)]
    assert abs(np.mean(est) - 0.5) < 0.05


def test_alpha_is_nan_when_the_series_decorrelates_too_slowly():
    # a random walk: its sample ACF stays above 1/e for a large fraction of N, so the range from 8 tau_c
    # to N/2 is too short for a scaling exponent; the broken line still describes the curve
    out = fluctuation_analysis(np.cumsum(fgn(2000, 0.5, 1)), wtf='dfa', k=2)
    assert np.isnan(out['alpha']) and np.isnan(out['ssr'])
    assert out['r1_alpha'] == pytest.approx(1.5, abs=0.15)


def test_alpha_is_less_affected_by_short_range_correlation_than_a_fixed_range():
    rng = np.random.default_rng(77)
    n, reps = 5000, 20
    ar9 = [z_score(ar1(0.9, n, rng)) for _ in range(reps)]   # exponent 0.5, correlation time about 10

    def fixed_range(x):
        # the slope over tau from 6 to N/2, as fitted without regard to the correlation time
        taus = _log_tau_grid(6, n // 2, 50)
        f = _fluctuations(np.cumsum(x - x.mean()), taus, 'dfa', 2, 1)
        return stats.theilslopes(np.log(f), np.log(taus))[0]

    assert np.mean([fixed_range(x) for x in ar9]) > 0.75
    assert np.mean([alpha(x) for x in ar9]) < 0.65
    # fGn with H = 0.75 against AR(1) with the same lag-1 autocorrelation
    rho1 = 0.5 * (2 ** 1.5 - 2)
    ar_m = [z_score(ar1(rho1, n, rng)) for _ in range(reps)]
    fg = [z_score(fgn(n, 0.75, 1000 + s)) for s in range(reps)]
    assert auc([alpha(x) for x in ar_m], [alpha(x) for x in fg]) > 0.9


def test_broken_line_recovers_two_exact_slopes():
    u = np.log(np.unique(np.round(np.exp(np.linspace(np.log(6), np.log(500), 50)))))
    s = np.log(40)
    v = 0.3 + 1.2 * np.minimum(u - s, 0) + 0.6 * np.maximum(u - s, 0)
    out = _broken_line_fit(u, v)
    assert out['r1_alpha'] == pytest.approx(1.2, abs=1e-12) and out['r2_alpha'] == pytest.approx(0.6, abs=1e-12)
    assert out['alphadiff'] == pytest.approx(0.6, abs=1e-12)
    assert out['r1_ssr'] == pytest.approx(0, abs=1e-20) and out['r2_ssr'] == pytest.approx(0, abs=1e-20)
    # a change of slope below the smallest candidate breakpoint is not resolved: NaN, not a forced split
    s = np.log(8)
    v = 1.2 * np.minimum(u - s, 0) + 0.6 * np.maximum(u - s, 0)
    assert all(np.isnan(val) for val in _broken_line_fit(u, v).values())


def test_mfdfa_monofractal_h_is_constant_in_q():
    qs = [-2, 0, 0.5, 2, 4]
    h = np.array([[alpha(fgn(2000, 0.7, s), q=q) for q in qs] for s in range(10)]).mean(axis=0)
    assert np.all(np.abs(h - 0.7) < 0.08), h
    assert np.ptp(h) < 0.1


def test_noninteger_q_between_neighbours():
    x = fgn(2000, 0.7, 3)
    a = alpha(x, q=2.5)
    assert np.isfinite(a) and abs(a - 0.7) < 0.2


def test_q2_is_rms_of_window_detrended_profile():
    # F_2 from the MF-DFA formula equals the pointwise RMS of all detrended profile points
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal((40, 25)) , axis=0)
    r = _detrend_windows(y, 1)
    from pyhctsa.operations.scaling import _dfa_fluctuation
    assert np.isclose(_dfa_fluctuation(r, 2), np.sqrt(np.mean(r ** 2)))
    var = np.mean(r ** 2, axis=0)
    assert np.isclose(_dfa_fluctuation(r, 4), np.mean(var ** 2) ** 0.25)
    assert np.isclose(_dfa_fluctuation(r, 0), np.exp(0.5 * np.mean(np.log(var))))
    assert np.isclose(_dfa_fluctuation(r, -1.5), np.mean(var ** -0.75) ** (-1 / 1.5))


def test_mean_offset_and_scale_invariance():
    x = fgn(1500, 0.6, 5)
    for k in (1, 2):
        ref = fluctuation_analysis(x - x.mean(), wtf='dfa', k=k)
        off = fluctuation_analysis(3 * x + 1000.0, wtf='dfa', k=k)
        for key in ('alpha', 'r1_alpha', 'r2_alpha', 'alphadiff'):
            assert np.isclose(ref[key], off[key], rtol=1e-8, atol=1e-8) or (np.isnan(ref[key]) and np.isnan(off[key])), (k, key)
    # with a lag the decimated series has its own mean
    a = fluctuation_analysis(x, wtf='dfa', k=1, lag=2)['alpha']
    b = fluctuation_analysis(x + 3.0, wtf='dfa', k=1, lag=2)['alpha']
    assert np.isclose(a, b, rtol=1e-8)


@pytest.mark.parametrize("wtf", ['rsrange', 'rsrangefit', 'std', 'iqr', 'range', 'endptdiff'])
def test_other_methods_give_half_for_white_noise(wtf):
    est = [fluctuation_analysis(np.random.default_rng(s).standard_normal(4000), wtf=wtf)['alpha'] for s in range(6)]
    assert abs(np.mean(est) - 0.5) < 0.1


def test_short_and_degenerate_inputs_give_nan_without_raising():
    x = np.random.default_rng(0).standard_normal(200)
    for n in range(0, 60):
        out = fluctuation_analysis(x[:n], wtf='dfa')   # never raises
        assert set(out) == FIELDS
        if n < 30:   # fewer than 8 distinct timescales from 8 to N/2
            assert all(np.isnan(v) for v in out.values())
    assert all(np.isnan(v) for v in fluctuation_analysis(np.r_[x, np.nan], wtf='dfa').values())
    with pytest.raises(ValueError):
        fluctuation_analysis(x, wtf='nothing')


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("tau", [5, 17, 100])
def test_vectorised_detrend_equals_polyfit_loop(k, tau):
    rng = np.random.default_rng(tau * 10 + k)
    y = np.cumsum(rng.standard_normal((tau, 30)), axis=0)
    ref = np.empty_like(y)
    tt = np.arange(tau)
    for j in range(y.shape[1]):
        ref[:, j] = y[:, j] - np.polyval(np.polyfit(tt, y[:, j], k), tt)
    np.testing.assert_allclose(_detrend_windows(y, k), ref, rtol=0, atol=1e-12)


def test_constant_series_does_not_raise():
    out = fluctuation_analysis(np.ones(500), wtf='dfa', q=-2)
    assert isinstance(out, dict)
