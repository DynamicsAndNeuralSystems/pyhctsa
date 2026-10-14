"""Regression tests for definitions converged between pyhctsa and hctsa v3 (fragile fits, random
draws, configuration slips)."""
import numpy as np
import pytest

from pyhctsa.operations.stationarity import std_nth_deriv_change


def test_std_nth_deriv_change_steep_growth_global_fit():
    # Differencing white noise multiplies the std by about 2 per order, so the exponential fit
    # must find b near log(2) with R^2 near 1 (a local optimizer from a fixed start could stop
    # at a spurious minimum with b around -9 and a negative R^2 when the curve is small in scale)
    rng = np.random.default_rng(1)
    for scale in (1.0, 0.01):
        y = scale * rng.standard_normal(1000)
        out = std_nth_deriv_change(y, 10)
        assert 0.55 < out['fexp_b'] < 0.75
        assert out['fexp_r2'] > 0.99
        assert out['fexp_a'] > 0


def test_std_nth_deriv_change_fit_matches_least_squares():
    # The fitted (a, b) is a least-squares optimum: no small perturbation lowers the SSE
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(500))
    out = std_nth_deriv_change(y, 10)
    n = np.arange(1, 11)
    ms = np.array([np.std(np.diff(y, k), ddof=1) for k in n])
    sse = lambda a, b: np.sum((ms - a * np.exp(b * n)) ** 2)
    s0 = sse(out['fexp_a'], out['fexp_b'])
    for da, db in [(1e-4, 0), (-1e-4, 0), (0, 1e-5), (0, -1e-5)]:
        assert sse(out['fexp_a'] * (1 + da), out['fexp_b'] + db) >= s0 * (1 - 1e-9)


def test_distribution_entropy_ks_configs_use_automatic_bandwidth():
    # hctsa's EN_DistributionEntropy(x_z,'ks',[],olremp): the empty numBins is the automatic
    # (normal-reference) bandwidth, not the histogram default of 10 used as a bandwidth
    from pyhctsa.calculator import FeatureCalculator
    from pyhctsa.operations.entropy import distribution_entropy
    from pyhctsa.utils import z_score
    fc = FeatureCalculator()
    ks = [k for k in fc.feature_funcs if k.startswith('distribution_entropy_ks')]
    assert len(ks) == 6
    y = np.random.default_rng(3).standard_t(4, 800)
    for lab in ks:
        olremp = float(lab.rsplit('_', 1)[1].replace('0p', '0.'))
        assert fc.feature_funcs[lab](y) == pytest.approx(distribution_entropy(z_score(y), 'ks', None, olremp), abs=1e-12)


def test_compare_ks_fit_singular_densities_bounded():
    # Fitted densities unbounded at 0 (gamma/Weibull shape < 1) or at the ends (a U-shaped beta)
    # are represented by cell averages, so the area between the curves stays within [0, 2]
    from pyhctsa.operations.distribution import compare_ks_fit
    rng = np.random.default_rng(4)
    x_pos = rng.gamma(0.5, 2.0, 1000)
    u_shape = rng.beta(0.4, 0.4, 1000)
    for x, d in [(x_pos, 'gamma'), (x_pos, 'wbl'), (u_shape, 'beta'), (rng.uniform(size=500), 'uni')]:
        out = compare_ks_fit(x, d)
        assert 0 <= out['adiff'] <= 2
        assert np.isfinite(out['peaksepy']) and np.isfinite(out['olapint']) and np.isfinite(out['relent'])
    assert np.isnan(compare_ks_fit(rng.uniform(size=500), 'uni')['peaksepx'])


def test_compare_ks_fit_gamma_needs_positive_data():
    from pyhctsa.operations.distribution import compare_ks_fit
    x = np.random.default_rng(5).gamma(2.0, 1.0, 500)
    assert isinstance(compare_ks_fit(x, 'gamma'), dict)
    x[10] = 0.0
    out = compare_ks_fit(x, 'gamma')
    assert all(np.isnan(v) for v in (out.values() if isinstance(out, dict) else [out]))


def test_compare_ks_fit_cell_average_close_to_pdf_for_smooth_fits():
    # For a smooth fitted density the cell average differs from the point value by O(dx^2)
    from pyhctsa.operations.distribution import compare_ks_fit
    x = np.random.default_rng(6).standard_normal(1000)
    out = compare_ks_fit(x, 'norm')
    # the overlap integral of a normal with itself, times its sd, is 1/(2 sqrt(pi)) ~ 0.2821
    assert out['olapint'] == pytest.approx(0.2821, abs=0.02)
    assert out['peaksepy'] == pytest.approx(0.0, abs=0.05)


def test_cepstrum_independent_of_the_rounding_of_the_mean():
    # The zero-frequency bin is set to the envelope, so a rounding-level change in the mean of a
    # z-scored series (|X(0)| ~ 1e-13 vs 0) does not shift the whole cepstrum
    from pyhctsa.operations.spectral import cepstrum
    rng = np.random.default_rng(7)
    y = rng.standard_normal(1000)
    y = (y - y.mean()) / y.std(ddof=1)
    a = cepstrum(y)
    b = cepstrum(y + 1e-15)
    for k in ('meanCeps', 'period', 'CPP', 'peakRatio'):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)


def test_polysat_fit_global_and_bounded():
    from pyhctsa.operations.spectral import _polysat_fit
    w = np.linspace(0, np.pi, 200)
    y = 3.0 * w ** 2 / (0.4 + w ** 2)
    f = _polysat_fit(w, y)
    assert f['a'] == pytest.approx(3.0, rel=1e-6) and f['b'] == pytest.approx(0.4, rel=1e-6)
    assert f['r2'] == pytest.approx(1.0, abs=1e-12)
    # a curve that keeps rising like w^2: b goes to the upper limit, a/b is the curvature
    f = _polysat_fit(w, 0.7 * w ** 2)
    assert f['b'] == pytest.approx(100 * np.pi ** 2, rel=1e-6)
    assert f['a'] / f['b'] == pytest.approx(0.7, rel=0.02)


def test_saturating_exp_fit():
    from pyhctsa.operations.nonlinearity import _saturating_exp_fit
    t = np.arange(0, 200, dtype=float)
    p = 2.5 * (1 - np.exp(-0.05 * t))
    f = _saturating_exp_fit(t, p)
    assert f['a'] == pytest.approx(2.5, rel=1e-6) and f['b'] == pytest.approx(-0.05, rel=1e-6)
    assert f['r2'] == pytest.approx(1.0, abs=1e-12)
    # instant saturation: the rate is reported at the limit of the search (5 per step)
    p = np.r_[0.0, np.full(199, 3.0)]
    f = _saturating_exp_fit(t, p)
    assert f['b'] == pytest.approx(-5.0, rel=1e-6) and f['a'] == pytest.approx(3.0, rel=1e-3)
    assert np.isnan(_saturating_exp_fit(t, np.zeros(200))['b'])


def test_mma_zero_fluctuation_segments():
    # A series with long constant runs: segments fitted exactly by the quadratic detrending are
    # left out of the q < 0 moments, so the exponents stay in a sensible range
    from pyhctsa.operations.scaling import mma
    rng = np.random.default_rng(8)
    y = np.zeros(1000)
    on = rng.random(1000) < 0.05
    y[on] = rng.exponential(1.0, on.sum())
    y = (y - y.mean()) / y.std(ddof=1)
    out = mma(y)
    assert np.isfinite(out['meanHurstExponent']) and out['maxHurstExponent'] < 5


def test_dvv_portable_reference_values():
    # The same input and seed give these values in hctsa (Tests/Robust_convergeTests.m)
    from pyhctsa.robust import bf_random
    from pyhctsa.operations.nonlinearity import dvv
    y = np.cumsum(bf_random(400, 7, 'normal'))
    y = (y - y.mean()) / y.std(ddof=1)
    o = dvv(y, 3, 50, 2, 20, 4, 0, 1)
    assert o['trend'] == pytest.approx(0.32953051863335436, abs=1e-10)
    assert o['max'] == pytest.approx(0.9565440085565462, abs=1e-10)
    assert o['meanDiff'] == pytest.approx(0.06196142307459889, abs=1e-10)
    assert o['rmsDiffSurr'] == pytest.approx(0.0069117290619591055, abs=1e-10)
    assert o['meanNormCDF'] == pytest.approx(0.3732281477819768, abs=1e-10)
    assert o['numZeroCrossings'] == 2


def test_marginal_tests_logjb():
    # log of the Jarque-Bera statistic n/6 (S^2 + (K-3)^2/4), biased moments (Jarque & Bera 1987);
    # the same value is pinned in hctsa's Tests/Robust_convergeTests.m
    from pyhctsa.operations.hypothesis_tests import marginal_tests
    x = np.array([1., 2, 3, 4, 10, 2, 3, 1, 0, 5])
    assert marginal_tests(x, 'logjb') == pytest.approx(1.5018106117098595, abs=1e-12)
    assert marginal_tests(3 * x + 7, 'logjb') == pytest.approx(1.5018106117098595, abs=1e-12)
    assert np.isnan(marginal_tests(np.ones(20), 'logjb'))
