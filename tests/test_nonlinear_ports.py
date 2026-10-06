"""Nonlinear ports: false nearest neighbors and the embedding dimension, TISEAN c1 and lyap_spec,
correlation sums at m = 1 (expected values generated with MATLAB hctsa and the TISEAN binaries)."""
import numpy as np
import pytest

from pyhctsa.operations import nonlinearity as nl
from pyhctsa.toolboxes.Tisean_3_0_1 import tisean as _tisean


def _x(n=400):
    t = np.arange(1, n + 1)
    x = (np.sin(0.3 * t) + 0.5 * np.cos(1.7 * t) ** 3 + 0.2 * np.sin(0.011 * t ** 1.5)
         + 0.4 * np.sin(0.77 * t ** 1.1))
    return (x - x.mean()) / x.std(ddof=1)


def _all_nan(out):
    """True for NaN, or for the dict of NaN fields a failed dict-valued function returns."""
    return all(np.isnan(v) for v in out.values()) if isinstance(out, dict) else bool(np.isnan(out))


def test_fnn_embedding_dimension():
    # BF_Embed(x, tau, 'fnn', true)
    assert nl._embedding_params(_x(), 'mi', 'fnn') == (3, 3)
    assert nl._embedding_params(_x(), 'ac1e', 'fnn') == (4, 2)
    assert nl._embedding_params(_x(), 'ac1e', ['fnn', 0.05]) == (4, 5)
    assert nl._embed_tau_m(_x(), ['mi', 'fnn']) == (3, 3)
    assert nl._embedding_params(_x(), 2, 6) == (2, 6)
    assert nl._embedding_params(np.ones(100), 'ac1e', 'fnn') is None
    with pytest.raises(ValueError):
        nl._embedding_params(_x(), 1, 'nonsense')


def test_fnn_users():
    x = _x()
    assert abs(nl.nlpe(x, 'fnn', 'mi', 5000, ('ac', 1))['msqerr'] - 0.207584162764) < 1e-9
    d = nl.local_density(x, 5, ('ac', 1), 'ac1e', 'fnn')
    assert abs(d['meanden'] + 2.37340122116) < 1e-9 and abs(d['stdden'] - 0.723311553717) < 1e-9
    d = nl.gp_corr_sum(x, -1, 0.1, ('ac', 1), 20, ('ac', 'fnn'))
    assert abs(d['robfit_a2'] - 3.66955087508) < 1e-9 and abs(d['meanlnCr'] + 8.32376735638) < 1e-9
    assert abs(nl.takens_estimator(x, -1, 0.05, ('ac', 1), ('mi', 'fnn')) - 2.14107670491) < 1e-9
    assert _all_nan(nl.nlpe(np.ones(100), 'fnn'))


def test_fnn():
    d = nl.fnn(_x(), 1, 10, ('ac', 1), False, escape_factor=5)
    assert abs(d['pfnn_3'] - 0.265233) < 1e-6 and d['pfnn_7'] == 0
    assert abs(d['nHood2_4'] - 0.1496798) < 1e-6 and abs(d['meannHood2'] - 0.131032955) < 1e-6
    assert abs(d['stdpfnn'] - 0.331280804587) < 1e-6 and abs(d['max1stepchange'] - 0.4061888) < 1e-6
    assert abs(d['mdrop'] + 0.108831911111) < 1e-6
    assert (d['firstunder09'], d['firstunder05'], d['firstunder02'], d['firstunder005']) == (2, 3, 4, 5)
    assert d['pdrop'] == pytest.approx(2 / 3)
    assert nl.fnn(_x(), 1, 10, ('ac', 1), True, 0.4, 5) == 3
    assert nl.fnn(_x(), 'ac', escape_factor=5, just_best=True, bestp=0.05) == 6
    assert np.isnan(nl.fnn(_x()[:8])) and np.isnan(nl.fnn(np.ones(100)))


def test_fnn_delay_is_the_lag_between_coordinates():
    # TISEAN 3.0.1 ignores -d for a scalar series (lag 1 whatever the delay); here the delay is the lag.
    # Reference values from hctsa's false_nearest binary (lag = delay) on a Lorenz x series.
    x = _lorenz_x(3000)
    f1 = _tisean.false_nearest(x, 1, 1, 5, 50, 5.0, 7)['pfnn']
    f30 = _tisean.false_nearest(x, 30, 1, 5, 50, 5.0, 7)['pfnn']
    # delay 1 is unchanged (the original program's values)
    assert np.allclose(f1, [0.9489149, 0.01035058, 0.002003339, 0, 0], atol=1e-6)
    # delay 30: the false-neighbor fraction stays high at m = 2 .. 5 (it was ~0 from m = 3 with lag 1)
    assert np.allclose(f30, [0.9824561, 0.4029484, 0.2250270, 0.2035503, 0.2013487], atol=1e-6)
    # a delay-d embedding of x with each value repeated d times is the lag-1 embedding of x
    d, z = 4, x[:1500]
    r = _tisean.false_nearest(np.repeat(z, d), d, 1, 4, 0, 5.0, None)['pfnn']
    r1 = _tisean.false_nearest(z, 1, 1, 4, 0, 5.0, None)['pfnn']
    assert np.allclose(r, r1, atol=0.01)


def _lorenz_x(n):
    def f(s):
        return np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
    s, h = np.array([1., 1., 1.]), 0.01
    def rk4(s):
        k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
        return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for _ in range(5000):
        s = rk4(s)
    x = np.empty(n)
    for i in range(n):
        s = rk4(rk4(s))
        x[i] = s[0]
    return x


def _x600():
    return _x(600)


def test_tisean_c1():
    x = _x600()
    # the c1 + c2d -a2 output of hctsa's patched TISEAN binary (-d1 -m1 -M5 -t12 -n300): [length scale, slope]
    # (centers: the first of the golden-ratio lattice ordering; float32 output, so rtol 1e-6)
    curves = nl._c2d_slopes(nl._c1_curves(x, 1, 1, 5, 12, 300))
    assert [c.shape[0] for c in curves] == [14] * 5
    for blk, first, last in [(0, (0.00616703741, 0.996788919), (0.898289323, 0.905884266)),
                             (2, (0.200101465, 2.73352361), (1.58181274, 1.49736857)),
                             (4, (0.349511266, 3.41429496), (1.92375433, 1.81590235))]:
        np.testing.assert_allclose(curves[blk][0], first, rtol=1e-6)
        np.testing.assert_allclose(curves[blk][-1], last, rtol=1e-6)
    # hctsa's NL_c1 (MATLAB); at least 500 reference points when the series is longer than 500
    d = nl.tisean_c1(x, 1, [1, 5], 0.02, 0.5)
    assert abs(d['bestestd'] - 0.992521771923) < 1e-8 and abs(d['bestestdstd'] - 0.0209969158853) < 1e-8
    assert abs(d['bestgoodness'] + 0.0440030841147) < 1e-8 and abs(d['mediand'] - 2.79020823714) < 1e-8
    assert abs(d['maxd'] - 3.5159757725) < 1e-8 and abs(d['meanstd'] - 0.0390353220819) < 1e-8
    assert abs(d['longestscr'] - 4.60231915198) < 1e-8
    d = nl.tisean_c1(x, 'ac', [2, 4], 10, 150)
    assert abs(d['bestestd'] - 2.03145078833) < 1e-8 and abs(d['longestscr'] - 0.907256921961) < 1e-8
    assert abs(d['ranged'] - 1.40261139417) < 1e-5  # the m = 4 slopes differ from the binary's in float32
    # no length is trimmed (TISEAN's c1 used to hang near multiples of 128): 512 differs from 511
    d = nl.tisean_c1(x[:512], 1, [1, 5], 0.02, 0.5)
    assert abs(d['bestestd'] - 0.982899175154) < 1e-8 and abs(d['longestscr'] - 4.6681318198) < 1e-8
    assert abs(nl.tisean_c1(x[:511], 1, [1, 5], 0.02, 0.5)['bestestd'] - 0.986398293077) < 1e-8
    assert _all_nan(nl.tisean_c1(x[:99])) and _all_nan(nl.tisean_c1(np.ones(300)))
    # 520 samples with delay 2 give 512 embedded points at m = 5: stock TISEAN's c1 never finishes
    d = nl.tisean_c1(x[:520], 2, [3, 6], 0.05, 0.3)
    assert abs(d['bestestd'] - 2.744996548) < 1e-8 and abs(d['longestscr'] - 0.659132525284) < 1e-8
    d = nl.tisean_c1(x[:520], 2, [1, 7], 26, 156)
    assert abs(d['bestestd'] - 0.989851166923) < 1e-8 and abs(d['meanstd'] - 0.101358611791) < 1e-8
    # a Theiler window that leaves no neighbors, or delay vectors longer than the series
    assert _all_nan(nl.tisean_c1(x[:300], 1, [2, 4], 200, 0.5))
    assert _all_nan(nl.tisean_c1(x[:150], 40, [1, 5], 3, 100))
    # more reference points than embedded points
    assert abs(nl.tisean_c1(x[:300], 2, [2, 6], 0.02, 1)['bestestd'] - 2.02472855667) < 1e-8
    with pytest.raises(ValueError):
        nl.tisean_c1(x, 'nonsense')


def test_tisean_c1_centers():
    # the centers are a fixed permutation of the embedded points (golden-ratio lattice), spread evenly
    for nmax, m, delay in [(600, 1, 1), (600, 4, 10), (101, 2, 3), (2, 1, 1)]:
        ju = np.zeros(nmax + 1, np.int64)
        nl._c1_centers(nmax, m, delay, ju)
        nvalid = nmax - (m - 1) * delay
        assert sorted(ju[:nvalid]) == list(range((m - 1) * delay + 1, nmax + 1))
    ju = np.zeros(601, np.int64)
    nl._c1_centers(600, 1, 1, ju)
    first = np.sort(ju[:50])
    assert np.max(np.diff(np.r_[0, first, 601])) < 3 * 600 / 50  # no large gap among the first 50


def test_tisean_c1_mean_over_reference_points_used():
    # the mean log radius is taken over the reference points used (stock TISEAN divided by
    # nref - (m - 1) * tau, which biased the estimates low and made them drift with nref: here
    # 2.7, 3.5, 3.7). 500 points, so the requested numbers of reference points are used as given.
    x = np.random.default_rng(7).standard_normal(500)
    est = [nl.tisean_c1(x, 10, [2, 4], 0.02, nref)['maxmd'] for nref in (100, 300, 500)]
    assert max(est) / min(est) < 1.1 and min(est) > 3.5


def _henon(n):
    # x * x, not x ** 2: the map is chaotic, so the series must not depend on the platform's pow()
    a, b, x = 1.4, 0.3, [0.1, 0.1]
    for _ in range(n + 100):
        x.append(1 - a * (x[-1] * x[-1]) + b * x[-2])
    x = np.array(x[102:])
    return (x - x.mean()) / x.std(ddof=1)


def test_lyap_spec():
    # the TISEAN lyap_spec binary, run on the same noisy embedding (the noise is BF_Random's, seed 42)
    d = nl.lyap_spec(_henon(800), 1, 3, 30, 'full', ('ac', 1))
    assert abs(d['LE1'] - 0.4762983) < 1e-9 and abs(d['LE2'] - 0.2482943) < 1e-9
    assert abs(d['LE3'] + 1.678287) < 1e-9
    assert d['numPos'] == 2 and abs(d['sumPos'] - 0.7245926) < 1e-9
    assert abs(d['sumAll'] + 0.9536944) < 1e-9 and abs(d['KYdim'] - 2.431745) < 1e-6
    d = nl.lyap_spec(_x(600), 1, 3, 30, 'full', ('ac', 1))
    assert abs(d['LE1'] + 0.03536173) < 1e-9 and abs(d['LE3'] + 0.3993465) < 1e-9
    assert d['numPos'] == 0 and d['KYdim'] == 0
    d = nl.lyap_spec(_x(600), 2, 4, 20, 'full', ('ac', 1))
    assert abs(d['LE2'] + 0.09210157) < 1e-9
    assert _all_nan(nl.lyap_spec(_x(600)[:200], 1, 3, 30))  # too short for the local fits
    assert _all_nan(nl.lyap_spec(np.ones(500)))
    with pytest.raises(ValueError):
        nl.lyap_spec(_x(600), 1, 2)


def test_surrogate_test_nlpe_fnn(monkeypatch):
    from pyhctsa.operations import surrogates as su
    x = _x(300)
    rng = np.random.RandomState(3)
    z = np.column_stack([x[rng.permutation(x.size)] for _ in range(12)])
    monkeypatch.setattr(su, '_make_surrogates', lambda *a, **k: z)
    out = su.surrogate_test(x, 'RandPerm', 12, ['nlpe', 'fnn'])
    assert {k.split('_')[0] for k in out} == {'nlpe', 'fnn'}
    assert {k.split('_', 1)[1] for k in out} == {'zscore', 'mediqr', 'prank'}
    # the statistics, as hctsa takes them: mean squared prediction errors for the series and each surrogate
    fnn_x = nl.fnn(x, 1, 2, ('ac', 1), False, escape_factor=5)['pfnn_2']
    fnn_s = np.array([nl.fnn(z[:, i], 1, 2, ('ac', 1), False, escape_factor=5)['pfnn_2'] for i in range(12)])
    assert out['fnn_zscore'] == pytest.approx((fnn_x - fnn_s.mean()) / fnn_s.std(ddof=1))
    nlpe_s = np.array([np.mean(nl._ms_nlpe(z[:, i], 3, 1, int(nl.theiler_window(z[:, i], ('ac', 1), 300))) ** 2)
                       for i in range(12)])
    assert out['nlpe_zscore'] == pytest.approx(
        (nl.nlpe(x, 3, 1, 5000, ('ac', 1))['msqerr'] - nlpe_s.mean()) / nlpe_s.std(ddof=1))
    assert out['nlpe_zscore'] < 0  # the series is more predictable than its permutations


def test_gp_corr_sum_m1():
    # embedding dimension 1 (TISEAN's d2 reads past the end of the data for it; the box that does is not used)
    d = nl.gp_corr_sum(_x(600), -1, 0.1, ('ac', 1), 20, (1, 1))
    assert abs(d['robfit_a2'] - 0.979887300056) < 1e-9 and abs(d['meanlnCr'] + 4.46294185584) < 1e-9
    assert abs(d['minlnr'] + 5.40215637362) < 1e-9
    c2 = _tisean.d2(_x(600), delay=1, embed=1, theiler=6, howoften=20, epsmax=0.3)['c2']
    assert len(c2) == 1 and c2[0].shape == (20, 2)
    with pytest.raises(ValueError):
        _tisean.d2(_x(600), embed=0)
