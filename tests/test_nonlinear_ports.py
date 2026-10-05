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
    assert nl._embedding_params(_x(), 'ac1e', 'fnn') == (4, 3)
    assert nl._embedding_params(_x(), 'ac1e', ['fnn', 0.05]) == (4, 5)
    assert nl._embed_tau_m(_x(), ['mi', 'fnn']) == (3, 3)
    assert nl._embedding_params(_x(), 2, 6) == (2, 6)
    assert nl._embedding_params(np.ones(100), 'ac1e', 'fnn') is None
    with pytest.raises(ValueError):
        nl._embedding_params(_x(), 1, 'nonsense')


def test_fnn_users():
    x = _x()
    assert abs(nl.nlpe(x, 'fnn', 'mi', 5000, ('ac', 1))['msqerr'] - 0.236829945826) < 1e-9
    d = nl.local_density(x, 5, ('ac', 1), 'ac1e', 'fnn')
    assert abs(d['meanden'] + 2.76300616221) < 1e-9 and abs(d['stdden'] - 0.902477556828) < 1e-9
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
    assert nl.fnn(_x(), 'ac', escape_factor=5, just_best=True, bestp=0.05) == 5
    assert np.isnan(nl.fnn(_x()[:8])) and np.isnan(nl.fnn(np.ones(100)))


def _x600():
    return _x(600)


def test_tisean_c1():
    x = _x600()
    # the c1 + c2d -a2 output of the TISEAN binary (-d1 -m1 -M5 -t12 -n300): [length scale, slope]
    curves = nl._c2d_slopes(nl._c1_curves(x, 1, 1, 5, 12, 300))
    assert [c.shape[0] for c in curves] == [14] * 5
    for blk, first, last in [(0, (0.00588210579, 1.00610375), (0.876543283, 0.908076644)),
                             (2, (0.201452777, 2.90096092), (1.58953869, 1.4778477)),
                             (4, (0.342286617, 3.37284517), (1.94325423, 1.78551936))]:
        np.testing.assert_allclose(curves[blk][0], first, rtol=1e-8)
        np.testing.assert_allclose(curves[blk][-1], last, rtol=1e-8)
    # hctsa's NL_c1 (MATLAB)
    d = nl.tisean_c1(x, 1, [1, 5], 0.02, 0.5)
    assert abs(d['bestestd'] - 0.990854776154) < 1e-8 and abs(d['bestestdstd'] - 0.0251984141023) < 1e-8
    assert abs(d['bestgoodness'] + 0.0398015858977) < 1e-8 and abs(d['mediand'] - 2.723457766) < 1e-8
    assert abs(d['maxd'] - 3.5391226425) < 1e-8 and abs(d['meanstd'] - 0.0496644940833) < 1e-8
    assert abs(d['longestscr'] - 4.61435478461) < 1e-8
    d = nl.tisean_c1(x, 'ac', [2, 4], 10, 150)
    assert abs(d['bestestd'] - 1.92153167667) < 1e-8 and abs(d['ranged'] - 0.896392209048) < 1e-8
    assert abs(d['longestscr'] - 1.70608968983) < 1e-8
    # a length with remainder <= 6 on division by 128 loses its last point, as in hctsa
    d = nl.tisean_c1(x[:512], 1, [1, 5], 0.02, 0.5)
    assert abs(d['bestestd'] - 0.983567228923) < 1e-8 and abs(d['longestscr'] - 4.72561510696) < 1e-8
    assert d == nl.tisean_c1(x[:511], 1, [1, 5], 0.02, 0.5)
    assert _all_nan(nl.tisean_c1(x[:99])) and _all_nan(nl.tisean_c1(np.ones(300)))
    # TISEAN's c1 never finishes here (too few neighbors outside the Theiler window): hctsa gives up
    assert _all_nan(nl.tisean_c1(x[:520], 2, [3, 6], 0.05, 0.3))
    with pytest.raises(ValueError):
        nl.tisean_c1(x, 'nonsense')


def _henon(n):
    a, b, x = 1.4, 0.3, [0.1, 0.1]
    for _ in range(n + 100):
        x.append(1 - a * x[-1] ** 2 + b * x[-2])
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
