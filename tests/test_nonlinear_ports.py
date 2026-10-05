"""Nonlinear ports: false nearest neighbors and the embedding dimension, TISEAN c1 and lyap_spec,
correlation sums at m = 1 (expected values generated with MATLAB hctsa and the TISEAN binaries)."""
import numpy as np
import pytest

from pyhctsa.operations import nonlinearity as nl


def _x(n=400):
    t = np.arange(1, n + 1)
    x = (np.sin(0.3 * t) + 0.5 * np.cos(1.7 * t) ** 3 + 0.2 * np.sin(0.011 * t ** 1.5)
         + 0.4 * np.sin(0.77 * t ** 1.1))
    return (x - x.mean()) / x.std(ddof=1)


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
    assert np.isnan(nl.nlpe(np.ones(100), 'fnn'))


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
