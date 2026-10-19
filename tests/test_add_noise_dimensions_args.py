"""add_noise ``num_draws`` and dimensions ``thwin`` (mirrors hctsa CO_AddNoise numDraws, NL_Dimensions thwin)."""
import numpy as np
import pytest

from pyhctsa.operations.correlation import add_noise
from pyhctsa.operations.nonlinearity import dimensions
from pyhctsa.robust import bf_random


def _ar1(n=300, seed=0):
    e = np.random.default_rng(seed).standard_normal(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + e[i]
    return (y - y.mean()) / y.std(ddof=1)


def test_noise_stream_prefix():
    # draws at a level are consecutive blocks of one stream, so 5 draws are the first 5 of 10
    a = bf_random(300 * 10, 3, 'normal').reshape(10, 300)
    b = bf_random(300 * 5, 3, 'normal').reshape(5, 300)
    assert np.array_equal(a[:5], b)


def test_add_noise_num_draws():
    y = _ar1()
    kw = dict(tau=1, ami_method='gaussian')
    default, ten, five = add_noise(y, **kw), add_noise(y, **kw, num_draws=10), add_noise(y, **kw, num_draws=5)
    assert default == ten
    assert five != ten
    with pytest.raises(ValueError):
        add_noise(y, **kw, num_draws=0)


def test_dimensions_thwin():
    y = _ar1()
    kw = dict(embed_params=['ac1e', ['fnn', 0.05]])
    d0, dth = dimensions(y, 50, **kw, thwin=0), dimensions(y, 50, **kw)
    assert dth == dimensions(y, 50, **kw, thwin=['ac', 1])
    assert d0['bc_meanm1'] == dth['bc_meanm1']  # box counting does not use the window
    assert d0['co_meanm1'] != dth['co_meanm1']
