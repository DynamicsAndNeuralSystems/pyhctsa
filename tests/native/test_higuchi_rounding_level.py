"""higuchi_fd leaves out scales whose curve length is zero up to rounding.

For a series periodic with period p, every subsequence at a scale k divisible by p is constant,
so L(k) = 0 and that scale is dropped. Before this rule only an exact zero was dropped: a length
at the rounding level (a computed periodic signal, or the exactly periodic series plus 1e-12
noise) entered the fit with log L(k) around -27 to -37, set by rounding noise alone.
"""
import warnings

import numpy as np
import pytest

from pyhctsa.operations.scaling import higuchi_fd
from pyhctsa.utils import z_score

KEYS = ("HFD", "intercept", "se_HFD", "ssr", "resac1")


def _hfd(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return higuchi_fd(y, 8)


def _period6():
    return z_score(np.tile([0.1, 0.7, -0.3, 0.9, -1.2, 0.4], 100))


@pytest.mark.parametrize("seed", range(3))
def test_exactly_periodic_series_is_insensitive_to_rounding_noise(seed):
    y = _period6()
    out = _hfd(y)
    outp = _hfd(y + 1e-12 * np.random.default_rng(seed).standard_normal(y.size))
    for k in KEYS:
        assert outp[k] == pytest.approx(out[k], abs=1e-8), k


def test_rounding_level_length_is_treated_as_zero():
    """A computed period-6 sinusoid (L(6) ~ 1e-16, not 0) gives the exactly periodic result."""
    t = np.arange(600)
    y = z_score(np.sin(2 * np.pi * t / 6) + 0.5 * np.cos(2 * np.pi * t / 3))
    exact = np.tile(y[:6], 100)     # exactly periodic: L(6) = 0
    out, out_exact = _hfd(y), _hfd(exact)
    for k in ("HFD", "intercept", "se_HFD", "ssr"):
        assert out[k] == pytest.approx(out_exact[k], rel=1e-9, abs=1e-12), k


def test_too_few_nondegenerate_scales_give_nan_with_or_without_noise():
    """Period 2: L(2), L(4), L(6), L(8) are zero, leaving four scales."""
    y = z_score(np.tile([1.0, -1.0], 300))
    yp = y + 1e-12 * np.random.default_rng(0).standard_normal(y.size)
    assert np.isnan(_hfd(y)["HFD"]) and np.isnan(_hfd(yp)["HFD"])


def test_ordinary_series_is_unaffected():
    """The rule only removes lengths at the rounding level: a random walk keeps all scales."""
    rng = np.random.default_rng(5)
    y = z_score(np.cumsum(rng.standard_normal(1000)))
    out = _hfd(y)
    assert out["HFD"] == pytest.approx(1.5, abs=0.1)
