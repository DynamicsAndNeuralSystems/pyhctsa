"""Edge-case behavior of moment_corr, modwt_var, sample_entropy, randomize_ordinal and moments."""
import numpy as np
import pytest
from scipy import stats

from pyhctsa.operations import distribution as ds
from pyhctsa.operations import entropy as en
from pyhctsa.operations import stationarity as st
from pyhctsa.operations import wavelet as wl
from pyhctsa.utils import frame_signal


def test_moment_corr_uses_only_full_windows():
    # N = 208: window 5, overlap 1, step 4; the last frame (samples 205..209) runs past the end
    x = np.random.default_rng(1).standard_normal(208)
    w, p, h = 5, 1, 4
    nfull = (208 - w) // h + 1
    assert frame_signal(x, w, p).shape[1] == nfull + 1
    xb = frame_signal(x, w, p)[:, :nfull]
    assert np.all(xb != 0)
    m1, m2 = xb.mean(axis=0), xb.std(axis=0, ddof=1)
    out = st.moment_corr(x)
    assert out['R'] == pytest.approx(np.corrcoef(m1, m2)[0, 1], abs=1e-12)
    assert out['density'] == pytest.approx(nfull / (np.ptp(m1) * np.ptp(m2)), rel=1e-12)


def test_moment_corr_unchanged_when_last_window_already_dropped():
    # N = 1000: window 20, overlap 4; 62 full windows
    x = np.random.default_rng(2).standard_normal(1000)
    xb = frame_signal(x, 20, 4)[:, :62]
    out = st.moment_corr(x)
    assert out['R'] == pytest.approx(np.corrcoef(xb.mean(axis=0), xb.std(axis=0, ddof=1))[0, 1], abs=1e-12)


@pytest.mark.parametrize('N', [40, 100, 155, 156, 400])
def test_modwt_var_always_has_a_scaling_band(N):
    y = np.random.default_rng(3).standard_normal(N)
    L = 6  # db3 filter length
    level = min(5, int(np.floor(np.log2((N - 1) / (L - 1) + 1))))
    w = wl._modwt(y, 'db3', level)
    wvar = wl._modwt_var(w, 'db3')
    assert wvar.size == level + 1
    out = wl.modwt_var(y, 'db3', 5)
    assert out['scalingFrac'] == pytest.approx(wvar[-1] / wvar.sum())


def test_sample_entropy_nan_when_none_at_next_length():
    # A = 0 matches at length m + 1, B > 0 at length m: the estimate is infinite, so NaN
    z = np.random.default_rng(1).standard_normal(200)
    z = (z - z.mean()) / z.std(ddof=1)
    r = 0.1 * np.std(z, ddof=1)
    out = en.sample_entropy(z, 5, r)
    _, a, b = en._sampen_c.calculate(z, 6, r, True)
    assert a[3] == 0 and b[3] == 2  # B = 2 pairs match on 3 values, none on 4
    assert np.isfinite(out['sampen2']) and np.isnan(out['sampen3'])
    assert np.isnan(out['sampen4'])  # no matches at length 4 (B = 0): NaN
    assert np.isnan(out['meanchsampen'])


def test_randomize_ordinal_guard_is_at_machine_precision(monkeypatch):
    z = np.random.default_rng(1).standard_normal(200)
    S = en._ro_pattern_dists(z, 1)
    orig = en._ro_norm_entropy

    def shifted(gap):
        def f(P):
            P = np.atleast_2d(P)
            if P.shape == S[7:8].shape and np.array_equal(P, S[7:8]):
                return orig(S[0:1]) - gap
            return orig(P)
        return f

    monkeypatch.setattr(en, '_ro_norm_entropy', shifted(1e-13))
    assert np.isfinite(en.randomize_ordinal(z, 1)['halftime'])
    monkeypatch.setattr(en, '_ro_norm_entropy', shifted(0.0))
    assert np.isnan(en.randomize_ordinal(z, 1)['halftime'])


def test_moments_are_skewness_and_kurtosis():
    y = np.random.default_rng(1).standard_normal(50) ** 2
    assert ds.moments(y, 3) == pytest.approx(stats.skew(y), rel=1e-12)
    assert ds.moments(y, 4) == pytest.approx(stats.kurtosis(y, fisher=False), rel=1e-12)
