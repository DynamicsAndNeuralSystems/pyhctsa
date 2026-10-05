"""Wave-6 entropy ports (expected values generated with MATLAB hctsa)."""
import numpy as np

from pyhctsa.operations.entropy import (
    bubble_entropy, fuzzy_entropy, permutation_entropy, permutation_entropy_complexity,
    randomize, wavelet_entropy)


def _x():
    n = np.arange(1, 301)
    x = np.sin(0.3 * n) + 0.5 * np.cos(1.7 * n) ** 3 + 0.2 * np.sin(0.011 * n ** 1.5)
    return (x - x.mean()) / x.std(ddof=1)


def test_fuzzy_entropy():
    d = fuzzy_entropy(_x(), 2, 0.2, 2)
    assert abs(d['fuzzyEn1'] - 1.18052465263) < 1e-9
    assert abs(d['fuzzyEn2'] - 1.47653378779) < 1e-9
    assert set(fuzzy_entropy(_x(), 3)) == {'fuzzyEn1', 'fuzzyEn2', 'fuzzyEn3'}
    nan = fuzzy_entropy(np.ones(100), 2)
    assert all(np.isnan(v) for v in nan.values())
    assert all(np.isnan(v) for v in fuzzy_entropy(_x()[:4], 2).values())


def test_bubble_entropy():
    assert abs(bubble_entropy(_x(), 5, 1)['bubbleEn'] - 1.03935885829) < 1e-9
    assert abs(bubble_entropy(_x(), 10, 'mi')['bubbleEn'] - 1.9567034885) < 1e-9
    assert np.isnan(bubble_entropy(np.ones(200))['bubbleEn'])
    assert np.isnan(bubble_entropy(_x()[:19], 10, 1)['bubbleEn'])


def test_permutation_entropy_complexity():
    d = permutation_entropy_complexity(_x(), 4, 1)
    assert abs(d['hNorm'] - 0.882883144624) < 1e-9
    assert abs(d['jsComplexity'] - 0.14018802479) < 1e-9
    d = permutation_entropy_complexity(_x(), 5, 'ac1e')
    assert abs(d['hNorm'] - 0.620720444885) < 1e-9
    assert abs(d['jsComplexity'] - 0.402123257975) < 1e-9
    assert abs(permutation_entropy(_x(), 4, 1)['normPermEn'] - 0.882883144624) < 1e-9
    d = permutation_entropy_complexity(np.ones(100), 4, 'ac1e')
    assert np.isnan(d['hNorm']) and np.isnan(d['jsComplexity'])
    assert all(np.isnan(v) for v in permutation_entropy(np.ones(100), 3, 'ac1e').values())


def test_wavelet_entropy():
    assert abs(wavelet_entropy(_x(), 'sym4', 5) - 0.556815846816) < 1e-9
    assert np.isnan(wavelet_entropy(_x()[:20], 'sym4', 5))  # level > floor(log2(N))
    assert np.isnan(wavelet_entropy(np.zeros(100)))


def test_randomize():
    # the random draws replicate MATLAB's (rng(0), randi), so the values agree
    d = randomize(_x(), 'statdist', 'default')
    assert abs(d['ac1diff'] - 0.707323602716) < 1e-9 and d['ac1hp'] == 4
    assert abs(d['xc1diff'] - 0.613406312685) < 1e-9
    assert abs(d['permen3_1diff'] - 0.0166988313631) < 1e-9 and d['statav5hp'] == 6
    assert abs(d['d1fexpc'] - 0.069342454) < 1e-6
    d = randomize(_x(), 'permute')
    assert abs(d['ac1diff'] - 0.831449012634) < 1e-9 and d['ac1hp'] == 3
    assert abs(d['xc1diff'] - 0.764558845981) < 1e-9
    assert len(d) == 64
