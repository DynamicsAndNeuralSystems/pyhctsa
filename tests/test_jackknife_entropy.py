"""Delete-one jackknife of plug-in entropies of pattern counts."""
from math import factorial

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import xlogy

from pyhctsa.operations.entropy import (dispersion_entropy, permutation_entropy,
                                        permutation_entropy_complexity, shannon_entropy)
from pyhctsa.utils import _jackknife_counts, _jackknife_entropy


def _h(p):
    return -np.sum(xlogy(p, p))


def _brute(counts, f):
    """Jackknife by explicit leave-one-out over every observation."""
    counts = np.asarray(counts, float)
    m = counts.sum()
    labels = np.repeat(np.arange(counts.size), counts.astype(int))
    acc = 0.0
    for i in range(m.astype(int)):
        loo = counts.copy()
        loo[labels[i]] -= 1
        acc += f(loo / (m - 1))
    return m * f(counts / m) - (m - 1) / m * acc


@pytest.mark.parametrize("counts", [[3, 1, 0, 5], [2, 2], [10, 0, 0, 1, 1, 4], [1, 1, 1, 1, 1]])
def test_helper_matches_bruteforce(counts):
    ref = _brute(counts, _h)
    assert _jackknife_counts(counts, _h) == pytest.approx(ref, abs=1e-12)
    assert _jackknife_entropy(counts) == pytest.approx(ref, abs=1e-12)


def test_helper_generic_function():
    f = lambda p: float(np.sum(p ** 2))
    counts = [4, 0, 2, 7]
    assert _jackknife_counts(counts, f) == pytest.approx(_brute(counts, f), abs=1e-12)


@pytest.mark.parametrize("m", [4, 5])
def test_permen_unbiased_on_noise(m):
    rng = np.random.default_rng(0)
    series = [rng.standard_normal(200) for _ in range(30)]
    vals = np.mean([permutation_entropy(y, m, 1)['permEn'] for y in series])
    truth = np.log2(factorial(m))
    # plug-in bias is about (m! - 1) / (2 M ln 2) bits: 0.17 bits at m = 4, 0.62 bits at m = 5
    plug_bias = (factorial(m) - 1) / (2 * (200 - m + 1) * np.log(2))
    assert abs(vals - truth) < max(0.15 * plug_bias, 0.05)
    assert abs(vals - truth) < 0.4 * plug_bias


def test_rank_invariance():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(300)
    y2 = np.exp(2 * y) + y ** 3
    a, b = permutation_entropy(y, 4, 1), permutation_entropy(y2, 4, 1)
    for k in ('permEn', 'normPermEn', 'permEnLE', 'ordAsym'):
        assert a[k] == b[k]
    a, b = permutation_entropy_complexity(y, 4, 1), permutation_entropy_complexity(y2, 4, 1)
    assert a == b
    assert shannon_entropy(y, 5, 2) == shannon_entropy(y2, 5, 2)


def test_complexity_runs_and_near_zero_on_noise():
    rng = np.random.default_rng(2)
    for m in (4, 5):
        c = [permutation_entropy_complexity(rng.standard_normal(400), m, 1) for _ in range(20)]
        js = np.array([d['jsComplexity'] for d in c])
        h = np.array([d['hNorm'] for d in c])
        assert np.all(np.isfinite(js))
        assert abs(js.mean()) < 0.03
        assert abs(h.mean() - 1) < 0.01
    # hNorm equals normPermEn
    y = rng.standard_normal(300)
    assert permutation_entropy_complexity(y, 4, 1)['hNorm'] == pytest.approx(
        permutation_entropy(y, 4, 1)['normPermEn'])


def test_dispersion_and_shannon_unbiased_on_noise():
    rng = np.random.default_rng(3)
    d = np.mean([dispersion_entropy(rng.standard_normal(200), 3, 3, 1)['normDispEn'] for _ in range(30)])
    assert abs(d - 1) < 0.02
    s = np.mean([shannon_entropy(rng.standard_normal(200), 5, 2) for _ in range(30)])
    assert abs(s - np.log(5)) < 0.03


def test_dropped_config():
    configs = Path(__file__).resolve().parents[1] / 'pyhctsa' / 'configurations'
    loader = type('L', (yaml.SafeLoader,), {})
    loader.add_constructor('!range', lambda ld, node: None)
    entry = yaml.load((configs / 'hctsa.yaml').read_text(), Loader=loader)['entropy']['shannon_entropy']
    assert not any(c.get('depth') == 3 and isinstance(c.get('num_bins'), list) for c in entry['configs'])
    mapping = yaml.safe_load((configs / 'hctsa_mapping.yaml').read_text())['entropy']['shannon_entropy']
    assert 'EN_Shannon_2t10_3' in mapping['dropped_hctsa_calls']
