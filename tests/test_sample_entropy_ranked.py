"""sample_entropy_ranked: SampEn of the ranks, a pyhctsa-only variant of sample_entropy (EN_SampEn) that does not
respond to the value distribution."""
import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import rankdata

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.operations.entropy import _ranks_random_ties, sample_entropy, sample_entropy_ranked


def _zranks(y):
    r = rankdata(y)
    return (r - r.mean()) / np.std(r, ddof=1)


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k], (k, a[k], b[k])


def _auc(a, b):
    """P(a > b) over all pairs (ties count one half); NaNs dropped."""
    a, b = np.asarray(a)[np.isfinite(a)], np.asarray(b)[np.isfinite(b)]
    return (np.sum(a[:, None] > b[None]) + 0.5 * np.sum(a[:, None] == b[None])) / (a.size * b.size)


def _ar1(phi, n, rng):
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 200))[200:]


# ---- values ----

def test_equals_sample_entropy_of_the_zscored_ranks():
    y = np.random.default_rng(1).standard_normal(300)
    out = sample_entropy_ranked(y, m=3, r=0.2)
    ref = sample_entropy(_zranks(y), m=3, r=0.2)
    assert list(out) == ["sampen1", "sampen2", "sampen3"]
    for k in (1, 2, 3):
        assert out[f"sampen{k}"] == pytest.approx(ref[f"sampen{k}"], rel=1e-12)


def test_small_case_by_hand():
    # ranks of y are 1..6 in the order below; z-scored ranks are spaced 1/sqrt(3.5) = 0.53 apart, so with r = 0.6
    # two values match exactly when their ranks are adjacent (or equal)
    y = np.array([10.0, 30.0, 20.0, 40.0, 50.0, 60.0])   # ranks 1 3 2 4 5 6
    # length-1 templates (first N-1 = 5 values, ranks 1 3 2 4 5): adjacent-rank pairs (1,2) (3,2) (3,4) (4,5) -> B1 = 4
    # extended to length 2, (r_i, r_{i+1}) vs (r_j, r_{j+1}) for those pairs: (1,3)-(2,4) match, (3,2)-(2,4) no,
    # (3,2)-(4,5) no, (4,5)-(5,6) match -> A1 = 2
    out = sample_entropy_ranked(y, m=1, r=0.6)
    assert out["sampen1"] == pytest.approx(-np.log(2 / 4))


def test_iid_value_matches_theory():
    # for iid continuous data sampen_k -> -log(2d - d^2), d = r / sqrt(12)
    y = np.random.default_rng(2).standard_normal(5000)
    d = 0.2 / np.sqrt(12)
    out = sample_entropy_ranked(y, m=2, r=0.2)
    assert out["sampen1"] == pytest.approx(-np.log(2 * d - d * d), abs=0.03)
    assert out["sampen2"] == pytest.approx(-np.log(2 * d - d * d), abs=0.05)


def test_diff1_ranks_the_increments():
    y = np.cumsum(np.random.default_rng(3).standard_normal(400))
    _same(sample_entropy_ranked(y, m=2, r=0.2, pre_process_how="diff1"), sample_entropy_ranked(np.diff(y), m=2, r=0.2))


def test_default_tolerance_is_0p1():
    y = np.random.default_rng(4).standard_normal(500)
    _same(sample_entropy_ranked(y), sample_entropy_ranked(y, m=2, r=0.1))


# ---- invariance ----

@pytest.mark.parametrize("g", [lambda x: np.exp(3 * x), lambda x: x ** 3, np.arctan, lambda x: 1e6 * x - 7,
                               lambda x: -x, lambda x: -np.exp(x)])
def test_invariant_under_monotone_transforms(g):
    y = np.random.default_rng(5).standard_normal(800)
    _same(sample_entropy_ranked(y, m=5, r=0.2), sample_entropy_ranked(g(y), m=5, r=0.2))


def test_invariant_under_increasing_transforms_with_ties():
    y = np.random.default_rng(6).poisson(2.0, 800).astype(float)
    a = sample_entropy_ranked(y, m=3, r=0.2)
    _same(a, sample_entropy_ranked(np.sqrt(y) * 5 + 1, m=3, r=0.2))
    _same(a, sample_entropy_ranked(np.log1p(y), m=3, r=0.2))


def test_deterministic_with_ties():
    y = np.random.default_rng(7).integers(0, 3, 600).astype(float)
    _same(sample_entropy_ranked(y, m=4, r=0.2), sample_entropy_ranked(y.copy(), m=4, r=0.2))


def test_random_ties_give_a_permutation_independent_of_the_values():
    y = np.random.default_rng(8).integers(0, 4, 500).astype(float)
    r = _ranks_random_ties(y)
    assert sorted(r) == list(range(1, 501))
    assert np.all(np.diff(y[np.argsort(r)]) >= 0)      # still ordered by value
    np.testing.assert_array_equal(r, _ranks_random_ties(10 * y + 3))
    x = np.random.default_rng(8).standard_normal(50)
    np.testing.assert_array_equal(_ranks_random_ties(x), rankdata(x))   # no ties: plain ranks


# ---- degenerate input: NaN, not an error ----

@pytest.mark.parametrize("y", [[], [1.0], [1.0, 1.0, 1.0, 1.0], np.ones(50), [0.0, np.nan, 1.0, 2.0, 3.0],
                               [0.0, np.inf, 1.0, 2.0, 3.0]])
def test_degenerate_input_gives_nan(y):
    out = sample_entropy_ranked(np.asarray(y, float), m=3, r=0.2)
    assert list(out) == ["sampen1", "sampen2", "sampen3"] and all(np.isnan(v) for v in out.values())


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_short_input_gives_nan_or_finite_values(n):
    for how in (None, "diff1"):
        out = sample_entropy_ranked(np.random.default_rng(n).standard_normal(n), m=5, r=0.2, pre_process_how=how)
        assert list(out) == [f"sampen{k}" for k in range(1, 6)]
        assert all(np.isnan(v) or np.isfinite(v) for v in out.values())


def test_diff1_of_a_linear_trend_is_nan():
    out = sample_entropy_ranked(np.arange(100.0), m=2, r=0.2, pre_process_how="diff1")
    assert all(np.isnan(v) for v in out.values())


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        sample_entropy_ranked(np.arange(10.0), m=0)
    with pytest.raises(ValueError):
        sample_entropy_ranked(np.arange(10.0), pre_process_how="diff2")


# ---- specificity: the marginal does not register, temporal structure does ----

def _values(gen, reps, n, seed0, **kw):
    return np.array([sample_entropy_ranked(gen(np.random.default_rng(seed0 + i), n), m=5, r=0.2, **kw)["sampen2"]
                     for i in range(reps)])


def _values_orig(gen, reps, n, seed0):
    out = []
    for i in range(reps):
        y = gen(np.random.default_rng(seed0 + i), n)
        out.append(sample_entropy((y - y.mean()) / y.std(ddof=1), m=5, r=0.2)["sampen2"])
    return np.array(out)


def test_iid_marginals_are_indistinguishable():
    reps, n = 40, 1000
    gauss = lambda rng, n: rng.standard_normal(n)
    t3 = lambda rng, n: rng.standard_t(3, n)
    binary = lambda rng, n: (rng.uniform(size=n) < 0.3).astype(float)
    counts = lambda rng, n: rng.poisson(1.0, n).astype(float)
    g = _values(gauss, reps, n, 100)
    for other in (t3, binary, counts):
        assert 0.3 < _auc(g, _values(other, reps, n, 500)) < 0.7
    # the original SampEn separates the marginals completely
    assert _auc(_values_orig(gauss, reps, n, 100), _values_orig(t3, reps, n, 500)) > 0.95


def test_temporal_structure_still_registers():
    reps, n = 40, 1000
    g = _values(lambda rng, n: rng.standard_normal(n), reps, n, 100)
    ar = _values(lambda rng, n: _ar1(0.5, n, rng), reps, n, 900)
    ar_exp = _values(lambda rng, n: np.exp(_ar1(0.5, n, rng)), reps, n, 900)
    np.testing.assert_array_equal(ar, ar_exp)            # monotone transform of the same AR(1) realisations
    assert _auc(g, ar) > 0.95                            # AR(1) reads as more regular than iid noise

    def logistic(rng, n):
        x = np.empty(n)
        x[0] = rng.uniform(0.1, 0.9)
        for i in range(1, n):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        return x + 0.05 * rng.standard_normal(n)
    assert _auc(g, _values(logistic, 10, n, 1300)) == 1.0


# ---- feature set ----

def test_registered_in_the_default_feature_set():
    labels = [k for k in FeatureCalculator().feature_funcs if k.startswith("sample_entropy_ranked")]
    assert labels == ["sample_entropy_ranked_5_0p2", "sample_entropy_ranked_5_0p2_diff1",
                      "sample_entropy_ranked_5_0p2_dec"]
