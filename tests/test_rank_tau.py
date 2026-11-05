"""Rank-based delay rules ('ac-rank', 'ac1e-rank', 'mi-rank'): the delay, and the order-based features
registered with these rules, are unchanged by strictly increasing transformations of the series."""
import numpy as np
import pytest
import yaml
from scipy.stats import rankdata

from pyhctsa.calculator import FeatureCalculator
from pyhctsa.utils import _average_ranks, _is_rank_rule, _tau_mi, get_tau, z_score
from conftest import HCTSA_YAML

RANK_CALLS = {
    'permutation_entropy_2_ac1e-rank', 'permutation_entropy_3_ac1e-rank',
    'permutation_entropy_4_ac1e-rank', 'permutation_entropy_5_ac1e-rank',
    'permutation_entropy_complexity_4_ac1e-rank', 'permutation_entropy_complexity_5_ac1e-rank',
    'bubble_entropy_10_mi-rank', 'ordinal_partition_network_ac1e-rank',
    'motif_two_diff_ac1e-rank', 'motif_two_median_ac1e-rank', 'motif_three_quantile_ac1e-rank',
    'transition_matrix_quantile_2_mi-rank', 'transition_matrix_quantile_3_mi-rank',
    'transition_matrix_quantile_4_mi-rank', 'transition_matrix_quantile_5_mi-rank',
    'transition_matrix_updown_2_ac-rank', 'transition_p_alphabet_2_to_20_ac1e-rank',
}
NOT_ORDINAL = {'permutation_entropy_5_ac1e-rank.normWPE'}  # weighted by amplitude by design


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def _skewed_remap(x, seed=0):
    """A strictly increasing map of the values of x onto a sorted gamma(1.5) sample (ties kept)."""
    u, inv = np.unique(x, return_inverse=True)
    g = np.sort(np.random.default_rng(seed).gamma(1.5, size=u.size))
    assert np.all(np.diff(g) > 0)
    return g[inv]


def _versions(x):
    return [x, np.exp(x), x ** 3, _skewed_remap(x)]


SERIES = {
    'ar1': _ar1(0.9, 1000, 1),
    'ar1_slow': _ar1(0.97, 1000, 2),
    'ar1_ties': np.round(2 * _ar1(0.9, 1000, 3)) / 2,  # many tied values, half-unit steps
}


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("rule", ['ac-rank', 'ac1e-rank', 'mi-rank'])
def test_rank_rules_are_invariant(name, rule):
    taus = [get_tau(v, rule) for v in _versions(SERIES[name])]
    assert taus[0] > 1 and taus.count(taus[0]) == len(taus)


@pytest.mark.parametrize("name", SERIES)
def test_rank_rules_apply_the_base_rule_to_average_ranks(name):
    y = SERIES[name]
    r = rankdata(y)
    assert np.array_equal(_average_ranks(y), r)
    assert get_tau(y, 'ac-rank') == get_tau(r, 'ac')
    assert get_tau(y, 'ac1e-rank') == get_tau(r, 'ac1e')
    assert get_tau(y, 'mi-rank') == _tau_mi(r, ranks=r)


def test_average_ranks_tie_only_equal_values():
    y = np.array([0.3, 1.0, 1.0, 1.0 + 1e-12, -2.0, 0.3])
    assert np.array_equal(_average_ranks(y), [2.5, 4.5, 4.5, 6.0, 1.0, 2.5])


def test_rank_rules_flagged():
    assert all(_is_rank_rule(r) for r in ('ac-rank', 'ac1e-rank', 'mi-rank'))
    assert not any(_is_rank_rule(r) for r in ('ac', 'ac1e', 'mi', 'mi-gaussian', 3, 'bogus-rank'))
    with pytest.raises(ValueError):
        get_tau(SERIES['ar1'], 'mi-gaussian-rank')


def test_constant_series_gives_nan():
    for rule in ('ac-rank', 'ac1e-rank', 'mi-rank'):
        assert np.isnan(get_tau(np.ones(100), rule))


def test_linear_rules_still_depend_on_the_values():
    # the Pearson rules are unchanged: a monotone transform can still move them
    x = SERIES['ar1_slow']
    assert get_tau(z_score(x), 'ac1e') != get_tau(z_score(np.exp(2 * x)), 'ac1e')


def _rank_rule_config(tmp_path):
    """The configs of hctsa.yaml that set their delay by a rank-based rule, as a config file."""
    cfg = yaml.safe_load(HCTSA_YAML.read_text(encoding="utf-8"))
    sub = {}
    for module, ops in cfg.items():
        for op, meta in ops.items():
            keep = [c for c in meta.get("configs") or [] if _is_rank_rule(c.get("tau"))]
            if keep:
                sub.setdefault(module, {})[op] = {**meta, "configs": keep}
    path = tmp_path / "rank_rules.yaml"
    path.write_text(yaml.safe_dump(sub), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def rank_calculator(tmp_path_factory):
    return FeatureCalculator(_rank_rule_config(tmp_path_factory.mktemp("rank")))


def test_rank_rules_registered_for_the_order_based_calls(rank_calculator):
    assert set(rank_calculator.feature_funcs) == RANK_CALLS


@pytest.mark.parametrize("name", SERIES)
def test_registered_calls_are_rank_invariant(rank_calculator, name):
    df = rank_calculator.extract(_versions(SERIES[name]))
    cols = [c for c in df.columns if c not in NOT_ORDINAL]
    assert {c.split('.')[0] for c in cols} == RANK_CALLS
    ref = df[cols].iloc[0].to_numpy()
    assert np.isfinite(ref).mean() > 0.9
    for i in range(1, len(df)):
        v = df[cols].iloc[i].to_numpy()
        same = (np.isnan(ref) & np.isnan(v)) | (np.abs(v - ref) <= 1e-12)
        assert same.all(), [c for c, s in zip(cols, same) if not s]
