"""Per-feature minimum series lengths (``min_length`` in hctsa.yaml) and the global length check.

Below its ``min_length`` a feature is not defined (hctsa's 'not appropriate', quality code 2): all of
its outputs are NaN and the set of output fields is unchanged. At ``min_length`` it is computed.
"""
import warnings

import numpy as np
import pytest
import yaml

from pyhctsa.calculator import FeatureCalculator, _eval_min_length
from pyhctsa.quality import Quality
from pyhctsa.utils import MIN_SERIES_LENGTH, _validate_data, decimate_ac1e, z_score

FC = FeatureCalculator()
BOUNDED = sorted(lab for lab, ml in FC.min_lengths.items() if ml is not None and ml > MIN_SERIES_LENGTH)


def _series(kind: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    if kind == "noise":
        return rng.standard_normal(n)
    e = rng.standard_normal(n + 200)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = 0.9 * x[t - 1] + e[t]
    return x[200:]


def _all_nan(v) -> bool:
    if isinstance(v, dict):
        return all(np.ndim(u) == 0 and np.isnan(float(np.real(u))) for u in v.values())
    return np.ndim(v) == 0 and np.isnan(float(np.real(v)))


# ---- global validation ----

def test_validate_data_rejects_fewer_than_three_samples():
    assert MIN_SERIES_LENGTH == 3
    assert not _validate_data(np.array([0.0, 1.0]))
    assert _validate_data(np.array([0.0, 1.0, 3.0]))
    assert not _validate_data(np.ones(10))
    assert not _validate_data(np.array([0.0, np.nan, 1.0]))
    assert not _validate_data(np.array([0.0, np.inf, 1.0]))


def test_extract_rejects_two_samples_and_warns_below_twenty():
    with pytest.raises(ValueError):
        FC.extract(np.array([0.0, 1.0]))
    sub = FeatureCalculator.__new__(FeatureCalculator)  # a calculator with two features only
    sub.feature_funcs = {k: FC.feature_funcs[k] for k in ("ac_1", "ac_40")}
    with pytest.warns(UserWarning, match="as short as"):
        df = sub.extract(_series("noise", 10))
    assert np.isfinite(df["ac_1"].iloc[0]) and np.isnan(df["ac_40"].iloc[0])
    # below its min_length a feature is NaN, coded 'undefined / not appropriate' (not an error)
    assert sub.quality["ac_40"].iloc[0] == Quality.NAN and sub.quality["ac_1"].iloc[0] == Quality.GOOD
    # a 2-sample series among valid ones is rejected (coded REJECTED), not computed
    with pytest.warns(UserWarning):
        sub.extract([_series("noise", 10), np.array([0.0, 1.0])], labels=["ok", "two"])
    assert (sub.quality.loc["two"] == Quality.REJECTED).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub.extract(_series("noise", 50))  # no warning at N >= 20


# ---- the min_length specifications ----

def test_min_lengths_cover_every_feature():
    assert set(FC.min_lengths) == set(FC.feature_funcs)
    for lab, ml in FC.min_lengths.items():
        assert ml is None or (isinstance(ml, int) and ml >= 0), lab


def test_min_length_examples():
    ml = FC.min_lengths
    assert ml["glscf_1_1_3"] == 6  # tau + 3 (3 pairs: 2 pairs only give +-1)
    assert ml["withinp_2_mean"] == 10  # two points beyond 2 sd need N - 1 > 2 p^2
    assert ml["variance_ratio_test_4_0"] == 9  # N > 2q
    assert ml["pol_var_1_6"] == 13  # two runs of D = 6 symbols: 2D + 1
    assert ml["surprise_dist_20_2_diff_500"] == 23  # memory + 1 (+1 'dist' prior, +1 'diff')
    assert ml["compare_test_sets_ss_2_uniform"] == 260  # 25 segments at most 50% overlapping
    assert ml["add_noise_1_kraskov1_4"] == 7  # KSG (k = 4) needs k + 2 = 6 pairs at lag 1
    # one bound per operation: where the code's own guard is at least as strict, no YAML bound
    assert ml["ac_40"] is None and ml["permutation_entropy_3_2"] is None


def test_eval_min_length():
    def f(y, tau=1, m=2):
        return 0.0
    assert _eval_min_length(None, f, {}, "f") is None
    assert _eval_min_length(7, f, {}, "f") == 7
    assert _eval_min_length("(m - 1) * num(tau, 1) + 2", f, {"tau": 3}, "f") == 5  # m from the default
    assert _eval_min_length("(m - 1) * num(tau, 1) + 2", f, {"tau": "ac1e", "m": 3}, "f") == 4
    assert _eval_min_length("None if m == 2 else 5", f, {}, "f") is None
    assert _eval_min_length("min(n for n in range(2, 99) if n * m > 9)", f, {}, "f") == 5
    with pytest.raises(ValueError):
        _eval_min_length("tau + 0.5", f, {}, "f")
    with pytest.raises(ValueError):
        _eval_min_length("undefined_name + 1", f, {}, "f")


def test_custom_config_min_length(tmp_path):
    cfg = {"correlation": {"autocorr": {"base_name": "ac", "min_length": "tau + 1", "ordered_args": ["tau"],
                                        "configs": [{"tau": 5, "zscore": True}, {"tau": 1, "zscore": True}]}}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    assert fc.min_lengths == {"ac_5": 6, "ac_1": 2}
    x = _series("noise", 5)
    assert np.isnan(fc.feature_funcs["ac_5"](x)) and np.isfinite(fc.feature_funcs["ac_1"](x))
    # applies to the series the function receives: here the decimated series is shorter
    cfg["correlation"]["autocorr"]["configs"] = [{"tau": 5, "zscore": True, "preprocess": "decimate_ac1e"}]
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    x = np.sin(2 * np.pi * np.arange(20) / 60)  # 1/e time of several samples
    n_dec = len(decimate_ac1e(z_score(x)))
    assert 6 <= len(x) and n_dec < 6
    assert np.isnan(fc.feature_funcs["ac_5_dec"](x))


# ---- each bounded feature: all NaN just below its bound, computed at it ----

@pytest.mark.parametrize("label", BOUNDED)
def test_all_nan_below_min_length(label):
    ml = FC.min_lengths[label]
    for kind in ("noise", "ar1"):
        assert _all_nan(FC.feature_funcs[label](_series(kind, ml - 1))), (label, kind)


@pytest.mark.parametrize("label", BOUNDED)
def test_runs_at_min_length(label):
    ml = FC.min_lengths[label]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("noise", "ar1"):
            FC.feature_funcs[label](_series(kind, ml))  # must not raise


def test_below_min_length_keeps_the_output_fields():
    lab = "motif_three_quantile"  # dict-valued, min_length 5
    ml = FC.min_lengths[lab]
    short = FC.feature_funcs[lab](_series("noise", ml - 1))
    long_ = FC.feature_funcs[lab](_series("noise", 200))
    assert isinstance(short, dict) and set(short) == set(long_)


# ---- fixes that came with the bounds ----

def test_capped_delay_is_at_least_one():
    from pyhctsa.operations.correlation import _resolve_delay
    x = _series("ar1", 8)
    assert _resolve_delay(x, "tau", cap=True) == 1  # hctsa's floor(N/10) would be 0


def test_lag_longer_than_series_gives_nan_not_an_error():
    from pyhctsa.operations.correlation import glscf, trev
    x = _series("noise", 4)
    assert np.isnan(glscf(x, 1, 1, 5))
    assert _all_nan(trev(x, 6))


# ---- whole feature set on short series ----

def test_short_series_keep_the_column_set_and_raise_nothing():
    x = _series("noise", 300)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols = list(FC.extract(x).columns)
        for n in (3, 8, 15):
            df = FC.extract(x[:n])
            assert list(df.columns) == cols, n
            assert FC.errors.empty, (n, FC.errors.feature.tolist())
            assert not (FC.quality.to_numpy() == Quality.ERROR).any(), n


def test_short_series_fixes_return_nan_not_errors():
    from pyhctsa.operations.symbolic import motif_three
    from pyhctsa.operations.model_fit import loop_local_simple, _arch_test_pvalues, _lbq_test_pvalues
    from pyhctsa.toolboxes.RAD.rad import rad_core
    x = _series("noise", 3)
    out = motif_three(x, 'quantile')  # 3 symbols: no word of length 4
    assert np.isnan(out["hhhh"]) and np.isfinite(out["hh"])
    out = loop_local_simple(_series("noise", 8))  # training windows up to 10 > N
    assert np.isnan(out["stde_peakpos"])
    assert np.isnan(rad_core(np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.5]), 1))  # tied sub-median pair
    assert np.isnan(_arch_test_pvalues(_series("noise", 10))).all()
    assert np.isnan(_lbq_test_pvalues(_series("noise", 10))).all()


def test_local_extrema_keeps_a_full_last_window_ending_in_zero():
    from pyhctsa.operations.stationarity import local_extrema
    x = _series("noise", 100)
    x[-1] = 0.0  # 100 = 2 full windows of 50: nothing is padded
    y = x.copy(); y[-1] = 1e-9
    a, b = local_extrema(x, 'l', 50), local_extrema(y, 'l', 50)
    assert a["meanmax"] == pytest.approx(b["meanmax"], rel=1e-6)  # both use 2 windows


# ---- per-field minimum lengths (field_min_length) ----

def test_field_min_length_blocks_single_fields():
    f = FC.feature_funcs["range_evolve"]
    out = f(_series("noise", 100))  # p1 is the range of 1 sample (always 0) until N = 101
    assert np.isnan(out["p1"]) and np.isfinite(out["p10"]) and np.isnan(out["l100"]) and np.isfinite(out["l50"])
    out = f(_series("noise", 101))
    assert np.isfinite(out["p1"]) and np.isfinite(out["l100"])
    out = FC.feature_funcs["histogram_ami_1_to_5_even_2"](_series("noise", 7))  # ami{k}: k + 3
    assert np.isfinite(out["ami4"]) and np.isnan(out["ami5"])
    out = FC.feature_funcs["local_global_l_500"](_series("noise", 300))  # the whole series: 0s and 1s
    assert np.isfinite(out["median"]) and np.isnan(out["std"]) and np.isnan(out["absmean"])
    assert FC.field_min_lengths["local_global_p_0p1"].bound("std") is None


def test_field_min_length_in_extract_is_coded_nan():
    sub = FeatureCalculator.__new__(FeatureCalculator)
    sub.feature_funcs = {"range_evolve": FC.feature_funcs["range_evolve"]}
    sub.extract(_series("noise", 100))
    assert sub.quality["range_evolve.p1"].iloc[0] == Quality.NAN
    assert sub.quality["range_evolve.p10"].iloc[0] == Quality.GOOD


def test_custom_config_field_min_length(tmp_path):
    cfg = {"correlation": {"histogram_ami": {
        "base_name": "hami", "field_min_length": {"ami{k}": "k + 1 + extra", "ami1": 50},
        "configs": [{"tau": [[1, 2, 3]], "meth": "even", "num_bins": 2, "zscore": True}]}}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):  # 'extra' is not a parameter
        FeatureCalculator(str(p))
    cfg["correlation"]["histogram_ami"]["field_min_length"] = {"ami{k}": "k + 5", "ami1": 50}
    p.write_text(yaml.safe_dump(cfg))
    fc = FeatureCalculator(str(p))
    lab = next(iter(fc.feature_funcs))
    b = fc.field_min_lengths[lab]
    assert b.bound("ami1") == 50 and b.bound("ami3") == 8 and b.bound("other") is None
    out = fc.feature_funcs[lab](_series("noise", 7))
    assert np.isnan(out["ami1"]) and np.isfinite(out["ami2"]) and np.isnan(out["ami3"])
