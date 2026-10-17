import numpy as np
import pytest

from pyhctsa.utils import get_dataset, z_score, _check_optional_deps, _validate_data, sign_change
import pyhctsa.calculator
import pyhctsa.utils
from conftest import e1000_available, load_e1000_subset

needs_e1000 = pytest.mark.skipif(not e1000_available(),
                                 reason="Empirical1000 not cached; set PYHCTSA_TEST_DOWNLOAD=1 to download it")


# 1. Dataset loading tests
class TestDataLoader:
    @needs_e1000
    def test_get_dataset_default(self):
        # test default behaviour (downloads Empirical1000 on first use)
        data = get_dataset()
        assert isinstance(data, list)
        assert len(data) == 1000
        assert all(isinstance(ts, np.ndarray) for ts in data)
        assert all(isinstance(x, float) for ts in data for x in ts)

    @needs_e1000
    def test_get_data_e1000(self):
        # the downloaded dataset starts with the series of the test subset
        data = get_dataset(which="e1000")
        assert len(data) == 1000, "Expected list of length 1000"
        for a, b in zip(data[:6], load_e1000_subset()):
            assert np.array_equal(a, b)

    def test_fetch_verified_checks_digest_and_caches(self, tmp_path):
        import hashlib
        src = tmp_path / "src.csv"
        src.write_text("1,2,3\n4,5\n")
        payload = src.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        dest = tmp_path / "cache" / "ds.csv"
        out = pyhctsa.utils._fetch_verified(src.as_uri(), str(dest), len(payload), digest)
        assert out == str(dest) and dest.read_bytes() == payload
        src.unlink()   # a verified cached copy is reused without downloading
        assert pyhctsa.utils._fetch_verified(src.as_uri(), str(dest), len(payload), digest) == str(dest)
        with pytest.raises(ConnectionError):   # missing source, nothing cached
            pyhctsa.utils._fetch_verified(src.as_uri(), str(tmp_path / "other.csv"), len(payload), digest)
        src.write_bytes(payload + b"x")         # corrupted download is rejected and not cached
        with pytest.raises(OSError, match="verification"):
            pyhctsa.utils._fetch_verified(src.as_uri(), str(tmp_path / "bad.csv"), len(payload), digest)
        assert not (tmp_path / "bad.csv").exists()

    def test_e1000_cache_dir_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PYHCTSA_DATA_DIR", str(tmp_path))
        assert pyhctsa.utils._data_cache_dir() == str(tmp_path)

    def test_get_dataset_unknown(self):
        # test error thrown when dataset not found
        with pytest.raises(NotImplementedError):
            get_dataset("unknown_dataset")

# 2. z-score function tests
class TestZScoring:
    def test_zscore_basic(self):
        # basic test - is the data zero mean and unit variance
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        x2 = np.array([10, 20, 30, 14, 32, 4])
        z = z_score(x)
        z2 = z_score(x2)
        assert isinstance(z, np.ndarray) # check that numpy array is returned
        assert isinstance(z2, np.ndarray)
        np.testing.assert_almost_equal(np.mean(z), 0, decimal=7)
        np.testing.assert_almost_equal(np.mean(z2), 0, decimal=7)
        np.testing.assert_almost_equal(np.std(z, ddof=1), 1, decimal=7)
        np.testing.assert_almost_equal(np.std(z2, ddof=1), 1, decimal=7)

    def test_zscore_constant_values(self):
        # check behaviour when values are constant
        x = [7, 7, 7, 7]
        with pytest.raises(ValueError):
            z_score(x)
    
    def test_zscore_nonumeric(self):
        # check handling of non-numeric values
        x = ['val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7']
        with pytest.raises(TypeError) as excinfo:
            z_score(x)
        assert "Input cannot be converted to numeric array" in str(excinfo.value)

    @pytest.mark.parametrize("x", [
        [1, 2, np.nan, 4],
        [1, 2, -np.nan, 3],
        [1, np.inf, 10, 4, 5],
        [1, 2, 3, -np.inf, 9, 8, 11]
    ])
    def test_zscore_with_nonfinite(self, x):
        # check behaviour when nonfinite values (nan/infs) are passed into zscore
        with pytest.raises(ValueError):
            z_score(x)

    def test_zscore_with_empty(self):
        # test behaviour when empty list/array is passed
        x = []
        with pytest.raises(ValueError):
            z_score(x)

# 3. Optional dep checks
class TestOptionalDepChecks:
    def test_optional_dep_check_basic(self):
        assert _check_optional_deps('numpy') is True
        assert _check_optional_deps('test') is False
    
# 4. Validate data checks
class TestValidateData:
    def test_data_too_short(self):
        dat = np.random.randn(10)
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_constant(self):
        dat = np.ones(1000)
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_nan(self):
        dat = np.random.randn(1000)
        dat[5] = np.nan
        assert _validate_data(dat) is False, "Expected False to be returned"
    def test_data_inf(self):
        dat = np.random.randn(1000)
        dat[5] = np.inf
        assert _validate_data(dat) is False, "Expected False to be returned"

# 5. Periphery function checks
class TestPeripheryFuncs:
    def test_sign_change_default(self):
        d1 = np.array([0.1, 2, 1, 5, -1, 2, 3, -1, -0.1, -2, 3])
        expected1 = np.array([False, False, False,  True,  True, False,
                              True, False, False,True])
        assert np.array_equal(sign_change(d1), expected1)
        d2 = np.array([1, 2, 3, 4])
        expected2 = np.array([False, False, False])
        assert np.array_equal(sign_change(d2), expected2)
    def test_sign_change_idx(self):
        d1 = np.array([0.1, 2, 1, 5, -1, 2, 3, -1, -0.1, -2, 3])
        expected1 = np.array([3, 4, 6, 9])
        assert np.array_equal(sign_change(d1, 1), expected1)
        d2 = np.array([1, 2, 3, 4])
        assert np.array_equal(sign_change(d2, 1), np.array([]))
    def test_preproc_decorator(self):
        simple_test_func = lambda x: sum(x)
        zs_true_abs_false = pyhctsa.calculator._preprocess_decorator(True, False)
        zs_true_abs_true = pyhctsa.calculator._preprocess_decorator(True, True)
        zs_false_abs_true = pyhctsa.calculator._preprocess_decorator(False, True)
        d = np.array([1, -3, -4, -8, 2, 3, 4, 2, 45, 10, 8, 6])
        assert simple_test_func(d) == 66, "Unexpected output for simple test function"
        assert pytest.approx(0) == zs_true_abs_false(simple_test_func)(d), "Expected near 0 output"
        assert np.isclose(zs_true_abs_true(simple_test_func)(d), 6.997516651978067), "Expected near 6.9975 output"
        assert zs_false_abs_true(simple_test_func)(d) == 96, "Expected output to be 96"
    def test_point_of_crossing(self):
        ds = load_e1000_subset()
        d = ds[0]
        fc_0p1, poc_0p1 = pyhctsa.utils.point_of_crossing(d, 0.1)
        assert fc_0p1 == 1, "Expected first crossing index to be 1."
        assert np.isclose(poc_0p1, 0.009535835889354775)
        fc_0p3, poc_0p3 = pyhctsa.utils.point_of_crossing(d, 0.3)
        assert fc_0p3 == 1, "Expected first crossing index to be 1."
        assert np.isclose(poc_0p3, 0.36128190336832033)
        d2 = ds[5]
        fc_0p9, poc_0p9 = pyhctsa.utils.point_of_crossing(d2, 0.9)
        assert fc_0p9 == 6, "Expected first crossing index to be 6."
        assert np.isclose(poc_0p9, 5.695366116499348)
    def test_make_buffer(self):
        d = get_dataset(which='sinusoid')[0]
        b3 = pyhctsa.utils.make_buffer(d, 3)
        assert b3.shape[0] == 333, "expected there to be 333 rows"
        assert b3.shape[1] == 3, "expected there to be 3 columns"
        b2 = pyhctsa.utils.make_buffer(d, 2)
        assert b2.shape[0] == 500, "expected there to be "
    def test_binarize(self):
        d = load_e1000_subset()[0]
        # test defaults
        res_default = pyhctsa.utils.binarize(d)
        assert int(np.sum(res_default)) == 4961
        res_diff = pyhctsa.utils.binarize(d, binarize_how='mean')
        assert int(np.sum(res_diff)) == 4219
        res_median = pyhctsa.utils.binarize(d, binarize_how='median')
        assert int(np.sum(res_median)) == 5000
        res_iqr = pyhctsa.utils.binarize(d, binarize_how='iqr')
        assert int(np.sum(res_iqr)) == 5000
        with pytest.raises(ValueError):
            pyhctsa.utils.binarize(d, binarize_how='test')
    def test_binarize_iqr_ties(self):
        # quantized series: the 75th percentile is a repeated value, and points equal to it
        # are inside the interquartile range (above the 25th percentile, up to the 75th)
        y = np.repeat([0.0, 1.0, 2.0], [30, 40, 30])
        res_iqr = pyhctsa.utils.binarize(np.roll(y, 17), binarize_how='iqr')
        assert int(np.sum(res_iqr)) == 70
    def test_xcorr(self):
        d1, d2 = load_e1000_subset()[0], load_e1000_subset()[1]
        # basic test, 5 lags
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, max_lags=5)
        assert out1[0] == -5
        assert out1[-1] == 5
        assert len(out2) == 11, "expected 11 values for lag 5"
        assert np.isclose(sum(out2), 8.025332010437447)
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, normed=False, max_lags=5)
        assert np.isclose(sum(out2), 7704.728115530612)
        # test invalid input i.e., mismtach in lengths
        with pytest.raises(ValueError) as excinfo:
            pyhctsa.utils.x_corr(d1, np.random.randn(100), max_lags=5)
        assert 'x and y must have the same number of samples' in str(excinfo.value)
        # test invalid max lags
        with pytest.raises(ValueError) as excinfo2:
            pyhctsa.utils.x_corr(d1, d2, max_lags=-5)
        assert 'max_lags must be between 1 and len(x) - 1' in str(excinfo2.value)
        # set max lags to none
        out1, out2 = pyhctsa.utils.x_corr(d1, d2, max_lags=None)
        d1_len = len(d1)
        assert abs(out1[0]) == (d1_len - 1), "expected max lag to be nx - 1 where nx is length of data"

# 6. Regression tests
class TestRegressions:
    def test_seeded_delay_time(self):
        # delay_time relied on a seeded RNG helper that was not imported
        from pyhctsa.operations.nonlinearity import delay_time
        y = np.random.default_rng(0).standard_normal(500)
        out1 = delay_time(y, random_seed=0)
        out2 = delay_time(y, random_seed=0)
        assert isinstance(out1, dict)
        assert out1 == out2, "same seed should give identical outputs"

    def test_std_nth_deriv_too_short(self):
        # too-short input should return NaN, not raise
        from pyhctsa.operations.stationarity import std_nth_deriv
        assert np.isnan(std_nth_deriv(np.arange(2.0), ndr=3))

    def test_name_mappings_do_not_clobber_range(self, tmp_path, module_config):
        # building the mappings must not change how !range is parsed elsewhere
        from pyhctsa.calculator import FeatureCalculator
        config_path = module_config("correlation")
        before = FeatureCalculator(config_path).feature_funcs
        pyhctsa.utils.make_function_name_mappings(csv_out_fpath=str(tmp_path / "mappings.csv"))
        after = FeatureCalculator(config_path).feature_funcs
        assert list(before) == list(after)
        assert "ac_1" in after and "ac_None" not in after
