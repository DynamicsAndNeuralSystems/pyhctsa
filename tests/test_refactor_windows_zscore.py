"""Regression pins for the window-geometry / make_buffer (#17) and z-scoring (#19) refactor.

The expected values were generated from the main checkout (0fa9013) before the refactor;
the refactored code must reproduce them at rtol 1e-12 (NaN where main gives NaN).
"""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.operations import correlation as co
from pyhctsa.operations import stationarity as st
from pyhctsa.utils import _zscore


def series():
    rng = np.random.default_rng(20261107)
    N = 600
    e = rng.standard_normal(N); ar = np.zeros(N)
    for t in range(1, N): ar[t] = 0.8 * ar[t - 1] + e[t]
    heavy = rng.standard_t(3, N)
    blk = np.r_[np.zeros(150), rng.standard_normal(N - 150)]
    return {'ar1': ar, 'heavy': heavy, 'const_block': blk}

CALLS = {
    'dyn_win': lambda y: st.dyn_win(y, 6),
    'sw_mean': lambda y: st.sliding_window(y, 'mean', 'std', 5, 2),
    'sw_std_inc0': lambda y: st.sliding_window(y, 'std', 'std', 30, 40),
    'sw_ac1': lambda y: st.sliding_window(y, 'AC1', 'std', 10, 2),
    'sw_asym': lambda y: st.sliding_window(y, 'asymAC1', 'std', 4, 1, 'ac1e'),
    'drifting_mean': lambda y: st.drifting_mean(y, 'fix', 70),
    'peak_intervals': lambda y: st.peak_intervals(y, 1),
    'ramping': lambda y: st.ramping_windows(y, 10, 'ac1e'),
    'crinkle_kurt': lambda y: co.crinkle_statistic_kurtosis(y),
    'nl_excess': lambda y: co.nonlinear_autocorr_excess(y, [1, 2, 3]),
    'nl_abs_surr': lambda y: co.nonlinear_autocorr_abs_surrogate_ratio(y, [1, 2], 10, 0),
    'rev_scan': lambda y: co.reversibility_scan(y, 2, 1, 10),
}
KEYS = {'dyn_win': ['stdmean', 'stdstd', 'stdac1', 'stdtaul'],
        'drifting_mean': ['max', 'min', 'meanmaxmin'],
        'peak_intervals': ['meanProm', 'cvInt', 'acInt1'],
        'ramping': ['var_tau', 'ac1_tau', 'asymac1_tau', 'asymac1_pearson_r']}


EXPECTED = {
    ('ar1', 'dyn_win', 'stdmean'): 0.09058550735176474,
    ('ar1', 'dyn_win', 'stdstd'): 0.05748478515261615,
    ('ar1', 'dyn_win', 'stdac1'): 0.012251866110737964,
    ('ar1', 'dyn_win', 'stdtaul'): 6.15976973931862,
    ('ar1', 'sw_mean'): 0.3376633353940254,
    ('ar1', 'sw_std_inc0'): 0.2518969776722617,
    ('ar1', 'sw_ac1'): 0.02734065472294154,
    ('ar1', 'sw_asym'): 0.05920112188338699,
    ('ar1', 'drifting_mean', 'max'): 0.30385531322454185,
    ('ar1', 'drifting_mean', 'min'): -0.6644647213768398,
    ('ar1', 'drifting_mean', 'meanmaxmin'): -0.18030470407614896,
    ('ar1', 'peak_intervals', 'meanProm'): 2.069983975039923,
    ('ar1', 'peak_intervals', 'cvInt'): 0.5545780027421374,
    ('ar1', 'peak_intervals', 'acInt1'): 0.365767043115137,
    ('ar1', 'ramping', 'var_tau'): 0.19999999999999998,
    ('ar1', 'ramping', 'ac1_tau'): 0.5111111111111111,
    ('ar1', 'ramping', 'asymac1_tau'): -0.06666666666666667,
    ('ar1', 'ramping', 'asymac1_pearson_r'): -0.041286034441251646,
    ('ar1', 'crinkle_kurt'): 0.03145608145991474,
    ('ar1', 'nl_excess'): -0.20492726228635827,
    ('ar1', 'nl_abs_surr'): 1.0033483121117843,
    ('ar1', 'rev_scan', 'max_abs_z'): 0.6438357585845649,
    ('ar1', 'rev_scan', 'peak_lag'): 2.0,
    ('ar1', 'rev_scan', 'peak_z'): 0.6438357585845649,
    ('ar1', 'rev_scan', 'sumsq_z'): -2.000932526778425,
    ('heavy', 'dyn_win', 'stdmean'): 0.02110842509738209,
    ('heavy', 'dyn_win', 'stdstd'): 0.06716093172259156,
    ('heavy', 'dyn_win', 'stdac1'): 0.021333511910642075,
    ('heavy', 'dyn_win', 'stdtaul'): 0.1138902890853925,
    ('heavy', 'sw_mean'): 0.08549935901585087,
    ('heavy', 'sw_std_inc0'): 0.3544881861619542,
    ('heavy', 'sw_ac1'): 0.07690223545138483,
    ('heavy', 'sw_asym'): 0.12017979176721393,
    ('heavy', 'drifting_mean', 'max'): 0.0980243520658922,
    ('heavy', 'drifting_mean', 'min'): -0.10452730482657453,
    ('heavy', 'drifting_mean', 'meanmaxmin'): -0.0032514763803411645,
    ('heavy', 'peak_intervals', 'meanProm'): 2.3243284545625293,
    ('heavy', 'peak_intervals', 'cvInt'): 0.5410851051549629,
    ('heavy', 'peak_intervals', 'acInt1'): 0.07552782521184195,
    ('heavy', 'ramping', 'var_tau'): -0.4666666666666666,
    ('heavy', 'ramping', 'ac1_tau'): 0.24444444444444444,
    ('heavy', 'ramping', 'asymac1_tau'): -0.3333333333333333,
    ('heavy', 'ramping', 'asymac1_pearson_r'): -0.46108867851095453,
    ('heavy', 'crinkle_kurt'): 0.05589748772660874,
    ('heavy', 'nl_excess'): 0.0072182178709339195,
    ('heavy', 'nl_abs_surr'): 1.0126218727027836,
    ('heavy', 'rev_scan', 'max_abs_z'): 2.3220281595167642,
    ('heavy', 'rev_scan', 'peak_lag'): 1.0,
    ('heavy', 'rev_scan', 'peak_z'): -2.3220281595167642,
    ('heavy', 'rev_scan', 'sumsq_z'): 0.6592726878969639,
    ('const_block', 'dyn_win', 'stdmean'): 0.014778247561213537,
    ('const_block', 'dyn_win', 'stdstd'): 0.10876861500149575,
    ('const_block', 'dyn_win', 'stdac1'): float('nan'),
    ('const_block', 'dyn_win', 'stdtaul'): float('nan'),
    ('const_block', 'sw_mean'): 0.06219151786116487,
    ('const_block', 'sw_std_inc0'): 0.4973114985920093,
    ('const_block', 'sw_ac1'): float('nan'),
    ('const_block', 'sw_asym'): 0.18892585987607613,
    ('const_block', 'drifting_mean', 'max'): 0.193616058781642,
    ('const_block', 'drifting_mean', 'min'): -0.12239662540116278,
    ('const_block', 'drifting_mean', 'meanmaxmin'): 0.035609716690239615,
    ('const_block', 'peak_intervals', 'meanProm'): 2.548892466641683,
    ('const_block', 'peak_intervals', 'cvInt'): 0.511412708017197,
    ('const_block', 'peak_intervals', 'acInt1'): -0.04525023142157043,
    ('const_block', 'ramping', 'var_tau'): 0.4944132324730442,
    ('const_block', 'ramping', 'ac1_tau'): float('nan'),
    ('const_block', 'ramping', 'asymac1_tau'): -0.4944132324730442,
    ('const_block', 'ramping', 'asymac1_pearson_r'): -0.44890855827593024,
    ('const_block', 'crinkle_kurt'): 0.7336193776452329,
    ('const_block', 'nl_excess'): 0.005207571262755313,
    ('const_block', 'nl_abs_surr'): 1.5705002862063864,
    ('const_block', 'rev_scan', 'max_abs_z'): 2.464311122542016,
    ('const_block', 'rev_scan', 'peak_lag'): 8.0,
    ('const_block', 'rev_scan', 'peak_z'): -2.464311122542016,
    ('const_block', 'rev_scan', 'sumsq_z'): -0.1921595918243521,
}


@pytest.fixture(scope='module')
def computed():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logging.disable(logging.CRITICAL)
        try:
            for sn, y in series().items():
                for nm, f in CALLS.items():
                    r = f(y.copy())
                    if isinstance(r, dict):
                        for k in KEYS.get(nm, sorted(r)[:4]):
                            out[(sn, nm, k)] = float(r[k])
                    else:
                        out[(sn, nm)] = float(r)
        finally:
            logging.disable(logging.NOTSET)
    return out


@pytest.mark.parametrize('key', list(EXPECTED))
def test_pinned_output(computed, key):
    np.testing.assert_allclose(computed[key], EXPECTED[key], rtol=1e-12, atol=0, equal_nan=True)


def test_window_starts():
    # step floor(wlen / inc_move), raised to 1; windows lie wholly inside the series
    np.testing.assert_array_equal(st._window_starts(100, 20, 2), np.arange(0, 81, 10))
    np.testing.assert_array_equal(st._window_starts(100, 20.0, 40), np.arange(81))
    np.testing.assert_array_equal(st._window_starts(10, 10, 1), [0])


def test_zscore_constant_none():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_array_equal(_zscore(x, constant='none'), _zscore(x))
    assert _zscore(np.ones(5), constant='none') is None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert _zscore(np.array([3.0]), constant='none') is None
        assert _zscore(np.array([1.0, np.nan, 2.0]), constant='none') is None
