"""Pinned outputs of the operations whose quantiles now go through ``utils._quantile`` / ``utils._iqr``.

The Hazen quantile and the interquartile range built from it were spelled out at each site
(``np.quantile(..., method='hazen')``, ``np.percentile(..., method='hazen')``); they now call one
helper. The two sites that use NumPy's default linear definition (``extremogram``, ``porta``) pass
``method='linear'``. These values were generated from the main checkout before the refactor
(pyhctsa 0fa9013), so the test checks that no output changed.
"""
import math

import numpy as np
import pytest

from pyhctsa.operations import (correlation, distribution, entropy, extreme_events, graph,  # noqa: F401
                                information, medical, model_fit, nonlinearity, physics, scaling,
                                spectral, stationarity, surrogates, symbolic)
from pyhctsa.utils import _iqr, _quantile


def series():
    n = 300
    r = np.random.default_rng(2026); e = r.standard_normal(n); ar = np.zeros(n)
    for t in range(1, n): ar[t] = 0.8 * ar[t - 1] + e[t]
    r = np.random.default_rng(2027); sine = np.sin(2 * np.pi * np.arange(n) / 23.7) + 0.4 * r.standard_normal(n)
    r = np.random.default_rng(2028); heavy = np.round(r.standard_t(3, n), 1)  # heavy tails with ties
    z = lambda x: (x - x.mean()) / x.std(ddof=1)
    return {'ar1': z(ar), 'sine': z(sine), 'heavy_ties': z(heavy)}


# (id, module, function, kwargs, pinned output keys; None pins every output)
CALLS = [
  ('oversampling', 'correlation', 'oversampling', {}, None),
  ('falling_sticks', 'correlation', 'falling_sticks', {}, ['q10_all']),
  ('stick_angles', 'correlation', 'stick_angles', {}, ['q1_p', 'q10_p', 'q90_p', 'q99_p', 'q1_n', 'q99_n', 'q1_all', 'q99_all']),
  ('embed2_dist', 'correlation', 'embed2_dist', {'tau': 1}, ['d_iqr', 'd_std']),
  ('embed2_shapes', 'correlation', 'embed2_shapes', {'tau': 1}, ['iqr', 'iqronrange', 'median']),
  ('withinp', 'distribution', 'withinp', {'p': 1.0, 'mean_or_median': 'median'}, None),
  ('spread_iqr', 'distribution', 'spread', {'spread_measure': 'iqr'}, None),
  ('quantile_0.3', 'distribution', 'quantile', {'p': 0.3}, None),
  ('quantile_0.99', 'distribution', 'quantile', {'p': 0.99}, None),
  ('pleft', 'distribution', 'pleft', {'th': 0.1}, None),
  ('mean_iqm', 'distribution', 'mean', {'mean_type': 'iqm'}, None),
  ('mean_midhinge', 'distribution', 'mean', {'mean_type': 'midhinge'}, None),
  ('custom_skewness', 'distribution', 'custom_skewness', {'what_skew': 'bowley'}, None),
  ('outlier_test', 'distribution', 'outlier_test', {'p': 2}, None),
  ('distribution_entropy', 'entropy', 'distribution_entropy', {'hist_or_ks': 'hist', 'num_bins': 10, 'olremp': 0.05}, None),
  ('moving_threshold', 'extreme_events', 'moving_threshold', {}, ['iqrq', 'medianq']),
  ('extreme_event_order', 'extreme_events', 'extreme_event_order', {}, ['propPosEvents', 'meanInterval', 'alternationRate', 'lzComplexity']),
  ('extremogram', 'extreme_events', 'extremogram', {}, None),
  ('visibility_graph', 'graph', 'visibility_graph', {}, ['mediank', 'iqrk', 'ol90', 'olu90', 'entropy']),
  ('automutual_info_stats', 'information', 'automutual_info_stats', {'max_tau': 40}, ['pcrossq10', 'pcrossq90']),
  ('porta', 'medical', 'porta', {}, None),
  ('local_density', 'nonlinearity', 'local_density', {}, ['iqrden']),
  ('poincare_section', 'nonlinearity', 'poincare_section', {'tau': 1}, ['iqrx', 'iqry', 'iqrds', 'iqrD', 'boxarea', 'hboxcounts5', 'tracepbox10']),
  ('return_time', 'nonlinearity', 'return_time', {}, ['iqr']),
  ('rqa', 'nonlinearity', 'rqa', {}, ['RR', 'DET', 'L_entr']),
  ('box_count_entropy_rate', 'nonlinearity', 'box_count_entropy_rate', {}, ['iqrstretch', 'medianstretch']),
  ('evt_local_dim', 'nonlinearity', 'evt_local_dim', {}, None),
  ('kramers_moyal', 'physics', 'kramers_moyal', {}, ['driftLin', 'diffSlope', 'pawula']),
  ('fluctuation_analysis', 'scaling', 'fluctuation_analysis', {'wtf': 'iqr'}, ['alpha', 'r1_alpha', 'r2_alpha']),
  ('spectral_summaries', 'spectral', 'spectral_summaries', {}, ['iqr', 'logiqr', 'q25', 'q75']),
  ('rhythmicity', 'spectral', 'rhythmicity', {}, ['lavi_iqr']),
  ('local_global', 'stationarity', 'local_global', {'subset_how': 'l', 'n': 100}, ['iqr']),
  ('moment_corr', 'stationarity', 'moment_corr', {'window_length': 0.05, 'w_overlap': 0.2, 'mom_1': 'median', 'mom_2': 'iqr'}, None),
  ('nstat_z', 'stationarity', 'nstat_z', {'num_seg': 4}, ['median', 'iqr', 'iqroffdiag']),
  ('surrogates', 'surrogates', 'surrogates', {'nsurr': 30, 'surrfn': 'tc3'}, ['iqrsfrommedian']),
  ('surprise', 'symbolic', 'surprise', {'what_prior': 'dist', 'memory': 0.2, 'num_groups': 3, 'coarse_grain_method': 'quantile'}, ['median', 'lq', 'uq', 'mean']),
  ('binary_stats_iqr', 'symbolic', 'binary_stats', {'binary_method': 'iqr'}, ['pupstat2', 'pstretch1', 'longstretch0', 'meanstretch1']),
  ('compare_test_sets', 'model_fit', 'compare_test_sets', {'the_model': 'ar', 'ord': 4, 'subset_how': 'spread', 'sample_p': [25, 0.1], 'steps': 1}, ['stde_iqr', 'ac1_iqr', 'meane_iqr', 'stdrat_iqr']),
]

PINS = {
    'oversampling|ar1': {'eta': 12.49178335825558, 'etaRobust': 7.453175255073525, 'numHalvings': 1.0, 'etaAfterDownsampling': 8.810775668302176},
    'oversampling|sine': {'eta': 7.183445207566253, 'etaRobust': 5.175780705124405, 'numHalvings': 0.0, 'etaAfterDownsampling': 7.183445207566253},
    'oversampling|heavy_ties': {'eta': 9.862736752500531, 'etaRobust': 3.4996807831453496, 'numHalvings': 0.0, 'etaAfterDownsampling': 9.862736752500531},
    'falling_sticks|ar1': {'q10_all': 0.7336792756153545},
    'falling_sticks|sine': {'q10_all': 0.8178549518802632},
    'falling_sticks|heavy_ties': {'q10_all': 1.3010601606662686},
    'stick_angles|ar1': {'q1_p': -2.0664624639876865, 'q10_p': -1.5223651378006495, 'q90_p': 1.3159811020285175, 'q99_p': 2.223637438864829, 'q1_n': -2.060110196443359, 'q99_n': 2.4218140209419383, 'q1_all': -2.061995579736563, 'q99_all': 2.36953011298491},
    'stick_angles|sine': {'q1_p': -2.0781986049498475, 'q10_p': -1.3920171798799252, 'q90_p': 1.2838709291535242, 'q99_p': 1.9793802360906414, 'q1_n': -1.953764483235236, 'q99_n': 1.9676638391537267, 'q1_all': -2.0360245962271004, 'q99_all': 1.9785469881841304},
    'stick_angles|heavy_ties': {'q1_p': -2.2473281315245774, 'q10_p': -1.4004306916695195, 'q90_p': 1.2544938095592493, 'q99_p': 2.271753138883035, 'q1_n': -2.4350501553074078, 'q99_n': 2.3470780299816942, 'q1_all': -2.2867679064552995, 'q99_all': 2.3020933531475696},
    'embed2_dist|ar1': {'d_iqr': 0.5480750157425966, 'd_std': 0.3918926538113},
    'embed2_dist|sine': {'d_iqr': 0.6522387651056023, 'd_std': 0.4991247954111904},
    'embed2_dist|heavy_ties': {'d_iqr': 1.3001717414660057, 'd_std': 1.1460211356840608},
    'embed2_shapes|ar1': {'iqr': 0.2548790754630338, 'iqronrange': 0.4853176916350917, 'median': 0.35611510791366907},
    'embed2_shapes|sine': {'iqr': 0.11619718309859156, 'iqronrange': 0.31428571428571433, 'median': 0.27816901408450706},
    'embed2_shapes|heavy_ties': {'iqr': 0.326530612244898, 'iqronrange': 0.5393258426966293, 'median': 0.3741496598639456},
    'withinp|ar1': {'value': 0.6533333333333333},
    # withinp (median mode) scales the IQR by the exact IQR_TO_SIGMA since cons-mad-constant (#61)
    # instead of 1/1.35: one sample of the sine series moves across the band (0.7167 on main).
    'withinp|sine': {'value': 0.72},
    'withinp|heavy_ties': {'value': 0.68},
    'spread_iqr|ar1': {'value': 1.2406424559498603},
    'spread_iqr|sine': {'value': 1.569826084696433},
    'spread_iqr|heavy_ties': {'value': 0.977985995398815},
    'quantile_0.3|ar1': {'value': -0.4800551726751431},
    'quantile_0.3|sine': {'value': -0.6883749033295706},
    'quantile_0.3|heavy_ties': {'value': -0.41517082901446467},
    'quantile_0.99|ar1': {'value': 2.947594469162145},
    'quantile_0.99|sine': {'value': 1.8951041055186817},
    'quantile_0.99|heavy_ties': {'value': 2.6765268338592083},
    'pleft|ar1': {'value': 1.6864996142429407},
    'pleft|sine': {'value': 1.5977724398658952},
    'pleft|heavy_ties': {'value': 1.703588508114065},
    'mean_iqm|ar1': {'value': -0.009128224257005983},
    'mean_iqm|sine': {'value': -0.02367561055139936},
    'mean_iqm|heavy_ties': {'value': -0.006292982890596794},
    'mean_midhinge|ar1': {'value': -0.0159770057693156},
    'mean_midhinge|sine': {'value': 0.0012492995064028456},
    'mean_midhinge|heavy_ties': {'value': -0.02082163732139411},
    'custom_skewness|ar1': {'value': -0.01621368924967248},
    'custom_skewness|sine': {'value': 0.0313775704489145},
    'custom_skewness|heavy_ties': {'value': 0.03225806451612909},
    'outlier_test|ar1': {'mean': -0.01640994092454195, 'std': 0.8730661152224988},
    'outlier_test|sine': {'mean': 0.0007118597891487308, 'std': 0.937452164234644},
    'outlier_test|heavy_ties': {'mean': 0.028944119827477875, 'std': 0.7566167872155627},
    'distribution_entropy|ar1': {'value': 0.3017274056414707},
    'distribution_entropy|sine': {'value': 0.20948352134416348},
    'distribution_entropy|heavy_ties': {'value': 0.42545932600041536},
    'moving_threshold|ar1': {'iqrq': 1.2601609053225982, 'medianq': 1.552840607311409},
    'moving_threshold|sine': {'iqrq': 1.0839232127546161, 'medianq': 1.93935812573081},
    'moving_threshold|heavy_ties': {'iqrq': 1.8553947145829395, 'medianq': 2.2912764292610666},
    'extreme_event_order|ar1': {'propPosEvents': 0.5, 'meanInterval': 8.689655172413794, 'alternationRate': 0.27586206896551724, 'lzComplexity': 0.9813781191217037},
    'extreme_event_order|sine': {'propPosEvents': 0.5, 'meanInterval': 9.931034482758621, 'alternationRate': 0.4827586206896552, 'lzComplexity': 1.635630198536173},
    'extreme_event_order|heavy_ties': {'propPosEvents': 0.5, 'meanInterval': 10.10344827586207, 'alternationRate': 0.4482758620689655, 'lzComplexity': 1.4720671786825557},
    'extremogram|ar1': {'max_chi': 0.5333333333333333, 'max_z': 8.371578903249574, 'peak_lag': 1.0},
    'extremogram|sine': {'max_chi': 0.14285714285714285, 'max_z': 1.5537971921347116, 'peak_lag': 12.0},
    'extremogram|heavy_ties': {'max_chi': 0.08333333333333333, 'max_z': 0.7505553499465135, 'peak_lag': 1.0},
    'visibility_graph|ar1': {'mediank': 3.0, 'iqrk': 3.0, 'ol90': 0.9707018103234492, 'olu90': 2.574917097544444, 'entropy': 1.1690373972720802},
    'visibility_graph|sine': {'mediank': 3.0, 'iqrk': 3.0, 'ol90': 0.9404697866236328, 'olu90': 2.534437403846074, 'entropy': 1.3028399693138775},
    'visibility_graph|heavy_ties': {'mediank': 3.0, 'iqrk': 3.0, 'ol90': 0.9192027484710411, 'olu90': 2.843349302624689, 'entropy': 1.4208211494467133},
    'automutual_info_stats|ar1': {'pcrossq10': 0.07692307692307693, 'pcrossq90': 0.02564102564102564},
    'automutual_info_stats|sine': {'pcrossq10': 0.15384615384615385, 'pcrossq90': 0.1282051282051282},
    'automutual_info_stats|heavy_ties': {'pcrossq10': 0.20512820512820512, 'pcrossq90': 0.20512820512820512},
    'porta|ar1': {'pV0': 30.201342281879196, 'pV1': 44.29530201342282, 'pV2LV': 7.718120805369128, 'pV2UV': 17.78523489932886},
    'porta|sine': {'pV0': 26.174496644295303, 'pV1': 45.63758389261745, 'pV2LV': 3.3557046979865772, 'pV2UV': 24.83221476510067},
    'porta|heavy_ties': {'pV0': 7.382550335570469, 'pV1': 39.59731543624161, 'pV2LV': 13.422818791946309, 'pV2UV': 39.59731543624161},
    'local_density|ar1': {'iqrden': 1.5031289986743528},
    'local_density|sine': {'iqrden': 1.1106280185320534},
    'local_density|heavy_ties': {'iqrden': 1.6092779316314556},
    'poincare_section|ar1': {'iqrx': 0.6036005709800908, 'iqry': 0.45979490214778646, 'iqrds': 0.9556753583773138, 'iqrD': 0.5112809938045844, 'boxarea': 4.1444170716882285, 'hboxcounts5': 2.848379976031844, 'tracepbox10': 0.3571428571428571},
    'poincare_section|sine': {'iqrx': 1.430965832502985, 'iqry': 0.552794265583484, 'iqrds': 1.2144642381555126, 'iqrD': 0.675592774299073, 'boxarea': 7.405807312257583, 'hboxcounts5': 2.576746780371153, 'tracepbox10': 0.20833333333333331},
    'poincare_section|heavy_ties': {'iqrx': 0.7874995618433942, 'iqry': 0.7548546118289896, 'iqrds': 0.9658840676831321, 'iqrD': 0.761923662212495, 'boxarea': 31.143263305849963, 'hboxcounts5': 3.262428890270546, 'tracepbox10': 0.18571428571428572},
    'return_time|ar1': {'iqr': 0.9766478801902948},
    'return_time|sine': {'iqr': 1.8128747628406612},
    'return_time|heavy_ties': {'iqr': 1.0696296156468286},
    'rqa|ar1': {'RR': 0.096278552071235, 'DET': 0.7680321688866549, 'L_entr': 1.4900370538704701},
    'rqa|sine': {'RR': 0.09804479203697121, 'DET': 0.7198452985255016, 'L_entr': 1.4008950001508487},
    'rqa|heavy_ties': {'RR': 0.09972514887769125, 'DET': 0.6798346348185577, 'L_entr': 1.1677833731435203},
    'box_count_entropy_rate|ar1': {'iqrstretch': 1.2384321003055205, 'medianstretch': 0.3666252747919003},
    'box_count_entropy_rate|sine': {'iqrstretch': 0.7296357388309656, 'medianstretch': 0.0825210236880034},
    'box_count_entropy_rate|heavy_ties': {'iqrstretch': 1.7667904578988534, 'medianstretch': 0.5439764596439662},
    'evt_local_dim|ar1': {'propValidPoles': float("nan"), 'meanLocalDim': float("nan"), 'stdLocalDim': float("nan"), 'meanTheta': float("nan"), 'stdTheta': float("nan"), 'corrDimTheta': float("nan")},
    'evt_local_dim|sine': {'propValidPoles': float("nan"), 'meanLocalDim': float("nan"), 'stdLocalDim': float("nan"), 'meanTheta': float("nan"), 'stdTheta': float("nan"), 'corrDimTheta': float("nan")},
    'evt_local_dim|heavy_ties': {'propValidPoles': float("nan"), 'meanLocalDim': float("nan"), 'stdLocalDim': float("nan"), 'meanTheta': float("nan"), 'stdTheta': float("nan"), 'corrDimTheta': float("nan")},
    'kramers_moyal|ar1': {'driftLin': float("nan"), 'diffSlope': float("nan"), 'pawula': float("nan")},
    'kramers_moyal|sine': {'driftLin': float("nan"), 'diffSlope': float("nan"), 'pawula': float("nan")},
    'kramers_moyal|heavy_ties': {'driftLin': float("nan"), 'diffSlope': float("nan"), 'pawula': float("nan")},
    'fluctuation_analysis|ar1': {'alpha': 0.19879926509694262, 'r1_alpha': 0.7386485578394638, 'r2_alpha': 0.21471134163934202},
    'fluctuation_analysis|sine': {'alpha': -0.016864190430719603, 'r1_alpha': 0.7616902327971025, 'r2_alpha': -0.012960423262657894},
    'fluctuation_analysis|heavy_ties': {'alpha': 0.5678463671514717, 'r1_alpha': 0.6871868814341967, 'r2_alpha': 0.5356767295434901},
    'spectral_summaries|ar1': {'iqr': 0.11609222520926946, 'logiqr': 2.0095338501044764, 'q25': 0.017971844287015627, 'q75': 0.13406406949628508},
    'spectral_summaries|sine': {'iqr': 0.11227162050556641, 'logiqr': 1.7968672977551194, 'q25': 0.022318465336354593, 'q75': 0.134590085841921},
    'spectral_summaries|heavy_ties': {'iqr': 0.32015228142925495, 'logiqr': 1.411864149316159, 'q25': 0.10315829301204454, 'q75': 0.4233105744412995},
    'rhythmicity|ar1': {'lavi_iqr': 0.07607450070871269},
    'rhythmicity|sine': {'lavi_iqr': 0.1482974148210795},
    'rhythmicity|heavy_ties': {'lavi_iqr': 0.11781818692138218},
    'local_global|ar1': {'iqr': 0.19856048130508097},
    'local_global|sine': {'iqr': 0.09249777098404399},
    'local_global|heavy_ties': {'iqr': 0.06451612903225801},
    'moment_corr|ar1': {'R': 0.5550553230136883, 'absR': 0.5550553230136883, 'density': 4.9240369264607695},
    'moment_corr|sine': {'R': 0.3217502924125236, 'absR': 0.3217502924125236, 'density': 8.0830287191531},
    'moment_corr|heavy_ties': {'R': -0.06599098957356196, 'absR': 0.06599098957356196, 'density': 37.85558181904096},
    'nstat_z|ar1': {'median': 0.8124932387240962, 'iqr': 0.06171399632547525, 'iqroffdiag': 0.05218057457769387},
    'nstat_z|sine': {'median': 0.744047438114306, 'iqr': 0.057710825737705695, 'iqroffdiag': 0.04494946855952631},
    'nstat_z|heavy_ties': {'median': 1.0112587491288565, 'iqr': 0.0760208385745611, 'iqroffdiag': 0.06996748869854563},
    'surrogates|ar1': {'iqrsfrommedian': 1.4777175469808959},
    'surrogates|sine': {'iqrsfrommedian': 0.8260777754452291},
    'surrogates|heavy_ties': {'iqrsfrommedian': 1.551537903444498},
    'surprise|ar1': {'median': 1.148622709242771, 'lq': 0.9203640572617908, 'uq': 1.3156767939059373, 'mean': 1.1622261803876606},
    'surprise|sine': {'median': 1.148622709242771, 'lq': 1.0282530526404763, 'uq': 1.201266442728193, 'mean': 1.1329601109923655},
    'surprise|heavy_ties': {'median': 1.0986122886681098, 'lq': 1.0055218656020977, 'uq': 1.2568362938830038, 'mean': 1.11999000272458},
    'binary_stats_iqr|ar1': {'pupstat2': 0.7647058823529411, 'pstretch1': 0.14333333333333334, 'longstretch0': 14.0, 'meanstretch1': 3.488372093023256},
    'binary_stats_iqr|sine': {'pupstat2': 0.948051948051948, 'pstretch1': 0.21, 'longstretch0': 8.0, 'meanstretch1': 2.380952380952381},
    'binary_stats_iqr|heavy_ties': {'pupstat2': 1.054054054054054, 'pstretch1': 0.25, 'longstretch0': 7.0, 'meanstretch1': 2.026666666666667},
    'compare_test_sets|ar1': {'stde_iqr': 0.11415603291747473, 'ac1_iqr': 0.24249442882530625, 'meane_iqr': 0.11670768133300422, 'stdrat_iqr': 0.07140005997821697},
    'compare_test_sets|sine': {'stde_iqr': 0.1584466859225243, 'ac1_iqr': 0.2321803078383256, 'meane_iqr': 0.10434872808700917, 'stdrat_iqr': 0.11873764440391787},
    'compare_test_sets|heavy_ties': {'stde_iqr': 0.2195496661575822, 'ac1_iqr': 0.22042605130826803, 'meane_iqr': 0.2196504844239674, 'stdrat_iqr': 0.024220190408709347},
}

_SERIES = series()


@pytest.mark.parametrize('cid, mod, fn, kw, keys', CALLS, ids=[c[0] for c in CALLS])
def test_quantile_refactor_pins(cid, mod, fn, kw, keys):
    f = getattr(globals()[mod], fn)
    for sname, y in _SERIES.items():
        out = f(y.copy(), **kw)
        if not isinstance(out, dict):
            out = {'value': out}
        for k, expected in PINS[f'{cid}|{sname}'].items():
            got = float(out[k])
            if math.isnan(expected):
                assert math.isnan(got), (cid, sname, k, got)
            else:
                assert got == pytest.approx(expected, rel=1e-12, abs=0), (cid, sname, k, got, expected)


def test_helpers_match_numpy():
    rng = np.random.default_rng(7)
    x = np.round(rng.standard_normal(57), 1)
    for q in (0.0, 0.01, 0.25, 0.5, 0.9, 1.0):
        assert _quantile(x, q) == np.quantile(x, q, method='hazen')
        assert _quantile(x, q, method='linear') == np.quantile(x, q)
    assert _iqr(x) == np.quantile(x, 0.75, method='hazen') - np.quantile(x, 0.25, method='hazen')
    X = rng.standard_normal((40, 5))
    np.testing.assert_array_equal(_iqr(X, axis=0), np.percentile(X, 75, axis=0, method='hazen')
                                  - np.percentile(X, 25, axis=0, method='hazen'))
    assert np.isnan(_quantile(np.array([1.0, np.nan, 2.0]), 0.5))
