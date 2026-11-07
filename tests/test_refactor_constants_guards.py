"""Pinned outputs of the operations touched by the constants-and-guards refactor.

The refactor put the MAD-to-sigma constants (``utils.MAD_TO_SIGMA``, ``utils.MAD_DIVISOR``), the
round-off guard (``utils._is_roundoff``) and ``_piecewise_poly_residual`` in ``utils``, moved
``_rank_map_gaussian`` to ``robust`` and ``_ar_rms_error`` to ``model_fit`` (breaking the
``model_fit`` <-> ``pre_process`` import cycle), and removed the alias wrappers
``_armax_residuals`` and ``_kstep_residuals_ss``. None of this changes any output.

The expected values below were generated from the main checkout before the refactor
(pyhctsa main at 0fa9013), on two seeded series.
"""
import numpy as np
import pytest

from pyhctsa.fitting import robust_linear_fit
from pyhctsa.operations import entropy as en
from pyhctsa.operations import medical as md
from pyhctsa.operations import model_fit as mf
from pyhctsa.operations import pre_process as pp
from pyhctsa.operations import spectral as sp
from pyhctsa.robust import bf_ks_density, bf_residual_stats
from pyhctsa.utils import _is_roundoff, get_tau, z_score

NAN = float('nan')


def _series():
    r = np.random.default_rng(11)
    e = r.standard_normal(300)
    a = np.zeros(300)
    for t in range(1, 300):
        a[t] = 0.8 * a[t - 1] + e[t]
    r = np.random.default_rng(12)
    t = np.arange(300)
    s = np.sin(2 * np.pi * t / 23.7) + 0.4 * r.standard_normal(300)
    return {'ar1': z_score(a), 'sine': z_score(s)}


SERIES = _series()

CALLS = {
    'hrv_classic': lambda y: md.hrv_classic(y),
    'specparam': lambda y: sp.specparam(y),
    'envelope_stats': lambda y: sp.envelope_stats(y),
    'rhythmicity': lambda y: sp.rhythmicity(y),
    'sinusoid_fit': lambda y: sp.sinusoid_fit(y),
    'distribution_entropy_ks': lambda y: en.distribution_entropy(y, 'ks'),
    'get_tau_mi': lambda y: get_tau(y, 'mi'),
    'robust_linear_fit': lambda y: np.concatenate([robust_linear_fit(y[:-1], y[1:])[0],
                                                   [robust_linear_fit(y[:-1], y[1:])[1]['se'][0]]]),
    'bf_ks_density': lambda y: bf_ks_density(y)[0][::10],
    'bf_residual_stats': lambda y: bf_residual_stats(y[1:] - 0.5 * y[:-1], y[1:]),
    'preproc_compare': lambda y: pp.preproc_compare(y),
    'preproc_iterate': lambda y: pp.preproc_iterate(y),
    'preproc_model_fit': lambda y: pp.preproc_model_fit(y),
    'garch_fit': lambda y: mf.garch_fit(y),
    'armax': lambda y: mf.armax(y, (2, 2), 0.5, 3),
    'steps_ahead_ss': lambda y: mf.steps_ahead(y, 'ss', 2, 3),
    'compare_test_sets_ss': lambda y: mf.compare_test_sets(y, 'ss', 2, 'uniform', (25, 0.1), 1),
    'gp_hyperparameters': lambda y: mf.gp_hyperparameters(y, 'covSEiso_covNoise', max_n=50, resample_how='first'),
    'gp_fit_across': lambda y: mf.gp_fit_across(y),
}

EXPECTED = {
    'hrv_classic|ar1': {'pnnrel025': 0.8260869565217391, 'pnnrel05': 0.6521739130434783, 'pnnrel1': 0.32441471571906355, 'pnnrel2': 0.05351170568561873, 'pnnrel3': 0.0, 'lfhf': 0.7005144860060833, 'vlf': 3.100191674367942, 'lf': 25.743484192924203, 'hf': 36.749395918560126, 'tri': 4.166666666666667, 'SD1': 447.58191387194006, 'SD2': 1341.5179575297273},
    'specparam|ar1': {'apExponent': 1.543703159741261, 'apOffset': 0.30624944010819544, 'numPeaks': 0.0, 'maxPeakFreq': NAN, 'maxPeakPower': NAN, 'maxPeakBW': NAN, 'totalPeakPower': 0.0, 'periodicFraction': 0.0, 'modelR2': 0.8156041118182904, 'modelMAE': 0.08488111314444795},
    'envelope_stats|ar1': {'full_cv': 0.4833063797409645, 'full_kurt': 2.8045763345538015, 'full_tau': 4.094592905797782, 'dom_cv': 0.5049282192721426, 'dom_skew': 0.38976631003839923, 'dom_kurt': 2.4006972870963814, 'dom_tau': 15.496207962530775, 'dom_ifspread': 0.010889741686716099},
    'rhythmicity|ar1': {'lavi_mean': 0.3362448796394521, 'lavi_median': 0.3155179481665234, 'lavi_std': 0.0901876001142985, 'lavi_iqr': 0.12525038463157045, 'lavi_max': 0.5112421050800628, 'lavi_min': 0.17965957769168361, 'lavi_maxOnMedian': 1.620326539427927, 'lavi_freqAtMax': 2.1553627491362284, 'lavi_freqAtMin': 2.5132741228718345, 'lavi_meanAboveMedian': 0.09703402315158656, 'lavi_fracSustained': 0.07692307692307693, 'lavi_fracTransient': 0.0},
    'sinusoid_fit|ar1': {'r2': 0.13546562638257675, 'adjr2': 0.1296438460888567, 'rmse': 0.9329288043099234, 'resAC1': 0.7666414477145047, 'resAC2': 0.5767457931039631, 'resrunsz': -10.409726994853541},
    'distribution_entropy_ks|ar1': [1.970222296574162],
    'get_tau_mi|ar1': [4.0],
    'robust_linear_fit|ar1': [0.0017209305773072338, 0.7967245685904693, 0.03665814691110904],
    'bf_ks_density|ar1': [9.255562567603058e-05, 0.011069655754865173, 0.03431835515490478, 0.1280751641894135, 0.25736062816044664, 0.40879152312411926, 0.2877101730479515, 0.10207718390065579, 0.024579517500850908, 0.00364756068195168],
    'bf_residual_stats|ar1': [0.40645555176419856, 0.2986262137243846, -3.9969363018511497],
    'preproc_compare|ar1': {'statav2': 0.07351547861016382, 'swms2_2': 0.014915685892036314, 'swms5_1': 0.04031233324773185, 'swms10_1': 0.03573925978631528, 'swss2_1': 0.22074896399099708, 'swss5_1': 0.17113807307731999, 'swss10_1': 0.09862292132937572, 'gauss1_kd_r2': -0.01153523254174138, 'gauss1_kd_resAC1': 0.0021569202443602054, 'gauss1_kd_resrunsz': 0.4020356365562936, 'kscn_peaksepy': -0.017446658372852475, 'kscn_peaksepx': -0.027527527527527962, 'kscn_olapint': 0.0037921530381242324, 'kscn_relent': 0.006986549048658532, 'olbt_m2': 0.004783528068250933, 'olbt_m5': -0.0009343242475037764, 'olbt_s5': -0.005914656316736566},
    'preproc_iterate|ar1': {'statav5_trend': -2.4620666826341835, 'statav5_jump': 3.731416849025093, 'swms5_2_trend': -2.605228740196594, 'swms5_2_jump': 3.844546185660663, 'swss5_2_trend': 2.5110091264327696, 'swss5_2_jump': 3.087790381857614, 'gauss1_kd_trend': -2.7463100074835483, 'gauss1_kd_jump': 3.6504588206164827, 'gauss1_hsqrt_trend': -2.3098666850640677, 'gauss1_hsqrt_jump': 3.531184497056363, 'norm_kscomp_trend': -2.610382152066633, 'norm_kscomp_jump': 2.9616854495836, 'ol_trend': 0.36514556556377564, 'ol_jump': 1.9362494909421275, 'xcn1_trend': NAN, 'xcn1_jump': NAN, 'xc1_trend': NAN, 'xc1_jump': NAN, 'normdiff_trend': NAN, 'normdiff_jump': NAN},
    'preproc_model_fit|ar1': {'stderat_d1': 1.6415641222884199, 'stderat_d2': 1.3287329674020867, 'stderat_p1_20': 1.5573788044869799, 'stderat_p2_5': 1.1220907201419525, 'stderat_rmgd': 1.03126264347702},
    'garch_fit|ar1': {'hasStructure': 0.0, 'offset': 0.0, 'constant': 0.9966555183946488, 'constanterr': NAN, 'GARCH_1': 0.0, 'GARCHerr_1': NAN, 'ARCH_1': 0.0, 'ARCHerr_1': NAN, 'leverage': NAN, 'leverageerr': NAN, 'distDoF': NAN, 'LLF': -1.4165019186207959},
    'armax|ar1': {'AR_1': -1.14002720546604, 'AR_2': 0.2722632055791465, 'MA_1': -0.3306967478137346, 'MA_2': -0.008906859438537381, 'maxda': 0.5440792036079743, 'maxdc': 0.5475904470150063, 'noisevar': 0.36289876796361326, 'lossfn': 0.3580601177240984, 'fpe': 0.3677374182031281, 'meane': -0.014479523181231765, 'meanabs': 0.7028614023858274, 'stde': 0.8777530379746259, 'maxonstd': 3.054322896511163, 'ac1': 0.7267310878623915, 'ac2': 0.4897653487484831, 'ac3': 0.2520446594843283, 'propbth': 0.88, 'taurat': 0.8945513451007979, 'sws': 0.17267380310389285, 'swm': 0.40749922997834787, 'ftbth': 4.0, 'normksstat': 0.05225769992681045, 'popt': 1.0, 'minsbc': -0.7905724074594094},
    'steps_ahead_ss|ar1': {'stde_h1': 0.9471225570029191, 'meanabs_h1': 0.9535371566281783, 'ac1_h1': 0.0030848336198310406, 'stde_h2': 0.9165131659906888, 'meanabs_h2': 0.8932884775480379, 'ac1_h2': 0.4741729215032184, 'stde_h3': 0.8748918691933406, 'meanabs_h3': 0.8673814053878763, 'ac1_h3': 0.6593475810938084, 'stde_meanabs_diff': 0.15190176364567207, 'stde_meandiff': 0.12766380010573308, 'stde_maxdiff': 0.17140940427298523, 'stde_stddiff': 0.06186562670753298, 'stde_ndown': 0.0},
    'compare_test_sets_ss|ar1': {'stde_mean': 0.5932594718014773, 'stde_std': 0.08272594774868247, 'stde_iqr': 0.11217338972809121, 'ac1_mean': 0.03382159630564277, 'ac1_median': 0.00470497968288885, 'ac1_std': 0.2517983578097202, 'ac1_iqr': 0.4793177930193604, 'meane_mean': 0.10317059780655458, 'meane_std': 0.08259475089156305, 'meane_iqr': 0.10362879230825131, 'stdrat_mean': 0.7987992748525978, 'stdrat_median': 0.8048936201290205, 'stdrat_std': 0.058308264483653865, 'stdrat_iqr': 0.0516427328688166},
    'gp_hyperparameters|ar1': {'logh1': 1.6897612314038541, 'logh2': 0.05604196497622301, 'logh3': -0.8545446360412418, 'nlml': 0.8578789954434987, 'stde': 0.38361090174275714, 'meanabs_std': 0.6453885005614811, 'std_mu_data': 0.9274582964264593, 'std_S_data': 0.009767339025851782, 'maxS': 0.5076287955714036, 'minS': 0.46030042394905557, 'meanS': 0.4624818920935805, 'oosmedabserr': NAN},
    'gp_fit_across|ar1': {'stde': 0.9566685618833833, 'meanabs_std': 0.8209008763311675, 'stdmu': 0.16099046091901195, 'meanS': 0.925685267818252, 'stdS': 0.00044411169870946075, 'nlml': 1.3287721351184583, 'logh1': 2.756004594224246, 'logh2': -0.9984368270677072, 'logh3': -0.1419918919614959, 'h_lonN': 0.0524561403508772},
    'hrv_classic|sine': {'pnnrel025': 0.8060200668896321, 'pnnrel05': 0.6053511705685619, 'pnnrel1': 0.3110367892976589, 'pnnrel2': 0.05016722408026756, 'pnnrel3': 0.0, 'lfhf': 0.002728216024158437, 'vlf': 0.4250497644337917, 'lf': 0.21899734924658845, 'hf': 80.27126419145706, 'tri': 6.122448979591836, 'SD1': 499.7229785059535, 'SD2': 1322.9803266689714},
    'specparam|sine': {'apExponent': 0.3801328250382594, 'apOffset': -0.17623895464369788, 'numPeaks': 0.0, 'maxPeakFreq': NAN, 'maxPeakPower': NAN, 'maxPeakBW': NAN, 'totalPeakPower': 0.0, 'periodicFraction': 0.0, 'modelR2': 0.3370856207465839, 'modelMAE': 0.0696680890528908},
    'envelope_stats|sine': {'full_cv': 0.3371685692226656, 'full_kurt': 2.558041846743786, 'full_tau': 0.8697942164930907, 'dom_cv': 0.13793303545161992, 'dom_skew': -1.3446034567499938, 'dom_kurt': 4.640099057805704, 'dom_tau': 18.566884721821424, 'dom_ifspread': 0.0011500278515764637},
    'rhythmicity|sine': {'lavi_mean': 0.3516502480266407, 'lavi_median': 0.35294829187567395, 'lavi_std': 0.09245338779176403, 'lavi_iqr': 0.11811071786228178, 'lavi_max': 0.536624458934526, 'lavi_min': 0.17122002368465736, 'lavi_maxOnMedian': 1.520405315131974, 'lavi_freqAtMax': 1.2589316974698097, 'lavi_freqAtMin': 1.079649235131062, 'lavi_meanAboveMedian': 0.07288097058537986, 'lavi_fracSustained': 0.0, 'lavi_fracTransient': 0.0},
    'sinusoid_fit|sine': {'r2': 0.7610980172007877, 'adjr2': 0.7594892496398503, 'rmse': 0.4904189539160877, 'resAC1': 0.06791383316269949, 'resAC2': -0.02935171127863397, 'resrunsz': -1.3879635993138053},
    'distribution_entropy_ks|sine': [1.7106396193583342],
    'get_tau_mi|sine': [4.0],
    'robust_linear_fit|sine': [0.006902666568615935, 0.7578792439441135, 0.039282824884364645],
    'bf_ks_density|sine': [0.000111683798319969, 0.008646591958122663, 0.07931540952507672, 0.23825311845777342, 0.31327122490253057, 0.28794011355191235, 0.2941013059964076, 0.15442119266295218, 0.030886676913445106, 0.0019507101949600956],
    'bf_residual_stats|sine': [0.22708914204205624, 0.35131113758952465, -4.228653985661812],
    'preproc_compare|sine': {'statav2': -0.44474682596034354, 'swms2_2': 0.3858199029020582, 'swms5_1': 0.011031567439936833, 'swms10_1': 0.04645086791246811, 'swss2_1': 0.38755473871500706, 'swss5_1': 0.0017630996857338768, 'swss10_1': 0.06166028562529506, 'gauss1_kd_r2': -0.06614652372459628, 'gauss1_kd_resAC1': 0.0030005279595568535, 'gauss1_kd_resrunsz': 0.0, 'kscn_peaksepy': -0.01760185483787835, 'kscn_peaksepx': 0.10740740740740762, 'kscn_olapint': -0.010592648493930325, 'kscn_relent': 0.023064663032452172, 'olbt_m2': -0.0016533342778481705, 'olbt_m5': -0.008986129885466687, 'olbt_s5': 0.01415811478727297},
    'preproc_iterate|sine': {'statav5_trend': -1.8583122512564783, 'statav5_jump': 3.490235206681201, 'swms5_2_trend': -2.2992292322337833, 'swms5_2_jump': 3.953176383930759, 'swss5_2_trend': 2.4587749565612107, 'swss5_2_jump': 3.641934867251773, 'gauss1_kd_trend': 1.9777562833800955, 'gauss1_kd_jump': 3.1805356991490017, 'gauss1_hsqrt_trend': 1.7229684834103471, 'gauss1_hsqrt_jump': 3.7356686132524373, 'norm_kscomp_trend': 1.7905846551977338, 'norm_kscomp_jump': 2.601095951792396, 'ol_trend': -1.2254912098011248, 'ol_jump': 2.3353499036030687, 'xcn1_trend': NAN, 'xcn1_jump': NAN, 'xc1_trend': NAN, 'xc1_jump': NAN, 'normdiff_trend': NAN, 'normdiff_jump': NAN},
    'preproc_model_fit|sine': {'stderat_d1': 1.4725203336729091, 'stderat_d2': 1.074162592505232, 'stderat_p1_20': 1.487334340481193, 'stderat_p2_5': 1.0518365138434016, 'stderat_rmgd': 1.0638893277751407},
    'garch_fit|sine': {'hasStructure': 0.0, 'offset': 0.0, 'constant': 0.9966666666666667, 'constanterr': NAN, 'GARCH_1': 0.0, 'GARCHerr_1': NAN, 'ARCH_1': 0.0, 'ARCHerr_1': NAN, 'leverage': NAN, 'leverageerr': NAN, 'distDoF': NAN, 'LLF': -1.4158015249646054},
    'armax|sine': {'AR_1': -1.9115562640544936, 'AR_2': 0.9829141891484994, 'MA_1': -1.6723502498612663, 'MA_2': 0.7460096052029674, 'maxda': 0.009541284841247126, 'maxdc': 0.040520313725915275, 'noisevar': 0.27266184749537997, 'lossfn': 0.2690263561954416, 'fpe': 0.2762973387953184, 'meane': 0.01790249750625929, 'meanabs': 0.43693668696704685, 'stde': 0.5490736527326076, 'maxonstd': 2.6950979619303252, 'ac1': 0.1461041050885978, 'ac2': 0.002851282231715088, 'ac3': -0.23449594672705443, 'propbth': 0.96, 'taurat': 0.34526027771182494, 'sws': 0.16416804175125205, 'swm': 0.16551843104248923, 'ftbth': 1.0, 'normksstat': 0.04676223716906641, 'popt': 1.0, 'minsbc': 0.025318390182079042},
    'steps_ahead_ss|sine': {'stde_h1': 0.7362870784110038, 'meanabs_h1': 0.7116711684746337, 'ac1_h1': 0.020131051041348384, 'stde_h2': 0.6313200191773921, 'meanabs_h2': 0.6072514935899266, 'ac1_h2': 0.11703785953463919, 'stde_h3': 0.5363935419521004, 'meanabs_h3': 0.5156368450560186, 'ac1_h3': 0.1407484044199863, 'stde_meanabs_diff': 0.10802118086555713, 'stde_meandiff': 0.0062730805144420865, 'stde_maxdiff': 0.008691468064165653, 'stde_stddiff': 0.003420116471893305, 'stde_ndown': 0.0},
    'compare_test_sets_ss|sine': {'stde_mean': 0.5982917905676659, 'stde_std': 0.09860629067658858, 'stde_iqr': 0.13636032740004178, 'ac1_mean': 0.07934947046350432, 'ac1_median': 0.054687109426880695, 'ac1_std': 0.19464302213881124, 'ac1_iqr': 0.22583416133351134, 'meane_mean': 0.08542813313408647, 'meane_std': 0.08395363584676051, 'meane_iqr': 0.07723734146985989, 'stdrat_mean': 0.9162987957268892, 'stdrat_median': 0.9081529327926331, 'stdrat_std': 0.1349019969860598, 'stdrat_iqr': 0.2145036267853485},
    'gp_hyperparameters|sine': {'logh1': 1.7615554901477346, 'logh2': -0.1494205488011105, 'logh3': -0.8151363981364801, 'nlml': 0.8426666769986556, 'stde': 0.4044015779952619, 'meanabs_std': 0.6506894808713993, 'std_mu_data': 0.7330377995313611, 'std_S_data': 0.008814747709975953, 'maxS': 0.5163416688164632, 'minS': 0.47462500928346174, 'meanS': 0.47678222482089005, 'oosmedabserr': NAN},
    'gp_fit_across|sine': {'stde': 0.9983348579260971, 'meanabs_std': 0.9910020985603706, 'stdmu': 5.835675140632057e-05, 'meanS': 0.867342720813748, 'stdS': 2.9160782297132985e-10, 'nlml': 1.4390936265738767, 'logh1': 2.756004594224246, 'logh2': -4.7231290819534815, 'logh3': -0.1423735748805146, 'h_lonN': 0.0524561403508772},
}


@pytest.mark.parametrize('key', list(EXPECTED))
def test_pinned_outputs(key):
    name, sname = key.split('|')
    got = CALLS[name](SERIES[sname])
    want = EXPECTED[key]
    if isinstance(want, dict):
        got_v = np.array([float(got[k]) for k in want])
        want_v = np.array(list(want.values()))
    else:
        got_v = np.ravel(np.asarray(got, dtype=float))[:len(want)]
        want_v = np.array(want)
    np.testing.assert_allclose(got_v, want_v, rtol=1e-12, atol=0, equal_nan=True)


def test_roundoff_guard_edges():
    r = np.random.default_rng(0)
    y = r.standard_normal(200)
    assert _is_roundoff(1e-15 * r.standard_normal(200), y)
    assert not _is_roundoff(1e-11 * r.standard_normal(200), y)
    assert _is_roundoff(np.full(50, 3.0), n=1000)  # ref defaults to x
    # exact fits and constant series give NaN through the guard
    assert np.all(np.isnan(bf_residual_stats(1e-15 * np.random.default_rng(1).standard_normal(200), y)))
    c = np.full(300, 2.0)
    assert all(np.isnan(v) for v in mf.gp_fit_across(c).values())
    assert all(np.isnan(v) for v in mf.gp_hyperparameters(c, 'covSEiso_covNoise', max_n=50,
                                                           resample_how='first').values())


def test_no_import_cycle():
    import ast
    import inspect
    src = inspect.getsource(mf)
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert not any(m and m.endswith('pre_process') for m in mods)
