"""Pinned outputs of the operations touched by the Theiler-window / delay-embedding refactor
(duplication audit themes #2, #3 and #18): `utils._theiler_int`, `utils.time_delay_embed` in place
of hand-written embeddings, and `_spectrum_stats` / `_enough_for_pca` / `_bf_embed_pair` in
`nonlinearity`. The values were generated from the main checkout (0fa9013) before the refactor,
so this checks that the refactor left every output unchanged (rtol 1e-12)."""
import logging
import warnings

import numpy as np
import pytest

from pyhctsa.operations import correlation as CO
from pyhctsa.operations import entropy as EN
from pyhctsa.operations import nonlinearity as NL

nan = float('nan')


def _series():
    e = np.random.default_rng(11).standard_normal(500)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    t = np.arange(500)
    sine = np.sin(2 * np.pi * t / 23) + 0.3 * np.random.default_rng(12).standard_normal(500)
    return {'ar1': ar, 'sine': sine}


CALLS = {
    'fnn': (NL.fnn, dict(tau=1, maxm=4)),
    'nlpe': (NL.nlpe, dict(de=3, tau=1)),
    'delay_time': (NL.delay_time, {}),
    'local_density': (NL.local_density, {}),
    'takens_estimator': (NL.takens_estimator, {}),
    'fractal_dimensions': (NL.fractal_dimensions, {}),
    'largest_lyap': (NL.largest_lyap, {}),
    'recurrence_times': (NL.recurrence_times, {}),
    'rqa': (NL.rqa, {}),
    'return_time': (NL.return_time, {}),
    'evt_local_dim': (NL.evt_local_dim, {}),
    'lyap_spec': (NL.lyap_spec, {}),
    'embed_pca': (NL.embed_pca, {}),
    'embed_kernel_pca': (NL.embed_kernel_pca, {}),
    'persistent_homology': (NL.persistent_homology, {}),
    'rpde': (EN.rpde, dict(m=3, tau='ac')),
    'approximate_entropy': (EN.approximate_entropy, dict(mnom=2, rth=0.1)),
    'randomize_ordinal': (EN.randomize_ordinal, {}),
    'dispersion_entropy': (EN.dispersion_entropy, dict(m=2, c=6, tau='ac')),
    'fuzzy_entropy': (EN.fuzzy_entropy, dict(M=3)),
    'time_rev_kld': (CO.time_rev_kld, {}),
    'embed2_shapes': (CO.embed2_shapes, {}),
    'joint_non_gaussianity': (CO.joint_non_gaussianity, {}),
}

# (operation, series) -> a scalar output, or the first (up to) four scalar outputs of a dict
EXPECTED = {
    ('approximate_entropy', 'ar1'): 0.9370175098654396,
    ('approximate_entropy', 'sine'): 0.9864007338153522,
    ('delay_time', 'ar1'): {'difftau12': 2.039235704929139, 'tau1': 0.050010132388752813, 'tau2': 2.089245837317892, 'tau3': 2.6790884677039517},
    ('delay_time', 'sine'): {'difftau12': 0.9861014589964531, 'tau1': 0.015166348296189354, 'tau2': 1.0012678072926424, 'tau3': 1.2002370335724786},
    ('dispersion_entropy', 'ar1'): {'dispEn': 3.5415193080485783, 'fDispEn': 2.2730817322391137, 'normDispEn': 0.9882797799791658, 'normFDispEn': 0.9479487106984459},
    ('dispersion_entropy', 'sine'): {'dispEn': 3.395493256727497, 'fDispEn': 2.3187573086705795, 'normDispEn': 0.9475304344813592, 'normFDispEn': 0.9669969055673411},
    ('embed2_shapes', 'ar1'): {'ac1': 0.5203403697743306, 'ac2': 0.32393509508503426, 'ac3': 0.2475065132483697, 'tau': 13.195694348902435},
    ('embed2_shapes', 'sine'): {'ac1': 0.05478835959974354, 'ac2': -0.01634471325594193, 'ac3': 0.04763173758261821, 'tau': 1.7702234333514308},
    ('embed_kernel_pca', 'ar1'): {'perc_1': 0.1902741061168778, 'perc_2': 0.1799070888296769, 'perc_3': 0.16902756202740368, 'std': 0.010624302225091238},
    ('embed_kernel_pca', 'sine'): {'perc_1': 0.3779238632835058, 'perc_2': 0.25474024790851924, 'perc_3': 0.10190638397604239, 'std': 0.1382739080862266},
    ('embed_pca', 'ar1'): {'perc_1': 0.3558439342765335, 'perc_2': 0.32659456060553865, 'perc_3': 0.3175615051179279, 'std': 0.020011106640527552},
    ('embed_pca', 'sine'): {'perc_1': 0.6193636655072488, 'perc_2': 0.3371491810421237, 'perc_3': 0.04348715345062746, 'std': 0.28795721870605623},
    ('evt_local_dim', 'ar1'): {'meanLocalDim': nan, 'meanTheta': nan, 'propValidPoles': nan, 'stdLocalDim': nan},
    ('evt_local_dim', 'sine'): {'meanLocalDim': nan, 'meanTheta': nan, 'propValidPoles': nan, 'stdLocalDim': nan},
    ('fnn', 'ar1'): 5.0,
    ('fnn', 'sine'): 5.0,
    ('fractal_dimensions', 'ar1'): {'maxDq': 6.892541471843439, 'maxq': 0.7596897056011878, 'meanDq': 5.865586674703977, 'rangeDq': 2.731254904816777},
    ('fractal_dimensions', 'sine'): {'maxDq': 5.500395117103076, 'maxq': 0.7754576365958549, 'meanDq': 4.899329677465131, 'rangeDq': 1.046892513676129},
    ('fuzzy_entropy', 'ar1'): {'fuzzyEn1': 1.28136700779501, 'fuzzyEn2': 1.590547722261595, 'fuzzyEn3': 1.403329901918295},
    ('fuzzy_entropy', 'sine'): {'fuzzyEn1': 1.1168510607643447, 'fuzzyEn2': 1.4211571744050793, 'fuzzyEn3': 1.2259497909483859},
    ('joint_non_gaussianity', 'ar1'): {'mahalKSstat': 0.020730520897730087, 'mardiaKurt': 7.82037151523612, 'mardiaSkew': -0.3523937878789187},
    ('joint_non_gaussianity', 'sine'): {'mahalKSstat': 0.2241811832186431, 'mardiaKurt': 5.074785312201711, 'mardiaSkew': -0.09024989483342846},
    ('largest_lyap', 'ar1'): {'p1': 0.0, 'p2': 1.042188610913818, 'p3': 1.3030459047831835, 'p4': 1.4004712028350212},
    ('largest_lyap', 'sine'): {'p1': 0.0, 'p2': 0.8783370073129559, 'p3': 0.8388101658326257, 'p4': 0.8309156659975376},
    ('local_density', 'ar1'): {'iqrden': 1.368382976129567, 'maxden': 0.5954242832274597, 'minden': -6.965886116479995, 'rangeden': 7.561310399707454},
    ('local_density', 'sine'): {'iqrden': 0.9686986047616193, 'maxden': 0.7663216788336564, 'minden': -5.430440225382411, 'rangeden': 6.196761904216068},
    ('lyap_spec', 'ar1'): {'LE1': -0.10526345793834326, 'LE2': -0.4191192936749948, 'LE3': -0.8702343292938036, 'sumPos': 0.0},
    ('lyap_spec', 'sine'): {'LE1': -0.17040694628124198, 'LE2': -0.42543878263906193, 'LE3': -0.8345847159425517, 'sumPos': 0.0},
    ('nlpe', 'ar1'): {'meanabs': 1.1009818609872208, 'meane': -0.06347264298001318, 'msqerr': 1.8634152276563145, 'stde': 1.3649671110528625},
    ('nlpe', 'sine'): {'meanabs': 0.4457343451246913, 'meane': 0.01146980442493018, 'msqerr': 0.31031063461704017, 'stde': 0.5574983760050419},
    ('persistent_homology', 'ar1'): {'maxPersistenceH1': 0.04460506621085445, 'persistenceEntropyH1': 5.102010515705922, 'totalPersistenceH0': 20.78245645848883, 'totalPersistenceH1': 2.3618143879703375},
    ('persistent_homology', 'sine'): {'maxPersistenceH1': 0.17066025107319538, 'persistenceEntropyH1': 4.805983636565744, 'totalPersistenceH0': 21.207483318857435, 'totalPersistenceH1': 2.0596891913713744},
    ('randomize_ordinal', 'ar1'): {'halftime': 0.144816627085592, 'normPermEn0': 0.9830429580112896, 'normPermEnInf': 0.9999966513340083, 'normPermEnRep1': 0.9993178194670023},
    ('randomize_ordinal', 'sine'): {'halftime': 0.1322315182872476, 'normPermEn0': 0.9790591326631329, 'normPermEnInf': 0.9999966513340083, 'normPermEnRep1': 0.9996694922024318},
    ('recurrence_times', 'ar1'): {'N_MPRT': 0.11349412840165951, 'N_MPRT_var': 0.00014479976344191782, 'T_MRT': 13.40334716264679, 'T_MRT_var': 2.452238253177699},
    ('recurrence_times', 'sine'): {'N_MPRT': 0.12948350071736012, 'N_MPRT_var': 0.0015032256819884132, 'T_MRT': 14.608895265423243, 'T_MRT_var': 3.285593836737224},
    ('return_time', 'ar1'): {'max': 2.875675775481676, 'pg05': 0.2, 'pzeros': 0.0, 'std': 0.6753581108007148},
    ('return_time', 'sine'): {'max': 9.646962235527266, 'pg05': 0.06, 'pzeros': 0.75, 'std': 2.00863949384222},
    ('rpde', 'ar1'): {'H': 1.7917594692280547, 'H_norm': 0.29002519869887244, 'meanNonZero': 80.33333333333333, 'propNonZero': 0.012448132780082987},
    ('rpde', 'sine'): {'H': 3.8682867025123455, 'H_norm': 0.6248933130054654, 'meanNonZero': 9.037037037037035, 'propNonZero': 0.11065573770491803},
    ('rqa', 'ar1'): {'DET': 0.7329540630041607, 'L_max': 15.0, 'L_mean': 2.981692573402418, 'RR': 0.09870428106875859},
    ('rqa', 'sine'): {'DET': 0.7682529062473865, 'L_max': 20.0, 'L_mean': 3.4430284857571216, 'RR': 0.09899326080837183},
    ('takens_estimator', 'ar1'): 3.8883988317429115,
    ('takens_estimator', 'sine'): 4.639253245381276,
    ('time_rev_kld', 'ar1'): {'abs': 0.07074833165221481, 'raw': 0.07074833165221481},
    ('time_rev_kld', 'sine'): {'abs': 0.03761991488637854, 'raw': 0.03761991488637854},
}


@pytest.mark.parametrize('op,sname', list(EXPECTED))
def test_pinned_output(op, sname):
    if op == 'persistent_homology':
        pytest.importorskip('ripser')
    func, kwargs = CALLS[op]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logging.disable(logging.CRITICAL)
        try:
            out = func(_series()[sname].copy(), **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    expected = EXPECTED[(op, sname)]
    if isinstance(expected, dict):
        got = [float(out[k]) for k in expected]
        expected = list(expected.values())
    else:
        got = float(out)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0, equal_nan=True)
