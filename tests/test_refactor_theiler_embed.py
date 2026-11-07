"""Checks for the Theiler-window / delay-embedding refactor (duplication audit themes #2, #3
and #18): `utils._theiler_int`, `utils.time_delay_embed` in place of hand-written embeddings, and
`_spectrum_stats` / `_enough_for_pca` / `_bf_embed_pair` in `nonlinearity`.

Two kinds of check:

1. Same-runner reference checks (the ones that matter): verbatim copies of the code the refactor
   replaced (`_ref_*`, copied from main 0fa9013, file and line given with each) against the new
   helpers, at rtol 1e-12 (or exact equality), on whatever machine runs the test.
2. Operation-level pins, generated on Ben's Mac (arm64) from the main checkout (0fa9013) before
   the refactor, with a tolerance per operation (`TOL`) set by what the operation computes, since
   other platforms reproduce them only to about 1e-12 to 1e-10 (BLAS and FFT summation order).
"""
import logging
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

from pyhctsa.operations import correlation as CO
from pyhctsa.operations import entropy as EN
from pyhctsa.operations import nonlinearity as NL
from pyhctsa.utils import _theiler_int, get_tau, theiler_window, time_delay_embed

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


# Per-operation tolerances for the pins (the values above are unchanged main-checkout values).
# Tier 1, rtol 1e-9: closed-form arithmetic (counts, sums, quantiles, entropies, ACF-based delays,
#   neighbor-distance statistics); CI drift on these is at most about 1e-11.
# Tier 2, rtol 1e-8: linear algebra and fitted slopes (PCA and kernel-PCA eigenvalues, covariance
#   inverses, QR-based Lyapunov spectra, least-squares scaling slopes).
T1, T2 = 1e-9, 1e-8
TOL = {
    'approximate_entropy': T1, 'delay_time': T1, 'dispersion_entropy': T1, 'fnn': T1,
    'fuzzy_entropy': T1, 'largest_lyap': T1, 'local_density': T1, 'nlpe': T1,
    'persistent_homology': T1, 'randomize_ordinal': T1, 'recurrence_times': T1,
    'return_time': T1, 'rpde': T1, 'rqa': T1, 'takens_estimator': T1, 'time_rev_kld': T1,
    'embed2_shapes': T1, 'evt_local_dim': T1,  # (all NaN on these series)
    'embed_pca': T2, 'embed_kernel_pca': T2,  # eigen-decompositions
    'joint_non_gaussianity': T2,  # Mahalanobis distances through a covariance inverse
    'lyap_spec': T2,  # QR re-orthonormalization of local Jacobians fitted by least squares
    'fractal_dimensions': T2,  # Dq are least-squares slopes over many neighbor distances
                               # (ubuntu and windows drift 8e-12 on 'sine'; up to 1.2e-10 seen)
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
    np.testing.assert_allclose(got, expected, rtol=TOL[op], atol=0, equal_nan=True)


# ---------------------------------------------------------------------------------------------
# Same-runner reference checks: the replaced code, copied verbatim from main 0fa9013
# ---------------------------------------------------------------------------------------------

def _ref_series():
    out = dict(_series())
    out['short'] = np.random.default_rng(13).standard_normal(60)
    out['rep'] = np.round(2 * np.random.default_rng(14).standard_normal(300)) / 2  # tied values
    out['const'] = np.ones(100)  # no ACF zero crossing: the NaN path
    return out


def _ref_theiler(y, spec, N=None):
    # The boilerplate repeated at the Theiler-window sites, e.g. main 0fa9013
    # pyhctsa/operations/nonlinearity.py:2110-2114 (fractal_dimensions):
    #     past = theiler_window(y, past)
    #     if np.isnan(past):
    #         logger.warning('No autocorrelation zero-crossing to set the Theiler window')
    #         return np.nan
    #     past = int(past)
    # (some sites, e.g. nonlinearity.py:141 fnn, skipped the final int()). NaN stands for the
    # early return.
    w = theiler_window(y, spec, N)
    if np.isnan(w):
        return np.nan
    return w


@pytest.mark.parametrize('spec', [('ac', 1), ['ac', 2], ('ac', 0.5), ('ac1e', 1), ('ac1e', 2.5),
                                  0, 1, 7, 7.0, 0.05, 0.3])
def test_ref_theiler_int(spec):
    for name, y in _ref_series().items():
        for N in (None, 50, len(y)):
            if N is None and y is None:
                continue
            old = _ref_theiler(y, spec, N)
            new = _theiler_int(y, spec, N, warn=False)
            if np.isnan(old):
                assert new is None, (name, spec, N)
            else:
                # the old value (int or integer-valued) and the new int are the same number,
                # so the sites that skipped int() see the same window too
                assert isinstance(new, int) and new == old and int(old) == new, (name, spec, N)


def _ref_embed_rpde(y, m, tau):
    # main 0fa9013 pyhctsa/operations/entropy.py:948-951 (rpde)
    n = len(y) - (m - 1) * tau
    return np.column_stack([y[k * tau:k * tau + n] for k in range(m)])


def _ref_embed_apen(x, length, tau):
    # main 0fa9013 pyhctsa/operations/entropy.py:1035-1036 (_apen_phis)
    n_vec = x.size - (length - 1) * tau
    return np.column_stack([x[j * tau:j * tau + n_vec] for j in range(length)])


def _ref_embed_ro(r, tau):
    # main 0fa9013 pyhctsa/operations/entropy.py:1584 (_ro_pattern_dists), with n = len(r)
    n = len(r)
    return np.column_stack([r[:n - 2 * tau], r[tau:n - tau], r[2 * tau:]])


def _ref_embed_disp(z, m, tau):
    # main 0fa9013 pyhctsa/operations/entropy.py:1877 and 1904 (dispersion_entropy)
    num_vectors = len(z) - (m - 1) * tau
    return z[np.arange(num_vectors)[:, None] + np.arange(m) * tau]


def _ref_embed_fuzzy(y, m, Nv):
    # main 0fa9013 pyhctsa/operations/entropy.py:2006 (fuzzy_entropy)
    return y[np.arange(Nv)[:, None] + np.arange(m)[None, :]]


def _ref_embed_lyap_spec(y, m, tau):
    # main 0fa9013 pyhctsa/operations/nonlinearity.py:4881 and 4886 (lyap_spec), with n = len(y)
    n_emb = len(y) - (m - 1) * tau
    return np.column_stack([y[j * tau:j * tau + n_emb] for j in range(m)])


def test_ref_delay_embeddings():
    for name, y in _ref_series().items():
        for m in (1, 2, 3, 5):
            for tau in (1, 2, 7):
                if len(y) - (m - 1) * tau < 2:
                    continue
                new = time_delay_embed(y, m, tau)
                for ref in (_ref_embed_rpde, _ref_embed_apen, _ref_embed_lyap_spec):
                    np.testing.assert_array_equal(new, ref(y, m, tau), err_msg=f'{ref.__name__} {name}')
                if m == 3:
                    r = np.unique(y, return_inverse=True)[1].astype(float) + 1.0
                    np.testing.assert_array_equal(time_delay_embed(r, 3, tau), _ref_embed_ro(r, tau))
                z = np.clip(np.floor(6 * (y - y.min()) / max(np.ptp(y), 1)) + 1, 1, 6).astype(np.int64)
                new_z = time_delay_embed(z, m, tau).astype(np.int64)
                old_z = _ref_embed_disp(z, m, tau)
                assert new_z.dtype == old_z.dtype
                np.testing.assert_array_equal(new_z, old_z)
        for M in (1, 2, 3):
            Nv = len(y) - M
            for m in range(1, M + 2):
                np.testing.assert_array_equal(time_delay_embed(y, m, 1)[:Nv], _ref_embed_fuzzy(y, m, Nv))


def _ref_embed_pca(y, tau='ac', m=3):
    # main 0fa9013 pyhctsa/operations/nonlinearity.py:885-929 (embed_pca body; warnings dropped)
    tau = get_tau(y, tau)
    if np.isnan(tau):
        return np.nan
    try:
        y_embed = time_delay_embed(y, m, int(tau))
    except ValueError:
        return np.nan
    if y_embed.shape[0] - 1 < m or m < 2:
        return np.nan
    pca = PCA().fit(y_embed)
    perc = pca.explained_variance_ / np.sum(pca.explained_variance_)
    _first_fn = NL._first_fn
    out = {}
    for i in range(m):
        out[f'perc_{i+1}'] = perc[i]
    out['std'] = np.std(perc, ddof=1)
    out['range'] = np.ptp(perc)
    out['min'] = np.min(perc)
    out['max'] = np.max(perc)
    out['top2'] = np.sum(perc[:2])
    csperc = np.cumsum(perc)
    out['nto50'] = _first_fn(csperc, 0.5, 'over')
    out['nto60'] = _first_fn(csperc, 0.6, 'over')
    out['nto70'] = _first_fn(csperc, 0.7, 'over')
    out['nto80'] = _first_fn(csperc, 0.8, 'over')
    out['nto90'] = _first_fn(csperc, 0.9, 'over')
    out['fb05'] = _first_fn(perc, 0.5, 'under')
    out['fb02'] = _first_fn(perc, 0.2, 'under')
    out['fb01'] = _first_fn(perc, 0.1, 'under')
    out['fb001'] = _first_fn(perc, 0.01, 'under')
    return out


@pytest.mark.parametrize('tau,m', [('ac', 2), ('ac', 3), (1, 4), (3, 5), ('ac', 1), (1, 40)])
def test_ref_embed_pca(tau, m):
    for name, y in _ref_series().items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logging.disable(logging.CRITICAL)
            try:
                old = _ref_embed_pca(y.copy(), tau, m)
                new = NL.embed_pca(y.copy(), tau, m)
            finally:
                logging.disable(logging.NOTSET)
        if not isinstance(old, dict):  # NaN (`dict_output` may expand it to a dict of NaNs)
            vals = list(new.values()) if isinstance(new, dict) else [new]
            assert np.all(np.isnan(np.asarray(vals, dtype=float))), (name, tau, m)
            continue
        assert list(new) == list(old), (name, tau, m)
        np.testing.assert_allclose([float(new[k]) for k in old], [float(v) for v in old.values()],
                                   rtol=1e-12, atol=0, err_msg=f'{name} tau={tau} m={m}')


def _ref_embed_pair(y, embed_params):
    # main 0fa9013 pyhctsa/operations/nonlinearity.py:262-265 (_embed_tau_m) and 1883-1891
    # (takens_estimator; fractal_dimensions at 2117-2125 is the same up to its log message)
    if not isinstance(embed_params, (list, tuple)) or len(embed_params) != 2:
        raise ValueError('Embedding parameters are formatted incorrectly -- need [tau, m]')
    params = NL._embedding_params(y, embed_params[0], embed_params[1])
    tau, m = (np.nan, np.nan) if params is None else params
    if np.isnan(tau):
        return None
    try:
        return time_delay_embed(y, m, tau)
    except ValueError:
        return None


@pytest.mark.parametrize('embed_params', [('ac', 3), [1, 2], (2, 5), ('ac', 'fnn'), (1, 200)])
def test_ref_embed_pair(embed_params):
    for name, y in _ref_series().items():
        logging.disable(logging.CRITICAL)
        try:
            old = _ref_embed_pair(y, embed_params)
            new = NL._bf_embed_pair(y, embed_params)
        finally:
            logging.disable(logging.NOTSET)
        if old is None:
            assert new is None, (name, embed_params)
        else:
            np.testing.assert_array_equal(new, old, err_msg=f'{name} {embed_params}')
