"""Pinned outputs of the operations touched by the spectral analytic-band refactor.

The analytic signal, usable-bin selection, dominant-band selection and max_n cropping of
spectral.py were factored into module-private helpers, _padded_nfft moved to utils (and is used
by medical.hrv_classic and nonlinearity._period_normalized_tau), and nonlinearity's inline max_n
crops now call _check_max_n. None of this may change an output.

Two kinds of check:

1. Same-runner reference (the equivalence check): verbatim copies of the inline code the helpers
   replaced (from main 0fa9013) are compared with the helpers at rtol 1e-12, atol 0, on whatever
   machine runs the test, so the check carries no platform dependence.
2. Operation-level pins: the values below were generated from the main checkout (0fa9013) before
   the refactor, on one machine (arm64 macOS). Other BLAS/FFT builds reproduce them only to about
   1e-12, so each operation has a tolerance set by the kind of computation (see TOLERANCES).
"""
import logging
import warnings

import numpy as np
import pytest
import scipy.fft

from pyhctsa.operations import medical as md
from pyhctsa.operations import nonlinearity as nl
from pyhctsa.operations import spectral as sp


def _series():
    rng = np.random.default_rng(7)
    n = 1200
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    sin = np.sin(2 * np.pi * np.arange(n + 1) / 23.7) + 0.4 * rng.standard_normal(n + 1)  # odd N
    return {'ar1': ar, 'sin_odd': sin}


SERIES = _series()

CALLS = {
    'phase_amp_coupling': (sp.cross_frequency_coupling, {}),
    'phase_amp_coupling_max_n': (sp.cross_frequency_coupling, {'max_n': 700, 'n_bands': 4}),
    'envelope_stats': (sp.envelope_stats, {}),
    'phase_fluctuation_scaling': (sp.phase_fluctuation_scaling, {'max_n': 900}),
    'bicoherence': (sp.bicoherence, {'max_n': 800}),
    'spectral_summaries_welch': (sp.spectral_summaries, {'psd_meth': 'welch'}),
    'hrv_classic': (md.hrv_classic, {}),
    'zero_one_test': (nl.zero_one_test, {'max_n': 600}),
    'nlpe': (nl.nlpe, {'max_n': 500}),
    'embed_cluster': (nl.embed_cluster, {'max_n': 300}),
    'embed_kernel_pca': (nl.embed_kernel_pca, {'max_n': 300}),
}

PINNED = {
    'phase_amp_coupling': {
        'ar1': {
            'maxMI': 0.0019463062346649022,
            'entropyMI': 0.928498802527566,
        },
        'sin_odd': {
            'maxMI': 0.0008695159348339754,
            'entropyMI': 0.9771054167147875,
        },
    },
    'phase_amp_coupling_max_n': {
        'ar1': {
            'maxMI': 0.0033204237143018823,
            'entropyMI': 0.9010631302983428,
        },
        'sin_odd': {
            'maxMI': 0.001199950034117532,
            'entropyMI': 0.9666245704500985,
        },
    },
    'envelope_stats': {
        'ar1': {
            'full_cv': 0.5246614478475472,
            'full_kurt': 3.072060821424426,
            'dom_cv': 0.5488229077973847,
            'dom_kurt': 2.582744449508469,
            'dom_ifspread': 0.006855560811158193,
        },
        'sin_odd': {
            'full_cv': 0.35638175239959974,
            'full_kurt': 2.899025885040078,
            'dom_cv': 0.08218266644905439,
            'dom_kurt': 5.313460719412068,
            'dom_ifspread': 0.00026455008098093616,
        },
    },
    'phase_fluctuation_scaling': {
        'ar1': {
            'meanFreq': 0.027943646441605623,
            'slope_short': 1.0057981579291562,
            'slope_long': 0.8058075799704743,
            'slope_diff': 0.19999057795868191,
        },
        'sin_odd': {
            'meanFreq': 0.04220170257008161,
            'slope_short': 0.991060325269172,
            'slope_long': 0.6102606944094591,
            'slope_diff': 0.3807996308597129,
        },
    },
    'bicoherence': {
        'ar1': {
            'meanBic': 0.045186423892335216,
            'maxBic': 0.23967121389081927,
            'skewBic': 1.5876372238264476,
            'meanBicDiag': 0.031573921076840955,
            'threshRatio': 0.995615912796739,
        },
        'sin_odd': {
            'meanBic': 0.04296040859262361,
            'maxBic': 0.4086824169286797,
            'skewBic': 2.83244350528225,
            'meanBicDiag': 0.023280156353465347,
            'threshRatio': 0.9909822547782275,
        },
    },
    'spectral_summaries_welch': {
        'ar1': {
            'maxS': 13.643921523814466,
            'logstd': 1.389451831981044,
            'fpoly2_r2': 0.8239741667739016,
            'linfitsemilog_all_sigma': 0.6157699250328597,
            'ncross_log_f50': 9.0,
        },
        'sin_odd': {
            'maxS': 14.707338053612684,
            'logstd': 0.8170941202649162,
            'fpoly2_r2': 0.654978995748215,
            'linfitsemilog_all_sigma': 0.4304900808864068,
            'ncross_log_f50': 2.0,
        },
    },
    'hrv_classic': {
        'ar1': {
            'pnnrel025': 0.8015012510425354,
            'pnnrel1': 0.3219349457881568,
            'lfhf': 0.5382233231864144,
            'hf': 33.36432224649133,
            'SD2': 2300.3659571954777,
        },
        'sin_odd': {
            'pnnrel025': 0.7908333333333334,
            'pnnrel1': 0.33166666666666667,
            'lfhf': 0.013843261662781424,
            'hf': 79.10401558546683,
            'SD2': 1094.20628317549,
        },
    },
    'zero_one_test': {
        'ar1': {
            'K': 0.9970494944460515,
            'Kstd': 0.0061092897482743925,
            'D': 0.5028596145899138,
            'Dstd': 0.7258279863282748,
        },
        'sin_odd': {
            'K': 0.990164685946041,
            'Kstd': 0.09299999307238939,
            'D': 0.16321255656806943,
            'Dstd': 0.04485092426252303,
        },
    },
    'nlpe': {
        'ar1': {
            'msqerr': 1.9221495919908007,
            'stde': 1.386604430175024,
            'ac3': 0.009024102742685964,
            'swm': 0.12244984454334751,
            'minsbc': 0.013230449339834749,
        },
        'sin_odd': {
            'msqerr': 0.593816823224132,
            'stde': 0.7713595763787796,
            'ac3': 0.04447896565328907,
            'swm': 0.11574162331978662,
            'minsbc': 0.020145685381460332,
        },
    },
    'embed_cluster': {
        'ar1': {
            'sep_pp': 0.6995662865272382,
            'sep_mahal': 0.9324923307820927,
            'valley': 0.0,
            'nongauss': 0.14591679817578074,
        },
        'sin_odd': {
            'sep_pp': 0.7277188301930316,
            'sep_mahal': 1.3099115233799308,
            'valley': 0.06815496129737497,
            'nongauss': 0.16254122461134957,
        },
    },
    'embed_kernel_pca': {
        'ar1': {
            'perc_1': 0.20742145426063172,
            'min': 0.15750470315035303,
            'nto80': 8.0,
            'fb001': 15.0,
            'std_ratio': 0.6067547691577171,
        },
        'sin_odd': {
            'perc_1': 0.31475794474870256,
            'min': 0.10179135987260728,
            'nto80': 6.0,
            'fb001': 12.0,
            'std_ratio': 0.44042472812291966,
        },
    },
    '_period_normalized_tau': {
        'ar1': 1.0,
        'sin_odd': 5.0,
    },
}


# ------------------------------------------------------------------------------
# 1. Same-runner reference: verbatim copies of the pre-refactor inline code
# ------------------------------------------------------------------------------

def _ref_envelope_stats(y, power_frac=0.5, trim_frac=0.05):
    # Verbatim body of envelope_stats from main 0fa9013, pyhctsa/operations/spectral.py lines
    # 1585-1635 (inline usable bins, analytic signal and dominant band).
    _envelope_summary = sp._envelope_summary
    y = np.asarray(y, dtype=float).ravel()
    out = {f: np.nan for f in ('full_cv', 'full_kurt', 'full_tau', 'dom_cv', 'dom_skew',
                               'dom_kurt', 'dom_tau', 'dom_ifspread')}
    N = len(y)
    if N < 50 or not np.all(np.isfinite(y)) or np.std(y, ddof=1) == 0:
        return out
    y = y - np.mean(y)

    # Frequency bins (DC and Nyquist excluded, as in phase_amp_coupling):
    half_n = N // 2 + 1
    usable = np.arange(1, half_n - 1 if N % 2 == 0 else half_n)
    Y = np.fft.fft(y)

    # Samples dropped from each end of the (circularly computed) analytic signal:
    n_trim = max(1, int(np.floor(trim_frac * N + 0.5)))
    keep = slice(n_trim, N - n_trim)

    # (a) Full band
    Yfull = np.zeros(N, dtype=complex)
    Yfull[usable] = 2 * Y[usable]
    env_full = np.abs(np.fft.ifft(Yfull)[keep])
    out['full_cv'], _, out['full_kurt'], out['full_tau'] = _envelope_summary(env_full)

    # (b) Dominant band: the largest periodogram peak +/- the smallest half-width
    # (at least 2 bins) holding power_frac of the power
    pw = np.abs(Y[usable]) ** 2
    n_bins = len(usable)
    i_peak = int(np.argmax(pw))
    cum_pow = np.concatenate(([0.0], np.cumsum(pw)))
    hws = np.arange(2, n_bins + 1)
    band_pow = (cum_pow[np.minimum(n_bins - 1, i_peak + hws) + 1]
                - cum_pow[np.maximum(0, i_peak - hws)])
    ok = np.flatnonzero(band_pow >= power_frac * cum_pow[-1])
    half_width = int(hws[ok[0]]) if ok.size else n_bins
    peak_bin = usable[i_peak]
    band = np.arange(max(usable[0], peak_bin - half_width),
                     min(usable[-1], peak_bin + half_width) + 1)

    Ydom = np.zeros(N, dtype=complex)
    Ydom[band] = 2 * Y[band]
    z_dom = np.fft.ifft(Ydom)
    env_dom = np.abs(z_dom[keep])
    out['dom_cv'], out['dom_skew'], out['dom_kurt'], out['dom_tau'] = _envelope_summary(env_dom)

    # Instantaneous frequency (cycles per sample): the unwrapped phase increments
    # over the trimmed segment, summarized robustly (scaled MAD, which equals the
    # standard deviation for a Gaussian)
    phi = np.unwrap(np.angle(z_dom[keep]))
    inst_freq = np.diff(phi) / (2 * np.pi)
    out['dom_ifspread'] = 1.4826 * np.median(np.abs(inst_freq - np.median(inst_freq)))
    return out


def _ref_pac_band_signals(y, band_bins):
    # Verbatim from phase_amp_coupling, main 0fa9013, spectral.py lines 966-971 (scipy.fft pair).
    n = len(y)
    y_fft = scipy.fft.fft(y)
    phase_band, amp_band = [], []
    for b in range(len(band_bins)):
        yb = np.zeros(n, dtype=complex)
        yb[band_bins[b]] = 2 * y_fft[band_bins[b]]
        zb = scipy.fft.ifft(yb)
        phase_band.append(np.angle(zb))
        amp_band.append(np.abs(zb))
    return np.array(phase_band), np.array(amp_band)


def _ref_pac_usable_bins(n):
    # Verbatim from phase_amp_coupling, main 0fa9013, spectral.py lines 946-951.
    half_n = n // 2 + 1 # one-sided bins: DC (0) up to Nyquist (half_n - 1, if n even)
    is_nyquist_bin = (n % 2 == 0)
    if is_nyquist_bin:
        usable_bins = np.arange(1, half_n - 1)
    else:
        usable_bins = np.arange(1, half_n)
    return usable_bins


def _ref_pfs_phase(y, half_width_bins):
    # Verbatim from phase_fluctuation_scaling, main 0fa9013, spectral.py lines 1716-1732
    # (dominant band and its unwrapped analytic phase; the length guard is left to the caller).
    N = len(y)
    half_n = N // 2 + 1
    usable = np.arange(1, half_n - 1 if N % 2 == 0 else half_n)
    Y = np.fft.fft(y)
    power = np.abs(Y[usable]) ** 2
    peak_bin = usable[int(np.argmax(power))]
    band = np.arange(max(usable[0], peak_bin - half_width_bins),
                     min(usable[-1], peak_bin + half_width_bins) + 1)
    Yband = np.zeros(N, dtype=complex)
    Yband[band] = 2 * Y[band]
    phi = np.unwrap(np.angle(np.fft.ifft(Yband)))
    return phi


def _ref_padded_nfft(n):
    # Verbatim from medical.hrv_classic, main 0fa9013, medical.py line 183.
    return max(256, 2 ** int(np.ceil(np.log2((n)))))


def _reference_corpus():
    # Seeded series of even and odd length (the pinned SERIES plus white noise and a short sine).
    rng = np.random.default_rng(11)
    extra = {'wn_even': rng.standard_normal(1000),
             'sine_short_odd': np.sin(np.arange(301) / 3.1) + 0.2 * rng.standard_normal(301)}
    return {**SERIES, **extra}


REF_SERIES = _reference_corpus()


@pytest.mark.parametrize('name', list(REF_SERIES))
def test_envelope_stats_matches_reference(name):
    y = REF_SERIES[name]
    out, ref = sp.envelope_stats(y), _ref_envelope_stats(y)
    for key, expected in ref.items():
        np.testing.assert_allclose(out[key], expected, rtol=1e-12, atol=0, err_msg=key)


@pytest.mark.parametrize('name', list(REF_SERIES))
def test_band_helpers_match_reference(name):
    y = REF_SERIES[name]
    n = len(y)
    usable = sp._usable_bins(n)
    np.testing.assert_array_equal(usable, _ref_pac_usable_bins(n))
    # phase_amp_coupling's per-band analytic signals (scipy.fft pair), 5 equal-width bands:
    edges = sp._even_split_points(0, len(usable), 6)
    band_bins = [usable[edges[b]:edges[b + 1]] for b in range(5)]
    y_fft = scipy.fft.fft(y)
    z = [sp._analytic_signal_fft(y_fft, bb, ifft=scipy.fft.ifft) for bb in band_bins]
    ref_phase, ref_amp = _ref_pac_band_signals(y, band_bins)
    np.testing.assert_allclose(np.angle(z), ref_phase, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.abs(z), ref_amp, rtol=1e-12, atol=0)
    # phase_fluctuation_scaling's dominant band and unwrapped phase (numpy FFT pair):
    for hw in (2, 5):
        Y = np.fft.fft(y)
        band = sp._dominant_band(np.abs(Y[usable]) ** 2, usable, hw)
        phi = np.unwrap(np.angle(sp._analytic_signal_fft(Y, band)))
        np.testing.assert_allclose(phi, _ref_pfs_phase(y, hw), rtol=1e-12, atol=0)
    assert sp._padded_nfft(n) == _ref_padded_nfft(n)


# ------------------------------------------------------------------------------
# 2. Operation-level pins (main-checkout values) with per-operation tolerances
# ------------------------------------------------------------------------------

# Tiers: closed-form arithmetic (FFT, sums, histograms, quantiles, entropies) rtol 1e-9, since
# BLAS/FFT ordering moves these by at most ~1e-11 across runners (dom_ifspread drifted 3e-12 on
# ubuntu); linear algebra and least squares rtol 1e-8; iterative optimizers rtol 1e-5. Counts
# (ncross, nto80, fb001) are compared at the tier of their operation and are exact in practice.
TIER_CLOSED, TIER_LINALG, TIER_ITER = 1e-9, 1e-8, 1e-5
TOLERANCES = {
    # FFT analytic signals and histogram-based modulation index:
    'phase_amp_coupling': TIER_CLOSED,
    'phase_amp_coupling_max_n': TIER_CLOSED,
    # FFT envelope, moments, ACF timescale, MAD of instantaneous frequency:
    'envelope_stats': TIER_CLOSED,
    # meanFreq is FFT arithmetic; the slopes are lstsq fits in log-log space:
    'phase_fluctuation_scaling': TIER_LINALG,
    # segment FFT triple products and their averages:
    'bicoherence': TIER_CLOSED,
    # maxS/logstd are arithmetic on the Welch PSD; fpoly2_r2 and linfitsemilog are polynomial fits:
    'spectral_summaries_welch': TIER_LINALG,
    # periodogram band powers, relative-difference fractions and Poincare SD2:
    'hrv_classic': TIER_CLOSED,
    # cumulative sums, mean-square displacement and correlation coefficients:
    'zero_one_test': TIER_CLOSED,
    # nearest-neighbor prediction errors and residual autocorrelation:
    'nlpe': TIER_CLOSED,
    # eigh of the embedding covariance and projections (sep_mahal: see KEY_TOLERANCES):
    'embed_cluster': TIER_LINALG,
    # eigvalsh of the centered kernel matrix and linear PCA:
    'embed_kernel_pca': TIER_LINALG,
}
# Outputs whose tier differs from their operation's:
KEY_TOLERANCES = {
    # means and pooled covariance of an EM 2-component Gaussian mixture fit (iterative optimizer):
    ('embed_cluster', 'sep_mahal'): TIER_ITER,
}


@pytest.mark.parametrize('op', list(CALLS))
@pytest.mark.parametrize('name', list(SERIES))
def test_pinned_outputs(op, name):
    f, kwargs = CALLS[op]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = f(SERIES[name], **kwargs)
    for key, expected in PINNED[op][name].items():
        np.testing.assert_allclose(out[key], expected, rtol=KEY_TOLERANCES.get((op, key), TOLERANCES[op]), atol=0,
                                   err_msg=f'{op}.{key}')


@pytest.mark.parametrize('name', list(SERIES))
def test_period_normalized_tau(name):
    assert nl._period_normalized_tau(SERIES[name]) == PINNED['_period_normalized_tau'][name]


def test_crop_messages(caplog):
    y = SERIES['ar1']
    with pytest.warns(UserWarning, match=r'^Time series \(1200 samples\) exceeds max_n = 700; '
                                         r'analyzing the first 700 samples$'):
        sp.cross_frequency_coupling(y, max_n=700)
    with caplog.at_level(logging.INFO, logger='pyhctsa'):
        nl.nlpe(y, max_n=500)
        nl.zero_one_test(y, max_n=600)
        nl.embed_kernel_pca(y, max_n=300)
    messages = [r.getMessage() for r in caplog.records]
    assert "Michael Small's nlpe code is only being evaluated on the first 500 (/1200) samples." in messages
    assert 'Time series (1200 samples) exceeds maxN=600; analyzing the first 600 samples.' in messages
    assert any(m.startswith('Cropping to the first 300 of ') and m.endswith('for kernel PCA (memory/time '
               'cap, not a convergence point)') for m in messages)
    with pytest.raises(ValueError, match="maxN must be an integer or 'full'"):
        nl.zero_one_test(y, max_n='half')
    with pytest.raises(ValueError, match="max_n must be an integer or 'full', got 'half'"):
        nl.embed_cluster(y, max_n='half')


def test_spectral_helpers():
    assert list(sp._usable_bins(8)) == [1, 2, 3]
    assert list(sp._usable_bins(9)) == [1, 2, 3, 4]
    assert sp._padded_nfft(257) == 512 and sp._MIN_NFFT == 256
