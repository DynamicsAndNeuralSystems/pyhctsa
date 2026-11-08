import numpy as np
from ..utils import _quantile
from numpy.typing import ArrayLike
from scipy import signal

from ..utils import _run_lengths
from ..robust import bf_hist_edges
from ..utils import MAD_DIVISOR, get_tau, _padded_nfft

def raw_hrv_meas(x: ArrayLike) -> dict:
    """
    Compute Poincaré plot-based HRV (Heart Rate Variability) measures from RR interval time series.

    This function computes the triangular histogram indices and Poincaré plot measures commonly used 
    in HRV analysis. It is specifically designed for time series consisting of consecutive 
    RR intervals measured in seconds. It is not suitable for other types of time series.

    The computed features are widely used in clinical and physiological studies of autonomic nervous 
    system activity. The Poincaré plot measures (SD1 and SD2) are standard metrics for short- and 
    long-term variability, while the triangular indices provide geometric summaries of the RR 
    distribution.

    Port of hctsa's ``MD_RawHRVMeas``, Copyright (C) 2016 Max A. Little and Ben D. Fulcher (GNU
    GPL, version 3 or later).

    References
    ----------
    .. [1] M. Brennan, M. Palaniswami, and P. Kamen, 
        "Do existing measures of Poincaré plot geometry reflect nonlinear features 
        of heart rate variability?", IEEE Transactions on Biomedical Engineering, 
        48(11), pp. 1342–1347, 2001.
    .. [2] Original MATLAB implementation adapted from: Max Little's `hrv_classic.m` 
        (http://www.maxlittle.net/)

    Parameters
    ----------
    x : array-like
        Time series of RR intervals in seconds.

    Returns
    -------
    out : dict
        Dictionary containing the following HRV features (SD1 and SD2 are in milliseconds,
        the series being multiplied by 1000 to convert from seconds)

        - 'tri10'   : Triangular histogram index using 10 bins.
        - 'tri20'   : Triangular histogram index using 20 bins.
        - 'trisqrt' : Triangular histogram index using a number of bins determined by 
                the square root rule.
        - 'SD1'     : Standard deviation of the Poincaré plot’s minor axis (short-term variability).
        - 'SD2'     : Standard deviation of the Poincaré plot’s major axis (long-term variability).
        - 'CVI'     : Cardiac vagal index, log10(16*SD1*SD2) [3].

    References (CVI)
    ----------------
    .. [3] Toichi et al., "A new method of assessing cardiac autonomic function and its
        comparison with spectral analysis and coefficient of variation of R-R interval",
        J. Auton. Nerv. Syst. 62(1-2), 79 (1997).
    """
    x = np.asarray(x)
    N = len(x)
    out = {}

    # triangular histogram index
    # 10 bins
    out['tri10'] = N/np.max(np.histogram(x, bins=bf_hist_edges(x, 10))[0])

    # 20 bins
    out['tri20'] = N/np.max(np.histogram(x, bins=bf_hist_edges(x, 20))[0])

    # (sqrt samples) bins
    out['trisqrt'] = N/np.max(np.histogram(x, bins=bf_hist_edges(x, 'sqrt'))[0])

    # Poincare plot measures
    diff_x = np.diff(x)
    sd_diff = np.std(diff_x, ddof=1)
    out['SD1'] = 1/np.sqrt(2) * sd_diff * 1000
    out['SD2'] = np.sqrt(2 * np.var(x, ddof=1) - (1/2) * sd_diff**2) * 1000

    # CVI: cardiac vagal index (Toichi et al., 1997)
    out['CVI'] = np.log10(out['SD1'] * out['SD2'] * 16)

    return out

def hrv_classic(y: ArrayLike) -> dict:
    """
    Compute classic heart rate variability (HRV) statistics.

    This function computes a variety of standard time-domain, frequency-domain, and
    geometric HRV measures from a time series of RR (or NN) intervals. The input is
    typically assumed to be in **seconds**.

    The following categories of HRV features are included:

    1. **pNNx-style measures (pnnrel025, pnnrel05, pnnrel1, pnnrel2, pnnrel3)**
    The proportion of successive differences whose magnitude exceeds 0.25, 0.5, 1, 2 and 3
    times the robust standard deviation of the successive differences,
    ``sigD = median(|d - median(d)|)/0.6745`` with ``d = diff(y)`` [1]. If ``sigD`` is
    at rounding-error level (``<= 1e-10*std(y)``, i.e. more than half the increments are
    equal) the mean absolute deviation about the median times ``sqrt(pi/2)`` is used
    instead; if that is also negligible (all increments equal) all five are NaN. Thresholds
    relative to the spread of the increments are used because fixed thresholds (such as
    pNN5 to pNN40, in the units of the series) give proportions close to 1 for almost every
    z-scored series.

    2. **Frequency-domain measures**
    The ratio of low- to high-frequency power (lfhf) and the percentages of total power in the
    very-low-, low- and high-frequency bands (vlf, lf, hf) of a Hann-windowed periodogram [2].
    The periodogram is computed with ``fs = 2*pi``, so the bands (VLF below 0.04, LF 0.04 to
    0.15, HF 0.15 to 0.4) are in radians per sample, not in hertz.

    3. **Triangular histogram index**
    A geometric measure of HRV based on the shape of the RR interval histogram.

    4. **Poincaré plot measures (SD1, SD2)**
    Geometric descriptors of the Poincaré plot reflecting short- and long-term variability [3]. 

    This implementation is adapted from original MATLAB code by Max A. Little
    (http://www.maxlittle.net/).

    Port of hctsa's ``MD_hrv_classic``, Copyright (C) Ben D. Fulcher (GNU GPL, version 3 or
    later), heavily derived from code provided by Max A. Little.

    References
    ----------
    .. [1] Mietus, J.E., et al., *The pNNx files: Re-examining a widely used 
        heart rate variability measure*, Heart, 88(4):378, 2002.
    .. [2] Malik, M., et al., *Heart rate variability: Standards of measurement, 
        physiological interpretation, and clinical use*, European Heart Journal, 17(3):354, 1996.
    .. [3] Brennan, M., et al., *Do existing measures of Poincaré plot geometry 
        reflect nonlinear features of heart rate variability?*, IEEE Transactions on 
        Biomedical Engineering, 48(11):1342, 2001.

    Parameters
    ----------
    y : array-like
        Input time series of RR intervals, assumed to be in seconds.

    Returns
    -------
    out: dict
        Dictionary containing various HRV features: pnnrel025, pnnrel05, pnnrel1,
        pnnrel2, pnnrel3 (pNNx-style statistics relative to a robust SD of the
        increments), frequency-domain power ratios (lfhf, vlf, lf, hf), triangular
        index (tri), and Poincaré measures (SD1, SD2).
    """

    # Standard defaults
    y = np.asarray(y)
    diff_y = np.diff(y)
    n = len(y)

    # ------------------------------------------------------------------------------
    # Calculate pNNx: proportion of |successive differences| exceeding c robust SDs
    # ------------------------------------------------------------------------------
    # cf. Mietus et. al. 2002, "The pNNx files: ...", Heart. The fixed thresholds x/1000
    # are replaced by multiples of the robust SD of the increments, so the measure
    # does not depend on the units of the series.
    d_y = np.abs(diff_y)
    med = np.median(diff_y)
    sig_d = np.median(np.abs(diff_y - med)) / MAD_DIVISOR  # robust (MAD-based) SD of increments
    tol_d = 1e-10 * np.std(y, ddof=1)  # a spread below this (rounding error) is treated as zero
    if sig_d <= tol_d:
        # over half the increments are equal: fall back to the mean absolute deviation
        # about the median (consistent with the SD for Gaussian increments)
        sig_d = np.mean(np.abs(diff_y - med)) * np.sqrt(np.pi / 2)
    if sig_d <= tol_d:
        sig_d = np.nan  # all increments equal

    def pnn_rel_fn(c):
        # (a comparison with NaN would otherwise give 0)
        return np.nan if np.isnan(sig_d) else float(np.mean(d_y > c * sig_d))

    out = {}
    out['pnnrel025'] = pnn_rel_fn(0.25)
    out['pnnrel05'] = pnn_rel_fn(0.5)
    out['pnnrel1'] = pnn_rel_fn(1)
    out['pnnrel2'] = pnn_rel_fn(2)
    out['pnnrel3'] = pnn_rel_fn(3)

    # ------------------------------------------------------------------------------
    # Calculate PSD
    # ------------------------------------------------------------------------------

    nfft = _padded_nfft(n)
    f, pxx = signal.periodogram(
        y,
        window=np.hanning(len(y)),
        detrend=False,
        scaling='density',
        fs=2 * np.pi,
        nfft=nfft
    )

    # Calculate spectral measures such as subband spectral power percentage, LF/HF ratio etc.
    lf_lo = 0.04  # /pi -- fraction of total power (max F is pi)
    lf_hi = 0.15
    hf_lo = 0.15
    hf_hi = 0.4

    f_bin_size = f[1] - f[0]

    i_lf = np.searchsorted(f, lf_lo, side='left')
    j_lf = np.searchsorted(f, lf_hi, side='right')
    i_hf = np.searchsorted(f, hf_lo, side='left')
    j_hf = np.searchsorted(f, hf_hi, side='right')
    j_vlf = np.searchsorted(f, lf_lo, side='right')

    lf_p = f_bin_size * np.sum(pxx[i_lf:j_lf])
    hf_p = f_bin_size * np.sum(pxx[i_hf:j_hf])
    vlf_p = f_bin_size * np.sum(pxx[:j_vlf])

    out['lfhf'] = lf_p / hf_p
    total = f_bin_size * np.sum(pxx)
    out['vlf'] = vlf_p / total * 100
    out['lf'] = lf_p / total * 100
    out['hf'] = hf_p / total * 100

    # Triangular histogram index
    # (equal-width bins spanning the data, with explicit edges)
    out['tri'] = len(y) / np.max(np.histogram(y, bins=bf_hist_edges(y, 10))[0])

    # Poincare plot measures:
    # cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    rmssd = np.std(diff_y, ddof=1)
    sigma = np.std(y, ddof=1)

    out["SD1"] = 1 / np.sqrt(2) * rmssd * 1000
    out["SD2"] = np.sqrt(2 * sigma**2 - (1 / 2) * rmssd**2) * 1000

    return out

def pol_var(x: ArrayLike, d: float = 1, D: int = 6, tau: int | str = 1) -> float:
    """
    Compute the POLVARd measure of a time series.

    The POLVARd (also called Plvar) measure quantifies the probability of 
    obtaining a sequence of consecutive ones or zeros in a symbolic sequence 
    derived from the input time series.

    This measure was originally introduced in [1].

    The original implementation applied this measure to RR interval sequences 
    (typically in milliseconds), with the symbolic threshold `d` representing 
    raw amplitude differences. This implementation generalizes it to 
    z-scored time series, such that `d` is specified in units of standard deviation.

    The function is derived from the MATLAB implementation by Max A. Little 
    (2009) and Ben D. Fulcher.

    Port of hctsa's ``MD_PolVar``, Copyright (C) 2016 Max A. Little and Ben D. Fulcher (GNU GPL,
    version 3 or later).

    References
    ----------
    .. [1] Wessel et al., "Short-term forecasting of life-threatening cardiac 
            arrhythmias based on symbolic dynamics and finite-time growth rates",
            Phys. Rev. E 61(1), 733 (2000).

    Parameters
    ----------
    x : array-like
        The input time series.
    d : float
        Symbolic coding threshold in units of standard deviation. Default is 1.
    D : int
        Word length for detecting consecutive sequences. Default is 6.
    tau : int or str, optional
        The step, in samples: an integer, or a rule for :func:`~pyhctsa.utils.get_tau` (e.g.
        ``'ac1e'``). The symbols code the changes ``|x_{t+tau} - x_t|`` at every ``t``; runs are
        counted along each chain ``t = p, p + tau, ...`` (the series sampled every ``tau``
        samples, from each of the ``tau`` phases), and the pooled count of D-runs is divided by
        the number of symbols. Default 1. NaN when a rule cannot set the delay.

    Returns
    -------
    float
        The probability of obtaining a sequence of D consecutive ones or zeros.

    Notes
    -----
    The value depends on the fraction of changes that reach the threshold, which is set by the
    lag-1 autocorrelation and the distribution of the increments, as well as on the clustering
    of large and small changes. :func:`pol_var_resid` sets the threshold from AR(2) residuals,
    which removes the first two dependences.
    """
    x = np.asarray(x)
    tau = get_tau(x, tau) if x.size > 1 else 1
    if np.isnan(tau):  # the delay could not be set
        return np.nan
    tau = int(tau)
    dx = np.abs(x[tau:] - x[:-tau]) # abs changes over tau samples
    N = len(dx) # number of changes (symbols)

    # binary representation of time series based on changes being greater than d
    x_sym = dx >= d # changes exceed some threshold, d

    pc = 0
    for ph in range(min(tau, N)):  # runs along each lag-tau chain
        chain = x_sym[ph::tau]
        pc += int(np.sum(_run_lengths(chain)[1] // D))

    p = pc / N

    return p

def pol_var_resid(x: ArrayLike, D: int = 3, ar_order: int = 2) -> float:
    """
    Clustering of large and small innovations: POLVAR on AR residuals with a median threshold.

    A variant of :func:`pol_var` that codes the residuals ``e_t`` of a least-squares
    AR(``ar_order``) fit (with intercept) as large (1) or small (0) against the median of
    ``|e_t|``, and counts runs of D equal symbols. Coding residuals
    rather than increments, at a threshold that fixes ``q = P(1) = 1/2``, removes the dependence
    of :func:`pol_var` on the lag-1 autocorrelation and the increment distribution, so the
    statistic measures the clustering of large and small innovations (volatility clustering,
    conditional heteroscedasticity [4]_). It is invariant to the location and scale of ``x``.

    With ``s_t = 1{|e_t| >= median|e|}`` (t = 1, ..., n = N - ``ar_order``) split into maximal
    runs of equal symbols with lengths ``L_1, L_2, ...``,

    ``pol_var_resid = (1/n) * sum_j floor(L_j / D)``,

    the number of non-overlapping words of D equal symbols per symbol. For a linear process with
    iid innovations the symbols are close to iid Bernoulli(1/2) and the statistic is close to
    ``1 / (2**D - 1)``; larger values mean that large and small innovations cluster in time.
    Static nonlinear transforms of a linear process also give clustered residual magnitudes.

    As in :func:`pol_var`, words of D zeros and of D ones are both counted. The published
    POLVAR statistic [1]_ [2]_ [3]_ codes raw beat-to-beat differences against a fixed threshold
    in ms and counts only the word of D zeros.

    The statistic combines standard parts: the run counting of POLVAR, introduced for heart-rate
    variability [3]_ [1]_; coding against the median, as in the runs test of Wald and Wolfowitz
    [5]_; and reading volatility clustering from the magnitudes of model residuals, as in tests
    for ARCH effects [4]_. In effect it is a runs test for the clustering of large innovations.
    No published statistic of this form is known.

    References
    ----------
    .. [1] N. Wessel, C. Ziehmann, J. Kurths, U. Meyerfeldt, A. Schirdewan and A. Voss,
           "Short-term forecasting of life-threatening cardiac arrhythmias based on symbolic
           dynamics and finite-time growth rates", Phys. Rev. E 61(1), 733 (2000).
    .. [2] U. Parlitz, S. Berg, S. Luther, A. Schirdewan, J. Kurths and N. Wessel, "Classifying
           cardiac biosignals using ordinal pattern statistics and symbolic dynamics", Comput.
           Biol. Med. 42(3), 319 (2012).
    .. [3] A. Voss et al., "The application of methods of non-linear dynamics for the improved
           and predictive recognition of patients threatened by sudden cardiac death",
           Cardiovasc. Res. 31(3), 419 (1996).
    .. [4] R. F. Engle, "Autoregressive conditional heteroscedasticity with estimates of the
           variance of United Kingdom inflation", Econometrica 50(4), 987 (1982).
    .. [5] A. Wald and J. Wolfowitz, "On a test whether two samples are from the same
           population", Ann. Math. Stat. 11(2), 147 (1940).

    Parameters
    ----------
    x : array-like
        The input time series.
    D : int
        Word length: the number of consecutive equal symbols counted as one word. Default is 3.
    ar_order : int
        Order of the autoregressive model whose residuals are coded. Default is 2.

    Returns
    -------
    float
        The number of non-overlapping words of D equal symbols per residual. NaN if there are
        fewer than ``2 * D`` residuals, if the series is constant, or if the AR model fits it
        exactly (e.g. a pure sinusoid, which is an exact AR(2) process), when there are no
        innovations to code.
    """
    x = np.asarray(x, dtype=float)
    D, p = int(D), int(ar_order)
    n = len(x) - p  # number of residuals
    if D < 1 or p < 0 or n < max(2 * D, p + 2) or not np.all(np.isfinite(x)):
        return np.nan
    scale = np.std(x)
    if scale == 0:
        return np.nan

    # least-squares AR(p) fit with an intercept: x_t ~ c + a_1 x_{t-1} + ... + a_p x_{t-p}
    design = np.column_stack([x[p - j:len(x) - j] for j in range(1, p + 1)] + [np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, x[p:], rcond=None)
    abs_resid = np.abs(x[p:] - design @ coef)

    threshold = np.median(abs_resid)
    if not threshold > 1e-10 * scale:  # (near-)exact fit: the symbols would code rounding error
        return np.nan
    x_sym = abs_resid >= threshold

    return float(np.sum(_run_lengths(x_sym)[1] // D) / n)

def pnn(x: ArrayLike) -> dict:
    """
    Compute pNNx measures of heart rate variability (HRV).

    The pNNx metrics quantify the proportion of successive RR intervals that 
    differ by more than x milliseconds, for x = 5, 10, 20, 30, ..., 100. This function 
    assumes the input `x` is a time series of consecutive RR intervals in seconds: the 
    differences are multiplied by 1000 to convert them to milliseconds.

    This measure is commonly used in clinical HRV analysis. It is not appropriate 
    to apply this method to z-scored or otherwise normalized time series, as 
    meaningful interpretation depends on absolute differences in time.

    This implementation is derived from `HRVClassic`, with the spectral 
    measures removed, focusing solely on pNNx.

    Port of hctsa's ``MD_pNN``, Copyright (C) 2016 Max A. Little and Ben D. Fulcher (GNU GPL,
    version 3 or later).

    References
    ----------
    .. [1] Mietus, J.E., et al. "The pNNx files: re-examining a widely used 
           heart rate variability measure." Heart 88(4): 378 (2002).

    Parameters
    ----------
    x : array-like
        Time series of RR intervals in seconds.

    Returns
    -------
    dict
        Dictionary containing pNNx values, such as:

        - 'pnn20': Proportion of successive differences > 20 ms
        - 'pnn50': Proportion of successive differences > 50 ms

    """
    x = np.asarray(x)
    diff_x = np.diff(x)
    N = len(x)

    # Calculate pNNx percentage
    Dx = np.abs(diff_x) * 1000 # seconds to milliseconds
    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    dx_sorted = np.sort(Dx)
    counts = (N - 1) - np.searchsorted(dx_sorted, pnns, side='right')

    out = {}
    for threshold, count in zip(pnns, counts):
        out["pnn" + str(threshold)] = np.int64(count) / (N - 1)

    return out

def porta(x: ArrayLike, num_levels: int = 6, tau: int | str = 1) -> dict:
    """
    Compute Porta's symbolic-dynamics word-type indices.

    Quantizes the time series into a small number of levels and classifies
    consecutive length-3 "words" of symbols by their pattern of variation:

    - ``0V``  : no variation (all three symbols equal)
    - ``1V``  : one variation (exactly one of the two transitions is flat)
    - ``2LV`` : two like variations (both transitions move the same direction)
    - ``2UV`` : two unlike variations (transitions move in opposite directions)

    Originally developed for heart-rate-variability analysis, quantifying the
    complexity/regularity of the symbolic dynamics of RR interval sequences.

    The levels are equal-width bins one robust standard deviation wide (for six levels),
    ``s = IQR / 1.349``, centred on the median: the span ``median +/- 3 s`` is divided into
    ``num_levels`` bins and values beyond it fall in the end bins. Porta et al. divide the full
    range ``[min, max]`` into equal bins [2]_, after replacing ectopic beats, artifacts and jumps
    of more than 30% in the RR series. Without that cleaning one outlier widens every range
    bin: a single 10-SD spike in 1000 samples moves the word-type percentages by 25-55 points,
    and the percentages then also track the shape of the value distribution. The robust span
    is the analogue of the original for uncleaned data: unchanged by a few outliers, and
    (for Gaussian data) about as wide as the range bins of a series of about 1000 samples.

    References
    ----------
    .. [1] A. Porta et al., "Entropy, entropy rate, and pattern classification as tools to typify
           complexity in short heart period variability series", IEEE Trans. Biomed. Eng.
           48(11) 1282 (2001).
    .. [2] A. Porta et al., "Symbolic analysis of 24h Holter heart period variability series:
           comparison between normal and heart failure patients", Computers in Cardiology 2005,
           575-578: uniform quantization of the full range into 6 levels, words of 3 symbols.

    Parameters
    ----------
    x : array-like
        The input time series.
    num_levels : int, optional
        The number of quantization levels spanning ``median +/- 3 s``. Default is 6, as in the
        original papers.
    tau : int or str, optional
        The step between the symbols of a word, in samples: an integer, or a rule for
        :func:`~pyhctsa.utils.get_tau` (e.g. ``'ac1e'``). The words are
        ``(s_t, s_{t+tau}, s_{t+2 tau})`` at every ``t``, with the levels set from the whole
        series, which pools the ``tau`` phases of the series sampled every ``tau`` samples.
        Default 1. All outputs are NaN when a rule cannot set the delay.

    Returns
    -------
    dict
        Dictionary containing the percentage of length-3 words of each type:

        - 'pV0'   : percentage of 0V words
        - 'pV1'   : percentage of 1V words
        - 'pV2LV' : percentage of 2LV words
        - 'pV2UV' : percentage of 2UV words
    """
    x = np.asarray(x, dtype=float)
    tau = get_tau(x, tau) if x.size > 1 and np.std(x) > 0 else 1
    if np.isnan(tau):  # the delay could not be set
        return {'pV0': np.nan, 'pV1': np.nan, 'pV2LV': np.nan, 'pV2UV': np.nan}
    tau = int(tau)
    num_words = x.size - 2 * tau

    if num_words < 1 or np.std(x) == 0:
        # Constant series: quantization is undefined
        return {'pV0': np.nan, 'pV1': np.nan, 'pV2LV': np.nan, 'pV2UV': np.nan}

    # quantize into 0:num_levels-1: equal bins over median +/- 3 robust SDs, open end bins
    num_levels = int(num_levels)
    q25, med, q75 = _quantile(x, [0.25, 0.5, 0.75], method='linear')
    scale = (q75 - q25) / 1.349  # robust standard deviation
    if scale <= 1e-12 * np.max(np.abs(x)):  # more than half the values tied
        scale = 1.2533 * np.mean(np.abs(x - med))  # mean absolute deviation (Gaussian-consistent)
    inner = med + (6 * scale / num_levels) * (np.arange(1, num_levels) - num_levels / 2)
    sym = np.searchsorted(inner, x, side='right')

    # transitions between symbols tau apart
    s = np.sign(sym[tau:] - sym[:-tau])

    codes = np.bincount((s[:-tau] + 1) * 3 + (s[tau:] + 1), minlength=9)

    n0V = codes[4]
    n1V = codes[1] + codes[3] + codes[5] + codes[7]
    n2LV = codes[0] + codes[8]
    n2UV = codes[2] + codes[6]

    return {
        'pV0': 100 * n0V / num_words,
        'pV1': 100 * n1V / num_words,
        'pV2LV': 100 * n2LV / num_words,
        'pV2UV': 100 * n2UV / num_words,
    }
