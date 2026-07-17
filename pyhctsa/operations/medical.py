import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from ..utils import bin_picker, histc

def raw_hrv_meas(x: ArrayLike) -> dict:
    """
    Compute Poincaré plot-based HRV (Heart Rate Variability) measures from RR interval time series.

    This function computes the triangular histogram indices and Poincaré plot measures commonly used 
    in HRV analysis. It is specifically designed for time series consisting of consecutive 
    RR intervals measured in milliseconds. It is not suitable for other types of time series.

    The computed features are widely used in clinical and physiological studies of autonomic nervous 
    system activity. The Poincaré plot measures (SD1 and SD2) are standard metrics for short- and 
    long-term variability, while the triangular indices provide geometric summaries of the RR 
    distribution.

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
        Time series of RR intervals in milliseconds.

    Returns
    -------
    out : dict
        Dictionary containing the following HRV features   

        - 'tri10'   : Triangular histogram index using 10 bins.
        - 'tri20'   : Triangular histogram index using 20 bins.
        - 'trisqrt' : Triangular histogram index using a number of bins determined by 
                the square root rule.
        - 'SD1'     : Standard deviation of the Poincaré plot’s minor axis (short-term variability).
        - 'SD2'     : Standard deviation of the Poincaré plot’s major axis (long-term variability).
    """
    x = np.asarray(x)
    N = x.size

    xmin = x.min()
    xmax = x.max()

    # Triangular histogram indices
    edges_10 = bin_picker(xmin, xmax, 10)
    max_count_10 = np.max(histc(x, edges_10))

    edges_20 = bin_picker(xmin, xmax, 20)
    max_count_20 = np.max(histc(x, edges_20))

    n_bins_sqrt = int(np.ceil(np.sqrt(N)))
    edges_sqrt = bin_picker(xmin, xmax, n_bins_sqrt)
    max_count_sqrt = np.max(histc(x, edges_sqrt))

    # Poincaré measures
    dx = np.diff(x)
    sd_diff = np.std(dx, ddof=1)
    var_x = np.var(x, ddof=1)

    return {
        'tri10': N / max_count_10,
        'tri20': N / max_count_20,
        'trisqrt': N / max_count_sqrt,
        'SD1': (1 / np.sqrt(2)) * sd_diff * 1000,
        'SD2': np.sqrt(2.0 * var_x - 0.5 * sd_diff**2) * 1000
    }

def hrv_classic(y: ArrayLike) -> dict:
    """
    Compute classic heart rate variability (HRV) statistics.

    This function computes a variety of standard time-domain, frequency-domain, and
    geometric HRV measures from a time series of RR (or NN) intervals. The input is
    typically assumed to be in **seconds**.

    The following categories of HRV features are included:

    1. **pNNx measures**
    Measures the proportion of interval differences greater than a given threshold `x` [1].

    2. **Frequency-domain measures**
    Power spectral density ratios computed over standard frequency bands (e.g., LF, HF) [2].

    3. **Triangular histogram index**
    A geometric measure of HRV based on the shape of the RR interval histogram.

    4. **Poincaré plot measures (SD1, SD2)**
    Geometric descriptors of the Poincaré plot reflecting short- and long-term variability [3]. 

    This implementation is adapted from original MATLAB code by Max A. Little
    (http://www.maxlittle.net/).

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
        Dictionary containing various HRV features, including pNNx statistics,
        frequency-domain power ratios, triangular index, and Poincaré measures.
    """

    # Standard defaults
    y = np.asarray(y)
    diff_y = np.diff(y)
    n = len(y)

    # ------------------------------------------------------------------------------
    # Calculate pNNx percentage
    # ------------------------------------------------------------------------------
    # pNNx: recommendation as per Mietus et. al. 2002, "The pNNx files: ...", Heart
    # strange to do this for a z-scored time series...
    d_y = np.abs(diff_y)
    pnn_x_fn = lambda x: np.mean(d_y > x / 1000)

    out = {}

    out['pnn5'] = pnn_x_fn(5)  # 0.005*sigma
    out['pnn10'] = pnn_x_fn(10)  # 0.01*sigma
    out['pnn20'] = pnn_x_fn(20)  # 0.02*sigma
    out['pnn30'] = pnn_x_fn(30)  # 0.03*sigma
    out['pnn40'] = pnn_x_fn(40)  # 0.04*sigma

    # ------------------------------------------------------------------------------
    # Calculate PSD
    # ------------------------------------------------------------------------------

    nfft = max(256, 2 ** int(np.ceil(np.log2((n)))))
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

    # Vectorised band selection. pxx[mask] keeps ascending-index order, so the
    # masked np.sum matches the original loop's summation order.
    ind_l = (f >= lf_lo) & (f <= lf_hi)
    ind_h = (f >= hf_lo) & (f <= hf_hi)
    ind_v = (f <= lf_lo)
    lf_p = f_bin_size * np.sum(pxx[ind_l])
    hf_p = f_bin_size * np.sum(pxx[ind_h])
    vlf_p = f_bin_size * np.sum(pxx[ind_v])

    out['lfhf'] = lf_p / hf_p
    total = f_bin_size * np.sum(pxx)
    out['vlf'] = vlf_p / total * 100
    out['lf'] = lf_p / total * 100
    out['hf'] = hf_p / total * 100

    # Triangular histogram index
    edges_10 = bin_picker(y.min(), y.max(), 10)
    hist = histc(y, edges_10)
    out['tri'] = len(y) / np.max(hist)

    # Poincare plot measures:
    # cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    rmssd = np.std(diff_y, ddof=1)
    sigma = np.std(y, ddof=1)

    out["SD1"] = 1 / np.sqrt(2) * rmssd * 1000
    out["SD2"] = np.sqrt(2 * sigma**2 - (1 / 2) * rmssd**2) * 1000

    return out

def pol_var(x: ArrayLike, d: float = 1, D: int = 6) -> float:
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

    Returns
    -------
    float
        The probability of obtaining a sequence of D consecutive ones or zeros.
    """
    x = np.asarray(x)
    dx = np.abs(np.diff(x)) # abs diff in consecutive values of the time series
    N = len(dx) # number of diffs in the input time series

    # binary representation of time series based on consecutive changes being greater than d/1000...
    x_sym = dx >= d # consec. diffs exceed some threshold, d
    z_seq = np.zeros(D)
    o_seq = np.ones(D)

    # search for D consecutive zeros/ones
    i = 0
    pc = 0

    while i <= (N-D):
        x_seq = x_sym[i:(i+D)]
        if np.array_equal(x_seq, z_seq) or np.array_equal(x_seq, o_seq):
            pc += 1
            i += D
        else:
            i += 1
    
    p = pc / N

    return p

def pnn(x: ArrayLike) -> dict:
    """
    Compute pNNx measures of heart rate variability (HRV).

    The pNNx metrics quantify the proportion of successive RR intervals that 
    differ by more than x milliseconds. This function assumes the input `x` is 
    a time series of consecutive RR intervals in milliseconds.

    This measure is commonly used in clinical HRV analysis. It is not appropriate 
    to apply this method to z-scored or otherwise normalized time series, as 
    meaningful interpretation depends on absolute differences in time.

    This implementation is derived from `HRVClassic`, with the spectral 
    measures removed, focusing solely on pNNx.

    References
    ----------
    .. [1] Mietus, J.E., et al. "The pNNx files: re-examining a widely used 
           heart rate variability measure." Heart 88(4): 378 (2002).

    Parameters
    ----------
    x : array-like
        Time series of RR intervals in milliseconds (ms).

    Returns
    -------
    dict
        Dictionary containing pNNx values, such as:

        - 'pNN20': Percentage of successive differences > 20 ms
        - 'pNN50': Percentage of successive differences > 50 ms

    """
    x = np.asarray(x)

    Dx = np.abs(np.diff(x)) * 1000
    N = Dx.size

    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    counts = np.sum(Dx[:, None] > pnns, axis=0)
    values = counts / N

    return dict(zip(
        ("pnn" + pnns.astype(str)),
        values
    ))
