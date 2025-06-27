import numpy as np
from typing import Union
from scipy import signal
from ..Utilities.utils import binpicker, histc
from numpy.typing import ArrayLike

def RawHRVMeas(x: ArrayLike) -> dict:
    """
    Compute Poincaré plot-based HRV (Heart Rate Variability) measures from RR interval time series.

    This function computes the triangular histogram indices and Poincaré plot measures commonly used 
    in HRV analysis. It is specifically designed for time series consisting of consecutive RR intervals 
    measured in milliseconds. It is not suitable for other types of time series.

    The computed features are widely used in clinical and physiological studies of autonomic nervous 
    system activity. The Poincaré plot measures (SD1 and SD2) are standard metrics for short- and 
    long-term variability, while the triangular indices provide geometric summaries of the RR 
    distribution.

    References
    ----------
    - M. Brennan, M. Palaniswami, and P. Kamen, 
    "Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability?", 
    IEEE Transactions on Biomedical Engineering, 48(11), pp. 1342–1347, 2001.
    - Original MATLAB implementation adapted from: Max Little's `hrv_classic.m`
    (http://www.maxlittle.net/)

    Parameters
    ----------
    x : array_like
        Time series of RR intervals in milliseconds.

    Returns
    -------
    out : dict
        Dictionary containing the following HRV features   
        - 'tri10'   : Triangular histogram index using 10 bins.
        - 'tri20'   : Triangular histogram index using 20 bins.
        - 'trisqrt' : Triangular histogram index using a number of bins determined by the square root rule.
        - 'SD1'     : Standard deviation of the Poincaré plot’s minor axis (short-term variability).
        - 'SD2'     : Standard deviation of the Poincaré plot’s major axis (long-term variability).
    """

    x = np.asarray(x)
    N = len(x)
    
    out = {}

    # triangular histogram index  
    # 10 bins  
    edges10 = binpicker(x.min(), x.max(), 10)
    hist_counts10 = histc(x, edges10)
    out['tri10'] = N/np.max(hist_counts10)

    # 20 bins
    edges20 = binpicker(x.min(), x.max(), 20)
    hist_counts20 = histc(x, edges20)
    out['tri20'] = N/np.max(hist_counts20)

    # (sqrt samples) bins
    edges_sqrt = binpicker(x.min(), x.max(), int(np.ceil(np.sqrt(N))))
    hist_counts_sqrt = histc(x, edges_sqrt)
    out['trisqrt'] = N/np.max(hist_counts_sqrt)

    # Poincare plot measures
    diffx = np.diff(x)
    out['SD1'] = 1/np.sqrt(2) * np.std(diffx, ddof=1) * 1000
    out['SD2'] = np.sqrt(2 * np.var(x, ddof=1) - (1/2) * np.std(diffx, ddof=1)**2) * 1000

    return out

def HRVClassic(y: ArrayLike) -> dict:
    """
    Compute classic heart rate variability (HRV) statistics.

    This function computes a variety of standard time-domain, frequency-domain, and 
    geometric HRV measures from a time series of RR (or NN) intervals. The input is 
    typically assumed to be in **seconds**.

    The following categories of HRV features are included:

    1. **pNNx measures**  
    Measures the proportion of interval differences greater than a given threshold `x`.  
    Reference:  
    Mietus, J.E., et al., *The pNNx files: Re-examining a widely used heart rate variability measure*,  
    Heart, 88(4):378, 2002.

    2. **Frequency-domain measures**  
    Power spectral density ratios computed over standard frequency bands (e.g., LF, HF).  
    Reference:  
    Malik, M., et al., *Heart rate variability: Standards of measurement, physiological interpretation, and clinical use*,  
    European Heart Journal, 17(3):354, 1996.

    3. **Triangular histogram index**  
    A geometric measure of HRV based on the shape of the RR interval histogram.

    4. **Poincaré plot measures (SD1, SD2)**  
    Geometric descriptors of the Poincaré plot reflecting short- and long-term variability.  
    Reference:  
    Brennan, M., et al., *Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability?*,  
    IEEE Transactions on Biomedical Engineering, 48(11):1342, 2001.

    This implementation is adapted from original MATLAB code by Max A. Little  
    (http://www.maxlittle.net/).

    Parameters
    ----------
    y : array_like
        Input time series of RR intervals, assumed to be in seconds.

    Returns
    -------
    out: dict
        Dictionary containing various HRV features, including pNNx statistics, 
        frequency-domain power ratios, triangular index, and Poincaré measures.
    """

    # Standard defaults
    y = np.asarray(y)
    diffy = np.diff(y)
    N = len(y)

    # ------------------------------------------------------------------------------
    # Calculate pNNx percentage
    # ------------------------------------------------------------------------------
    # pNNx: recommendation as per Mietus et. al. 2002, "The pNNx files: ...", Heart
    # strange to do this for a z-scored time series...
    Dy = np.abs(diffy)
    PNNxfn = lambda x : np.mean(Dy > x/1000)

    out = {}

    out['pnn5'] = PNNxfn(5) # 0.0055*sigma
    out['pnn10'] = PNNxfn(10) # 0.01*sigma
    out['pnn20'] = PNNxfn(20) # 0.02*sigma
    out['pnn30'] = PNNxfn(30) # 0.03*sigma
    out['pnn40'] = PNNxfn(40) # 0.04*sigma

    # ------------------------------------------------------------------------------
    # Calculate PSD
    # ------------------------------------------------------------------------------

    nfft = max(256, 2**int(np.ceil(np.log2((N)))))
    F, Pxx = signal.periodogram(y, window = np.hanning(len(y)), detrend=False, scaling='density', fs=2*np.pi, nfft=nfft)

    # Calculate spectral measures such as subband spectral power percentage, LF/HF ratio etc.
    LF_lo = 0.04 # /pi -- fraction of total power (max F is pi)
    LF_hi = 0.15
    HF_lo = 0.15
    HF_hi = 0.4

    fbinsize = F[1] - F[0]

    indl = []
    for x in F:
        if x >= LF_lo and x <= LF_hi:
            indl.append(1)
        else :
            indl.append(0)
    indh = []
    for x in F:
        if x >= HF_lo and x <= HF_hi:
            indh.append(1)
        else:
            indh.append(0)

    indv = []
    for x in F:
        if x <= LF_lo:
            indv.append(1)
        else :
            indv.append(0)

    indlPxx = []
    for i in range(0, len(Pxx)):
        if indl[i] == 1:
            indlPxx.append(Pxx[i])
    lfp = fbinsize * np.sum(indlPxx)

    indhPxx = []
    for i in range(0, len(Pxx)):
        if indh[i] == 1:
            indhPxx.append(Pxx[i])
    hfp = fbinsize * np.sum(indhPxx)

    indvPxx = []
    for i in range(0, len(Pxx)):
        if indv[i] == 1:
            indvPxx.append(Pxx[i])
    vlfp = fbinsize * np.sum(indvPxx)

    out['lfhf'] = lfp / hfp
    total = fbinsize * np.sum(Pxx)
    out['vlf'] = vlfp/total * 100
    out['lf'] = lfp/total * 100
    out['hf'] = hfp/total * 100

    # Triangular histogram index
    edges10 = binpicker(y.min(), y.max(), 10)
    hist = histc(y, edges10)
    out['tri'] = len(y)/np.max(hist)

    # Poincare plot measures:
    # cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    rmssd = np.std(diffy, ddof=1)
    sigma = np.std(y, ddof=1)

    out["SD1"] = 1/np.sqrt(2) * rmssd * 1000
    out["SD2"] = np.sqrt(2 * sigma**2 - (1/2) * rmssd**2) * 1000

    return out

def PolVar(x : ArrayLike, d : float = 1, D : int = 6) -> float:
    """
    Compute the POLVARd measure of a time series.

    The POLVARd (also called Plvar) measure quantifies the probability of 
    obtaining a sequence of consecutive ones or zeros in a symbolic sequence 
    derived from the input time series.

    This measure was originally introduced in:
        Wessel et al., "Short-term forecasting of life-threatening cardiac 
        arrhythmias based on symbolic dynamics and finite-time growth rates",
        Phys. Rev. E 61(1), 733 (2000).

    The original implementation applied this measure to RR interval sequences 
    (typically in milliseconds), with the symbolic threshold `d` representing 
    raw amplitude differences. This implementation generalizes it to 
    z-scored time series, such that `d` is specified in units of standard deviation.

    The function is derived from the MATLAB implementation by Max A. Little 
    (2009) and Ben D. Fulcher.

    References
    ----------
    .. [1] Wessel et al., Phys. Rev. E 61(1), 733 (2000).
    .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework for 
           Automated Time-Series Phenotyping Using Massive Feature Extraction", 
           Cell Systems 5: 527 (2017). DOI: 10.1016/j.cels.2017.10.001
    .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative time-series 
           analysis: the empirical structure of time series and their methods", 
           J. Roy. Soc. Interface 10(83) 20130048 (2013). 
           DOI: 10.1098/rsif.2013.0048

    Parameters
    ----------
    x : array_like
        The input time series.
    d : float
        Symbolic coding threshold in units of standard deviation.
    D : int
        Word length for detecting consecutive sequences (commonly D=6).

    Returns
    -------
    float
        The probability of obtaining a sequence of D consecutive ones or zeros.
    """
    x = np.asarray(x)
    dx = np.abs(np.diff(x)) # abs diff in consecutive values of the time series
    N = len(dx) # number of diffs in the input time series

    # binary representation of time series based on consecutive changes being greater than d/1000...
    xsym = dx >= d # consec. diffs exceed some threshold, d
    zseq = np.zeros(D)
    oseq = np.ones(D)

    # search for D consecutive zeros/ones
    i = 0
    pc = 0

    # seqcnt = 0
    while i <= (N-D):
        xseq = xsym[i:(i+D)]
        if np.array_equal(xseq, zseq) or np.array_equal(xseq, oseq):
            pc += 1
            i += D
        else:
            i += 1
    
    p = pc / N

    return p 

def PNN(x : ArrayLike) -> dict:
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
    .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework for 
           Automated Time-Series Phenotyping Using Massive Feature Extraction", 
           Cell Systems 5: 527 (2017). DOI: 10.1016/j.cels.2017.10.001
    .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative 
           time-series analysis: the empirical structure of time series and 
           their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013). 
           DOI: 10.1098/rsif.2013.0048

    Parameters
    ----------
    x : array_like
        Time series of RR intervals in milliseconds.

    Returns
    -------
    dict
        Dictionary containing pNNx values, such as:
            - 'pNN20': Percentage of successive differences > 20 ms
            - 'pNN50': Percentage of successive differences > 50 ms

    """
    x = np.asarray(x)
    diffx = np.diff(x)
    N = len(x)

    # Calculate pNNx percentage

    Dx = np.abs(diffx) * 1000 # assume milliseconds as for RR intervals
    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    out = {}
    for x in pnns:
        out["pnn" + str(x) ] = sum(Dx > x) / (N-1)

    return out
