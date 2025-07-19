import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Optional
from numba import njit
from math import factorial
from sklearn.neighbors import KDTree
from scipy.stats import gaussian_kde
from antropy.entropy import _xlogx
from ..Utilities.utils import ZScore, make_buffer, binpicker, histc
from ..Toolboxes.physionet import sampen as _sampen_c 
from ..Operations.Correlation import FirstCrossing
from ..Toolboxes.Michael_Small import shannon
from ..Toolboxes.Max_Little import close_returns as _close_returns_c

def ShannonEntropy(y : ArrayLike, numBins : Union[int, list[int]] = 2, depth : Union[int, list[int]] = 3) -> Union[float, dict, None]:
    y = np.asarray(y)
    binRangeSize = np.size(numBins)
    depthRangeSize = np.size(depth)
    out = None
    if binRangeSize == 1:
        if depthRangeSize == 1:
            # %% Evaluate the shannon entropy of discretization - scales with depth, so it's nice to normalize by this factor
            out = shannon.entropy(y, numBins, depth) / depth
        elif depthRangeSize > 1:
            # % Range over depths specified in the vector and return statistics on results
            numDepths = depthRangeSize
            ents = np.zeros(numDepths)
            for i in range(numDepths):
                ents[i] = shannon.entropy(y, numBins, depth[i]) / depth[i]
            out = {}
            #% Output statistics on variation across the range tested:
            out['maxent'] = np.max(ents)
            out['minent'] = np.min(ents)
            out['medent'] = np.median(ents)
            out['meanent'] = np.mean(ents)
            out['stdent'] = np.std(ents, ddof=1)
    elif binRangeSize > 1:
        if depthRangeSize == 1:
            #%% (*) Statistics over different bin numbers (constant depth)
            ents = np.zeros(binRangeSize)
            for i in range(binRangeSize):
                ents[i] = shannon.entropy(y, numBins[i], depth)
            out = {}
            out['maxent'] = np.max(ents)
            out['minent'] = np.min(ents)
            out['medent'] = np.median(ents)
            out['meanent'] = np.mean(ents)
            out['stdent'] = np.std(ents, ddof=1)
        elif depthRangeSize > 1:
            raise NotImplementedError("Comparing both bins and depth not implemented.")
    
    return out 

def DistributionEntropy(y : ArrayLike, histOrKS : str = 'hist', numBins : int = 10, olremp : float = 0) -> float:
    # (1) Remove outliers?
    y = np.asarray(y)
    if olremp != 0:
        yHat = y[(y >= np.quantile(y, olremp, method='hazen')) & (y <= np.quantile(y, 1-olremp, method='hazen'))]
        if yHat.size == 0:
            return np.nan
        else:
            out = DistributionEntropy(y, histOrKS, numBins) - DistributionEntropy(yHat, histOrKS, numBins)
            return out
    # (2) Form the histogram
    if histOrKS == 'hist':
        # use histogram to calculate pdf
        if isinstance(numBins, int):
            binEdges = binpicker(y.min(), y.max(), nbins=numBins)
            px = histc(y, binEdges)
            px = np.divide(px, np.sum(px))[:-1]
        elif numBins in ['sturges', 'fd', 'sqrt', 'auto']:
            binEdges = np.histogram_bin_edges(y, bins=numBins)
            px = histc(y, binEdges)[:-1]
            px = np.divide(px, np.sum(px))
        else:
            raise ValueError(f"Unknown binning method: {numBins}. Choose either a valid rule or manually specify numBins.")
        binWidths = np.diff(binEdges)
    elif histOrKS == 'ks':
        # use kernel density estimate to calculate pdf
        if isinstance(numBins, float):
            #uses specified width
            bw = numBins
            kde = gaussian_kde(y, bw_method=bw)
            xr = np.linspace(min(y) - 3 * bw, max(y) + 3 * bw, 100) # 3 x bandwidth padding
            px = kde(xr)
        elif numBins in ['', ' ', '[]', 'none']:
            # determine the optimal width
            kde = gaussian_kde(y, bw_method='silverman') # normal-approx equivalent as per docs
            actual_bw = kde.factor * np.std(y)  # Convert factor to actual bandwidth
            xr = np.linspace(min(y) - 3 * actual_bw, max(y) + 3 * actual_bw, 100)
            px = kde(xr)
        else:
            raise ValueError(f"Unknown type for {numBins}. Either set to a float (which specifies the width, or leave empty.)")
        binWidths = np.ones(len(px)) * (xr[1] - xr[0])

    # (3) Compute the entropy sum and return it as output
    P = px[px>0]
    logP = np.log(px[px>0]/binWidths[px>0])

    return -np.sum(P * logP)

def MultiScaleEntropy(
    y: ArrayLike,
    scaleRange: Optional[Union[list, range]] = None,
    m: int = 2,
    r: float = 0.15,
    preProcessHow: Optional[str] = None
) -> dict:
    """
    Compute multiscale entropy (MSE) of a time series using sample entropy across multiple scales.

    Parameters
    ----------
    y : array-like
        Input time series (list or NumPy array).
    scaleRange : list or range, optional
        List or range of scales (window sizes) to use for coarse-graining. Default is range(1, 11).
    m : int, optional
        Embedding dimension for sample entropy (default: 2).
    r : float, optional
        Similarity threshold for sample entropy (default: 0.15).
    preProcessHow : str, optional
        Preprocessing method. Supported:
            - 'diff1': Use z-scored first differences.
            - 'rescale_tau': Rescale using autocorrelation time.

    Returns
    -------
    dict
        Dictionary containing sample entropy at each scale and summary statistics:
            - 'sampen_s{scale}': SampEn at each scale
            - 'maxSampEn', 'maxScale', 'minSampEn', 'minScale', 'meanSampEn', 'stdSampEn', 'cvSampEn', 'meanch'
    """
    y = np.asarray(y)
    if scaleRange is None:
        scaleRange = range(1, 10)
    minTsLength = 20
    numScales = len(scaleRange)

    if preProcessHow is not None:
        if preProcessHow == 'diff1':
            y = ZScore(np.diff(y))
        elif preProcessHow == 'rescale_tau':
            tau = FirstCrossing(y, 'ac', 0, 'discrete')
            y_buffer = make_buffer(y, tau)
            y = np.mean(y_buffer, 1)
            y = ZScore(y)
        else:
            raise ValueError(f"Unknown preprocessing setting: {preProcessHow}")    
    
    # Coarse-graining across scales
    y_cg = []
    for i in range(numScales):
        buffer_size = scaleRange[i]
        y_buffer = make_buffer(y, buffer_size)
        y_cg.append(np.mean(y_buffer, 1))
    
    # Run sample entropy for each m and r value at each scale
    samp_ens = np.zeros(numScales)
    for si in range(numScales):
        if len(y_cg[si]) >= minTsLength:
            samp_en_struct = SampleEntropy(y_cg[si], m, r)
            samp_ens[si] = samp_en_struct[f'sampen{m}']
        else:
            samp_ens[si] = np.nan


    # Outputs: multiscale entropy
    if np.all(np.isnan(samp_ens)):
        if preProcessHow:
            pp_text = f"after {preProcessHow} pre-processing"
        else:
            pp_text = ""
        print(f"Warning: Not enough samples ({len(y)} {pp_text}) to compute SampEn at multiple scales")
        return {'out': np.nan}

    # Output raw values
    out = {f'sampen_s{scaleRange[i]}': samp_ens[i] for i in range(numScales)}

     # Summary statistics of the variation
    max_samp_en = np.nanmax(samp_ens)
    max_ind = np.nanargmax(samp_ens)
    min_samp_en = np.nanmin(samp_ens)
    min_ind = np.nanargmin(samp_ens)

    out.update({
        'maxSampEn': max_samp_en,
        'maxScale': scaleRange[max_ind],
        'minSampEn': min_samp_en,
        'minScale': scaleRange[min_ind],
        'meanSampEn': np.nanmean(samp_ens),
        'stdSampEn': np.nanstd(samp_ens, ddof=1),
        'cvSampEn': np.nanstd(samp_ens, ddof=1) / np.nanmean(samp_ens),
        'meanch': np.nanmean(np.diff(samp_ens))
    })

    return out

def SampleEntropy(y: ArrayLike, M: int = 2, r: Optional[float] = None, preProcessHow: Optional[str] = None) -> dict:
    """
    Compute Sample Entropy (SampEn) of a time series.

    This function calculates SampEn for embedding dimensions from 0 to M. The implementation
    uses the PhysioNet C code (sampen.c by Doug Lake) for efficiency and accuracy.
    Can specify to first apply an incremental differencing of the time series
    thus yielding the 'Control Entropy': "Control Entropy: A complexity measure for nonstationary signals"
    E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009).

    Parameters
    ----------
    y : array-like
        Input time series
    M : int, optional
        Maximum embedding dimension (default: 2)
    r : float, optional
        Similarity threshold. If None, set to 0.1 * std(y)
    preProcessHow : str, optional
        Preprocessing method:
            - 'diff1': Use first differences

    Returns
    -------
    dict
        Dictionary containing:
            - 'sampen{m}': Sample entropy for each m from 0 to M
            - 'quadSampEn{m}': Quadratic sample entropy for each m
            - 'meanchsampen': Mean change in sample entropy values
    """
    y = np.asarray(y, dtype=np.float64)
    if r is None:
        r = 0.1 * np.std(y, ddof=1)
    if preProcessHow == 'diff1':
        y = np.diff(y)

    sampEN = _sampen_c.calculate(y, M+1, r)
    sampEN = sampEN[:-1] # always that extra one for the M = 0 
    out = {}
    for m in range(M + 1):
        out[f'sampen{m}'] = sampEN[m]
        out[f'quadSampEn{m}'] = sampEN[m] + np.log(2 * r)
    if M > 1:
        out['meanchsampen'] = np.mean(np.diff(sampEN))
    return out

def PermEn(y: ArrayLike, m: int = 2, tau: int = 1) -> dict:
    """
    Permutation Entropy (PermEn) of a time series.

    Computes the permutation entropy and its normalized version for a given time series,
    as described in:
        C. Bandt and B. Pompe, "Permutation Entropy: A Natural Complexity Measure for Time Series",
        Phys. Rev. Lett. 88(17) 174102 (2002).
    This implementation modifies code from the antropy package:
    https://github.com/raphaelvallat/antropy to provide both raw and normalized permutation entropy values.

    Parameters
    ----------
    y : array-like
        Input time series.
    m : int, optional
        Embedding dimension (order of the permutation entropy, default: 2).
    tau : int, optional
        Time-delay for the embedding (default: 1).

    Returns
    -------
    dict
        A dictionary containing the permutation entropy and normalized permutation entropy.
    """
    y = np.asarray(y)
    ran_order = range(m)
    hashmult = np.power(m, ran_order)
    assert tau > 0, "delay must be greater than zero."
    sorted_idx = _embed(y, order=m, delay=tau).argsort(kind="quicksort")
    Nx = sorted_idx.shape[0]
    assert Nx > 5, "Time series too short to embed." # need at least 5 embedding vectors to actually do a computation
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = np.true_divide(c, c.sum())
    pe = - _xlogx(p).sum()
    pe_norm = pe / np.log2(factorial(m))
    out = {"permEn": pe, "normPermEn": pe_norm}
    return out

def RPDE(y: ArrayLike, m: int = 2, tau: int = 1, epsilon: float = 0.12, TMax : int = -1) -> dict:
    """
    Recurrence period density entropy (RPDE).

    Fast RPDE analysis on an input signal to obtain an estimate of the normalized entropy (H_norm)
    and other related statistics. Based on Max Little's original rpde code.

    Parameters
    ----------
    y : array-like
        Input signal (must be a 1D array or list).
    m : int, optional
        Embedding dimension (default: 2).
    tau : int, optional
        Embedding time delay (default: 1).
    epsilon : float, optional
        Recurrence neighbourhood radius (default: 0.12).
    TMax : int, optional
        Maximum recurrence time. If not specified (default: -1), all recurrence times are returned.

    Returns
    -------
    dict
        Dictionary containing:
            - 'H_norm': Estimated normalized RPDE value.
            - 'H': Unnormalized entropy.
            - 'rpd': Recurrence period density (probability distribution).
            - 'propNonZero': Proportion of non-zero entries in rpd.
            - 'meanNonZero': Mean value of non-zero rpd entries (rescaled by N).
            - 'maxRPD': Maximum value of rpd (rescaled by N).

    References
    ----------
    M. Little, P. McSharry, S. Roberts, D. Costello, I. Moroz (2007),
    "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection",
    BioMedical Engineering OnLine 2007, 6:23.
    """

    y = np.asarray(y)
    rpd = np.array(_close_returns_c.close_returns(y, m, tau, epsilon))
    if TMax > -1:
        rpd = rpd[:TMax]
    rpd = np.divide(rpd, np.sum(rpd))
    N = len(rpd)
    ip = rpd > 0
    H = -np.sum(rpd[ip] * np.log(rpd[ip]))
    H_norm = np.divide(H, np.log(N))
    out = {}
    out['H'] = H
    out["H_norm"] = H_norm

    # prop of non-zero entries
    out['propNonZero'] = np.mean(rpd > 0) # proportion of rpds that are non-zero
    out['meanNonZero'] = np.mean(rpd[ip]) * N # mean value when rpd is non-zero (rescale by N)
    out['maxRPD'] = np.max(rpd) * N # maximum value of rpd (rescale by N)

    return out 

def ApproximateEntropy(x : ArrayLike, mnom : int = 1, rth : float = 0.2) -> float:
    """
    Approximate Entropy of a time series

    ApEn(m,r).

    Parameters
    -----------
    y : array-like
        The input time series
    mnom : int, optional
        The embedding dimension (default is 1)
    rth : float, optional
        The threshold for judging closeness/similarity (default is 0.2)

    Returns
    --------
    float
        The Approximate Entropy value

    References:
    -----------
    S. M. Pincus, "Approximate entropy as a measure of system complexity",
    P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)

    For more information, cf. http://physionet.org/physiotools/ApEn/
    """
    x = np.asarray(x)
    r = rth * np.std(x, ddof=1) # threshold of similarity
    phi = _app_samp_entropy(x, order=mnom, r=r, metric="chebyshev", approximate=True)
    return np.subtract(phi[0], phi[1])

def _embed(x, order, delay=1):
    """Safe embedding that supports order=1."""
    x = np.asarray(x)
    if order < 1:
        raise ValueError("Order must be at least 1.")
    N = x.shape[0]
    if N - (order - 1) * delay <= 0:
        raise ValueError("Time series is too short for the given order and delay.")
    return np.array([x[i:i + order * delay:delay] for i in range(N - (order - 1) * delay)])

def _app_samp_entropy(x, order, r, metric="chebyshev", approximate=True):
    """Modified version of _app_samp_entropy that supports order=1."""
    phi = np.zeros(2)
    emb_data1 = _embed(x, order, 1)
    if approximate:
        pass
    else:
        emb_data1 = emb_data1[:-1]

    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64)

    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi

def ComplexityInvariantDistance(y : ArrayLike) -> dict:
    """
    Complexity-invariant distance

    Computes two estimates of the 'complexity' of a time series based on the 
    stretched-out length of the lines in its line graph. These features are 
    based on the method described by Batista et al. (2014), designed for use 
    in complexity-invariant distance calculations.

    Parameters
    ----------
    y : array-like
        One-dimensional time series input.

    Returns
    -------
    dict
        A dictionary containing the following features:
        
        - 'CE1' : float
            Root mean square of successive differences.
        - 'CE2' : float
            Mean length of line segments between consecutive points using 
            Euclidean distance (Pythagorean theorem).
        - 'minCE1' : float
            Minimum CE1 value computed from sorted time series.
        - 'minCE2' : float
            Minimum CE2 value computed from sorted time series.
        - 'CE1_norm' : float
            Normalized CE1: CE1 / minCE1.
        - 'CE2_norm' : float
            Normalized CE2: CE2 / minCE2.

    References
    ----------
    Batista, G. E. A. P. A., Keogh, E. J., Tataw, O. M., & de Souza, V. M. A. 
    (2014). CID: an efficient complexity-invariant distance for time series. 
    Data Mining and Knowledge Discovery, 28(3), 634–669. 
    https://doi.org/10.1007/s10618-013-0312-3
    """
    y = np.asarray(y)
    #% Original definition (in Table 2 of paper cited above)
    # % sum -> mean to deal with non-equal time-series lengths
    # % (now scales properly with length)

    f_CE1 = lambda y: np.sqrt(np.mean(np.power(np.diff(y), 2)))
    #% Definition corresponding to the line segment example in Fig. 9 of the paper
    #% cited above (using Pythagoras's theorum):
    f_CE2 = lambda y: np.mean(np.sqrt(1 + np.power(np.diff(y), 2)))

    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    # % Defined as a proportion of the minimum such value possible for this time series,
    # % this would be attained from putting close values close; i.e., sorting the time
    # % series
    y_sorted = np.sort(y)
    minCE1 = f_CE1(y_sorted)
    minCE2 = f_CE2(y_sorted)

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':       CE1,
           'CE2':       CE2,
           'minCE1':    minCE1,
           'minCE2':    minCE2,
           'CE1_norm':  CE1_norm,
           'CE2_norm':  CE2_norm}

    return out

def LZComplexity(x: ArrayLike, nbits: int = 2, preProc: Union[str, list] = [], rng : int = 0) -> float:
    """
    Compute the normalized Lempel-Ziv (LZ) complexity of an n-bit encoding of a time series.

    This function measures the complexity of a time series by counting the number of distinct
    symbol sequences (phrases) in its n-bit symbolic encoding, normalized by the expected
    number for a random (noise) sequence. Optionally, a preprocessing step can be applied
    before symbolization.

    Parameters
    ----------
    x : array-like
        Input time series (1-D array or list).
    nbits : int, optional
        Number of bits (alphabet size) to encode the data into (default: 2).
    preProc : str or list, optional
        Preprocessing method to apply before symbolization. Currently supported:
            - 'diff': Use z-scored first differences of the time series.
    rng : int, optional
        Random seed for reproducibility (default: 0). Used for adding small random
        noise to break ties during symbolization.

    Returns
    -------
    float
        Normalized Lempel-Ziv complexity: the number of distinct symbol sequences
        divided by the expected number for a noise sequence.
    """
    rng = np.random.RandomState(rng) # fix the seed for reproducibility
    x = np.asarray(x, dtype=np.float64).ravel()
    if preProc == "diff":
        x = ZScore(np.diff(x))

    if x.size == 0 or nbits < 2:
        return 0.0

    symbols = _symbolise_lz(x, nbits, rng)
    c = _lz_complexity(symbols)

    return (c * np.log(x.size)) / (x.size * np.log(nbits))

@njit(cache=True, fastmath=True)
def _lz_complexity(symbols: np.ndarray) -> int:
    """
    Count phrases exactly as in the original Python function.
    Input must be a 1-D int32/64 NumPy array whose values start at 1.
    """
    n = symbols.size
    if n == 0:
        return 0

    c  = 1 # phrase counter
    ns = 1 # phrase start
    nq = 1  # phrase length
    k  = 2 # overall scan pointer

    while k < n:
        is_substring = False
        max_i = ns - nq
        # brute-force search
        for i in range(max_i + 1):
            match = True
            for j in range(nq):
                if symbols[i + j] != symbols[ns + j]:
                    match = False
                    break
            if match:
                is_substring = True
                break

        if is_substring:
            nq += 1
        else:
            c  += 1
            ns += nq
            nq  = 1
        k += 1

    return c

def _symbolise_lz(x: np.ndarray, n_bins: int, rng) -> np.ndarray:
    nx = x.size
    noisy = x + np.finfo(np.float64).eps * rng.randn(nx)
    order = np.argsort(noisy, kind="mergesort") 
    ranks = np.arange(1, nx + 1)
    symbols = np.floor(ranks * (n_bins / (nx + 1)))

    out = np.empty_like(symbols)
    out[order] = symbols + 1                          
    return out
