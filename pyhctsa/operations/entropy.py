from math import factorial
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from numba import njit
from antropy.entropy import _xlogx
from scipy.stats import gaussian_kde
from sklearn.neighbors import KDTree

from ..operations.correlation import first_crossing
from ..toolboxes.Michael_Small import shannon
from ..toolboxes.Max_Little import close_returns as _close_returns_c
from ..toolboxes.physionet import sampen as _sampen_c
from ..utils import bin_picker, histc, make_buffer, z_score

def shannon_entropy(
    y: ArrayLike,
    num_bins: Union[int, list[int]] = 2,
    depth: Union[int, list[int]] = 3
) -> Union[float, dict, None]:
    """
    Approximate Shannon entropy of a time series.

    Uses a num_bins-bin encoding and depth-symbol sequences.
    Uniform population binning is used, and the implementation uses Michael Small's code
    MS_shannon.c.

    In this wrapper function, you can evaluate the code at a given n and d, and
    also across a range of depth and num_bins to return statistics on how the obtained
    entropies change.

    References
    ----------
    .. [1] M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
        Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
        Vol. 52 (2005).
    .. [2] Michael Small's code is available at http://small.eie.polyu.edu.hk/matlab/

    Parameters
    ----------
    y : array-like
        The input time series.
    num_bins : int or list of int, optional
        The number of bins to discretize the time series into (i.e., alphabet size).
    depth : int or list of int, optional
        The length of strings to analyze.

    Returns
    -------
    float or dict or None
        The normalized Shannon entropy for a given setting, or summary statistics
        (max, min, median, mean, std) across a range of numBins or depths.
    """
    y = np.asarray(y)
    bin_range_size = np.size(num_bins)
    depth_range_size = np.size(depth)
    out = None

    if bin_range_size == 1:
        if depth_range_size == 1:
            # %% Evaluate the shannon entropy of discretization - scales with depth, so it's nice to normalize by this factor
            out = shannon.entropy(y, num_bins, depth) / depth
        elif depth_range_size > 1:
            # % Range over depths specified in the vector and return statistics on results
            num_depths = depth_range_size
            ents = np.zeros(num_depths)
            for i in range(num_depths):
                ents[i] = shannon.entropy(y, num_bins, depth[i]) / depth[i]
            out = {}
            #% Output statistics on variation across the range tested:
            out['maxent'] = np.max(ents)
            out['minent'] = np.min(ents)
            out['medent'] = np.median(ents)
            out['meanent'] = np.mean(ents)
            out['stdent'] = np.std(ents, ddof=1)

    elif bin_range_size > 1:
        if depth_range_size == 1:
            #%% (*) Statistics over different bin numbers (constant depth)
            ents = np.zeros(bin_range_size)
            for i in range(bin_range_size):
                ents[i] = shannon.entropy(y, num_bins[i], depth)
            out = {}
            out['maxent'] = np.max(ents)
            out['minent'] = np.min(ents)
            out['medent'] = np.median(ents)
            out['meanent'] = np.mean(ents)
            out['stdent'] = np.std(ents, ddof=1)
        elif depth_range_size > 1:
            raise NotImplementedError("Comparing both bins and depth not implemented.")

    return out

def distribution_entropy(
    y: ArrayLike,
    hist_or_ks: str = 'hist',
    num_bins: Union[str, int] = 10,
    olremp: float = 0
) -> float:
    """
    Distributional entropy.

    Estimates entropy from the distribution of a data vector. The distribution is estimated
    either using a histogram with numBins bins, or as a kernel-smoothed distribution using
    a Gaussian kernel.

    An optional additional parameter can be used to remove a proportion of the most extreme
    positive and negative deviations from the mean as an initial pre-processing step.

    Parameters
    ----------
    y : array-like
        The input time series.
    hist_or_ks : str
        Whether to use a histogram ('hist') or kernel-smoothed ('ks') distribution.
    num_bins : int or list of int, optional

        - (for 'hist'): an integer, uses a histogram with that many bins
        - (for 'ks'): a positive real number, for the bandwidth parameter for the kernel density estimate.

    olremp : float, optional
        The proportion of outliers at both extremes to remove.
        (e.g., if olremp = 0.01; keeps only the middle 98% of data; 0 keeps all data. 
        This parameter ought to be less than 0.5, which keeps none of the data).

    Returns
    -------
    float
        Estimate of entropy from the distribution.
    """
    # (1) Remove outliers?
    y = np.asarray(y)
    if olremp != 0:
        y_hat = y[
            (y >= np.quantile(y, olremp, method='hazen')) &
            (y <= np.quantile(y, 1 - olremp, method='hazen'))
        ]
        if y_hat.size == 0:
            return np.nan
        else:
            out = (
                distribution_entropy(y, hist_or_ks, num_bins)
                - distribution_entropy(y_hat, hist_or_ks, num_bins)
            )
            return out

    # (2) Form the histogram
    if hist_or_ks == 'hist':
        # use histogram to calculate pdf
        if isinstance(num_bins, int):
            bin_edges = bin_picker(x_min=y.min(), x_max=y.max(), n_bins=num_bins)
            px = histc(y, bin_edges)
            px = np.divide(px, np.sum(px))[:-1]
        elif num_bins in ['sturges', 'fd', 'sqrt', 'auto']:
            bin_edges = np.histogram_bin_edges(y, bins=num_bins)
            px = histc(y, bin_edges)[:-1]
            px = np.divide(px, np.sum(px))
        else:
            raise ValueError(
                f"Unknown binning method: {num_bins}. Choose either a valid rule or manually specify numBins."
            )
        bin_widths = np.diff(bin_edges)

    elif hist_or_ks == 'ks':
        # use kernel density estimate to calculate pdf
        if isinstance(num_bins, float):
            # uses specified width
            bw = num_bins
            kde = gaussian_kde(y, bw_method=bw)
            xr = np.linspace(min(y) - 3 * bw, max(y) + 3 * bw, 100)  # 3 x bandwidth padding
            px = kde(xr)
        elif num_bins in ['', ' ', '[]', 'none']:
            # determine the optimal width
            kde = gaussian_kde(y, bw_method='silverman')  # normal-approx equivalent as per docs
            actual_bw = kde.factor * np.std(y)  # Convert factor to actual bandwidth
            xr = np.linspace(min(y) - 3 * actual_bw, max(y) + 3 * actual_bw, 100)
            px = kde(xr)
        else:
            raise ValueError(
                f"Unknown type for {num_bins}. Either set to a float (which specifies the width, or leave empty.)"
            )
        bin_widths = np.ones(len(px)) * (xr[1] - xr[0])

    # (3) Compute the entropy sum and return it as output
    p = px[px > 0]
    log_p = np.log(px[px > 0] / bin_widths[px > 0])

    return -np.sum(p * log_p)

def multi_scale_entropy(
    y: ArrayLike,
    scale_range: Optional[Union[list, range]] = None,
    m: int = 2,
    r: float = 0.15,
    pre_process_how: Optional[str] = None
) -> dict:
    """
    Compute multiscale entropy (MSE) of a time series using sample entropy across multiple scales.

    Parameters
    ----------
    y : array-like
        Input time series (list or NumPy array).
    scale_range : list or range, optional
        List or range of scales (window sizes) to use for coarse-graining. Default is range(1, 11).
    m : int, optional
        Embedding dimension for sample entropy (default: 2).
    r : float, optional
        Similarity threshold for sample entropy (default: 0.15).
    pre_process_how : str, optional
        Preprocessing method. Supported:

        - 'diff1': Use z-scored first differences.
        - 'rescale_tau': Rescale using autocorrelation time.

    Returns
    -------
    dict
        Dictionary containing sample entropy at each scale and summary statistics.
    """
    y = np.asarray(y)
    m = int(m)
    if scale_range is None:
        scale_range = range(1, 10)
    min_ts_length = 20
    num_scales = len(scale_range)

    if pre_process_how is not None:
        if pre_process_how == 'diff1':
            y = z_score(np.diff(y))
        elif pre_process_how == 'rescale_tau':
            tau = first_crossing(y, 'ac', 0, 'discrete')
            y_buffer = make_buffer(y, tau)
            y = np.mean(y_buffer, 1)
            y = z_score(y)
        else:
            raise ValueError(f"Unknown preprocessing setting: {pre_process_how}")    
    
    # Coarse-graining across scales
    y_cg = []
    for i in range(num_scales):
        buffer_size = scale_range[i]
        y_buffer = make_buffer(y, buffer_size)
        y_cg.append(np.mean(y_buffer, 1))
    
    # Run sample entropy for each m and r value at each scale
    samp_ens = np.zeros(num_scales)
    for si in range(num_scales):
        if len(y_cg[si]) >= min_ts_length:
            samp_en_struct = sample_entropy(y_cg[si], m, r)
            samp_ens[si] = samp_en_struct[f'sampen{m}']
        else:
            samp_ens[si] = np.nan


    # Outputs: multiscale entropy
    if np.all(np.isnan(samp_ens)):
        if pre_process_how:
            pp_text = f"after {pre_process_how} pre-processing"
        else:
            pp_text = ""
        print(f"Warning: Not enough samples ({len(y)} {pp_text}) to compute SampEn at multiple scales")
        return {'out': np.nan}

    # Output raw values
    out = {f'sampen_s{scale_range[i]}': samp_ens[i] for i in range(num_scales)}

     # Summary statistics of the variation
    max_samp_en = np.nanmax(samp_ens)
    max_ind = np.nanargmax(samp_ens)
    min_samp_en = np.nanmin(samp_ens)
    min_ind = np.nanargmin(samp_ens)

    out.update({
        'maxSampEn': max_samp_en,
        'maxScale': scale_range[max_ind],
        'minSampEn': min_samp_en,
        'minScale': scale_range[min_ind],
        'meanSampEn': np.nanmean(samp_ens),
        'stdSampEn': np.nanstd(samp_ens, ddof=1),
        'cvSampEn': np.nanstd(samp_ens, ddof=1) / np.nanmean(samp_ens),
        'meanch': np.nanmean(np.diff(samp_ens))
    })

    return out

def sample_entropy(y: ArrayLike, m: int = 2, r: Optional[float] = None,
                    pre_process_how: Optional[str] = None) -> dict:
    """
    Compute Sample Entropy (SampEn) of a time series.

    This function calculates SampEn for embedding dimensions from 0 to m. The implementation
    uses the PhysioNet C code (sampen.c by Doug Lake) [1] for efficiency and accuracy.
    Can specify to first apply an incremental differencing of the time series
    thus yielding the 'Control Entropy' [2].

    References
    ----------
    .. [1] "Sample Entropy Estimation 1.0.0", 
        https://physionet.org/content/sampen/1.0.0/c/sampen-1.1.c
    .. [2] "Control Entropy: A complexity measure for nonstationary signals"
        E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009).

    Parameters
    ----------
    y : array-like
        Input time series
    m : int, optional
        Maximum embedding dimension (default: 2)
    r : float, optional
        Similarity threshold. If None, set to 0.1 * std(y)
    pre_process_how : str, optional
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
    m = int(m)
    y = np.asarray(y, dtype=np.float64)
    if r is None:
        r = 0.1 * np.std(y, ddof=1)
    if pre_process_how == 'diff1':
        y = np.diff(y)

    samp_en = _sampen_c.calculate(y, m+1, r)
    samp_en = samp_en[:-1] # always that extra one for the M = 0
    out = {}
    for mi in range(m + 1):
        out[f'sampen{mi}'] = samp_en[mi]
        out[f'quadSampEn{mi}'] = samp_en[mi] + np.log(2 * r)
    if m > 1:
        out['meanchsampen'] = np.mean(np.diff(samp_en))

    return out

def permutation_entropy(y: ArrayLike, m: int = 2, tau: int = 1) -> dict:
    """
    Permutation Entropy (PermEn) of a time series.

    Computes the permutation entropy and its normalised version for a given time series,
    as described in [1]. 

    This implementation modifies code from the antropy package:
    https://github.com/raphaelvallat/antropy to provide both raw and 
    normalised permutation entropy values.

    References
    ----------
    .. [1] C. Bandt and B. Pompe, "Permutation Entropy: A Natural 
        Complexity Measure for Time Series",
        Phys. Rev. Lett. 88(17) 174102 (2002).

    Parameters
    ----------
    y : array-like
        Input time series.
    m : int, optional
        Embedding dimension (order of the permutation entropy, default: 2).
    tau : int or str, optional
        Time-delay for the embedding (default: 1).

    Returns
    -------
    dict
        A dictionary containing the permutation entropy and normalized permutation entropy.
    """
    m = int(m)
    if tau == 'ac':
        tau = int(first_crossing(y, 'ac', 0, 'discrete'))
    else:
        tau = int(tau)
    y = np.asarray(y)
    ran_order = range(m)

    hash_mult = np.power(m, ran_order)
    assert tau > 0, "delay must be greater than zero."

    sorted_idx = _embed(y, order=m, delay=tau).argsort(kind="quicksort")
    nx = sorted_idx.shape[0]
    assert nx > 5, "Time series too short to embed. Need at least 5 embedding vectors to compute permutation entropy."
    
    hash_val = (np.multiply(sorted_idx, hash_mult)).sum(1)
    _, c = np.unique(hash_val, return_counts=True)
    p = np.true_divide(c, c.sum())
    pe = - _xlogx(p).sum()
    pe_norm = pe / np.log2(factorial(m))
    out = {"permEn": pe, "normPermEn": pe_norm}

    return out

def rpde(y: ArrayLike, m: int = 2, tau: int = 1, epsilon: float = 0.12, t_max: int = -1) -> dict:
    """
    Recurrence period density entropy (RPDE).

    Fast RPDE analysis on an input signal to obtain an estimate of the normalized entropy (H_norm)
    and other related statistics. Based on Max Little's original rpde code [1]. 

    References
    ----------
    .. [1] M. Little, P. McSharry, S. Roberts, D. Costello, I. Moroz (2007),
        "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for 
        Voice Disorder Detection", BioMedical Engineering OnLine 2007, 6:23.

    Parameters
    ----------
    y : array-like
        Input signal (must be a 1D array or list).
    m : int, optional
        Embedding dimension (default: 2).
    tau : int or str, optional
        Embedding time delay (default: 1).
    epsilon : float, optional
        Recurrence neighbourhood radius (default: 0.12).
    t_max : int, optional
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

    """
    if tau == 'ac':
        # use the first zero crossing of the ACF
        tau = int(first_crossing(y, 'ac', 0, 'discrete'))
    y = np.asarray(y)
    rpd = np.array(_close_returns_c.close_returns(y, m, tau, epsilon))
    if t_max > -1:
        rpd = rpd[:t_max]
    rpd = np.divide(rpd, np.sum(rpd))
    N = len(rpd)
    ip = rpd > 0
    H = -np.sum(rpd[ip] * np.log(rpd[ip]))
    H_norm = np.divide(H, np.log(N)) # log(N) is the H for an i.i.d. process
    out = {}
    #% Entropy and normalized entropy:
    out['H'] = H
    out["H_norm"] = H_norm

    # prop of non-zero entries
    out['propNonZero'] = np.mean(rpd > 0) # proportion of rpds that are non-zero
    out['meanNonZero'] = np.mean(rpd[ip]) * N # mean value when rpd is non-zero (rescale by N)
    out['maxRPD'] = np.max(rpd) * N # maximum value of rpd (rescale by N)

    return out

def approximate_entropy(x: ArrayLike, mnom: int = 1, rth: float = 0.2) -> float:
    """
    Approximate entropy (ApEn) of a time series.

    Computes :math:`\\mathrm{ApEn}(m, r)`.

    For details, see the PhysioNet documentation:
    https://physionet.org/physiotools/apen/

    References
    ----------
    .. [1] S. M. Pincus, "Approximate entropy as a measure of system complexity,"
        *Proc. Natl. Acad. Sci. USA*, 88(6), 2297 (1991).

    Parameters
    ----------
    x : array-like
        Input time series.

    mnom : int, optional
        Embedding dimension :math:`m`. Default is ``1``.

    rth : float, optional
        Similarity threshold :math:`r`. Default is ``0.2``.

    Returns
    -------
    float
        Approximate entropy value.
    """
    x = np.asarray(x)
    r = rth * np.std(x, ddof=1) # threshold of similarity
    phi = _app_samp_entropy(x, order=mnom, r=r, metric="chebyshev", approximate=True)

    return np.subtract(phi[0], phi[1])

def _embed(x: ArrayLike, order: int, delay: int = 1) -> ArrayLike:
    """Safe embedding that supports order=1."""
    x = np.asarray(x)
    if order < 1:
        raise ValueError("Order must be at least 1.")
    N = x.shape[0]
    if N - (order - 1) * delay <= 0:
        raise ValueError("Time series is too short for the given order and delay.")
    return np.array([x[i:i + order * delay:delay] for i in range(N - (order - 1) * delay)])

def _app_samp_entropy(
        x: ArrayLike, 
        order: int, 
        r: float, 
        metric: str = "chebyshev", 
        approximate: bool = True) -> ArrayLike:
    """Modified version of `_app_samp_entropy` that supports order=1."""
    phi = np.zeros(2)
    emb_data1 = _embed(x, order, 1)
    if approximate:
        pass
    else:
        emb_data1 = emb_data1[:-1]

    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True).astype(np.float64)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))

    return phi

def complexity_invariant_distance(y: ArrayLike) -> dict:
    """
    Complexity-invariant distance

    Computes two estimates of the 'complexity' of a time series based on the 
    stretched-out length of the lines in its line graph. These features are 
    based on the method described by Batista et al. (2014) [1], designed for use 
    in complexity-invariant distance calculations.

    References
    ----------
    .. [1] Batista, G. E. A. P. A., Keogh, E. J., Tataw, O. M., & de Souza, V. M. A. 
        (2014). CID: an efficient complexity-invariant distance for time series. 
        Data Mining and Knowledge Discovery, 28(3), 634–669. 
        https://doi.org/10.1007/s10618-013-0312-3

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
    min_CE1 = f_CE1(y_sorted)
    min_CE2 = f_CE2(y_sorted)

    CE1_norm = CE1 / min_CE1
    CE2_norm = CE2 / min_CE2

    out = {'CE1':       CE1,
           'CE2':       CE2,
           'minCE1':    min_CE1,
           'minCE2':    min_CE2,
           'CE1_norm':  CE1_norm,
           'CE2_norm':  CE2_norm}

    return out

def lempel_ziv_complexity(x: ArrayLike, n_bits: int = 2,
                          pre_proc: Union[str, None] = None, rng: int = 0) -> float:
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
    n_bits : int, optional
        Number of bits (alphabet size) to encode the data into (default: 2).
    pre_proc : str, optional
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
    if pre_proc == "diff":
        x = z_score(np.diff(x))

    if x.size == 0 or n_bits < 2:
        return 0.0

    symbols = _symbolise_lz(x, n_bits, rng)
    c = _lz_complexity(symbols)

    return (c * np.log(x.size)) / (x.size * np.log(n_bits))

@njit(cache=True, fastmath=True)
def _lz_complexity(symbols: np.ndarray) -> int:
    """
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
    """Helper function for lempel_ziv_complexity"""
    nx = x.size
    noisy = x + np.finfo(np.float64).eps * rng.randn(nx)
    order = np.argsort(noisy, kind="mergesort")
    ranks = np.arange(1, nx + 1)
    symbols = np.floor(ranks * (n_bins / (nx + 1)))

    out = np.empty_like(symbols)
    out[order] = symbols + 1
    return out
