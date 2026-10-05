from math import factorial
from typing import Optional, Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
import pywt
from numpy.typing import ArrayLike
from numba import njit
from antropy.entropy import _xlogx
from scipy.stats import norm, rankdata
from sklearn.neighbors import KDTree

from ..toolboxes.Michael_Small import shannon
from ..toolboxes.Max_Little import close_returns as _close_returns_c
from ..toolboxes.physionet import sampen as _sampen_c
from ..robust import bf_hist_edges, bf_ks_density, bf_random
from ..utils import (_ml_rng, _zscore_matlab, get_tau, make_buffer, pre_process,
                     time_delay_embed, z_score)


def _entropy_summary(ents: np.ndarray) -> dict:
    """Summary statistics on a set of entropy values."""
    return {
        'maxent': np.max(ents),
        'minent': np.min(ents),
        'medent': np.median(ents),
        'meanent': np.mean(ents),
        'stdent': np.std(ents, ddof=1),
    }


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
        The number of bins to discretize the time series into (i.e., alphabet size). Default is 2.
    depth : int or list of int, optional
        The length of strings to analyze. Default is 3.

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
            # Entropy scales with depth, so normalize by this factor
            out = shannon.entropy(y, int(num_bins), int(depth)) / int(depth)
        elif depth_range_size > 1:
            # Range over depths and return statistics on the results
            ents = np.array([
                shannon.entropy(y, int(num_bins), int(d)) / int(d) for d in depth
            ])
            out = _entropy_summary(ents)

    elif bin_range_size > 1:
        if depth_range_size == 1:
            # Statistics over different bin numbers (constant depth)
            # Entropy scales with depth, so normalize by this factor
            ents = np.array([
                shannon.entropy(y, int(n), int(depth)) / int(depth) for n in num_bins
            ])
            out = _entropy_summary(ents)
        elif depth_range_size > 1:
            raise NotImplementedError("Comparing both bins and depth not implemented.")

    return out

def distribution_entropy(
    y: ArrayLike,
    hist_or_ks: str = 'hist',
    num_bins: Union[str, int, float, None] = 10,
    olremp: float = 0
) -> float:
    """
    Distributional entropy.

    Estimates entropy from the distribution of a data vector. The distribution is estimated
    either using a histogram (equal-width bins spanning the data, see
    :func:`~pyhctsa.robust.bf_hist_edges`) with numBins bins, or as a kernel-smoothed
    distribution using a Gaussian kernel with a normal-reference bandwidth (see
    :func:`~pyhctsa.robust.bf_ks_density`).

    An optional additional parameter can be used to remove a proportion of the most extreme
    positive and negative deviations from the mean as an initial pre-processing step.

    Parameters
    ----------
    y : array-like
        The input time series.
    hist_or_ks : str
        Whether to use a histogram ('hist') or kernel-smoothed ('ks') distribution. Default is ``'hist'``.
    num_bins : int or str or float or None, optional

        - (for 'hist'): an integer, the number of equal-width bins; or the name of a rule for
          the number of bins ('sturges', 'fd', 'sqrt', 'auto'; written out as formulae in
          :func:`~pyhctsa.robust.bf_hist_edges`, so not NumPy's or MATLAB's ``histcounts``
          rules of the same names, which round the bin width to a 'nice' value);
        - (for 'ks'): a positive real number, the bandwidth (standard deviation of the Gaussian
          kernel) of the kernel density estimate; or empty (``''`` / ``None``) for the default
          bandwidth, the normal-reference rule ``sigma * (4 / (3 N)) ** (1/5)`` with
          ``sigma = median(|y - median(y)|) / 0.6745`` (see :func:`~pyhctsa.robust.bf_ks_density`).

        Default is 10.

    olremp : float, optional
        The proportion of outliers at both extremes to remove.
        (e.g., if olremp = 0.01; keeps only the middle 98% of data; 0 keeps all data. 
        This parameter ought to be less than 0.5, which keeps none of the data). Default is 0.

    Returns
    -------
    float
        Estimate of entropy from the distribution (in nats), or, if ``olremp`` is nonzero, the
        entropy of the full time series minus that of the trimmed time series. NaN if everything
        is removed by the trimming, or if the data (after trimming) are constant, for which the
        differential entropy is not defined.

    Notes
    -----
    The 'ks' density is evaluated on a 200-point grid spanning the 0.1% to 99.9% quantiles
    of ``y`` plus a 10% margin, converted to probability mass per grid cell and renormalized
    (NaN if those quantiles coincide). A fixed absolute bandwidth makes the estimate drift with
    the length of the series, so the automatic selection is preferable.
    """
    # (1) Remove outliers?
    y = np.asarray(y, dtype=float)
    if olremp != 0:
        y_hat = y[
            (y >= np.quantile(y, olremp, method='hazen')) &
            (y <= np.quantile(y, 1 - olremp, method='hazen'))
        ]
        if y_hat.size == 0:
            return np.nan
        return (
            distribution_entropy(y, hist_or_ks, num_bins)
            - distribution_entropy(y_hat, hist_or_ks, num_bins)
        )

    # (2) Form the histogram
    if hist_or_ks == 'hist':
        # use histogram to calculate pdf
        if np.ptp(y) == 0:  # constant: the differential entropy is not defined
            return np.nan
        if isinstance(num_bins, (int, np.integer)) and not isinstance(num_bins, bool):
            bin_edges = bf_hist_edges(y, int(num_bins))
        elif isinstance(num_bins, str) and num_bins in ['sturges', 'fd', 'sqrt', 'auto']:
            bin_edges = bf_hist_edges(y, num_bins)
        else:
            raise ValueError(
                f"Unknown binning method: {num_bins}. Choose either a valid rule or manually specify numBins."
            )
        # (the last bin includes its right edge, as MATLAB's histcounts)
        px = np.histogram(y, bins=bin_edges)[0].astype(float)
        px = px / np.sum(px)
        bin_widths = np.diff(bin_edges)

    elif hist_or_ks == 'ks':
        # Evaluate the kernel density estimate on an explicit, length-stable grid. The range
        # of a sample grows with N (as ~sqrt(2 log N) for Gaussian data), and with it the
        # log(bin width) term in the entropy sum, so the grid is anchored to extreme quantiles
        # instead (consistent estimators, so the interval converges as N grows).
        num_grid_pts = 200
        lo, hi = np.quantile(y, [0.001, 0.999], method='hazen')
        if not hi > lo:  # degenerate (near-constant) input
            return np.nan
        pad = 0.1 * (hi - lo)  # a little headroom beyond the quantile range
        xr = np.linspace(lo - pad, hi + pad, num_grid_pts)
        if num_bins is None or (isinstance(num_bins, str) and num_bins in ['', ' ', '[]', 'none']) \
                or (isinstance(num_bins, (list, tuple, np.ndarray)) and len(num_bins) == 0):
            bw = None  # the default (normal-reference) bandwidth
        elif isinstance(num_bins, (int, float, np.integer, np.floating)) and not isinstance(num_bins, bool):
            # uses the specified width (the standard deviation of the Gaussian kernel). NB: a
            # fixed absolute bandwidth makes the density estimate inconsistent (for consistency
            # the bandwidth must shrink with the sample size), so the smoothness of the
            # estimated density, and hence its entropy, drifts with N: hctsa no longer registers
            # the fixed-bandwidth settings, but the option remains for a specific smoothing scale.
            bw = float(num_bins)
        else:
            raise ValueError(
                f"Unknown type for {num_bins}. Either set to a float (which specifies the width, or leave empty.)"
            )
        px = bf_ks_density(y, xr, bw)[0]
        bin_widths = np.ones(len(px)) * (xr[1] - xr[0])
        # The density must be converted to probability mass per cell for the entropy sum
        # below (shared with 'hist'), and renormalized (the grid truncates some tail mass)
        px = px * bin_widths
        px = px / np.sum(px)

    else:
        raise ValueError(f"Unknown distribution estimator: {hist_or_ks}. Use 'hist' or 'ks'.")

    # (3) Compute the entropy sum and return it as output
    mask = px > 0
    p = px[mask]
    log_p = np.log(p / bin_widths[mask])
    out = -np.sum(p * log_p)

    if hist_or_ks == 'hist':
        # Miller-Madow correction for the downward bias of the plug-in estimate,
        # using this call's own sample size (y is y_hat inside the olremp recursion)
        out += (np.count_nonzero(px) - 1) / (2 * len(y))

    return out

_MAD_TO_SIGMA = 0.6745


def _bisquare_weights(r: np.ndarray) -> np.ndarray:
    return (np.abs(r) < 1) * (1 - r ** 2) ** 2


def _robustfit(x: np.ndarray, y: np.ndarray, tune: float = 4.685) -> tuple:
    """
    Robust straight-line fit by iteratively reweighted least squares with Tukey's bisquare
    weights: a port of MATLAB's ``[b, stats] = robustfit(x, y)`` (default options).

    Follows ``statrobustfit``: leverage-adjusted residuals, residual scale from the MAD
    (``median(|r|) / 0.6745`` over the largest residuals, excluding the smallest ``p - 1``),
    the same convergence rule, and the same standard errors (a robust estimate of the error
    scale combined with the OLS one, from ``statrobustsigma``).

    Returns
    -------
    (b, se) : tuple of ndarray
        The intercept and slope, and their standard errors. NaN arrays if the design is rank
        deficient or there are fewer than 3 points.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    X = np.column_stack([np.ones(n), x])
    p = 2
    nan2 = np.full(2, np.nan)
    if n <= p:
        return nan2, nan2

    Q, R = np.linalg.qr(X)
    tol = abs(R[0, 0]) * max(n, p) * np.finfo(float).eps
    if np.sum(np.abs(np.diag(R)) > tol) < p:
        return nan2, nan2
    b = np.linalg.solve(R, Q.T @ y)

    E = np.linalg.solve(R.T, X.T).T  # X / R
    h = np.minimum(0.9999, np.sum(E * E, axis=1))
    adjfactor = 1.0 / np.sqrt(1.0 - h)

    dfe = n - p
    ols_s = np.linalg.norm(y - X @ b) / np.sqrt(dfe)
    tiny_s = 1e-6 * np.std(y, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0

    def madsigma(r, rank):
        rs = np.sort(np.abs(r))
        return np.median(rs[max(1, rank) - 1:]) / _MAD_TO_SIGMA

    D = np.sqrt(np.finfo(float).eps)
    b0 = np.zeros(2)
    wxrank = p
    w = np.ones(n)
    it = 0
    while it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > 50:
            logger.warning("Iteration limit reached in robust fit")
            break
        r = y - X @ b
        radj = r * adjfactor
        s = madsigma(radj, wxrank)
        w = _bisquare_weights(radj / (max(s, tiny_s) * tune))
        b0 = b
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        b = np.linalg.lstsq(Xw, y * sw, rcond=None)[0]
        wxrank = int(np.linalg.matrix_rank(Xw))

    # Standard errors
    r = y - X @ b
    radj = r * adjfactor
    mad_s = madsigma(radj, p)
    if np.all((w < D) | (w > 1 - D)):
        included = w > 1 - D
        robust_s = np.linalg.norm(r[included]) / np.sqrt(np.sum(included) - p)
    else:
        # statrobustsigma
        st = max(mad_s, tiny_s) * tune
        u = radj / st
        phi = u * _bisquare_weights(u)
        delta = 0.0001
        u1 = u - delta
        phi0 = u1 * _bisquare_weights(u1)
        u1 = u + delta
        phi1 = u1 * _bisquare_weights(u1)
        dphi = (phi1 - phi0) / (2 * delta)
        m1 = np.mean(dphi)
        m2 = np.sum((1 - h) * phi ** 2) / (n - p)
        K = 1 + (p / n) * (1 - m1) / m1
        robust_s = K * np.sqrt(m2) * st / m1
    sigma = max(robust_s, np.sqrt((ols_s ** 2 * p ** 2 + robust_s ** 2 * n) / (p ** 2 + n)))
    RI = np.linalg.solve(R, np.eye(p))
    C = (RI @ RI.T) * sigma ** 2
    se = np.sqrt(np.maximum(np.finfo(float).eps, np.diag(C)))
    return b, se


def multi_scale_entropy(
    y: ArrayLike,
    scale_range: Optional[Union[list, range]] = None,
    m: int = 2,
    r: float = 0.15,
    pre_process_how: Optional[str] = None,
    what_entropy: str = 'sampen',
    num_classes: int = 6
) -> Union[dict, float]:
    """
    Multiscale entropy (MSE) of a time series.

    At each scale ``s`` the time series is coarse-grained by averaging over non-overlapping
    windows of ``s`` samples (scale 1 is the original series), and the entropy of the
    coarse-grained series is computed: by default the sample entropy, SampEn(m, r)
    (:func:`sample_entropy`), as in the multiscale entropy of Costa et al. [1]. Scales are
    handled as Composite Multiscale Entropy [2]: the value at scale ``s`` is the mean over all
    ``s`` possible starting offsets of the windows, instead of just offset 0 (single-offset
    estimates can swing several-fold depending on the arbitrary start). Offsets for which the
    coarse-grained series has fewer than 20 samples are omitted from the mean, and a scale
    for which all are is NaN. The entropy is also summarized across scales (extremes and where
    they occur, mean, spread, trend).

    References
    ----------
    .. [1] M. Costa, A. L. Goldberger and C.-K. Peng, "Multiscale entropy analysis of
        biological signals", Phys. Rev. E 71, 021906 (2005).
    .. [2] S.-D. Wu, C.-W. Wu, S.-G. Lin, C.-C. Wang and K.-Y. Lee, "Time series analysis using
        composite multiscale entropy", Entropy 15(3), 1069 (2013).
    .. [3] H. Azami, M. Rostaghi, D. Abasolo and J. Escudero, "Refined Composite Multiscale
        Dispersion Entropy and its Application to Biomedical Signals", IEEE Trans. Biomed. Eng.
        64(12), 2872 (2017).

    Parameters
    ----------
    y : array-like
        Input time series.
    scale_range : list or range, optional
        Scales (window sizes) for coarse-graining. Default is ``range(1, 11)``.
    m : int, optional
        Embedding dimension (length of the sequences to match). Default is 2.
    r : float, optional
        Similarity threshold for sample entropy, an absolute value (it is not rescaled with
        the scale). It is a fraction of the standard deviation of the input if ``y`` is
        z-scored. Unused for the dispersion settings. Default is 0.15.
    pre_process_how : str, optional
        Pre-processing applied (and the result z-scored) before coarse-graining:

        - 'diff1': incremental differences;
        - 'rescale_tau': first coarse-grain at the first zero-crossing of the autocorrelation function;
        - `None`: none.

        Default is `None`.
    what_entropy : {'sampen', 'dispen', 'fdispen'}, optional
        The entropy evaluated at each scale: sample entropy (``'sampen'``, the classical
        multiscale entropy), normalized dispersion entropy (``'dispen'``, i.e. multiscale
        dispersion entropy [3]; :func:`dispersion_entropy` with ``tau = 1``) or its
        fluctuation-based variant (``'fdispen'``). Output names carry the corresponding
        suffix (``dispen_s1``, ``meanDispEn``, ...). Default is ``'sampen'``.
    num_classes : int, optional
        The number of amplitude classes for the dispersion settings. Default is 6.

    Returns
    -------
    dict or float
        A dictionary with (names for ``what_entropy = 'sampen'``; ``'dispen'``/``'fdispen'``
        replace ``SampEn`` by ``DispEn``/``FDispEn`` and ``sampen`` by ``dispen``/``fdispen``):

        - 'sampen_s{k}': the entropy at each scale ``k`` in ``scale_range``;
        - 'maxSampEn', 'minSampEn': the maximum and minimum across scales, with
          'maxScale' and 'minScale' the scales at which they occur;
        - 'meanSampEn', 'stdSampEn', 'cvSampEn': the mean, standard deviation and
          coefficient of variation across scales;
        - 'meanch': the mean change from one scale to the next;
        - 'slope', 'slopeSE': the slope, and its standard error, of a robust (bisquare,
          as MATLAB's ``robustfit``) linear fit of the entropy against scale; NaN unless at
          least 4 scales have valid values.

        NaN (scalar) if no scale has enough samples.
    """
    y = np.asarray(y, dtype=float)
    m = int(m)
    if scale_range is None:
        scale_range = range(1, 11)
    scale_range = list(scale_range)
    min_ts_length = 20
    num_scales = len(scale_range)

    if what_entropy not in ('sampen', 'dispen', 'fdispen'):
        raise ValueError(
            f"Unknown entropy '{what_entropy}' (expected 'sampen', 'dispen' or 'fdispen')")
    en_name, en_prefix = {'sampen': ('SampEn', 'sampen'), 'dispen': ('DispEn', 'dispen'),
                          'fdispen': ('FDispEn', 'fdispen')}[what_entropy]

    # Pre-processing happens BEFORE the coarse-graining, and the result is z-scored
    if pre_process_how:
        y = pre_process(y, pre_process_how)
        if np.isscalar(y) or np.ndim(y) == 0:  # e.g., an undefined autocorrelation time
            logger.warning(f"Could not apply '{pre_process_how}' pre-processing")
            return np.nan
        y = _zscore_matlab(y)

    # Composite coarse-graining and entropy across scales: at each scale, the mean over all
    # `scale` possible non-overlapping starting offsets (Eq. (16) of Costa et al. is the
    # offset-0 coarse-graining)
    samp_ens = np.zeros(num_scales)
    for si, scale in enumerate(scale_range):
        scale = int(scale)
        offset_vals = np.full(scale, np.nan)
        for off in range(scale):
            y_cg = np.mean(make_buffer(y[off:], scale), axis=1) if y.size - off >= scale else np.empty(0)
            if len(y_cg) < min_ts_length:
                continue
            if what_entropy == 'sampen':
                offset_vals[off] = sample_entropy(y_cg, m, r)[f'sampen{m}']
            else:
                disp = dispersion_entropy(y_cg, m, num_classes, 1)
                if isinstance(disp, dict):
                    offset_vals[off] = disp['normDispEn' if what_entropy == 'dispen' else 'normFDispEn']
        samp_ens[si] = np.mean(offset_vals[~np.isnan(offset_vals)]) if not np.all(np.isnan(offset_vals)) else np.nan

    # Outputs: multiscale entropy
    if np.all(np.isnan(samp_ens)):
        pp_text = f"after {pre_process_how} pre-processing" if pre_process_how else ""
        logger.warning(f"Not enough samples ({len(y)} {pp_text}) to compute {en_name} at multiple scales")
        return np.nan

    # Output raw values
    out = {f'{en_prefix}_s{scale_range[i]}': samp_ens[i] for i in range(num_scales)}

    # Summary statistics of the variation (max, min, mean, std, diff all ignore NaN, as hctsa)
    valid = samp_ens[~np.isnan(samp_ens)]
    max_ind = int(np.nanargmax(samp_ens))
    min_ind = int(np.nanargmin(samp_ens))
    mean_val = np.mean(valid)
    std_val = np.std(valid, ddof=1) if valid.size > 1 else 0.0
    out[f'max{en_name}'] = samp_ens[max_ind]
    out['maxScale'] = scale_range[max_ind]
    out[f'min{en_name}'] = samp_ens[min_ind]
    out['minScale'] = scale_range[min_ind]
    out[f'mean{en_name}'] = mean_val
    out[f'std{en_name}'] = std_val
    with np.errstate(divide='ignore', invalid='ignore'):
        out[f'cv{en_name}'] = std_val / mean_val
    d = np.diff(samp_ens)
    d = d[~np.isnan(d)]
    out['meanch'] = np.mean(d) if d.size else np.nan

    # Trend across scales: a robust linear fit of the entropy against scale
    good = ~np.isnan(samp_ens)
    if good.sum() >= 4:
        b, se = _robustfit(np.asarray(scale_range, dtype=float)[good], samp_ens[good])
        out['slope'] = b[1]
        out['slopeSE'] = se[1]
    else:
        out['slope'] = np.nan
        out['slopeSE'] = np.nan

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
        Input time series.
    m : int, optional
        Maximum embedding dimension. Default is 2.
    r : float, optional
        Similarity threshold. If None, set to 0.1 * std(y). Default is `None`.
    pre_process_how : str, optional

        Preprocessing method:
            - 'diff1': Use first differences
            - `None`: No pre-processing.
        
        Default is `None`.

    Returns
    -------
    dict
        Dictionary containing:
            - 'sampen{m}': Sample entropy for each m from 0 to M
            - 'quadSampEn{m}': Quadratic sample entropy for each m
            - 'meanchsampen': Mean change in sample entropy values

        As in hctsa's ``sampen_mex``, ``sampen{k}`` (and ``quadSampEn{k}``) is NaN for
        ``k >= 1`` when no template of length ``k`` matched (there are no matches to
        form the ratio of), and 0 when templates of length ``k`` matched but none of
        length ``k + 1`` did.
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

def _ordinal_pattern_rank(x: np.ndarray) -> np.ndarray:
    """
    Index in 0..m!-1 of the ordinal pattern (the argsort permutation) of each row of `x`.

    The permutation is encoded by its Lehmer code, as hctsa's BF_OrdinalPatternRank
    does (ties are broken by position, as MATLAB's stable `sort`).
    """
    m = x.shape[1]
    ix = np.argsort(x, axis=1, kind='stable')
    rank = np.zeros(x.shape[0], dtype=np.int64)
    for k in range(m - 1):
        lehmer = np.sum(ix[:, k + 1:] < ix[:, [k]], axis=1)
        rank += lehmer * factorial(m - 1 - k)
    return rank

def permutation_entropy(y: ArrayLike, m: int = 2, tau: Union[int, str] = 1) -> dict:
    """
    Permutation Entropy (PermEn) of a time series.

    Computes the permutation entropy and its normalised version for a given time series,
    as described in [1], along with a weighted permutation entropy [2] and a
    measure of the time-reversal asymmetry of the ordinal patterns.

    The ordinal patterns are ranked as in hctsa's EN_PermEn (BF_OrdinalPatternRank).

    References
    ----------
    .. [1] C. Bandt and B. Pompe, "Permutation Entropy: A Natural 
        Complexity Measure for Time Series",
        Phys. Rev. Lett. 88(17) 174102 (2002).
    .. [2] B. Fadlallah, B. Chen, A. Keil and J. Principe, "Weighted-permutation
        entropy: A complexity measure for time series incorporating amplitude
        information", Phys. Rev. E 87, 022911 (2013).

    Parameters
    ----------
    y : array-like
        Input time series.
    m : int, optional
        Embedding dimension (order of the permutation entropy). Default is 2.
    tau : int or str, optional
        Time-delay for the embedding: an integer, or a rule understood by
        :func:`~pyhctsa.utils.get_tau`: ``'ac'`` (first zero-crossing of the
        autocorrelation function), ``'ac1e'`` (floor of its first 1/e crossing) or ``'mi'``
        (the smaller of the first minimum of the Kraskov automutual information and the
        ``'ac1e'`` delay). All outputs are NaN if the delay cannot be determined (e.g., a
        constant series). Default is 1.

    Returns
    -------
    dict
        A dictionary containing:

        - 'permEn': the permutation entropy (bits),
        - 'normPermEn': permEn normalized by log2(m!),
        - 'permEnLE': the permutation entropy of Bandt and Pompe with patterns of
          probability below 1/N floored at 1/N (natural log, divided by m - 1),
        - 'normWPE': the weighted permutation entropy normalized by log2(m!)
          (pattern probabilities are the sums of the variances of the m values of
          the embedding vectors with that pattern; NaN if all vectors are constant),
        - 'ordAsym': the total variation distance between the distribution of
          ordinal patterns of the series and that of the same embedding vectors read
          backward (0 for a time-reversible pattern distribution).
    """
    m = int(m)
    y = np.asarray(y)
    nan_out = {"permEn": np.nan, "normPermEn": np.nan, "permEnLE": np.nan,
               "normWPE": np.nan, "ordAsym": np.nan}
    tau = get_tau(y, tau)
    if np.isnan(tau):  # the delay could not be determined (e.g., constant series)
        return nan_out
    tau = int(tau)
    assert tau > 0, "delay must be greater than zero."

    try:
        embedded = time_delay_embed(y, m, tau)
    except ValueError:
        return nan_out
    nx = embedded.shape[0]
    if nx < 5:
        logger.warning("Time series too short to embed. Need at least 5 embedding vectors to compute permutation entropy.")
        return nan_out

    num_perms = factorial(m)
    perm_idx = _ordinal_pattern_rank(embedded)
    count_perms = np.bincount(perm_idx, minlength=num_perms)
    p = count_perms / nx
    pe = - _xlogx(p).sum()
    pe_norm = pe / np.log2(num_perms)

    # Permutation entropy with a floor of 1/N on the pattern probabilities
    p_le = np.maximum(1 / nx, p)
    pe_le = -np.sum(p_le * np.log(p_le)) / (m - 1) if m > 1 else np.nan

    # Weighted permutation entropy: each vector is weighted by the (population)
    # variance of its m values
    w = np.var(embedded, axis=1)
    if np.sum(w) > 0:
        pw = np.bincount(perm_idx, weights=w, minlength=num_perms) / np.sum(w)
        pw = pw[pw > 0]
        norm_wpe = -np.sum(pw * np.log2(pw)) / np.log2(num_perms)
    else:
        norm_wpe = np.nan

    # Time-reversal asymmetry of the ordinal patterns
    count_perms_rev = np.bincount(_ordinal_pattern_rank(embedded[:, ::-1]), minlength=num_perms)
    ord_asym = 0.5 * np.sum(np.abs(count_perms - count_perms_rev)) / nx

    return {"permEn": pe, "normPermEn": pe_norm, "permEnLE": pe_le,
            "normWPE": norm_wpe, "ordAsym": ord_asym}

def rpde(y: ArrayLike, m: int = 2, tau: Union[int, str] = 1, epsilon: float = 0.12, t_max: int = -1) -> dict:
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
        Embedding dimension. Default is 2.
    tau : int or str, optional
        Embedding time delay: an integer, or a rule understood by
        :func:`~pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``). NaN is returned if
        it cannot be determined (e.g. for a constant series). Default is 1.
    epsilon : float, optional
        Recurrence neighbourhood radius. Default is 0.12.
    t_max : int, optional
        Maximum recurrence time. If not specified, all recurrence times are returned. Default is -1.

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
    tau = get_tau(y, tau)
    if np.isnan(tau):
        # the delay could not be determined (e.g., constant series)
        logger.warning('Could not determine embedding parameters for this time series')
        return np.nan
    y = np.asarray(y)
    m = int(m)
    tau = int(tau)
    try:
        rpd = np.array(_close_returns_c.close_returns(y, m, tau, epsilon))
    except Exception:
        return np.nan
    if t_max > -1:
        rpd = rpd[:t_max]
    rpd = np.divide(rpd, np.sum(rpd))
    N = len(rpd)
    ip = rpd > 0
    H = -np.sum(rpd[ip] * np.log(rpd[ip]))
    H_norm = np.divide(H, np.log(N))  # log(N) is the H for an i.i.d. process

    return {
        'H': H,
        'H_norm': H_norm,
        'propNonZero': np.mean(ip),        # proportion of rpds that are non-zero
        'meanNonZero': np.mean(rpd[ip]) * N,  # mean value when rpd is non-zero (rescaled by N)
        'maxRPD': np.max(rpd) * N,         # maximum value of rpd (rescaled by N)
    }

def approximate_entropy(x: ArrayLike, mnom: int = 1, rth: float = 0.2,
                        tau: Union[int, str] = 1) -> float:
    """
    Approximate entropy (ApEn) of a time series.

    Computes :math:`\\mathrm{ApEn}(m, r)`, with delay vectors
    :math:`(x_i, x_{i+\\tau}, \\ldots, x_{i+(m-1)\\tau})`.

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
        Embedding dimension :math:`m`. Default is 1.0
    rth : float, optional
        Similarity threshold :math:`r`. Default is 0.2.
    tau : int or str, optional
        The time delay between the elements of a pattern: an integer, or a rule
        understood by :func:`~pyhctsa.utils.get_tau` (``'ac'``: first zero-crossing of the
        autocorrelation function; ``'ac1e'``: floor of its first 1/e crossing; ``'mi'``: the
        smaller of the first Kraskov automutual-information minimum and the ``'ac1e'``
        delay). The default of 1 uses consecutive samples.

    Returns
    -------
    float
        Approximate entropy value. NaN if the delay cannot be determined, or if the
        series is too short for the pattern length and delay (fewer than two delay
        vectors).
    """
    x = np.asarray(x)
    tau = get_tau(x, tau)
    if np.isnan(tau):  # the delay could not be determined (e.g., constant series)
        return np.nan
    tau = int(tau)
    mnom = int(mnom)
    # number of delay vectors of length m and m + 1
    if len(x) - (mnom - 1) * tau < 2 or len(x) - mnom * tau < 2:
        return np.nan
    r = rth * np.std(x, ddof=1) # threshold of similarity
    phi = _app_samp_entropy(x, order=mnom, r=r, metric="chebyshev", approximate=True, tau=tau)

    return np.subtract(phi[0], phi[1])

def _app_samp_entropy(
        x: ArrayLike,
        order: int,
        r: float,
        metric: str = "chebyshev", 
        approximate: bool = True,
        tau: int = 1) -> ArrayLike:
    """Modified version of `_app_samp_entropy` that supports order=1 and a time delay `tau`."""
    order = int(order)
    phi = np.zeros(2)
    emb_data1 = time_delay_embed(x, order, tau)
    if not approximate:
        emb_data1 = emb_data1[:-1]

    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True).astype(np.float64)
    emb_data2 = time_delay_embed(x, order + 1, tau)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))

    return phi

def bubble_entropy(y: ArrayLike, m: int = 10, tau: Union[int, str] = 1) -> Union[dict, float]:
    """
    Bubble entropy of a time series.

    Manis et al.'s bubble entropy [1], an ordinal entropy that depends very little on its
    embedding dimension. The series is cut into overlapping runs of ``m`` values spaced
    ``tau`` samples apart. For each run, the number of swaps a bubble sort needs to put it
    in order (equivalently, the number of pairs in which an earlier value exceeds a later
    one, from 0 to ``m(m-1)/2``) is counted. The Renyi entropy of order 2 of the
    distribution of this swap count, ``H_m = -log(sum_k p_k**2)``, is computed for runs of
    ``m`` and of ``m + 1`` values. The bubble entropy is the increase in entropy on going
    from ``m`` to ``m + 1`` values, ``H_(m+1) - H_m``, divided by ``log((m+1)/(m-1))`` to
    normalize for the dimension. Low values indicate series whose runs have predictable
    orderings. For white noise the long-series value is about 0.64 for ``m = 5`` and 0.69
    for ``m = 10`` (rising slowly toward 0.75 as ``m`` grows). Port of hctsa's
    ``EN_BubbleEn``.

    A swap is counted only when an earlier value is strictly greater than a later one, so
    tied values are never swapped (as in a standard bubble sort). The estimate is a small
    difference between two entropies, so it is noisy when the series is short relative to
    the number of possible swap counts: for series of about 1000 samples, embedding
    dimensions much above 10 give poorly reproducible values.

    With a delay set by the timescale of the series, the runs of ``m`` values span
    ``(m-1)*tau`` samples, so for a slowly decorrelating series (a large delay) there are
    few runs and the value is noisy. hctsa uses ``'mi'``, which is rarely NaN: it falls
    back on the first automutual-information minimum when the autocorrelation function
    never decays to 1/e. The ``'ac1e'`` delay is NaN when the autocorrelation function
    never decays to 1/e (as for many random walks) or the runs are too few, and the first
    zero crossing (``'ac'``) is less stable still.

    References
    ----------
    .. [1] G. Manis, M. D. Aktaruzzaman and R. Sassi, "Bubble Entropy: An Entropy Almost
        Free of Parameters", IEEE Trans. Biomed. Eng. 64(11), 2711 (2017).

    Parameters
    ----------
    y : array-like
        The input time series.
    m : int, optional
        The embedding dimension, at least 2. Default is 10.
    tau : int or str, optional
        The time delay for the embedding: an integer, or a rule understood by
        :func:`~pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.

    Returns
    -------
    dict
        A dictionary with a single field, 'bubbleEn', the bubble entropy
        ``(H_(m+1) - H_m) / log((m+1)/(m-1))``. NaN if the series is constant, the delay
        cannot be determined, or the series is too short for runs of ``m + 1`` values
        (fewer than 10 runs).
    """
    m = int(m)
    if m < 2:
        raise ValueError(f"The embedding dimension must be at least 2 (m = {m} given)")
    out = {'bubbleEn': np.nan}
    y = np.asarray(y, dtype=float).ravel()
    if not np.std(y, ddof=1) > 0:  # constant (or non-finite) series
        return out
    tau = get_tau(y, tau)  # resolve a rule once, so both dimensions share one delay
    if np.isnan(tau):
        return out
    tau = int(tau)

    # Renyi-2 entropy of the swap-count distribution at m and m + 1
    H = np.zeros(2)
    for j in range(2):
        mm = m + j
        n_vec = y.size - (mm - 1) * tau
        if n_vec < 10:
            return out
        x = time_delay_embed(y, mm, tau)
        num_swaps = np.zeros(n_vec, dtype=np.int64)
        for a in range(mm - 1):
            for b in range(a + 1, mm):
                num_swaps += x[:, a] > x[:, b]  # one swap per inversion
        p = np.bincount(num_swaps) / n_vec
        H[j] = -np.log(np.sum(p ** 2))

    out['bubbleEn'] = (H[1] - H[0]) / np.log((m + 1) / (m - 1))
    return out

def permutation_entropy_complexity(y: ArrayLike, m: int = 2, tau: Union[int, str] = 1) -> dict:
    """
    Jensen-Shannon statistical complexity of ordinal patterns.

    Computes the Bandt-Pompe ordinal-pattern distribution (as in
    :func:`permutation_entropy`) and pairs its normalized Shannon entropy with the
    Jensen-Shannon statistical complexity of Rosso et al. [1]: the entropy-complexity
    plane used to separate chaotic, stochastic and periodic dynamics that can look alike
    under entropy alone.

    Entropy is near its extremes (0 or ``log(m!)``) for both fully ordered *and* fully
    random sequences. The statistical complexity ``C = Q_J[P, P_uniform] * H[P]`` is
    instead close to zero at both those extremes and peaks for structured-but-disordered
    ('chaotic') ordinal-pattern distributions, a distinct axis of information from
    entropy alone. 'hNorm' reproduces the 'normPermEn' of :func:`permutation_entropy`.
    Port of hctsa's ``EN_PermEnComplexity``.

    At ``m = 2`` there are only two ordinal states, so H and C are both unimodal,
    symmetric functions of a single probability: they are then forced to be near-perfect
    reparameterizations of one another regardless of the input data, making
    'jsComplexity' redundant with plain permutation entropy at that order. ``m = 3`` was
    also found redundant on real-world data; only ``m = 4`` and ``m = 5`` are registered
    in the default hctsa feature set.

    References
    ----------
    .. [1] O.A. Rosso, H.A. Larrondo, M.T. Martin, A. Plastino and M.A. Fuentes,
        "Distinguishing noise from chaos", Phys. Rev. Lett. 99, 154102 (2007).
    .. [2] M.T. Martin, A. Plastino and O.A. Rosso, "Generalized statistical complexity
        measures: Geometrical and analytical properties", Physica A 369(2), 439 (2006),
        for the Q_0 normalization.
    .. [3] P.W. Lamberti, M.T. Martin, A. Plastino and O.A. Rosso, "Intensive entropic
        non-triviality measure", Physica A 334(1-2), 119 (2004).

    Parameters
    ----------
    y : array-like
        The input time series.
    m : int, optional
        The embedding dimension (order of the ordinal patterns). Default is 2.
    tau : int or str, optional
        The time delay for the embedding: an integer, or a rule understood by
        :func:`~pyhctsa.utils.get_tau` (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.

    Returns
    -------
    dict
        A dictionary containing:

        - 'hNorm': the normalized Shannon entropy of the ordinal-pattern distribution,
          ``H[P] = S[P] / log2(m!)``, in [0, 1],
        - 'jsComplexity': the Jensen-Shannon statistical complexity,
          ``C[P] = Q_J[P, P_uniform] * H[P]``, in [0, 1].

        Both are NaN if the delay cannot be determined or the series is too short to
        embed (fewer than 5 embedding vectors).
    """
    m = int(m)
    nan_out = {'hNorm': np.nan, 'jsComplexity': np.nan}
    y = np.asarray(y, dtype=float).ravel()
    tau = get_tau(y, tau)
    if np.isnan(tau):
        return nan_out
    tau = int(tau)
    if y.size - (m - 1) * tau < 5:  # need at least 5 embedding vectors
        logger.warning("Time series too short to embed")
        return nan_out
    x = time_delay_embed(y, m, tau)
    nx = x.shape[0]

    num_perms = factorial(m)
    p = np.bincount(_ordinal_pattern_rank(x), minlength=num_perms) / nx

    # Normalized Shannon entropy of P
    p_0 = p[p > 0]
    s_p = -np.sum(p_0 * np.log2(p_0))
    s_max = np.log2(num_perms)
    h_norm = s_p / s_max

    # Jensen-Shannon statistical complexity of P relative to the uniform distribution Pe
    pe = 1 / num_perms
    p_mix = (p + pe) / 2  # every entry > 0, since pe > 0
    s_mix = -np.sum(p_mix * np.log2(p_mix))
    js_div = s_mix - s_p / 2 - s_max / 2  # the entropy of Pe is log2(m!)

    # Normalizing constant so that Q_J is in [0, 1], attained for P a point mass
    n = float(num_perms)
    q0 = -2 / (((n + 1) / n) * np.log2(n + 1) - 2 * np.log2(2 * n) + np.log2(n))

    return {'hNorm': h_norm, 'jsComplexity': q0 * js_div * h_norm}

def wavelet_entropy(y: ArrayLike, wavelet_name: str = 'sym4', level: int = 5) -> float:
    """
    Wavelet entropy of a time series.

    Decomposes ``y`` via the maximal-overlap discrete wavelet transform (MODWT) into
    ``level`` detail scales plus the remaining smooth (scaling) band, i.e., ``level + 1``
    bands, computes each band's share of the signal's total energy,
    ``p_j = E_j / sum(E)``, and returns the Shannon entropy of this relative-energy
    distribution across bands, normalized to [0, 1] by its maximum possible value,
    ``log2(level + 1)`` [1]. Low values mean the energy is concentrated in few bands;
    high values that it is spread evenly across bands. Port of hctsa's ``EN_wentropy``
    (MATLAB's ``wentropy`` with a global energy distribution of the MODWT); the MODWT is
    computed here by the pyramid algorithm with circular boundary handling, using the
    filters of PyWavelets.

    The output is invariant to rescaling ``y``, and is bounded in [0, 1] (the value 1 is
    reached when the energy is equal in all ``level + 1`` bands). ``level`` is fixed by
    default (rather than left to depend on the series length) because the number of
    levels sets the normalizing denominator, so letting it grow with the length of ``y``
    introduces a strong length dependence. With ``level`` fixed, the value for white noise
    is independent of length (about 0.75). ``level = 5`` needs about 64 samples or more
    for a non-degenerate decomposition.

    References
    ----------
    .. [1] O. A. Rosso, S. Blanco, J. Yordanova, V. Kolev, A. Figliola, M. Schuermann,
        E. Basar, "Wavelet entropy: a new tool for analysis of short duration brain
        electrical signals", J. Neurosci. Methods 105(1), 65 (2001).

    Parameters
    ----------
    y : array-like
        The input time series.
    wavelet_name : str, optional
        The wavelet used for the MODWT decomposition, a PyWavelets name for an orthogonal
        wavelet (e.g., ``'sym4'``, ``'db2'``, ``'haar'``). Default is ``'sym4'``.
    level : int, optional
        The number of decomposition levels. Default is 5.

    Returns
    -------
    float
        The normalized wavelet entropy. NaN if the decomposition fails (e.g., for a series
        too short for the requested number of levels: ``level`` may not exceed
        ``floor(log2(N))``) or the series has no energy.
    """
    y = np.asarray(y, dtype=float).ravel()
    level = int(level)
    n = y.size
    if level < 1 or n < 2 or level > int(np.floor(np.log2(n))) or not np.all(np.isfinite(y)):
        return np.nan
    try:
        wav = pywt.Wavelet(wavelet_name)
    except ValueError:
        return np.nan
    g = np.asarray(wav.dec_lo) / np.sqrt(2)  # MODWT scaling and wavelet filters
    h = np.asarray(wav.dec_hi) / np.sqrt(2)
    idx = np.arange(n)

    # MODWT pyramid algorithm with circular boundary: the band energies do not depend on
    # the (circular) time alignment of the coefficients
    v = y
    energy = np.zeros(level + 1)
    for j in range(1, level + 1):
        shifts = 2 ** (j - 1) * np.arange(g.size)
        gather = v[(idx[:, None] - shifts[None, :]) % n]
        energy[j - 1] = np.sum((gather @ h) ** 2)
        v = gather @ g
    energy[level] = np.sum(v ** 2)

    total = energy.sum()
    if not total > 0:
        return np.nan
    p = energy / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(level + 1))

_RANDOMIZE_STATS = ('xcn1', 'xc1', 'd1', 'ac1', 'ac2', 'ac3', 'ac4', 'permen3_1', 'statav5',
                    'swss5_1')


def _randomize_stats(y: np.ndarray, y_rand: np.ndarray) -> list:
    """The ten statistics comparing a series, ``y``, with a randomized version, ``y_rand``."""
    from .stationarity import sliding_window, stat_av
    from .correlation import autocorr
    n = y.size

    # Cross-correlation with the original signal at lags -1 and +1 (xcorr 'coeff')
    norm_xc = np.sqrt(np.sum(y ** 2) * np.sum(y_rand ** 2))
    with np.errstate(all='ignore'):
        xcn1 = np.sum(y[:-1] * y_rand[1:]) / norm_xc
        xc1 = np.sum(y[1:] * y_rand[:-1]) / norm_xc

    # Norm of the differences between the original and randomized signals
    d1 = np.linalg.norm(y - y_rand) / n

    def safe(f, *args):
        try:
            return float(f(*args))
        except Exception:  # data-dependent failure (e.g., a series too short): NaN
            return np.nan

    with np.errstate(all='ignore'):
        ac = np.asarray(autocorr(y_rand, [1, 2, 3, 4], 'Fourier'), dtype=float).ravel()
    if ac.size != 4:
        ac = np.full(4, np.nan)

    # Normalized permutation entropy, PermEn(3, 1)
    permen3_1 = safe(lambda v: permutation_entropy(v, 3, 1)['normPermEn'], y_rand)
    # Stationarity
    statav5 = safe(stat_av, y_rand, 'seg', 5)
    swss5_1 = safe(sliding_window, y_rand, 'std', 'std', 5, 1)
    return [xcn1, xc1, d1, ac[0], ac[1], ac[2], ac[3], permen3_1, statav5, swss5_1]


def _randomize_run(y: np.ndarray, randomize_how: str, draws: np.ndarray) -> np.ndarray:
    """
    Randomize ``y`` one point at a time for ``2N`` steps, recording the statistics at the
    start and every ``N/10`` steps.

    ``draws`` has shape ``(2N, 2)``: the (0-based) random indices consumed by each step,
    in the order they are drawn.
    """
    n = y.size
    num_calcs = 2.0 / 0.1  # randp_max / rand_inc
    calc_ints = int(np.floor(2 * n / num_calcs))
    if calc_ints == 0:
        calc_ints = 1  # round up for short time series
    calc_pts = list(range(0, 2 * n + 1, calc_ints))
    if calc_pts[-1] != 2 * n:
        calc_pts.append(2 * n)
    row_of = {pt: k for k, pt in enumerate(calc_pts)}

    stats = np.zeros((len(calc_pts), len(_RANDOMIZE_STATS)))
    y_rand = y.copy()
    stats[0] = _randomize_stats(y, y_rand)  # initial condition: apply on itself

    for i in range(1, 2 * n + 1):
        a, b = draws[i - 1]
        if randomize_how == 'statdist':
            # substitute a random element by a random element of the original series
            # (MATLAB evaluates the right-hand index first: the first draw is the source)
            y_rand[b] = y[a]
        elif randomize_how == 'dyndist':
            # substitute a random element by a random element of the current,
            # already partially randomized, series
            y_rand[b] = y_rand[a]
        elif randomize_how == 'permute':
            # swap two random elements, so that the distribution never changes
            y_rand[a], y_rand[b] = y_rand[b], y_rand[a]
        else:
            raise ValueError(f"Unknown randomization method '{randomize_how}'.")
        k = row_of.get(i)
        if k is not None:
            stats[k] = _randomize_stats(y, y_rand)
    return stats


def _randomize_fit(stats: np.ndarray) -> dict:
    """Exponential fits and summaries of the trajectory of each statistic."""
    from ..toolboxes.matlab.matlab_fit import goodness_of_fit, lsqcurvefit_trr

    def model2(p, x):
        return p[0] * np.exp(p[1] * x)

    def model3(p, x):
        return p[0] * np.exp(p[1] * x) + p[2]

    r = np.arange(1, stats.shape[0] + 1, dtype=float)  # an 'x-axis' for the fits
    out = {}
    for i, name in enumerate(_RANDOMIZE_STATS):
        v = stats[:, i]
        if name in ('xcn1', 'xc1'):
            model, start = model2, [v[0], -0.1]
        elif name in ('ac1', 'ac2', 'ac3'):
            model, start = model2, [v[0], -0.2]
        elif name == 'ac4':
            model, start = model2, [v[0], -0.4]
        elif name in ('d1', 'permen3_1'):
            model, start = model3, [-v[-1], -0.2, v[-1]]
        else:  # statav5, swss5_1
            model, start = model3, [-v[-1], -0.1, v[-1]]
        num_coeffs = len(start)

        # Exponential fit (a * exp(b * k), plus an offset c for some), as MATLAB's fit
        try:
            with np.errstate(all='ignore'):
                p = np.asarray(lsqcurvefit_trr(model, start, r, v), dtype=float)
                gof = goodness_of_fit(v, model(p, r), num_coeffs)
            if not np.all(np.isfinite(p)):
                raise ValueError('non-finite fit')
        except Exception:
            p = np.full(num_coeffs, np.nan)
            gof = {'rsquare': np.nan, 'rmse': np.nan}
        out[name + 'fexpa'] = p[0]
        out[name + 'fexpb'] = p[1]
        if num_coeffs == 3:
            out[name + 'fexpc'] = p[2]
        out[name + 'fexpr2'] = gof['rsquare']
        out[name + 'fexprmse'] = gof['rmse']

        # Extra statistics: the absolute change, and the first checkpoint at which the
        # statistic passes halfway between its start and end values
        out[name + 'diff'] = abs(v[-1] - v[0])
        half = 0.5 * (v[-1] + v[0])
        passed = np.flatnonzero(v > half) if v[-1] > v[0] else np.flatnonzero(v < half)
        out[name + 'hp'] = float(passed[0] + 1) if passed.size else np.nan
    return out


def randomize(y: ArrayLike, randomize_how: str = 'statdist',
              random_seed: Union[int, str, None] = None) -> dict:
    """
    How properties of the series change as it is progressively randomized.

    Randomizes a copy of the input (z-scored) series one point at a time, according to a
    randomization procedure, repeated ``2N`` times for a series of length ``N``, and
    compares statistics of the randomized copy with the original at 21 checkpoints: at the
    start and after every ``N/10`` steps. Port of hctsa's ``EN_Randomize``.

    The random draws are those of MATLAB's Mersenne Twister (``rng(seed, 'twister')``,
    ``randi``) when a seed is given, so the result is reproducible and, for the same seed,
    follows the same randomization as hctsa.

    Parameters
    ----------
    y : array-like
        The input (z-scored) time series.
    randomize_how : {'statdist', 'dyndist', 'permute'}, optional
        What one step of randomization does:

        - ``'statdist'``: overwrites a random element of the series with a randomly chosen
          element of the original series,
        - ``'dyndist'``: overwrites a random element of the series with another random
          element of the current, partially randomized, series,
        - ``'permute'``: swaps two randomly chosen elements of the series, so that the
          distribution of values never changes and only the temporal properties do.

        Default is ``'statdist'``.
    random_seed : int or {'default', 'none'}, optional
        How to set the random seed, as hctsa's ``BF_ResetSeed``: an integer seed;
        ``'default'`` (or None) seeds with 0; ``'none'`` does not seed (the run is then not
        reproducible). Default is None.

    Returns
    -------
    dict
        For each of ten statistics measured at each checkpoint, six or seven fields
        describing its trajectory over the 21 checkpoints. The statistics are:

        - 'xcn1', 'xc1': the cross-correlation of the original and randomized series at
          lags -1 and +1,
        - 'd1': the distance between the original and randomized series,
          ``norm(y - y_rand) / N``,
        - 'ac1', 'ac2', 'ac3', 'ac4': the autocorrelation of the randomized series at
          lags 1 to 4,
        - 'permen3_1': the normalized permutation entropy of the randomized series,
          PermEn(3, 1),
        - 'statav5': StatAv with 5 segments (the standard deviation of the segment means),
        - 'swss5_1': the standard deviation across 5 non-overlapping windows of the local
          standard deviation, relative to the full-series standard deviation.

        The fields are named by joining a statistic's name to a suffix. Fits of
        ``a * exp(b * k)`` (``k`` the checkpoint number 1..21) for 'xcn1', 'xc1', 'ac1',
        'ac2', 'ac3' and 'ac4' have the suffixes 'fexpa', 'fexpb' (the parameters),
        'fexpr2' (R^2), 'fexprmse' (the standard error of the fit), 'diff' and 'hp'. Fits
        of ``a * exp(b * k) + c`` for 'd1', 'permen3_1', 'statav5' and 'swss5_1' have the
        same suffixes plus 'fexpc' (the offset ``c``). In all cases 'diff' is the absolute
        change ``|s_end - s_start|`` of the statistic between the first and last
        checkpoints and 'hp' is the number of the first checkpoint at which the statistic
        passes halfway between its start and end values (NaN if it never does).

    Notes
    -----
    'diff' is an absolute change, not a change relative to the starting value, because the
    starting value (e.g., the autocorrelation of the original series at lag 2) can be near
    0, where a relative change is unstable. The exponential fits use a port of MATLAB's
    trust-region nonlinear least squares and the same starting points as hctsa; a fit that
    fails gives NaN.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if randomize_how not in ('statdist', 'dyndist', 'permute'):
        raise ValueError(f"Unknown randomization method '{randomize_how}'.")
    if not np.isclose(np.mean(y), 0, atol=1e-6) or not np.isclose(np.std(y, ddof=1), 1, atol=1e-6):
        logger.warning('The input time series should be z-scored for randomize.')

    # Random indices, in the order a MATLAB run draws them (randi(N) = floor(N*rand) + 1)
    if random_seed is None or (isinstance(random_seed, str) and random_seed == 'default'):
        rng = _ml_rng(0)
    elif isinstance(random_seed, str):
        if random_seed != 'none':
            raise ValueError(f"Not sure how to reset using '{random_seed}'")
        rng = np.random.RandomState()
    else:
        rng = _ml_rng(int(random_seed))
    draws = np.floor(n * rng.random_sample(4 * n)).astype(np.int64).reshape(2 * n, 2)

    return _randomize_fit(_randomize_run(y, randomize_how, draws))

def dispersion_entropy(y: ArrayLike, m: int = 2, c: int = 6, tau: Union[int, str] = 1,
                       mapping_how: str = 'ncdf') -> Union[dict, float]:
    """
    Dispersion entropy of a time series.

    Maps the time series onto ``c`` amplitude classes, replaces each run of ``m`` values
    (spaced ``tau`` samples apart) by the sequence of classes it visits (a 'dispersion
    pattern'), and returns the Shannon entropy of the resulting pattern distribution.
    Port of hctsa's ``EN_DispEn``.

    Unlike permutation entropy (:func:`permutation_entropy`), which records only the rank
    ordering within each embedding vector and so discards amplitude information entirely
    ([1, 2, 3] and [1, 2, 300] are the same pattern), dispersion entropy assigns each point
    to an amplitude class first, so the size of an excursion, not just its direction, shapes
    the symbol sequence. It is also markedly cheaper than sample entropy and degrades more
    gracefully on short, noisy series.

    The fluctuation-based variant is also returned. It symbolizes the differences between
    successive classes rather than the classes themselves, and so responds to the size of
    class-to-class changes rather than to absolute amplitude level.

    References
    ----------
    .. [1] M. Rostaghi and H. Azami, "Dispersion Entropy: A Measure for Time-Series
        Analysis", IEEE Signal Processing Letters 23(5) 610 (2016).
    .. [2] H. Azami and J. Escudero, "Amplitude- and Fluctuation-Based Dispersion
        Entropy", Entropy 20(3) 210 (2018).

    Parameters
    ----------
    y : array-like
        The input time series.
    m : int, optional
        The embedding dimension. The number of possible patterns grows as ``c**m``, so ``m``
        must stay small for the pattern frequencies to be estimable. Default is 2.
    c : int, optional
        The number of amplitude classes (must be at least 2). Default is 6.
    tau : int or str, optional
        The time delay: an integer, or a rule understood by :func:`~pyhctsa.utils.get_tau`
        (``'ac'``, ``'ac1e'``, ``'mi'``). Default is 1.
    mapping_how : {'ncdf', 'linear'}, optional
        How to map the time series onto (0, 1) before classifying:

        - ``'ncdf'``: the normal cumulative distribution function with the series' own mean
          and standard deviation (the mapping the method was introduced with; a linear
          mapping assigns most points to a few classes whenever the maximum or minimum is far
          from the median, so a single outlier can collapse the symbolization).
        - ``'linear'``: a min-max rescaling onto [0, 1] (outlier-sensitive).

        Default is ``'ncdf'``.

    Returns
    -------
    dict or float
        A dictionary with:

        - 'dispEn': the dispersion entropy (nats),
        - 'normDispEn': 'dispEn' normalized by its maximum possible value, ``log(c**m)``,
        - 'fDispEn': the fluctuation-based dispersion entropy (nats; NaN for ``m = 1``),
        - 'normFDispEn': 'fDispEn' normalized by ``log((2c-1)**(m-1))`` (NaN for ``m = 1``).

        NaN (scalar) if the delay cannot be determined, the series is constant, or it is
        too short for the embedding (fewer than 5 embedding vectors).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    m = int(m)
    c = int(c)
    if c < 2:
        raise ValueError(f"Need at least two amplitude classes (c = {c} given)")
    if m < 1:
        raise ValueError(f"Embedding dimension must be at least 1 (m = {m} given)")

    tau = get_tau(y, tau)
    if np.isnan(tau):  # data-dependent: no correlation length could be estimated
        return np.nan
    tau = int(tau)

    num_vectors = n - (m - 1) * tau
    if num_vectors < 5:
        logger.warning(f"Time series (N = {n}) too short for dispersion entropy at m = {m}, tau = {tau}")
        return np.nan

    # Map the series onto (0, 1), then onto the c amplitude classes
    if mapping_how == 'ncdf':
        sigma = np.std(y, ddof=1)
        if not sigma > 0:
            logger.warning("Constant time series has no dispersion structure")
            return np.nan
        y_mapped = norm.cdf(y, loc=np.mean(y), scale=sigma)
    elif mapping_how == 'linear':
        y_range = np.max(y) - np.min(y)
        if not y_range > 0:
            logger.warning("Constant time series has no dispersion structure")
            return np.nan
        y_mapped = (y - np.min(y)) / y_range
    else:
        raise ValueError(f"Unknown mapping '{mapping_how}' (expected 'ncdf' or 'linear')")

    # Classes 1..c as z = round(c*y + 0.5) (Rostaghi & Azami); round half away from zero as
    # MATLAB (the argument is positive), then clamp into 1..c (the top of the range rounds to c+1)
    v = c * y_mapped + 0.5
    z = np.floor(v)
    z = z + ((v - z) >= 0.5)
    z = np.clip(z, 1, c).astype(np.int64)

    # Z[i, k] = z[i + k*tau], one row per embedding vector
    emb = z[np.arange(num_vectors)[:, None] + np.arange(m) * tau]

    # Dispersion entropy: the patterns are the class sequences themselves (c^m of them),
    # encoded as base-c integers
    place_values = c ** np.arange(m - 1, -1, -1, dtype=np.int64)
    pattern_idx = (emb - 1) @ place_values
    p = np.bincount(pattern_idx, minlength=c ** m) / num_vectors
    p = p[p > 0]
    disp_en = -np.sum(p * np.log(p))
    out = {'dispEn': disp_en, 'normDispEn': disp_en / np.log(float(c) ** m)}

    # Fluctuation-based: the patterns are the successive class differences, each in
    # -(c-1)..(c-1), giving (2c-1)^(m-1) patterns
    if m < 2:
        out['fDispEn'] = np.nan
        out['normFDispEn'] = np.nan
        return out
    d_z = np.diff(emb, axis=1) + (c - 1)  # shift -(c-1)..(c-1) onto 0..2c-2
    num_fluct = (2 * c - 1) ** (m - 1)
    f_place_values = (2 * c - 1) ** np.arange(m - 2, -1, -1, dtype=np.int64)
    f_idx = d_z @ f_place_values
    p_f = np.bincount(f_idx, minlength=num_fluct) / num_vectors
    p_f = p_f[p_f > 0]
    f_disp_en = -np.sum(p_f * np.log(p_f))
    out['fDispEn'] = f_disp_en
    out['normFDispEn'] = f_disp_en / np.log(float(num_fluct))
    return out

def fuzzy_entropy(y: ArrayLike, M: int = 2, r: float = 0.2, n: float = 2) -> dict:
    """
    Fuzzy entropy of a time series.

    Chen et al.'s fuzzy entropy [1, 2], a smooth relative of sample entropy
    (:func:`sample_entropy`). The series is cut into overlapping runs of ``m``
    consecutive values, and the mean of each run is subtracted from it, so runs are
    compared by shape and not by level. Two runs are not simply 'matching' or 'not
    matching', as in sample entropy: they are given a similarity ``exp(-(d/r)**n)``,
    where ``d`` is the largest absolute difference between corresponding values of the
    two (baseline-removed) runs. ``phi_m`` is the mean similarity over all pairs of
    distinct runs of length ``m``, and the fuzzy entropy at dimension ``m`` is
    ``log(phi_m) - log(phi_(m+1))``. Low values indicate regular, predictable series;
    high values irregular ones. The smooth similarity makes the measure continuous in
    ``r`` and defined for short series for which sample entropy would find no matches.

    All runs of length 1, ..., M+1 are taken from the same ``N - M`` starting points, so
    that successive dimensions are compared on the same footing. The distances are
    computed in blocks, so memory use does not grow with the square of the series length
    (the run time does: it is O(N^2 M)). Port of hctsa's ``EN_FuzzyEn``.

    The similarity is written here as ``exp(-(d/r)**n)``, so that ``r`` is a distance (in
    units of the standard deviation of ``y``). Chen et al. (2007) write it as
    ``exp(-d**n/r)``, in which ``r`` is not a distance: for ``n = 2``, their ``r = 0.2``
    on standardized data is a Gaussian width of ``sqrt(0.2) = 0.45`` standard deviations,
    against 0.2 here. Values of ``r`` are therefore not directly comparable with those
    quoted in that paper. To get the fuzzy entropy of the increments of a series, give
    ``np.diff(y)`` as the input.

    References
    ----------
    .. [1] W. Chen, Z. Wang, H. Xie and W. Yu, "Characterization of surface EMG signal
        based on fuzzy entropy", IEEE Trans. Neural Syst. Rehabil. Eng. 15(2), 266
        (2007).
    .. [2] W. Chen, J. Zhuang, W. Yu and Z. Wang, "Measuring complexity using FuzzyEn,
        ApEn, and SampEn", Med. Eng. Phys. 31(1), 61 (2009).

    Parameters
    ----------
    y : array-like
        The input time series.
    M : int, optional
        The largest embedding dimension: the fuzzy entropy is returned for
        ``m = 1, ..., M``. Default is 2.
    r : float, optional
        The width of the similarity function, as a fraction of the standard deviation of
        ``y`` (the width in the units of ``y`` is ``r * std(y)``, so the measure is
        unchanged by any rescaling of ``y``). Default is 0.2.
    n : float, optional
        The exponent of the similarity function ``exp(-(d/r)**n)`` (larger values make
        the similarity closer to a hard threshold). Default is 2.

    Returns
    -------
    dict
        Fields 'fuzzyEn1', 'fuzzyEn2', ..., 'fuzzyEnM': the fuzzy entropy at each
        embedding dimension (nats). At ``m = 1`` the run mean removed is the value itself,
        so ``phi_1 = 1`` and 'fuzzyEn1' is ``-log(phi_2)``. All fields are NaN if the
        series is constant, has fewer than ``M + 3`` points, or has no pair of runs with
        a nonzero similarity.
    """
    y = np.asarray(y, dtype=float).ravel()
    M = int(M)
    N = y.size
    out = {f'fuzzyEn{m}': np.nan for m in range(1, M + 1)}

    sd = np.std(y, ddof=1) if N > 1 else np.nan
    Nv = N - M  # number of starting points shared by every embedding dimension
    if not np.isfinite(sd) or sd == 0 or Nv < 3:
        return out
    width = r * sd

    phi = np.zeros(M + 1)
    block_size = max(1, int(2e6 // Nv))  # cap the size of the distance block
    for m in range(1, M + 2):
        Z = y[np.arange(Nv)[:, None] + np.arange(m)[None, :]]
        Z = Z - Z.mean(axis=1, keepdims=True)  # remove each run's own mean (local baseline)
        total = 0.0
        for i0 in range(0, Nv, block_size):
            Zi = Z[i0:i0 + block_size]
            D = np.abs(Zi[:, 0][:, None] - Z[:, 0][None, :])
            for k in range(1, m):
                np.maximum(D, np.abs(Zi[:, k][:, None] - Z[:, k][None, :]), out=D)  # Chebyshev
            total += np.exp(-(D / width) ** n).sum() - Zi.shape[0]  # drop self-similarity (=1)
        phi[m - 1] = total / (Nv * (Nv - 1))

    if np.any(phi <= 0):
        return out
    for m in range(1, M + 1):
        out[f'fuzzyEn{m}'] = np.log(phi[m - 1]) - np.log(phi[m])
    return out

def complexity_invariant_distance(y: ArrayLike) -> dict:
    """
    Complexity-invariant distance.

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

    # Original definition (Table 2 of the cited paper). sum -> mean to deal with
    # non-equal time-series lengths (now scales properly with length).
    def f_CE1(v):
        return np.sqrt(np.mean(np.power(np.diff(v), 2)))

    # Definition corresponding to the line segment example in Fig. 9 of the cited
    # paper (using Pythagoras's theorem).
    def f_CE2(v):
        return np.mean(np.sqrt(1 + np.power(np.diff(v), 2)))

    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    # Defined as a proportion of the minimum value possible for this time series,
    # attained by placing close values close together; i.e., sorting the series.
    y_sorted = np.sort(y)
    min_CE1 = f_CE1(y_sorted)
    min_CE2 = f_CE2(y_sorted)

    return {
        'CE1': CE1,
        'CE2': CE2,
        'minCE1': min_CE1,
        'minCE2': min_CE2,
        'CE1_norm': CE1 / min_CE1,
        'CE2_norm': CE2 / min_CE2,
    }

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
        Number of bits (alphabet size) to encode the data into. Default is 2.
    pre_proc : str, optional
        Preprocessing method to apply before symbolization. Currently supported:

            - 'diff': Use z-scored first differences of the time series.
            - `None`: No pre-processing.

        Default is `None`.

    rng : int, optional
        Unused (ties are now broken deterministically by average ranks); kept for
        backward compatibility. Default is 0.

    Returns
    -------
    float
        Normalized Lempel-Ziv complexity: the number of distinct symbol sequences
        divided by the expected number for a noise sequence.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if pre_proc == "diff":
        x = z_score(np.diff(x))

    if x.size == 0 or n_bits < 2:
        return 0.0

    symbols = _symbolise_lz(x, n_bits)
    c = _lz_complexity(symbols)

    # normalize by the number of symbols actually present (as MS_complexitybs.c does),
    # which can be fewer than n_bits when many values are tied
    bins = int(symbols.max())

    return (c * np.log(x.size)) / (x.size * np.log(bins))

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
        # brute-force search over all start positions i < ns (Q may overlap itself)
        for i in range(ns):
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

def _symbolise_lz(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Helper function for lempel_ziv_complexity: equiprobable symbols from ranks.

    Tied values share an (average) rank and therefore always get the same symbol.
    """
    nx = x.size
    ranks = rankdata(x, method="average")
    return np.floor(ranks * (n_bins / (nx + 1))) + 1
