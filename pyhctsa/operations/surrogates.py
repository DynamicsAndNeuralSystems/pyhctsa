import warnings
from typing import Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, zmap

from ..operations.correlation import tc3
from ..operations.information import automutual_info, first_min
from ..operations.physics import _ksdensity

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _sd_give_me_stats(stat_x: float, stat_surr: ArrayLike, left_right_both: str) -> dict:
    """Compute statistiscs on the surrogate distribution."""
    num_surrs = len(stat_surr)
    out = {}
    if np.isnan(stat_surr).any():
        logger.warning("SDgivemestats failed")
        return np.nan
    #% ASSUME GAUSSIAN DISTRIBUTION:
    #% so can use 1/2-sided z-statistic
    z_stat = zmap(np.atleast_1d(stat_x), stat_surr, ddof=1)[0]
    p = None
    if left_right_both == 'both':
        p = 2 * norm.sf(np.abs(z_stat))
    elif left_right_both == 'right':
        p = norm.sf(z_stat)
    elif left_right_both == 'left':
        p = norm.cdf(z_stat)
    out['p'] = p
    out['zscore'] = z_stat

    # kernel density of the (z-scored) surrogates' values, evaluated where the series' value lies
    # (MATLAB's ksdensity rule, as in hctsa)
    sigma = np.std(stat_surr, ddof=1)
    mu = np.mean(stat_surr)
    if sigma == 0 or not np.isfinite(sigma):
        # all surrogates have the same value of this statistic: cannot do a
        # meaningful z-score, so do it raw
        zsc_surr = np.asarray(stat_surr, dtype=float)
        xval = stat_x
    else:
        zsc_surr = (stat_surr - mu) / sigma
        xval = (stat_x - mu) / sigma
    f, xi = _ksdensity(zsc_surr)
    if (xval < xi.min()) or (xval > xi.max()):
        out["f"] = 0.0  # out of range: assume p = 0 here
    else:
        out["f"] = float(f[int(np.argmin(np.abs(xval - xi)))])

    # what fraction of the range is the sample in? 
    medsurr = np.median(stat_surr)
    iqrsurr = np.quantile(stat_surr, q=.75, method='hazen') - np.quantile(stat_surr, q=.25, method='hazen')
    if iqrsurr == 0:
        out['mediqr'] = np.nan
    else:
        out['mediqr'] = np.abs(stat_x-medsurr)/iqrsurr

    # rank-based p-value
    # number of surrogates strictly below the series' value (the series is ranked
    # ahead of any tied surrogates):
    num_below = int(np.sum(stat_surr < stat_x))
    num_at_least = num_surrs - num_below  # number of surrogates at least as large
    num_extreme = {'right': num_at_least,  # series should be larger than the surrogates
                   'left': num_below,
                   'both': min(num_below, num_at_least)}[left_right_both]  # the more extreme tail
    prank = (num_extreme + 1) / (num_surrs + 1)
    if left_right_both == 'both':
        prank = min(2 * prank, 1)  # two-sided: double, capped at 1

    out['prank'] = prank

    return out

def _make_surrogates(x: ArrayLike, surr_method: str = 'RP', num_surrs: int = 1,
                    random_seed: int = 42) -> ArrayLike:
    """
    Generates surrogate time series.

    Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    Used bits of aaft code that references (and presumably was obtained from) [1].

    References
    ----------
    .. [1] "Surrogate data test for nonlinearity including monotonic
        transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.

    Parameters
    ----------
    x : array-like
        The input time series.
    surr_method : str
        The method for generating surrogates:

        - 'RP': Random phase surrogates
        - 'AAFT': Amplitude adjusted Fourier transform. NOTE: **Not yet implemented.**
        - 'TFT': Truncated Fourier transform. NOTE: **Not yet implemented.**
        
        Default is ``'RP'``.
            
    num_surrs : int, optional
        The number of surrogates to generate. Default is 1.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        Array of surrogate time series.
    """
    x = np.asarray(x)
    N = len(x)
    out = np.zeros(shape=(N, num_surrs))
    if surr_method == 'RP':
        # random phase surrogates: the magnitude spectrum of x (at full length N)
        # is kept, and every phase is randomized except the DC (and, for even N,
        # Nyquist) phase, which is kept (necessarily 0 or pi for a real signal).
        # Randomized phases are negated onto the conjugate-symmetric half so the
        # result is real for any N.
        n_free = (N // 2 - 1) if (N % 2 == 0) else ((N - 1) // 2)

        # RNG
        rng = np.random.RandomState(random_seed)

        # FFT
        z = np.fft.fft(x)
        z_mag = np.abs(z)
        z_phase = np.angle(z)

        for s in range(num_surrs):
            rand_phase = rng.uniform(0.0, 2.0 * np.pi, size=max(n_free, 0))
            if N % 2 == 0:
                new_phase = np.concatenate((
                    z_phase[0:1],
                    rand_phase,
                    z_phase[N // 2:N // 2 + 1],
                    -rand_phase[::-1]
                ))
            else:
                new_phase = np.concatenate((
                    z_phase[0:1],
                    rand_phase,
                    -rand_phase[::-1]
                ))

            # Apply randomized phases, keep magnitudes; back to the time domain
            x_new = np.fft.ifft(z_mag * np.exp(1j * new_phase)).real
            out[:, s] = x_new

    elif surr_method == "AAFT":
        raise NotImplementedError("AAFT not yet implemented.")
    
    elif surr_method == "TFT":
        raise NotImplementedError("TFT not yet implemented.")
    
    else:
        raise ValueError(f"Unknown method: {surr_method}")
    
    return out

def _first_min_per_surrogate(z: np.ndarray, min_what: str) -> np.ndarray:
    """First minimum of the automutual information function of each surrogate (NaN on failure)."""
    out = np.zeros(z.shape[1])
    for i in range(z.shape[1]):
        try:
            out[i] = first_min(z[:, i], min_what)
        except Exception:
            out[i] = np.nan
    return out

def surrogate_test(
    x: ArrayLike,
    surr_meth: str = 'RP',
    num_surrs: int = 99,
    the_test_stat: Union[str, ArrayLike] = 'amikraskov1',
    random_seed: int = 42
) -> dict:
    """
    Analyzes test statistics obtained from surrogate time series.

    This function is based on [1].

    The generation of surrogates is done by the periphery function, `_make_surrogates`.

    References
    ----------
    .. [1] "Surrogate data test for nonlinearity including nonmonotonic transforms"
        D. Kugiumtzis, Phys. Rev. E 62(1) R25 (2000).
    .. [2] "Testing for nonlinearity in irregular fluctuations with long-term trends"
            T. Nakamura, M. Small, Y. Hirata, Phys. Rev. E 74(2) 026205 (2006).
    .. [3] "Surrogate time series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346 (2000).

    Parameters
    ----------
    x : array-like
        The input time series.
    surr_meth : str, optional
        The method for generating surrogate time series:
        
        - 'RP': random phase surrogates that maintain linear correlations in
            the data but destroy any nonlinear structure through phase randomization.
        - 'AAFT': amplitude-adjusted Fourier transform method maintains
            linear correlations but destroys nonlinear structure through phase
            randomization, yet preserves the approximate amplitude distribution.
            NOTE: **Not yet implemented.**
        - 'TFT': preserves low-frequency phases but randomizes high-frequency phases
            (as a way of dealing with non-stationarity, cf. [2]
            "A new surrogate data method for nonstationary time series",
            D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)).
            NOTE: **Not yet implemented.**
        
        Default is ``'RP'``.

    num_surrs : int, optional
        The number of surrogates to compute. Default is 99 for a 0.01 significance 
        level 1-sided test.
    the_test_stat : str or array-like, optional
        The test statistic(s) to evaluate on all surrogates and the original time series.
        A single name or a list of names; output is returned for each statistic:

        - 'amikraskov1': the automutual information at lag 1, estimated with the Kraskov
            nearest-neighbor estimator (``automutual_info(y, 1, 'kraskov1', 4)``); tested
            one-sided (surrogates should have lower values), cf. [2].
        - 'fmmikraskov': the first minimum of the Kraskov automutual information function
            (``first_min(y, 'mi-kraskov1')``); tested one-sided.
        - 'o3': a third-order statistic used in [3] (the mean cubed increment at lag 1);
            tested two-sided.
        - 'tc3': a time-reversal asymmetry measure (``tc3`` at lag 1); tested two-sided.
        - 'amigaussian1' and 'fmmigaussian': as 'amikraskov1' and 'fmmikraskov' but with
            the Gaussian estimate of the automutual information. That is a function of
            the autocorrelation only, which random-phase surrogates preserve, so
            these tests cannot detect anything (their p-values are close to uniform for
            every series); hctsa no longer registers them. The earlier names 'ami1' and
            'fmmi' (which both meant the Gaussian estimate) are accepted as
            'amigaussian1' and 'fmmigaussian' with a deprecation warning.

        Default is ``'amikraskov1'``. The statistics 'nlpe' and 'fnn' of hctsa's
        ``SD_SurrogateTest`` are not implemented. Outputs for each statistic ``s``
        (``amikraskov``, ``fmmikraskov``, ``amigaussian``, ``fmmigaussian``, ``o3``, ``tc3``)
        are ``s_p`` (p-value of a one- or two-sided z-test of the series' value against the
        Gaussian fitted to the surrogates' values), ``s_zscore``, ``s_f`` (kernel-smoothed
        density of the z-scored surrogates' values at the series' value; 0 outside the density
        estimate), ``s_mediqr`` (distance from the surrogates' median in interquartile ranges;
        NaN if the interquartile range is 0) and ``s_prank`` (rank-based p-value
        ``(k + 1) / (num_surrs + 1)``, with ``k`` the number of surrogates at least as extreme
        as the series in the tested direction; doubled and capped at 1 for two-sided tests).

    random_seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict or float
        Dictionary of statistics comparing the original time series to its
        surrogates for each test statistic. NaN if a statistic cannot be computed on
        every surrogate (hctsa raises an error, which its operation wrapper turns into NaN).
    """
    x = np.asarray(x)
    n = len(x)

    if isinstance(the_test_stat, str):
        the_test_stat = [the_test_stat]  # a bare string would be searched for substrings
    the_test_stat = list(the_test_stat)
    # earlier names of the Gaussian-estimate statistics
    for old, new in (('ami1', 'amigaussian1'), ('fmmi', 'fmmigaussian')):
        if old in the_test_stat:
            warnings.warn(
                f"surrogate_test statistic '{old}' is deprecated: use '{new}' (the Gaussian "
                f"estimate) or the Kraskov versions 'amikraskov1' and 'fmmikraskov'.",
                DeprecationWarning, stacklevel=2)
            the_test_stat = [new if t == old else t for t in the_test_stat]

    #Generate surrogate time series
    z = _make_surrogates(x, surr_method=surr_meth, num_surrs=num_surrs, random_seed=random_seed)
    # z is matrix where each column is a surrogate time series
    #% Evaluate test statistic on each surrogate
    out = {}

    def _compare(label, stat_x, stat_surr, side):
        some_stats = _sd_give_me_stats(stat_x, stat_surr, side)
        if not isinstance(some_stats, dict):
            return False  # NaN in the surrogates' statistics: hctsa errors
        for k, v in some_stats.items():
            out[f'{label}_{k}'] = v
        return True

    if 'amikraskov1' in the_test_stat:
        # Kraskov AMI(1) of the surrogates compared to that of the signal itself. Unlike the
        # Gaussian estimate (a function of the autocorrelation, which random-phase surrogates
        # preserve) it responds to nonlinear dependence between x(t) and x(t+1).
        ami_x = automutual_info(x, 1, 'kraskov1', 4)
        ami_surr = np.array([automutual_info(z[:, i], 1, 'kraskov1', 4)
                             for i in range(num_surrs)])
        # surrogates should have lower AMI than the original signal
        if not _compare('amikraskov', ami_x, ami_surr, 'right'):
            return np.nan

    if 'fmmikraskov' in the_test_stat:
        # first minimum of the Kraskov automutual information of the surrogates compared to
        # that of the signal itself
        fmmi_x = first_min(x, 'mi-kraskov1')
        fmmi_surr = _first_min_per_surrogate(z, 'mi-kraskov1')
        if np.isnan(fmmi_surr).any():
            logger.warning("fmmikraskov failed")
            return np.nan
        # the first minimum should be at a higher lag for the signal than for the surrogates
        if not _compare('fmmikraskov', fmmi_x, fmmi_surr, 'right'):
            return np.nan

    if 'amigaussian1' in the_test_stat:
        # AMI(1) of the surrogates compared to that of the signal itself, with the Gaussian
        # estimate (as in Nakamura et al. (2006))
        ami_x = automutual_info(x, 1, 'gaussian')
        ami_surr = np.array([automutual_info(z[:, i], 1, 'gaussian')
                             for i in range(num_surrs)])
        if not _compare('amigaussian', ami_x, ami_surr, 'right'):
            return np.nan

    if 'fmmigaussian' in the_test_stat:
        fmmi_x = first_min(x, 'mi-gaussian')
        fmmi_surr = _first_min_per_surrogate(z, 'mi-gaussian')
        if np.isnan(fmmi_surr).any():
            logger.warning("fmmigaussian failed")
            return np.nan
        # FMMI should be higher for the signal than for the surrogates
        if not _compare('fmmigaussian', fmmi_x, fmmi_surr, 'right'):
            return np.nan

    if 'o3' in the_test_stat:
        #% Third-order statistic in Schreiber, Schmitz (Physica D)
        tau = 1
        o3_x = (1.0 / (n - tau)) * np.sum((x[tau:] - x[:n - tau]) ** 3)
        o3_surr = np.zeros(num_surrs, dtype=float)
        for i in range(num_surrs):
            o3_surr[i] = (1.0 / (n - tau)) * np.sum((z[tau:, i] - z[:n - tau, i]) ** 3)
        if not _compare('o3', o3_x, o3_surr, 'both'):
            return np.nan

    if 'tc3' in the_test_stat:
        # tc3 statistic -- another time-reversal asymmetry measure
        tau = 1
        tmp = tc3(x, tau)
        tc3_x = tmp['raw']
        tc3_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            tmp = tc3(z[:, i], tau)
            tc3_surr[i] = tmp['raw']
        if not _compare('tc3', tc3_x, tc3_surr, 'both'):
            return np.nan

    return out
