import warnings
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde, norm, zmap

from ..operations.correlation import tc3
from ..operations.information import automutual_info, first_min

warnings.filterwarnings("ignore", category=RuntimeWarning)

def sd_give_me_stats(stat_x : float, stat_surr : ArrayLike, left_right_both : str) -> dict:
    """Compute statistiscs on the surrogate distribution."""
    num_surrs = len(stat_surr)
    out = {}
    if np.isnan(stat_surr).any():
        raise ValueError("SDgivemestats failed")
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

    # fit a kenerel distribution to zscored distributions
    sigma = np.std(stat_surr, ddof=1)
    mu = np.mean(stat_surr)
    if sigma == 0 or not np.isfinite(sigma):
        # all surrogates have same value of this statisitc
        # cannot do a meaningful zscore
        c = float(stat_surr[0])
        bw = 1.0 # bandwidth scale
        xi = np.linspace(c - 3 * bw, c + 3 * bw, 100)
        # tiny gaussian around c
        f = norm.pdf(xi, loc=c, scale=bw)
        if (stat_x < xi.min()) or (stat_x > xi.max()):
            out["f"] = 0.0
        else:
            idx = int(np.argmin(np.abs(stat_x - xi)))
            out["f"] = float(f[idx])
    
    else:
        # z-score branch
        zscstatsurr = (stat_surr - mu) / sigma
        zscstatx = (stat_x - mu) / sigma
        kde = gaussian_kde(zscstatsurr)  # Scott's rule
        xi = np.linspace(zscstatsurr.min(), zscstatsurr.max(), 1000)
        f = kde(xi)
        xval = float(zscstatx)
        if (xval < xi.min()) or (xval > xi.max()):
            out["f"] = 0.0  # out of range → assume p=0 (as in the MATLAB comment)
        else:
            minhere = int(np.argmin(np.abs(xval - xi)))
            out["f"] = float(f[minhere])
        
    # what fraction of the range is the sample in? 
    medsurr = np.median(stat_surr)
    iqrsurr = np.quantile(stat_surr, q=.75, method='hazen') - np.quantile(stat_surr, q=.25, method='hazen')
    if iqrsurr == 0:
        out['mediqr'] = np.nan
    else:
        out['mediqr'] = np.abs(stat_x-medsurr)/iqrsurr

    # rank statistic 
    ix = np.argsort(np.concatenate(([stat_x], stat_surr)))
    # Where did the original index 0 (i.e., stat_x) end up?
    xfitshere = np.where(ix == 0)[0][0]
    if left_right_both == 'right':  # x smaller than distribution → flip distance from top
        xfitshere = num_surrs + 1 - xfitshere
    elif left_right_both == 'both':
        xfitshere = min(xfitshere, num_surrs + 1 - xfitshere)

    if xfitshere is None:  
        prank = 1 / (num_surrs + 1)
    else:
        prank = (1 + xfitshere) / (num_surrs + 1)

    if left_right_both == 'both':
        prank *= 2

    out['prank'] = prank

    return out

def make_surrogates(x : ArrayLike, surr_method : str, num_surrs : int = 1, 
                    random_seed : int = 42) -> ArrayLike:
    """
    Generates surrogate time series.

    Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    Used bits of aaft code that references (and presumably was obtained from)
    "Surrogate data test for nonlinearity including monotonic
    transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.

    Parameters
    ----------
    x : array-like
        The input time series.
    surr_method : str
        The method for generating surrogates:
            - 'RP' -- random phase surrogates
            - 'AAFT' -- amplitude adjusted Fourier transform.
                NOTE: **Not yet implemented.**
            - 'TFT' -- truncated Fourier transform.
                NOTE: **Not yet implemented.**
    num_surrs : int, optional
        The number of surrogates to generate.
    random_seed : int, optional
        Random seed for reproducibility. 

    Returns
    -------
    np.ndarray
        Array of surrogate time series.
    """
    x = np.asarray(x)
    N = len(x)
    out = np.zeros(shape=(N, num_surrs))
    if surr_method == 'RP':
        # random phase surrogates
        n2 = (N // 2) if (N % 2 == 0) else ((N - 1) // 2)  # floor(N/2)
        fft_len = 2 * n2  

        # RNG
        rng = np.random.RandomState(random_seed)

        # FFT
        z = np.fft.fft(x, n=fft_len)
        z_mag = np.abs(z)
        z_phase = np.angle(z)

        for s in range(num_surrs):
            if n2 - 1 > 0:
                rand_phase = rng.uniform(0.0, 2.0 * np.pi, size=n2 - 1)
            else:
                rand_phase = np.empty(0)

            new_phase = np.concatenate((
                np.array([0.0]),
                rand_phase,
                np.array([z_phase[n2]]),     
                -rand_phase[::-1]
            ))

            # Symmetric magnitudes: [zMag(1:n2+1), flipud(zMag(2:n2))]
            mag_sym = np.concatenate((
                z_mag[0:n2 + 1],
                z_mag[1:n2][::-1]
            ))

            # Apply randomized phases, keep magnitudes
            z_new = mag_sym * np.exp(1j * new_phase)

            # Back to time domain; ifft length N
            x_new = np.fft.ifft(z_new, n=N).real
            out[:, s] = x_new

    elif surr_method == "AAFT":
        raise NotImplementedError("AAFT not yet implemented.")
    
    elif surr_method == "TFT":
        raise NotImplementedError("TFT not yet implemented.")
    
    else:
        raise ValueError(f"Unknown method: {surr_method}")
    
    return out

def surrogate_test(
    x: ArrayLike,
    surr_meth: str = "RP",
    num_surrs: int = 99,
    the_test_stat: Union[str, ArrayLike] = "ami1",
    random_seed: int = 42
) -> dict:
    """
    Analyzes test statistics obtained from surrogate time series.

    This function is based on [1].

    The generation of surrogates is done by the periphery function, `make_surrogates`.

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

    num_surrs : int, optional
        The number of surrogates to compute (default is 99 for a 0.01 significance 
        level 1-sided test).
    the_test_stat : str or array-like, optional
        The test statistic(s) to evaluate on all surrogates and the original time series.
        Can specify multiple options and will return output for each specified test statistic:

        - 'ami': the automutual information at lag 1, cf. [2]
        - 'fmmi': the first minimum of the automutual information function.
        - 'o3': a third-order statistic used in [3].
        - 'tc3': a time-reversal asymmetry measure.
        Outputs of the function include a z-test between the two distributions, and
        some comparative rank-based statistics.
        
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary of statistics comparing the original time series to its 
        surrogates for each test statistic.
    """
    x = np.asarray(x)
    n = len(x)

    #Generate surrogate time series
    z = make_surrogates(x, surr_method=surr_meth, num_surrs=num_surrs, random_seed=random_seed)
    # z is matrix where each column is a surrogate time series
    #% Evaluate test statistic on each surrogate
    out = {}

    if "ami1" in the_test_stat:
        ami_fn = lambda time_series, time_delay: automutual_info(time_series, time_delay, 'gaussian')
        ami_x = ami_fn(x, 1)
        ami_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            ami_surr[i] = ami_fn(z[:, i], 1)
        some_stats = sd_give_me_stats(ami_x, ami_surr, "right")
        for (k, v) in zip(some_stats.keys(), some_stats.values()):
            out[f'ami_{k}'] = v

    if "fmmi" in the_test_stat:
        #% Investigate the first minimum of mutual information of surrogates compared to
        #% that of signal itself
        fmmi_x = first_min(x, 'mi')
        fmmi_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            try:
                fmmi_surr[i] = first_min(z[:, i], 'mi')
            except Exception:
                fmmi_surr[i] = np.nan

        if np.isnan(fmmi_surr).any():
            raise ValueError("fmmi failed")
        #% FMMI should be higher for signal than surrogates
        some_stats = sd_give_me_stats(fmmi_x, fmmi_surr, "right")
        for (k, v) in zip(some_stats.keys(), some_stats.values()):
            out[f'fmmi_{k}'] = v

    if "o3" in the_test_stat:
        #% Third-order statistic in Schreiber, Schmitz (Physica D)
        tau = 1
        o3_x = (1.0 / (n - tau)) * np.sum((x[tau:] - x[:n - tau]) ** 3)
        o3_surr = np.zeros(num_surrs, dtype=float)
        for i in range(num_surrs):
            o3_surr[i] = (1.0 / (n - tau)) * np.sum((z[tau:, i] - z[:n - tau, i]) ** 3)
        some_stats = sd_give_me_stats(o3_x, o3_surr, "both")
        for (k, v) in zip(some_stats.keys(), some_stats.values()):
            out[f'o3_{k}'] = v

    if "tc3" in the_test_stat:
        # tc3 statistic -- another time-reversal asymmetry measure
        tau = 1
        tmp = tc3(x, tau)
        tc3_x = tmp['raw']
        tc3_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            tmp = tc3(z[:, i], tau)
            tc3_surr[i] = tmp['raw']
        some_stats = sd_give_me_stats(tc3_x, tc3_surr, "both")
        for (k, v) in zip(some_stats.keys(), some_stats.values()):
            out[f'tc3_{k}'] = v

    return out
