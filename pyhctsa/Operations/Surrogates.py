import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy.stats import zmap, norm, gaussian_kde

from pyhctsa.Operations.Information import AutoMutualInfo
from pyhctsa.Operations.correlation import FirstMin, TC3

def SDgivemestats(statx : float, statsurr : ArrayLike, leftrightboth : str) -> dict:
    """Compute statistiscs on the surrogate distribution."""
    numSurrs = len(statsurr)
    out = {}
    if np.isnan(statsurr).any():
        raise ValueError("SDgivemestats failed")
    #% ASSUME GAUSSIAN DISTRIBUTION:
    #% so can use 1/2-sided z-statistic
    zStat = zmap(np.atleast_1d(statx), statsurr, ddof=1)[0]
    p = None
    if leftrightboth == 'both':
        p = 2 * norm.sf(np.abs(zStat))
    elif leftrightboth == 'right':
        p = norm.sf(zStat)
    elif leftrightboth == 'left':
        p = norm.cdf(zStat)
    out['p'] = p
    out['zscore'] = zStat

    # fit a kenerel distribution to zscored distributions
    sigma = np.std(statsurr, ddof=1)
    mu = np.mean(statsurr)
    if sigma == 0 or not np.isfinite(sigma):
        # all surrogates have same value of this statisitc
        # cannot do a meaningful zscore
        c = float(statsurr[0])
        bw = 1.0 # bandwidth scale
        xi = np.linspace(c - 3 * bw, c + 3 * bw, 100)
        # tiny gaussian around c
        f = norm.pdf(xi, loc=c, scale=bw)
        if (statx < xi.min()) or (statx > xi.max()):
            out["f"] = 0.0
        else:
            idx = int(np.argmin(np.abs(statx - xi)))
            out["f"] = float(f[idx])
    
    else:
        # z-score branch
        zscstatsurr = (statsurr - mu) / sigma
        zscstatx = (statx - mu) / sigma
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
    medsurr = np.median(statsurr)
    iqrsurr = np.quantile(statsurr, q=.75, method='hazen') - np.quantile(statsurr, q=.25, method='hazen')
    if iqrsurr == 0:
        out['mediqr'] = np.nan
    else:
        out['mediqr'] = np.abs(statx-medsurr)/iqrsurr

    # rank statistic 
    ix = np.argsort(np.concatenate(([statx], statsurr)))
    # Where did the original index 0 (i.e., statx) end up?
    xfitshere = np.where(ix == 0)[0][0]
    if leftrightboth == 'right':  # x smaller than distribution → flip distance from top
        xfitshere = numSurrs + 1 - xfitshere
    elif leftrightboth == 'both':
        xfitshere = min(xfitshere, numSurrs + 1 - xfitshere)

    if xfitshere is None:  
        prank = 1 / (numSurrs + 1)
    else:
        prank = (1 + xfitshere) / (numSurrs + 1)

    if leftrightboth == 'both':
        prank *= 2

    out['prank'] = prank

    return out

def MakeSurrogates(x : ArrayLike, surrMethod : str, numSurrs : int = 1, randomSeed : int = 42) -> ArrayLike:
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
    surrMethod : str
        The method for generating surrogates:
            - 'RP' -- random phase surrogates
            - 'AAFT' -- amplitude adjusted Fourier transform.
                NOTE: **Not yet implemented.**
            - 'TFT' -- truncated Fourier transform.
                NOTE: **Not yet implemented.**
    numSurrs : int, optional
        The number of surrogates to generate.
    randomSeed : int, optional
        Random seed for reproducibility. 

    Returns
    -------
    np.ndarray
        Array of surrogate time series.
    """
    x = np.asarray(x)
    N = len(x)
    out = np.zeros(shape=(N, numSurrs))
    if surrMethod == 'RP':
        # random phase surrogates
        n2 = (N // 2) if (N % 2 == 0) else ((N - 1) // 2)  # floor(N/2)
        fft_len = 2 * n2  

        # RNG
        rng = np.random.RandomState(randomSeed)

        # FFT
        z = np.fft.fft(x, n=fft_len)
        z_mag = np.abs(z)
        z_phase = np.angle(z)

        for s in range(numSurrs):
            if n2 - 1 > 0:
                randphase = rng.uniform(0.0, 2.0 * np.pi, size=n2 - 1)
            else:
                randphase = np.empty(0)

            new_phase = np.concatenate((
                np.array([0.0]),
                randphase,
                np.array([z_phase[n2]]),     
                -randphase[::-1]
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

    elif surrMethod == "AAFT":
        raise NotImplementedError("AAFT not yet implemented.")
    
    elif surrMethod == "TFT":
        raise NotImplementedError("TFT not yet implemented.")
    
    else:
        raise ValueError(f"Unknown method: {surrMethod}")
    
    return out

def SurrogateTest(x : ArrayLike, 
                surrMeth : str = "RP",
                numSurrs : int = 99, 
                theTestStat : Union[str, ArrayLike] = "ami1", 
                randomSeed : int = 42) -> dict:
    """
    SD_SurrogateTest: Analyzes test statistics obtained from surrogate time series.

    This function is based on:
        "Surrogate data test for nonlinearity including nonmonotonic transforms"
        D. Kugiumtzis, Phys. Rev. E 62(1) R25 (2000).

    The generation of surrogates is done by the periphery function, SD_MakeSurrogates.

    Parameters
    ----------
    x : array-like
        The input time series.
    surrMeth : str, optional
        The method for generating surrogate time series:
            - 'RP': random phase surrogates that maintain linear correlations in
              the data but destroy any nonlinear structure through phase randomization.
            - 'AAFT': amplitude-adjusted Fourier transform method maintains
              linear correlations but destroys nonlinear structure through phase
              randomization, yet preserves the approximate amplitude distribution.
              NOTE: **Not yet implemented.**
            - 'TFT': preserves low-frequency phases but randomizes high-frequency phases
              (as a way of dealing with non-stationarity, cf.:
              "A new surrogate data method for nonstationary time series",
              D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)).
              NOTE: **Not yet implemented.**
    numSurrs : int, optional
        The number of surrogates to compute (default is 99 for a 0.01 significance level 1-sided test).
    theTestStat : str or array-like, optional
        The test statistic(s) to evaluate on all surrogates and the original time series.
        Can specify multiple options and will return output for each specified test statistic:
            - 'ami': the automutual information at lag 1, cf.
              "Testing for nonlinearity in irregular fluctuations with long-term trends"
              T. Nakamura, M. Small, Y. Hirata, Phys. Rev. E 74(2) 026205 (2006).
            - 'fmmi': the first minimum of the automutual information function.
            - 'o3': a third-order statistic used in:
              "Surrogate time series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346 (2000).
            - 'tc3': a time-reversal asymmetry measure.
        Outputs of the function include a z-test between the two distributions, and
        some comparative rank-based statistics.
    randomSeed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary of statistics comparing the original time series to its surrogates for each test statistic.
    """
    x = np.asarray(x)
    N = len(x)
    #Generate surrogate time series
    z = MakeSurrogates(x, surrMethod=surrMeth, numSurrs=numSurrs, randomSeed=randomSeed)
    # z is matrix where each column is a surrogate time series
    #% Evaluate test statistic on each surrogate
    out = {}
    if "ami1" in theTestStat:
        ami_fn = lambda timeSeries, timeDelay : AutoMutualInfo(timeSeries, timeDelay, 'gaussian')
        AMIx = ami_fn(x, 1)
        AMIsurr = np.zeros(numSurrs)
        for i in range(numSurrs):
            AMIsurr[i] = ami_fn(z[:, i], 1)
        someStats = SDgivemestats(AMIx, AMIsurr, "right")
        for (k, v) in zip(someStats.keys(), someStats.values()):
            out[f'ami_{k}'] = v
    if "fmmi" in theTestStat:
        #% Investigate the first minimum of mutual information of surrogates compared to
        #% that of signal itself
        fmmix = FirstMin(x, 'mi')
        fmmiSurr = np.zeros(numSurrs)
        for i in range(numSurrs):
            try:
                fmmiSurr[i] = FirstMin(z[:, i], 'mi')
            except:
                fmmiSurr[i] = np.nan
        
        if np.isnan(fmmiSurr).any():
            raise ValueError("fmmi failed")
        #% FMMI should be higher for signal than surrogates
        someStats = SDgivemestats(fmmix, fmmiSurr, "right")
        for (k, v) in zip(someStats.keys(), someStats.values()):
            out[f'fmmi_{k}'] = v
    if "o3" in theTestStat:
        #% Third-order statistic in Schreiber, Schmitz (Physica D)
        tau = 1
        o3x = (1.0 / (N - tau)) * np.sum((x[tau:] - x[:N - tau]) ** 3)
        o3surr = np.zeros(numSurrs, dtype=float)
        for i in range(numSurrs):
            o3surr[i] = (1.0 / (N - tau)) * np.sum((z[tau:, i] - z[:N - tau, i]) ** 3)
        someStats = SDgivemestats(o3x, o3surr, "both")
        for (k, v) in zip(someStats.keys(), someStats.values()):
            out[f'o3_{k}'] = v
    if "tc3" in theTestStat:
        # TC3 statistic -- another time-reversal asymmetry measure
        tau = 1
        tmp = TC3(x, tau)
        tc3x = tmp['raw']
        tc3surr = np.zeros(numSurrs)
        for i in range(numSurrs):
            tmp = TC3(z[:, i], tau)
            tc3surr[i] = tmp['raw']
        someStats = SDgivemestats(tc3x, tc3surr, "both")
        for (k, v) in zip(someStats.keys(), someStats.values()):
            out[f'tc3_{k}'] = v
            
    return out
