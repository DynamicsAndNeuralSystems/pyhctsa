
import numpy as np
from numpy.typing import ArrayLike 
from typing import Union
from ..Utilities.utils import pointOfCrossing
from loguru import logger

def AutoCorr(y: ArrayLike, tau: Union[int, list] = 1, method: str = 'Fourier') -> Union[float, np.ndarray]:
    """
    Compute the autocorrelation of an input time series.

    Parameters:
    -----------
    y : array_like
        A scalar time series column vector.
    tau : int, list, optional
        The time-delay. If tau is a scalar, returns autocorrelation for y at that
        lag. If tau is a list, returns austocorrelations for y at that set of
        lags. If empty list, returns the full function for the 'Fourier' estimation method.
    method : str, optional
        The method of computing the autocorrelation: 'Fourier',
        'TimeDomainStat', or 'TimeDomain'.

    Returns:
    --------
    float or array
        The autocorrelation at the given time lag(s).

    """
    y = np.array(y)
    N = len(y)  # time-series length

    if tau:
        # if list is not empty
        if np.max(tau) > N - 1:  # -1 because acf(1) is lag 0
            logger.warning(f"Time lag {np.max(tau)} is too long for time-series length {N}.")
        if np.any(np.array(tau) < 0):
            logger.warning('Negative time lags not applicable.')
    
    if method == 'Fourier':
        n_fft = 2 ** (int(np.ceil(np.log2(N))) + 1)
        F = np.fft.fft(y - np.mean(y), n_fft)
        F = F * np.conj(F)
        acf = np.fft.ifft(F)  # Wiener–Khinchin
        acf = acf / acf[0]  # Normalize
        acf = np.real(acf)
        acf = acf[:N]
        
        if not tau:  # list empty, return the full function
            out = acf
        else:  # return a specific set of values
            tau = np.atleast_1d(tau)
            out = np.zeros(len(tau))
            for i, t in enumerate(tau):
                if (t > len(acf) - 1) or (t < 0):
                    out[i] = np.nan
                else:
                    out[i] = acf[t]
    
    elif method == 'TimeDomainStat':
        sigma2 = np.std(y, ddof=1)**2  # time-series variance
        mu = np.mean(y)  # time-series mean
        
        def acf_y(t):
            return np.mean((y[:N-t] - mu) * (y[t:] - mu)) / sigma2
        
        tau = np.atleast_1d(tau)
        out = np.array([acf_y(t) for t in tau])
    
    elif method == 'TimeDomain':
        tau = np.atleast_1d(tau)
        out = np.zeros(len(tau))
        
        for i, t in enumerate(tau):
            if np.any(np.isnan(y)):
                good_r = (~np.isnan(y[:N-t])) & (~np.isnan(y[t:]))
                print(f'NaNs in time series, computing for {np.sum(good_r)}/{len(good_r)} pairs of points')
                y1 = y[:N-t]
                y1n = y1[good_r] - np.mean(y1[good_r])
                y2 = y[t:]
                y2n = y2[good_r] - np.mean(y2[good_r])
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean(y1n * y2n) / np.std(y1[good_r], ddof=0) / np.std(y2[good_r], ddof=0)
            else:
                y1 = y[:N-t]
                y2 = y[t:]
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean((y1 - np.mean(y1)) * (y2 - np.mean(y2))) / np.std(y1, ddof=0) / np.std(y2, ddof=0)
    
    else:
        raise ValueError(f"Unknown autocorrelation estimation method {method}")
    
    return out

def FirstCrossing(y: ArrayLike, corrFun: str = 'ac', threshold: float = 0.0, whatOut: str = 'both') -> Union[dict, float]:
    """
    The first crossing of a given autocorrelation function across a given threshold.

    Parameters
    -----------
    y : array_like
        The input time series
    corrFun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    whatOut : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns
    --------
    dict or float
        The first crossing information, format depends on whatOut
    """
    # Select the self-correlation function
    if corrFun == 'ac':
        # Autocorrelation at all time lags
        corrs = AutoCorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corrFun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = pointOfCrossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if whatOut == 'both':
        out = {
            'firstCrossing': first_crossing_index - 1,
            'pointOfCrossing': point_of_crossing_index - 1
        }
    elif whatOut == 'discrete':
        out = first_crossing_index - 1
    elif whatOut == 'continuous':
        out = point_of_crossing_index - 1
    else:
        raise ValueError(f"Unknown output format '{whatOut}'")

    return out
