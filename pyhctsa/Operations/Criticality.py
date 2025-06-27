import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from ..Operations.Correlation import FirstCrossing

def RAD(x: ArrayLike, tau : Union[int, str] = 1, centre : bool = True) -> float:
    """
    Compute the Rescaled Auto-Density (RAD) feature of a time series.

    The RAD is a metric for inferring the distance to criticality in a system,
    designed to be robust to uncertainty in noise strength. It is calibrated using
    experiments on the Hopf bifurcation with variable and unknown measurement noise.

    This method was devised and implemented by Brendan Harris (@brendanjohnharris, GitHub, 2023).

    Parameters
    ----------
    x : array-like
        The input time series (1D array).
    tau : int or str, optional
        The embedding and differencing delay, in units of the time step (default: 1).
        If a string, must be "tau", in which case the delay is set to the first
        crossing of the autocorrelation function.
    centre : bool, optional
        Whether to center the time series at zero and take absolute values before analysis (default: True).

    Returns
    -------
    float
        The RAD feature value, quantifying proximity to criticality.

    References
    ----------
    Harris, B. (2023). Rescaled Auto-Density (RAD) metric for criticality inference.
    """

    # ensure that x is in the form of a numpy array
    x = np.asarray(x)
    
    # if specified: centre the time series and take the absolute value
    if centre:
        x = x - np.median(x)
        x = np.abs(x)
    
    # if specified, make tau the first crossing of the AC function
    if isinstance(tau, str):
        if tau == "tau":
            tau = FirstCrossing(x, 'ac', 0, 'discrete')
        else:
            raise ValueError(f"Unknown operation {tau}")

    # Delay embed at interval tau
    y = x[tau:]
    x = x[:-tau]

    # Median split
    subMedians = x < np.median(x)
    superMedianSD = np.std(x[~subMedians], ddof=1)
    subMedianSD = np.std(x[subMedians], ddof=1)

    # Properties of the auto-density
    sigma_dx = np.std(y - x, ddof=1)
    densityDifference = (1/superMedianSD) - (1/subMedianSD)

    # return RAD
    return sigma_dx * densityDifference
