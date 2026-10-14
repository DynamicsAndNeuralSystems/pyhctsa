r"""
          _____                    _____                    _____          
         /\    \                  /\    \                  /\    \         
        /::\    \                /::\    \                /::\    \        
       /::::\    \              /::::\    \              /::::\    \       
      /::::::\    \            /::::::\    \            /::::::\    \      
     /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
    /:::/__\:::\    \        /:::/__\:::\    \        /:::/  \:::\    \    
   /::::\   \:::\    \      /::::\   \:::\    \      /:::/    \:::\    \   
  /::::::\   \:::\    \    /::::::\   \:::\    \    /:::/    / \:::\    \  
 /:::/\:::\   \:::\____\  /:::/\:::\   \:::\    \  /:::/    /   \:::\ ___\ 
/:::/  \:::\   \:::|    |/:::/  \:::\   \:::\____\/:::/____/     \:::|    |
\::/   |::::\  /:::|____|\::/    \:::\  /:::/    /\:::\    \     /:::|____|
 \/____|:::::\/:::/    /  \/____/ \:::\/:::/    /  \:::\    \   /:::/    / 
       |:::::::::/    /            \::::::/    /    \:::\    \ /:::/    /  
       |::|\::::/    /              \::::/    /      \:::\    /:::/    /   
       |::| \::/____/               /:::/    /        \:::\  /:::/    /    
       |::|  ~|                    /:::/    /          \:::\/:::/    /     
       |::|   |                   /:::/    /            \::::::/    /      
       \::|   |                  /:::/    /              \::::/    /       
        \:|   |                  \::/    /                \::/____/        
         \|___|                   \/____/                  ~~              
                                                                           
"""
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ..operations.correlation import first_crossing
from ..toolboxes.RAD.rad import rad_centre, rad_core
from ..utils import get_tau

def rad(x: ArrayLike, tau: Union[int, str] = 1, centre: bool = True) -> float:
    """
    Compute the Rescaled Auto-Density (RAD) feature of a time series.

    The RAD is a metric for inferring the distance to criticality in a system,
    designed to be robust to uncertainty in noise strength. It is calibrated using
    experiments on the Hopf bifurcation with variable and unknown measurement noise.

    This method was devised and implemented by Brendan Harris (@brendanjohnharris, GitHub, 2023).
    Derived from Harris's ``RAD.m`` (https://github.com/brendanjohnharris/Criticality, MIT licence,
    Copyright (c) 2021 Brendan Harris and contributors), via hctsa's ``CR_RAD``; the computation
    is in :mod:`pyhctsa.toolboxes.RAD.rad`.

    References
    ----------
    .. [1] Harris et al., "Tracking the Distance to Criticality in Systems with 
        Unknown Noise", Phys. Rev. X 14, 031021 (2024)

    Parameters
    ----------
    x : array-like
        The input time series (1D array).
    tau : int or str, optional
        The embedding and differencing delay, in units of the time step. A string sets it
        from the series (after any `centre` transformation); NaN is returned if the delay
        is undefined:

        - ``'tau'``: the first zero crossing of the autocorrelation function;
        - ``'ac1e'``: the floor of the first 1/e crossing of the autocorrelation function;
        - ``'mi'``: the smaller of the first minimum of the Kraskov automutual information
          and the ``'ac1e'`` delay (see :func:`pyhctsa.utils.get_tau`).

        Default is 1.
    centre : bool, optional
        Whether to center the time series at zero and take absolute values before 
        analysis. Default is `True`.

    Returns
    -------
    float
        The RAD feature value, quantifying proximity to criticality.
    """

    # ensure that x is in the form of a numpy array
    x = np.asarray(x)
    
    # if specified: centre the time series and take the absolute value
    if centre:
        x = rad_centre(x)
    
    # if specified, make tau the first crossing of the AC function
    if isinstance(tau, str):
        if tau == "tau":
            tau = first_crossing(x, 'ac', 0, 'discrete')
            if tau is None or np.isnan(tau):
                # delay undefined (e.g., an all-NaN autocorrelation function)
                return np.nan
            tau = int(tau)
        elif tau in ("ac1e", "mi"):
            # adaptive delay: see BF_GetTau
            tau = get_tau(x, tau)
            if np.isnan(tau):
                return np.nan
        else:
            raise ValueError(f"Unknown operation {tau}")

    return rad_core(x, tau)
