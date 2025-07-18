import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from pyhctsa.Operations.Correlation import FirstCrossing, AutoCorr


def ForcePotential(y : ArrayLike, whatPotential : str = 'dblwell', params : Union[list, None] = None) -> dict:
    """
    Couples the values of the time series to a dynamical system.

    The input time series forces a particle in the given potential well.

    Args:
    y (array-like): The input time series.
    what_potential (str): The potential function to simulate:
                          'dblwell' (a double well potential function) or
                          'sine' (a sinusoidal potential function).
    params (list): The parameters for simulation, should be in the form:
                   [alpha, kappa, deltat]

    Returns:
    dict: Statistics summarizing the trajectory of the simulated particle.
    """
    y = np.array(y)
    if params is None:
        if whatPotential == 'dblwell':
            params = [2, 0.1, 0.1]
        elif whatPotential == 'sine':
            params = [1, 1, 1]
        else:
            raise ValueError(f"Unknown system {whatPotential}")
    else:
        # check params
        if not isinstance(params, list):
            raise ValueError("Expected list of parameters.")
        else:
            if len(params) != 3:
                raise ValueError("Expected 3 parameters.")
    
    N = len(y) # length of the time series

    alpha, kappa, deltat = params

    # specify potential function
    if whatPotential == 'sine':
        V = lambda x: -np.cos(x/alpha)
        F = lambda x: np.sin(x/alpha)/alpha
    elif whatPotential == 'dblwell':
        F = lambda x: -x**3 + alpha**2 * x
        V = lambda x: ((x**4) / 4) - (alpha**2) * ((x**2) / 2)
    else:
        raise ValueError(f"Unknown potential function {whatPotential}")
    
    x = np.zeros(N) # position
    v = np.zeros(N) # velocity

    for i in range(1, N):
        x[i] = x[i-1] + v[i-1]*deltat + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat**2
        v[i] = v[i-1] + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat

    # check the trajectory didn't blow out
    if np.isnan(x[-1]) or np.abs(x[-1]) > 1E10:
        return np.nan
    
    # Output some basic features of the trajectory
    out = {}
    out['mean'] = np.mean(x) # mean position
    out['median'] = np.median(x) # median position
    out['std'] = np.std(x, ddof=1) # std. dev.
    out['range'] = np.ptp(x)
    out['proppos'] = np.sum(x >0)/N
    out['pcross'] = np.sum(x[:-1] * x[1:] < 0) / (N - 1)
    out['ac1'] = np.abs(AutoCorr(x, 1, 'Fourier')[0])
    out['ac10'] = np.abs(AutoCorr(x, 10, 'Fourier')[0])
    out['ac50'] = np.abs(AutoCorr(x, 50, 'Fourier')[0])
    out['tau'] = FirstCrossing(x, 'ac', 0, 'continuous')
    out['finaldev'] = np.abs(x[-1]) # final position

    # additional outputs for dbl well
    if whatPotential == 'dblwell':
        out['pcrossup'] = np.sum((x[:-1] - alpha) * (x[1:] - alpha) < 0) / (N - 1)
        out['pcrossdown'] = np.sum((x[:-1] + alpha) * (x[1:] + alpha) < 0) / (N - 1)

    return out
