import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from pyhctsa.Operations.Correlation import FirstCrossing, AutoCorr
from pyhctsa.Operations.Stationarity import SlidingWindow
from scipy.stats import ansari, gaussian_kde
from statsmodels.sandbox.stats.runs import runstest_1samp


def Walker(y : ArrayLike, walkerRule : str = 'prop', walkerParams : Union[None, float, int, list] = None) -> dict:
    """
    Simulates a hypothetical walker moving through the time domain.
    
    The hypothetical particle (or 'walker') moves in response to values of the
    time series at each point. Outputs from this operation are summaries of the
    walker's motion, and comparisons of it to the original time series.
    
    Parameters
    ----------
    y : array-like
        The input time series.
    walkerRule : str, optional
        The kinematic rule by which the walker moves in response to the
        time series over time:
        
        - 'prop': the walker narrows the gap between its value and that
          of the time series by a given proportion p.
          walkerParams = p
        - 'biasprop': the walker is biased to move more in one
          direction; when it is being pushed up by the time
          series, it narrows the gap by a proportion p_up,
          and when it is being pushed down by the time series,
          it narrows the gap by a (potentially different)
          proportion p_down. walkerParams = [pup, pdown]
        - 'momentum': the walker moves as if it has mass m and inertia
          from the previous time step and the time series acts
          as a force altering its motion in a classical
          Newtonian dynamics framework. walkerParams = m (the mass).
          
    walkerParams : float, int, or list, optional
        The parameters for the specified walkerRule, explained above.
        
    Returns
    -------
    dict
        Various statistics summarizing properties of the residuals between the
        walker's trajectory and the original time series.
    """
    N = len(y)

    # Define default values and type requirements for each rule
    WALKER_CONFIGS = {
        'prop': {
            'default': 0.5,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        },
        'biasprop': {
            'default': [0.1, 0.2],
            'valid_types': (list,),
            'error_msg': 'must be a list'
        },
        'momentum': {
            'default': 2,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        }
    }

    if walkerRule not in WALKER_CONFIGS:
        valid_rules = ", ".join(f"'{rule}'" for rule in WALKER_CONFIGS.keys())
        raise ValueError(f"Unknown walker_rule: '{walkerRule}'. Choose from: {valid_rules}")
    
    # get configuration for the specified rule
    config = WALKER_CONFIGS[walkerRule]

    # use the default value if no parameters provided
    if walkerParams is None:
        walkerParams = config['default']

    if not isinstance(walkerParams, config["valid_types"]):
        raise ValueError(
            f"walkerParams {config['error_msg']} for walker rule: '{walkerRule}'"
        )
    
    # Do the walk
    w = np.zeros(N)

    if walkerRule == 'prop':
        #  % walker starts at zero and narrows the gap between its position
        #and the time series value at that point by the proportion given
        #in walkerParams, to give the value at the subsequent time step
        p = walkerParams
        w[0] = 0 # start at zero
        for i in range(1, N):
            w[i] = w[i-1] + p * (y[i-1] - w[i-1])
        
    elif walkerRule == 'biasprop':
        #walker is biased in one or the other direction (i.e., prefers to
        # go up, or down). Requires a vector of inputs: [p_up, p_down]
        pup, pdown = walkerParams

        w[0] = 0
        for i in range(1, N):
            if y[i] > y[i-1]: # time series inceases
                w[i] = w[i-1] + pup*(y[i-1]-w[i-1])
            else:
                w[i] = w[i-1] + pdown*(y[i-1]-w[i-1])
    elif walkerRule == 'momentum':
        #  % walker moves as if it had inertia from the previous time step,
        # i.e., it 'wants' to move the same amount; the time series acts as
        # a force changing its motion
        m = walkerParams # 'inertial mass'

        w[0] = y[0]
        w[1] = y[1]
        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1]-w[i-2])
            w[i] = w_inert + (y[i]-w_inert)/m # dissipative term
            #  % equation of motion (s-s_0=ut+F/m*t^2)
            # where the 'force' F is the change in the original time series
            # at that point
    else:
        raise ValueError(f"Unknown rule : {walkerRule}")
    
    # Get statistics on the walk
    out = {}
    # the walk itself
    out['w_mean'] = np.mean(w)
    out['w_median'] = np.median(w)
    out['w_std'] = np.std(w, ddof=1)
    out['w_ac1'] = AutoCorr(w, 1, 'Fourier')[0] # lag 1 autocorr
    out['w_ac2'] = AutoCorr(w, 2, 'Fourier')[0] # lag 2 autocorr
    out['w_tau'] = FirstCrossing(w, 'ac', 0, 'continuous')
    out['w_min'] = np.min(w)
    out['w_max'] = np.max(w)
    # fraction of time series length that walker crosses time series
    out['w_propzcross'] = (np.sum((w[:-1] * w[1:]) < 0)) / (N-1)

    # Differences between the walk at signal
    out['sw_meanabsdiff'] = np.mean(np.abs(y - w))
    out['sw_taudiff'] = FirstCrossing(y, 'ac', 0, 'continuous') - FirstCrossing(w, 'ac', 0 , 'continuous')
    out['sw_stdrat'] =  np.std(w, ddof=1)/np.std(y, ddof=1)
    out['sw_ac1rat'] = out['w_ac1']/AutoCorr(y, 1)[0]
    out['sw_minrat'] = np.min(w)/np.min(y)
    out['sw_maxrat'] = np.max(w)/np.max(y)
    out['sw_propcross'] = np.sum((w[:-1] - y[:-1]) * (w[1:] - y[1:]) < 0)/(N-1)

    #% test from same distribution: Ansari-Bradley test
    _, pval = ansari(w, y)
    out['sw_ansarib_pval'] = pval

    r = np.linspace(
        min(min(y), min(w)),
        max(max(y), max(w)),
        200
    )

    kde_y = gaussian_kde(y)
    kde_w = gaussian_kde(w)
    dy = kde_y(r)
    dw = kde_w(r)
    out['sw_distdiff'] = np.sum(np.abs(dy - dw))

    #% (iii) Looking at residuals between time series and walker
    res = w - y
    _, runs_pval = runstest_1samp(res, cutoff='mean')
    out['res_runstest'] = runs_pval
    out['res_swss5_1'] = SlidingWindow(res, 'std', 'std', 5, 1) # sliding window stationarity
    out['res_ac1'] = AutoCorr(res, 1)[0] # auto correlation at lag-1

    return out


def ForcePotential(y : ArrayLike, whatPotential : str = 'dblwell', params : Union[list, None] = None) -> dict:
    """
    Couples the values of the time series to a dynamical system.

    The input time series forces a particle in the given potential well.

    The time series contributes to a forcing term on a simulated particle in either:
        (i) A quartic double-well potential with potential energy V(x) = x^4/4 - alpha^2 x^2/2,
            or a force F(x) = -x^3 + alpha^2 x
        (ii) A sinusoidal potential with V(x) = -cos(x/alpha),
            or F(x) = sin(x/alpha)/alpha

    Parameters
    ----------
    y : array-like
        The input time series.
    whatPotential : str, optional
        The potential function to simulate:
            - 'dblwell': a double well potential function
            - 'sine': a sinusoidal potential function
    params : list, optional
        The parameters for simulation, should be in the form [alpha, kappa, deltat]:
            For 'dblwell':
                - alpha: controls the relative positions of the wells
                - kappa: coefficient of friction
                - deltat: time step for the simulation
            For 'sine':
                - alpha: controls the period of oscillations in the potential
                - kappa: coefficient of friction
                - deltat: time step for the simulation

    Returns
    --------
    dict
        Statistics summarizing the trajectory of the simulated particle.
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
