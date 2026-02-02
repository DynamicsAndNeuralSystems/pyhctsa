from typing import Union

from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import iqr
from scipy.interpolate import interp1d
import statsmodels.api as sm

from ..toolboxes.Max_Little import fastdfa
from ..utils import make_mat_buffer
from ..operations.correlation import autocorr

def fast_dfa(y: ArrayLike) -> float:
    """
    Measures the scaling exponent of the time series using a fast implementation
    of detrended fluctuation analysis (DFA).

    This is a Python wrapper for Max Little's fastdfa code.
    The original fastdfa code is by 

    References
    ----------
    .. [1] Max A. Little, http://www.maxlittle.net/software/index.php

    Parameterss
    ----------
    y : array-like
        Input time series (1D array), fed straight into the fastdfa script.

    Returns
    -------
    float
        Estimated scaling exponent from log-log linear fit of fluctuation vs interval.
    """
    y = np.asarray(y)
    intervals, flucts = fastdfa.fastdfa(y)
    idx = np.argsort(intervals)
    intervals_sorted = intervals[idx]
    flucts_sorted = flucts[idx]

    # Log-log linear fit
    coeffs = np.polyfit(np.log10(intervals_sorted), np.log10(flucts_sorted), 1)
    alpha = coeffs[0]
    
    return alpha

def fluctuation_analysis(x : ArrayLike, q: Union[float, int] = 2, 
                         wtf : str = 'rsrange', tau_step: int = 1, k : int = 1, 
                         lag: Union[int, None] = None, log_inc: bool = True):
    N = len(x)

    # Compute integrated sequence
    if (lag is None) | (lag == 1):
        # normal cumsum
        y = np.cumsum(x)
    else:
        # if a lag is specified, do a decimation...
        y = np.cumsum(x[::lag])
    
    # perform scaling over a range of tau, up to a fifth of the time-series length
    if log_inc:
        taur = np.unique(np.floor(np.exp(np.linspace(np.log(5), np.log(np.floor(N/2)), tau_step)) + 0.5))
    else:
        taur = np.arange(5, np.floor(N/2) + 1, tau_step)  # maybe increased??
    ntau = len(taur)  # analyze the time series across this many timescales
    if ntau < 8:  # fewer than 8 points
        print(f'This time series (N = {N}) is too short to analyze using this fluctuation analysis')
        out = np.nan
        return out
    
    F = np.zeros(ntau)
    # % 2) Compute the fluctuation function, F
    for i in range(ntau):
        tau = int(taur[i]) # time scale on which to compute fluctuations
        y_buff = make_mat_buffer(y, tau)
        if y_buff.shape[1] > (N // tau):  # zero-padded, remove trailing set of points...
            y_buff = y_buff[:, :-1]
        nn = y_buff.shape[1] * tau

        if wtf == 'nothing':
            y_dt = y_buff.reshape(nn, 1)
        elif wtf == 'endptdiff':
            y_dt = y_buff[-1, :] - y_buff[0, :]
        elif wtf == "range":
            y_dt = np.max(y_buff, axis=0) - np.min(y_buff, axis=0)
        elif wtf == 'std':
            raise NotImplementedError(f"{wtf} not yet implemented.")
        elif wtf == 'iqr':
            raise NotImplementedError(f"{wtf} not yet implemented.")
        elif wtf == 'dfa':
            tt = np.arange(1, tau + 1).reshape(-1, 1)  # faux time range (column vector)
            for j in range(y_buff.shape[1]):
                # fit a polynomial of order k in each subsegment
                p = np.polyfit(tt.flatten(), y_buff[:, j], k)
                # remove the trend, store back in y_buff
                y_buff[:, j] = y_buff[:, j] - np.polyval(p, tt.flatten())
            
            # reshape to a column vector, y_dt (detrended)
            y_dt = y_buff.reshape(-1, 1)
        elif wtf == 'rsrange':
            b = y_buff[0, :]
            m = y_buff[-1, :] - b
            y_buff = y_buff - (np.linspace(0, 1, tau).reshape(-1, 1) * m + np.ones((tau, 1)) * b)
            y_dt = np.ptp(y_buff, axis=0)
        elif wtf == 'rsrangefit':
            tt = np.arange(1, tau + 1).reshape(-1, 1)  # faux time range (column vector)
            for j in range(y_buff.shape[1]):
                # fit a polynomial of order k in each subsegment
                p = np.polyfit(tt.flatten(), y_buff[:, j], k)
                # remove the trend, store back in y_buff
                y_buff[:, j] = y_buff[:, j] - np.polyval(p, tt.flatten())

            y_dt = np.ptp(y_buff, axis=0)
        else:
            raise ValueError(f"Unknown fluctuation analysis method: {wtf}")
        F[i] = np.mean(y_dt**q)**(1/q)
    # % Smooth unevenly-distributed points in log space:
    if log_inc:
        logtt = np.log(taur)
        logFF = np.log(F)
        num_timescales = ntau
    else: # need to smooth the unevenly-distributed points (using a spline)
        logtaur = np.log(taur)
        logF = np.log(F)
        num_timescales = 50
        logtt = np.linspace(np.min(logtaur), np.max(logtaur), num_timescales)
        logFF = interp1d(logtaur, logF, kind='cubic')(logtt)
    #% Linear fit the log-log plot: full range
    out = _robust_linear_fit(logtt, logFF, np.arange(1, num_timescales), '')

    sserr = np.full(num_timescales, np.nan)  # don't choose the end points
    min_points = 6

    for i in range(min_points, num_timescales - min_points):
        r1 = slice(0, i + 1)  # 1:i in MATLAB (inclusive)
        p1 = np.polyfit(logtt[r1], logFF[r1], 1)
        
        r2 = slice(i, num_timescales)  # i:numTimeScales in MATLAB
        p2 = np.polyfit(logtt[r2], logFF[r2], 1)
        
        # Sum of errors from fitting lines to both segments:
        sserr[i] = (np.linalg.norm(np.polyval(p1, logtt[r1]) - logFF[r1]) + 
                    np.linalg.norm(np.polyval(p2, logtt[r2]) - logFF[r2]))
    
    break_pt = np.where(sserr == np.nanmin(sserr))[0][0]  # find first occurrence of minimum
    r1 = np.arange(0, break_pt + 1)
    r2 = np.arange(break_pt, num_timescales)

    out['prop_r1'] = len(r1) / num_timescales
    out['logtausplit'] = logtt[break_pt]
    out['ratsplitminerr'] = np.nanmin(sserr) / out['ssr']
    out['meanssr'] = np.nanmean(sserr)
    out['stdssr'] = np.nanstd(sserr)

    out2 = _robust_linear_fit(logtt, logFF, r1, 'r1_')
    out3 = _robust_linear_fit(logtt, logFF, r2, 'r2_')

    out_final = out | out2 | out3

    if np.isnan(out_final['r1_alpha']) or np.isnan(out_final['r2_alpha']):
        out_final['alpha_rat'] = np.nan
    else:
        out_final['alpha_rat'] = out_final['r1_alpha']/out_final['r2_alpha']

    return out_final


def _robust_linear_fit(logtt, logFF, the_range, field_name):
    X = sm.add_constant(logtt[the_range])
    rlm = sm.RLM(logFF[the_range], X, M=sm.robust.norms.TukeyBiweight())
    results = rlm.fit()
    linfit = results.params  # [intercept, slope]
    out = {}
    # Store results in dictionary (Python equivalent of MATLAB struct)
    out[f'{field_name}linfitint'] = linfit[0]  # linear fit intercept
    out[f'{field_name}alpha'] = linfit[1]  # linear fit gradient
    out[f'{field_name}se1'] = results.bse[0]  # standard error in intercept
    out[f'{field_name}se2'] = results.bse[1]  # standard error in slope
    out[f'{field_name}ssr'] = np.mean(results.resid**2)  # mean squares residual
    out[f'{field_name}resac1'] = autocorr(results.resid, 1, 'Fourier')[0]  # autocorr at lag 1
    return out





