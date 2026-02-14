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

def fluctuation_analysis(x: ArrayLike, q: Union[float, int] = 2,
                         wtf: str = 'rsrange', tau_step: int = 1, k: int = 1,
                         lag: Union[int, None] = None, log_inc: bool = True):
    """
    Implements fluctuation analysis by a variety of methods.
    
    Much of our implementation is based on the well-explained discussion of scaling
    methods [1]_.

    The main difference between algorithms for estimating scaling exponents amount
    to differences in how fluctuations, F, are quantified in time-series segments.
    Many alternatives are implemented in this function.
    
    Parameters
    ----------
    x : ArrayLike
        The input time series.
    q : Union[float, int], optional
        The parameter in the fluctuation function. q = 2 (default) gives RMS 
        fluctuations.
    wtf : str, optional
        What to fluctuate. Options are:
        
        - 'endptdiff': Calculates the differences in end points in each segment
        - 'range': Calculates the range in each segment
        - 'std': Takes the standard deviation in each segment [1]_
        - 'iqr': Takes the interquartile range in each segment
        - 'dfa': Removes a polynomial trend of order k in each segment
        - 'rsrange': Returns the range after removing a straight line fit [2]_
        - 'rsrangefit': Fits a polynomial of order k and returns the range [2]_
        
        For 'rsrangefit', an optional timelag can be applied for computing the 
        cumulative sum (integrated profile) [3]_.
    tau_step : int, optional
        number of tau (locInc true), or increments in tau for linear range
    k : int, optional
        polynomial order of detrending (for 'dfa' & 'rsrangefit')
    lag : int or None, optional
        optional time-lag, as in Alvarez-Ramirez [3]_
    log_inc : bool, optional
        whether to use logarithmic increments in tau (it should be logarithmic)
    
    Returns
    -------
    dict
        Statistics of fitting a linear function to a plot of log(F) as
        a function of log(tau), and for fitting two straight lines to the same data,
        choosing the split point at tau = tau_{split} as that which minimizes the
        combined fitting errors.
    
    References
    ----------
    .. [1] "Power spectrum and detrended fluctuation analysis: Application to daily
        temperatures" P. Talkner and R. O. Weber, Phys. Rev. E 62(1) 150 (2000)
    .. [2] D. C. Caccia et al., "Analyzing exact fractal time series: evaluating dispersional
        analysis and rescaled range methods", Physica A 246(3-4) 609 (1997)
    .. [3] J. Alvarez-Ramirez et al., "Using detrended fluctuation analysis for lagged 
        correlation analysis of nonstationary signals", Phys. Rev. E 79(5) 057202 (2009)
    """
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


def _robust_linear_fit(log_tt, log_ff, the_range, field_name):
    """
    Robust linear fit using Tukey's biweight function for M-estimation. 
    """
    x = sm.add_constant(log_tt[the_range])
    rlm = sm.RLM(log_ff[the_range], x, M=sm.robust.norms.TukeyBiweight())
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
