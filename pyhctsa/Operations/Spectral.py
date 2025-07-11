import scipy.fft
import numpy as np
from pyhctsa.Operations.Distribution import Moments
from pyhctsa.Operations.Correlation import AutoCorr, FirstCrossing
from pyhctsa.Utilities.utils import make_mat_buffer, signChange
from scipy.optimize import curve_fit
from numpy.typing import ArrayLike

def SpectralSummaries(y : ArrayLike, psdMeth : str = 'fft', windowType : str = 'none') -> dict:

    y = np.asarray(y)
    Ny = len(y)

    window = None
    # Set window (for periodogram and welch):
    if windowType == 'none':
        window = []
    elif windowType == 'hamming':
        window = np.hamming(Ny)
    elif windowType == 'hann':
        window = np.hanning(Ny)
    elif windowType == 'bartlett':
        window = np.bartlett(Ny)
    elif windowType == 'boxcar':
        window = scipy.signal.windows.boxcar(Ny)
    elif windowType == 'rect':
        window = np.ones(Ny)
    else:
        raise ValueError(f"Unknown window: {windowType}")

    # Compute the Fourier Transform
    if psdMeth == 'fft':
        Fs = 1 # sampling freq
        NFFT = 2**(int(np.ceil(np.log2(Ny)))) # next power of 2
        f = (Fs/2) * np.linspace(0, 1, int(NFFT/2)+1) # freq
        w = 2 * np.pi * f # angular freq
        S = scipy.fft.fft(y, NFFT) # do the fourier transform
        S = 2*np.abs(S[:int(NFFT/2)+1])**2/Ny # single-sided power spectral density
        S = S/(2*np.pi) # convert to angular freq space

    elif psdMeth == 'welch':
        # welch power spectral density estimate
        Fs = 1
        N = 2**(int(np.ceil(np.log2(Ny))))
        f, S = scipy.signal.welch(y, window=window, noverlap=0, nfft=N, fs=Fs)
        w = 2 * np.pi * f # angular frequency
        S = S/(2*np.pi) # adjust so that area remains normalized in angular frequency space

    # elif psdMeth == 'periodogram':
    #     if nf:
    #         w = np.linspace(0, np.pi, nf)
    #         S, w = scipy.signal.periodogram(y, window=window, )
    #     else:
    #         w, S = scipy.signal.periodogram(y, window=window)
    else:
        raise ValueError(f"Unknown spectral estimation method: {psdMeth}.")
    
    if not np.any(np.isfinite(S)):
        return np.nan
    
    N = len(S)
    logS = np.log(S)
    dw = w[1] - w[0] # spacing increment in w

    # Simple measures of the power spectrum
    # Peaks 
    out = {}
    i_maxS = np.argmax(S)
    out = {'maxS': S[i_maxS], 'maxw': w[i_maxS]}
    r, l = np.where(S[i_maxS+1:] < S[i_maxS])[0], np.where(S[:i_maxS] < S[i_maxS])[0]
    out['maxWidth'] = w[i_maxS + 1 + r[0]] - w[l[-1]] if len(r) > 0 and len(l) > 0 else 0

    right_indices = np.where(S[i_maxS+1:] < out['maxS'])[0]
    if len(right_indices) > 0:
        right_idx = i_maxS + 1 + right_indices[0]
    else:
        right_idx = None

    # Find last index before i_maxS where S < maxS
    left_indices = np.where(S[:i_maxS] < out['maxS'])[0]
    if len(left_indices) > 0:
        left_idx = left_indices[-1]
    else:
        left_idx = None

    # Calculate maxWidth
    if right_idx is not None and left_idx is not None:
        out['maxWidth'] = w[right_idx] - w[left_idx]
    else:
        out['maxWidth'] = 0


    minDist_w = 0.02
    ptsPerw = len(S)/np.pi
    minPkDist = np.ceil(minDist_w*ptsPerw)
    pkHeight, pkLoc = _findpeaks(S, minPkDist, 'descend')
    pkWidth = scipy.signal.peak_widths(S, pkLoc)[0]
    pkProm = (scipy.signal.peak_prominences(S, pkLoc)[0])
    pkWidth = pkWidth/ptsPerw
    pkLoc = pkLoc/ptsPerw # diff due to indexing difference

    # Characterize mean peak prominence
    out['numPeaks'] = len(pkHeight)
    out['numPromPeaks_1'] = np.sum(pkProm > 1) # number of peaks with prominence of at least 1
    out['numPromPeaks_2'] = np.sum(pkProm > 2) # number of peaks with prominence of at least 2
    out['numPromPeaks_5'] = np.sum(pkProm > 5)# number of peaks with prominence of at least 5
    out['numPeaks_overmean'] = np.sum(pkProm>np.mean(pkProm)) # number of peaks with prominence greater than the mean (low for skewed distn)
    out['maxProm'] = np.max(pkProm)
    out['meanProm_2'] = np.mean(pkProm[pkProm > 2]) # mean peak prominence of those with prominence of at least 2
    out['meanPeakWidth_prom2'] = np.mean(pkWidth[pkProm > 2])
    out['width_weighted_prom'] = np.sum(pkWidth*pkProm)/np.sum(pkProm)

    # Power in top N peaks
    nn = lambda x : np.arange(0, np.minimum(x, out['numPeaks']-1))
    out['peakPower_2'] = np.sum(pkHeight[nn(2)]*pkWidth[nn(2)])
    out['peakPower_5'] = np.sum(pkHeight[nn(5)]*pkWidth[nn(5)])
    out['peakPower_prom2'] = np.sum(pkHeight[pkProm > 2]*pkWidth[pkProm > 2]) # power in peaks with prominence of at least 2
    # note any features which depend on pKLoc will yield slightly diff answers due to one-indexing, but should be perfectly correlated
    out['w_weighted_peak_prom'] = np.sum(pkLoc*pkProm)/np.sum(pkProm)
    out['w_weighted_peak_height'] = np.sum(pkLoc*pkHeight)/np.sum(pkHeight)#where are prominent peaks located on average (weighted by height)
    # Number of peaks required to get to 50% of power in peaks
    peakPower = pkHeight*pkWidth
    out['numPeaks_50power'] = np.where(np.cumsum(peakPower) > 0.5 * np.sum(peakPower))[0][0]
    out['peakpower_1'] = peakPower[0]/sum(peakPower)

    # Distribution
    # quantiles
    iqr75 = np.quantile(S, 0.75, method='hazen')
    iqr25 = np.quantile(S, 0.25, method='hazen')
    out['iqr'] = iqr75 - iqr25
    out['logiqr'] = np.quantile(logS, 0.75, method='hazen') - np.quantile(logS, 0.25, method='hazen')
    out['q25'] = iqr25
    out['median'] = np.median(S)
    out['q75'] = iqr75

    # Moments
    out['std'] = np.std(S, ddof=1)
    out['stdlog'] = np.log(out['std'])
    out['logstd'] = np.std(logS, ddof=1)
    out['mean'] = np.mean(S)
    out['logmean'] = np.mean(logS)
    for i in range(3, 6):
        out[f'mom{i}'] = Moments(S, i)

    # Autocorrelation of amplitude spectrum:
    autoCorrs_S = AutoCorr(S, [1, 2, 3, 4], 'Fourier')
    out['ac1'] = autoCorrs_S[0]
    out['ac2'] = autoCorrs_S[1]
    out['tau'] = FirstCrossing(S, 'ac', 0, 'continuous') # first zero crossing


    # Shape of cumulative sum curve
    csS = np.cumsum(S)
    f_frac_w_max = lambda f: w[np.where(csS >= csS[-1] * f)[0][0]]
    # @ what frequency is csS a fraction p of its maximum?
    out['wmax_5'] = f_frac_w_max(0.05)
    out['wmax_10'] = f_frac_w_max(0.1)
    out['wmax_25'] = f_frac_w_max(0.25)
    out['centroid'] = f_frac_w_max(0.5)
    out['wmax_75'] = f_frac_w_max(0.75)
    out['wmax_90'] = f_frac_w_max(0.9)
    out['wmax_95'] = f_frac_w_max(0.95)
    out['wmax_99'] = f_frac_w_max(0.99)

    #Width of saturation measures
    out['w10_90'] = out['wmax_90'] - out['wmax_10'] # % from 10% to 90%:
    out['w25_75'] = out['wmax_75'] - out['wmax_25']

    # Fit some functions to this cumulative sum:
    # Quadratic
    a, b, c = np.polyfit(w, csS, deg=2)
    out['fpoly2csS_p1'] = a
    out['fpoly2csS_p2'] = b
    out['fpoly2csS_p3'] = c
    quad = lambda x, a, b, c : a*x**2 + b*x + c 
    residuals = quad(w, a, b, c) - csS
    sum_sq_err = np.sum(residuals**2)
    out['fpoly2_sse'] = sum_sq_err
    out['fpoly2_r2'] = 1 - (sum_sq_err/(np.sum((csS - np.mean(csS))**2)))

    # Fit polysat a*x^2/(b+x^2) (has zero derivative at zero, though)
    # polysat = lambda x, a, b : (a*(x**2))/(b + x**2)
    # popt, _ = curve_fit(polysat, w, csS, p0=[csS[-1], 100])
    # a, b = popt
    # out['fpolysat_a'] = a
    # out['fpolysat_b'] = b
    # residuals = polysat(w, a, b) - csS
    # sum_sq_err = np.sum(residuals**2)
    # out['fpolysat_r2'] = 1 - (sum_sq_err/(np.sum((csS - np.mean(csS))**2)))
    # out['fpolysat_rmse'] = np.sqrt(np.mean(residuals**2))


    # Shannon spectral entropy
    Hshann = - S * np.log(S)
    out['spect_shann_ent'] = np.sum(Hshann)
    out['spect_shann_ent_norm'] = np.mean(Hshann)

    #"Spectral Flatness Measure"
    #which is given in dB as 10 log_10(gm/am) where gm is the geometric mean and am
    # is the arithmetic mean of the power spectral density
    out['sfm'] = 10*np.log10(np.exp(np.mean(np.log(S)))/np.mean(S))

    # Areas under power spectrum
    out['areatopeak'] = np.sum(S[0:np.argmax(S)+1])*dw
    out['ylogareatopeak'] = np.sum(logS[0:np.argmax(S)+1])*dw #% (semilogy)

    # TODO: RobustFits

    # Power in specific frequency bands
    # % 2 bands
    split = make_mat_buffer(S, int(np.floor(N/2)))
    if split.shape[1] > 2:
        split = split[:, :2]
    out['area_2_1'] = np.sum(split[:,0])*dw
    out['logarea_2_1'] = np.sum(np.log(split[:,0]))*dw
    out['area_2_2'] = np.sum(split[:,1])*dw
    out['logarea_2_2'] = np.sum(np.log(split[:, 1]))*dw
    out['statav2_m'] = np.std(np.mean(split, axis=0), ddof=1)/np.std(S, ddof=1)
    out['statav2_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1)/np.std(S, ddof=1)

    # 3 bands
    split = make_mat_buffer(S, int(np.floor(N/3)))
    if split.shape[1] > 3:
        split = split[:, :3]
    out['area_3_1'] = np.sum(split[:,0])*dw
    out['logarea_3_1'] = np.sum(np.log(split[:,0]))*dw
    out['area_3_2'] = np.sum(split[:,1])*dw
    out['logarea_3_2'] = np.sum(np.log(split[:, 1]))*dw
    out['area_3_3'] = np.sum(split[:,2])*dw
    out['logarea_3_3'] = np.sum(np.log(split[:, 2]))*dw
    out['statav3_m'] = np.std(np.mean(split, axis=0), ddof=1)/np.std(S, ddof=1)
    out['statav3_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1)/np.std(S, ddof=1)

    # 4 bands
    split = make_mat_buffer(S, int(np.floor(N/4)))
    if split.shape[1] > 4:
        split = split[:, :4]
    out['area_4_1'] = np.sum(split[:,0])*dw
    out['logarea_4_1'] = np.sum(np.log(split[:,0]))*dw
    out['area_4_2'] = np.sum(split[:,1])*dw
    out['logarea_4_2'] = np.sum(np.log(split[:, 1]))*dw
    out['area_4_3'] = np.sum(split[:,2])*dw
    out['logarea_4_3'] = np.sum(np.log(split[:, 2]))*dw
    out['area_4_4'] = np.sum(split[:,3])*dw
    out['logarea_4_4'] = np.sum(np.log(split[:, 3]))*dw
    out['statav4_m'] = np.std(np.mean(split, axis=0), ddof=1)/np.std(S, ddof=1)
    out['statav4_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1)/np.std(S, ddof=1)

    # 5 bands
    split = make_mat_buffer(S, int(np.floor(N/5)))
    if split.shape[1] > 5:
        split = split[:, :5]
    out['area_5_1'] = np.sum(split[:,0])*dw
    out['logarea_5_1'] = np.sum(np.log(split[:,0]))*dw
    out['area_5_2'] = np.sum(split[:,1])*dw
    out['logarea_5_2'] = np.sum(np.log(split[:, 1]))*dw
    out['area_5_3'] = np.sum(split[:,2])*dw
    out['logarea_5_3'] = np.sum(np.log(split[:, 2]))*dw
    out['area_5_4'] = np.sum(split[:,3])*dw
    out['logarea_5_4'] = np.sum(np.log(split[:, 3]))*dw
    out['area_5_5'] = np.sum(split[:,4])*dw
    out['logarea_5_5'] = np.sum(np.log(split[:, 4]))*dw
    out['statav5_m'] = np.std(np.mean(split, axis=0), ddof=1)/np.std(S, ddof=1)
    out['statav5_s'] = np.std(np.std(split, ddof=1, axis=0), axis=0, ddof=1)/np.std(S, ddof=1)

    # Count crossings:
    # Get a horizontal line and count the number of crossings with the power spectrum
    ncrossfn_rel = lambda f : np.sum(signChange(S - f * np.max(S)))
    out['ncross_f05'] = ncrossfn_rel(0.05)
    out['ncross_f01'] = ncrossfn_rel(0.1)
    out['ncross_f02'] = ncrossfn_rel(0.2)
    out['ncross_f05'] = ncrossfn_rel(0.5)
    
    return out

def _findpeaks(S, minPkDist=0, sort_str='none'):
    """
    MATLAB-compatible findpeaks implementation
    
    Parameters:
    S: input signal
    minPkDist: minimum peak distance
    sort_str: 'none', 'ascend', or 'descend'
    
    Returns:
    pkHeight, pkLoc
    """
    # find ALL local maxima
    # a peak is considered to be a point higher than both neighbors
    # Handle infinite values
    inf_peaks = np.where(np.isinf(S) & (S > 0))[0]
    
    # Find finite peaks by checking if each point is greater than both neighbors
    finite_peaks = []
    for i in range(1, len(S) - 1):
        if not np.isinf(S[i]) and not np.isnan(S[i]):
            if S[i] > S[i-1] and S[i] > S[i+1]:
                finite_peaks.append(i)
    
    finite_peaks = np.array(finite_peaks, dtype=int)
    
    # Combine finite and infinite peaks
    all_peaks = np.concatenate([finite_peaks, inf_peaks]) if len(inf_peaks) > 0 else finite_peaks
    all_peaks = np.sort(all_peaks)
    
    if len(all_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # apply minimum peak distance constraint
    if minPkDist > 0:
        # start with largest peaks and remove smaller ones in neighborhood
        peak_heights = S[all_peaks]
        
        # sort by height (descending)
        sort_idx = np.argsort(peak_heights)[::-1]
        sorted_peaks = all_peaks[sort_idx]
        
        # keep track of which peaks to delete
        to_delete = np.zeros(len(sorted_peaks), dtype=bool)
        
        for i in range(len(sorted_peaks)):
            if not to_delete[i]:
                current_peak = sorted_peaks[i]
                # mark all peaks within minPkDist of current peak for deletion
                for j in range(len(sorted_peaks)):
                    if not to_delete[j]:
                        distance = abs(sorted_peaks[j] - current_peak)
                        if distance <= minPkDist and distance > 0:
                            to_delete[j] = True
        
        # keep only non-deleted peaks
        final_peaks = sorted_peaks[~to_delete]
        
        # convert back to original indices for sorting
        back_to_original = np.zeros(len(final_peaks), dtype=int)
        for i, peak in enumerate(final_peaks):
            back_to_original[i] = np.where(all_peaks == peak)[0][0]
        
        final_peaks = all_peaks[np.sort(back_to_original)]
    else:
        final_peaks = all_peaks
    
    if len(final_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    pkHeight = S[final_peaks]
    pkLoc = final_peaks.astype(int)
    
    if sort_str == 'descend':
        sort_idx = np.argsort(pkHeight)[::-1]
        pkHeight = pkHeight[sort_idx]
        pkLoc = pkLoc[sort_idx]
    elif sort_str == 'ascend':
        sort_idx = np.argsort(pkHeight)
        pkHeight = pkHeight[sort_idx]
        pkLoc = pkLoc[sort_idx]
    
    return pkHeight, pkLoc
