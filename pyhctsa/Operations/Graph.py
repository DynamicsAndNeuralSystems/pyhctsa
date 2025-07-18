from ts2vg import NaturalVG
import numpy as np
from numpy.typing import ArrayLike
import scipy
from scipy.stats import norm, expon
from pyhctsa.Operations.Correlation import AutoCorr, FirstCrossing
from pyhctsa.Operations.Entropy import DistributionEntropy

def _horiz_vgraph(ts_data):
    # helper function for Visibility graph
    # Ensure ts_data is a NumPy array
    ts_data = np.asarray(ts_data)
    N = len(ts_data)
    # Initialize an empty adjacency matrix
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        # --- Look forward for the first taller neighbor ---
        # We only need to look forward if we are not the last node
        if i < N - 1:
            # Create a slice of the data from the next element to the end
            forward_slice = ts_data[i+1:]
            # Find the indices of all nodes in the slice that are taller than the current node
            # np.where returns a tuple of arrays, we take the first element [0]
            taller_nodes_fwd = np.where(forward_slice > ts_data[i])[0]
            # If any taller nodes were found
            if taller_nodes_fwd.size > 0:
                # The first element in this array corresponds to the nearest taller node
                first_taller_relative_idx = taller_nodes_fwd[0]  
                # Convert the relative index (from the slice) to an absolute index (from the original series)
                first_taller_absolute_idx = i + 1 + first_taller_relative_idx      
                # Set the connection in the adjacency matrix
                A[i, first_taller_absolute_idx] = 1
        if i > 0:
            # Create a slice of the data from the beginning up to the current node
            backward_slice = ts_data[:i]
            # Find the indices of all nodes in the slice that are taller than the current node
            taller_nodes_bwd = np.where(backward_slice > ts_data[i])[0]
            # If any taller nodes were found
            if taller_nodes_bwd.size > 0:
                closest_taller_absolute_idx = taller_nodes_bwd[-1]
                A[closest_taller_absolute_idx, i] = 1

    A = np.maximum(A, A.T)
    
    return A


def VisibilityGraph(y : ArrayLike, meth : str = 'horiz', maxL : int = 5000) -> dict:
    y = np.asarray(y)
    N = len(y)
    if N > maxL:
        # too long to store in memory
        print(f"Time series ({N} > {maxL}) is too long for visibility graph. Analyzing the first {maxL} samples.")
        y = y[:maxL]
        N = len(y)
    y = y - np.min(y) # adjust so that the minimum of y is at zero

    # Compute the visibility graph:
    if meth == 'horiz':
        A = _horiz_vgraph(y)
        k = A.sum(axis=0)

    elif meth == 'norm':
        vg = NaturalVG()
        vg.build(y,only_degrees=True)
        k = vg._degrees

    out = {}
    # Degree distribution: basic statistics
    m, c = scipy.stats.mode(k)
    out['mode'] = m
    out['propmode'] = sum(k == out['mode'])/sum(k)
    out['meank'] = np.mean(k) # mean number of links per node
    out['mediank'] = np.median(k)
    out['stdk'] = np.std(k, ddof=1)
    out['maxk'] = np.max(k)
    out['mink'] = np.min(k)
    out['rangek'] = np.ptp(k)
    out['iqrk'] = np.quantile(k, .75, method='hazen') - np.quantile(k, .25, method='hazen') 
    out['skewnessk'] = scipy.stats.skew(k)
    out['maxonmedian'] = np.max(k)/np.median(k) # max on median (indicator of outlier)
    out['ol90'] = np.mean(k[(k >= np.quantile(k, 0.05, method='hazen')) & (k <= np.quantile(k, 0.95, method='hazen'))])/np.mean(k)
    out['olu90'] = np.mean(k[k >= np.quantile(k, 0.95, method='hazen')] - np.mean(k))/np.std(k, ddof=1)

    # Fit distributions to degree distribution

    # Entropy of distribution 
    out['entropy'] = DistributionEntropy(k, 'hist', 'sqrt')

    #Using likelihood now:
    out['gaussnlogL'] = -np.sum(norm.logpdf(k, loc=np.mean(k), scale=np.std(k, ddof=1)))
    out['expnlogL'] = -np.sum(expon.logpdf(k, scale=np.mean(k)))
    

    # Autocorr
    out['kac1'] = AutoCorr(k, 1, 'Fourier')[0]
    out['kac2'] = AutoCorr(k, 2, 'Fourier')[0]
    out['kac3'] = AutoCorr(k, 3, 'Fourier')[0]
    out['ktau'] = FirstCrossing(k, 'ac', 0, 'continuous')

    return out
