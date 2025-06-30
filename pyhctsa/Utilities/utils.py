
import numpy as np
import csv
from numpy.typing import ArrayLike
from typing import Union
import os
from functools import wraps

def get_dataset(which : str = "e1000"):
    """
    Load data for testing and validation.
    Options are either 'e1000' or 'sinusoid'. 
    """
    dataset = []
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    if which == "e1000":
        print("Loading empirical1000 dataset...")
        # Get the absolute path to the data directory
        data_path = os.path.join(utils_dir, "../../data/e1000.csv")
        data_path = os.path.normpath(data_path)
        with open(data_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                try:
                    time_series = [float(value) for value in row if value != '']
                    dataset.append(time_series)
                except ValueError as e:
                    print(f"Skipping row due to conversion error: {row}")
                continue
    elif which == "sinusoid":
        print("Loading sinusoid dataset...")
        data_path = os.path.join(utils_dir, "../../data/sinusoid.txt")
        dataset.append(np.loadtxt(data_path))
    else:
        raise NotImplementedError("Dataset not found.")
    print(f"Loaded dataset of {len(dataset)} time series.")
    return dataset
    
def preprocess_decorator(zscore=False, absval=False):
    """
    Z-score or take the absolute value of the time series before passing into
    the master operation.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            if zscore:
                x = ZScore(x)
            if absval:
                x = np.abs(x)
            return func(x, *args, **kwargs)
        return wrapper
    return decorator

def ZScore(x : ArrayLike) -> np.ndarray:
    """
    Z-score the input data vector.

    This function standardizes the input array by removing the mean and scaling to unit variance.
    It performs the z-scoring operation twice to reduce numerical error, as recommended for 
    high-precision applications.

    Parameters
    ----------
    x : array-like
        Input data vector (1D array or list of numbers).

    Returns
    -------
    np.ndarray
        The z-scored version of the input data.

    Raises
    ------
    ValueError
        If the input contains NaN values.
    """
    # Convert input to numpy array
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input cannot be converted to numeric array: {e}")
    
    if x.size == 0:
        raise ValueError("Input array is empty.")

    # Check for NaNs, infs, etc. i.e., check data is finite
    if not np.isfinite(x).all():
        raise ValueError(f'data contains non-finite values (NaN/inf) at idxs: {np.argwhere(np.isfinite(x) == False)}')
    
    # robust checks for constant values
    var_x = np.var(x, ddof=1)
    if var_x < 1e-10:
        raise ValueError(f"Data has sample variance {var_x:.2e} < 1e-10. Values appear to be constant.")
    data_range = np.ptp(x)  # peak-to-peak (max - min)
    if data_range < 1e-10:
        raise ValueError(f"Data range {data_range:.2e} < 1e-10. Values appear to be constant.")

    # Z-score twice to reduce numerical error
    zscored_data = np.divide((x - np.mean(x)), np.std(x, ddof=1))
    zscored_data = np.divide((zscored_data - np.mean(zscored_data)), np.std(zscored_data, ddof=1))

    return zscored_data

def histc(x, bins):
    # reproduce the behaviour of MATLAB's histc function
    map_to_bins = np.digitize(x, bins) # Get indices of the bins to which each value in input array belongs.
    res = np.zeros(bins.shape)
    for el in map_to_bins:
        res[el-1] += 1 # Increment appropriate bin.
    return res

def binpicker(xmin, xmax, nbins, bindwidthEst=None):
    """
    Choose histogram bins. 
    A 1:1 port of the internal MATLAB function.


    Parameters:
    -----------
    xmin : float
        Minimum value of the data range.
    xmax : float
        Maximum value of the data range.
    nbins : int or None
        Number of bins. If None, an automatic rule is used.

    Returns:
    --------
    edges : numpy.ndarray
        Array of bin edges.
    """
    if bindwidthEst == None:
        rawBinWidth = abs(xmax - xmin)/nbins
    else:
        rawBinWidth = bindwidthEst


    if xmin is not None:
        if not np.issubdtype(type(xmin), np.floating):
            raise ValueError("Input must be float type when number of bins is specified.")

        xscale = max(abs(xmin), abs(xmax))
        xrange = xmax - xmin

        # Make sure the bin width is not effectively zero
        rawBinWidth = max(rawBinWidth, np.spacing(xscale))

        # If the data are not constant, place the bins at "nice" locations
        if xrange > max(np.sqrt(np.spacing(xscale)), np.finfo(xscale).tiny):
            # Choose the bin width as a "nice" value
            pow_of_ten = 10 ** np.floor(np.log10(rawBinWidth))
            rel_size = rawBinWidth / pow_of_ten  # guaranteed in [1, 10)

            # Automatic rule specified
            if nbins is None:
                if rel_size < 1.5:
                    bin_width = 1 * pow_of_ten
                elif rel_size < 2.5:
                    bin_width = 2 * pow_of_ten
                elif rel_size < 4:
                    bin_width = 3 * pow_of_ten
                elif rel_size < 7.5:
                    bin_width = 5 * pow_of_ten
                else:
                    bin_width = 10 * pow_of_ten

                left_edge = max(min(bin_width * np.floor(xmin / bin_width), xmin), -np.finfo(xmax).max)
                nbins_actual = max(1, np.ceil((xmax - left_edge) / bin_width))
                right_edge = min(max(left_edge + nbins_actual * bin_width, xmax), np.finfo(xmax).max)

            # Number of bins specified
            else:
                bin_width = pow_of_ten * np.floor(rel_size)
                left_edge = max(min(bin_width * np.floor(xmin / bin_width), xmin), -np.finfo(xmin).max)
                if nbins > 1:
                    ll = (xmax - left_edge) / nbins
                    ul = (xmax - left_edge) / (nbins - 1)
                    p10 = 10 ** np.floor(np.log10(ul - ll))
                    bin_width = p10 * np.ceil(ll / p10)

                nbins_actual = nbins
                right_edge = min(max(left_edge + nbins_actual * bin_width, xmax), np.finfo(xmax).max)

        else:  # the data are nearly constant
            if nbins is None:
                nbins = 1

            bin_range = max(1, np.ceil(nbins * np.spacing(xscale)))
            left_edge = np.floor(2 * (xmin - bin_range / 4)) / 2
            right_edge = np.ceil(2 * (xmax + bin_range / 4)) / 2

            bin_width = (right_edge - left_edge) / nbins
            nbins_actual = nbins

        if not np.isfinite(bin_width):
            edges = np.linspace(left_edge, right_edge, nbins_actual + 1)
        else:
            edges = np.concatenate([
                [left_edge],
                left_edge + np.arange(1, nbins_actual) * bin_width,
                [right_edge]
            ])
    else:
        # empty input
        if nbins is not None:
            edges = np.arange(nbins + 1, dtype=float)
        else:
            edges = np.array([0.0, 1.0])

    return edges

def simple_binner(xData, numBins) -> tuple:
    """
    Generate a histogram from equally spaced bins.
   
    Parameters:
    xData (array-like): A data vector.
    numBins (int): The number of bins.

    Returns:
    tuple: (N, binEdges)
        N (numpy.ndarray): The counts
        binEdges (numpy.ndarray): The extremities of the bins.
    """
    minX = np.min(xData)
    maxX = np.max(xData)
    
    # Linearly spaced bins:
    binEdges = np.linspace(minX, maxX, numBins + 1)
    N = np.zeros(numBins, dtype=int)
    
    for i in range(numBins):
        if i < numBins - 1:
            N[i] = np.sum((xData >= binEdges[i]) & (xData < binEdges[i+1]))
        else:
            # the final bin
            N[i] = np.sum((xData >= binEdges[i]) & (xData <= binEdges[i+1]))
    
    return N, binEdges

def pointOfCrossing(x, threshold, oneIndexing=True):
    """
    Linearly interpolate to the point of crossing a threshold
    
    Parameters:
        x (array-like): a vector
        threshold (float): a threshold x crosses
        ondeIndexing (bool): whether to use zero or one indexing for consistency with MATLAB implementation.
    
    Returns:
        tuple: (firstCrossing, pointOfCrossing)
        firstCrossing (int): the first discrete value after which a crossing event has occurred
        pointOfCrossing (float): the (linearly) interpolated point of crossing

    """
    x = np.asarray(x)

    if x[0] > threshold:
        crossings = np.where((x - threshold) < 0)[0]
    else:
        crossings = np.where((x - threshold) > 0)[0]

    if crossings.size == 0:
        # Never crosses
        N = len(x)
        firstCrossing = N
        pointOfCrossing = N
    else:
        firstCrossing = crossings[0]
        # Continuous version
        valueBefore = x[firstCrossing - 1]
        valueAfter = x[firstCrossing]
        pointOfCrossing = firstCrossing - 1 + (threshold - valueBefore) / (valueAfter - valueBefore)

        if oneIndexing:
            firstCrossing += 1
            pointOfCrossing += 1

    return firstCrossing, pointOfCrossing

def signChange(y : Union[list, np.ndarray], doFind=0):
    """
    Where a data vector changes sign.
    """
    if doFind == 0:
        return (np.multiply(y[1:],y[0:len(y)-1]) < 0)
    indexs = np.where((np.multiply(y[1:],y[0:len(y)-1]) < 0))[0]

    return indexs

def make_buffer(y, bufferSize):
    """
    Make a buffered version of a time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    bufferSize : int
        The length of each buffer segment.

    Returns
    -------
    y_buffer : ndarray
        2D array where each row is a segment of length `bufferSize` 
        corresponding to consecutive, non-overlapping segments of the input time series.
    """
    y = np.asarray(y) 
    N = len(y)

    numBuffers = int(np.floor(N/bufferSize))

    # may need trimming
    y_buffer = y[:numBuffers*bufferSize]
    # then reshape
    y_buffer = y_buffer.reshape((numBuffers,bufferSize))

    return y_buffer

def binarize(y, binarizeHow='diff'):
    """
    Converts an input vector into a binarized version.

    Parameters:
    -----------
    y : array_like
        The input time series
    binarizeHow : str, optional
        Method to binarize the time series: 'diff', 'mean', 'median', 'iqr'.
    Returns:
    --------
    yBin : array_like
        The binarized time series
    """
    if binarizeHow == 'diff':
        # Binary signal: 1 for stepwise increases, 0 for stepwise decreases
        yBin = stepBinary(np.diff(y))
    
    elif binarizeHow == 'mean':
        # Binary signal: 1 for above mean, 0 for below mean
        yBin = stepBinary(y - np.mean(y))
    
    elif binarizeHow == 'median':
        # Binary signal: 1 for above median, 0 for below median
        yBin = stepBinary(y - np.median(y))
    
    elif binarizeHow == 'iqr':
        # Binary signal: 1 if inside interquartile range, 0 otherwise
        iqr = np.quantile(y,[.25,.75], method='hazen')
        iniqr = np.logical_and(y > iqr[0], y<iqr[1])
        yBin = np.zeros(len(y))
        yBin[iniqr] = 1
    else:
        raise ValueError(f"Unknown binary transformation setting '{binarizeHow}'")

    return yBin

def stepBinary(X):
    # Transform real values to 0 if <=0 and 1 if >0:
    Y = np.zeros(len(X))
    Y[X > 0] = 1

    return Y
