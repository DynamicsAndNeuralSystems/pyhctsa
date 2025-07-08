
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


def make_mat_buffer(X, n, p=0, opt=None):
    # helper function
    '''Mimic MATLAB routine to generate buffer array
    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html.
    Taken from: https://stackoverflow.com/questions/38453249/does-numpy-have-a-function-equivalent-to-matlabs-buffer 

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from X
    '''

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result

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

def xcorr(x, y, normed=True, maxlags=10):
    # taken from https://github.com/colizoli/xcorr_python 
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c


def RM_histogram2(*args):
    """
    rm_histogram2() computes the two dimensional frequency histogram of two row vectors x and y

    Takes in either two or three parameters:
        rm_histogram(x, y)
        rm_histogram(x, y, descriptor)

    x, y : the row vectors to be analyzed
    descriptor : the descriptor of the histogram where:

        descriptor = [lowerx, upperx, ncellx, lowery, uppery, ncelly]
            lower? : the lowerbound of the ? dimension of the histogram
            upper? : the upperbound of the dimension of the histogram
            ncell? : the number of cells of the ? dimension of the histogram

    :return: a tuple countaining a) the result (the 2d frequency histogram), b) descriptor (the descriptor used)

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen
    """

    nargin = len(args)

    if nargin < 1:
        print("Usage: result = rm_histogram2(X, Y)")
        print("       result = rm_histogram2(X,Y)")
        print("Where: descriptor = [lowerX, upperX, ncellX; lowerY, upperY, ncellY")

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 3:
        print("Error: too many arguments")
        return

    if nargin == 2:
        minx = np.amin(x)
        maxx = np.amax(x)
        deltax = (maxx - minx) / (lenx - 1)
        ncellx = np.ceil(lenx ** (1 / 3))

        miny = np.amin(y)
        maxy = np.amax(y)
        deltay = (maxy - miny) / (leny - 1)
        ncelly = ncellx
        descriptor = np.array(
            [[minx - deltax / 2, maxx + deltax / 2, ncellx], [miny - deltay / 2, maxy + deltay / 2, ncelly]])
    else:
        descriptor = args[2]

    lowerx = descriptor[0, 0]  # python indexes one less then matlab indexes, since starts at zero
    upperx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowery = descriptor[1, 0]
    uppery = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    # checking descriptor to make sure it is valid, otherwise print an error

    if ncellx < 1:
        print("Error: invalid number of cells in X dimension")

    if ncelly < 1:
        print("Error: invalid number of cells in Y dimension")

    if upperx <= lowerx:
        print("Error: invalid bounds in X dimension")

    if uppery <= lowery:
        print("Error: invalid bounds in Y dimension")

    result = np.zeros([int(ncellx), int(ncelly)],
                      dtype=int)  # should do the same thing as matlab: result(1:ncellx,1:ncelly) = 0;

    xx = np.around((x - lowerx) / (upperx - lowerx) * ncellx + 1 / 2)
    yy = np.around((y - lowery) / (uppery - lowery) * ncelly + 1 / 2)

    xx = xx.astype(int)  # cast all the values in xx and yy to ints for use in indexing, already rounded in previous step
    yy = yy.astype(int)

    for n in range(0, lenx):
        indexx = xx[n]
        indexy = yy[n]

        indexx -= 1  # adjust indices to start at zero, not one like in MATLAB
        indexy -= 1

        if indexx >= 0 and indexx <= ncellx - 1 and indexy >= 0 and indexy <= ncelly - 1:
            result[indexx, indexy] = result[indexx, indexy] + 1

    return result, descriptor
    