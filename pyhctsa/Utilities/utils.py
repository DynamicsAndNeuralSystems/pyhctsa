
import numpy as np
import csv
from numpy.typing import ArrayLike
import os
from functools import wraps

def get_dataset(which : str = "e1000"):
    """
    Load data for testing and validation.
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
