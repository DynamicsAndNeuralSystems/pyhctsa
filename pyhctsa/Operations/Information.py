import os
import numpy as np
from typing import Union, Any, Optional, Dict, List
from numpy.typing import ArrayLike
import logging

import jpype as jp
from scipy import stats

from ..Utilities.utils import signChange, RM_histogram2

def first_min(
    y: list,
    min_what: str = "mi-gaussian",
    extra_param: Union[int, float, None] = None,
    min_not_max: Union[bool, None] = True,
) -> int:
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    min_what : str, optional
        The type of correlation to minimize. Options are 'ac' for autocorrelation,
        or 'mi' for automutual information. By default, 'mi' specifies the
        'gaussian' method from the Information Dynamics Toolkit. Other options
        include 'mi-kernel', 'mi-kraskov1', 'mi-kraskov2' (from Information Dynamics Toolkit, JIDT),
        or 'mi-hist' (histogram-based method). Default is 'mi'.
    extra_param : any, optional
        An additional parameter required for the specified `min_what` method (e.g., for Kraskov).
    min_not_max : bool, optional
        If False, return the maximum instead of the minimum. Default is True.

    Returns
    -------
    int
        The time of the first minimum (or maximum if `min_not_max` is False).
    """
    from ..Operations.correlation import autocorr

    y = np.asarray(y)
    n = len(y)
    
    # Define the autocorrelation function
    if min_what in ["ac", "corr"]:
        corrfn = lambda x: autocorr(y, tau=x, method="Fourier")
    elif min_what == "mi-hist":
        # if extraParam is none, use default num of bins in MutualInformation (default : 10)
        if extra_param:
            # coerce into integer if not int
            extra_param = int(extra_param)
        corrfn = lambda x: _mi_bin(y[:-x], y[x:], "range", "range", extra_param or 10)
    elif min_what == "mi-kraskov2":
        # (using Information Dynamics Toolkit)
        # extraParam is the number of nearest neighbors
        corrfn = lambda x: automutual_info(y, x, "kraskov2", extra_param)
    elif min_what == "mi-kraskov1":
        # (using Information Dynamics Toolkit)
        corrfn = lambda x: automutual_info(y, x, "kraskov1", extra_param)
    elif min_what == "mi-kernel":
        corrfn = lambda x: automutual_info(y, x, "kernel", extra_param)
    elif min_what in ["mi", "mi-gaussian"]:
        corrfn = lambda x: automutual_info(y, x, "gaussian", extra_param)
    else:
        raise ValueError(f"Unknown correlation type specified: {min_what}")

    # search for a minimum (incrementally through time lags until a minimum is found)
    auto_corr = np.zeros(n - 1)  # pre-allocate maximum length autocorrelation vector
    if min_not_max:
        # FIRST LOCAL MINUMUM
        for i in range(1, n):
            auto_corr[i - 1] = corrfn(i)
            # Hit a NaN before got to a minimum -- there is no minimum
            if np.isnan(auto_corr[i - 1]):
                logging.warning(
                    f"No minimum in {min_what} [[time series too short to find it?]]"
                )
                return np.nan
            # we're at a local minimum
            if (i == 2) and (auto_corr[1] > auto_corr[0]):
                # already increases at lag of 2 from lag of 1: a minimum (since ac(0) is maximal)
                return 1
            elif (i > 2) and auto_corr[i - 3] > auto_corr[i - 2] < auto_corr[i - 1]:
                # minimum at previous i
                return i - 1  # I found the first minimum!
    else:
        # FIRST LOCAL MAXIMUM
        for i in range(1, n):
            auto_corr[i - 1] = corrfn(i)
            # Hit a NaN before got to a max -- there is no max
            if np.isnan(auto_corr[i - 1]):
                logging.warning(
                    f"No minimum in {min_what} [[time series too short to find it?]]"
                )
                return np.nan

            # we're at a local maximum
            if i > 2 and auto_corr[i - 3] < auto_corr[i - 2] > auto_corr[i - 1]:
                return i - 1
    return np.nan

def _mi_bin(v1 : ArrayLike, v2 : ArrayLike, r1 : Union[str, list] = 'range', 
            r2 : Union[str, list] = 'range', num_bins : int = 10) -> float:
    """
    Compute mutual information between two data vectors using bin counting.

    Parameters:
    -----------
        v1 (array-like): The first input vector
        v2 (array-like): The second input vector
        r1 (str or list): The bin-partitioning method for v1 ('range', 'quantile', or [min, max])
        r2 (str or list): The bin-partitioning method for v2 ('range', 'quantile', or [min, max])
        num_bins (int): The number of bins to partition each vector into (default : 10)

    Returns:
    --------
        float: The mutual information computed between v1 and v2
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()

    if len(v1) != len(v2):
        raise ValueError("Input vectors must be the same length")

    N = len(v1)

    # Create histograms
    edges_i = _give_me_edges(r1, v1, num_bins)
    edges_j = _give_me_edges(r2, v2, num_bins)

    ni, _ = np.histogram(v1, edges_i)
    nj, _ = np.histogram(v2, edges_j)

    # Create a joint histogram
    hist_xy, _, _ = np.histogram2d(v1, v2, [edges_i, edges_j])

    # Normalize counts to probabilities
    p_i = ni[:num_bins] / N
    p_j = nj[:num_bins] / N
    p_ij = hist_xy / N
    p_ixp_j = np.outer(p_i, p_j)

    # Calculate mutual information
    mask = (p_ixp_j > 0) & (p_ij > 0)
    if np.any(mask):
        mi = np.sum(p_ij[mask] * np.log(p_ij[mask] / p_ixp_j[mask]))
    else:
        print("The histograms aren't catching any points. Perhaps due to an inappropriate custom range for binning the data.")
        mi = np.nan

    return mi

def _give_me_edges(r, v, n_bins):
    EE = 1E-6 # this small addition gets lost in the last bin
    if n_bins <= 0:
        raise ValueError(f"nbins must be > 0, got {n_bins}")
    if r == 'range':
        return np.linspace(np.min(v), np.max(v) + EE, n_bins + 1)
    elif r == 'quantile': # bin edges based on quantiles
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[-1] += EE
        return edges
    elif isinstance(r, (list, np.ndarray)) and len(r) == 2: # a two-component vector
        return np.linspace(r[0], r[1] + EE, n_bins + 1)
    else:
        raise ValueError(f"Unknown partitioning method '{r}'")

def automutual_info_stats(
    y: ArrayLike,
    max_tau: Optional[int] = None,
    est_method: str = 'kernel',
    extra_param: Optional[Union[int, str]] = None
) -> Dict[str, float]:
    """
    Calculate statistics on the automutual information (AMI) function of a time series.

    This function computes various statistics on how the automutual information changes
    with increasing time delay, including basic statistics, periodicities, and crossings.

    Parameters
    ----------
    y : array-like
        Input time series (1D).
    max_tau : int, optional
        Maximum time delay to investigate. If None, uses N/4 where N is the length
        of the time series, but won't exceed N/2.
    est_method : {'gaussian', 'kernel', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information (passed to automutual_info).
        Default is 'kernel'.
    extra_param : int or str, optional
        Extra parameter for the estimator (passed to automutual_info).
        For Kraskov estimators, sets the number of nearest neighbors 'k'.

    Returns
    -------
    dict
        Dictionary containing AMI statistics.
    """
    from ..Operations.correlation import autocorr

    y = np.asarray(y)
    n = len(y)  # length of the time series

    # max_tau: the maximum time delay to investigate
    if max_tau is None:
        max_tau = np.ceil(n / 4)
    max_tau_0 = max_tau

    # Don't go above N/2
    max_tau = min(max_tau, np.ceil(n / 2))

    # Get the AMI data
    max_tau = int(max_tau)
    max_tau_0 = int(max_tau_0)
    time_delay = list(range(1, max_tau + 1))
    ami = automutual_info(
        y,
        time_delay=time_delay,
        est_method=est_method,
        extra_param=extra_param
    )
    ami = np.array(list(ami.values()))

    out = {}  # create dict for storing results

    # Output the raw values
    for i in range(1, max_tau_0 + 1):
        if i <= max_tau:
            out[f'ami{i}'] = ami[i - 1]
        else:
            out[f'ami{i}'] = np.nan

    # Basic statistics
    lami = len(ami)
    out['mami'] = np.mean(ami)
    out['stdami'] = np.std(ami, ddof=1)

    # First minimum of mutual information across range
    dami = np.diff(ami)
    extrema_i = np.where((dami[:-1] * dami[1:]) < 0)[0]
    out['pextrema'] = len(extrema_i) / (lami - 1)
    out['fmmi'] = min(extrema_i) + 1 if len(extrema_i) > 0 else lami

    # Look for periodicities in local maxima
    maxima_i = np.where((dami[:-1] > 0) & (dami[1:] < 0))[0] + 1
    dmaxima_i = np.diff(maxima_i)
    # Is there a big peak in dmaxima? (no need to normalize since a given method inputs its range; but do it anyway... ;-))
    out['pmaxima'] = len(dmaxima_i) / (lami // 2)
    if len(dmaxima_i) == 0:  # fewer than 2 local maxima
        out['modeperiodmax'] = np.nan
        out['pmodeperiodmax'] = np.nan
    else:
        out['modeperiodmax'] = stats.mode(dmaxima_i, keepdims=True).mode[0]
        out['pmodeperiodmax'] = np.sum(dmaxima_i == out['modeperiodmax']) / len(dmaxima_i)

    # Look for periodicities in local minima
    minima_i = np.where((dami[:-1] < 0) & (dami[1:] > 0))[0] + 1
    dminima_i = np.diff(minima_i)

    out['pminima'] = len(dminima_i) / (lami // 2)

    if len(dminima_i) == 0:  # fewer than 2 local minima
        out['modeperiodmin'] = np.nan
        out['pmodeperiodmin'] = np.nan
    else:
        out['modeperiodmin'] = stats.mode(dminima_i, keepdims=True).mode[0]
        out['pmodeperiodmin'] = np.sum(dminima_i == out['modeperiodmin']) / len(dminima_i)

    # Number of crossings at mean/median level, percentiles
    out['pcrossmean'] = np.mean(np.diff(np.sign(ami - np.mean(ami))) != 0)
    out['pcrossmedian'] = np.mean(np.diff(np.sign(ami - np.median(ami))) != 0)
    out['pcrossq10'] = np.mean(signChange(ami - np.percentile(ami, 10, method='hazen')))
    out['pcrossq90'] = np.mean(signChange(ami - np.percentile(ami, 90, method='hazen')))

    # ac1
    out['amiac1'] = autocorr(ami, 1, 'Fourier')[0]

    return out

def automutual_info(
    y: ArrayLike,
    time_delay: Union[int, str, List[int]] = 1,
    est_method: str = 'kernel',
    extra_param: Optional[Union[int, str]] = None
) -> Union[float, Dict[str, float]]:
    """
    Compute time-delayed automutual information of a time series.

    Calculates the mutual information between a time series and its time-delayed version
    using various estimation methods from the JIDT (Java Information Dynamics Toolkit).

    References
    ----------
    .. [1] Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). 
        Estimating mutual information. Physical Review E, 69(6), 066138.

    Parameters
    ----------
    y : array-like
        Input time series (1D).
    time_delay : int, str, or list of int, optional
        Time lag(s) for automutual information calculation. Can be:
            - int: a fixed lag
            - list of int: multiple lags
            - 'ac': first zero-crossing of autocorrelation
            - 'tau': same as 'ac'
        Default is 1.
    est_method : {'gaussian', 'kernel', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information:
            - 'gaussian': Assumes Gaussian variables
            - 'kernel': Kernel density estimation (default)
            - 'kraskov1': Kraskov estimator 1 (KSG1)
            - 'kraskov2': Kraskov estimator 2 (KSG2)
    extra_param : int or str, optional 
        Extra parameter for the estimator. For Kraskov estimators,
        this sets the number of nearest neighbors 'k' (default is 3).

    Returns
    -------
    float or dict
        If single time_delay:
            float: The automutual information value
        If multiple time_delay:
            dict: Keys are f"ami{delay}", values are corresponding AMI values
    """
    from ..Operations.distribution import first_crossing
    if isinstance(time_delay, str) and time_delay in ['ac', 'tau']:
        time_delay = first_crossing(y, corr_fun='ac', threshold=0, what_out='discrete')
        
    y = np.asarray(y).flatten()
    N = len(y)
    minSamples = 5 # minimum 5 samples to compute mutual information (could make higher?)

    # Loop over time delays if a vector
    if not isinstance(time_delay, list):
        time_delay = [time_delay]
    
    numTimeDelays = len(time_delay)
    amis = np.full(numTimeDelays, np.nan)

    if numTimeDelays > 1:
        time_delay = np.sort(time_delay)
    
    # initialise the MI calculator object if using non-Gaussian estimator
    if est_method != 'gaussian':
        # assumes the JVM has already been started up
        miCalc = _initialize_MI(est_method=est_method, extra_param=extra_param, add_noise=False) # NO ADDED NOISE!
    
    for k, delay in enumerate(time_delay):
        # check enough samples to compute automutual info
        if delay > N - minSamples:
            # time sereis too short - keep the remaining values as NaNs
            break
        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        if est_method == 'gaussian':
            r, _ = stats.pearsonr(y1, y2)
            amis[k] = -0.5*np.log(1 - r**2)
        else:
            # Reinitialize for Kraskov:
            miCalc.initialise(1, 1)
            # Set observations to time-delayed versions of the time series:
            y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
            y2_jp = jp.JArray(jp.JDouble)(y2)
            miCalc.setObservations(y1_jp, y2_jp)
            # compute
            amis[k] = miCalc.computeAverageLocalOfObservations()
        
    if np.isnan(amis).any():
        print(f"Warning: Time series (N={N}) is too short for automutual information calculations up to lags of {max(time_delay)}")
    if numTimeDelays == 1:
        # return a scalar if only one time delay
        return amis[0]
    else:
        # return a dict for multiple time delays
        return {f"ami{delay}": ami for delay, ami in zip(time_delay, amis)}

def mutual_info(
    y1: ArrayLike,
    y2: ArrayLike,
    est_method: str = 'kernel',
    extra_param: Optional[Union[int, str]] = None
) -> float:
    """
    Compute mutual information between two time series using JIDT estimators.

    This function calculates the mutual information between two time series using
    various estimators from the Java Information Dynamics Toolkit (JIDT).

    Note: This function requires the infodynamics.jar Java library and JPype.

    References
    ----------
    .. [1] Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). Estimating mutual information.
        Physical Review E, 69(6), 066138. https://doi.org/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    y1 : ArrayLike
        First input time series
    y2 : ArrayLike
        Second input time series
    est_method : str, optional
        Estimation method to use:
        - 'gaussian': Assumes Gaussian variables
        - 'kernel': Kernel density estimation (default)
        - 'kraskov1': Kraskov estimator 1 (KSG1)
        - 'kraskov2': Kraskov estimator 2 (KSG2)
    extra_param : Union[int, str], optional
        Extra parameter for the estimator:
        - For Kraskov estimators: number of nearest neighbors 'k' (default: 3)
        - For other methods: ignored

    Returns
    -------
    float
        Estimated mutual information between the input time series
    """
    # Initialize miCalc object (don't add noise!):
    miCalc = _initialize_MI(est_method=est_method, extra_param=extra_param, add_noise=False)
    # Set observations to two time series:
    y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
    y2_jp = jp.JArray(jp.JDouble)(y2) # convert observations to java double
    miCalc.setObservations(y1_jp, y2_jp)

    # Compute mutual information
    out = miCalc.computeAverageLocalOfObservations()

    return out

def _initialize_MI(
    est_method: str = "gaussian",
    extra_param: Optional[Union[int, str]] = None,
    add_noise: bool = False,
    verbose: bool = False
) -> Any:  # Returns a Java object, use Any since we can't type hint JPype objects
    """
    Initialize a mutual information calculator from JIDT (Java Information Dynamics Toolkit).
    
    Creates and configures a mutual information estimator by starting the JVM if needed,
    loading the appropriate JIDT class, and setting up the calculation parameters.

    Parameters
    ----------
    est_method : str, optional
        Estimation method to use:
        - 'gaussian': Assumes Gaussian variables (simplest)
        - 'kernel': Kernel density estimation
        - 'kraskov1': Kraskov estimator 1 (KSG1)
        - 'kraskov2': Kraskov estimator 2 (KSG2)
        Default is 'gaussian'.

    extra_param : Union[int, str], optional
        Configuration parameter for the estimator:
        - For Kraskov methods: number of nearest neighbors 'k'
        - For other methods: ignored
        Default is None (uses k=3 for Kraskov).

    add_noise : bool, optional
        Whether to add small random noise for Kraskov estimators:
        - True: Add noise (helpful for deterministic signals)
        - False: No noise (default)
    
    verbose : bool, optional
        Display JVM debug info. 

    Returns
    -------
    Any
        Initialized JIDT mutual information calculator object.
        Type varies by estimation method chosen.
    """

    if not jp.isJVMStarted():
        jarloc = (
            os.path.dirname(os.path.abspath(__file__)) + "/../Toolboxes/infodynamics-dist/infodynamics.jar"
        )
        # change to debug info
        if verbose:
            logging.debug(f"Starting JVM with java class {jarloc}.")
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)


    if est_method == 'gaussian':
        implementingClass = 'infodynamics.measures.continuous.gaussian'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateGaussian()
    elif est_method == 'kernel':
        implementingClass = 'infodynamics.measures.continuous.kernel'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKernel()
    elif est_method == 'kraskov1':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov1()
    elif est_method == 'kraskov2':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov2()
    else:
        raise ValueError(f"Unknown mutual information estimation method '{est_method}'")

    # Add neighest neighbor option for KSG estimator
    if est_method in ['kraskov1', 'kraskov2']:
        if extra_param != None:
            if isinstance(extra_param, int):
                logging.warning("Number of nearest neighbors needs to be a string. Setting this for you...")
                extra_param = str(extra_param)
            miCalc.setProperty('k', extra_param) # 4th input specifies number of nearest neighbors for KSG estimator
        else:
            miCalc.setProperty('k', '3') # use 3 nearest neighbors for KSG estimator as default
        
    # Make deterministic if kraskov1 or 2 (which adds a small amount of noise to the signal by default)
    if (est_method in ['kraskov1', 'kraskov2']) and (add_noise == False):
        miCalc.setProperty('NOISE_LEVEL_TO_ADD','0')
    
    # Specify a univariate calculation
    miCalc.initialise(1,1)

    return miCalc

def rm_automutual_information(y : ArrayLike, tau : int = 1) -> float:
    """
    Estimates the mutual information of two stationary signals with 
    independent pairs of samples using various approaches.

    Based on a wrapper initially developed by Ben D. Fulcher in MATLAB,
    which is based on rm_information.py initially developed by Rudy Moddemeijer in MATLAB,
    and translated to to python by Tucker Cullen.

    Parameters
    ------
    y : array-like
        The input time series.

    Returns
    -------
    float:
        Estimate of the auto-mutual information
    """
    if tau >= len(y):
        return np.nan
    elif tau == 0:
        # handle the case when tau = 0 (no lag)
        y1 = y2 = y
    else:
        y1 = y[:-tau]
        y2 = y[tau:]

    out = _rm_info(y1, y2)[0]

    return out

def _rm_info(*args):
    """
    Estimates the mutual information of two stationary signals with independent pairs of samples using various approaches.

    Parameters
    ----------
    x : array-like
        First input time series (row vector).
    y : array-like
        Second input time series (row vector).
    descriptor : array-like, optional
        Histogram descriptor array of shape (2, 3):
            [[LOWERBOUNDX, UPPERBOUNDX, NCELLX],
             [LOWERBOUNDY, UPPERBOUNDY, NCELLY]]
        If not provided, will be computed automatically.
    approach : {'unbiased', 'mmse', 'biased'}, optional
        Method for estimating mutual information:
            - 'unbiased': The unbiased estimate (default)
            - 'mmse': The minimum mean square error estimate
            - 'biased': The biased estimate
    base : float, optional
        The base of the logarithm (default: e).

    Returns
    -------
    estimate : float
        The mutual information estimate.
    nbias : float
        The N-bias of the estimate.
    sigma : float
        The standard error of the estimate.
    descriptor : np.ndarray
        The descriptor of the histogram used, see also RM_histogram2.

    Notes
    -----
    - See also: http://www.cs.rug.nl/~rudy/matlab/
    - Original MATLAB function by R. Moddemeijer, minor modifications by Ben Fulcher.
    - Python translation by Tucker Cullen.
    """

    nargin = len(args)

    if nargin < 1:
        print("Takes in 2-5 parameters: ")
        print("rm_information(x, y)")
        print("rm_information(x, y, descriptor)")
        print("rm_information(x, y, descriptor, approach)")
        print("rm_information(x, y, descriptor, approach, base)")
        print()

        print("Returns a tuple containing: ")
        print("estimate, nbias, sigma, descriptor")
        return

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

    if nargin > 5:
        print("Error: too many arguments")
        return

    if nargin < 2:
        print("Error: not enough arguments")
        return

    # setting up variables depending on amount of inputs

    if nargin == 2:
        hist = RM_histogram2(x, y)  # call outside function from rm_histogram2.py
        h = hist[0]
        descriptor = hist[1]

    if nargin >= 3:
        hist = RM_histogram2(x, y, args[2])  # call outside function from rm_histogram2.py, args[2] represents the given descriptor
        h = hist[0]
        descriptor = hist[1]

    if nargin < 4:
        approach = 'unbiased'
    else:
        approach = args[3]

    if nargin < 5:
        base = np.e  # as in e = 2.71828
    else:
        base = args[4]

    lowerboundx = descriptor[0, 0]  #not sure why most of these were included in the matlab script, most of them go unused
    upperboundx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowerboundy = descriptor[1, 0]
    upperboundy = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    estimate = 0
    sigma = 0
    count = 0

    # determine row and column sums

    hy = np.sum(h, 0)
    hx = np.sum(h, 1)

    ncellx = ncellx.astype(int)
    ncelly = ncelly.astype(int)

    for nx in range(0, ncellx):
        for ny in range(0, ncelly):
            if h[nx, ny] != 0:
                logf = np.log(h[nx, ny] / hx[nx] / hy[ny])
            else:
                logf = 0

            count = count + h[nx, ny]
            estimate = estimate + h[nx, ny] * logf
            sigma = sigma + h[nx, ny] * (logf ** 2)

    # biased estimate

    estimate = estimate / count
    sigma = np.sqrt((sigma / count - estimate ** 2) / (count - 1))
    estimate = estimate + np.log(count)
    nbias = (ncellx - 1) * (ncelly - 1) / (2 * count)

    # conversion to unbiased estimate

    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0

        # conversion to minimum mse estimate

    if approach[0] == 'm':
        estimate = estimate - nbias
        nbias = 0
        lamda = (estimate ** 2) / ((estimate ** 2) + (sigma ** 2))
        nbias = (1 - lamda) * estimate
        estimate = lamda * estimate
        sigma = lamda * sigma

        # base transformations

    estimate = estimate / np.log(base)
    nbias = nbias / np.log(base)
    sigma = sigma / np.log(base)

    return estimate, nbias, sigma, descriptor
