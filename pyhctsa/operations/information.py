import logging
import os
from typing import Any, Dict, List, Optional, Union, Callable

import jpype as jp
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ..utils import sign_change

def _get_corr_fn(y: np.ndarray, min_what: str, extra_param: Union[int, float, None]) -> Callable:
    """Helper to return the correct correlation function based on method type."""
    from ..operations.correlation import autocorr, automutual_info

    if min_what in ['ac', 'corr']:
        return lambda x: autocorr(y, tau=x, method='Fourier')
    elif min_what == 'mi-hist':
        num_bins = int(extra_param) if extra_param else 10
        return lambda x: _mi_bin(y[:-x], y[x:], 'range', 'range', num_bins)
    elif min_what == 'mi-kraskov2':
        return lambda x: automutual_info(y, x, 'kraskov2', extra_param)
    elif min_what == 'mi-kraskov1':
        return lambda x: automutual_info(y, x, 'kraskov1', extra_param)
    elif min_what == 'mi-kernel':
        return lambda x: automutual_info(y, x, 'kernel', extra_param)
    elif min_what in ['mi', 'mi-gaussian']:
        return lambda x: automutual_info(y, x, 'gaussian', extra_param)
    else:
        raise ValueError(f"Unknown correlation type specified: {min_what}")

def first_min(
    y: list,
    min_what: str = 'mi-gaussian',
    extra_param: Union[int, float, None] = None,
) -> int:
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.

    min_what : str, optional
        Correlation measure to minimize.

        Autocorrelation:

        - ``'ac'``: Autocorrelation.

        Automutual information (AMI):

        - ``'mi'``: AMI using the JIDT Gaussian estimator (default for AMI).
        - ``'mi-kernel'``: AMI using the JIDT kernel estimator.
        - ``'mi-kraskov1'``: AMI using the JIDT Kraskov estimator (variant 1).
        - ``'mi-kraskov2'``: AMI using the JIDT Kraskov estimator (variant 2).
        - ``'mi-hist'``: AMI using a histogram-based estimator.

        Default is ``'mi-gaussian'``.

    extra_param : any, optional
        Additional parameter required by the chosen ``min_what`` method
        (e.g., a k-nearest-neighbours parameter for Kraskov-based AMI).

    Returns
    -------
    int
        The time of the first minimum.
    """
    y = np.asarray(y)
    n = len(y)
    corrfn = _get_corr_fn(y, min_what, extra_param)
    
    auto_corr = np.zeros(n - 1)
    for i in range(1, n):
        auto_corr[i - 1] = corrfn(i)
        
        if np.isnan(auto_corr[i - 1]):
            logging.warning(f"No minimum in {min_what}: encountered NaN.")
            return np.nan
        
        # Check for minimum
        if (i == 2) and (auto_corr[1] > auto_corr[0]):
            return 1
        elif (i > 2) and auto_corr[i - 3] > auto_corr[i - 2] < auto_corr[i - 1]:
            return i - 1
            
    return np.nan

def first_max(
    y: list,
    max_what: str = 'mi-gaussian',
    extra_param: Union[int, float, None] = None,
) -> int:
    """
    Time of first maximum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    max_what : str, optional
        Correlation measure to maximize.

        Autocorrelation:

        - ``'ac'``: Autocorrelation.

        Automutual information (AMI):

        - ``'mi'``: AMI using the JIDT Gaussian estimator (default for AMI).
        - ``'mi-kernel'``: AMI using the JIDT kernel estimator.
        - ``'mi-kraskov1'``: AMI using the JIDT Kraskov estimator (variant 1).
        - ``'mi-kraskov2'``: AMI using the JIDT Kraskov estimator (variant 2).
        - ``'mi-hist'``: AMI using a histogram-based estimator.

        Default is ``'mi'``.

    extra_param : any, optional
        An additional parameter required for the specified `max_what` method (e.g., for Kraskov).

    Returns
    -------
    int
        The time of the first maximum.
    """
    y = np.asarray(y)
    n = len(y)
    corrfn = _get_corr_fn(y, max_what, extra_param)
    
    auto_corr = np.zeros(n - 1)
    for i in range(1, n):
        auto_corr[i - 1] = corrfn(i)
        
        if np.isnan(auto_corr[i - 1]):
            logging.warning(f"No maximum in {max_what}: encountered NaN.")
            return np.nan

        # Check for maximum
        if i > 2 and auto_corr[i - 3] < auto_corr[i - 2] > auto_corr[i - 1]:
            return i - 1
            
    return np.nan

def _mi_bin(v1: ArrayLike, v2: ArrayLike, r1: Union[str, list] = 'range',
            r2: Union[str, list] = 'range', num_bins: int = 10) -> float:
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
        Input time series.
    max_tau : int, optional
        Maximum time delay to investigate. If None, uses N/4 where N is the length
        of the time series, but won't exceed N/2. Default is `None`.
    est_method : {'gaussian', 'kernel', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information (passed to automutual_info).
        Default is ``'kernel'``.
    extra_param : int or str, optional
        Extra parameter for the estimator (passed to automutual_info).
        For Kraskov estimators, sets the number of nearest neighbors 'k'. Default is `None`.

    Returns
    -------
    dict
        Dictionary containing AMI statistics.
    """
    from ..operations.correlation import autocorr

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
    # Is there a big peak in dmaxima? (no need to normalize since a given method 
    # inputs its range; but do it anyway... ;-))
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
    out['pcrossq10'] = np.mean(sign_change(ami - np.percentile(ami, 10, method='hazen')))
    out['pcrossq90'] = np.mean(sign_change(ami - np.percentile(ami, 90, method='hazen')))

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
        Input time series.
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

        Default is `kernel`.

    extra_param : int or str, optional
        Extra parameter for the estimator. For Kraskov estimators,
        this sets the number of nearest neighbors 'k'.
        Default is 3.

    Returns
    -------
    float or dict
        If single time_delay:
            float: The automutual information value
        If multiple time_delay:
            dict: Keys are f"ami{delay}", values are corresponding AMI values
    """
    from ..operations.distribution import first_crossing

    if isinstance(time_delay, str) and time_delay in ['ac', 'tau']:
        time_delay = first_crossing(y, corr_fun='ac', threshold=0, what_out='discrete')

    y = np.asarray(y).flatten()
    n = len(y)
    min_samples = 5  # minimum 5 samples to compute mutual information (could make higher?)

    # Loop over time delays if a vector
    if not isinstance(time_delay, list):
        time_delay = [time_delay]

    num_time_delays = len(time_delay)
    amis = np.full(num_time_delays, np.nan)

    if num_time_delays > 1:
        time_delay = np.sort(time_delay)

    # initialise the MI calculator object if using non-Gaussian estimator
    if est_method != 'gaussian':
        # assumes the JVM has already been started up
        mi_calc = _initialize_MI(est_method=est_method, extra_param=extra_param, add_noise=False)  # NO ADDED NOISE!

    for k, delay in enumerate(time_delay):
        # check enough samples to compute automutual info
        if delay > n - min_samples:
            # time sereis too short - keep the remaining values as NaNs
            break

        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        if est_method == 'gaussian':
            r, _ = stats.pearsonr(y1, y2)
            amis[k] = -0.5 * np.log(1 - r**2)
        else:
            # Reinitialize for Kraskov:
            mi_calc.initialise(1, 1)
            # Set observations to time-delayed versions of the time series:
            y1_jp = jp.JArray(jp.JDouble)(y1)  # convert observations to java double
            y2_jp = jp.JArray(jp.JDouble)(y2)
            mi_calc.setObservations(y1_jp, y2_jp)
            # compute
            amis[k] = mi_calc.computeAverageLocalOfObservations()

    if np.isnan(amis).any():
        print(
            f"Warning: Time series (n={n}) is too short for automutual information calculations "
            f"up to lags of {max(time_delay)}"
        )

    if num_time_delays == 1:
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
        First input time series.
    y2 : ArrayLike
        Second input time series.
    est_method : str, optional
        Estimation method to use:
        
        - 'gaussian': Assumes Gaussian variables
        - 'kernel': Kernel density estimation (default)
        - 'kraskov1': Kraskov estimator 1 (KSG1)
        - 'kraskov2': Kraskov estimator 2 (KSG2)

        Default is `kernel`.

    extra_param : Union[int, str], optional
        Extra parameter for the estimator:

        - For Kraskov estimators: number of nearest neighbors 'k' (default: 3)
        - For other methods: ignored

        Default is `None`.

    Returns
    -------
    float
        Estimated mutual information between the input time series
    """
    # Initialize miCalc object (don't add noise!):
    mi_calc = _initialize_MI(est_method=est_method, extra_param=extra_param, add_noise=False)
    # Set observations to two time series:
    y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
    y2_jp = jp.JArray(jp.JDouble)(y2) # convert observations to java double
    mi_calc.setObservations(y1_jp, y2_jp)

    # Compute mutual information
    out = mi_calc.computeAverageLocalOfObservations()

    return out

def _initialize_MI(
    est_method: str = 'gaussian',
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
        
        Default is ``'gaussian'``.

    extra_param : Union[int, str], optional
        Configuration parameter for the estimator:

        - For Kraskov methods: number of nearest neighbors 'k'
        - For other methods: ignored

        Default is `None` (uses k=3 for Kraskov).

    add_noise : bool, optional
        Whether to add small random noise for Kraskov estimators:

        - True: Add noise (helpful for deterministic signals)
        - False: No noise.

        Default is `False`.
    
    verbose : bool, optional
        Display JVM debug info. Default is `False`.

    Returns
    -------
    Any
        Initialized JIDT mutual information calculator object.
        Type varies by estimation method chosen.
    """

    if not jp.isJVMStarted():
        jarloc = (
            os.path.dirname(os.path.abspath(__file__)) + "/../toolboxes/infodynamics-dist/infodynamics.jar"
        )
        # change to debug info
        if verbose:
            logging.debug(f"Starting JVM with java class {jarloc}.")
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc, interrupt=False)


    if est_method == 'gaussian':
        implementing_class = 'infodynamics.measures.continuous.gaussian'
        mi_calc = jp.JPackage(implementing_class).MutualInfoCalculatorMultiVariateGaussian()
    elif est_method == 'kernel':
        implementing_class = 'infodynamics.measures.continuous.kernel'
        mi_calc = jp.JPackage(implementing_class).MutualInfoCalculatorMultiVariateKernel()
    elif est_method == 'kraskov1':
        implementing_class = 'infodynamics.measures.continuous.kraskov'
        mi_calc = jp.JPackage(implementing_class).MutualInfoCalculatorMultiVariateKraskov1()
    elif est_method == 'kraskov2':
        implementing_class = 'infodynamics.measures.continuous.kraskov'
        mi_calc = jp.JPackage(implementing_class).MutualInfoCalculatorMultiVariateKraskov2()
    else:
        raise ValueError(f"Unknown mutual information estimation method '{est_method}'")

    # Add neighest neighbor option for KSG estimator
    if est_method in ['kraskov1', 'kraskov2']:
        if extra_param is not None:
            if isinstance(extra_param, int):
                logging.warning("Number of nearest neighbors needs to be a string. Setting this for you...")
                extra_param = str(extra_param)
            mi_calc.setProperty('k', extra_param) # 4th input specifies number of nearest neighbors for KSG estimator
        else:
            mi_calc.setProperty('k', '3') # use 3 nearest neighbors for KSG estimator as default
        
    # Make deterministic if kraskov1 or 2 (which adds a small amount of noise to the signal by default)
    if (est_method in ['kraskov1', 'kraskov2']) and (add_noise is False):
        mi_calc.setProperty('NOISE_LEVEL_TO_ADD','0')
    
    # Specify a univariate calculation
    mi_calc.initialise(1,1)

    return mi_calc

def rm_automutual_information(y: ArrayLike, tau: int = 1) -> float:
    """
    Estimates the mutual information of two stationary signals with 
    independent pairs of samples using various approaches.

    Based on a wrapper initially developed by Ben D. Fulcher in MATLAB,
    which is based on rm_information.py initially developed by Rudy Moddemeijer in MATLAB,
    and translated to to python by Tucker Cullen.

    Parameters
    ----------
    y : array-like
        The input time series.
    tau: int
        Time lag for automutual information calculation. Default is 1.

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
    Estimates the mutual information of two stationary signals with independent 
    pairs of samples using various approaches.

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
        The descriptor of the histogram used, see also _rm_histogram_2.

    Notes
    -----
    - See also: http://www.cs.rug.nl/~rudy/matlab/
    - Original MATLAB function by R. Moddemeijer, minor modifications by Ben Fulcher.
    - Python translation by Tucker Cullen.
    """
    n_args = len(args)
    if n_args < 1:
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

    x_shape = x.shape
    y_shape = y.shape

    len_x = x_shape[0]  # how many elements are in the row vector
    len_y = y_shape[0]

    if len(x_shape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(y_shape) != 1:
        print("Error: invalid dimension of y")
        return

    if len_x != len_y:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if n_args > 5:
        print("Error: too many arguments")
        return

    if n_args < 2:
        print("Error: not enough arguments")
        return

    # setting up variables depending on amount of inputs
    if n_args == 2:
        hist = _rm_histogram_2(x, y)  # call outside function from rm_histogram2.py
        h = hist[0]
        descriptor = hist[1]

    if n_args >= 3:
        hist = _rm_histogram_2(
            x, y, args[2]
        )  # call outside function from rm_histogram2.py, args[2] represents the given descriptor
        h = hist[0]
        descriptor = hist[1]

    if n_args < 4:
        approach = 'unbiased'
    else:
        approach = args[3]

    if n_args < 5:
        base = np.e  # as in e = 2.71828
    else:
        base = args[4]

    # not sure why most of these were included in the matlab script, most of them go unused
    lower_bound_x = descriptor[0, 0]
    upper_bound_x = descriptor[0, 1]
    n_cell_x = descriptor[0, 2]
    lower_bound_y = descriptor[1, 0]
    upper_bound_y = descriptor[1, 1]
    n_cell_y = descriptor[1, 2]

    estimate = 0
    sigma = 0
    count = 0

    # determine row and column sums
    h_y = np.sum(h, 0)
    h_x = np.sum(h, 1)

    n_cell_x = n_cell_x.astype(int)
    n_cell_y = n_cell_y.astype(int)

    for n_x in range(0, n_cell_x):
        for n_y in range(0, n_cell_y):
            if h[n_x, n_y] != 0:
                log_f = np.log(h[n_x, n_y] / h_x[n_x] / h_y[n_y])
            else:
                log_f = 0

            count = count + h[n_x, n_y]
            estimate = estimate + h[n_x, n_y] * log_f
            sigma = sigma + h[n_x, n_y] * (log_f**2)

    # biased estimate
    estimate = estimate / count
    sigma = np.sqrt((sigma / count - estimate**2) / (count - 1))
    estimate = estimate + np.log(count)
    nbias = (n_cell_x - 1) * (n_cell_y - 1) / (2 * count)

    # conversion to unbiased estimate
    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0

    # conversion to minimum mse estimate
    if approach[0] == 'm':
        estimate = estimate - nbias
        nbias = 0
        lamda = (estimate**2) / ((estimate**2) + (sigma**2))
        nbias = (1 - lamda) * estimate
        estimate = lamda * estimate
        sigma = lamda * sigma

    # base transformations
    estimate = estimate / np.log(base)
    nbias = nbias / np.log(base)
    sigma = sigma / np.log(base)

    return estimate, nbias, sigma, descriptor


def _rm_histogram_2(*args):
    """
    Compute a two-dimensional frequency histogram from two row vectors.
    
    Computes the two-dimensional frequency histogram of two row vectors x and y,
    optionally using a provided histogram descriptor to specify bin boundaries and counts.

    Parameters
    ----------
    *args : variable length argument list
        Can be called with 2 or 3 arguments:
        
        - _rm_histogram_2(x, y): Auto-computes histogram descriptor
        - _rm_histogram_2(x, y, descriptor): Uses provided descriptor
        
        Where:
        
        x : array-like
            First row vector to be analyzed.
        y : array-like
            Second row vector to be analyzed. Must be equal length to x.
        descriptor : np.ndarray, optional
            Histogram descriptor of shape (2, 3) specifying bin parameters:
            [[lower_x, upper_x, ncell_x],
             [lower_y, upper_y, ncell_y]]
            where:
            - lower_?: lower bound of the ? dimension
            - upper_?: upper bound of the ? dimension
            - ncell_?: number of bins in the ? dimension

    Returns
    -------
    tuple
        A tuple containing:
        - result : np.ndarray
            2D frequency histogram of shape (ncell_x, ncell_y).
        - descriptor : np.ndarray
            The histogram descriptor used (auto-computed if not provided).

    Raises
    ------
    ValueError
        If x and y have different lengths.
        If descriptor dimensions are invalid.

    Notes
    -----
    This is a helper function for mutual information calculations.
    
    Original MATLAB function and logic by Rudy Moddemeijer.
    Translated to Python by Tucker Cullen.

    See Also
    --------
    _rm_info : Mutual information estimation using histograms.

    References
    ----------
    .. [1] http://www.cs.rug.nl/~rudy/matlab/
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
