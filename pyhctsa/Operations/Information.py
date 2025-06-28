from typing import Union, Any, Optional, Dict
import numpy as np
import jpype as jp
import os
from numpy.typing import ArrayLike
from scipy import stats
from loguru import logger
from ..Utilities.utils import signChange
from ..Operations.Correlation import FirstCrossing, AutoCorr

def AutoMutualInfoStats(
    y: ArrayLike,
    maxTau: Optional[int] = None,
    estMethod: str = 'kernel',
    extraParam: Optional[Union[int, str]] = None) -> Dict[str, float]:
    """
    Calculate statistics on the automutual information (AMI) function of a time series.

    This function computes various statistics on how the automutual information changes
    with increasing time delay, including basic statistics, periodicities, and crossings.

    Parameters
    ----------
    y : array-like
        Input time series (1D).
    maxTau : int, optional
        Maximum time delay to investigate. If None, uses N/4 where N is the length
        of the time series, but won't exceed N/2.
    estMethod : {'gaussian', 'kernel', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information (passed to AutoMutualInfo).
        Default is 'kernel'.
    extraParam : int or str, optional
        Extra parameter for the estimator (passed to AutoMutualInfo).
        For Kraskov estimators, sets the number of nearest neighbors 'k'.

    Returns
    -------
    dict
        Dictionary containing AMI statistics.
    """
    y = np.asarray(y)
    N = len(y) # length of the time series
    
    # maxTau: the maximum time delay to investigate
    if maxTau is None:
        maxTau = np.ceil(N/4)
    maxTau0 = maxTau

    # Don't go above N/2
    maxTau = min(maxTau, np.ceil(N/2))

    # Get the AMI data
    maxTau = int(maxTau)
    maxTau0 = int(maxTau0)
    timeDelay = list(range(1, maxTau+1))
    ami = AutoMutualInfo(y, timeDelay=timeDelay, estMethod=estMethod, extraParam=extraParam)
    ami = np.array(list(ami.values()))

    out = {} # create dict for storing results
    # Output the raw values
    for i in range(1, maxTau0+1):
        if i <= maxTau:
            out[f'ami{i}'] = ami[i-1]
        else:
            out[f'ami{i}'] = np.nan

    # Basic statistics
    lami = len(ami)
    out['mami'] = np.mean(ami)
    out['stdami'] = np.std(ami, ddof=1)

    # First minimum of mutual information across range
    dami = np.diff(ami)
    extremai = np.where((dami[:-1] * dami[1:]) < 0)[0]
    out['pextrema'] = len(extremai) / (lami - 1)
    out['fmmi'] = min(extremai) + 1 if len(extremai) > 0 else lami

    # Look for periodicities in local maxima
    maximai = np.where((dami[:-1] > 0) & (dami[1:] < 0))[0] + 1
    dmaximai = np.diff(maximai)
    # Is there a big peak in dmaxima? (no need to normalize since a given method inputs its range; but do it anyway... ;-))
    out['pmaxima'] = len(dmaximai) / (lami // 2)
    if len(dmaximai) == 0:  # fewer than 2 local maxima
        out['modeperiodmax'] = np.nan
        out['pmodeperiodmax'] = np.nan
    else:
        out['modeperiodmax'] = stats.mode(dmaximai, keepdims=True).mode[0]
        out['pmodeperiodmax'] = np.sum(dmaximai == out['modeperiodmax']) / len(dmaximai)

    # Look for periodicities in local minima
    minimai = np.where((dami[:-1] < 0) & (dami[1:] > 0))[0] + 1
    dminimai = np.diff(minimai)

    out['pminima'] = len(dminimai) / (lami // 2)

    if len(dminimai) == 0:  # fewer than 2 local minima
        out['modeperiodmin'] = np.nan
        out['pmodeperiodmin'] = np.nan
    else:
        out['modeperiodmin'] = stats.mode(dminimai, keepdims=True).mode[0]
        out['pmodeperiodmin'] = np.sum(dminimai == out['modeperiodmin']) / len(dminimai)
    
    # Number of crossings at mean/median level, percentiles
    out['pcrossmean'] = np.mean(np.diff(np.sign(ami - np.mean(ami))) != 0)
    out['pcrossmedian'] = np.mean(np.diff(np.sign(ami - np.median(ami))) != 0)
    out['pcrossq10'] = np.mean(signChange(ami - np.percentile(ami, 10, method='hazen')))
    out['pcrossq90'] = np.mean(signChange(ami - np.percentile(ami, 90, method='hazen')))

    # ac1 
    out['amiac1'] = AutoCorr(ami, 1, 'Fourier')[0]

    return out 

def AutoMutualInfo(
    y: ArrayLike,
    timeDelay: Union[int, str, list[int]] = 1,
    estMethod: str = 'kernel',
    extraParam: Optional[Union[int, str]] = None
) -> Union[float, Dict[str, float]]:
    """
    Compute time-delayed automutual information of a time series.

    Calculates the mutual information between a time series and its time-delayed version
    using various estimation methods from the JIDT (Java Information Dynamics Toolkit).

    Parameters
    ----------
    y : array-like
        Input time series (1D).
    timeDelay : int, str, or list of int, optional
        Time lag(s) for automutual information calculation. Can be:
            - int: a fixed lag
            - list of int: multiple lags
            - 'ac': first zero-crossing of autocorrelation
            - 'tau': same as 'ac'
        Default is 1.
    estMethod : {'gaussian', 'kernel', 'kraskov1', 'kraskov2'}, optional
        Method for estimating mutual information:
            - 'gaussian': Assumes Gaussian variables
            - 'kernel': Kernel density estimation (default)
            - 'kraskov1': Kraskov estimator 1 (KSG1)
            - 'kraskov2': Kraskov estimator 2 (KSG2)
    extraParam : int or str, optional 
        Extra parameter for the estimator. For Kraskov estimators,
        this sets the number of nearest neighbors 'k' (default is 3).

    Returns
    -------
    float or dict
        If single timeDelay:
            float: The automutual information value
        If multiple timeDelays:
            dict: Keys are f"ami{delay}", values are corresponding AMI values

    References
    ----------
    Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). 
    Estimating mutual information. Physical Review E, 69(6), 066138.
    """

    if isinstance(timeDelay, str) and timeDelay in ['ac', 'tau']:
        timeDelay = FirstCrossing(y, corr_fun='ac', threshold=0, what_out='discrete')
        
    y = np.asarray(y).flatten()
    N = len(y)
    minSamples = 5 # minimum 5 samples to compute mutual information (could make higher?)

    # Loop over time delays if a vector
    if not isinstance(timeDelay, list):
        timeDelay = [timeDelay]
    
    numTimeDelays = len(timeDelay)
    amis = np.full(numTimeDelays, np.nan)

    if numTimeDelays > 1:
        timeDelay = np.sort(timeDelay)
    
    # initialise the MI calculator object if using non-Gaussian estimator
    if estMethod != 'gaussian':
        # assumes the JVM has already been started up
        miCalc = _initialize_MI(estMethod, extraParam=extraParam, addNoise=False) # NO ADDED NOISE!
    
    for k, delay in enumerate(timeDelay):
        # check enough samples to compute automutual info
        if delay > N - minSamples:
            # time sereis too short - keep the remaining values as NaNs
            break
        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        if estMethod == 'gaussian':
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
        print(f"Warning: Time series (N={N}) is too short for automutual information calculations up to lags of {max(timeDelay)}")
    if numTimeDelays == 1:
        # return a scalar if only one time delay
        return amis[0]
    else:
        # return a dict for multiple time delays
        return {f"ami{delay}": ami for delay, ami in zip(timeDelay, amis)}

def MutualInfo(
    y1: ArrayLike,
    y2: ArrayLike,
    estMethod: str = 'kernel',
    extraParam: Optional[Union[int, str]] = None
) -> float:
    """
    Compute the mutual information between two time series using the Java Information Dynamics Toolkit (JIDT).

    Parameters
    ----------
    y1 : array-like
        First input time series (1D array).
    y2 : array-like
        Second input time series (1D array).
    estMethod : str, optional
        The estimation method to use. Options are:
            - 'gaussian'  : Gaussian estimator
            - 'kernel'    : Kernel estimator
            - 'kraskov1'  : Kraskov estimator 1 (KSG1)
            - 'kraskov2'  : Kraskov estimator 2 (KSG2)
        Default is 'kernel'.
    extraParam : any, optional
        Extra parameter for the estimator. For Kraskov estimators, this sets the number
        of nearest neighbors 'k' (default is 3). If provided as an integer, it will be
        converted to a string.

    Returns
    -------
    float
        The estimated mutual information between the two input time series.

    References
    ----------
    Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). Estimating mutual information.
    Physical Review E, 69(6), 066138. https://doi.org/10.1103/PhysRevE.69.066138

    Notes
    -----
    This function requires the infodynamics.jar Java library and JPype.
    """
    # Initialize miCalc object (don't add noise!):
    miCalc = _initialize_MI(estMethod=estMethod, extraParam=extraParam, addNoise=False)
    # Set observations to two time series:
    y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
    y2_jp = jp.JArray(jp.JDouble)(y2) # convert observations to java double
    miCalc.setObservations(y1_jp, y2_jp)

    # Compute mutual information
    out = miCalc.computeAverageLocalOfObservations()

    return out

def _initialize_MI(
    estMethod: str = "gaussian",
    extraParam: Optional[Union[int, str]] = None,
    addNoise: bool = False
) -> Any:  # Returns a Java object, use Any since we can't type hint JPype objects
    """
    Helper function to initialize a mutual information calculator object from the Java
    Information Dynamics Toolkit (JIDT).

    This function starts the Java Virtual Machine (JVM) if it is not already running,
    loads the appropriate mutual information estimator class based on the specified
    estimation method, and configures its parameters.

    Parameters
    ----------
    estMethod : str
        The estimation method to use. Must be one of
            - 'gaussian'  : Gaussian estimator
            - 'kernel'    : Kernel estimator
            - 'kraskov1'  : Kraskov estimator 1 (KSG1)
            - 'kraskov2'  : Kraskov estimator 2 (KSG2)
    extraParam : any, optional
        Extra parameter for the estimator. For Kraskov estimators, this sets the number
        of nearest neighbors 'k' (default is 3). If provided as an integer, it will be
        converted to a string.
    addNoise : bool, optional
        If False (default), disables the small random noise that is added by default
        in Kraskov estimators for determinism.

    Returns
    -------
    miCalc : Java object
        An initialized mutual information calculator object ready for use.
    """
    # Check to see whether a jpype JVM has been started.
    if not jp.isJVMStarted():
        jarloc = (
            os.path.dirname(os.path.abspath(__file__)) + "/../Toolboxes/infodynamics-dist/infodynamics.jar"
        )
        # change to debug info
        logger.debug(f"Starting JVM with java class {jarloc}.")
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)


    if estMethod == 'gaussian':
        implementingClass = 'infodynamics.measures.continuous.gaussian'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateGaussian()
    elif estMethod == 'kernel':
        implementingClass = 'infodynamics.measures.continuous.kernel'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKernel()
    elif estMethod == 'kraskov1':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov1()
    elif estMethod == 'kraskov2':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov2()
    else:
        raise ValueError(f"Unknown mutual information estimation method '{estMethod}'")

    # Add neighest neighbor option for KSG estimator
    if estMethod in ['kraskov1', 'kraskov2']:
        if extraParam != None:
            if isinstance(extraParam, int):
                logger.warning("Number of nearest neighbors needs to be a string. Setting this for you...")
                extraParam = str(extraParam)
            miCalc.setProperty('k', extraParam) # 4th input specifies number of nearest neighbors for KSG estimator
        else:
            miCalc.setProperty('k', '3') # use 3 nearest neighbors for KSG estimator as default
        
    # Make deterministic if kraskov1 or 2 (which adds a small amount of noise to the signal by default)
    if (estMethod in ['kraskov1', 'kraskov2']) and (addNoise == False):
        miCalc.setProperty('NOISE_LEVEL_TO_ADD','0')
    
    # Specify a univariate calculation
    miCalc.initialise(1,1)

    return miCalc
