import warnings
from typing import Union
import logging
logger = logging.getLogger('pyhctsa')

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ..operations.correlation import tc3, trev
from ..operations.information import automutual_info, first_min
from ..operations.nonlinearity import _ms_nlpe, fnn, nlpe
from ..robust import bf_ks_density, bf_random, bf_random_seed
from ..utils import dict_output, get_tau, theiler_window

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _sd_give_me_stats(stat_x: float, stat_surr: ArrayLike, left_right_both: str) -> dict:
    """Compute statistics on the surrogate distribution: the z-score of the series' value against
    the surrogates' (NaN if they are all equal), its distance from their median in interquartile ranges
    (NaN if the interquartile range is 0) and a rank-based p-value (as hctsa, which no longer returns
    a p-value or kernel density of the z-scored value, redundant with these)."""
    num_surrs = len(stat_surr)
    out = {}
    if np.isnan(stat_surr).any():
        logger.warning("SDgivemestats failed")
        return np.nan
    # ASSUME GAUSSIAN DISTRIBUTION: z-statistic of the series' value
    sigma = np.std(stat_surr, ddof=1)
    if not sigma > 0:
        # all surrogates have the same value of this statistic: no meaningful z-score
        out['zscore'] = np.nan
    else:
        out['zscore'] = (stat_x - np.mean(stat_surr)) / sigma

    # what fraction of the range is the sample in? 
    medsurr = np.median(stat_surr)
    iqrsurr = np.quantile(stat_surr, q=.75, method='hazen') - np.quantile(stat_surr, q=.25, method='hazen')
    if iqrsurr == 0:
        out['mediqr'] = np.nan
    else:
        out['mediqr'] = np.abs(stat_x-medsurr)/iqrsurr

    # rank-based p-value
    # number of surrogates strictly below the series' value (the series is ranked
    # ahead of any tied surrogates):
    num_below = int(np.sum(stat_surr < stat_x))
    num_at_least = num_surrs - num_below  # number of surrogates at least as large
    num_extreme = {'right': num_at_least,  # series should be larger than the surrogates
                   'left': num_below,
                   'both': min(num_below, num_at_least)}[left_right_both]  # the more extreme tail
    prank = (num_extreme + 1) / (num_surrs + 1)
    if left_right_both == 'both':
        prank = min(2 * prank, 1)  # two-sided: double, capped at 1

    out['prank'] = prank

    return out

def _random_phase(x: np.ndarray, u: np.ndarray, fc: float = 0) -> np.ndarray:
    """
    Random-phase-Fourier-transform surrogate of x (hctsa's ``SUB_RandomPhase``), with the uniform
    random numbers ``u`` (one per free frequency bin, in (0, 1)) setting the phases.

    The magnitude spectrum of x (at full length N) is kept, and every phase is randomized
    except the DC (and, for even N, Nyquist) phase, which is kept (necessarily 0 or pi for
    a real signal). Randomized phases are negated onto the conjugate-symmetric half so the
    result is real for any N. The phases of the ``fc`` lowest free frequencies are kept
    rather than randomized (truncated Fourier transform surrogates), 0 for none.
    """
    N = len(x)
    n_free = max((N - 1) // 2, 0)
    z = np.fft.fft(x)
    z_mag = np.abs(z)
    z_phase = np.angle(z)

    rand_phase = 2.0 * np.pi * np.asarray(u, dtype=float)
    n_keep = int(np.floor(min(fc, n_free)))
    if n_keep > 0:
        rand_phase[:n_keep] = z_phase[1:1 + n_keep]  # preserve low-frequency phases (TFT)

    if N % 2 == 0:
        new_phase = np.concatenate((z_phase[0:1], rand_phase, z_phase[N // 2:N // 2 + 1],
                                    -rand_phase[::-1]))
    else:
        new_phase = np.concatenate((z_phase[0:1], rand_phase, -rand_phase[::-1]))

    # apply the randomized phases, keep the magnitudes; back to the time domain
    return np.fft.ifft(z_mag * np.exp(1j * new_phase)).real


def _make_surrogates(x: ArrayLike, surr_method: str = 'RP', num_surrs: int = 1,
                    random_seed: Union[int, str, None] = 0,
                    extra_params: Union[float, None] = None) -> ArrayLike:
    """
    Generates surrogate time series (hctsa's ``SD_MakeSurrogates``).

    Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    Used bits of aaft code that references (and presumably was obtained from) [1].

    The random numbers come from the portable generator :func:`~pyhctsa.robust.bf_random`, so
    the surrogates are the same as hctsa's for the same seed, and NumPy's global random state is
    untouched. All the surrogates are drawn in one go: column k of the random numbers (a block
    of consecutive draws) makes surrogate k. (AAFT draws its Gaussian noise from the seed and
    its phases from the seed + 1.)

    References
    ----------
    .. [1] "Surrogate data test for nonlinearity including monotonic
        transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.

    Parameters
    ----------
    x : array-like
        The input time series.
    surr_method : str
        The method for generating surrogates:

        - 'RP': Random phase surrogates: linear correlations are kept and any nonlinear
            structure is destroyed by the phase randomization.
        - 'AAFT': Amplitude adjusted Fourier transform: as 'RP', but the amplitude
            distribution is also (approximately) kept. This is Theiler's algorithm II.
        - 'TFT': Truncated Fourier transform: the phases of the low frequencies are kept and
            the others randomized (a way of dealing with non-stationarity, [2]). The cut-off
            is ``extra_params``.
        - 'RandPerm': Random permutations of the samples, which destroy all temporal
            structure (unlike the above, which keep the linear or amplitude properties).

        Default is ``'RP'``.

    num_surrs : int, optional
        The number of surrogates to generate. Default is 1.
    random_seed : int, str or None, optional
        The seed of the random numbers, as hctsa's ``BF_RandomSeed``: a number, ``'default'`` (0),
        or ``None``/``'none'`` (a seed from NumPy's global stream). Default is 0.
    extra_params : float, optional
        The cut-off frequency for 'TFT', in frequency bins (a value below 1 is taken as a
        proportion of N). Default is N/8.

    Returns
    -------
    np.ndarray
        Array of surrogate time series, one per column.

    References
    ----------
    .. [2] "A new surrogate data method for nonstationary time series", D. L. Guarin Lopez
        et al., arXiv 1008.1804 (2010).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    out = np.zeros(shape=(N, num_surrs))
    seed = bf_random_seed(random_seed)
    n_free = (N - 1) // 2  # number of random phases per surrogate

    def uniform_block(rows, s):  # (rows, num_surrs): column k is a block of consecutive draws
        return bf_random(rows * num_surrs, s).reshape(rows, num_surrs, order='F')

    if surr_method == 'RP':
        phases = uniform_block(n_free, seed)
        for s in range(num_surrs):
            out[:, s] = _random_phase(x, phases[:, s])

    elif surr_method == "AAFT":
        # sort and rank order the data
        ix = np.argsort(x, kind='stable')
        x_sorted = x[ix]
        x_ro = np.argsort(ix, kind='stable')  # rank-ordered permutation
        noise = bf_random(N * num_surrs, seed, 'normal').reshape(N, num_surrs, order='F')
        phases = uniform_block(n_free, seed + 1)
        for s in range(num_surrs):
            # random-order white Gaussian-distributed noise, ranked as x
            n_sort = np.sort(noise[:, s])
            y = n_sort[x_ro]
            # random-phase surrogate of y (phase-randomized noise ranked as x)
            y_rp = _random_phase(y, phases[:, s])
            # rank order x with respect to y_rp
            ix_yrp = np.argsort(y_rp, kind='stable')
            y_ro = np.argsort(ix_yrp, kind='stable')
            out[:, s] = x_sorted[y_ro]

    elif surr_method == "TFT":
        if extra_params is None:
            logger.warning("No cut-off frequency specified for TFT: setting N/8")
            fc = int(np.floor(N / 8 + 0.5))  # MATLAB round
        else:
            fc = extra_params
            if fc < 1:
                fc = N * fc
        phases = uniform_block(n_free, seed)
        for s in range(num_surrs):
            out[:, s] = _random_phase(x, phases[:, s], fc)

    elif surr_method == "RandPerm":
        perms = np.argsort(uniform_block(N, seed), axis=0, kind='stable')
        for s in range(num_surrs):
            out[:, s] = x[perms[:, s]]

    else:
        raise ValueError(f"Unknown method: {surr_method}")

    return out

def _first_min_per_surrogate(z: np.ndarray, min_what: str) -> np.ndarray:
    """First minimum of the automutual information function of each surrogate (NaN on failure)."""
    out = np.zeros(z.shape[1])
    for i in range(z.shape[1]):
        try:
            out[i] = first_min(z[:, i], min_what)
        except Exception:
            out[i] = np.nan
    return out

@dict_output
def surrogate_test(
    x: ArrayLike,
    surr_meth: str = 'RP',
    num_surrs: int = 99,
    the_test_stat: Union[str, ArrayLike] = 'amikraskov1',
    random_seed: Union[int, str, None] = 0,
    extrap: Union[float, None] = None
) -> dict:
    """
    Analyzes test statistics obtained from surrogate time series.

    This function is based on [1].

    The generation of surrogates is done by the periphery function, `_make_surrogates`.

    References
    ----------
    .. [1] "Surrogate data test for nonlinearity including nonmonotonic transforms"
        D. Kugiumtzis, Phys. Rev. E 62(1) R25 (2000).
    .. [2] "Testing for nonlinearity in irregular fluctuations with long-term trends"
            T. Nakamura, M. Small, Y. Hirata, Phys. Rev. E 74(2) 026205 (2006).
    .. [3] "Surrogate time series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346 (2000).

    Parameters
    ----------
    x : array-like
        The input time series.
    surr_meth : str, optional
        The method for generating surrogate time series:
        
        - 'RP': random phase surrogates that maintain linear correlations in
            the data but destroy any nonlinear structure through phase randomization.
        - 'AAFT': amplitude-adjusted Fourier transform method maintains
            linear correlations but destroys nonlinear structure through phase
            randomization, yet preserves the approximate amplitude distribution.
        - 'TFT': preserves low-frequency phases but randomizes high-frequency phases
            (as a way of dealing with non-stationarity, cf. [2]
            "A new surrogate data method for nonstationary time series",
            D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)); the cut-off frequency is
            ``extrap``.
        - 'RandPerm': random permutations of the samples (destroying all temporal structure).

        Default is ``'RP'``.

    num_surrs : int, optional
        The number of surrogates to compute. Default is 99 for a 0.01 significance 
        level 1-sided test.
    the_test_stat : str or array-like, optional
        The test statistic(s) to evaluate on all surrogates and the original time series.
        A single name or a list of names; output is returned for each statistic:

        - 'amikraskov1': the automutual information at lag 1, estimated with the Kraskov
            nearest-neighbor estimator (``automutual_info(y, 1, 'kraskov1', 4)``); tested
            one-sided (surrogates should have lower values), cf. [2].
        - 'fmmikraskov': the first minimum of the Kraskov automutual information function
            (``first_min(y, 'mi-kraskov1')``); tested one-sided.
        - 'o3': a third-order statistic used in [3] (the mean cubed increment at lag 1);
            tested two-sided.
        - 'tc3': a time-reversal asymmetry measure (``tc3`` at lag 1); tested two-sided.
        - 'amigaussian1' and 'fmmigaussian': as 'amikraskov1' and 'fmmikraskov' but with
            the Gaussian estimate of the automutual information. That is a function of
            the autocorrelation only, which random-phase surrogates preserve, so
            these tests cannot detect anything (their p-values are close to uniform for
            every series); hctsa no longer registers them. The earlier names 'ami1' and
            'fmmi' (which both meant the Gaussian estimate) are accepted as
            'amigaussian1' and 'fmmigaussian' with a deprecation warning.

        - 'nlpe': the mean squared error of the locally constant nonlinear prediction
            (:func:`pyhctsa.operations.nonlinearity.nlpe`, embedding dimension 3, delay 1,
            Theiler window of one autocorrelation time), for the series and each surrogate; tested
            one-sided (nonlinear structure makes the series more predictable: its error should be
            lower than the surrogates'). Slow.
        - 'fnn': the fraction of false nearest neighbors at embedding dimension 2
            (:func:`pyhctsa.operations.nonlinearity.fnn`, delay 1, escape factor 5);
            tested one-sided.

        Default is ``'amikraskov1'``. Outputs for each statistic ``s``
        (``amikraskov``, ``fmmikraskov``, ``amigaussian``, ``fmmigaussian``, ``o3``, ``tc3``,
        ``nlpe``, ``fnn``) are ``s_zscore`` (the z-statistic of the series' value against the
        Gaussian fitted to the surrogates' values; NaN if all surrogates have the same value),
        ``s_mediqr`` (distance from the surrogates' median in interquartile ranges;
        NaN if the interquartile range is 0) and ``s_prank`` (rank-based p-value
        ``(k + 1) / (num_surrs + 1)``, with ``k`` the number of surrogates at least as extreme
        as the series in the tested direction; doubled and capped at 1 for two-sided tests).
        As in hctsa, the p-value and kernel density of earlier versions (``s_p`` and ``s_f``)
        are no longer returned.

    random_seed : int, str or None, optional
        The seed of the surrogates (see :func:`_make_surrogates`; ``'default'`` is 0, as in hctsa's
        registered calls). Default is 0.
    extrap : float, optional
        The cut-off frequency for 'TFT' surrogates (see ``_make_surrogates``).

    Returns
    -------
    dict or float
        Dictionary of statistics comparing the original time series to its
        surrogates for each test statistic. NaN if a statistic cannot be computed on
        every surrogate (hctsa raises an error, which its operation wrapper turns into NaN).
    """
    x = np.asarray(x)
    n = len(x)

    if isinstance(the_test_stat, str):
        the_test_stat = [the_test_stat]  # a bare string would be searched for substrings
    the_test_stat = list(the_test_stat)
    # earlier names of the Gaussian-estimate statistics
    for old, new in (('ami1', 'amigaussian1'), ('fmmi', 'fmmigaussian')):
        if old in the_test_stat:
            warnings.warn(
                f"surrogate_test statistic '{old}' is deprecated: use '{new}' (the Gaussian "
                f"estimate) or the Kraskov versions 'amikraskov1' and 'fmmikraskov'.",
                DeprecationWarning, stacklevel=2)
            the_test_stat = [new if t == old else t for t in the_test_stat]

    #Generate surrogate time series
    z = _make_surrogates(x, surr_method=surr_meth, num_surrs=num_surrs, random_seed=random_seed,
                         extra_params=extrap)
    # z is matrix where each column is a surrogate time series
    #% Evaluate test statistic on each surrogate
    out = {}

    def _compare(label, stat_x, stat_surr, side):
        some_stats = _sd_give_me_stats(stat_x, stat_surr, side)
        if not isinstance(some_stats, dict):
            return False  # NaN in the surrogates' statistics: hctsa errors
        for k, v in some_stats.items():
            out[f'{label}_{k}'] = v
        return True

    if 'amikraskov1' in the_test_stat:
        # Kraskov AMI(1) of the surrogates compared to that of the signal itself. Unlike the
        # Gaussian estimate (a function of the autocorrelation, which random-phase surrogates
        # preserve) it responds to nonlinear dependence between x(t) and x(t+1).
        ami_x = automutual_info(x, 1, 'kraskov1', 4)
        ami_surr = np.array([automutual_info(z[:, i], 1, 'kraskov1', 4)
                             for i in range(num_surrs)])
        # surrogates should have lower AMI than the original signal
        if not _compare('amikraskov', ami_x, ami_surr, 'right'):
            return np.nan

    if 'fmmikraskov' in the_test_stat:
        # first minimum of the Kraskov automutual information of the surrogates compared to
        # that of the signal itself
        fmmi_x = first_min(x, 'mi-kraskov1')
        fmmi_surr = _first_min_per_surrogate(z, 'mi-kraskov1')
        if np.isnan(fmmi_surr).any():
            logger.warning("fmmikraskov failed")
            return np.nan
        # the first minimum should be at a higher lag for the signal than for the surrogates
        if not _compare('fmmikraskov', fmmi_x, fmmi_surr, 'right'):
            return np.nan

    if 'amigaussian1' in the_test_stat:
        # AMI(1) of the surrogates compared to that of the signal itself, with the Gaussian
        # estimate (as in Nakamura et al. (2006))
        ami_x = automutual_info(x, 1, 'gaussian')
        ami_surr = np.array([automutual_info(z[:, i], 1, 'gaussian')
                             for i in range(num_surrs)])
        if not _compare('amigaussian', ami_x, ami_surr, 'right'):
            return np.nan

    if 'fmmigaussian' in the_test_stat:
        fmmi_x = first_min(x, 'mi-gaussian')
        fmmi_surr = _first_min_per_surrogate(z, 'mi-gaussian')
        if np.isnan(fmmi_surr).any():
            logger.warning("fmmigaussian failed")
            return np.nan
        # FMMI should be higher for the signal than for the surrogates
        if not _compare('fmmigaussian', fmmi_x, fmmi_surr, 'right'):
            return np.nan

    if 'o3' in the_test_stat:
        #% Third-order statistic in Schreiber, Schmitz (Physica D)
        tau = 1
        o3_x = (1.0 / (n - tau)) * np.sum((x[tau:] - x[:n - tau]) ** 3)
        o3_surr = np.zeros(num_surrs, dtype=float)
        for i in range(num_surrs):
            o3_surr[i] = (1.0 / (n - tau)) * np.sum((z[tau:, i] - z[:n - tau, i]) ** 3)
        if not _compare('o3', o3_x, o3_surr, 'both'):
            return np.nan

    if 'tc3' in the_test_stat:
        # tc3 statistic -- another time-reversal asymmetry measure
        tau = 1
        tmp = tc3(x, tau)
        tc3_x = tmp['raw']
        tc3_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            tmp = tc3(z[:, i], tau)
            tc3_surr[i] = tmp['raw']
        if not _compare('tc3', tc3_x, tc3_surr, 'both'):
            return np.nan

    if 'nlpe' in the_test_stat:
        # locally constant phase space prediction error; embedding parameters fixed
        de, tau = 3, 1
        tmp = nlpe(x, de, tau, 5000, ('ac', 1))
        if not isinstance(tmp, dict):  # hctsa errors on taking the field of a NaN
            return np.nan
        nlpe_x = tmp['msqerr']
        nlpe_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            th = theiler_window(z[:, i], ('ac', 1), n)
            if np.isnan(th):
                return np.nan
            res = _ms_nlpe(z[:, i], de, tau, int(th))
            nlpe_surr[i] = np.mean(np.asarray(res, dtype=float) ** 2)  # the mean squared error, as for the series
        # nonlinear structure makes the series more predictable: lower error than the surrogates
        if not _compare('nlpe', nlpe_x, nlpe_surr, 'left'):
            return np.nan

    if 'fnn' in the_test_stat:
        # false nearest neighbors at d = 2
        tmp = fnn(x, 1, 2, ('ac', 1), False, escape_factor=5)
        if not isinstance(tmp, dict):
            return np.nan
        fnn_x = tmp['pfnn_2']
        fnn_surr = np.zeros(num_surrs)
        for i in range(num_surrs):
            tmp = fnn(z[:, i], 1, 2, ('ac', 1), False, escape_factor=5)
            if not isinstance(tmp, dict):
                return np.nan
            fnn_surr[i] = tmp['pfnn_2']
        if not _compare('fnn', fnn_x, fnn_surr, 'right'):
            return np.nan

    return out


@dict_output
def surrogates(
    y: ArrayLike,
    tau: Union[int, str] = 1,
    nsurr: int = 50,
    surr_method: int = 1,
    surrfn: str = 'tc3',
    random_seed: Union[int, str, None] = 0
) -> Union[dict, float]:
    """
    Surrogate data test of a nonlinear statistic, tc3 or trev (hctsa's ``SD_Surrogates``).

    Generates surrogate time series and tests them against the original time series
    according to a test statistic: T_{C3} (:func:`tc3`) or T_{rev} (:func:`trev`), both
    reimplementations of the expressions originally used by the TSTOOL package's tc3/trev
    functions. The statistic is computed on the series and on each surrogate, and the
    outputs describe where the series' value lies in the distribution of the surrogates'
    values (a Gaussian fit to them, their median and interquartile range, and a
    kernel-smoothed density).

    Parameters
    ----------
    y : array-like
        The input time series.
    tau : int or str, optional
        The time lag used in the test statistic: an integer, or a string that sets it
        from the series: 'ac' for the first zero-crossing of the autocorrelation function,
        'ac1e' for the floor of its first 1/e crossing, or 'mi' for the smaller of the first
        minimum of the Kraskov automutual information and the 'ac1e' delay (see
        ``utils.get_tau``). The delay is set once, from the original series, and used for
        the series and all its surrogates. For 'tc3' the statistic is
        ``<x_n x_{n-tau} x_{n-2tau}> / |<x_n x_{n-tau}>|^(3/2)``; for 'trev' it is
        ``<d^3> / <d^2>^(3/2)`` for the increments ``d = x_{n+tau} - x_n``. Default is 1.
    nsurr : int, optional
        The number of surrogates to generate. Default is 50.
    surr_method : int, optional
        The method of generating surrogates: 1 randomizes the phases of the Fourier
        spectrum ('RP'), 2 is Theiler's algorithm II ('AAFT'), 3 permutes the samples
        randomly ('RandPerm'). Default is 1.
    surrfn : {'tc3', 'trev'}, optional
        The statistic to evaluate on all surrogates. Default is ``'tc3'``.
    random_seed : int, str or None, optional
        The seed of the surrogates (see :func:`_make_surrogates`; ``'default'`` is 0, as in hctsa's
        registered calls). Default is 0.

    Returns
    -------
    dict or float
        With ``s`` the value of the statistic on the series, and the surrogates' values
        having mean ``muhat``, standard deviation ``sigmahat``, median and interquartile
        range ``iqrsurr``:

        - ``'meansurr'``, ``'stdsurr'``: the mean and standard deviation of the statistic
          over the surrogates.
        - ``'normpatponmax'``: the Gaussian density N(muhat, sigmahat) at ``s`` relative to
          its peak value.
        - ``'stdfrommean'``: ``|s - muhat| / sigmahat``.
        - ``'ztestp'``: the p-value of a z-test of ``s`` against N(muhat, sigmahat).
        - ``'iqrsfrommedian'``: ``|s - median| / iqrsurr`` (NaN if ``iqrsurr`` is 0).
        - ``'kspminfromext'``: the smaller of the kernel-density probabilities of a value
          below and above ``s`` (0 if ``s`` lies above the density grid).
        - ``'ksphereonmax'``: the kernel density at ``s`` relative to the peak of
          N(muhat, sigmahat) (0 if ``s`` lies above the density grid).
        - ``'ksiqrsfrommode'``: ``|s - (mode of the kernel density)| / iqrsurr`` (NaN if
          ``iqrsurr`` is 0).

        ``normpatponmax``, ``stdfrommean``, ``ztestp``, ``kspminfromext`` and ``ksphereonmax``
        are NaN if the surrogates all have the same value (the kernel density uses
        :func:`~pyhctsa.robust.bf_ks_density`). The output is NaN if ``tau`` cannot be
        determined or the statistic fails for any surrogate.
    """
    y = np.asarray(y, dtype=float).ravel()

    # (1) time delay, tau: resolved once, from the original series
    if isinstance(tau, str) and tau == 'ac':
        from ..operations.distribution import first_crossing
        tau = first_crossing(y, 'ac', 0, 'discrete')
    elif isinstance(tau, str):
        tau = get_tau(y, tau)  # 'ac1e' or 'mi'
    if tau is None or np.isnan(tau):
        return np.nan
    tau = int(tau)

    # (3) surrogate data method: TSTOOL's numeric convention -> _make_surrogates
    native_methods = {1: 'RP', 2: 'AAFT', 3: 'RandPerm'}
    if surr_method not in native_methods:
        raise ValueError(f"Unknown surrogate method {surr_method}")

    # (4) the statistic, evaluated identically on the original series and on each surrogate
    if surrfn == 'tc3':
        stat_fn = tc3
    elif surrfn == 'trev':
        stat_fn = trev
    else:
        raise ValueError(f"Unknown surrogate function '{surrfn}'")

    def stat(x):
        res = stat_fn(x, tau)
        return res['raw'] if isinstance(res, dict) else np.nan

    tc3_y = stat(y)
    surr = _make_surrogates(y, native_methods[surr_method], nsurr, random_seed)
    tc3_surr = np.array([stat(surr[:, i]) for i in range(nsurr)])

    if np.isnan(tc3_surr).any():
        logger.warning(f"Surrogate statistic '{surrfn}' failed for a surrogate")
        return np.nan  # (hctsa errors if it fails for all; its z-test errors on a NaN std)

    out = {}
    # 1) fit a Gaussian to the surrogates
    muhat = np.mean(tc3_surr)
    sigmahat = np.std(tc3_surr, ddof=1)
    if sigmahat == 0:
        # all surrogates give an identical value of this statistic: cannot meaningfully
        # assess how many stds/z the data value is from them
        out['normpatponmax'] = np.nan
        out['stdfrommean'] = np.nan
        out['ztestp'] = np.nan
    else:
        # probability of the data given Gaussian surrogates
        out['normpatponmax'] = np.exp(-0.5 * ((tc3_y - muhat) / sigmahat) ** 2)
        # 2) stds from mean
        out['stdfrommean'] = np.abs(tc3_y - muhat) / sigmahat
        # (~equivalent to a z-test:)
        out['ztestp'] = 2 * norm.sf(np.abs((tc3_y - muhat) / sigmahat))

    # iqrs from median
    iqrsurr = np.quantile(tc3_surr, 0.75, method='hazen') - np.quantile(tc3_surr, 0.25, method='hazen')
    if iqrsurr == 0:
        out['iqrsfrommedian'] = np.nan
    else:
        out['iqrsfrommedian'] = np.abs(tc3_y - np.median(tc3_surr)) / iqrsurr

    # 3) basic info on the surrogates
    out['stdsurr'] = sigmahat
    out['meansurr'] = muhat

    # 4) kernel density test
    ksf, ksx, _ = bf_ks_density(tc3_surr)
    ksdx = ksx[1] - ksx[0]
    above = np.flatnonzero(ksx > tc3_y)
    if np.any(np.isnan(ksf)):  # all surrogates have the same value: no scale to smooth over
        out['kspminfromext'] = np.nan
        out['ksphereonmax'] = np.nan
    elif above.size == 0:  # off the scale!
        out['kspminfromext'] = 0.0
        out['ksphereonmax'] = 0.0
    else:
        ihit = above[0]
        pfromleft = ksdx * np.sum(ksf[:ihit + 1])
        out['kspminfromext'] = min(pfromleft, 1 - pfromleft)
        # relative to the peak of the fitted Gaussian (undefined for a zero-width one)
        out['ksphereonmax'] = (ksf[ihit] * sigmahat * np.sqrt(2 * np.pi)) if sigmahat > 0 else np.nan

    # iqrs from mode
    imode = int(np.argmax(ksf)) if not np.any(np.isnan(ksf)) else 0
    if iqrsurr == 0:
        out['ksiqrsfrommode'] = np.nan
    else:
        out['ksiqrsfrommode'] = np.abs(ksx[imode] - tc3_y) / iqrsurr

    return out
