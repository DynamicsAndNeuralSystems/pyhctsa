from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ansari

from ..operations.correlation import autocorr, first_crossing
from ..operations.stationarity import sliding_window
from ..robust import bf_ks_density, bf_runs_z
from ..utils import get_tau, matlab_quantile

def walker(y: ArrayLike, walker_rule: str = 'prop',
           walker_params: Union[None, float, int, list] = None) -> dict:
    """
    Simulates a hypothetical walker moving through the time domain.

    The hypothetical particle (or 'walker') moves in response to values of the
    time series at each point. Outputs from this operation are summaries of the
    walker's motion, and comparisons of it to the original time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    walker_rule : str, optional
        The kinematic rule by which the walker moves in response to the
        time series over time:

        - 'prop': the walker narrows the gap between its value and that of the
          previous value of the time series by a given proportion p:
          w[i] = w[i-1] + p * (y[i-1] - w[i-1]), with w[0] = 0.
          walker_params = p
        - 'biasprop': biased motion; when the time series has just gone up
          (y[i-1] > y[i-2]) the walker narrows the gap to y[i-1] by a proportion
          p_up, and otherwise (including at the first step) by p_down.
          walker_params = [pup, pdown]
        - 'momentum': the walker moves with mass m and inertia from the
          previous step: it extrapolates its previous step,
          w_inert = 2 w[i-1] - w[i-2], then closes a fraction 1/m of the gap
          between w_inert and y[i-1]. walker_params = m
        - 'runningvar': inertial motion as above, with values rescaled by the
          ratio of the standard deviation of the last wl+1 values of y (up to
          y[i-1]) to that of the walker over the same window, including its
          provisional new value. walker_params = [m, wl] (inertial mass,
          window length)

        Default is ``'prop'``.

    walker_params : float, int, or list, optional
        The parameters for the specified walker_rule. Default is None
        (use pre-defined defaults).

    Returns
    -------
    dict
        Summaries of the walker's trajectory (``w_mean``, ``w_median``, ``w_std``,
        ``w_min``, ``w_max``, ``w_ac1``, ``w_ac2``, ``w_tau``, ``w_propzcross``),
        of the walker compared with the series (``sw_meanabsdiff``,
        ``sw_taudiff``, ``sw_stdrat``, ``sw_minrat``, ``sw_maxrat``,
        ``sw_ac1diff`` (lag-1 autocorrelation of w minus that of y),
        ``sw_propcross``, ``sw_ansarib_pval``, ``sw_distdiff`` (L1 distance
        between the kernel-smoothed densities of y and w, at most 2)), and of
        the residual w - y (``res_runsz`` (signed z-statistic of a runs test for randomness,
        :func:`pyhctsa.robust.bf_runs_z`: negative when it has fewer runs about its median
        than expected), ``res_swss5_1``, ``res_ac1``).
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    # Default values and type requirements for each rule
    WALKER_CONFIGS = {
        'prop': {
            'default': 0.5,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        },
        'biasprop': {
            'default': [0.1, 0.2],
            'valid_types': (list,),
            'error_msg': 'must be a list'
        },
        'momentum': {
            'default': 2,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        },
        'runningvar': {
            'default': [1.5, 50],
            'valid_types': (list,),
            'error_msg': 'must be a list'
        }
    }

    if walker_rule not in WALKER_CONFIGS:
        valid_rules = ", ".join(f"'{rule}'" for rule in WALKER_CONFIGS)
        raise ValueError(f"Unknown walker_rule: '{walker_rule}'. Choose from: {valid_rules}")

    config = WALKER_CONFIGS[walker_rule]

    if walker_params is None:
        walker_params = config['default']

    if not isinstance(walker_params, config["valid_types"]):
        raise ValueError(
            f"walker_params {config['error_msg']} for walker rule: '{walker_rule}'"
        )

    # ------------------------------------------------------------------
    # Do the walk
    # ------------------------------------------------------------------
    w = np.zeros(N)

    if walker_rule == 'prop':
        # walker narrows the gap between its position and the series value
        # by the proportion walker_params at each step
        p = walker_params
        for i in range(1, N):
            w[i] = w[i-1] + p * (y[i-1] - w[i-1])

    elif walker_rule == 'biasprop':
        # biased motion: [p_up, p_down]
        pup, pdown = walker_params
        for i in range(1, N):
            # direction of the change just observed, y[i-1] vs y[i-2];
            # p_down at the first step (no previous change)
            if i >= 2 and y[i-1] > y[i-2]:
                w[i] = w[i-1] + pup * (y[i-1] - w[i-1])
            else:
                w[i] = w[i-1] + pdown * (y[i-1] - w[i-1])

    elif walker_rule == 'momentum':
        # walker moves with inertia; the series acts as a force
        m = walker_params  # 'inertial mass'
        w[0] = y[0]
        w[1] = y[1]
        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1] - w[i-2])
            w[i] = w_inert + (y[i-1] - w_inert) / m  # dissipative term

    elif walker_rule == 'runningvar':
        # inertial motion rescaled by local standard deviation
        m, wl = walker_params
        wl = int(wl)
        w[0] = y[0]
        w[1] = y[1]
        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1] - w[i-2])
            w_mom = w_inert + (y[i-1] - w_inert) / m  # dissipative term
            # MATLAB: if i > wl + 1, with i the 1-based index (= i + 1 here).
            if i > wl:
                # w[i] is not yet computed, so the local std of the walker is
                # built from its previous wl values plus the provisional w_mom.
                # The series is read one step lagged: y[i-wl-1 : i] (wl+1 samples).
                sy = np.std(y[i-wl-1:i], ddof=1)
                sw = np.std(np.append(w[i-wl:i], w_mom), ddof=1)
                w[i] = w_mom * (sy / sw)
            else:
                w[i] = w_mom

    # ------------------------------------------------------------------
    # Statistics on the walk
    # ------------------------------------------------------------------
    out = {}

    # (i) The walk itself
    out['w_mean'] = np.mean(w)
    out['w_median'] = np.median(w)
    out['w_std'] = np.std(w, ddof=1)
    out['w_ac1'] = autocorr(w, 1, 'Fourier')
    out['w_ac2'] = autocorr(w, 2, 'Fourier')
    out['w_tau'] = first_crossing(w, 'ac', 0, 'continuous')
    out['w_min'] = np.min(w)
    out['w_max'] = np.max(w)
    out['w_propzcross'] = np.sum((w[:-1] * w[1:]) < 0) / (N - 1)

    # (ii) Differences between the walk and the signal
    out['sw_meanabsdiff'] = np.mean(np.abs(y - w))
    out['sw_taudiff'] = (first_crossing(y, 'ac', 0, 'continuous')
                         - first_crossing(w, 'ac', 0, 'continuous'))
    out['sw_stdrat'] = np.std(w, ddof=1) / np.std(y, ddof=1)
    # a difference, not a ratio, which blows up when y has ac1 near 0
    out['sw_ac1diff'] = out['w_ac1'] - autocorr(y, 1, 'Fourier')
    out['sw_minrat'] = np.min(w) / np.min(y)
    out['sw_maxrat'] = np.max(w) / np.max(y)
    out['sw_propcross'] = np.sum((w[:-1] - y[:-1]) * (w[1:] - y[1:]) < 0) / (N - 1)

    # Ansari-Bradley test: same distribution?
    _, pval = ansari(w, y)
    out['sw_ansarib_pval'] = pval

    # L1 distance between the kernel-smoothed densities of y and w, on a common
    # grid of 200 points (sum times grid spacing: at most 2, independent of the
    # range of the data)
    r = np.linspace(min(np.min(y), np.min(w)), max(np.max(y), np.max(w)), 200)
    dy, _, _ = bf_ks_density(y, r)
    dw, _, _ = bf_ks_density(w, r)
    out['sw_distdiff'] = np.sum(np.abs(dy - dw)) * (r[1] - r[0])

    # (iii) Residuals between time series and walker
    res = w - y
    out['res_runsz'] = bf_runs_z(res)  # runs test z-statistic
    out['res_swss5_1'] = sliding_window(res, 'std', 'std', 5, 1)
    out['res_ac1'] = autocorr(res, 1)

    return out

def force_potential(y: ArrayLike, what_potential: str = 'dblwell',
                    params: Union[list, None] = None) -> dict:
    """
    Couple a time series to a driven dynamical system.

    The input time series acts as an external forcing term on a simulated
    particle evolving in a specified potential well.

    Two potential functions are available:

    1. **Quartic double-well potential**

    .. math::

        V(x) = \\frac{x^4}{4} - \\frac{\\alpha^2 x^2}{2},

    with corresponding force

    .. math::

        F(x) = -\\frac{dV}{dx} = -x^3 + \\alpha^2 x.

    2. **Sinusoidal potential**

    .. math::

        V(x) = -\\cos\\left(\\frac{x}{\\alpha}\\right),

    with corresponding force

    .. math::

        F(x) = -\\frac{1}{\\alpha}
        \\sin\\left(\\frac{x}{\\alpha}\\right).

    The time series provides a forcing contribution to the particle dynamics,
    which are integrated numerically.

    Parameters
    ----------
    y : array-like
        Input time series.

    what_potential : str, optional
        Potential function to simulate.

        - ``'dblwell'``: Quartic double-well potential.
        - ``'sine'``: Sinusoidal potential.

        Default is ``'dblwell'``

    params : list of float, optional
        Simulation parameters in the form ``[alpha, kappa, deltat]``.

        - ``alpha``: Controls the well separation (``"dblwell"``) or the
        oscillation period (``"sine"``).
        - ``kappa``: Friction (damping) coefficient.
        - ``deltat``: Integration time step.

        Default is `None` (use pre-defined defaults).

    Returns
    -------
    dict
        Summary statistics of the simulated trajectory, including mean,
        range, proportion of positive values, zero-crossing rate,
        autocorrelation, ``meanabs`` (the mean magnitude of the position, a robust
        summary of how far the particle is from the center; it is not sensitive to the
        exact end of the run) and standard deviation.
    """
    y = np.asarray(y, dtype=np.float64)

    DEFAULT_PARAMS = {'dblwell': [2, 0.1, 0.1], 'sine': [1, 1, 1]}
    if what_potential not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown potential function {what_potential}")

    if params is None:
        params = DEFAULT_PARAMS[what_potential]
    if not isinstance(params, list):
        raise ValueError("Expected list of parameters.")
    if len(params) != 3:
        raise ValueError("Expected 3 parameters.")

    N = len(y) # length of the time series
    alpha, kappa, deltat = params

    # force F(x) = -dV/dx for the chosen potential V(x)
    if what_potential == 'sine':
        # V(x) = -cos(x / alpha)
        F = lambda x: -np.sin(x/alpha)/alpha
    else:  # 'dblwell': V(x) = x^4 / 4 - alpha^2 x^2 / 2
        F = lambda x: -x**3 + alpha**2 * x

    x = np.zeros(N) # position
    v = np.zeros(N) # velocity

    for i in range(1, N):
        x[i] = x[i-1] + v[i-1]*deltat + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat**2
        v[i] = v[i-1] + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat

    # check the trajectory didn't blow out
    if np.isnan(x[-1]) or np.abs(x[-1]) > 1E10:
        return np.nan
    
    # Output some basic features of the trajectory
    out = {}
    out['mean'] = np.mean(x) # mean position
    out['median'] = np.median(x) # median position
    out['std'] = np.std(x, ddof=1) # std. dev.
    out['range'] = np.ptp(x)
    out['proppos'] = np.sum(x >0)/N
    out['pcross'] = np.sum(x[:-1] * x[1:] < 0) / (N - 1)
    out['ac1'] = np.abs(autocorr(x, 1, 'Fourier'))
    out['ac10'] = np.abs(autocorr(x, 10, 'Fourier'))
    out['ac50'] = np.abs(autocorr(x, 50, 'Fourier'))
    out['tau'] = first_crossing(x, 'ac', 0, 'continuous')
    out['meanabs'] = np.mean(np.abs(x)) # mean magnitude of the position

    # additional outputs for dbl well
    if what_potential == 'dblwell':
        out['pcrossup'] = np.sum((x[:-1] - alpha) * (x[1:] - alpha) < 0) / (N - 1)
        out['pcrossdown'] = np.sum((x[:-1] + alpha) * (x[1:] + alpha) < 0) / (N - 1)

    return out


def kramers_moyal(y: ArrayLike, tau: Union[int, str] = 1, num_bins: int = 15) -> Union[dict, float]:
    """
    How the series' average drift and noise intensity depend on its current level, from its increments.

    Treats the time series as a sampled Langevin process,
    ``dx = D1(x) dt + sqrt(2 D2(x)) dW``, and estimates the conditional-moment
    (Kramers-Moyal) coefficients as a function of the current level x, from increments
    over `tau` samples::

        D1(x) = E[dx | x] / tau            (drift: the deterministic restoring force)
        D2(x) = E[dx^2 | x] / (2 tau)      (diffusion: the local noise intensity)
        D4(x) = E[dx^4 | x] / (24 tau)

    with conditioning on x done in equiprobable (quantile) bins.

    The features summarize the shape of D1 and D2 across levels: a linear D1 with a constant
    D2 is an Ornstein-Uhlenbeck (AR(1)-like) process; a cubic D1 indicates a nonlinear
    potential (e.g., bistability); a D2 that varies with x indicates state-dependent noise
    (note that a monotone transformation of an Ornstein-Uhlenbeck process also has
    state-dependent D2, so this is partly a property of the marginal distribution); the
    Pawula ratio D4/D2^2 is small for continuous diffusion and large when increments are
    dominated by jumps.

    Parameters
    ----------
    y : array-like
        The input time series (z-scored in hctsa).
    tau : int or str, optional
        The increment lag, in samples (default: 1); or a rule that sets it from the series:
        'ac' (first zero-crossing of the autocorrelation function), 'ac1e' (floor of its
        first 1/e crossing), or 'mi' (the smaller of the first minimum of the Kraskov
        automutual information and the 'ac1e' delay); see `pyhctsa.utils.get_tau`.
    num_bins : int, optional
        The number of equiprobable bins used to condition on x (default: 15).

    Returns
    -------
    dict or float
        Dictionary containing:

        - `driftLin`, `driftQuad`, `driftCubic`: coefficients of a count-weighted cubic fit of
          D1(x) (`driftCubic` < 0 for a stiffening restoring force).
        - `driftNonlinGain`: fraction of the residual variance of a linear drift fit that is
          removed by the cubic fit (0 = linear drift).
        - `diffSlope`, `diffCurv`: linear and quadratic coefficients of a quadratic fit of
          D2(x), each relative to the mean of D2 across bins (0 = additive noise).
        - `diffTailRatio`: mean D2 in the outer bins (the lowest and highest fifth of the
          bins) / mean D2 in the central bins.
        - `pawula`: mean over bins of D4/D2^2 (tau/2 for Gaussian increments; larger for
          jumps). Unlike the shape of D2, not changed by a monotone transformation of the data.

        NaN (a single float) if there are fewer than 20 * `num_bins` increments, if no delay
        can be estimated for a `tau` rule, or if tied values make the quantile bins coincide.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if isinstance(tau, str):
        tau = get_tau(y, tau)  # adaptive delay
        if np.isnan(tau):
            return np.nan  # data-dependent: no correlation length could be estimated
    tau = int(tau)

    x = y[:-tau]
    dx = y[tau:] - x
    if x.size < 20 * num_bins:
        return np.nan  # data-dependent: too few samples per bin

    # Conditional moments in equiprobable bins of x
    edges = matlab_quantile(x, np.linspace(0, 1, num_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_ = np.digitize(x, edges[1:-1])  # 0..num_bins-1; bin i is [edges[i], edges[i+1])
    n = np.bincount(bin_, minlength=num_bins).astype(np.float64)
    if np.any(n == 0):
        return np.nan  # data-dependent: heavily tied values (quantile edges coincide)
    xc = np.empty(num_bins)
    D1 = np.empty(num_bins)
    D2 = np.empty(num_bins)
    D4 = np.empty(num_bins)
    for b in range(num_bins):
        m = bin_ == b
        xc[b] = np.median(x[m])
        D1[b] = np.mean(dx[m]) / tau
        D2[b] = np.mean(dx[m] ** 2) / (2 * tau)
        D4[b] = np.mean(dx[m] ** 4) / (24 * tau)

    # Drift: linear vs cubic fit (weighted by bin counts, which are ~equal)
    w = np.sqrt(n)
    V3 = np.column_stack([np.ones(num_bins), xc, xc ** 2, xc ** 3])
    c3 = np.linalg.lstsq(V3 * w[:, None], D1 * w, rcond=None)[0]
    V1 = V3[:, :2]
    c1 = np.linalg.lstsq(V1 * w[:, None], D1 * w, rcond=None)[0]
    res1 = np.sum(n * (D1 - V1 @ c1) ** 2)
    res3 = np.sum(n * (D1 - V3 @ c3) ** 2)
    out = {
        'driftLin': c3[1],
        'driftQuad': c3[2],
        'driftCubic': c3[3],
        'driftNonlinGain': 1 - res3 / res1,
    }

    # Diffusion: quadratic fit, relative to the mean of D2 across bins (always positive), not
    # to the fitted D2 at x = 0, which can be near zero or negative for an unconstrained fit
    V2 = V3[:, :3]
    b2 = np.linalg.lstsq(V2 * w[:, None], D2 * w, rcond=None)[0]
    out['diffSlope'] = b2[1] / np.mean(D2)
    out['diffCurv'] = b2[2] / np.mean(D2)
    num_outer = max(1, num_bins // 5)
    is_outer = np.zeros(num_bins, dtype=bool)
    is_outer[:num_outer] = True
    is_outer[num_bins - num_outer:] = True
    is_central = np.zeros(num_bins, dtype=bool)
    half = num_outer // 2
    is_central[int(np.ceil(num_bins / 2)) - 1 + np.arange(-half, half + 1)] = True
    out['diffTailRatio'] = np.mean(D2[is_outer]) / np.mean(D2[is_central])

    # Pawula ratio (jumps vs continuous diffusion); the mean over bins is more reliable than the median
    out['pawula'] = np.mean(D4 / D2 ** 2)
    return out
