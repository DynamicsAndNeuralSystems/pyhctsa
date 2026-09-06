import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import betainc, betaincc, betaln


def betafit(x: np.ndarray) -> np.ndarray:
    """Beta-distribution MLE matching MATLAB's betafit.

    Values within eps of 0 or 1 are handled with a mixed likelihood that
    assigns them the probability mass of the corresponding tail, so data
    scaled to have max(x) == 1 (as in HT_DistributionTest) remain fittable.
    """
    n = len(x)
    with np.errstate(divide='ignore'):
        sumlogx = np.sum(np.log(x))
        sumlog1mx = np.sum(np.log1p(-x))
    # Moment-style starting point
    tmp1 = np.exp(sumlog1mx / n)
    tmp2 = np.exp(sumlogx / n)
    tmp3 = 1 - tmp1 - tmp2
    pstart = np.log([0.5 * (1 - tmp1) / tmp3, 0.5 * (1 - tmp2) / tmp3])

    xl = np.sqrt(np.finfo(float).tiny)  # tolerance above zero
    xu = 1 - np.finfo(float).eps / 2
    is0 = x < xl
    is1 = x > xu
    n0, n1 = int(np.sum(is0)), int(np.sum(is1))

    if n0 == 0 and n1 == 0:
        def negloglike(logp):
            a, b = np.exp(logp)
            return n * betaln(a, b) - (a - 1) * sumlogx - (b - 1) * sumlog1mx
    else:
        x2 = x[~is0 & ~is1]
        n2 = len(x2)
        sumlogx2 = np.sum(np.log(x2))
        sumlog1mx2 = np.sum(np.log1p(-x2))

        def negloglike(logp):
            a, b = np.exp(logp)
            nll = n2 * betaln(a, b) - (a - 1) * sumlogx2 - (b - 1) * sumlog1mx2
            with np.errstate(divide='ignore'):
                if n0 > 0:  # Pr(X <= xl) for data that are zeros
                    nll -= n0 * np.log(betainc(a, b, xl))
                if n1 > 0:  # Pr(X >= xu) for data that are ones
                    nll -= n1 * np.log(betaincc(a, b, xu))
            return nll

    # Nelder-Mead on log-parameters, matching fminsearch's tolerances/limits
    res = minimize(negloglike, pstart, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 400, 'maxfev': 400})
    return np.exp(res.x)

def evfit(x: np.ndarray) -> tuple:
    """MLE of the type-I extreme value (Gumbel, minima) distribution,
    matching MATLAB's evfit: the profile likelihood is solved for the scale
    parameter, then the location parameter follows in closed form."""
    xmax = np.max(x)
    mean_x = np.mean(x)

    def profile(sigma):
        w = np.exp((x - xmax) / sigma)  # shift by xmax for numerical stability
        return np.sum(x * w) / np.sum(w) - mean_x - sigma

    # Method-of-moments starting value; profile() decreases through its root
    sigma0 = np.sqrt(6) * np.std(x, ddof=1) / np.pi
    lo = hi = sigma0
    if profile(sigma0) > 0:
        while profile(hi) > 0:
            hi *= 2
    else:
        while profile(lo) < 0:
            lo /= 2
    sigma = brentq(profile, lo, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps)
    mu = xmax + sigma * np.log(np.mean(np.exp((x - xmax) / sigma)))
    return mu, sigma