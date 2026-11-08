"""
Tests for the Gaussian-process hyperparameter layout refactor (one ``_gp_layout`` in place of the
per-role position walkers in ``model_fit.py``).

Two checks:

1. A same-runner reference check (the one that matters): verbatim copies of the removed walkers,
   the old ``_gp_init_hyp`` loop and the old ``_gp_shape_bounds`` from main 0fa9013 are compared
   with ``_gp_layout``, the new ``_gp_init_hyp`` and the new ``_gp_shape_bounds`` on every ordering
   of up to four components; positions must be equal exactly and initial values to rtol 1e-12.
   This has no platform dependence.
2. Operation-level pins, whose expected values were generated from the main checkout (0fa9013)
   before the refactor on Ben's Mac, at a tolerance set per output (see ``CASES``).
"""
import itertools

import numpy as np
import pytest

import pyhctsa.operations.model_fit as mf

RTOL = 1e-12
# GP marginal-likelihood optimization (an iterative optimizer): tier 3 of the CI-fix policy.
# CI drift on the identified outputs below was at most 8e-9 (ubuntu, windows).
RTOL_GP = 1e-5


# --- Reference copies of the pre-refactor code (main 0fa9013, pyhctsa/operations/model_fit.py) ---

# model_fit.py:3781 on main 0fa9013
def _ref_gp_alpha_pos(components: list) -> list:
    """0-based positions of the log shape parameters of ``covRQiso`` among the covariance hyperparameters."""
    pos = 0
    alpha = []
    for name, _ in components:
        if name in ('covSEiso', 'covMaterniso'):
            pos += 2
        elif name == 'covPeriodic':
            pos += 3
        elif name == 'covRQiso':
            alpha.append(pos + 2)
            pos += 3
        else:
            pos += 1
    return alpha


# model_fit.py:3874 on main 0fa9013
def _ref_gp_amp_pos(components: list) -> list:
    """0-based positions of the log signal amplitudes among the covariance hyperparameters."""
    pos = 0
    amps = []
    for name, _ in components:
        if name in ('covSEiso', 'covMaterniso', 'covRQiso'):
            amps.append(pos + 1)
        elif name == 'covPeriodic':
            amps.append(pos + 2)
        pos += {'covSEiso': 2, 'covMaterniso': 2, 'covPeriodic': 3, 'covRQiso': 3}.get(name, 1)
    return amps


# model_fit.py:3887 on main 0fa9013
def _ref_gp_shape_pos(components: list) -> list:
    """
    0-based positions, among the covariance hyperparameters, of those that are not identified
    without correlation structure: the length scales, the period of ``covPeriodic`` and the
    shape of ``covRQiso``.
    """
    pos = 0
    shape = []
    for name, _ in components:
        if name in ('covSEiso', 'covMaterniso'):
            shape.append(pos)
        elif name == 'covPeriodic':
            shape += [pos, pos + 1]
        elif name == 'covRQiso':
            shape += [pos, pos + 2]
        pos += {'covSEiso': 2, 'covMaterniso': 2, 'covPeriodic': 3, 'covRQiso': 3}.get(name, 1)
    return shape


# model_fit.py:3906 on main 0fa9013
def _ref_gp_shape_bounds(components: list, tt: np.ndarray) -> dict:
    """
    The values reported for the hyperparameters of :func:`_gp_shape_pos` when the series shows no
    correlation structure, keyed by position: for a length scale, the sampling interval (a bound:
    the data do not resolve a length scale above it); NaN for periods and shape parameters.
    """
    dt = np.mean(np.diff(tt))
    lens = set(_ref_gp_len_pos(components))
    return {p: (np.log(dt) if p in lens else np.nan) for p in _ref_gp_shape_pos(components)}


# model_fit.py:3917 on main 0fa9013
def _ref_gp_len_pos(components: list) -> list:
    """0-based positions of the log length scales among the covariance hyperparameters."""
    pos = 0
    lens = []
    for name, _ in components:
        if name in ('covSEiso', 'covMaterniso', 'covRQiso'):
            lens.append(pos)
        pos += {'covSEiso': 2, 'covMaterniso': 2, 'covPeriodic': 3, 'covRQiso': 3}.get(name, 1)
    return lens


# model_fit.py:4020 on main 0fa9013
def _ref_gp_noise_pos(components: list) -> list:
    """
    0-based positions of the ``covNoise`` standard deviations among the covariance
    hyperparameters, found as ``MF_GP_LearnHyperp`` does (while it sets the initial values):
    every component advances the position by its own number of hyperparameters (2 for a
    degree-parameterized ``covMaterniso``).
    """
    pos = 0
    noise = []
    for name, degree in components:
        if name in ('covSEiso', 'covMaterniso'):
            pos += 2
        elif name in ('covPeriodic', 'covRQiso'):
            pos += 3
        elif name == 'covNoise':
            noise.append(pos)
            pos += 1
        else:
            pos += 1
    return noise


# model_fit.py:5206 on main 0fa9013 (docstring omitted)
def _ref_gp_init_hyp(components: list, tt: np.ndarray) -> np.ndarray:
    n_cov = int(sum({'covSEiso': 2, 'covPeriodic': 3, 'covRQiso': 3, 'covNoise': 1,
                     'covMaterniso': 2}[name] for name, _ in components))
    typical_dt = np.mean(np.diff(tt))  # typical time step: a length-scale prior
    data_span = np.max(tt) - np.min(tt)
    hyp = np.zeros(n_cov)
    pos = 0
    for name, degree in components:
        if name in ('covSEiso', 'covMaterniso'):
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = 0.0                  # log-magnitude
            pos += 2
        elif name == 'covPeriodic':
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = np.log(data_span / 10)  # period (guess: ~10 cycles across the data)
            hyp[pos + 2] = 0.0                  # log-magnitude
            pos += 3
        elif name == 'covRQiso':
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = 0.0                  # log-magnitude
            hyp[pos + 2] = 0.0                  # log-alpha (shape)
            pos += 3
        elif name == 'covNoise':
            hyp[pos] = np.log(0.1)              # noise magnitude
            pos += 1
        else:
            pos += 1                            # unrecognized component: leave at zero
    return np.r_[hyp, np.log(0.1)]


# Every ordering (with repetition) of one to four components: covMaterniso with and without a
# degree, and one unrecognized name (parse_cov rejects it, but the walkers gave it one position).
_POOL = [('covSEiso', None), ('covMaterniso', None), ('covMaterniso', 3), ('covPeriodic', None),
         ('covRQiso', None), ('covNoise', None), ('covUnknown', None)]
COMPONENT_LISTS = [list(c) for n in range(1, 5) for c in itertools.product(_POOL, repeat=n)]


def test_gp_layout_matches_reference_walkers():
    tts = [np.arange(1, 81, dtype=float) * 0.7,
           np.cumsum(np.random.default_rng(3).uniform(0.2, 2.0, 120))]
    for comps in COMPONENT_LISTS:
        layout = mf._gp_layout(comps)
        assert layout['alpha'] == _ref_gp_alpha_pos(comps), comps
        assert layout['amp'] == _ref_gp_amp_pos(comps), comps
        assert layout['shape'] == _ref_gp_shape_pos(comps), comps
        assert layout['len'] == _ref_gp_len_pos(comps), comps
        assert layout['noise'] == _ref_gp_noise_pos(comps), comps
        assert mf._gp_noise_pos(comps) == _ref_gp_noise_pos(comps), comps
        recognized = all(name != 'covUnknown' for name, _ in comps)
        for tt in tts:
            new_b, ref_b = mf._gp_shape_bounds(comps, tt), _ref_gp_shape_bounds(comps, tt)
            assert list(new_b) == list(ref_b), comps
            np.testing.assert_allclose(list(new_b.values()), list(ref_b.values()), rtol=RTOL,
                                       atol=0, equal_nan=True, err_msg=str(comps))
            if recognized:
                np.testing.assert_allclose(mf._gp_init_hyp(comps, tt), _ref_gp_init_hyp(comps, tt),
                                           rtol=RTOL, atol=0, err_msg=str(comps))
            else:
                # The old _gp_init_hyp raised KeyError on an unrecognized name (its n_cov lookup);
                # unreachable, since parse_cov rejects such names before.
                with pytest.raises(KeyError):
                    _ref_gp_init_hyp(comps, tt)


def _series():
    rng = np.random.default_rng(7)
    wn = rng.standard_normal(300)
    e = rng.standard_normal(300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    return {k: (y - y.mean()) / y.std(ddof=1) for k, y in (('wn', wn), ('ar1', ar))}


SERIES = _series()

# On branch fix-gp-unidentified-amplitudes (#67): where the fitted model does not beat white noise
# (both ``wn`` cases and ``gp_fit_across`` on ``ar1``, whose training points are too sparse for the
# AR(1) correlation), the signal amplitude is NaN, the noise is the white-noise standard deviation
# and nlml is that of the white-noise model, by design; those pins were moved to the new values
# (main's: ar1 nlml 1.300436272482262, logh2 -0.08815140550692771, logh3 -0.5452414046928566;
# wn logh3 -0.35719140570889624; covRQiso logh2 -1.9962119439868855, logh4 -0.06755834772024763,
# nlml 1.3603913867845026). The prediction outputs (stde, stdmu, meanS) are unchanged.
CASES = [
    ('ar1', lambda y: mf.gp_fit_across(y, 'covSEiso_covNoise'),
     {'stde': 0.9177860630020508, 'meanS': 0.723678877103032, 'nlml': 1.4202118466028613,
      'logh2': np.nan, 'logh3': 0.053576715480535816}),
    ('wn', lambda y: mf.gp_fit_across(y, 'covSEiso_covNoise'),
     {'stde': 1.0226552606920438, 'stdmu': 0.018258212994875116, 'logh3': -0.28804460879053595}),
    ('ar1', lambda y: mf.gp_hyperparameters(y, 'covSEiso_covNoise', 1, 200, 'first'),
     {'logh1': 0.6171109040249246, 'logh2': -0.16161561102675429, 'logh3': -0.8616060751511386,
      'nlml': 1.029397269591294, 'meanS': 0.4989946447392767}),
    ('ar1', lambda y: mf.gp_hyperparameters(y, 'covMaterniso3_covNoise', 1, 200, 'first'),
     {'logh1': 0.8960543012209071, 'logh2': -0.11666246313149084, 'logh3': -0.9909786549775478,
      'stde': 0.2548620189842625}),
    ('wn', lambda y: mf.gp_hyperparameters(y, 'covRQiso_covNoise', 1, 200, 'first'),
     {'logh1': 0.0, 'logh2': np.nan, 'logh3': np.nan, 'logh4': -0.05738379311870417,
      'nlml': 1.3615547400859684}),
    ('wn', lambda y: mf.gp_hyperparameters(y, 'covSEiso_covPeriodic_covNoise', 1, 100, 'first'),
     {'logh1': 0.0, 'logh2': np.nan, 'logh3': np.nan, 'logh4': np.nan, 'logh5': np.nan,
      'logh6': -0.05288807443961939, 'nlml': 1.3660504587650533,
     # On main, logh2, logh5, logh6 and nlml (-1.4736004215830034, -0.4964149089149668,
     # -0.36405912244996264, 1.3378221522489182) were not pinned: the split of the variance between
     # the two components was unidentified on white noise (flat marginal likelihood), so the
     # optimizer stopped at a different point per platform (logh5 drifted by up to a factor of 8
     # between CI runners). On branch fix-gp-unidentified-amplitudes (#67) the amplitudes are NaN
     # and the noise and nlml are those of the white-noise model, functions of the data alone.
     }),
]


@pytest.mark.parametrize('series, fn, expected', CASES)
def test_gp_outputs_pinned(series, fn, expected):
    out = fn(SERIES[series])
    for k, v in expected.items():
        np.testing.assert_allclose(out[k], v, rtol=RTOL_GP, atol=0, equal_nan=True, err_msg=k)


def test_gp_layout_mixed_components():
    comps = [('covPeriodic', None), ('covRQiso', None), ('covMaterniso', 5), ('covNoise', None),
             ('covSEiso', None)]
    layout = mf._gp_layout(comps)
    assert layout['alpha'] == [5]
    assert layout['amp'] == [2, 4, 7, 10]
    assert layout['shape'] == [0, 1, 3, 5, 6, 9]
    assert layout['len'] == [3, 6, 9]
    assert layout['noise'] == [8]
    tt = np.arange(1, 81, dtype=float) * 0.7
    bounds = mf._gp_shape_bounds(comps, tt)
    assert list(bounds) == [0, 1, 3, 5, 6, 9]
    assert np.isnan([bounds[0], bounds[1], bounds[5]]).all()
    np.testing.assert_allclose([bounds[3], bounds[6], bounds[9]], -0.3566749439387323, rtol=RTOL, atol=0)
    comps2 = [('covNoise', None), ('covRQiso', None), ('covNoise', None), ('covPeriodic', None)]
    np.testing.assert_allclose(
        mf._gp_init_hyp(comps2, tt),
        [-2.3025850929940455, -0.3566749439387323, 0.0, 0.0, -2.3025850929940455,
         -0.3566749439387323, 1.7101878155342434, 0.0, -2.3025850929940455], rtol=RTOL, atol=0)
