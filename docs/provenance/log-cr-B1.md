# cr-B1: ARFIT replacement (stepwise AR fit, residual test, eigenmodes)

New module `pyhctsa/operations/_ar_stepwise.py` (`_arfit`, `_arfit_residuals`, `_arfit_modes`); the stubs in
`operations/model_fit.py` are removed and the names imported from the new module (callers unchanged).

## Sources used
- `_arfit`: Neumaier & Schneider (2001) sec. 3.1-3.2 as set out in specs/thirdparty/arfit.md sec. 4.1-4.2
  (regularized QR of [K; sqrt(delta) D], delta = (q^2+q+1) eps; criteria from trailing entries of the last
  column of R; one factorization for all orders and for the final estimate).
- `_arfit_residuals`: Li & McLeod (1981) modified portmanteau statistic, spec sec. 4.3; chi2 survival from scipy.
- `_arfit_modes`: Neumaier & Schneider sec. 2.2, 4 and Appendix, spec sec. 4.4 (companion matrix via
  np.linalg.eig, unit-norm eigenvectors, first-order error propagation, Student-t margins). The eigenvector
  phase rotation is not applied (only the norm affects any output).
- Libraries: numpy (qr, solve, inv, eig, sliding_window_view), scipy.stats (chi2, t).
- Fixture `thirdparty_blackbox.json` (arfit, MF_arfit) used for comparison only.

## Agreement with the hctsa fixtures
All 22 `arfit` rows (A, C, sbc, fpe, dof, Uinv, siglev, res, Aerr, per/tau with margins, exctn, lambda) and all
5 `MF_arfit` series (`ar_fit(y,1,8,'sbc')`, all keys) pass. Worst deviations: A 3e-16, sbc 2e-15, Uinv 9e-18,
residuals 6e-16, siglev relative 3e-8 (chi2 tail vs. fixture, small values), eigenmode quantities within test
tolerances 1e-6 to 1e-8. `short12` raises ValueError('Time series (N = 12) too short.'); `const50` is finite with
A = 1/3 and `ar_fit` returns NaN.

## Tests
`tests/native/test_thirdparty_B1.py` (30 tests: fixtures, argument errors, constant/short series, AR(2)
recovery within 3 SE, intercept model, residual definition and whiteness test, eigenvalue/period/damping
identities, real-eigenvalue period conventions, first-minimizer rule). Also pass: the 7 previously failing
arfit-related tests in lic-stub-failing-tests.txt (test_p2_c arfit18 x6, test_wp4_sysid n4sid,
test_fit_subsegments_rand_draws).

## Provenance statement
No third-party source (ARFIT or any other toolbox code), no quarantined material, no other worktree or branch,
no transcripts and no earlier pyhctsa code was consulted. Only the spec, the cited papers' content as relayed in
the spec, the fixtures and numpy/scipy documentation were used.
