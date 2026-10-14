# pyhctsa validation harness

This directory contains the parity-testing framework used to validate `pyhctsa`
against the original MATLAB [`hctsa`](https://github.com/benfulcher/hctsa)
implementation. Each Python feature is run against its MATLAB counterpart on a
common reference corpus, and the per-feature Pearson correlation is recorded.

The inclusion criterion for a ported feature is a Pearson **r ≥ 0.9** against the
MATLAB reference. Features that fall below this threshold are retained only for
documented reasons; see [`lessthan0p9.csv`](./lessthan0p9.csv) and the notes at
the end of this file.

## Contents

| File | Description |
| --- | --- |
| `run_validation.py` | Entry point: runs the harness and writes the results CSV. |
| `validate_deterministic.yaml` | Per-feature test configuration (signatures, arg maps, test cases) for deterministic functions. |
| `validate_stochastic.yaml` | Per-feature test configuration for stochastic functions. |
| `validation_results_HCTSA.csv` | Full precomputed results (correlation per feature). |
| `lessthan0p9.csv` | Subset of features below the r ≥ 0.9 threshold, with rationale. |


## Inspecting the results without MATLAB

**You do not need MATLAB to verify the validation claim.** The complete
precomputed outputs are committed to this directory:

- [`validation_results_HCTSA.csv`](./validation_results_HCTSA.csv) — the
  per-feature correlations underpinning the r ≥ 0.9 claim.
- [`lessthan0p9.csv`](./lessthan0p9.csv) — the features that did not meet the
  threshold, with the reason each was kept.

These can be read, audited, or re-analysed directly. MATLAB is only required to
**regenerate** the reference vectors from scratch (see below).

## Requirements (to re-run the harness)

Re-running the harness calls the original MATLAB `hctsa` implementation directly,
so a local MATLAB installation is required in addition to the Python
dependencies:

1. **MATLAB** (a licensed local installation). The harness was developed and run
   against MATLAB R20XX; any reasonably recent release should work.
2. **The MATLAB Engine API for Python**, which lets Python call MATLAB
   functions. Install it from your MATLAB installation:

   ```bash
   cd "matlabroot/extern/engines/python"
   python -m pip install .
   ```

   Replace `matlabroot` with the output of `matlabroot` in your MATLAB console.
   See the [official instructions](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
   for version-compatibility details (the Engine API requires a Python version
   supported by your MATLAB release).
3. **The `hctsa` repository on the MATLAB path.** Clone
   [`hctsa`](https://github.com/benfulcher/hctsa) and ensure it (and its
   subdirectories) are added to the MATLAB path, e.g. via `startup.m` or:

   ```matlab
   addpath(genpath('/path/to/hctsa'))
   ```

## Running the harness

With the requirements above in place, you can run the deterministic validation using:

```bash
python run_validation.py --config validate_deterministic.yaml --dataset e1000 --mode deterministic --hctsa-path <path> --jidt-path <path> --max-workers 6
```
It is recommended to set the max workers to no more than the number of physical cores on your computer.

Similarly, the stochastic validation can be run using:
```bash
python run_validation.py --config validate_stochastic.yaml --dataset e1000 --mode stochastic --hctsa-path <path> --jidt-path <path> --max-workers 6
```

The harness spins up one isolated MATLAB engine per worker process (via
`ProcessPoolExecutor`) and evaluates every configured feature on both sides,
writing the per-feature correlations to `validation_results_HCTSA.csv`.


## Regenerating the MATLAB reference

If you change a Python implementation, or fix a discrepancy where the MATLAB
behaviour is itself being corrected, regenerate the MATLAB reference outputs
before re-running the comparison so that the harness is comparing against
up-to-date ground truth. This step requires MATLAB and the `hctsa` path as above.

## Stochastic features and the shared-noise path

Several operations are stochastic (e.g. `CO_AddNoise`, `SD_SurrogateTest`,
`FC_Surprise`). For these, MATLAB's and NumPy's random number streams differ even
under a matched seed, so a direct per-series comparison measures RNG divergence
rather than port fidelity.

To validate these faithfully, the harness includes a **shared-noise parity path**:
it reproduces the exact random draw MATLAB makes internally and injects the same
values into the Python implementation. Under matched randomness these features
collapse to r ≈ 1.0, isolating any residual difference to the algorithm rather
than the RNG. This is a stronger statement than the cross-corpus r ≥ 0.9 gate.

## Features below the r ≥ 0.9 threshold

A small subset of features (~117) sit below the threshold and are documented in
[`lessthan0p9.csv`](./lessthan0p9.csv). They fall into a few well-understood
categories: realization dependence under independent RNG draws; ill-posed or
near-singular numerics (e.g. high-lag partial autocorrelation, chaotic
integration); third-party wrapper behaviour (e.g. `hmmlearn`, `statsmodels`); and
unavoidable KDE-bandwidth differences between SciPy and MATLAB.