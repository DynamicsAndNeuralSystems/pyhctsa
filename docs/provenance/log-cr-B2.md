# Clean-room log: work package B2 (worktree SP/cr-B2, branch cr-B2), 2026-10-09

Specs: SP/cleanroom/specs/thirdparty/small_lz.md, small_shannon.md, small_nlpe.md (+ README, thirdparty_blackbox.json,
gen/thirdparty_inputs.json). Orchestrator decision applied: LZ complexity uses the published phrase count c (not c - 1).

## Renames (callers updated)
| old | new | module |
|---|---|---|
| `entropy._lz_complexity` | `_lempel_ziv_phrase_count` | `operations/_symbolic_complexity.py` |
| `entropy._symbolise_lz` | `_equiprobable_symbols` | same |
| `extreme_events._lz_complexity_bs` | `_normalized_lz_complexity` (takes 0-based labels; k = max + 1) | same |
| `entropy._shannon_block_entropy` | unchanged name, moved | same |
| `nonlinearity._ms_nlpe` | `_nn_prediction_residuals` | `operations/_nlpe.py` |
| `nonlinearity._ms_embed` | deleted (direct indexing) | |

`lempel_ziv_complexity` now calls `_normalized_lz_complexity(symbols - 1)`; `extreme_event_order` calls
`_normalized_lz_complexity(event_type)`; `nlpe` and `surrogate_test` call `_nn_prediction_residuals`.

## Sources per function
- `_lempel_ziv_phrase_count`: Lempel & Ziv (1976) exhaustive-history definition as stated in the spec (components =
  shortest substring not copyable from the text before its last symbol; final non-exhaustive component counted). Own
  O(n^2) scan (longest copyable prefix over all earlier start positions), numba-jitted. Verified against a brute-force
  parser written in the test from the definition (all binary strings of length 8, random binary/ternary 1..60) and the
  spec's worked example (0001101001000101 -> 6).
- `_equiprobable_symbols`: rank rule in spec sec. 3 (scipy `rankdata` average ranks, floor(R n/(N+1)) + 1).
- `_normalized_lz_complexity`: Kaspar & Schuster (1987) normalization c ln N / (N ln k).
- `_shannon_block_entropy`: Shannon (1948) block entropy; binning rule exactly as spec sec. 2 (order-statistic thresholds,
  `searchsorted`); words encoded as base-n int64 (fallback `np.unique(axis=0)` for huge alphabets).
- `_nn_prediction_residuals`: zeroth-order predictor (Lorenz 1969; Farmer & Sidorowich 1987; Sugihara & May 1990),
  Theiler window (Theiler 1986), spec sec. 2. Brute force in row blocks (32 MB), `argmin` gives smallest-index ties.

## Agreement with hctsa fixtures (MATLAB R2026a)
- MS_complexitybs (9 sequences): hctsa value equals the (c - 1) form to 1e-12 for all; the library returns the c form
  (offset log N/(N log k)), as decided.
- EN_LZComplexity (7 series, n = 2..10, raw and diff): same, (c - 1) form exact to 1e-12; library is c form.
- EX_ExtremeEventOrder lzComplexity (5 series x 2 thresholds): library minus hctsa = ln N/(N ln 2) for an integer N >= 10
  (the one-component offset), i.e. identical otherwise.
- MS_shannon (10 series/hand cases, all bins/depths) and EN_Shannon wrapper outputs: exact to 1e-12.
- MS_nlpe residual vectors (4 settings): exact (atol 1e-12); NL_nlpe `msqerr` for the 5 long series: <= 1e-8.
  The full NL_nlpe feature comparison skips (residual_analysis needs ARFIT, another work package).
- Tests: tests/native/test_thirdparty_B2.py (25 pass, 1 skip) with fixture subset tests/native/fixtures_thirdparty_B2.json.
  Other failing tests in the touched areas (permutation entropy, fnn, residual_analysis) are NotImplementedError stubs
  of other work packages (antropy, ARFIT).

## Notes
- The shannon C-extension entry in `pyhctsa/_build/ext_modules.py` is now dead (toolboxes/Michael_Small was already
  deleted); removal is left to the packaging (L3) agent.
- NLPE is O(M^2 de); 'full' on very long series is slow (a KD-tree with the smallest-index tie rule would be the option).

## Statement
No third-party source (Michael Small's MS_complexity*, MS_shannon, MS_nlpe, MS_embed or any other toolbox code), no
quarantined material, no other worktree or older commit, no removed derived pyhctsa code, and no Claude transcripts were
consulted. Sources: the specs and fixtures above, the cited papers' definitions, numpy/scipy/numba documentation.
