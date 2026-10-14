# Third-party licenses and acknowledgements

*pyhctsa* is licensed under the [GNU General Public License v3.0 or later](LICENSE) (SPDX: `GPL-3.0-or-later`).
This file lists the third-party code that is bundled with or adapted into *pyhctsa*, the licence of each component,
the methods that were implemented from published descriptions, and the libraries that *pyhctsa* uses as dependencies
(section 4.2).
Each component keeps its own copyright notice. Because every licence below is compatible with the GPL, the combined
work is distributed under GPL-3.0-or-later. Third-party source files live in
[`pyhctsa/toolboxes/<name>/`](pyhctsa/toolboxes), each directory with its own `LICENSE`/`COPYING` and `NOTICE` file
(these are shipped in the sdist and wheel). We thank all of the authors.

## 1. The *hctsa* code base and the author's own helpers

*pyhctsa* is a Python implementation of the feature library of [*hctsa*](https://github.com/benfulcher/hctsa)
(Ben D. Fulcher). The *hctsa* feature code (the `Operations` and the GPL `PeripheryFunctions`) is
GPL-3.0-or-later, Copyright (C) Ben D. Fulcher. The `MD_*` (medical/HRV) functions are Copyright (C) 2016 Max A. Little
and Ben D. Fulcher, GPL-3.0-or-later; `MD_hrv_classic` is heavily derived from code provided by Max A. Little.

A few *hctsa* helper functions were released under CC BY-NC-SA 4.0 in *hctsa*. They were written solely by Ben D.
Fulcher, who has relicensed their Python translations in *pyhctsa* under GPL-3.0-or-later (2026-10-08):

- `BF_RandomSeed` (`bf_random_seed`), `BF_SpreadPerm` (`bf_spread_perm`), `BF_SimpleBinner` (`simple_binner`),
  `BF_Whiten` (`_whiten`) and `ZG_hmm_fit` (the fitting wrapper around the hidden Markov model).

The relicensing covers only those translations. It does not change the licence of the *hctsa* originals.

## 2. Bundled third-party code

| Component | Copyright / authors | Licence (SPDX) | Location in pyhctsa |
|---|---|---|---|
| [TISEAN 3.0.1](https://www.pks.mpg.de/tisean/) | (c) 1998-2007 R. Hegger, H. Kantz, T. Schreiber | GPL-2.0-or-later | `pyhctsa/toolboxes/Tisean_3_0_1/` |
| [GPML 4.2](http://gaussianprocess.org/gpml/code/matlab/doc/) | (c) 2005-2018 C. E. Rasmussen & H. Nickisch | BSD-2-Clause | `pyhctsa/toolboxes/gpml/` |
| [catch22](https://github.com/DynamicsAndNeuralSystems/catch22) | (C) 2018- C. H. Lubba, B. D. Fulcher and contributors | GPL-3.0-only (catch22 repository); used here under GPL-3.0-or-later terms | `pyhctsa/toolboxes/c22/` |
| SPLINEFIT (inside catch22 `splinefit.c`) | (c) 2010 J. Lundgren | BSD-2-Clause | `pyhctsa/toolboxes/c22/` |
| [Delay vector variance toolbox](https://www.commsp.ee.ic.ac.uk/~mandic/dvv.htm) | (c) 2008 D. P. Mandic (with T. Gautama) | GPL-2.0-or-later | `pyhctsa/toolboxes/DVV/` |
| [PhysioNet `sampen`](https://physionet.org/content/sampen/1.0.0/) | (C) 2002-2004 D. Lake; revisions G. Moody | GPL-2.0-or-later | `pyhctsa/toolboxes/physionet/` |
| [PhysioNet MMA](https://physionet.org/content/mma/1.0.0/) | (C) 2014 J. Gieraltowski | GPL-2.0-or-later | `pyhctsa/toolboxes/physionet/` |
| Steps and bumps toolkit (`l1pwc`, after l1_tf) | (c) 2010 M. A. Little; after S.-J. Kim, K. Koh, S. Boyd, D. Gorinevsky | GPL-2.0-or-later | `pyhctsa/toolboxes/Max_Little/` |
| RAD (rescaled auto-density), [Criticality](https://github.com/brendanjohnharris/Criticality) | (c) 2021 B. Harris and contributors | MIT | `pyhctsa/toolboxes/RAD/` |
| [PyWavelets](https://github.com/PyWavelets/pywt) (cut-down `cwt`) | (c) 2006-2012 F. Wasilewski; (c) 2012- The PyWavelets Developers | MIT | `pyhctsa/toolboxes/pywavelets/` |

Notes per component.

- **TISEAN** (d2, c2g, c2t, poincare, false_nearest, boxcount, lyap_r, lyap_spec, c1/d1/c2d, nrlazy, nstat_z):
  made re-entrant, called in-process from NumPy arrays; the Fortran programs were translated to Python. The copyright
  to each program is held by the person(s) named in its source code. Hegger, Kantz & Schreiber, *Chaos* 9, 413 (1999).
  Licence text: `pyhctsa/toolboxes/Tisean_3_0_1/COPYING` (GPL-2.0).
- **GPML**: Python ports of the Gaussian-likelihood inference, the covariance functions and Rasmussen's `minimize`.
  The Laplace inference path and `brentmin` are not included. Rasmussen & Nickisch, *J. Mach. Learn. Res.* 11, 3011
  (2010). Licence text reproduced in section 5.1.
- **catch22 / SPLINEFIT**: C sources modified for pyhctsa. The "All rights reserved" lines in the catch22 headers are
  Xcode template text and do not restrict the GPL-3.0 grant. Lubba et al., *Data Min. Knowl. Discov.* 33, 1821 (2019).
  SPLINEFIT licence text reproduced in section 5.2.
- **DVV**: Python port of `DVV_dvv.m` and `DVV_surrogate.m`. Gautama, Mandic & Van Hulle, *Physica D* 190, 167 (2004).
  Licence text: `pyhctsa/toolboxes/DVV/COPYING` (GPL-2.0).
- **PhysioNet sampen and MMA**: the `sampen.c` header is retained; the PhysioNet pages additionally cite the Open
  Data Commons Attribution License v1.0, satisfied by this attribution. Richman & Moorman, *Am. J. Physiol.* 278,
  H2039 (2000); Gieraltówski, Zebrowski & Baranowski, *Phys. Rev. E* 85, 021915 (2012). Licence text:
  `pyhctsa/toolboxes/physionet/COPYING` (GPL-2.0).
- **l1pwc**: Little & Jones, *ICASSP* (2010); Kim, Koh, Boyd & Gorinevsky, *SIAM Rev.* 51, 339 (2009). Licence text:
  `pyhctsa/toolboxes/Max_Little/COPYING` (GPL-2.0).
- **RAD**: Harris et al., *Phys. Rev. X* 14, 031021 (2024). Licence text reproduced in section 5.3.
- **PyWavelets**: Lee et al., *J. Open Source Softw.* 4, 1237 (2019). Licence text reproduced in section 5.4.

GPL-2.0-or-later components are distributed here under their "or any later version" option, as part of a work
licensed GPL-3.0-or-later; the GPL-2.0 text is included for fidelity.

## 3. Code credited to its authors

These components are used with credit. No unlicensed third-party source is included.

| Component | Credit |
|---|---|
| `falling_sticks` (CO_FallingSticks) | Adapted from the Python FALLstick implementation written by Eugene Chon, a student of B. D. Fulcher, for *hctsa*. |
| `MD_*` heart-rate-variability measures | (C) 2016 Max A. Little and Ben D. Fulcher. |
| Permutation entropy variant (`permEnLE`) | Bruce Land and Damian Elias, Cornell University (https://people.ece.cornell.edu/land/PROJECTS/Complexity/). The variant is a one-line formula. |
| Visibility-graph measures | Lacasa et al., *PNAS* 105, 4972 (2008); Luque et al., *Phys. Rev. E* 80, 046103 (2009). The degree algorithms in *pyhctsa* are original. |
| KSG and Gaussian mutual-information estimators (`pyhctsa/ksg.py`) | Kraskov, Stoegbauer & Grassberger, *Phys. Rev. E* 69, 066138 (2004). The Java Information Dynamics Toolkit (J. T. Lizier, *Front. Robot. AI* 1, 11 (2014)) is the implementation *hctsa* uses and the reference for the method's conventions; the estimators in *pyhctsa* are written from the published equations and include no JIDT code. |
| TSTOOL methods (`CO_TC3`, `CO_trev`, `NL_DelayTime`, `NL_FractalDimensions`, `NL_ReturnTime`) | C. Merkwirth, U. Parlitz, I. Wedekind, D. Engster and W. Lauterborn (TSTOOL), via the *hctsa* reimplementations. The expressions are re-implemented; no TSTOOL source is included. |

## 4. Implemented from published descriptions (clean-room), and dependency credits

### 4.1 Clean-room re-implementations (implemented from the published method)

The following were written by implementers who worked only from the published method and a written specification, and
who did not read the original source ("implemented from \<paper\>"). The provenance logs are in
[`docs/provenance/`](docs/provenance).

| Method | Implemented from | Original software credited |
|---|---|---|
| ARFIT stepwise autoregressive fitting | Neumaier & Schneider, *ACM Trans. Math. Softw.* 27, 27 (2001); Schneider & Neumaier, *ACM Trans. Math. Softw.* 27, 58 (2001) | ARFIT, T. Schneider and A. Neumaier |
| Lempel-Ziv complexity, Shannon block entropy, nonlinear prediction error | Kaspar & Schuster, *Phys. Rev. A* 36, 842 (1987); Small, *Applied Nonlinear Time Series Analysis* (2005) | M. Small's time-series code |
| Theiler, crinkle and time-reversal statistics | Theiler et al., *Physica D* 58, 77 (1992); Kaplan & Glass, *Understanding Nonlinear Dynamics* (1995); Ramsey & Rothman, *J. Money Credit Bank.* 28, 1 (1996) | D. Kaplan. His `DK_*` files in *hctsa* carry a GPL-3 notice; the statistics here are re-implemented from his published methods, and none of his source is included |
| nsAMDF | Ozkurt et al., *NeuroImage* 223, 117356 (2020) | T. E. Ozkurt |
| RPDE and close returns | Little et al., *BioMed. Eng. OnLine* 6, 23 (2007) | M. A. Little |
| Fractal-dimension error function | van de Water & Schram, *Phys. Rev. A* 37, 3118 (1988) | TSTOOL |
| Approximate entropy (ApEn) | Pincus, *Proc. Natl. Acad. Sci. USA* 88, 2297 (1991) | antropy (R. Vallat), whose kernel earlier pyhctsa versions used |
| Exponential smoothing (`exp_smoothing`, `MF_ExpSmoothing`) | Brown, *Smoothing, Forecasting and Prediction of Discrete Time Series* (1963); Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, sec. 8.1 | S. Arora, whose design of the feature in *hctsa* (a training/held-out split with a smoothing parameter chosen on the training part, followed by residual analysis) is credited as a method; the smoothing itself is computed with statsmodels |
| Signal-processing helpers (`buffer`, `wavedec`/`wrcoef`, `medfilt1`) | MATLAB function specifications | |

### 4.2 Libraries used as dependencies

No code from these is copied; they are called through their public interfaces and are installed separately
(licences as declared in each package's metadata; all are GPL-compatible).

| Library | Used for | Licence |
|---|---|---|
| [NumPy](https://numpy.org) | arrays and numerical routines | BSD-3-Clause |
| [SciPy](https://scipy.org) | signal processing, statistics, optimization, linear algebra | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | tabular results and datasets | BSD-3-Clause |
| [PyYAML](https://pyyaml.org) | reading the feature configuration files | MIT |
| [statsmodels](https://www.statsmodels.org) | time-series models, including exponential smoothing (the design of *hctsa*'s `MF_ExpSmoothing` is credited to S. Arora above) | BSD-3-Clause |
| [numba](https://numba.pydata.org) | compiled inner loops | BSD-2-Clause |
| [scikit-learn](https://scikit-learn.org) | PCA, Gaussian mixtures, neighbor searches and silhouette scores | BSD-3-Clause |
| [arch](https://github.com/bashtage/arch) | GARCH models, Phillips-Perron and variance-ratio tests | NCSA (University of Illinois/NCSA Open Source License) |
| [PyWavelets](https://pywavelets.readthedocs.io) | discrete wavelet transforms (a cut-down `cwt` is also bundled, section 2) | MIT |
| [rich](https://github.com/Textualize/rich) | progress bars | MIT |
| [pathos](https://github.com/uqfoundation/pathos) | process-based parallel computation | BSD-3-Clause |
| [lmfit](https://lmfit.github.io/lmfit-py/) | sinusoid model fitting | BSD-3-Clause |
| [hmmlearn](https://github.com/hmmlearn/hmmlearn) | Gaussian hidden Markov model (replaces Z. Ghahramani's HMM code used in *hctsa*) | BSD-3-Clause |
| [ripser](https://github.com/scikit-tda/ripser.py) (optional, `tda` extra) | persistent homology | MIT |

## 5. Licence texts

GPL-3.0-or-later (pyhctsa itself, catch22): [`LICENSE`](LICENSE),
`pyhctsa/toolboxes/c22/LICENSE`. GPL-2.0 (TISEAN, DVV, PhysioNet, Max Little's
l1pwc): `COPYING` in each toolbox directory.

### 5.1 GPML (BSD-2-Clause, "FreeBSD License")

(`pyhctsa/toolboxes/gpml/LICENSE`, reproduced from http://gaussianprocess.org/gpml/code/matlab/Copyright)

```text
 GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 4.2
    for GNU Octave 3.2.x and Matlab 7.x

The code is released under the FreeBSD License.

Copyright (c) 2005-2018 Carl Edward Rasmussen & Hannes Nickisch. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY CARL EDWARD RASMUSSEN & HANNES NICKISCH ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CARL EDWARD RASMUSSEN & HANNES NICKISCH OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of Carl Edward Rasmussen & Hannes Nickisch.

The code and associated documentation is available from http://gaussianprocess.org/gpml/code.
```

### 5.2 SPLINEFIT (BSD-2-Clause)

(Text as reproduced in Jonas Lundgren's `__splinefit__.m` shipped with GNU Octave.)

```text
Copyright (c) 2010, Jonas Lundgren
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### 5.3 RAD / Criticality (MIT)

(`pyhctsa/toolboxes/RAD/LICENSE`, from https://github.com/brendanjohnharris/Criticality)

```text
MIT License

Copyright (c) 2021 Brendan Harris <brendanjohnharris@gmail.com> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 5.4 PyWavelets (MIT)

(`pyhctsa/toolboxes/pywavelets/LICENSE`, from https://github.com/PyWavelets/pywt)

```text
Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
Copyright (c) 2012-     The PyWavelets Developers <https://github.com/PyWavelets/pywt>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
