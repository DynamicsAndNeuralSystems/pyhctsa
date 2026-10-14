Acknowledgements and third-party code
=====================================

``pyhctsa`` is licensed under the `GNU General Public License v3.0 or later <https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/LICENSE>`_.
It is a Python implementation of the feature library of `hctsa <https://github.com/benfulcher/hctsa>`_ (Ben D. Fulcher).
The *hctsa* feature code is GPL-3.0-or-later; the few *hctsa* helper functions that were released under CC BY-NC-SA 4.0
were written solely by Ben D. Fulcher, who has relicensed their Python translations in ``pyhctsa`` under
GPL-3.0-or-later.

Bundled third-party code
------------------------

Some features use code written by others. Third-party code is kept in
`pyhctsa/toolboxes/<name>/ <https://github.com/DynamicsAndNeuralSystems/pyhctsa/tree/main/pyhctsa/toolboxes>`_, each
directory with its original copyright notice and license; full texts are in
`THIRD_PARTY_LICENSES.md <https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/THIRD_PARTY_LICENSES.md>`_.
We thank the authors.

.. list-table::
   :header-rows: 1
   :widths: 24 20 14 16 26

   * - Component
     - Authors
     - License
     - Location in pyhctsa
     - Reference
   * - TISEAN 3.0.1 (d2, c2g/c2t, poincare, false_nearest, boxcount, lyap_r, lyap_spec, c1/d1/c2d, nrlazy, nstat_z)
     - R. Hegger, H. Kantz, T. Schreiber
     - GPL-2.0-or-later
     - ``toolboxes/Tisean_3_0_1/``
     - Hegger, Kantz & Schreiber, Chaos 9, 413 (1999)
   * - GPML (Gaussian process inference, covariance functions, ``minimize``)
     - C. E. Rasmussen, H. Nickisch
     - BSD-2-Clause (FreeBSD)
     - ``toolboxes/gpml/``
     - Rasmussen & Nickisch, JMLR 11, 3011 (2010)
   * - catch22 (PD_PeriodicityWang, splinefit)
     - C. H. Lubba, B. D. Fulcher et al.; splinefit after J. Lundgren
     - GPL-3.0; splinefit BSD-2-Clause
     - ``toolboxes/c22/``
     - Lubba et al., Data Min. Knowl. Disc. 33, 1821 (2019)
   * - Delay vector variance (DVV) toolbox
     - D. P. Mandic, T. Gautama
     - GPL-2.0-or-later
     - ``toolboxes/DVV/``
     - Gautama, Mandic & Van Hulle, Physica D 190, 167 (2004)
   * - Sample entropy (sampen.c)
     - D. Lake, J. R. Moorman, G. Moody (PhysioNet)
     - GPL-2.0-or-later
     - ``toolboxes/physionet/``
     - Richman & Moorman, Am. J. Physiol. 278, H2039 (2000)
   * - Multiscale multifractal analysis (MMA)
     - J. Gieraltowski (PhysioNet)
     - GPL-2.0-or-later
     - ``toolboxes/physionet/``
     - Gieraltowski, Zebrowski & Baranowski, Phys. Rev. E 85, 021915 (2012)
   * - Steps and bumps toolkit (l1pwc)
     - M. A. Little; after S.-J. Kim, K. Koh, S. Boyd, D. Gorinevsky (l1_tf)
     - GPL-2.0-or-later
     - ``toolboxes/Max_Little/``
     - Little & Jones, ICASSP (2010)
   * - Rescaled auto-density (RAD)
     - B. Harris
     - MIT
     - ``toolboxes/RAD/``
     - Harris et al., Phys. Rev. X 14, 031021 (2024)
   * - Continuous wavelet transform (cut-down ``pywt.cwt``)
     - The PyWavelets developers
     - MIT
     - ``toolboxes/pywavelets/``
     - Lee et al., JOSS 4, 1237 (2019)

Further credits
---------------

- The falling sticks statistic (``CO_FallingSticks``) is adapted from the Python FALLstick implementation written by
  Eugene Chon, a student of B. D. Fulcher, for *hctsa*.
- The heart-rate-variability measures (``MD_*``) are Copyright (C) 2016 Max A. Little and Ben D. Fulcher.
- The permutation-entropy variant is that of B. Land and D. Elias (Cornell University).
- The visibility-graph measures follow Lacasa et al., PNAS 105, 4972 (2008) and Luque et al., Phys. Rev. E 80, 046103
  (2009).
- The Kraskov-Stoegbauer-Grassberger and Gaussian mutual-information estimators (``pyhctsa/ksg.py``) are written from the
  published method (Kraskov et al., Phys. Rev. E 69, 066138, 2004); the Java Information Dynamics Toolkit (J. T. Lizier,
  Front. Robot. AI 1, 11, 2014) is credited as the method reference used by *hctsa*. No JIDT code is included.
- Several methods follow TSTOOL (C. Merkwirth, U. Parlitz, I. Wedekind, D. Engster, W. Lauterborn): the tc3, trev,
  delay-time, fractal-dimension and return-time measures.

Clean-room re-implementations
-----------------------------

The following methods are implemented in ``pyhctsa`` from their published descriptions, without use of the original
source code, and we credit the authors whose software defined them in *hctsa*: ARFIT stepwise autoregressive fitting
(T. Schneider, A. Neumaier; Neumaier & Schneider, ACM TOMS 27, 27 (2001)); Lempel-Ziv complexity, Shannon block
entropy, nonlinear prediction error as in M. Small's toolbox (Kaspar & Schuster, Phys. Rev. A 36, 842 (1987); Small,
*Applied Nonlinear Time Series Analysis*, 2005); the Theiler, crinkle and time-reversal statistics (D. Kaplan; Theiler
et al., Physica D 58, 77 (1992); Kaplan & Glass, *Understanding Nonlinear Dynamics*, 1995); nsAMDF (T. E. Özkurt;
NeuroImage 223, 117356 (2020)); RPDE and close returns (M. A. Little; Little et al., BioMed. Eng. OnLine 6, 23 (2007));
and the fractal-dimension error function (van de Water & Schram, Phys. Rev. A 37, 3118 (1988)). Approximate entropy is
implemented from Pincus, PNAS 88, 2297 (1991). Clean-room provenance logs are in
`docs/provenance/ <https://github.com/DynamicsAndNeuralSystems/pyhctsa/tree/main/docs/provenance>`_.

Daniel Kaplan's statistics are re-implemented from his published methods; his ``DK_*`` files in *hctsa* are GPL-3 and
none of that source is included. The exponential-smoothing feature (``MF_ExpSmoothing``) follows the design of the code
provided by S. Arora for *hctsa* (training/held-out split, a smoothing parameter chosen on the training part, residual
analysis); the smoothing itself is computed with statsmodels.

Dependencies
------------

Runtime dependencies, called through their public interfaces (no code copied; all GPL-compatible):
NumPy, SciPy, pandas, statsmodels and scikit-learn (BSD-3-Clause), numba (BSD-2-Clause), PyYAML, PyWavelets and rich
(MIT), pathos, lmfit and `hmmlearn <https://github.com/hmmlearn/hmmlearn>`_ (BSD-3-Clause; hmmlearn replaces
Z. Ghahramani's HMM code used in *hctsa*), `arch <https://github.com/bashtage/arch>`_ (NCSA), and, optionally,
`ripser <https://github.com/scikit-tda/ripser.py>`_ (MIT). See
`THIRD_PARTY_LICENSES.md <https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/THIRD_PARTY_LICENSES.md>`_,
section 4.2.
