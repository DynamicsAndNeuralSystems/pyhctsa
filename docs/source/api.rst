API Reference
=============

Core
----

.. autosummary::
   :toctree: generated/
   :recursive:

   pyhctsa.calculator.FeatureCalculator
   pyhctsa.distribute.LocalDistributor

Utilities
---------

.. autosummary::
   :toctree: generated/

   pyhctsa.utils.get_dataset
   pyhctsa.utils.z_score
   pyhctsa.utils.dict_output
   pyhctsa.utils.nan_outputs

Shared hctsa helpers
--------------------

Ports of hctsa's ``BF_*`` helper functions, shared by the methods below (the portable random generator and
its seeds, robust fits, explicit histogram bin edges and kernel densities, and the removal of points).

.. autosummary::
   :toctree: generated/

   pyhctsa.robust.bf_random
   pyhctsa.robust.bf_random_seed
   pyhctsa.robust.bf_tie_break_noise
   pyhctsa.robust.bf_remove_points
   pyhctsa.robust.bf_runs_z
   pyhctsa.robust.bf_residual_stats
   pyhctsa.robust.bf_theil_sen
   pyhctsa.robust.bf_exp_fit
   pyhctsa.robust.bf_fit_density_curve
   pyhctsa.robust.bf_gauss_mix2
   pyhctsa.robust.bf_fit_sinusoids
   pyhctsa.robust.bf_ks_density
   pyhctsa.robust.bf_hist_edges
   pyhctsa.robust.bf_quantile_edges
   pyhctsa.robust.bf_half_sample_mode

.. _tsanalysismeths:

Time-Series Analysis Method Modules
-----------------------------------

.. _changepointmeths:

Changepoint
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.changepoint.stepdetect
   pyhctsa.operations.changepoint.l1pwc_sweep_lambda
   pyhctsa.operations.changepoint.wavelet_var_chg

.. _correlationmeths:

Correlation
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   pyhctsa.operations.correlation.oversampling
   pyhctsa.operations.correlation.time_rev_kld
   pyhctsa.operations.correlation.autocorr
   pyhctsa.operations.correlation.autocorr_x2_shape
   pyhctsa.operations.correlation.add_noise
   pyhctsa.operations.correlation.theiler_q
   pyhctsa.operations.correlation.crinkle_statistic
   pyhctsa.operations.correlation.time_rev_kaplan
   pyhctsa.operations.correlation.pos_neg_asymmetry
   pyhctsa.operations.correlation.embed2_angle_tau
   pyhctsa.operations.correlation.embed2
   pyhctsa.operations.correlation.periodicity_wang
   pyhctsa.operations.correlation.compare_min_ami
   pyhctsa.operations.correlation.histogram_ami
   pyhctsa.operations.correlation.stick_angles
   pyhctsa.operations.correlation.falling_sticks
   pyhctsa.operations.correlation.nonlinear_autocorr
   pyhctsa.operations.correlation.partial_autocorr
   pyhctsa.operations.correlation.embed2_dist
   pyhctsa.operations.correlation.embed2_basic
   pyhctsa.operations.correlation.embed2_shapes
   pyhctsa.operations.correlation.fzcglscf
   pyhctsa.operations.correlation.glscf
   pyhctsa.operations.correlation.first_crossing
   pyhctsa.operations.correlation.translate_shape
   pyhctsa.operations.correlation.autocorr_shape
   pyhctsa.operations.correlation.trev
   pyhctsa.operations.correlation.tc3
   pyhctsa.operations.correlation.joint_non_gaussianity
   pyhctsa.operations.correlation.matrix_profile
   pyhctsa.operations.correlation.quantilogram
   pyhctsa.operations.correlation.remove_points

.. _criticalitymeths:

Criticality
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.criticality.rad

.. _distmeths:

Distribution
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.distribution.cumulants
   pyhctsa.operations.distribution.compare_ks_fit
   pyhctsa.operations.distribution.withinp
   pyhctsa.operations.distribution.unique
   pyhctsa.operations.distribution.spread
   pyhctsa.operations.distribution.quantile
   pyhctsa.operations.distribution.proportion_values
   pyhctsa.operations.distribution.pleft
   pyhctsa.operations.distribution.min_max
   pyhctsa.operations.distribution.mean
   pyhctsa.operations.distribution.high_low_mu
   pyhctsa.operations.distribution.fit_mle
   pyhctsa.operations.distribution.cv
   pyhctsa.operations.distribution.custom_skewness
   pyhctsa.operations.distribution.burstiness
   pyhctsa.operations.distribution.moments
   pyhctsa.operations.distribution.outlier_include
   pyhctsa.operations.distribution.outlier_test
   pyhctsa.operations.distribution.trimmed_mean
   pyhctsa.operations.distribution.histogram_asymmetry
   pyhctsa.operations.distribution.histogram_mode
   pyhctsa.operations.distribution.remove_points
   pyhctsa.operations.distribution.fit_kernel_smooth
   pyhctsa.operations.distribution.simple_fit
   pyhctsa.operations.distribution.tail_index

.. _entropymeths:

Entropy
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.entropy.shannon_entropy
   pyhctsa.operations.entropy.distribution_entropy
   pyhctsa.operations.entropy.multi_scale_entropy
   pyhctsa.operations.entropy.sample_entropy
   pyhctsa.operations.entropy.permutation_entropy
   pyhctsa.operations.entropy.rpde
   pyhctsa.operations.entropy.approximate_entropy
   pyhctsa.operations.entropy.complexity_invariant_distance
   pyhctsa.operations.entropy.lempel_ziv_complexity
   pyhctsa.operations.entropy.bubble_entropy
   pyhctsa.operations.entropy.dispersion_entropy
   pyhctsa.operations.entropy.fuzzy_entropy
   pyhctsa.operations.entropy.permutation_entropy_complexity
   pyhctsa.operations.entropy.randomize
   pyhctsa.operations.entropy.wavelet_entropy

.. _extemeeventsmeths:

Extreme Events
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.extreme_events.moving_threshold
   pyhctsa.operations.extreme_events.extreme_event_order

.. _graphmeths:

Graph
~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.graph.visibility_graph
   pyhctsa.operations.graph.ordinal_partition_network

.. _hypothesistestsmeths:

Hypothesis Tests
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.hypothesis_tests.variance_ratio_test
   pyhctsa.operations.hypothesis_tests.hypothesis_test
   pyhctsa.operations.hypothesis_tests.distribution_test
   pyhctsa.operations.hypothesis_tests.independence_tests
   pyhctsa.operations.hypothesis_tests.marginal_tests

.. _informationmeths:

Information
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.information.first_min
   pyhctsa.operations.information.first_max
   pyhctsa.operations.information.automutual_info_stats
   pyhctsa.operations.information.automutual_info
   pyhctsa.operations.information.rm_automutual_information
   pyhctsa.operations.information.multivariate_ami

.. _medicalmeths:

Medical
~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.medical.raw_hrv_meas
   pyhctsa.operations.medical.hrv_classic
   pyhctsa.operations.medical.pol_var
   pyhctsa.operations.medical.pnn
   pyhctsa.operations.medical.porta

.. _modelfitmeths:

Model Fit
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.model_fit.hmm_fit
   pyhctsa.operations.model_fit.fit_subsegments
   pyhctsa.operations.model_fit.loop_local_simple
   pyhctsa.operations.model_fit.local_simple
   pyhctsa.operations.model_fit.residual_analysis
   pyhctsa.operations.model_fit.exp_smoothing
   pyhctsa.operations.model_fit.ar_cov
   pyhctsa.operations.model_fit.ar_fit
   pyhctsa.operations.model_fit.is_seasonal
   pyhctsa.operations.model_fit.gp_fit_across
   pyhctsa.operations.model_fit.gp_local_prediction
   pyhctsa.operations.model_fit.compare_ar
   pyhctsa.operations.model_fit.compare_test_sets
   pyhctsa.operations.model_fit.garch_compare
   pyhctsa.operations.model_fit.garch_fit
   pyhctsa.operations.model_fit.gp_hyperparameters
   pyhctsa.operations.model_fit.state_space_comp_order
   pyhctsa.operations.model_fit.state_space_n4sid
   pyhctsa.operations.model_fit.armax
   pyhctsa.operations.model_fit.hmm_compare_n_states
   pyhctsa.operations.model_fit.steps_ahead

.. _nonlinearitymeths:

Nonlinearity
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.nonlinearity.nsamdf
   pyhctsa.operations.nonlinearity.nlpe
   pyhctsa.operations.nonlinearity.embed_pca
   pyhctsa.operations.nonlinearity.ssa
   pyhctsa.operations.nonlinearity.zero_one_test
   pyhctsa.operations.nonlinearity.local_density
   pyhctsa.operations.nonlinearity.tisean_d2
   pyhctsa.operations.nonlinearity.poincare_section
   pyhctsa.operations.nonlinearity.delay_time
   pyhctsa.operations.nonlinearity.box_count_entropy_rate
   pyhctsa.operations.nonlinearity.dvv
   pyhctsa.operations.nonlinearity.dimensions
   pyhctsa.operations.nonlinearity.evt_local_dim
   pyhctsa.operations.nonlinearity.embed_cluster
   pyhctsa.operations.nonlinearity.embed_kernel_pca
   pyhctsa.operations.nonlinearity.fnn
   pyhctsa.operations.nonlinearity.fractal_dimensions
   pyhctsa.operations.nonlinearity.gp_corr_sum
   pyhctsa.operations.nonlinearity.largest_lyap
   pyhctsa.operations.nonlinearity.lyap_spec
   pyhctsa.operations.nonlinearity.persistent_homology
   pyhctsa.operations.nonlinearity.rqa
   pyhctsa.operations.nonlinearity.recurrence_times
   pyhctsa.operations.nonlinearity.return_time
   pyhctsa.operations.nonlinearity.takens_estimator
   pyhctsa.operations.nonlinearity.tisean_c1

.. _physicsmeths:

Physics
~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.physics.walker
   pyhctsa.operations.physics.force_potential
   pyhctsa.operations.physics.kramers_moyal

.. _preprocessmeths:

Pre-Process
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.pre_process.preproc_compare
   pyhctsa.operations.pre_process.preproc_iterate
   pyhctsa.operations.pre_process.preproc_model_fit
   pyhctsa.operations.pre_process.preproc_schreiber_denoise

.. _scalingmeths:

Scaling
~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.scaling.fast_dfa
   pyhctsa.operations.scaling.fluctuation_analysis
   pyhctsa.operations.scaling.mma
   pyhctsa.operations.scaling.higuchi_fd
   pyhctsa.operations.scaling.mfdfa

.. _spectralmeths:

Spectral
~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.spectral.phase_amp_coupling
   pyhctsa.operations.spectral.spectral_summaries
   pyhctsa.operations.spectral.spectral_summaries_phase
   pyhctsa.operations.spectral.specparam
   pyhctsa.operations.spectral.cepstrum
   pyhctsa.operations.spectral.bicoherence
   pyhctsa.operations.spectral.envelope_stats
   pyhctsa.operations.spectral.phase_fluctuation_scaling
   pyhctsa.operations.spectral.sinusoid_fit
   pyhctsa.operations.spectral.spectral_time_freq

.. _stationaritymeths:

Stationarity
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.stationarity.local_distributions
   pyhctsa.operations.stationarity.dyn_win
   pyhctsa.operations.stationarity.moment_corr
   pyhctsa.operations.stationarity.simple_stats
   pyhctsa.operations.stationarity.local_extrema
   pyhctsa.operations.stationarity.kpss_test
   pyhctsa.operations.stationarity.range_evolve
   pyhctsa.operations.stationarity.drifting_mean
   pyhctsa.operations.stationarity.local_global
   pyhctsa.operations.stationarity.fit_polynomial
   pyhctsa.operations.stationarity.ts_length
   pyhctsa.operations.stationarity.std_nth_deriv
   pyhctsa.operations.stationarity.trend
   pyhctsa.operations.stationarity.stat_av
   pyhctsa.operations.stationarity.sliding_window
   pyhctsa.operations.stationarity.ramping_windows
   pyhctsa.operations.stationarity.slow_feature_analysis
   pyhctsa.operations.stationarity.pp_test
   pyhctsa.operations.stationarity.peak_intervals
   pyhctsa.operations.stationarity.drifting_auto_corr
   pyhctsa.operations.stationarity.drifting_mean_cusum
   pyhctsa.operations.stationarity.spread_random_local
   pyhctsa.operations.stationarity.std_nth_deriv_change
   pyhctsa.operations.stationarity.nstat_z

.. _surrogatesmeths:

Surrogates
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.surrogates.surrogate_test
   pyhctsa.operations.surrogates.surrogates

.. _symbolicmeths:

Symbolic
~~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.symbolic.surprise
   pyhctsa.operations.symbolic.motif_two
   pyhctsa.operations.symbolic.motif_three
   pyhctsa.operations.symbolic.binary_stretch
   pyhctsa.operations.symbolic.binary_stats
   pyhctsa.operations.symbolic.transition_matrix
   pyhctsa.operations.symbolic.transition_p_alphabet
   pyhctsa.operations.symbolic.coarse_grain
   pyhctsa.operations.symbolic.binary_stats_ar1

.. _waveletmeths:

Wavelet
~~~~~~~
.. autosummary::
   :toctree: generated/

   pyhctsa.operations.wavelet.wl_coeffs
   pyhctsa.operations.wavelet.detail_coeffs
   pyhctsa.operations.wavelet.cwt
   pyhctsa.operations.wavelet.dwt_coeff
   pyhctsa.operations.wavelet.scal_2_freq
   pyhctsa.operations.wavelet.wfbm
   pyhctsa.operations.wavelet.modwt_var
   pyhctsa.operations.wavelet.wpd_best_tree
