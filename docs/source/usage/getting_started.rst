📘 Getting Started
==================

Computing features
------------------

To compute features on your data, a `FeatureCalculator` object must first be instantiated using:

.. code-block:: python

   from pyhctsa.calculator import FeatureCalculator
   calc = FeatureCalculator()

By default, the `FeatureCalculator` will initialize the full feature set. 
If you would like to specify a custom feature set, you can pass the corresponding configuration .YAML file as an argument to the `FeatureCalculator`:

.. code-block:: python

   custom_calc = FeatureCalculator(config_path="subset.yaml")

``len(calc.feature_funcs)`` is the number of feature functions (parameter settings of an operation) the configuration defines: 1091 for the default feature set.
Each returns one feature or several, 6406 in all.
Once a `FeatureCalculator` has been initialized, you can call the extract method to compute time series features on either a single time-series instance or a list of multiple instances:

.. code-block:: python

    from pyhctsa.utils import get_dataset

    e1000 = get_dataset()
    data = e1000[0] # your data as a list, array, or pandas series
    res = calc.extract(data)

``get_dataset()`` returns the Empirical1000 dataset of 1000 empirical time series
(`doi:10.6084/m9.figshare.5436136.v11 <https://doi.org/10.6084/m9.figshare.5436136.v11>`_). It is not
bundled with pyhctsa: the first call downloads it (33 MB) to ``~/.cache/pyhctsa`` (set the
``PYHCTSA_DATA_DIR`` environment variable to use another directory).

``get_dataset('1000x1000')`` returns the `1000×1000 collection <https://dynamicsandneuralsystems.github.io/1000x1000/>`_
(`doi:10.6084/m9.figshare.34013331.v1 <https://doi.org/10.6084/m9.figshare.34013331.v1>`_): 1000 synthetic
time series, each 1000 samples long, simulated from 133 generating mechanisms (noise and linear
processes, maps, chaotic flows, oscillators, point processes and others). It is downloaded (7.5 MB)
and cached in the same way.

Note that each time-series instances does not have to be the same length to compute a vector of features. The results of the extraction will be returned in a pandas dataframe of shape 
N×F, where N is the number of time-series instances and F is the number of time-series features.

Many features are sensitive to how finely a series is sampled relative to its own dynamics.
Before extracting features from an oversampled dataset, see :doc:`sampling_rate` for how to check the sampling rate and resample all series with one factor.
Trends leak into many features too: see :doc:`trends` for how to diagnose a trend and, if you decide to remove it, how to detrend every series in the same way.

.. _repeated-values:

🔁 Series with many repeated values
-----------------------------------
Counts and coarsely quantised data repeat a few values many times.
Features that rank, bin or symbolise a series, or compare distances between its values, must split these ties.
A rescaled or shifted series z-scores to the same values only up to rounding, and the rounding decides how ties fall.
On 1000 Poisson counts with mean 3, multiplying by 1.05 or 3.7, or adding 7.3, changed 376-393 of the 6097 z-scored features (beyond a relative 1e-6), mostly symbolic ones (``motif_three``, ``surprise``) and local forecasters.
Rounded and 10-level processes, and Empirical1000 series that are nearly all ties, changed 124-544; Gaussian noise changed at most 5.
Series stored to a few significant digits changed up to about 100 even without repeats, through tied distances.
A few estimators treat ties in their own fixed way, but pyhctsa does not jitter or alter the data.

Results at a fixed scaling are reproducible.
Use the same units and scaling for every series you compare, and consider whether rank- and symbol-based features are meaningful for your data.
:meth:`~pyhctsa.calculator.FeatureCalculator.extract` warns when more than 90% of a series' samples share their value with another sample.

🤖 Calling Individual Operations
--------------------------------
If you would like to run individual operations on your data, you can access the corresponding functions from their respective modules directly. 
For example, to compute the `raw_hrv_meas` features on your data, the `raw_hrv_meas` master operation can be accessed from the `medical` module:

.. code-block:: python
    
    from pyhctsa.operations.medical import raw_hrv_meas
    data = ... # your ArrayLike data
    res = raw_hrv_meas(data) # result as either a dictionary or scalar value

Note that individual operations can only be called directly on individual time-series instances.

🏗️ Parallel Computing
---------------------
Time-series feature extraction is computationally intensive. 
To speed up processing, pyhctsa allows you to distribute the workload across multiple CPU 
cores on your local machine using the `LocalDistributor`:

.. code-block:: python
    
    from pyhctsa.distribute import LocalDistributor
    from pyhctsa.calculator import FeatureCalculator
    from pyhctsa.utils import get_dataset

    if __name__ == "__main__":
        data = get_dataset()[:10]

        # initialize the calculator
        calc = FeatureCalculator()

        # create a LocalDistributor and specify the number of workers
        # it is generally recommended to set n_workers to the number of physical CPU cores
        dist = LocalDistributor(n_workers=4)

        # pass the distributor to the .extract() method
        res = calc.extract(data, distributor=dist)

.. important::

   In a script, the code that starts the extraction must be under ``if __name__ == "__main__":``, as above.
   The workers are new Python processes that import your script, and without the guard each of them would
   start the extraction again: Python then raises a ``RuntimeError`` about the "bootstrapping phase" and the
   call never returns.
   The guard is not needed in a notebook or an interactive session.

    