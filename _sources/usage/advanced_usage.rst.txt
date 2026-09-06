📕 Advanced usage
=================

⚙️ (1) Custom YAML configurations
---------------------------------
The YAML configuration file provides the flexibility for users to specify their own feature subsets as well as incorporate new time-series features and feature-computing functions.
Below, the general structure of a `pyhctsa` YAML file is specified.
In general, a YAML file follows a systematic nested structure. If a field is not applicable (e.g., no dependencies), 
it can either be left empty or excluded from the configuration altogether.

.. code-block:: yaml

    module:
        function_name:
            base_name: function_name
            dependencies:
            configs:
                - {param1: 1.0, param2: 2.0, zscore: True, abs: True}
            legacy_name: MD_polvar
            ordered_args: ['d', 'D']

In plain English, the YAML configuration instructs `pyhctsa` to ::
    
    1. Pre-process the time-series data by first z-score normalising and then taking the absolute value.
    2. Go to `pyhctsa.operations.<module>` and evaluate the function <function_name> on the data with param1 set to 1.0 and param2 set to 2.0.
    3. Construct a unique identifier for the output as <base_name>_<d>_<D>


.. rubric:: Field Descriptions

``module``
   The top-level key (e.g., ``distribution``) must exactly match a module in
   ``pyhctsa.operations`` (case-sensitive). If a function is not nested within the
   specified module, ``pyhctsa`` will skip its computation.

``function_name``
   The name of the function as defined in the module (case-sensitive), e.g.,
   ``add_noise`` for ``pyhctsa.operations.distribution.add_noise``.

``base_name``
   The operation name used in feature labels. Users may set this freely.

``dependencies``
   Any additional external dependencies required by the feature-computing function,
   beyond standard Python scientific libraries. May be left empty or omitted.

``configs``
   A list of parameter sets, where each entry defines a unique configuration.
   See below for a detailed explanation.

``legacy_name``
   Maps functions from the MATLAB HCTSA to equivalent ``pyhctsa`` implementations.
   May be left empty or omitted if not applicable.

``ordered_args``
   Specifies the order in which parameters from each configuration entry are
   injected into the feature function. Also used to construct the time-series feature identifiers.

.. rubric:: Configs

In `pyhctsa` YAML files, configs follow a specific convention. It is important to adhere to these conventions when specifying new functions or creating custom subsets to ensure the output is as expected.
When specifying a configuration, function-specific parameters are always followed by pre-processing arguments. Consider a generic function:

.. code-block:: python

    def feature_function(x, param1, param2):
        return feature

Here, the function accepts a raw time-series input (as an array), ``x``, and two
parameters ``param1`` and ``param2``. In this example, both parameters take scalar
values, but in general, parameters can be of type ``str``, ``bool``, ``array``,
``list``, ``int``, or ``float``.
The following YAML is specified:

.. code-block:: yaml

    module name:
        feature_function:
            base_name: function_name
            dependencies:
            configs:
                - {param1: 1.0, param2: 1.0}
                - {param1: 2.0, param2: 2.0}
            legacy_name:
            ordered_args: ['param1', 'param2']

In this example, two separate configurations are specified: one where ``param1 = 1.0``
and ``param2 = 1.0``, and another where ``param1 = 2.0`` and ``param2 = 2.0``. Accordingly, `feature_function` 
will be evaluated twice, once for the first configuration (``param1 = 1.0``
and ``param2 = 1.0``), and once for the second (``param1 = 2.0`` and ``param2 = 2.0``).

.. rubric:: Pre-processing arguments

By default, functions are evaluated on raw (unprocessed) time-series data. In many
cases, it is preferable (or necessary) to compute features in a way that is invariant
to the absolute scale of the data. This can be achieved by pre-processing the time
series prior to feature computation. The two pre-processing options available in
``pyhctsa`` are:

* **z-score:** Normalise the data by subtracting the mean and dividing by the standard deviation. See the :func:`z_score <pyhctsa.utils.z_score>` function documentation.
* **absolute (abs) value:** Take the absolute value of the time series.  

.. warning::

   By default, `pyhctsa` will always **z-score the data before applying the absolute value operation** (if set to True). 

To use either or both of these pre-processing options, they can be specified in the YAML:

.. code-block:: yaml

    module name:
        feature_function:
            base_name: function_name
            dependencies:
            configs:
                - {param1: 1.0, param2: 1.0, zscore: True}
                - {param1: 2.0, param2: 2.0, zscore: True, abs: True}
            legacy_name:
            ordered_args: ['param1', 'param2']

In this example, `pyhctsa` will first z-score the time-series data before computing `feature_function` for the first configuration, while for the second configuration, it will first z-score the data, 
then take the absolute value. Unless otherwise specified in the configuration, `zscore=False` and `abs=False` by default.

To make the YAML structure example more concrete, consider the :func:`pol_var <pyhctsa.operations.medical.pol_var>` function:

.. code-block:: yaml

    medical: 
        pol_var:
            base_name: pol_var
            dependencies:
            configs:
            - {d: 1.0, D: 3, zscore: True}
            - {d: 1.0, D: 4, zscore: True}
            - {d: 1.0, D: 5, zscore: True}
            - {d: 1.0, D: 6, zscore: True}
            legacy_name: MD_polvar
            ordered_args: ['d', 'D']

As per the API, the :func:`pol_var <pyhctsa.operations.medical.pol_var>` function accepts two parameters: ``d`` and ``D``. A total of 4 unique configurations will be evaluated by
`pyhctsa` corresponding to different inputs for ``D``. As per the configuration, the time-series input is first wrapped by a z-score operation before being input into the `pol_var` function.
In this case, each time the :func:`pol_var <pyhctsa.operations.medical.pol_var>` function is evaluated a single scalar value is returned, thus yielding
four time-series features across the four configurations: ::

    pol_var_1_3
    pol_var_1_4
    pol_var_1_5
    pol_var_1_6

Once a YAML configuration has been created (i.e., as a `.yaml` file), it can be loaded into the :func:`FeatureCalculator <pyhctsa.calculator.FeatureCalculator>` as follows:

.. code-block:: python

    from pyhctsa.calculator import FeatureCalculator

    calc = FeatureCalculator(config_path='<path_to_custom_yaml>.yaml')
    calc.extract(data)

☰ (2) Feature filtering
-----------------------
When constructing custom feature subsets, users may wish to control which time-series features are returned by an operation.
Consider the `binary_stats` function located in the `Symbolic` module. See :func:`binary_stats <pyhctsa.operations.symbolic.binary_stats>` for details.
The default configuration for this function in `pyhctsa` is:

.. code-block:: yaml

    symbolic: 
        binary_stats:
        base_name: binary_stats
        dependencies:
        configs:
            - {binary_method: 'mean', zscore: True}
        legacy_name: SB_BinaryStats
        ordered_args: ['binary_method']

We can then run the :func:`FeatureCalculator <pyhctsa.calculator.FeatureCalculator>` with this default configuration:

.. code-block:: python

    from pyhctsa.calculator import FeatureCalculator
    from pyhctsa.utils import get_dataset

    data = get_dataset(which='e1000')[0]
    calc = FeatureCalculator(config_path='<path_to_yaml>')
    res = calc.extract(data)
    print(f'Number of time-series features: {res.shape[1]}')

With these default settings, the :func:`binary_stats <pyhctsa.operations.symbolic.binary_stats>` function returns 18 different time-series features: :: 

    'binary_stats_mean.pupstat2',
    'binary_stats_mean.pstretch1',
    'binary_stats_mean.longstretch0',
    'binary_stats_mean.longstretch0norm',
    'binary_stats_mean.meanstretch0',
    'binary_stats_mean.meanstretch0norm',
    'binary_stats_mean.stdstretch0',
    'binary_stats_mean.stdstretch0norm',
    'binary_stats_mean.longstretch1',
    'binary_stats_mean.longstretch1norm',
    'binary_stats_mean.meanstretch1',
    'binary_stats_mean.meanstretch1norm',
    'binary_stats_mean.stdstretch1',
    'binary_stats_mean.stdstretch1norm',
    'binary_stats_mean.meanstretchdiff',
    'binary_stats_mean.stdstretchdiff',
    'binary_stats_mean.diff21stretch1',
    'binary_stats_mean.diff21stretch0'

If we would like to only return a specific time-series feature (or group of features), we can specify this in the configuration YAML using the `select` key.
For example, to isolate the feature `pupstat2`, we can specify the following:

.. code-block:: yaml

    symbolic: 
        binary_stats:
        base_name: binary_stats
        dependencies:
        configs:
            - {binary_method: 'mean', zscore: True, select: 'pupstat2'}
        legacy_name: SB_BinaryStats
        ordered_args: ['binary_method']

Running the `FeatureCalculator` with this configuration:

.. code-block:: python

    calc = FeatureCalculator(config_path='<path_to_new_yaml>')
    res = calc.extract(data)
    print(f'Number of time-series features: {res.shape[1]}')

Now, we can see that only a single time-series feature is returned: ::

    'binary_stats_mean.pupstat2'

Alternatively, if we would like to specify which features to **discard**, we can use the `exclude` key:

.. code-block:: yaml

    symbolic: 
        # same as before
        configs:
            - {binary_method: 'mean', zscore: True, exclude: 'pupstat2'}
        # same as before

Here, all time-series features excluding that specified in the configuration (`pupstat2`) will be returned.

To keep/discard multiple features, we can specify each in a list of strings as follows:

.. code-block:: yaml

    symbolic: 
        # same as before
        configs:
            - {binary_method: 'mean', zscore: True, select: ['pupstat2', 'pstretch1', 'meanstretchdiff']}
        # same as before


