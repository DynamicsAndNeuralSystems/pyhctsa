HCTSA function mappings
=======================

For existing `hctsa` users, the following table provides a mapping between existing `hctsa`
functions and equivalent `pyhctsa` implementations for reference.

.. warning::

   The conceptual groupings in pyhctsa modules (e.g., ``medical``, ``distribution``, etc.) do not always 
   correspond to the same groupings in the original `HCTSA <https://github.com/benfulcher/hctsa/tree/main/Operations>`_. 

.. note::

   Names follow the current `hctsa` ``main`` branch. In August 2026, `hctsa` standardized the
   capitalization of 33 function names (e.g., ``DN_cv`` → ``DN_CV``, ``SC_fastdfa`` → ``SC_FastDFA``,
   ``EN_MS_LZcomplexity`` → ``EN_LZComplexity``, ``NL_TISEAN_d2`` → ``NL_d2``), and later renamed
   ``SB_BinaryStretch`` → ``SB_BinaryGapHomogeneity``. A few ported functions (``CO_RM_AMInformation``,
   ``DN_HighLowMu``, ``SY_DynWin``) have since been removed from `hctsa` and are listed under their last name.

.. csv-table::
   :file: legacy_function_name_mappings.csv
   :header-rows: 1
   :widths: auto
   