Conventions 
===========

Package Imports 
---------------
We use the following conventions when organising dependencies in `pyhctsa`:
    - Standard python libraries (**first**), third party imports (**next**), local imports (**last**)
    - Within each section, imports are listed alphabetically for easier scanning.

.. code-block:: python

   # Standard python libraries
   import time
   from typing import union

   # third-party imports
   import numpy as np
   import pandas as pd

   # local imports
   from pyhctsa.operations.physics import walker

Naming Conventions 
------------------
`pyhctsa` follows the standard `PEP 8 <https://peps.python.org/pep-0008/>`_ style guide. Specifically:

    - variable names: `snake_case <https://peps.python.org/pep-0008/#function-and-variable-names>`_
    - function names: `snake_case <https://peps.python.org/pep-0008/#function-and-variable-names>`_
    - module names: `snake_case <https://peps.python.org/pep-0008/#package-and-module-names>`_
    - class names: `PascalCase <https://peps.python.org/pep-0008/#class-names>`_
    - constants: `ALL_CAPITALS <https://peps.python.org/pep-0008/#constants>`_
 
Docstring Conventions
---------------------
Below is an example of the docstring convention for feature-computing functions in pyhctsa:

.. code-block:: python

    def feature_function(x : ArrayLike) -> float:
        """
        Description of the function including what it computes.
        Reference to literature provided with [1].
        Also see [2] for supporting literature.
        
        References
        ----------
        .. [1] Moore, J.B., "Supporting literature", IEEE, 2026.
        .. [2] Moore, J.B., "Second source", IEEE, 2026.
        
        Parameters
        ----------
        x : array-like
            Time series data.
        
        Returns
        -------
        float
            The feature value as a scalar.
        """
        x = np.asarray(x)
        x += 0.1
        out = np.mean(x)
        return out
Output Conventions
------------------
The calculator turns the output of a feature function into feature columns, so the shape of the output decides
the column names, and it should match hctsa's:

    - a function that returns one number (even for a single lag, as ``autocorr(y, 1)``) returns a ``float``
      (column ``label``); a function that returns several outputs returns a ``dict`` (columns ``label.field``,
      with hctsa's field names, which are case-sensitive);
    - a function that cannot compute anything for the input it was given (hctsa: ``out = NaN``) returns ``nan``
      in the scalar case, and a ``dict`` with every field NaN in the dict case: decorate dict-returning functions
      with :func:`~pyhctsa.utils.dict_output` and return ``np.nan`` at the failure, and the decorator fills in the
      field names (see :func:`~pyhctsa.utils.nan_outputs`);
    - NaN means 'not appropriate for this input' (a constant series, too few samples, a fit that does not exist);
      do not raise for these;
    - return only the fields hctsa's function returns, with its names; hctsa registers a subset of them, which
      the ``select:`` and ``exclude:`` keys of ``hctsa.yaml`` express;
    - random draws come from :func:`~pyhctsa.robust.bf_random` through :func:`~pyhctsa.robust.bf_random_seed`
      (as hctsa's ``BF_Random``), never from NumPy's global stream.
