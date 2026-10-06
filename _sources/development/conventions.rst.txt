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