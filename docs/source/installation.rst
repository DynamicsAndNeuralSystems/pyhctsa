Installation
============

`pyhctsa` requires `Python 3.10 or higher <https://pypi.org/project/pyhctsa/>`_.

Using pip
---------
The easiest way to install `pyhctsa` is using `pip <https://pypi.org/project/pyhctsa/>`_:

.. code-block:: bash

   > pip install pyhctsa

This gives 6783 of the 6791 features of the *hctsa* v3.0.0 feature set.
The other 8, the persistent-homology features (``NL_PersistentHomology``), need the optional
`ripser <https://github.com/scikit-tda/ripser.py>`_ package; without it the calculator skips them and
warns. For the full feature set, install with the ``tda`` extra:

.. code-block:: bash

   > pip install "pyhctsa[tda]"

From Source
-----------
You can also install `pyhctsa` directly from the source code:


.. code-block:: bash

   git clone https://github.com/DynamicsAndNeuralSystems/pyhctsa.git
   cd pyhctsa
   pip install -e . 

Verify Installation 
-------------------
To verify that `pyhctsa` is installed correctly, you can run:

.. code-block:: bash

   pip show pyhctsa

If `pyhctsa` has been successfully installed, this shows its version, license and dependencies.
