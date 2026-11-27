Installation
============

`pyhctsa` requires `Python 3.10 or higher <https://pypi.org/project/pyhctsa/>`_.

Using pip
---------
The easiest way to install `pyhctsa` is using `pip <https://pypi.org/project/pyhctsa/>`_:

.. code-block:: bash

   > pip install pyhctsa

This gives all 6344 features of the default feature set on Python 3.10 to 3.13.
Two dependencies are used by only a few features.
If either cannot be imported, the calculator skips those features and warns, naming the package to
install, instead of failing.

`ripser <https://github.com/scikit-tda/ripser.py>`_, for the 8 persistent-homology features
(``persistent_homology``), is installed with `pyhctsa`.
It publishes wheels for Python 3.10 to 3.14 on Linux, macOS and Windows (x64), so no compiler is
needed.
If your environment lacks it (after ``pip install --no-deps``, for example), add it with:

.. code-block:: bash

   > pip install ripser

`hmmlearn <https://github.com/hmmlearn/hmmlearn>`_, for the 34 hidden-Markov-model features
(``hmm_fit``, ``hmm_compare_n_states``), is installed with `pyhctsa` on Python 3.10 to 3.13.
It does not yet publish wheels for Python 3.14, so there `pyhctsa` gives 6310 features.
To get the other 34, install ``hmmlearn`` yourself, which builds it from source and needs a C++
compiler:

.. code-block:: bash

   > pip install "pyhctsa[hmm]"

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
