.. pyhctsa documentation master file, created by
   sphinx-quickstart on Thu Feb 19 08:52:24 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**pyhctsa**: Python Toolkit of Highly Comparative Time-Series Analysis Features
===============================================================================

.. image:: _static/pyhctsa_logo.svg
   :width: 400
   :alt: pyhctsa logo
   :align: center
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa

|pypi| |pyversions| |tests| |zenodo| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/pyhctsa?style=flat-square
   :target: https://pypi.org/project/pyhctsa/
   :alt: PyPI Version

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/pyhctsa?style=flat-square
   :target: https://pypi.org/project/pyhctsa/
   :alt: Supported Python Versions

.. |tests| image:: https://img.shields.io/github/actions/workflow/status/DynamicsAndNeuralSystems/pyhctsa/run_unit_tests.yaml?branch=main&style=flat-square
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml
   :alt: Unit Test Status

.. |zenodo| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18529935-blue?style=flat-square
   :target: https://doi.org/10.5281/zenodo.18529935
   :alt: Zenodo DOI

.. |license| image:: https://img.shields.io/github/license/DynamicsAndNeuralSystems/pyhctsa.svg
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/LICENSE
   
Install
-------
Before installing `pyhctsa <https://pypi.org/project/pyhctsa/>`_, we strongly recommend setting up a `virtual environment <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>`_ to prevent dependency clashes: 

.. code-block:: bash

   pip install pyhctsa

Quickstart
----------
See :doc:`Getting started (Quickstart) <usage/getting_started>` for how to get up and running with `pyhctsa`.

API Reference
-------------
See the :doc:`API reference <api>` for `pyhctsa`.

License & Citation
------------------
**License**: GNU General Public License Version 3. See `LICENSE <https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/LICENSE>`_.

© 2026 Joshua Moore and Ben Fulcher. You may use, modify, and distribute this software with appropriate attribution.

Citation
--------
If you use `pyhctsa` in your work or publications, please cite:


   Moore, J. B., & Fulcher, B. D. (2026). pyhctsa: Python Toolkit of Highly Comparative Time Series Analysis Features [Software]. Zenodo. 
   https://doi.org/10.5281/zenodo.18652238

Or in BibTeX (version-agnostic):

.. code-block:: bibtex

   @software{pyhctsa:2026,
     author       = {Moore, Joshua B. and Fulcher, Ben D.},
     title        = {pyhctsa: Python Toolkit of Highly Comparative Time Series Analysis Features},
     year         = {2026},
     publisher    = {Zenodo},
     doi          = {10.5281/zenodo.18529934},
     url          = {https://doi.org/10.5281/zenodo.18529934}
   }

.. toctree::
   :maxdepth: 3
   :hidden:

   Home <self>
   installation
   usage/index
   api
   development/index
   mappings/index
