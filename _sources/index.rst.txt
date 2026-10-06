.. pyhctsa documentation master file, created by
   sphinx-quickstart on Thu Feb 19 08:52:24 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**Welcome to the pyhctsa documentation**.
=========================================

.. image:: _static/pyhctsa_logo.svg
   :width: 400
   :alt: pyhctsa logo
   :align: center
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa

|pypi| |pyversions| |tests| |joss| |pyOpenSci| |zenodo| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/pyhctsa?style=flat-square
   :target: https://pypi.org/project/pyhctsa/
   :alt: PyPI Version

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/pyhctsa?style=flat-square
   :target: https://pypi.org/project/pyhctsa/
   :alt: Supported Python Versions

.. |tests| image:: https://img.shields.io/github/actions/workflow/status/DynamicsAndNeuralSystems/pyhctsa/run_unit_tests.yaml?branch=main&style=flat-square
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml
   :alt: Unit Test Status

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.10581/status.svg
   :target: https://doi.org/10.21105/joss.10581
   :alt: JOSS publication

.. |pyOpenSci| image:: https://pyopensci.org/badges/peer-reviewed.svg
   :target: https://github.com/pyOpenSci/software-submission/issues/282
   :alt: pyOpenSci publication

.. |zenodo| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18529935-blue?style=flat-square
   :target: https://doi.org/10.5281/zenodo.18529935
   :alt: Zenodo DOI

.. |license| image:: https://img.shields.io/github/license/DynamicsAndNeuralSystems/pyhctsa.svg
   :target: https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/LICENSE
   
The **PY**\ thon toolkit for **H**\ ighly **C**\ omparative **T**\ ime-**S**\ eries **A**\ nalysis (``pyhctsa``) is a living library of time-series analysis methods.
With over 4500 time-series features derived from interpretable theory, ``pyhctsa`` is the most comprehensive feature set in native Python.


Installation
------------
Before installing `pyhctsa <https://pypi.org/project/pyhctsa/>`_, we strongly recommend setting up a `virtual environment <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>`_ to prevent dependency clashes: 

.. termynal:: 

      $ conda create -n pyhctsa python=3.12 -y
      $ conda activate pyhctsa
      $ pip install pyhctsa
      -->
      pyhctsa installed

Navigation
----------
Select from the cards below to navigate the `pyhctsa` documentation:

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :material-regular:`start;3em` Start Here
      :link: installation
      :link-type: doc

      Install pyhctsa and get up and running quickly.
   
   .. grid-item-card:: :material-regular:`play_lesson;3em` Usage Guide
      :link: usage/index
      :link-type: doc

      Tutorials and guides for using pyhctsa.
   
   .. grid-item-card:: :material-regular:`data_exploration;3em` Method List
      :link: methods/index
      :link-type: doc

      List and description of the time-series analysis methods
      included in pyhctsa.

   .. grid-item-card:: :material-regular:`api;3em` API Reference
      :link: api
      :link-type: doc

      General API reference for pyhctsa.

   .. grid-item-card:: :material-regular:`code;3em` Development
      :link: development/index
      :link-type: doc

      Developers guide for contributors.

   .. grid-item-card:: :material-regular:`merge;3em` Mappings
      :link: mappings/index
      :link-type: doc

      Function name mappings for existing hctsa (MATLAB) users.
   
   .. grid-item-card:: :material-regular:`groups;3em` Authors
      :link: authors
      :link-type: doc

      The people behind pyhctsa.

   .. grid-item-card:: :material-regular:`attribution;3em` License
      :link: license
      :link-type: doc

      GNU General Public License Version 3.

Citation
--------
If you use `pyhctsa` in your work or publications, please cite our JOSS paper:


   Moore et al., (2026). pyhctsa: A Python package for highly comparative time-series analysis. 
   Journal of Open Source Software, 11(123), 10581, https://doi.org/10.21105/joss.10581

Or in BibTeX (version-agnostic):

.. dropdown:: 

   .. code-block:: bibtex

      @article{Moore2026, 
         doi = {10.21105/joss.10581}, 
         url = {https://doi.org/10.21105/joss.10581}, 
         year = {2026}, 
         publisher = {The Open Journal}, 
         volume = {11}, 
         number = {123}, 
         pages = {10581}, 
         author = {Moore, Joshua B. and Fulcher, Ben D.}, 
         title = {pyhctsa: A Python package for highly comparative time-series analysis}, 
         journal = {Journal of Open Source Software}}

News and updates
----------------

.. article-info::
    :avatar: https://www.svgrepo.com/show/354057/medium-icon.svg
    :avatar-link: https://medium.com/@joshua.moore_17408/pyhctsa-is-here-and-it-might-change-how-you-analyze-time-series-d01abdfbf6f5
    :avatar-outline: white
    :author: Joshua Moore
    :date: Apr 18, 2026
    :read-time: 8 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-0


Developed and maintained by the
`Dynamics and Neural Systems Group <https://dynamicsandneuralsystems.github.io/>`_
at the University of Sydney.

.. image:: _static/dns_logo_horizontal.svg
   :alt: Dynamics and Neural Systems Group
   :target: https://dynamicsandneuralsystems.github.io/
   :width: 300px
   :align: center

© 2026 Joshua Moore and Ben Fulcher. You may use, modify, and distribute this software with appropriate attribution.

.. toctree::
   :maxdepth: 3
   :hidden:

   Home <self>
   installation
   usage/index
   methods/index
   api
   development/index
   mappings/index
   authors
   license
