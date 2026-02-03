---
title: 'pyhctsa: A Python package for highly comparative time-series analysis'
tags:
  - Python
  - feature extraction
  - dynamics
  - time-series feature extraction
  - highly comparative
authors:
  - name: Joshua B. Moore
    orcid: 0000-0002-5237-1087
    equal-contrib: true
    affiliation: 1
  - name: Ben D. Fulcher
    equal-contrib: true
    orcid: 0000-0002-3003-4055
    affiliation: 1
affiliations:
 - name: School of Physics, The University of Sydney
   index: 1
   ror: "0384j8v12"
date: 3 February 2026
bibliography: paper.bib
---

# Summary

Across diverse fields of science, finance, and industry, the generation of time-series data XXX.
With this data, there is an increasing demand for tools to quantify meaningful patterns
Drawing on a vast interdisciplinary literature of algorithmic methods, highly comparative time-series analysis (hctsa) offers a unified framework to compare thousands of time-series features, each capturing distinct aspects of the data.
Such time-series features  


# Statement of need

`pyhctsa` is a Python package for highly comparative time-series feature analysis, based on the original implementation (`HCTSA`) in MATLAB [@Fulcher:2017; @Fulcher:2013].
The API for `pyhctsa` was developed to provide a user-friendly interface 


enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`pyhctsa` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. 
The original implementation in MATLAB has already been used in a number of scientific publications.
By open-sourcing, `pyhctsa` will enable XXX something about bringing these tools to more scientists and being used in their workflows, scientific discovery, etc. etc.

# Software design

The design philosophy of `pyhctsa` is based on the original MATLAB implementation [@Fulcher:2017], however to meet the needs of the open-source data science ecosystem, `pyhctsa` integrates the modular framework established in `pyspi` [@Cliff:2023], a software package previously developed by our group for comparative pairwise feature analysis.
To ensure functional parity with the MATLAB implementation, `pyhctsa` retains the original function nomenclature, conceptual groupings, and parameter specifications.

At its core, the `pyhctsa` codebase is structured around three components: (1) __feature functions__, i.e., standalone routines that compute and return one or more features from a time-series input; (2) a __configuration file__, which specifies the parameter combinations to be systematically passed to these functions; and (3) the __extractor class__ (`FeatureCalculator`), which manages the execution of configured functions and consolidates the resulting feature values into a structured output. 

While `pyhctsa` is designed to be 'plug-and-play' for standard use cases, it also provides the flexibility to specify bespoke parameter sets (i.e., feature sets) for specialised applications or research needs.

## Implementation fidelity
While `pyhctsa` prioritises algorithmic equivalence with the original `HCTSA` toolbox, discrepancies in time-series feature values may arise from fundamental differences in how MATLAB and Python handle floating-point arithmetic and linear algebra routines. Furthermore, where the legacy implementation relied on proprietary MATLAB toolboxes and functions, we have integrated functionally similar open-source Python alternatives (e.g., `gaussian_kde` in Python instead of `ksdensity` for kernel density estimation).
With these discrepancies in mind, we conducted a rigorous empirical validation
To acheive this, we computed features for a large, heterogenous dataset of 1000 empirical time series [@Fulcher:]using both the original MATLAB implementation and `pyhctsa`.
For each feature-computing function, we then computed the Pearson correlation 



# Research impact statement

`Gala` has demonstrated significant research impact and grown both its user base 
and contributor community since its initial release. The package has evolved 
through contributions from over 18 developers beyond the original core developer 
(@adrn), with community members adding new features, reporting bugs, and 
suggesting new features. 

While `Gala` started as a tool primarily to support the core developer's 
research, it has expanded organically to support a range of applications across 
domains in astrophysics related to Milky Way and galactic dynamics. The package 
has been used in over 400 publications (according to Google Scholar) spanning 
topics in galactic dynamics such as modeling stellar streams [@Pearson:2017], 
Milky Way mass modeling, and interpreting kinematic and stellar population 
trends in the Galaxy. `Gala` is integrated within the Astropy ecosystem as an 
affiliated package and has built functionality that extends the widely-used 
`astropy.units` and `astropy.coordinates` subpackages. `Gala`'s impact extends 
beyond citations in research: Because of its focus on usability and user 
interface design, `Gala` has also been incorporated into graduate-level galactic 
dynamics curricula at multiple institutions. 

`Gala` has been downloaded over 100,000 times from PyPI and conda-forge yearly 
(or ~2,000 downloads per week) over the past few years, demonstrating a broad 
and active user community. Users span career stages from graduate students to 
faculty and other established researchers and represent institutions around the 
world. This broad adoption and active participation validate `Gala`'s role as 
core community infrastructure for galactic dynamics research.

# AI usage disclosure

AI tools were used in the development of this software, including the writing of function docstrings and in some cases, the porting of feature functions from their original implementation in MATLAB to python. To ensure. 
No AI was used in the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References