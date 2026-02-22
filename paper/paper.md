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
    equal-contrib: false
    affiliation: 1
  - name: Ben D. Fulcher
    equal-contrib: false
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
Across diverse fields, the generation of time-series data has reached an unprecedented scale. Large feature sets have emerged as a powerful solution to distill these datasets into interpretable summary statistics that can yield new insights and facilitate a wide range of statistical learning tasks. While the MATLAB-based `hctsa` (which stands for "highly comparative time-series analysis") offers the broadest coverage of features to date -- containing over 7000 time-series features and supporting over 500 publications -- its proprietary implementation presents accessibility barriers for open science and industry applications. We address these limitations with `pyhctsa`, which implements the majority of the `hctsa` feature set in native Python, bringing a uniquely comprehensive resource of scientific time-series analysis methods to the free and open-source software (FOSS) ecosystem for the first time. `pyhctsa` includes over 4500 time-series features, making it the most extensive Python-based feature set currently available, leading in both total volume of features and conceptual coverage.


# Statement of need
The highly comparative approach, formalised via the `hctsa` package in MATLAB, aims to implement and compare as many existing time-series analysis methods as possible. By simultaneously computing a large set of summary statistics from a diverse scientific literature of methods -- ranging from linear correlation to information theoretic quantities, model fits, scaling, stationarity, and more -- `hctsa` offers the broadest feature coverage of any existing feature set to date. The utility of `hctsa` is well-documented; for example, `hctsa` has identified EEG markers of seizure disorders [@Faiman:2023], uncovered the dynamical structure of sleep EEG, including substructure in sleep states [@Decat:2022], and characterised subtle behavioural differences in the movement patterns of _C._ _elegans_ [@Fulcher:2017]. Despite its many strengths, the reliance of `hctsa` on the proprietary MATLAB ecosystem presents several barriers to broader adoption:

- __Financial barriers__: The cost-prohibitive nature of commercial and academic MATLAB licenses, often compounded by the need for specialised, paid toolboxes of proprietary algorithms, restricts access for independent researchers, students, practitioners, and even institutions in resource-limited settings.
- __Workflow fragmentation__: As many data-analysis workflows are Python-based, `hctsa` users are often forced into inefficient and convoluted multi-language workflows, adding friction and hindering the development of workflows.
- __Transparency__: As a proprietary environment, MATLAB can obscure the traceability of its algorithms where readable code depends on inaccessible, built-in subroutines and closed-source dependencies. This conflicts with the open-source ethos of modern science, which prioritises algorithmic transparency and reproducibility.

Given these limitations, there is a clear need for a FOSS implementation of `hctsa`. While several FOSS packages for feature extraction exist -- such as `kats` [@Jiang:2022] (40 features; Python), `tsfresh` [@Christ:2018] (783 features; Python), `TSFEL` [@Barandas:2020] (156 features; Python), and `feasts` (43 features; R) [@Feasts:2026] -- they often overlap in common algorithms while missing many others from the broader scientific literature [@Henderson:2025]. Crucially, none of these existing packages in the FOSS ecosystem match the scale or the interdisciplinary scope of `hctsa`.

Our proposed solution, `pyhctsa`, addresses this gap in the coverage of existing FOSS packages by providing a uniquely comprehensive resource for time-series analysis in native Python. By porting the majority of the `hctsa` feature set, our package brings over 4500 validated time-series features to the FOSS ecosystem for the first time. `pyhctsa` can be straightforwardly integrated into existing Python-based data science workflows for performing statistical learning on time series (including classification and regression problems) in ways that connect the analyst to interpretable algorithms derived from scientific theory. In the following sections, we describe the architecture and workflow of `pyhctsa` (summarised by the schematic in \autoref{fig:schematic}), detail the methodology used to ensure algorithmic consistency with the original MATLAB implementations, and provide a code example demonstrating its use. 


# Software design
As a package for large-scale feature extraction, the primary objective of any `pyhctsa` workflow is to facilitate the transformation of an input time-series dataset into a dataframe of extracted features, as shown schematically in \autoref{fig:schematic}(i) and (ii). To achieve this, its development was guided by several design considerations:

1. __Extensibility__: A modular architecture that allows users to contribute new time-series analysis methods without modifying the core `pyhctsa` codebase. This ensures the software remains a flexible wrapper that scales alongside new algorithmic developments.
2. __Semantic consistency with hctsa__: To preserve the semantic meaning of algorithms across platforms, `pyhctsa` retains original `hctsa` function identifiers and parameter specifications, normalised to Pythonic case conventions. Original function prefixes (e.g., `DN_`) have been dropped from function names and refactored into dedicated Python modules to group conceptually similar algorithms (e.g., distribution for all functions previously prefixed with `DN_`). For example, the `hctsa` function `DN_Spread` is implemented in `pyhctsa` as `distribution.spread`.
3. __Usability__: To support general time-series analysis and machine learning workflows, function outputs are coerced into a standardised format (pandas `DataFrame`) for further processing.
4. __Function generalisation__: Functions are written to separate the core time-series analysis algorithm from parameter settings and data pre-processing (e.g., normalisation). This allows the same algorithm to be reused across contexts without internal modifications to the code.
5. __Clear documentation__: Time-series analysis methods are clearly documented with a structured docstring comprised of (1) a description of the time-series analysis method; (2) input requirements, data types and outputs; and (3) references to supporting literature, where applicable. 

The design philosophy of `pyhctsa` draws from the modular "calculator"-based approach of `pyspi` [@Cliff:2023], a software package previously developed for large-scale extraction of pairwise dependence metrics for multivariate time-series data (developed in the same research group as the authors). At its core, `pyhctsa` distinguishes between an algorithm (the underlying mathematical function) and a feature (the specific output of that function under a unique parameter set). For instance, a single "mean"-computing algorithm, implemented as a function in `pyhctsa`, can produce arithmetic, harmonic, or interquartile mean features depending on the parameters provided to that function. A feature set is therefore constructed by iterating across both the library of algorithms and their various parameter combinations.

In `pyhctsa` this iteration is managed systematically through a YAML configuration file (\autoref{fig:schematic}(iii)) that explicitly maps algorithms to their specific parameter combinations. This configuration-driven architecture is the basis for the package's rich flexibility and customisation. While `pyhctsa` is designed to be "plug-and-play" for standard use cases, users can modify the YAML to clearly specify which functions and parameters to evaluate, allowing analyses to scale from a few dozen to thousands of features. At runtime, `pyhctsa` iterates through the YAML mappings to generate a set of functions to be executed on the input dataset, which can consist of either equal-length time series (structured as a  array) or variable-length instances (list of arrays), as shown in \autoref{fig:schematic}(iv). The results are aggregated into a standardised "feature matrix", depicted in \autoref{fig:schematic}(v) where columns represent the time-series features and rows represent the time-series instances on which the features were computed.


## Validating function implementations
To provide the FOSS community with the wide coverage of time-series analysis methods implemented in `hctsa`, we aimed to port the majority of its library to native Python, while remaining as consistent as possible with the original algorithms. However, achieving strict numerical equivalence between Python and MATLAB proved challenging due to fundamental differences in floating-point arithmetic and linear algebra routines. In some cases, `hctsa` functions which rely on proprietary MATLAB toolboxes required porting with similar open-source Python alternatives (e.g., using Python's `scipy.stats.gaussian_kde` in place of MATLAB's `ksdensity` for kernel density estimation), which also contributed to differences in the resulting feature values. 

Rather than insisting on absolute numerical agreement to verify the capture of a common time-series property, we adopted a pragmatic approach: ported algorithms must instead exhibit a similar variation across a wide range of data. To quantify the behavioural consistency between implementations, we computed the Pearson correlation coefficient $r$ between the outputs of MATLAB and Python functions across a benchmark dataset of 1000 diverse (simulated and empirical) time series [@Fulcher:2021]. We took a high $r$ as an indication of consistent behaviour between Python and MATLAB implementations across the benchmark dataset (i.e., the two implementations order time-series data along a common axis). We then retained only those ported functions that demonstrated strong statistical agreement (defined as $r \geq 0.9$) with original MATLAB versions, yielding a final suite of 4580 validated features for the current release of `pyhctsa` [@Moore:2026]. In total, 44 algorithms from the original `hctsa` feature set were excluded due to either absence of open-source equivalents (e.g., `TSTOOL`-based methods) or failure to meet the predefined threshold for statistical agreement of features.


![__Overview of the pyhctsa workflow__. __(i)__ Input time-series dataset comprising one or more individual time-series instances. __(ii)__ Large-scale feature extraction transforms each time series into a feature vector, forming a $N \times F$ feature matrix containing over 4500 features ($F > 4500$), drawn from a diverse literature of time-series analysis methods. __(iii)__ Feature sets are generated from a configuration YAML file that specifies function parameter combinations. __(iv)__ Supported data formats include NumPy arrays for datasets of time-series instances with equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is structured as a pandas `DataFrame` where each row corresponds to a univariate time series and each column is an extracted feature. \label{fig:schematic}](pyhctsa_graphical_abstract.pdf){width=100%}

# Code Example
The `pyhctsa` API streamlines convoluted function calls, allowing users to straightforwardly compute thousands of features in two lines of code:
```Python 
from pyhctsa.calculator import FeatureCalculator
# 1. Initialise the engine with the full suite of > 4500 features
calc = FeatureCalculator()
# 2. Extract features from a 1D or 2D array-like dataset
data_matrix = calc.extract(data)
```
For further examples and code tutorials, see the `pyhctsa` GitHub repository [@Moore:2026].

# Outlook and future developments
`pyhctsa` is intended to function as a living library of time-series analysis methods; as such, future developments will be driven by contributions from the broader scientific community. Developed with extensibility in mind, `pyhctsa` enables users to freely adapt or add their own (e.g., domain-specific) functions for any application. To further broaden the coverage of cutting-edge methods in our package, we have established clear contributor guidelines for those wishing to integrate their algorithms into the existing library. To uphold a high standard of quality, we require that all time-series analysis methods be supported by a published, peer-reviewed paper. Detailed information regarding these guidelines can be found in the `pyhctsa` GitHub repository [@Moore:2026]. 

# Acknowledgements

We acknowledge the contributors and maintainers of the original `hctsa` package in MATLAB. We are also grateful to Brendan Harris and Trent Henderson for testing the package and providing useful feedback.

# References
