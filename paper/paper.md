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
<!-- Check validity of supporting > 500 publications claim - based on google scholar citations -->
Across diverse fields of science, finance, and industry, the generation of time-series data has reached an unprecedented scale. Accordingly, there is an increasing need for tools and methods to distill complex, high-dimensional datasets down into interpretable summaries or "features" that capture the varied statistical and dynamical properties underlying the time-series data. From uncovering empirical structure in large datasets [@Fulcher:2013] to seizure and stellar light curve classification [@Barbara:2022], feature-based approaches facilitate a wide range of statistical learning tasks involving time series, wherein the features are derived from interpretable theory and can thus yield insights that can guide understanding and help motivate further experiments. Among these approaches, the MATLAB-based `hctsa` package (which stands for "highly comparative time-series analysis") is the most comprehensive, containing implementations of over 7000 time-series features that can quantify a wide range of distributional and correlation-based statistical properties of time series. Although `hctsa` has supported over 500 publications, its MATLAB-based implementation has presented accessibility issues that have prevented its incorporation into analysis pipelines for open science and across industry applications. Our software package `pyhctsa` addresses these limitations by implementing the majority of the `hctsa` feature library in native Python, bringing a uniquely comprehensive resource of scientific time-series analysis methods to the free and open-source software (FOSS) ecosystem for the first time. The design philosophy of `pyhctsa` is centred around extensibility, with scope to become a living library of time-series analysis methods through further user contributions. Thus far, `pyhctsa` has implemented over 4500 time-series features, making it the most comprehensive Python package for time-series feature extraction by both total volume of features and conceptual coverage.


# Statement of need
Numerous FOSS packages for time-series feature extraction exist across various programming languages such as `kats` [@Jiang:2022] (40 features; Python), `tsfresh` [@Christ:2018] (783 features; Python), `tsfel` [@Barandas:2020] (156 features; Python), and `feasts` (43 features; R) [@Feasts:2026]. However, because these libraries were developed in isolation for different specific purposes, such as tracking human movement (`tsfel`) or econometrics (`kats`), they often overlap in the algorithms they implement while missing many others from the broader scientific literature. This presents a challenge for new problems: there are no clear rules on which tool or feature set to use for the given data, and picking one often means inheriting the narrow focus of its original creators. As a result, these packages may not generalise to new domains where the most informative and useful representations of the data lie outside their scope of feature coverage. 

Since the success of an analysis depends heavily on the choice of features used to represent the data, it is advantageous, and often necessary, to be able to compute and compare thousands of candidate features from across the diversity of available methods. The highly comparative approach, formalised via the `hctsa` package in MATLAB, addresses these limitations by drawing thousands of features from the diverse scientific literature of time-series analysis methods — ranging from linear correlation to information-theoretic quantities, model fits, scaling, stationarity, and more. The utility of `hctsa` is well-documented across diverse fields, including neuroscience [@Faiman:2023], geoscience [@Goel:2024], engineering [@Gorgannejad:2023], and biology [@Paul:2021; @Decat:2022], among others. While `hctsa` remains the gold standard for feature coverage, its reliance on the proprietary MATLAB ecosystem presents several barriers to broader adoption:

- __Financial barriers__: The high cost of commercial and academic MATLAB licenses, often compounded by the need for specialised, paid toolboxes of proprietary algorithms, restricts access for independent researchers, students, and even institutions in resource-limited settings.
- __Workflow fragmentation__: As many data-analysis workflows are Python-based, `hctsa` users are often forced into inefficient and convoluted multi-language workflows, hindering the development of end-to-end pipelines.
- __Transparency__: As a proprietary environment, MATLAB can obscure the traceability of its algorithms, where readable code depends on inaccessible, built-in subroutines and closed-source dependencies. This conflicts with the open-source ethos of modern science, which prioritises algorithmic transparency and reproducibility.

Given these limitations, there is a clear need for a FOSS implementation of `hctsa` that, unlike existing open-source packages, synthesises the diverse literature of time-series analysis methods. Our proposed solution, the Python software `pyhctsa`, can be straightforwardly integrated into existing python-based data science workflows for performing statistical learning on time series (including classification and regression problems) in ways that connect the analyst to interpretable algorithms derived from scientific theory. In the following sections, we describe the architecture of `pyhctsa`, the methodology used to ensure the consistency of its algorithms with MATLAB versions, and provide a code example demonstrating its use. 

# Software design
To ensure `pyhctsa` serves as a scalable tool for users, its development was guided by several design considerations:

1. __Extensibility__: A modular architecture that allows users to contribute new time-series analysis methods without modifying the core `pyhctsa` codebase. This ensures the software remains a flexible wrapper that scales alongside new algorithmic developments.
2. __Semantic consistency with hctsa__: To preserve the semantic meaning of algorithms across platforms, `pyhctsa` retains original `hctsa` function identifiers and parameter specifications, normalised to Pythonic case conventions. For example, the `hctsa` function `AutoCorr` is implemented in `pyhctsa` as `autocorr`. As a result, specific configurations (e.g., a function paired with a defined time-lag) correspond to the same conceptual operation in both the Python and MATLAB environments.
3. __Usability__: To support general time-series analysis and machine learning workflows, function outputs should be coerced into a standardised format (pandas `DataFrame`) for futher processing.
4. __Function generalisation__: Functions are written to separate the core time-series analysis algorithm from parameter settings and data pre-processing (e.g., normalisation). This allows the same algorithm to be reused across contexts without internal modifications to the code.
5. __Clear documentation__: Time-series analysis methods are clearly documented with a structured docstring comprised of (1) a description of the time-series analysis method; (2) input requirements, data types and outputs; and (3) references to supporting literature, where applicable. 

To realise these goals, the design philosophy of `pyhctsa` draws from the modular "calculator"-based approach of `pyspi` [@Cliff:2023], a software package previously developed for large-scale extraction of pairwise dependence metrics for multivariate time-series data (developed in the same reseach group as the authors). At its core, `pyhctsa` distinguishes between a core algorithm and a time-series feature. For example, the auto-correlation (AC) algorithm is implemented as a generic function `autocorr(x, tau, method)`, where `x` is the time-series input, `tau` and `method` are algorithm-specific parameters. While the core algorithm remains constant, varying the parameters `tau` and `method` yields different insights into the data (e.g., different time-lags and estimation methods). Consequently, a single algorithm can produce many distinct features (e.g., AC1, AC2, etc.), each representing a specific permutation of the underlying algorithm. 

Managing a library of hundreds of algorithms, each with many parameterisations, requires a systematic approach to configuration. Following the `pyspi` model, our package automates feature extraction through a YAML file that maps abstract algorithms (e.g., the autocorrelation) to their specific parameterisations. At runtime, `pyhctsa` programatically iterates through these instructions to generate a set of independent "tasks" by injecting the specified parameters into their corresponding functions. The results, obtained by executing each task on an input time series, are then aggregated into a standardised "feature matrix" (as a pandas `DataFrame`) where columns represent the specific features defined in the YAML and rows represent the time-series instances on which the features were computed. This configuration-driven architecture ensures that the resulting feature set (the collection of all computed outputs on a time series) is entirely reproducible and easily modified by updating the YAML registry, thus facilitating the extensibility required to become a living library of methods. While `pyhctsa` is designed to be "plug-and-play" for standard use cases, the YAML file provides the flexibility to specify custom parameter sets (i.e., feature sets) for specialised applications or research needs, allowing users to scale their analysis from dozens to thousands of features.

## Validating function implementations
Although we aimed to translate as many time-series analysis methods (implemented in `hctsa`) as equivalently as possible, minor discrepancies in ported function outputs can arise from fundamental differences in floating-point arithmetic and linear algebra routines between Python and MATLAB (despite identical function logic). In some cases, `hctsa` functions which rely on proprietary MATLAB toolboxes were ported with similar open-source Python alternatives (e.g., using Python's `scipy.stats.gaussian_kde` in place of MATLAB's `ksdensity` for kernel density estimation), which may further contribute to differences. For these reasons, validating Python implementations of time-series analysis methods based on strict numerical equivalence with MATLAB proved challenging. Given these constraints, we opted for a more pragmatic validation approach by computing the Pearson correlation coefficient $r$ between the outputs of MATLAB and Python functions across a benchmark dataset of 1000 empirical time series [@Fulcher:2021]. A high $r$ implies strong consistency between Python and MATLAB implementations (across the benchmark dataset), while a low $r$ suggests a fundamental structural discrepancy (e.g., diverging logic) that cannot be explained by systematic offsets (e.g., scaling or constant shifts) in the outputs. We retained only those ported functions that demonstrated strong statistical agreement (defined as $r \geq 0.9$) with original MATLAB versions, yielding a final suite of 4580 validated features for the current release of `pyhctsa` [@Moore:2026]. 


![__Overview of the pyhctsa workflow__. __(i)__ Input time-series dataset comprising one or more individual time-series instances. __(ii)__ Large-scale feature extraction transforms each time series into a feature vector, forming a $N \times F$ feature matrix containing over 4500 features ($F > 4500$), drawn from a diverse literature of time-series analysis methods. __(iii)__ Feature sets are generated from a configuration YAML file that specifies function parameter combinations. __(iv)__ Supported data formats include numpy arrays for datasets of time-series instances with equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is structured as a pandas `DataFrame` where rows are instances and columns are extracted features.](pyhctsa_graphical_abstract.pdf){width=100%}

# Code Example
The `pyhctsa` API streamlines convoluted function calls, enabling fast and automated feature extraction. As a result, users can transition from raw data to a data matrix of over 4500 features in two lines of code:
```Python 
from pyhctsa.calculator import FeatureCalculator
# 1. Initialise the engine with the full suite of > 4500 features
calc = FeatureCalculator()
# 2. Extract features from a 1D or 2D array-like dataset
data_matrix = calc.extract(data)
```
For futher examples and code tutorials, see the `pyhctsa` GitHub repository [@Moore:2026].

# Outlook and future developments
`pyhctsa` is intended to function as a living library; as such, future developments will be driven by contributions from the broader scientific community. Developed with extensibility in mind, `pyhctsa` enables users to freely adapt or add their own (e.g., domain-specific) functions for any application. To further broaden the coverage of cutting-edge methods in our package, we have established clear contributor guidelines for those wishing to integrate their algorithms into the existing library. To uphold a high standard of quality, we require that all time-series analysis methods be supported by a published, peer-reviewed paper. Detailed information regarding these guidelines can be found in the `pyhctsa` GitHub repository [@Moore:2026]. 

# Acknowledgements

We acknowledge the contributors and maintainers of the original `hctsa` package in MATLAB.

# References
