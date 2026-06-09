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
Methods for quantifying interpretable dynamical structure in time-series data have been central to many applications across science and industry, from medical diagnosis from physiological signals to demand forecasting in complex supply chains. These methods often yield real-valued summary statistics, or features of the dynamics that, because they are derived from underlying theory, can facilitate an understanding of the structured processes underlying time-series data and guide decision-making. Large feature sets, which aggregate large numbers of diverse time-series analysis methods, provide a powerful way to simultaneously leverage diverse representations of dynamical structure. The MATLAB-based `hctsa` (which stands for “highly comparative time-series analysis”) has the broadest coverage of methods to date, but its proprietary implementation is a barrier to open science and industry applications. Here we introduce `pyhctsa`, which implements the majority of the hctsa feature set in native Python, bringing this comprehensive and unified resource of time-series analysis methods to the free and open-source software (FOSS) ecosystem for the first time. `pyhctsa` includes over 4500 time-series features, making it the most extensive Python-based feature set currently available in both number and conceptual coverage.


# Statement of need
For a given time-series analysis problem, a fundamental challenge lies in identifying which analysis methods from a large and diverse methodological literature are most useful. Traditionally, this selection has relied on the subjective practice of hand-selecting features, often guided by expert knowledge or domain-specific conventions, which risks overlooking more informative data representations captured by methods in other fields. A highly comparative approach addresses this selection bias by systematically implementing and comparing as many existing time-series analysis methods as possible. The `hctsa` package facilitates such broad methodological comparison by implementing a comprehensive library of over 7000 diverse time-series methods, including statistics of the time-series distribution, linear correlation properties (including spectral structure), information theoretic measures including entropies and complexities, model fit statistics, self-affine scaling, stationarity metrics, symbolic motifs, and others. The approach has been applied broadly, including to problems in biology [@Jones:2023;@Phaniraj:2023;@He:2020], astronomy [@Barbara:2022], neuroimaging [@Faiman:2023;@Yang:2024], engineering [@Gorgannejad:2023;@Dabou:2021], and medicine [@Nahian:2021;@Kim:2022], among others. 

Despite its many strengths, the reliance of `hctsa` on the proprietary MATLAB ecosystem presents several barriers to broader adoption:

- __Financial__: The cost of licenses for MATLAB and its associated toolboxes is a financial barrier to use.
- __Workflow fragmentation__: As many data-analysis workflows are Python-based, `hctsa` users are often forced into inefficient and convoluted multi-language workflows.
- __Transparency__: As a proprietary environment, the source code for many MATLAB algorithms are not directly accessible. This conflicts with the ethos of open science, which prioritizes algorithmic transparency and reproducibility.

Given these limitations, there is a clear need for a FOSS implementation of `hctsa`.

# State of the field
While several FOSS packages for extracting sets of general time-series features from data exist -- such as `Kats` [@Jiang:2022] (40 features; Python), `tsfresh` [@Christ:2018] (783 features; Python), `TSFEL` [@Barandas:2020] (156 features; Python), and `feasts` (43 features; R) [@Feasts:2026] -- they each have a different scope of inclusion [@Henderson:2025] and, crucially, none match the scale and interdisciplinary coverage of `hctsa`.

Here we report on `pyhctsa`, which fills the gap in existing FOSS packages with an efficient and extendable architecture for computing a comprehensive set of time-series features from data in native Python. By porting the majority of the `hctsa` library, `pyhctsa` brings an implementation of over 4500 features while preserving the broad methodological reach of the original software. `pyhctsa` can be straightforwardly integrated into existing Python-based data science workflows for performing statistical learning on time series (including classification and regression problems) in ways that connect the analyst to interpretable algorithms derived from scientific theory. 

# Software design
As a package for large-scale feature extraction, the aim of `pyhctsa` is to compute a comprehensive time-series feature set on a given set of univariate time-series data, as shown schematically in \autoref{fig:schematic}(i) and (ii). To achieve this, `pyhctsa` development was guided by several design considerations:

1. __Extensibility__: A modular architecture that can scale to accommodate new time-series analysis methods without requiring modifications to the core `pyhctsa` codebase.
2. __Semantic consistency with hctsa__: To preserve the semantic meaning of algorithms across platforms, `pyhctsa` retains original `hctsa` analysis function names and parameter specifications, normalized to Pythonic case conventions.
3. __Usability__: To support general time-series analysis and machine learning workflows, function outputs are formatted into a standardized format (`pandas` `DataFrame`) for further processing.
4. __Function generalization__: Functions are written to separate the core time-series analysis algorithm from parameter settings. This allows the same algorithm to be reused across contexts without internal modifications to the code.
5. __Clear documentation__: Time-series analysis methods are clearly documented with a structured docstring and references to supporting literature where applicable. 

In `pyhctsa` features are generated from a YAML file (\autoref{fig:schematic}(iii)) that explicitly maps implemented algorithms to their parameter combinations. This YAML-based configuration enables rich flexibility and customisation: users can specify which algorithms to evaluate and as a result analyses can scale from a small selection of features to several thousand as needed. At runtime, `pyhctsa` iterates through the YAML mappings to generate a set of functions to be executed on the input dataset, which can consist of either equal-length time series (structured as an array) or variable-length instances (list of arrays), as shown in \autoref{fig:schematic}(iv). The results are aggregated into a "feature matrix", depicted in \autoref{fig:schematic}(v), where columns represent time-series features, and rows represent the time-series instances on which the features were computed.

![__Overview of the pyhctsa workflow__. __(i)__ Input time-series dataset comprising one or more univariate time-series instances. __(ii)__ Large-scale feature extraction transforms each time series into a feature vector, forming a $N \times F$ time series $\times$ feature matrix containing over 4500 features ($F > 4500$), drawn from a diverse literature of time-series analysis methods. __(iii)__ Feature sets are generated from a configuration YAML file that specifies function parameter combinations. __(iv)__ Supported data formats include `NumPy` arrays for datasets of time-series instances with equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is structured as a `pandas` `DataFrame` where each row corresponds to a univariate time series and each column is an extracted feature. \label{fig:schematic}](pyhctsa_graphical_abstract.pdf){width=100%}


## Validating function implementations
To provide broad access to the time-series analysis methods implemented in `hctsa`, we ported the majority of its library to native Python, while remaining as consistent as possible with the original algorithms. Achieving strict numerical equivalence between Python and MATLAB was often unattainable due to differences in floating-point arithmetic and linear algebra routines. Consequently, to verify the capture of a common time-series property, we required that ported algorithms exhibit a similar variation across a wide range of data. To achieve this, 
we computed the Pearson correlation $r$ between the outputs of a given MATLAB and Python algorithm across a benchmark of 1000 diverse (simulated and empirical) time series [@Fulcher:2021].
We then retained only those ported functions that demonstrated strong statistical agreement (defined as $r \geq 0.9$) with original MATLAB implementations. At release, `pyhctsa` includes 119 algorithms (73% of the original library) and maintains similar conceptual coverage to `hctsa`. Together, these algorithms yield over 4500 validated features. The remaining 44 algorithms from `hctsa` were excluded due to the absence of open-source equivalents or failure to meet the threshold for statistical agreement.


# Code Example
The `pyhctsa` API allows users to straightforwardly compute thousands of features in two lines of code that first specifies the set of calculations to be performed, and then runs that computation across a given time-series dataset:
```Python 
from pyhctsa.calculator import FeatureCalculator
calc = FeatureCalculator()
data_matrix = calc.extract(data)
```
Code tutorials are available at the `pyhctsa` `GitHub` repository [@Moore:2026].

# Research impact statement
The highly comparative framework implemented in `hctsa` has demonstrated broad research impact, with real-world applications across biology [@Jones:2023;@Phaniraj:2023;@He:2020], astronomy [@Barbara:2022], neuroimaging [@Faiman:2023;@Yang:2024], engineering [@Gorgannejad:2023;@Dabou:2021], and medicine [@Nahian:2021;@Kim:2022], among others. `pyhctsa` provides immediate significance by bringing the majority of the widely-used time-series analysis methods implemented in `hctsa` to the FOSS ecosystem for the first time. As an open-source package, `pyhctsa` overcomes the existing accessibility barriers of `hctsa`, enabling a broader community to incorporate these methods into machine learning and time-series analysis workflows.

# AI usage disclosure
Some function docstrings in the software were generated with the assistance of Large Language Models (LLMs). All AI-generated contributions were reviewed, edited, and verified by the human maintainers.

# References
