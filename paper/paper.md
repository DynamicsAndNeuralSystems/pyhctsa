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

Given these limitations, there is a clear need for a FOSS implementation of `hctsa`. While several FOSS packages for extracting sets of general time-series features from data exist -- such as `Kats` [@Jiang:2022] (40 features; Python), `tsfresh` [@Christ:2018] (783 features; Python), `TSFEL` [@Barandas:2020] (156 features; Python), and `feasts` (43 features; R) [@Feasts:2026] -- they each have a different scope of inclusion [@Henderson:2025] and, crucially, none match the scale and interdisciplinary coverage of `hctsa`.

Here we report on `pyhctsa`, which fills the gap in existing FOSS packages with an efficient and extendable architecture for computing a comprehensive set of time-series features from data in native Python. By porting the majority of the `hctsa` library, `pyhctsa` brings an implementation of over 4500 features while preserving the broad methodological reach of the original software. `pyhctsa` can be straightforwardly integrated into existing Python-based data science workflows for performing statistical learning on time series (including classification and regression problems) in ways that connect the analyst to interpretable algorithms derived from scientific theory. 

In the following sections, we describe the architecture and workflow of `pyhctsa` (summarized by the schematic in \autoref{fig:schematic}), detail the methodology used to ensure algorithmic consistency with the original MATLAB implementations, and provide a code example demonstrating its use. 


# Software design
As a package for large-scale feature extraction, the aim of `pyhctsa` is to compute a comprehensive time-series feature set on a given set of univariate time-series data, as shown schematically in \autoref{fig:schematic}(i) and (ii). To achieve this, `pyhctsa` development was guided by several design considerations:

1. __Extensibility__: A modular architecture that allows users to contribute new time-series analysis methods without modifying the core `pyhctsa` codebase. This ensures the software can scale to accommodate new time-series analysis methods.
2. __Semantic consistency with hctsa__: To preserve the semantic meaning of algorithms across platforms, `pyhctsa` retains original `hctsa` analysis function names and parameter specifications, normalized to Pythonic case conventions. Original function prefixes (e.g., `DN_`) have been dropped from function names and refactored into dedicated Python modules to group conceptually similar algorithms (e.g., distribution for all functions previously prefixed with `DN_`). For example, the `hctsa` function `DN_Spread` is implemented in `pyhctsa` as `distribution.spread`.
3. __Usability__: To support general time-series analysis and machine learning workflows, function outputs are formatted into a standardized format (`pandas` `DataFrame`) for further processing.
4. __Function generalization__: Functions are written to separate the core time-series analysis algorithm from parameter settings. This allows the same algorithm to be reused across contexts without internal modifications to the code.
5. __Clear documentation__: Time-series analysis methods are clearly documented with a structured docstring comprised of (1) a description of the time-series analysis method; (2) input requirements, data types, and outputs; and (3) references to supporting literature, where applicable. 

The design philosophy of `pyhctsa` draws from the modular "calculator"-based approach of `pyspi` [@Cliff:2023], a software package previously developed for large-scale extraction of pairwise dependence metrics for multivariate time-series data (developed in the same research group as the authors). At its core, `pyhctsa` distinguishes between an algorithm (the underlying mathematical function) and a feature (the specific output of that function under a unique parameter set). For instance, a single "mean"-computing algorithm, implemented as a function in `pyhctsa`, can produce arithmetic, harmonic, or interquartile mean features depending on the parameters provided to that function. A feature set is therefore constructed by iterating across both the library of algorithms and their various parameter combinations.

In `pyhctsa` this iteration is performed systematically with a YAML configuration file (\autoref{fig:schematic}(iii)) that explicitly maps algorithms to their parameter combinations. While `pyhctsa` is designed to be "plug-and-play" for standard use cases, the YAML-based configuration forms the basis for rich flexibilty and customisation: users can specify which algorithms and parameters to evaluate and as a result analyses can scale from a small selection of features to several thousand as needed. At runtime, `pyhctsa` iterates through the YAML mappings to generate a set of functions to be executed on the input dataset, which can consist of either equal-length time series (structured as an array) or variable-length instances (list of arrays), as shown in \autoref{fig:schematic}(iv). The results are aggregated into a standardized "feature matrix", depicted in \autoref{fig:schematic}(v), where columns represent the time-series features, and rows represent the time-series instances on which the features were computed.
<!-- 
\begin{table}[h]
\centering
\caption{Summary of the algorithms (functions) and validated time-series features implemented in \textit{pyhctsa} v0.2.0 at the time of release, grouped by conceptual category. The number of features corresponds to the aggregated outputs calculated across unique parameter settings for all algorithms in that category. \label{tab:feature_summary}}
\begin{tabular}{lcc}
\hline
\textbf{category} & \textbf{num. functions} & \textbf{num. features} \\ \hline
correlation       & 23 & 852 \\
criticality       & 1  & 3   \\
distribution      & 21 & 320 \\
entropy           & 9  & 263 \\
extreme events    & 1  & 44  \\
graph             & 1  & 40  \\
hypothesis tests  & 2  & 19  \\
information       & 5  & 201 \\
medical           & 4  & 40  \\
model fit         & 7  & 321 \\
physics           & 2  & 274 \\
preprocess        & 1  & 260 \\
scaling           & 2  & 241 \\
spectral          & 1  & 228 \\
stationarity      & 15 & 454 \\
surrogates        & 1  & 20  \\
symbolic          & 7  & 740 \\
wavelet           & 6  & 188 \\ \hline
\textbf{Total}    & \textbf{119} & \textbf{4508} \\ \hline
\end{tabular}
\end{table} -->


![__Overview of the pyhctsa workflow__. __(i)__ Input time-series dataset comprising one or more univariate time-series instances. __(ii)__ Large-scale feature extraction transforms each time series into a feature vector, forming a $N \times F$ time series $\times$ feature matrix containing over 4500 features ($F > 4500$), drawn from a diverse literature of time-series analysis methods. __(iii)__ Feature sets are generated from a configuration YAML file that specifies function parameter combinations. __(iv)__ Supported data formats include `NumPy` arrays for datasets of time-series instances with equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is structured as a `pandas` `DataFrame` where each row corresponds to a univariate time series and each column is an extracted feature. \label{fig:schematic}](pyhctsa_graphical_abstract.pdf){width=100%}



## Validating function implementations
To provide the open-source community with broad access to the time-series analysis methods implemented in `hctsa`, we ported the majority of its library to native Python, while remaining as consistent as possible with the original algorithms. However, achieving strict numerical equivalence between Python and MATLAB proved challenging due to fundamental differences in floating-point arithmetic and linear algebra routines. In some cases, `hctsa` functions that rely on proprietary MATLAB toolboxes required porting with similar open-source Python alternatives (e.g., using Python's `scipy.stats.gaussian_kde` in place of MATLAB's `ksdensity` for kernel density estimation), which also contributed to implementation-based differences in the resulting feature values. 

Rather than insisting on exact numerical agreement to verify the capture of a common time-series property, we adopted a pragmatic approach: ported algorithms must instead exhibit a similar variation across a wide range of data. To quantify the behavioral consistency between implementations of a given algorithm, we computed the Pearson correlation coefficient $r$ between the outputs of MATLAB and Python functions across a benchmark dataset of 1000 diverse (simulated and empirical) time series [@Fulcher:2021]. We took a high $r$ as an indication of consistent behavior between Python and MATLAB implementations across the benchmark dataset. We then retained only those ported functions that demonstrated strong statistical agreement (defined as $r \geq 0.9$) with original MATLAB versions.

At release, `pyhctsa` includes 119 algorithms (73% of the original library) and maintains similar conceptual coverage to `hctsa`. Together, these algorithms yield over 4500 validated features. 
The remaining 44 algorithms from `hctsa` were excluded due to either the absence of open-source equivalents or failure to meet the predefined threshold for statistical agreement.


# Code Example
The `pyhctsa` API streamlines convoluted function calls, allowing users to straightforwardly compute thousands of features in two lines of code that first specifies the set of calculations to be performed, and then runs that computation across a given time-series dataset:
```Python 
from pyhctsa.calculator import FeatureCalculator
# 1. Initialize the engine with the full suite of > 4500 features
calc = FeatureCalculator()
# 2. Extract features from a 1D or 2D array-like dataset
data_matrix = calc.extract(data)
```
For further examples and code tutorials, see the `pyhctsa` `GitHub` repository [@Moore:2026].

# Outlook and future developments
`pyhctsa` is intended to function as a living library of time-series analysis methods; as such, we aim for its future development to be driven by contributions from the broader scientific community. Developed with extensibility in mind, `pyhctsa` enables users to freely adapt or add their own (e.g., domain-specific) methods for any application. To further broaden the coverage of methods within `pyhctsa`, we have established clear contributor guidelines for those wishing to integrate their algorithms into the existing library. To uphold a high standard of quality, we require that all time-series analysis methods be supported by a published, peer-reviewed paper. Detailed information regarding these guidelines can be found in the `pyhctsa` `GitHub` repository [@Moore:2026]. 

# Acknowledgements

We acknowledge the contributors and maintainers of the original `hctsa` package in MATLAB. We are also grateful to Brendan Harris and Trent Henderson for testing the package and providing useful feedback.

# AI usage disclosure
No generative AI tools were used in the writing of this manuscript. Some function docstrings in the software were generated with the assistance of Large Language Models (LLMs). All AI-generated contributions were reviewed, edited, and verified by the human maintainers.

# References
