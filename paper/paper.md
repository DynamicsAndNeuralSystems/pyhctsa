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

Across diverse fields of science, finance, and industry, the generation of time-series data has reached an unprecedented scale and complexity. With this data, there is an increasing demand for tools to distill interpretable summaries or `features' that capture the varied statistical and dynamical properties of the time series. [sentence about importance/usefuleness of low dim. summaries]

Drawing on a vast interdisciplinary literature of quantitative methods, highly comparative time-series analysis (HCTSA) offers a unified framework to automatically compute and compare thousands of time-series features. Although `HCTSA` has supported over XXX publications, its original MATLAB-based implementation presents significant accesibility and scalability barriers for many researchers. `pyhctsa` addresses these limitations by offering a native Python implementation, bringing the analytical power of the `HCTSA` library to the open-source ecosystem. By integrating with the modern Python data stack, `pyhctsa` will enable a broader community of reseachers to perform large scale, automated feature extraction in a transparent and reproducible environment. 


# Statement of need
[A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problem the software is designed to solve, who the target audience is, and its relation to other work.]
[importance of time-series feature-based representations, problem with bespoke feature sets which rely on expert knowledge, motivation for comprehensive toolkits.]

While existing solutions such as `Kats` [@Jiang:2022], `tsfresh` [@Christ:2018], and `tsfel` [@Barandas:2020] offer automated time-series feature extraction in Python, 
[Kats only a handful of distributional and spectral features]



Automated time-series feature extraction packages in R such as `theft` [@Henderson:2026], Python such as `Kats` [@Jiang:2022], `tsfresh` [@Christ:2018], and `tsfel` [@Barandas:2020]. 
Python packages limited in scope, often stop at spectral features [bring in HCTSA as being better, but in MATLAB].

While `HCTSA` has supported over XXX publications across a myriad of disciplines -- from computational neuroscience [Ref], to engineering [Ref] and geoscience [Ref] -- the proprietary nature of its original implementation in MATLAB presents significant accessibility barriers 
to researchers [without expensive licenses].
[reference to open source science and Python with statistics]


# Software design

The design philosophy of `pyhctsa` is based on the original MATLAB implementation [@Fulcher:2017], however to meet the needs of the open-source data science ecosystem, `pyhctsa` integrates the modular framework established in `pyspi` [@Cliff:2023], a software package previously developed by our group for comparative pairwise feature analysis.
To ensure functional parity with the MATLAB implementation, `pyhctsa` retains the original function nomenclature, conceptual groupings, and parameter specifications.

At its core, the `pyhctsa` codebase is structured around three components: (1) __feature functions__, i.e., standalone routines that compute and return one or more features from a time-series input; (2) a __configuration file__, which specifies the parameter combinations to be systematically passed to these functions; and (3) the __extractor class__ (`FeatureCalculator`), which manages the execution of configured functions and consolidates the resulting feature values into a structured output. 

While `pyhctsa` is designed to be 'plug-and-play' for standard use cases, it also provides the flexibility to specify bespoke parameter sets (i.e., feature sets) for specialised applications or research needs. Additionally, the modular architecture of `pyhctsa` enables the seamless integration of new feature functions and configurations, allowing the package to evolve alongside emerging research.  

## Implementation fidelity
While `pyhctsa` prioritises algorithmic equivalence with the original `HCTSA` toolbox, discrepancies in time-series feature values may arise from fundamental differences in how MATLAB and Python handle floating-point arithmetic and linear algebra routines. Furthermore, where the legacy implementation relied on proprietary MATLAB toolboxes and functions, we have integrated functionally similar open-source Python alternatives (e.g., `gaussian_kde` in Python instead of `ksdensity` for kernel density estimation). 

To verify the functional equivalence of `pyhctsa`, each feature-generating function was empirically validated against its original `HCTSA` counterpart. Using a heterogeneous dataset of 1000 empirical time series [@Fulcher:2021], we systematically compared the outputs of both implementations. Similarity was quantified by calculating the Pearson correlation coefficient $r$ between resulting feature vectors. We note that because a single function often produces multiple distinct time-series features, each output feature was validated individually. To ensure ported functions were sufficiently similar, we applied a stringent inclusion criterion by retaining only those features that demonstrated strong statistical agreement ($r \geq 0.9$) with the original MATLAB implementation.

![__Overview of the pyhctsa workflow__. __(i)__ input time-series dataset comprising one or more individual time-series instances. __(ii)__ Large-scale feature extraction transforms each time seris into a feature vector, forming a $N \times F$ feature matrix. __(iii)__ Feature sets are generated from a configuration YAML file that specifies feature function parameter combinations. __(iv)__ Supported data formats include 1D and 2D numpy arrays for time-series instances of equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is stuctured as a pandas DataFrame where rows are instances and columns are extracted features.](pyhctsa_graphical_abstract.png){width=100%}

# Code Example
```Python 
from pyhctsa.calculator import FeatureCalculator

data = ... # your data as a 1D array-like or 2D array-like
calc = FeatureCalculator() # instantiate a FeatureCalculator instance
res = calc.extract(data) # call the extract method on your data
```

# Outlook and future developments
[Modular architecture allows for seamless integration of new features. Contribution guidelines.]
Any time-series feature, provided it is accompanied by a published research paper, falls within the scope for conisderation in the `pyhctsa` library.
Clear contribution guidelines etc. here [ref].


# Acknowledgements

We acknowledge the contributers and maintainers of the original HCTSA package in MATLAB. 

# References