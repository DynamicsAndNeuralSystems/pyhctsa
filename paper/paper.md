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
\begin{center}
\includegraphics[width=0.25\linewidth]{pyhctsa_logo-02.pdf}
\end{center}

Across diverse fields of science, finance, and industry, the generation of time-series data has reached an unprecedented scale. To navigate this abundance, there is an increasing need for tools and methods that distill complex, high-dimensional datasets down into interpretable summaries or `features' that capture the varied statistical and dynamical properties underlying the time-series data. From uncovering empirical structure in large datasets [@Fulcher:2013] to seizure and stellar light curve classification [@Barbara:2022], feature-based approaches play an integral role in tasks that would otherwise struggle with raw time-series values alone.

Drawing on a vast interdisciplinary literature of quantitative methods, highly comparative time-series analysis (HCTSA) provides a unified framework for automatically computing and comparing thousands of time-series features. Although `HCTSA` has supported over XXX publications, its original MATLAB-based implementation presents significant accessibility and scalability barriers for many researchers. `pyhctsa` addresses these limitations by offering a native Python implementation, bringing the analytical power of the `HCTSA` library to the open-source ecosystem. By integrating with the modern Python data stack, `pyhctsa` will enable a broader community of researchers to perform large-scale, automated feature extraction in a transparent and reproducible environment. 


# Statement of need
While Python libraries such as `Kats` [@Jiang:2022], `tsfresh` [@Christ:2018], and `tsfel` [@Barandas:2020] provide accessible workflows for automated feature extraction, they are generally designed around specific sets of distributional, spectral, and basic nonlinear methods. Such packages lack the interdisciplinary breadth required for a truly data-agnostic analysis, where the most informative features should be discovered from a wide net of candidate methods, rather than being selected from a relatively narrow, bespoke feature set. The highly comparative approach addresses these limitations by drawing thousands of features from a diverse scientific literature — ranging from linear correlation to information-theoretic quantities, model fits, scaling, stationarity, and others. This framework has been well-established in MATLAB as `HCTSA` [@Fulcher:2013] and has demonstrated significant utility across a wide range of domains, including neuroscience [@Faiman:2023], geoscience [@Goel:2024], engineering [@Gorgannejad:2023], and biology [@Paul:2021; @Decat:2022], among others.

While `HCTSA` remains the gold standard for feature breadth, its reliance on the proprietary MATLAB ecosystem presents several barriers to its widespread adoption:

- __Financial barriers__: The high cost of commercial and academic MATLAB licenses, often compounded by the need for specialised, paid toolboxes of proprietary algorithms, restricts access for independent researchers, students, and even institutions in resource-limited settings.
- __Workflow fragmentation__: The modern data science stack is predominantly Python-based, and as such, researchers wishing to utilise `HCTSA` are forced into inefficient and convoluted multi-language workflows, hindering the development of seamless end-to-end pipelines.
- __Limited scalability__: MATLAB does not integrate straightforwardly with modern machine learning frameworks (e.g., `scikit-learn`, `PyTorch`, or `TensorFlow`), making it difficult to incorporate highly comparative analysis into automated production models and pipelines.
- __Transparency__: As a closed-source environment, MATLAB can act as a "black box" by obscuring implementations of core functions. This contrasts with the open-source ethos of modern science, which prioritises algorithmic transparency and reproducibility.

There is, therefore, a clear need for a native Python implementation that brings the highly comparative framework to the open-source community, enabling more transparent, accessible, and scalable time-series analysis. Our software package, `pyhctsa`, addresses the above limitations by leveraging the modern Python scientific stack to deliver a platform for comprehensive, automated feature extraction that is compatible with existing machine learning workflows.
In the following sections, we describe the architecture of `pyhctsa`, the methodology used to ensure the reliability of its algorithms, and provide a code example demonstrating its ease of use. 

# Software design

The design philosophy of `pyhctsa` is inspired by the original MATLAB implementation [@Fulcher:2017], however to meet the needs of the open-source data science ecosystem, `pyhctsa` integrates a modular framework similar to that in `pyspi` [@Cliff:2023], a software package previously developed by the  Dynamics and Neural Systems Group for comparative pairwise feature analysis.
To ensure functional parity with the MATLAB implementation, `pyhctsa` retains the original function nomenclature, conceptual groupings, and parameter specifications.

At its core, the `pyhctsa` codebase is structured around three components:

1. __Feature functions__, i.e., standalone routines that compute and return one or more features from a time-series input.
2. A __configuration file__, which specifies the parameter combinations to be systematically passed to these functions; and 
3. the __extractor class__ (`FeatureCalculator`), which manages the execution of configured functions and consolidates the resulting feature values into a structured `DataFrame` output. 

While `pyhctsa` is designed to be 'plug-and-play' for standard use cases, it also provides the flexibility to specify bespoke parameter sets (i.e., feature sets) for specialised applications or research needs. Additionally, the modular architecture of `pyhctsa` enables the seamless integration of new feature functions and configurations, allowing the package to evolve alongside emerging research.  

## Implementation fidelity
While `pyhctsa` prioritises algorithmic equivalence with the original `HCTSA` toolbox, discrepancies in time-series feature values may arise from fundamental differences in how MATLAB and Python handle floating-point arithmetic and linear algebra routines. Furthermore, where the legacy implementation relied on proprietary MATLAB toolboxes and functions, we have integrated functionally similar open-source Python alternatives (e.g., `gaussian_kde` in Python instead of `ksdensity` for kernel density estimation). 

To verify the functional equivalence of `pyhctsa`, each feature-generating function was empirically validated against its original `HCTSA` counterpart. Using a heterogeneous dataset of 1000 empirical time series [@Fulcher:2021], we systematically compared the outputs of both implementations. Similarity was quantified by calculating the Pearson correlation coefficient $r$ between resulting feature vectors. We note that because a single function often produces multiple distinct time-series features, each output feature was validated individually. To ensure ported functions were sufficiently similar, we applied a stringent inclusion criterion by retaining only those features that demonstrated strong statistical agreement ($r \geq 0.9$) with the original MATLAB implementation.

![__Overview of the pyhctsa workflow__. __(i)__ input time-series dataset comprising one or more individual time-series instances. __(ii)__ Large-scale feature extraction transforms each time series into a feature vector, forming a $N \times F$ feature matrix. __(iii)__ Feature sets are generated from a configuration YAML file that specifies feature function parameter combinations. __(iv)__ Supported data formats include 1D and 2D numpy arrays for time-series instances of equal length, or lists of arrays for time series of different lengths. __(v)__ Post-feature extraction, the output is structured as a pandas `DataFrame` where rows are instances and columns are extracted features.](pyhctsa_graphical_abstract.pdf){width=100%}

# Code Example
The `pyhctsa` API prioritises simplicity by abstracting complex feature extraction routines into a streamlined workflow. As a result, users can transition from raw data to a comprehensive feature matrix of over 4000 features in just two lines of code:
```Python 
from pyhctsa.calculator import FeatureCalculator
# 1. Initialise the engine with the full suite of > 4000 features
calc = FeatureCalculator()
# 2. Extract features from a 1D or 2D array-like dataset
data_matrix = calc.extract(data)
```
By default, `FeatureCalculator` handles the initialisation of the entire `hctsa` library, allowing researchers to focus on data analysis rather than manual feature selection.

# Outlook and future developments
`pyhctsa` is intended to function as a living library of time-series analysis methods and, as such, its evolution is expected to be driven by contributions from the broader scientific community. We actively invite researchers to integrate their methods into the package to ensure it remains at the cutting edge of the field. To ensure the library remains a robust resource for reproducible science, we have established clear contributor guidelines. For example, a core requirement for the integration of new time-series features is that the underlying methods must be backed by a peer-reviewed, published paper. 

While the current release of `pyhctsa` delivers the core high-throughput feature extraction functionality of the `HCTSA` framework, future developments will focus on incorporating the visualisation, clustering, and classification tools of the original MATLAB toolkit, alongside methods for evaluating and ranking top-performing features. As with its MATLAB counterpart, such developments are expected to establish `pyhctsa` as a leading Python-native time-series analysis suite.

# Acknowledgements

We acknowledge the contributors and maintainers of the original HCTSA package in MATLAB.

# References