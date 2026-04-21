<p align="center">
  <picture>
    <source srcset="img/pyhctsa_logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="img/pyhctsa_logo.png" alt="pyhctsa logo" height="300"/>
  </picture>
</p>

<h1 align="center"><em>pyhctsa</em>: Python Toolkit for Highly Comparative Time-Series Analysis</h1>

<p align="center" style="font-size:0; line-height:0;">
  <a href="https://pypi.org/project/pyhctsa/">
    <img src="https://img.shields.io/pypi/v/pyhctsa?style=flat-square" style="margin-right:8px;" />
  </a>
  <a href="https://pypi.org/project/pyhctsa/">
    <img src="https://img.shields.io/pypi/pyversions/pyhctsa?style=flat-square" style="margin-right:8px;" />
  </a>
  <a href="https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/DynamicsAndNeuralSystems/pyhctsa/run_unit_tests.yaml?branch=main&style=flat-square" style="margin-right:8px;" />
  </a>
  <a href="https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml">
    <img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/coverage-badge/coverage.svg" style="margin-right:8px;" />
  </a>
  <a href="https://doi.org/10.5281/zenodo.18529935">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18529935-blue?style=flat-square" />
  </a>
</p>


# pyhctsa

## ⬇️ Installation
To install _pyhctsa_ you can call:
```
pip install pyhctsa
```

## ✨ Basic Usage
A `FeatureCalculator` object must first be instantiated using:
```Python
from pyhctsa.calculator import FeatureCalculator
calc = FeatureCalculator()
```
By default, the `FeatureCalculator` will initialize the full feature set. If you would like to specify a custom feature set, you can pass the corresponding configuration .YAML file as an argument to the `FeatureCalculator`:
```Python
custom_calc = FeatureCalculator(config_path="subset.yaml")
```
The number of master operations (callable functions) specified by the .yaml will be displayed for verification e.g., `Loaded 700 master operations.`

Once a `FeatureCalculator` has been initialized, you can call the `extract` method to compute time series features on either a single time-series instance or a list of multiple instances:
```Python
from pyhctsa.utils import get_dataset

e1000 = get_dataset()
data = e1000[0] # your data as a list, array, or pandas series
res = calc.extract(data)
``` 
Note that each time-series instances does *not* have to be the same length to compute a vector of features. 
The results of the extraction will be returned in a pandas dataframe of shape $N \times F$, where $N$ is the number of time-series instances and $F$ is the number of time-series features.

You can also inspect the quality of the extracted feature values by calling ```calc.summary()```.  

# 📘 Tutorials
New to _pyhctsa_? Step-by-step tutorials and example workflows are available in the repository
👉 [`/tutorials`](./tutorials)

## 🤖 Advanced Usage
## Calling individual operations
If you would like to run individual operations on your data, you can access the corresponding functions from their respective modules directly.
For example, to compute the `raw_hrv_meas` features on your data, the `raw_hrv_meas` master operation can be accessed from the `medical` module:
```Python
from pyhctsa.operations.medical import raw_hrv_meas

data = ... # your ArrayLike data
res = raw_hrv_meas(data) # result as either a dictionary or scalar value
```
Note that individual operations can only be called directly on individual time-series instances.

## 🏗️ Parallel Computing
Time-series feature extraction is computationally intensive. 
To speed up processing, pyhctsa allows you to distribute the workload across multiple CPU cores on your local machine using the `LocalDistributor`:
```Python
from pyhctsa.distributed import LocalDistributor
from pyhctsa.calculator import FeatureCalculator

# initialize the calculator
calc = FeatureCalculator()

# create a LocalDistributor and specify the number of workers
# it is generally recommended to set n_workers to the number of physical CPU cores
dist = LocalDistributor(n_workers=4)

# pass the distributor to the .extract() method
res = calc.extract(data, distributor=dist)
```

## ℹ️ Note for Windows users
Some features require Java (JDK) to be installed. If you encounter a `JVM not found` error:

1. Ensure Java Development Kit (JDK) is installed on your system
   - Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or use OpenJDK
   - Minimum version required: JDK 11

2. Before importing pyhctsa, set the `JAVA_HOME` environment variable using the location of the JDK installation on your system:
```Python
import os
os.environ['JAVA_HOME'] = "C:\Program Files\Java\jdk-11" # replace with relevant path
from pyhctsa.calculator import FeatureCalculator
# rest of your code...
```

# 🔑 Licenses

## Internal licenses
Code for computing features from time-series data is licensed as [GNU General Public License version 3](http://www.gnu.org/licenses/gpl-3.0.en.html).

## External packages and dependencies
While the majority of features in _pyhctsa_ rely on standard Python libraries, a small subset of features require external toolboxes.

The following external time-series analysis code packages are provided with the software (in the `toolboxes` directory), and are used by our main feature-extraction calculator to compute meaningful structural features from time series:

- Joseph T. Lizier's [Java Information Dynamics Toolkit (JIDT)](https://github.com/jlizier/jidt) for studying  information-theoretic measures of computation in complex systems, version 1.3 (GPL license).
- Time-series analysis code developed by [Michael Small](https://github.com/m-small) (unlicensed).
- Max Little's [time-series analysis code](http://www.maxlittle.net/software/index.php) (GPL License).
- [TISEAN package for nonlinear time-series analysis](http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html), version 3.0.1 (GPL license).

The following codebases have been adapted directly into Python code within _pyhctsa_, rather than being included as external toolboxes:
- Danny Kaplan's Code for embedding statistics (GPL license).
- Histogram code by Rudy Moddemeijer (unlicensed).

## AI Usage Disclosure
Portions of this codebase (including tests and function documentation) were refactored and generated with the assistance of Large Language Models (LLMs). All AI-generated contributions have been reviewed and verified by the human maintainers.
