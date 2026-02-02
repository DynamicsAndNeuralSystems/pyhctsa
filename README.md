<p align="center">
  <picture>
    <source srcset="img/pyhctsa_logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="img/pyhctsa_logo.png" alt="pyhctsa logo" height="300"/>
  </picture>
</p>

<h1 align="center"><em>pyhctsa</em>: Python Toolkit of Highly Comparative Time-Series Analysis Features</h1>

# pyhctsa

## ⬇️ Installation
To install pyhctsa locally, you can call:
```
pip install -e .
```

This will install pyhctsa in development mode. 

## ✨ Basic Usage
A `FeatureCalculator` object must first be instantiated using:
```Python
from pyhctsa.calculator import FeatureCalculator
calc = FeatureCalculator()
```
By default, the `FeatureCalculator` will initialize the full feature set. If you would like to specify a custom feature set, you can pass the corresponding configuration .YAML file as an argument to the `FeatureCalculator`:
```Python
customCalc = FeatureCalculator(config_path="subset.yaml")
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
