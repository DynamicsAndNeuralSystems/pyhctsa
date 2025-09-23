# py-HCTSA

## Installation
To install py-hctsa locally, you can call:
```
pip install -e .
```

This will install pyhctsa in development mode. 

## Basic Usage
A `FeatureCalculator` object must first be instantiated using:
```Python
from pyhctsa.FeatureCalculator.calculator import FeatureCalculator
calc = FeatureCalculator()
```
By default, the `FeatureCalculator` will initialize the full feature set (> 800 master operations). If you would like to specify a custom feature set, you can pass the corresponding configuration .YAML file as an argument to the `FeatureCalculator`:
```Python
customCalc = FeatureCalculator(configPath="subset.yaml")
```
The number of master operations (callable functions) specified by the .yaml will be displayed for verification e.g., `Loaded 813 master operations.`

Once a `FeatureCalculator` has been initialized, you can call the `extract` method to compute time series features on either a single time-series instance or a list of multiple instances:
```Python
from pyhctsa.Utilities.utils import get_dataset

e1000 = get_dataset()
data = e1000[0] # your data as a list, array, or pandas series
res = calc.extract(data)
``` 
Note that each time-series instances does *not* have to be the same length to compute a vector of features. 
The results of the extraction will be returned in a pandas dataframe of shape $N \times F$, where $N$ is the number of time-series instances and $F$ is the number of time-series features.
You can also inspect the quality of the extracted feature values by calling ```calc.summary()```.  


## Configuration Files
