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
customCalc = FeatureCalculator(configFile="subset.yaml")
```
Once a `FeatureCalculator` has been initialized, you can call the `extract` method to compute time series features:
```Python
from pyhctsa.Utilities.utils import get_dataset

e1000 = get_dataset()
data = e1000[0] # your data as a list, array, or pandas series
res = calc.extract(data)
``` 
The results of the extraction will be returned in a pandas dataframe of shape $N \times F$, where $N$ is the number of time-series instances and $F$ is the number of time-series features.

## Configuration Files
