
## Description

This package contains code for the Weak-Supervision Variational Auto-Encoder (WS-VAE). The WS-VAE is a weak
supervision model for binary classification tasks, specifically designed to be robust to the quality of weak
labels. Paper: https://proceedings.mlr.press/v202/tonolini23a/tonolini23a.pdf. This implementation is built with
TensorFLow 2.


## Installation

Copy this package to where you need it, then do the following:
1) move to the package's directory
```console
cd WS-VAE
```

2) create a Python 3.6 environment
```console
conda create --name wsvae python=3.6
```

3) activate the environment
```console
source activate wsvae
```

4) install requirements
```console
pip install -r requirements.txt
```

And you are good to go!

## Quick Example

Here is an example of how to train the model and use it for inference (the same example is run in the script bin/run_ws_vae.py).

#### Imports

```python
import sys
import os
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
from var_logger import load_wrench_data  # a function to load data from the format of the Wrench benchmark
from ws_vae import WeakSupervisionVAE  # the WS-VAE class
```

#### Load Data
The WS-VAE can run with any array of numeric features and binary weak labels (1=Positive, 0=Negative, -1=Abstain) 
as inputs (use var_logger.WeakSupervisionDataset() to format the data). 
In this example, we will load data from the Wrench Benchmark ([link](https://github.com/JieyuZ2/wrench)). 
In particular, we use the census tabular data set ([link](http://archive.ics.uci.edu/ml/datasets/Census+Income)). 
To reproduce this example, download the data from Wrench (link above) and put it in the 'data' folder.
```python
#### Load dataset
dataset_name = 'census'  # data set folder
# train split
dataset_path = os.path.join(path_to_package, 'data', dataset_name, 'train.json')
train_data = load_wrench_data(dataset_path)
# test split
dataset_path = os.path.join(path_to_package, 'data', dataset_name, 'test.json')
test_data = load_wrench_data(dataset_path)
```
NOTE: If you want to use text data from the Wrench framework, You will have to first infer numeric feature vectors with a pre-trained
language model, e.g., BERT. This can be done within the Wrench framework. You can also format your own data with the class var_logger.WeakSupervisionDataset() (see doc string in src/var_logger.py).

#### Train the WS-VAE
```python
model = WeakSupervisionVAE(train_data)
model.fit(gamma=0.1)  # see paper for hyper-parameters choice
```
The fit function can also take optional validation data with ground-truth labels, in which case it will be used for early stopping. 
As for the experiments in the paper, in this example we perform fully unsupervised weak labelling, with no validation ground-truth.

#### Infer with the WS-VAE
The WS-VAE supports various types of inference:
```python
labels = model.predict(test_data)  # infer hard label (1 or 0)
probabilities = model.predict_proba(test_data)  # infer class probabilities (p_class=0 and p_class=1 in the form [n_samples, 2])
mu, log_var = model.predict_continuous_label(test_data)  # infer mean and log variance of continuous label distribution (y_c in the paper)
```

#### Test the WS-VAE
The WS-VAE class also includes a testing function to return different evaluation metrics over a given test set:
```python
metric = model.test(test_data, metric_fn='f1_binary')
```
The argument 'metric_fn' specifies what evaluation to perform and what metrics to report. Options are:

| metric_fn | Returns |
|:--------|:---------|
| 'f1_binary' | Binary f1-score |
| 'auc' | ROC-AUC Score |
| 'acc' | Classification accuracy |
| 'kappa' | Cohen-Kappa agreement score between ground-truth and inferred labels | 
| 'matrix' | Confusion matrix | 

## References
If you use this package, please reference the following article:
- Tonolini, F., Aletras, N., Jiao, Y., & Kazai, G. (2023, July). Robust weak supervision with variational auto-encoders. In International Conference on Machine Learning (pp. 34394-34408). PMLR.