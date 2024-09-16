OptuML: Hyperparameter Optimization for Multiple Machine Learning Algorithms using Optuna
=============================

```
  ⣰⡁ ⡀⣀ ⢀⡀ ⣀⣀    ⡎⢱ ⣀⡀ ⣰⡀ ⡀⢀ ⡷⢾ ⡇    ⠄ ⣀⣀  ⣀⡀ ⢀⡀ ⡀⣀ ⣰⡀   ⡎⢱ ⣀⡀ ⣰⡀ ⠄ ⣀⣀  ⠄ ⣀⣀ ⢀⡀ ⡀⣀
  ⢸  ⠏  ⠣⠜ ⠇⠇⠇   ⠣⠜ ⡧⠜ ⠘⠤ ⠣⠼ ⠇⠸ ⠧⠤   ⠇ ⠇⠇⠇ ⡧⠜ ⠣⠜ ⠏  ⠘⠤   ⠣⠜ ⡧⠜ ⠘⠤ ⠇ ⠇⠇⠇ ⠇ ⠴⠥ ⠣⠭ ⠏ 
```

`OptuML` (for *Optu*na and *ML*) is a Python module that provides hyperparameter optimization for several machine learning algorithms using the [Optuna](https://optuna.org/) framework. The module supports a variety of algorithms and allows easy hyperparameter tuning through a scikit-learn-like API.

[![Python manual install](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml) [![Python pip install](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml)

## Features

- **Multiple Algorithms**: Supports hyperparameter optimization for the following algorithms:
  - Scikit-learn zoo, plus:
  - CatBoost
  - XGBoost
- **Optuna Framework**: Leverages Optuna for powerful hyperparameter search.
- **Maximize or Minimize**: Allows setting the optimization direction (`maximize` or `minimize`).
- **Scikit-learn API**: Provides a consistent interface for `fit()`, `predict()`, `predict_proba()`, and `score()` methods.
- **Control Output**: Optionally run Optuna with granular verbosity settings (`verbose` as `bool` or `int`).
- **Cross-validation**: Easily integrate cross-validation with custom scoring metrics (e.g., accuracy, ROC AUC).

## Installation

### a) pip

`pip install optuml`

or upgrade with:

`pip install optuml --upgrade`


### b) Manual way


You can install the required packages via `pip`:

```bash
pip install optuna scikit-learn catboost xgboost numpy wrapt_timeout_decorator
```

Next just fetch the `optuml.py` [file from the repo](optuml/optuml.py) and put it in the directory with your script.

## Usage

### Basic Example

Here’s how you can use the `Optimizer` class to optimize hyperparameters for different machine learning algorithms using the **Iris** dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from OptuML import Optimizer

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the optimizer for SVC
optimizer = Optimizer(algorithm="SVC", n_trials=50, cv=3, scoring="accuracy", verbose=True)

# Fit the optimizer to the training data
optimizer.fit(X_train, y_train)

# Predict on the test set
y_pred = optimizer.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the best hyperparameters found during optimization
print(f"Best Hyperparameters: {optimizer.best_params_}")
```

### Available Algorithms

The `Optimizer` class supports the following algorithms. You can specify the `algorithm` parameter to choose which one to use:

##### Classifiers

| Algorithm                | Type                           |
|--------------------------|--------------------------------|
| `AdaBoostClassifier`    | AdaBoost Classifier            |
| `CatBoostClassifier`    | CatBoost Classifier            |
| `GaussianNB`            | Gaussian Naive Bayes           |
| `KNeighborsClassifier`  | k-Nearest Neighbors Classifier |
| `MLPClassifier`         | Multi-layer Perceptron Classifier |
| `RandomForestClassifier`| Random Forest Classifier       |
| `SVC`                  | Support Vector Classifier      |
| `XGBClassifier`         | XGBoost Classifier             |
| `QDA`                  | Quadratic Discriminant Analysis |

##### Regressors

| Algorithm                | Type                           |
|--------------------------|--------------------------------|
| `AdaBoostRegressor`     | AdaBoost Regressor             |
| `CatBoostRegressor`     | CatBoost Regressor             |
| `KNeighborsRegressor`   | k-Nearest Neighbors Regressor  |
| `MLPRegressor`          | Multi-layer Perceptron Regressor  |
| `RandomForestRegressor` | Random Forest Regressor        |
| `SVR`                  | Support Vector Regressor       |
| `XGBRegressor`          | XGBoost Regressor              |


### Controlling Verbosity

You can control the verbosity of Optuna's output by using the `verbose` parameter:

- Set `verbose=True` for standard logging.
- Use an `int` value to specify more granular verbosity levels (e.g., `optuna.logging.DEBUG`).

```python
optimizer = Optimizer(algorithm="SVC", n_trials=50, cv=3, scoring="accuracy", verbose=True)
```

- and/or show a progress bar:

```python
optimizer = Optimizer(algorithm="SVC", n_trials=50, cv=3, scoring="accuracy", show_progress_bar=True)
```

## API Reference

### `Optimizer`

#### Parameters

- **`algorithm`** (`str`): The machine learning algorithm to optimize.
- **`direction`** (`str`, default `"maximize"`): Direction of optimization. Can be `"maximize"` or `"minimize"`.
- **`verbose`** (`bool` or `int`, default `False`): Controls Optuna's verbosity.
- **`n_trials`** (`int`, default `100`): Number of optimization trials to run.
- **`timeout`** (`float`, optional): Maximum time (in seconds) for the optimization process.
- **`cv`** (`int`, default `5`): Number of cross-validation folds.
- **`scoring`** (`str`, default `"accuracy"`): Scoring metric to use during cross-validation.
- **`random_state`** (`int`, optional): Seed for random number generation.
- **`cv_timeout`** (`int`, default `120`) Timeout for a signle cv process within a trial

#### Methods

- **`fit(X, y)`**: Fit the model using hyperparameter optimization.
- **`predict(X)`**: Make predictions using the best model found during optimization.
- **`predict_proba(X)`**: Predict class probabilities (if supported by the model).
- **`score(X, y)`**: Score the model using the test data.
