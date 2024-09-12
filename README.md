# OptuML: Hyperparameter Optimization for Multiple Machine Learning Algorithms using Optuna

`OptuML` is a Python module that provides hyperparameter optimization for several machine learning algorithms using the [Optuna](https://optuna.org/) framework. The module supports a variety of algorithms and allows easy hyperparameter tuning through a scikit-learn-like API.

[![Python tests](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml)

## Features

- **Multiple Algorithms**: Supports hyperparameter optimization for the following algorithms:
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (kNN)
  - Random Forest
  - CatBoost
  - XGBoost
  - Logistic Regression
  - Decision Tree
- **Optuna Framework**: Leverages Optuna for powerful hyperparameter search.
- **Maximize or Minimize**: Allows setting the optimization direction (`maximize` or `minimize`).
- **Scikit-learn API**: Provides a consistent interface for `fit()`, `predict()`, `predict_proba()`, and `score()` methods.
- **Control Output**: Optionally run Optuna with verbose logging (`verbose=True`) or in silent mode (`verbose=False`).
- **Cross-validation**: Easily integrate cross-validation with custom scoring metrics (e.g., accuracy, ROC AUC).

## Installation

### Prerequisites

Ensure that the following Python packages are installed:

- `optuna`
- `scikit-learn`
- `catboost`
- `xgboost`
- `numpy`

You can install the required packages via `pip`:

```bash
pip install optuna scikit-learn catboost xgboost numpy
```

### Installation

Just fetch the `optuml.py` [file from the repo](optuml.py) and put it in the directory with your script.


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

# Convert to a binary classification problem
X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the optimizer for SVM
optimizer = Optimizer(algorithm="SVM", n_trials=50, cv=3, scoring="accuracy", verbose=True, random_state=42)

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

- `"SVM"` (Support Vector Machine)
- `"kNN"` (k-Nearest Neighbors)
- `"RandomForest"` (Random Forest)
- `"CatBoost"` (CatBoost Classifier)
- `"XGBoost"` (XGBoost Classifier)
- `"LogisticRegression"` (Logistic Regression)
- `"DecisionTree"` (Decision Tree Classifier)

### Example of Optimizing Different Algorithms

You can iterate over different algorithms and optimize hyperparameters for each one:

```python
algorithms = ['SVM', 'kNN', 'RandomForest', 'CatBoost', 'XGBoost', 'LogisticRegression', 'DecisionTree']

for algorithm in algorithms:
    print(f"Optimizing {algorithm}")
    optimizer = Optimizer(algorithm=algorithm, n_trials=50, cv=3, scoring="accuracy", verbose=False, random_state=42)
    optimizer.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = optimizer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {algorithm}: {accuracy}")
    print(f"Best Hyperparameters for {algorithm}: {optimizer.best_params_}")
    print("="*60)
```

### Custom Scoring and Direction

You can also optimize for different scoring metrics like ROC AUC or use the `direction` parameter to **minimize** or **maximize** the objective:

```python
# Instantiate the optimizer for RandomForest with ROC AUC optimization
optimizer = Optimizer(algorithm="RandomForest", n_trials=50, cv=3, scoring="roc_auc", direction="maximize", verbose=True, random_state=42)

# Fit the optimizer to the training data
optimizer.fit(X_train, y_train)

# Predict probabilities and calculate ROC AUC
y_proba = optimizer.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])
print(f"ROC AUC: {roc_auc}")

# Print the best hyperparameters
print(f"Best Hyperparameters: {optimizer.best_params_}")
```

### Controlling Verbosity

You can control the verbosity of Optuna's output by using the `verbose` parameter:

- Set `verbose=True` to enable detailed Optuna logging.
- Set `verbose=False` to suppress logging and run the optimizer silently.

```python
optimizer = Optimizer(algorithm="SVM", n_trials=50, cv=3, scoring="accuracy", verbose=True, random_state=42)
```

## API Reference

### `Optimizer`

#### Parameters:

- **`algorithm`** (`str`): The machine learning algorithm to optimize. Options are `'SVM'`, `'kNN'`, `'RandomForest'`, `'CatBoost'`, `'XGBoost'`, `'LogisticRegression'`, `'DecisionTree'`.
- **`direction`** (`str`, default `"maximize"`): Direction of optimization. Can be `"maximize"` or `"minimize"`.
- **`verbose`** (`bool`, default `False`): If `True`, Optuna logging will be enabled. If `False`, the optimizer will run silently.
- **`n_trials`** (`int`, default `100`): Number of optimization trials to run.
- **`timeout`** (`float`, optional): Maximum time (in seconds) for the optimization process.
- **`cv`** (`int`, default `5`): Number of cross-validation folds.
- **`scoring`** (`str`, default `"accuracy"`): Scoring metric to use during cross-validation.
- **`random_state`** (`int`, optional): Seed for random number generation.

#### Methods:

- **`fit(X, y)`**: Fit the model using hyperparameter optimization.
- **`predict(X)`**: Make predictions using the best model found during optimization.
- **`predict_proba(X)`**: Predict class probabilities (if supported by the model).
- **`score(X, y)`**: Score the model using the test data.
- **`optimization_time()`**: Get the total time taken for optimization.

## Contributing

Contributions to the project are welcome! Please feel free to submit issues or pull requests on GitHub. You can also fork the repository and make your changes.

## Running Tests

You can run unit tests using `pytest`:

```bash
pip install pytest
pytest
```

### Contact

If you have any questions or feedback, feel free to open an issue on GitHub.
