# OptuML: Hyperparameter Optimization for Machine Learning Algorithms using Optuna

```
 ⣰⡁ ⡀⣀ ⢀⡀ ⣀⣀    ⢀⡀ ⣀⡀ ⣰⡀ ⡀⢀ ⣀⣀  ⡇   ⠄ ⣀⣀  ⣀⡀ ⢀⡀ ⡀⣀ ⣰⡀   ⡎⢱ ⣀⡀ ⣰⡀ ⠄ ⣀⣀  ⠄ ⣀⣀ ⢀⡀ ⡀⣀
 ⢸  ⠏  ⠣⠜ ⠇⠇⠇   ⠣⠜ ⡧⠜ ⠘⠤ ⠣⠼ ⠇⠇⠇ ⠣   ⠇ ⠇⠇⠇ ⡧⠜ ⠣⠜ ⠏  ⠘⠤   ⠣⠜ ⡧⠜ ⠘⠤ ⠇ ⠇⠇⠇ ⠇ ⠴⠥ ⠣⠭ ⠏ 
```

`OptuML` (*Optu*na + *ML*) is a Python module providing hyperparameter optimization for machine learning algorithms using the [Optuna](https://optuna.org/) framework. The module offers a scikit-learn compatible API with enhanced features for robust optimization.

[![Python manual install](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml) [![Python pip install](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml) [![pypi version](https://img.shields.io/pypi/v/optuml)](https://pypi.org/project/optuml/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17305964.svg)](https://doi.org/10.5281/zenodo.17305963)

## tl;dr

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuml import Optimizer

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train optimizer
clf = Optimizer(algorithm="RandomForestClassifier", n_trials=50, cv=5, scoring="accuracy")
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## Key Features

- **Comprehensive Algorithm Support**: Full scikit-learn algorithm zoo plus CatBoost and XGBoost
- **Full Scikit-learn Compatibility**: Seamless integration with pipelines, cross-validation, and all sklearn tools
- **Robust Optimization**: Powered by Optuna with early stopping, timeout protection, and parallel execution
- **Type-Safe Design**: Separate optimizers for classification and regression with proper type checking
- **Production Ready**: Cross-platform compatibility, comprehensive error handling, and extensive validation
- **Flexible Configuration**: Control every aspect of the optimization process

## Installation

### Option A: pip (recommended)

```bash
pip install optuml
```

With optional algorithm support:

```bash
pip install optuml[all]          # CatBoost + XGBoost + LightGBM
pip install optuml[catboost]     # CatBoost only
pip install optuml[xgboost]      # XGBoost only
pip install optuml[lightgbm]     # LightGBM only
```

or upgrade:

```bash
pip install optuml --upgrade
```

### Option B: Manual installation

```bash
# Install required dependencies
pip install optuna scikit-learn numpy

# Optional: Install additional algorithms
pip install catboost xgboost

# Download the module
wget https://raw.githubusercontent.com/filipsPL/optuml/main/optuml/optuml.py
```

## Quick Start

### Classification Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuml import Optimizer

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train optimizer
clf = Optimizer(
    algorithm="RandomForestClassifier",
    n_trials=50,
    cv=5,
    scoring="accuracy",
    random_state=42,
    show_progress_bar=True
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# View results
print(f"Accuracy: {accuracy:.3f}")
print(f"Best parameters: {clf.best_params_}")
print(f"Optimization took: {clf.study_time_:.2f} seconds")
print(f"Trials completed: {clf.n_trials_completed_}")
```

### Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from optuml import Optimizer

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train optimizer
reg = Optimizer(
    algorithm="XGBRegressor",
    n_trials=100,
    cv=5,
    scoring="r2",
    early_stopping_patience=10,  # Stop if no improvement for 10 trials
    n_jobs=-1,  # Use all CPU cores for CV
    verbose=True
)
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
```

## Supported Algorithms

### Classification Algorithms

| Algorithm                        | Description                     | Key Features                              |
| -------------------------------- | ------------------------------- | ----------------------------------------- |
| `SVC`                            | Support Vector Classifier       | Non-linear kernels, probability estimates |
| `LogisticRegression`             | Logistic Regression             | L1/L2/Elastic-Net regularization          |
| `RidgeClassifier`                | Ridge Classifier                | L2 regularization, fast linear model      |
| `KNeighborsClassifier`           | k-Nearest Neighbors             | Distance weighting, various metrics       |
| `RandomForestClassifier`         | Random Forest                   | Feature importance, OOB score             |
| `ExtraTreesClassifier`           | Extremely Randomized Trees      | Faster than RF, reduced variance          |
| `AdaBoostClassifier`             | AdaBoost                        | Boosted ensemble, learning rate tuning    |
| `GradientBoostingClassifier`     | Gradient Boosting               | Sequential boosting, feature subsampling  |
| `HistGradientBoostingClassifier` | Histogram Gradient Boosting     | Fast GBDT, native NaN support             |
| `MLPClassifier`                  | Neural Network                  | Multiple architectures, early stopping    |
| `GaussianNB`                     | Gaussian Naive Bayes            | Fast, probabilistic                       |
| `QDA`                            | Quadratic Discriminant Analysis | Non-linear boundaries                     |
| `DecisionTreeClassifier`         | Decision Tree                   | Multiple criteria, pruning                |
| `SGDClassifier`                  | Stochastic Gradient Descent     | Multiple losses, L1/L2/ElasticNet, online |
| `CatBoostClassifier`*            | CatBoost                        | Categorical features, GPU support         |
| `XGBClassifier`*                 | XGBoost                         | Regularization, missing values            |
| `LGBMClassifier`*                | LightGBM                        | Fast GBDT, leaf-wise growth               |

### Regression Algorithms

| Algorithm                       | Description                 | Key Features                             |
| ------------------------------- | --------------------------- | ---------------------------------------- |
| `SVR`                           | Support Vector Regression   | Epsilon-insensitive loss                 |
| `LinearRegression`              | Linear Regression           | Simple, interpretable                    |
| `Ridge`                         | Ridge Regression            | L2 regularization, stable on collinear   |
| `Lasso`                         | Lasso Regression            | L1 regularization, feature selection     |
| `ElasticNet`                    | Elastic Net                 | L1+L2 regularization, sparse solutions   |
| `KNeighborsRegressor`           | k-Nearest Neighbors         | Local regression                         |
| `RandomForestRegressor`         | Random Forest               | Reduces overfitting                      |
| `ExtraTreesRegressor`           | Extremely Randomized Trees  | Faster than RF, reduced variance         |
| `AdaBoostRegressor`             | AdaBoost                    | Sequential learning                      |
| `GradientBoostingRegressor`     | Gradient Boosting           | Sequential boosting, feature subsampling |
| `HistGradientBoostingRegressor` | Histogram Gradient Boosting | Fast GBDT, native NaN support            |
| `MLPRegressor`                  | Neural Network              | Non-linear patterns                      |
| `DecisionTreeRegressor`         | Decision Tree               | Non-parametric                           |
| `SGDRegressor`                  | Stochastic Gradient Descent | Multiple losses, L1/L2/ElasticNet, online |
| `CatBoostRegressor`*            | CatBoost                    | Handles categoricals                     |
| `XGBRegressor`*                 | XGBoost                     | High performance                         |
| `LGBMRegressor`*                | LightGBM                    | Fast GBDT, leaf-wise growth              |

*Optional dependencies (install separately)

## Advanced Features

### Early Stopping

Stop optimization when no improvement is observed:

```python
optimizer = Optimizer(
    algorithm="XGBClassifier",
    n_trials=1000,
    early_stopping_patience=20  # Stop after 20 trials without improvement
)
```

### Parallel Cross-Validation

Speed up optimization using multiple CPU cores:

```python
optimizer = Optimizer(
    algorithm="RandomForestClassifier",
    n_trials=100,
    cv=10,
    n_jobs=-1  # Use all available cores
)
```

### Custom Scoring Metrics

Use any scikit-learn compatible scoring metric:

```python
optimizer = Optimizer(
    algorithm="SVC",
    scoring="roc_auc",  # For classification
    # scoring="neg_mean_squared_error",  # For regression
    # scoring="f1_weighted",  # For imbalanced classes
)
```

### Timeout Protection

Set time limits for optimization:

```python
optimizer = Optimizer(
    algorithm="MLPClassifier",
    timeout=300,  # Total optimization timeout (5 minutes)
    cv_timeout=30,  # Per-trial timeout (30 seconds)
    n_trials=1000  # Will stop at timeout even if trials remain
)
```

### Access to Optuna Study

Get detailed optimization information:

```python
# After fitting
optimizer.fit(X_train, y_train)

# Access the Optuna study object
study = optimizer.study_
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")

# Plot optimization history (requires plotly)
import optuna.visualization as vis
fig = vis.plot_optimization_history(study)
fig.show()

# Plot parameter importances
fig = vis.plot_param_importances(study)
fig.show()
```

### Pipeline Integration

Full compatibility with scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with OptuML
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('optimizer', Optimizer(algorithm="SVC", n_trials=50))
])

# Use like any sklearn pipeline
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Type-Specific Optimizers

For more control, use the specific optimizer classes:

```python
from optuml.optuml import ClassifierOptimizer, RegressorOptimizer

# Classifier with all classifier-specific methods
clf = ClassifierOptimizer(
    algorithm="RandomForestClassifier",
    n_trials=100
)
clf.fit(X_train, y_train)
probas = clf.predict_proba(X_test)
decision = clf.decision_function(X_test)  # If supported

# Regressor with regression-specific defaults
reg = RegressorOptimizer(
    algorithm="RandomForestRegressor",
    n_trials=100,
    scoring="r2"  # Default for regressors
)
```

## API Reference

### Main Classes

#### `Optimizer`
Universal optimizer that automatically selects between classification and regression.

#### `ClassifierOptimizer`
Specialized optimizer for classification algorithms with methods like `predict_proba()` and `decision_function()`.

#### `RegressorOptimizer`
Specialized optimizer for regression algorithms with appropriate default scoring metrics.

### Common Parameters

| Parameter                 | Type       | Default    | Description                                |
| ------------------------- | ---------- | ---------- | ------------------------------------------ |
| `algorithm`               | str        | required   | ML algorithm to optimize                   |
| `n_trials`                | int        | 100        | Number of optimization trials              |
| `cv`                      | int        | 5          | Cross-validation folds                     |
| `scoring`                 | str/None   | Auto*      | Scoring metric for CV                      |
| `direction`               | str        | "maximize" | Optimization direction                     |
| `timeout`                 | float/None | None       | Total optimization timeout (seconds)       |
| `cv_timeout`              | float      | 120        | Single CV evaluation timeout               |
| `random_state`            | int/None   | None       | Random seed for reproducibility            |
| `n_jobs`                  | int        | 1          | Parallel jobs for CV (-1 for all cores)    |
| `early_stopping_patience` | int/None   | None       | Trials without improvement before stopping |
| `verbose`                 | bool/int   | False      | Verbosity level                            |
| `show_progress_bar`       | bool       | False      | Show optimization progress                 |

*Auto defaults: "accuracy" for classifiers, "r2" for regressors

### Methods

| Method                 | Description                        | Available For    |
| ---------------------- | ---------------------------------- | ---------------- |
| `fit(X, y)`            | Optimize hyperparameters and train | All              |
| `predict(X)`           | Make predictions                   | All              |
| `score(X, y)`          | Evaluate model performance         | All              |
| `predict_proba(X)`     | Predict class probabilities        | Classifiers      |
| `decision_function(X)` | Get decision values                | Some classifiers |
| `get_params()`         | Get optimizer parameters           | All              |
| `set_params(**params)` | Set optimizer parameters           | All              |

### Attributes (after fitting)

| Attribute             | Description                        |
| --------------------- | ---------------------------------- |
| `best_estimator_`     | Trained model with best parameters |
| `best_params_`        | Best hyperparameters found         |
| `best_score_`         | Best cross-validation score        |
| `study_`              | Optuna study object                |
| `study_time_`         | Total optimization time            |
| `n_trials_completed_` | Number of completed trials         |
| `classes_`            | Class labels (classifiers only)    |
| `n_features_in_`      | Number of input features           |
| `feature_names_in_`   | Feature names (if available)       |

## Troubleshooting

### Issue: "No successful trials completed"
**Solution**: Increase `cv_timeout` or reduce `cv` folds:
```python
optimizer = Optimizer(algorithm="SVC", cv_timeout=300, cv=3)
```

### Issue: CatBoost/XGBoost/LightGBM not available
**Solution**: Install optional dependencies:
```bash
pip install optuml[all]
# or individually:
pip install catboost xgboost lightgbm
```

### Issue: Optimization takes too long
**Solutions**:
1. Use parallel CV: `n_jobs=-1`
2. Set timeout: `timeout=600`
3. Use early stopping: `early_stopping_patience=10`
4. Reduce trials: `n_trials=50`

### Issue: Memory errors with large datasets
**Solutions**:
1. Use algorithms with lower memory footprint (e.g., `LogisticRegression`, `SGDClassifier`, or `SGDRegressor`)
2. Reduce CV folds

## Best Practices

1. **Start with fewer trials**: Begin with `n_trials=20-50` for exploration, then increase for final optimization

2. **Use appropriate scoring metrics**: 
   - Imbalanced classification: `"f1_weighted"`, `"roc_auc"`
   - Regression: `"r2"`, `"neg_mean_squared_error"`
   
3. **Enable early stopping** for large trial counts:
   ```python
   Optimizer(n_trials=1000, early_stopping_patience=20)
   ```

4. **Set random state** for reproducibility:
   ```python
   Optimizer(random_state=42)
   ```

5. **Use parallel processing** for faster optimization:
   ```python
   Optimizer(n_jobs=-1)
   ```

## Benchmark

See [this page](benchmark/README.md) for benchmark results.

## Citation

If you use OptuML in your research, please cite:

```bibtex
@software{stefaniak_optuml_2024,
  author       = {Filip Stefaniak},
  title        = {OptuML: Hyperparameter Optimization for Multiple Machine Learning Algorithms using Optuna},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17305963},
  url          = {https://doi.org/10.5281/zenodo.17305963}
}
```
