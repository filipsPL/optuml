# OptuML: Hyperparameter Optimization for Machine Learning Algorithms using Optuna

```
 ⣰⡁ ⡀⣀ ⢀⡀ ⣀⣀    ⢀⡀ ⣀⡀ ⣰⡀ ⡀⢀ ⣀⣀  ⡇   ⠄ ⣀⣀  ⣀⡀ ⢀⡀ ⡀⣀ ⣰⡀   ⡎⢱ ⣀⡀ ⣰⡀ ⠄ ⣀⣀  ⠄ ⣀⣀ ⢀⡀ ⡀⣀
 ⢸  ⠏  ⠣⠜ ⠇⠇⠇   ⠣⠜ ⡧⠜ ⠘⠤ ⠣⠼ ⠇⠇⠇ ⠣   ⠇ ⠇⠇⠇ ⡧⠜ ⠣⠜ ⠏  ⠘⠤   ⠣⠜ ⡧⠜ ⠘⠤ ⠇ ⠇⠇⠇ ⠇ ⠴⠥ ⠣⠭ ⠏ 
```

`OptuML` (*Optu*na + *ML*) is a Python module providing hyperparameter optimization for machine learning algorithms using the [Optuna](https://optuna.org/) framework. The module offers a scikit-learn compatible API with enhanced features for robust optimization.

[![Python manual install](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-package.yml) [![Python pip install](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/python-pip.yml) [![pypi version](https://img.shields.io/pypi/v/optuml)](https://pypi.org/project/optuml/) [![CodeQL Advanced](https://github.com/filipsPL/optuml/actions/workflows/codeql.yml/badge.svg)](https://github.com/filipsPL/optuml/actions/workflows/codeql.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17305964.svg)](https://doi.org/10.5281/zenodo.17305963)
 
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

print(accuracy)
# 0.9111111111111111
print(y_pred[:10])
# [1 1 1 1 0 0 2 2 0 0]
```

## tl;dr why this module?

*I want to make a fair comparison of ML methods, where 'fair' means that each method has tuned hyperparameters, making it the best version of itself.*

Two design choices support that fairness: scale-sensitive algorithms (SVM, KNN, MLP, regularized-linear models) are automatically standardized so they are not handicapped against scale-invariant trees (see [Feature Scaling](#feature-scaling)), and [`nested_score()`](#unbiased-evaluation-nested-cross-validation) provides an *unbiased* generalization estimate for reporting, since `best_score_` is optimistically biased by the hyperparameter search.


## Key Features

- **Comprehensive Algorithm Support**: Full scikit-learn algorithm zoo plus optional CatBoost, XGBoost, LightGBM and fuzzy-rough-learn (FRNN, FROVOCO)
- **Full Scikit-learn Compatibility**: Seamless integration with pipelines, cross-validation, and all sklearn tools
- **Robust Optimization**: Powered by Optuna with early stopping, timeout protection, and parallel execution
- **Type-Safe Design**: Separate optimizers for classification and regression with proper type checking
- **Production Ready**: Cross-platform compatibility, comprehensive error handling, and extensive validation
- **Flexible Configuration**: Control every aspect of the optimization process
- **Benchmarking** of multiple algorithms at once, see [benchmarking](#algorithm-benchmarking)

## Installation

### Option A: pip (recommended)

```bash
pip install optuml
```

With optional algorithm support:

```bash
pip install optuml[all]          # CatBoost + XGBoost + LightGBM + fuzzy-rough-learn + interpret + imodels
pip install optuml[catboost]     # CatBoost only
pip install optuml[xgboost]      # XGBoost only
pip install optuml[lightgbm]     # LightGBM only
pip install optuml[frlearn]      # fuzzy-rough-learn only (FRNN / FROVOCO)
pip install optuml[interpret]    # InterpretML (Explainable Boosting Machine)
pip install optuml[imodels]      # imodels (RuleFit / FIGS)
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
pip install catboost xgboost fuzzy-rough-learn interpret-core imodels

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
| `RoughSetClassifier`             | Rough Sets (Pawlak)             | Reducts + readable if-then rules          |
| `LDA`                            | Linear Discriminant Analysis    | Linear boundaries, optional shrinkage     |
| `GaussianProcessClassifier`      | Gaussian Process                | RBF/Matern/RQ kernels, probabilistic (small data) |
| `LinearSVC`                      | Linear SVM                      | Scales to many samples, hinge/L1 options  |
| `CatBoostClassifier`*            | CatBoost                        | Categorical features, GPU support         |
| `XGBClassifier`*                 | XGBoost                         | Regularization, missing values            |
| `LGBMClassifier`*                | LightGBM                        | Fast GBDT, leaf-wise growth               |
| `FRNNClassifier`*                | Fuzzy Rough Nearest Neighbour   | OWA-weighted fuzzy-rough k-NN             |
| `FROVOCOClassifier`*             | Fuzzy Rough OVO Combination     | Imbalanced multiclass (one-vs-one)        |
| `ExplainableBoostingClassifier`* | Explainable Boosting Machine    | Glass-box GAM, per-feature curves, native NaN |
| `RuleFitClassifier`*             | RuleFit                         | Sparse linear model over tree-derived rules |
| `FIGSClassifier`*                | FIGS                            | Fast Interpretable Greedy-tree Sums       |

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
| `GaussianProcessRegressor`      | Gaussian Process            | RBF/Matern/RQ kernels, smooth fits (small data) |
| `KernelRidge`                   | Kernel Ridge                | Linear/RBF/poly/laplacian/sigmoid kernels |
| `PLSRegression`                 | Partial Least Squares       | Many correlated features, latent components |
| `BayesianRidge`                 | Bayesian Ridge              | Automatic regularization, small data      |
| `HuberRegressor`                | Huber Regression            | Robust to outliers                        |
| `LinearSVR`                     | Linear SVR                  | Epsilon-insensitive loss, scales to many samples |
| `CatBoostRegressor`*            | CatBoost                    | Handles categoricals                     |
| `XGBRegressor`*                 | XGBoost                     | High performance                         |
| `LGBMRegressor`*                | LightGBM                    | Fast GBDT, leaf-wise growth              |
| `FRNNRegressor`*                | Fuzzy-rough k-NN            | Fuzzy-rough local regression             |
| `ExplainableBoostingRegressor`* | Explainable Boosting Machine | Glass-box GAM, per-feature curves, native NaN |
| `RuleFitRegressor`*             | RuleFit                     | Sparse linear model over tree-derived rules |
| `FIGSRegressor`*                | FIGS                        | Fast Interpretable Greedy-tree Sums      |

*Optional dependencies (install separately)

### Fuzzy-Rough Algorithms (`fuzzy-rough-learn`)

`FRNNClassifier`, `FROVOCOClassifier` and `FRNNRegressor` come from the [`fuzzy-rough-learn`](https://pypi.org/project/fuzzy-rough-learn/) package (import name `frlearn`). Install it with `pip install optuml[frlearn]` or `pip install fuzzy-rough-learn`; they are listed only when it is installed.

```python
from optuml import Optimizer

optimizer = Optimizer(algorithm="FROVOCOClassifier", n_trials=50).fit(X, y)
```

Caveats:

- **`predict_proba` is not calibrated.** It returns fuzzy-rough membership scores, clipped at 0 and normalised per row. They rank classes well (fine for `roc_auc`), but they are not probabilities, so do not use them for thresholding or log-loss. These classifiers have no `decision_function`.
- **Upstream is unmaintained.** The last `fuzzy-rough-learn` release is 0.2.2 (June 2022). It has worked with NumPy 2.4, scikit-learn 1.8 and Python 3.14, but future releases of those libraries may break it.
- **Input:** features are range-normalised internally, so these algorithms are not affected by `scale`. They do not accept `NaN` and require dense input (sparse input raises `TypeError`).

### Rough-Set Classifier (Pawlak)

`RoughSetClassifier` implements classic [rough set theory](https://en.wikipedia.org/wiki/Rough_set), introduced by [Zdzisław Pawlak](https://en.wikipedia.org/wiki/Zdzis%C5%82aw_Pawlak) in 1982. It is built in (no extra dependency). Features are discretised into bins; objects in the same bins on a set of attributes are *indiscernible*. A class is described by its **lower approximation** (objects that certainly belong) and **upper approximation** (objects that possibly belong). The classifier:

1. discretises each feature (`n_bins`, `strategy`), inside each CV fold;
2. finds a **reduct**, a small subset of attributes that preserves the dependency degree (greedy QuickReduct; `reduct_epsilon > 0` allows an approximate reduct, which is more robust to noise);
3. turns the indiscernibility classes over the reduct into **certain** rules (lower approximation) and **possible** rules (boundary region, used if `use_possible_rules`), dropping rules with support below `min_support`;
4. predicts by matching the rules, backing off to fewer reduct attributes when nothing matches.

```python
from optuml import Optimizer

opt = Optimizer(algorithm="RoughSetClassifier", n_trials=50).fit(X, y)
rs = opt.best_estimator_
print(rs.reduct_)                                  # selected feature indices
for rule in rs.describe_rules(feature_names=names, max_rules=5):
    print(rule)
# IF petal length (cm) in [5.35, inf) AND ... THEN class=2 [certain, support=6, confidence=1.00]
```

Its value is interpretability (explicit rules and the selected attributes), not raw accuracy: discretisation loses information, so on continuous data it usually scores below the fuzzy-rough and tree models. Fuzzy-rough methods (`FRNNClassifier`, `FROVOCOClassifier` above) generalise the same lower/upper-approximation idea to graded similarity, which avoids discretisation; Pawlak's model is their special case with all-or-nothing similarity.

### Interpretable Models (EBM, RuleFit, FIGS)

Three optional glass-box model families are available. They are listed only when their package is installed (`pip install optuml[interpret]` for EBM, `pip install optuml[imodels]` for RuleFit and FIGS).

- **`ExplainableBoostingClassifier` / `ExplainableBoostingRegressor`** ([InterpretML](https://interpret.ml/), package `interpret-core`): a glass-box generalized additive model (GAM) trained by cyclic boosting; every feature's effect is a single plottable curve.
- **`RuleFitClassifier` / `RuleFitRegressor`** ([imodels](https://github.com/csinva/imodels)): a sparse linear model (Lasso) over rules extracted from tree ensembles plus, optionally, the original linear terms.
- **`FIGSClassifier` / `FIGSRegressor`** (imodels): Fast Interpretable Greedy-tree Sums, a small sum of greedily grown trees.

```python
opt = Optimizer(algorithm="FIGSClassifier", n_trials=50).fit(X, y)
print(opt.best_estimator_)   # a readable sum of trees
```

Search spaces:

| Model   | Tuned parameters |
| ------- | ---------------- |
| EBM     | `learning_rate` 1e-3..0.2 (log), `max_bins` 32..1024 (log), `max_leaves` 2..4, `min_samples_leaf` 2..20 |
| RuleFit | `n_estimators` 50..200, `tree_size` 2..8, `max_rules` 10..60, `include_linear` |
| FIGS    | `max_rules` 2..40, `min_impurity_decrease` 0..0.05, `max_trees` and `max_depth` (`None` reachable via sentinels) |

Caveats:

- **EBMs are main-effects-only.** OptuML trains EBMs with `interactions=0`, `outer_bags=4` and `n_jobs=1`. Pairwise-interaction detection made each CV fit 10-50x slower (over 4 minutes per 5-fold CV on breast cancer with library defaults), so EBM's default `"3x"` interactions are intentionally not searched.
- **NaN:** only EBM accepts `NaN` in `X`; RuleFit and FIGS reject it.
- **Multiclass RuleFit:** imodels' RuleFit is binary-only, so OptuML wraps `RuleFitClassifier` in a `OneVsRestClassifier` (one rule set per class, in `best_estimator_.estimators_`). With two classes this is a single RuleFit model.
- **RuleFit is slow-ish.** Each fit takes about 1-4 s because of its inner Lasso path, so 50+ trials take minutes.

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

### Feature Scaling

Many algorithms (SVM, KNN, MLP, and regularized-linear models) perform poorly on unscaled features, while trees are scale-invariant. The `scale` parameter controls standardization so every algorithm competes on equal footing:

```python
# scale="auto" (default): standardize only scale-sensitive algorithms
optimizer = Optimizer(algorithm="SVC", n_trials=50)          # SVC gets a StandardScaler
optimizer = Optimizer(algorithm="RandomForestClassifier")    # trees are left as-is

# Force or disable scaling explicitly
Optimizer(algorithm="SVC", scale=True)    # always standardize
Optimizer(algorithm="SVC", scale=False)   # never standardize
```

The scaler is fit **inside each cross-validation fold** (and on the final training data), so no information leaks from validation folds. When scaling applies, `best_estimator_` is a `Pipeline([("scaler", StandardScaler()), ("model", ...)])`. Sparse input keeps its sparsity (`with_mean=False`).

For `SVR`, `LinearSVR`, `MLPRegressor` and `SGDRegressor` the target is standardized as well (the model is wrapped in a `TransformedTargetRegressor`, so `best_estimator_.named_steps["model"]` is that wrapper and predictions are returned on the original scale). Their `epsilon`/`C`/learning-rate ranges assume a unit-scale target.

Missing values: `HistGradientBoosting*`, `XGB*`, `LGBM*`, `CatBoost*` and `ExplainableBoosting*` accept `NaN` in `X` (they handle it natively); all other algorithms still reject it.

### Unbiased Evaluation (Nested Cross-Validation)

`best_score_` is the best score found *during* the search and is therefore an optimistic estimate of generalization. For an unbiased number to report, use `nested_score()`, which runs a full, independent optimization on each outer fold:

```python
optimizer = Optimizer(algorithm="SVC", n_trials=50, random_state=42)

scores = optimizer.nested_score(X, y, outer_cv=5)   # one full optimization per fold
print(f"Unbiased estimate: {scores.mean():.3f} +/- {scores.std():.3f}")

optimizer.fit(X, y)
print(f"Optimistic best_score_: {optimizer.best_score_:.3f}")  # typically higher
```

It does not require `fit()` to have been called (it refits internally). Reproducible folds come from `random_state`; cross-validation always uses a shuffled, seeded `StratifiedKFold`/`KFold`, and you may also pass a splitter object as `cv`.

### Timeout Protection

Set time limits for optimization:

```python
optimizer = Optimizer(
    algorithm="MLPClassifier",
    timeout=300,  # Total study timeout: stops launching trials after 5 minutes
    cv_timeout=30,  # Per-trial timeout: prunes any trial whose CV exceeds 30 seconds
    n_trials=1000  # Upper bound on trials; timeout will usually cut this short
)
```

> **`cv_timeout` is a soft timeout.** When it fires, the trial is pruned and the search moves on immediately, but the CV work cannot be killed (Python threads are not cancellable) and keeps running in the background until it finishes. Repeated timeouts can pile up concurrent background jobs that slow down later trials, so set it generously and treat it as a safety guard rather than a hard limit.

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

### Algorithm Benchmarking

`AlgorithmBenchmark` runs every supported algorithm (or a chosen subset) on your data, optimizes each one independently, and reports a ranked comparison — without any scikit-learn estimator constraints. Check [sample script](examples/example-benchmark-all.py) and [outputs](examples/benchmark_results/).

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from optuml import AlgorithmBenchmark

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bench = AlgorithmBenchmark(
    task="classification",   # "classification" or "regression"
    n_trials=50,
    random_state=42,
)
bench.fit(X_train, y_train)

# Ranked results as a DataFrame (requires pandas) or list of dicts
print(bench.summary())
#                      algorithm  best_score  n_trials_completed  fit_time error
# 0         RandomForestClassifier    0.983333                  50      4.21  None
# 1           ExtraTreesClassifier    0.975000                  50      3.87  None
# ...

print(bench.best_algorithm_)   # e.g. "RandomForestClassifier"
print(bench.best_score_)       # best CV score across all algorithms

# Use the winning estimator directly
predictions = bench.best_estimator_.predict(X_test)

# Or drill into any individual optimizer
rf_optimizer = bench.optimizers_["RandomForestClassifier"]
print(rf_optimizer.best_params_)
```

All algorithms and the dummy baseline are scored on **the same CV folds**: with an int `cv`, one shuffled splitter is built per `fit()` (seeded by `random_state`, or a seed drawn once when it is `None`) and shared by every run; it is exposed as `bench.cv_`.

> **The ranking is exploratory.** Each algorithm's `best_score` is the best of its trials on the same folds it is reported on, so it is optimistically biased, and not uniformly: algorithms with larger search spaces gain more than parameter-free ones (e.g. `LinearRegression`, 1 trial). To compare finalists without bias, use [`nested_score()`](#unbiased-evaluation-nested-cross-validation):
>
> ```python
> for name in bench.summary()["algorithm"][:3]:          # top-3 finalists
>     print(name, bench.optimizers_[name].nested_score(X_train, y_train).mean())
> ```

To benchmark a specific subset of algorithms:

```python
bench = AlgorithmBenchmark(
    task="regression",
    algorithms=["Ridge", "RandomForestRegressor", "XGBRegressor"],
    n_trials=50,
    scoring="r2",
)
bench.fit(X_train, y_train)
```

Run algorithms in parallel across CPU cores with `n_jobs_algorithms`:

```python
bench = AlgorithmBenchmark(
    task="classification",
    n_trials=50,
    n_jobs_algorithms=-1,   # one process per algorithm, all cores
)
```

#### `AlgorithmBenchmark` Parameters

| Parameter                 | Type           | Default      | Description                                         |
| ------------------------- | -------------- | ------------ | --------------------------------------------------- |
| `task`                    | str            | required     | `"classification"` or `"regression"`                |
| `algorithms`              | list or `"all"`| `"all"`      | Algorithms to benchmark                             |
| `n_trials`                | int            | 50           | Optuna trials per algorithm                         |
| `timeout`                 | float/None     | None         | Per-algorithm study timeout (seconds)               |
| `cv`                      | int/splitter   | 5            | CV folds, shared by all algorithms (see `cv_`)      |
| `scoring`                 | str/None       | Auto*        | Scoring metric                                      |
| `cv_timeout`              | float          | 120          | Per-trial *soft* CV timeout (seconds)               |
| `random_state`            | int/None       | None         | Random seed forwarded to every `Optimizer`          |
| `early_stopping_patience` | int/None       | None         | Early stopping patience per algorithm               |
| `n_jobs`                  | int            | 1            | Parallel CV jobs inside each `Optimizer`            |
| `n_jobs_algorithms`       | int            | 1            | Algorithms to run in parallel (`-1` = all cores)    |
| `verbose`                 | bool/int       | False        | Verbosity forwarded to each `Optimizer`             |
| `include_dummy`           | bool           | True         | Add a dummy baseline (never wins) for reference     |
| `scale`                   | bool or "auto" | "auto"       | Scaling policy forwarded to every `Optimizer`       |

*Auto defaults: `"accuracy"` for classification, `"r2"` for regression

#### Attributes after `fit()`

| Attribute           | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `best_algorithm_`   | Name of the best-scoring algorithm                       |
| `best_estimator_`   | Fitted sklearn estimator from the winning optimizer      |
| `best_score_`       | Best CV score across all algorithms                      |
| `best_params_`      | Hyperparameters of the winning optimizer                 |
| `results_`          | List of per-algorithm result dicts (including failures)  |
| `optimizers_`       | `dict[algorithm_name, Optimizer]` for full introspection |

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
| `algorithm`               | str            | required   | ML algorithm to optimize                          |
| `n_trials`                | int            | 100        | Number of optimization trials                     |
| `cv`                      | int or splitter| 5          | CV folds (shuffled, seeded) or a splitter object  |
| `scoring`                 | str/None       | Auto*      | Scoring metric for CV                             |
| `scale`                   | bool or "auto" | "auto"     | Standardize scale-sensitive algorithms**          |
| `direction`               | str            | "maximize" | Optimization direction (keep "maximize"***)       |
| `timeout`                 | float/None     | None       | Total optimization timeout (seconds)              |
| `cv_timeout`              | float          | 120        | Single CV evaluation *soft* timeout               |
| `random_state`            | int/None       | None       | Random seed for reproducibility                   |
| `n_jobs`                  | int            | 1          | Parallel jobs for CV (-1 for all cores)           |
| `early_stopping_patience` | int/None       | None       | Trials without improvement before stopping        |
| `verbose`                 | bool/int       | False      | Verbosity level                                   |
| `show_progress_bar`       | bool           | False      | Show optimization progress                        |

*Auto defaults: "accuracy" for classifiers, "r2" for regressors
**`"auto"` scales only scale-sensitive algorithms (SVM, KNN, MLP, regularized-linear); `True`/`False` force/disable. See [Feature Scaling](#feature-scaling).
***All sklearn scoring strings are higher-is-better (including `neg_*`); only use `"minimize"` with a custom lower-is-better objective.

### Methods

| Method                 | Description                        | Available For    |
| ---------------------- | ---------------------------------- | ---------------- |
| `fit(X, y)`            | Optimize hyperparameters and train | All              |
| `predict(X)`           | Make predictions                   | All              |
| `score(X, y)`          | Evaluate model (accuracy / R²)†    | All              |
| `nested_score(X, y)`   | Unbiased estimate via nested CV    | All              |
| `predict_proba(X)`     | Predict class probabilities        | Classifiers      |
| `decision_function(X)` | Get decision values                | Some classifiers |
| `get_params()`         | Get optimizer parameters           | All              |
| `set_params(**params)` | Set optimizer parameters           | All              |

†`score()` always returns accuracy (classifiers) or R² (regressors), regardless of the `scoring` used to optimize — standard scikit-learn behavior. Use `sklearn.metrics` on `predict`/`predict_proba` for other metrics.

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

### Issue: CatBoost/XGBoost/LightGBM/fuzzy-rough-learn/interpret/imodels not available
**Solution**: Install optional dependencies:
```bash
pip install optuml[all]
# or individually:
pip install catboost xgboost lightgbm fuzzy-rough-learn interpret-core imodels
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

## Benchmark results

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
