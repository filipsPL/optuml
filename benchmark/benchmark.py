"""
OptuML Benchmark: Compare default vs optimized hyperparameters.

Runs benchmarks and saves results to CSV. To generate plots and markdown:
    python benchmark/benchmark-plot.py

Usage:
    python benchmark/benchmark.py
"""

import csv
import os
import sys
import time
import warnings
import numpy as np
import optuna
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

# Default-parameter estimator classes
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Optional dependencies
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from optuml import Optimizer

# Suppress noisy warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
QUICK_TRIALS = 20
FULL_TRIALS = 50
PER_ALGO_TIMEOUT = 60  # seconds per optimization run

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

CLASSIFICATION_DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
}

REGRESSION_DATASETS = {
    "Diabetes": load_diabetes,
}

CLASSIFIERS = {
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "LogisticRegression": lambda **kw: LogisticRegression(max_iter=1000, **kw),
    "RidgeClassifier": RidgeClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GaussianNB": GaussianNB,
    "QDA": QDA,
    "MLPClassifier": lambda **kw: MLPClassifier(max_iter=1000, **kw),
    "SGDClassifier": lambda **kw: SGDClassifier(max_iter=1000, **kw),
}

if CATBOOST_AVAILABLE:
    CLASSIFIERS["CatBoostClassifier"] = lambda **kw: CatBoostClassifier(verbose=0, random_seed=kw.pop("random_state", None), **kw)

if XGBOOST_AVAILABLE:
    CLASSIFIERS["XGBClassifier"] = lambda **kw: XGBClassifier(verbosity=0, **kw)

if LIGHTGBM_AVAILABLE:
    CLASSIFIERS["LGBMClassifier"] = lambda **kw: LGBMClassifier(verbose=-1, **kw)

REGRESSORS = {
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "MLPRegressor": lambda **kw: MLPRegressor(max_iter=1000, **kw),
    "SGDRegressor": lambda **kw: SGDRegressor(max_iter=1000, **kw),
}

if CATBOOST_AVAILABLE:
    REGRESSORS["CatBoostRegressor"] = lambda **kw: CatBoostRegressor(verbose=0, random_seed=kw.pop("random_state", None), **kw)

if XGBOOST_AVAILABLE:
    REGRESSORS["XGBRegressor"] = lambda **kw: XGBRegressor(verbosity=0, **kw)

if LIGHTGBM_AVAILABLE:
    REGRESSORS["LGBMRegressor"] = lambda **kw: LGBMRegressor(verbose=-1, **kw)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def score_default(estimator_factory, X_train, y_train, X_test, y_test, scoring, cv):
    """Evaluate an estimator with default hyperparameters."""
    model = estimator_factory(random_state=RANDOM_STATE) if _accepts_random_state(estimator_factory) else estimator_factory()
    t0 = time.time()
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    elapsed = time.time() - t0
    return cv_scores.mean(), test_score, elapsed


def score_optuml(algorithm, X_train, y_train, X_test, y_test, scoring, cv, n_trials):
    """Evaluate an algorithm with OptuML optimization."""
    opt = Optimizer(
        algorithm=algorithm,
        n_trials=n_trials,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        timeout=PER_ALGO_TIMEOUT,
        cv_timeout=30,
    )
    t0 = time.time()
    opt.fit(X_train, y_train)
    test_score = opt.score(X_test, y_test)
    elapsed = time.time() - t0
    return opt.best_score_, test_score, elapsed, opt.best_params_


def _accepts_random_state(factory):
    """Check if factory/class accepts random_state (directly or via **kwargs)."""
    import inspect
    from inspect import Parameter

    try:
        sig = inspect.signature(factory)
        if "random_state" in sig.parameters:
            return True
        # Also return True if the factory accepts **kwargs (e.g. lambda wrappers)
        return any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (ValueError, TypeError):
        return False


def run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring):
    """Run default + quick + full for a single algorithm, with progress output."""
    sys.stdout.write(f"  {algo_name}... ")
    sys.stdout.flush()

    # Default
    try:
        def_cv, def_test, def_time = score_default(factory, X_train, y_train, X_test, y_test, scoring, CV_FOLDS)
    except Exception as e:
        def_cv, def_test, def_time = float("nan"), float("nan"), 0.0
        sys.stdout.write(f"[default failed: {e}] ")

    # OptuML quick
    try:
        quick_cv, quick_test, quick_time, _ = score_optuml(algo_name, X_train, y_train, X_test, y_test, scoring, CV_FOLDS, QUICK_TRIALS)
    except Exception as e:
        quick_cv, quick_test, quick_time = float("nan"), float("nan"), 0.0
        sys.stdout.write(f"[quick failed: {e}] ")

    # OptuML full
    try:
        full_cv, full_test, full_time, _ = score_optuml(algo_name, X_train, y_train, X_test, y_test, scoring, CV_FOLDS, FULL_TRIALS)
    except Exception as e:
        full_cv, full_test, full_time = float("nan"), float("nan"), 0.0
        sys.stdout.write(f"[full failed: {e}] ")

    total_time = def_time + quick_time + full_time
    print(f"done ({total_time:.1f}s)")

    return {
        "algorithm": algo_name,
        "def_cv": def_cv,
        "def_test": def_test,
        "quick_cv": quick_cv,
        "quick_test": quick_test,
        "full_cv": full_cv,
        "full_test": full_test,
        "time_s": total_time,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = ["dataset", "task", "algorithm", "def_cv", "def_test", "quick_cv", "quick_test", "full_cv", "full_test", "time_s"]


def save_csv(all_records, path):
    """Write all benchmark records to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_classification_benchmark(all_records):
    """Run benchmark on classification datasets, appending records to all_records."""
    print("=" * 115)
    print("CLASSIFICATION BENCHMARK")
    print("=" * 115)

    for ds_name, loader in CLASSIFICATION_DATASETS.items():
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        scoring = "accuracy"

        print(f"\n--- {ds_name} (n={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))}) ---")
        print(f"    Scoring: {scoring}, CV folds: {CV_FOLDS}, timeout: {PER_ALGO_TIMEOUT}s per run\n")

        for algo_name, factory in CLASSIFIERS.items():
            record = run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring)
            record["dataset"] = ds_name
            record["task"] = "classification"
            all_records.append(record)


def run_regression_benchmark(all_records):
    """Run benchmark on regression datasets, appending records to all_records."""
    print("\n" + "=" * 115)
    print("REGRESSION BENCHMARK")
    print("=" * 115)

    for ds_name, loader in REGRESSION_DATASETS.items():
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        scoring = "r2"

        print(f"\n--- {ds_name} (n={len(X)}, features={X.shape[1]}) ---")
        print(f"    Scoring: {scoring}, CV folds: {CV_FOLDS}, timeout: {PER_ALGO_TIMEOUT}s per run\n")

        for algo_name, factory in REGRESSORS.items():
            record = run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring)
            record["dataset"] = ds_name
            record["task"] = "regression"
            all_records.append(record)


if __name__ == "__main__":
    all_records = []
    run_classification_benchmark(all_records)
    run_regression_benchmark(all_records)

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    save_csv(all_records, csv_path)
    print("\nTo generate plots and markdown run:")
    print(f"  python benchmark/benchmark-plot.py {csv_path}")
