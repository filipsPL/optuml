"""
OptuML Benchmark: Compare default vs optimized hyperparameters.

Compares algorithm performance across 4 classical datasets using:
  - Default scikit-learn hyperparameters
  - OptuML quick optimization (20 trials)
  - OptuML full optimization (50 trials)

Usage:
    python benchmark/benchmark.py
"""

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
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

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
    "AdaBoostClassifier": AdaBoostClassifier,
    "LogisticRegression": lambda **kw: LogisticRegression(max_iter=1000, **kw),
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GaussianNB": GaussianNB,
    "MLPClassifier": lambda **kw: MLPClassifier(max_iter=1000, **kw),
}

REGRESSORS = {
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "MLPRegressor": lambda **kw: MLPRegressor(max_iter=1000, **kw),
}

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
    """Check if factory/class accepts random_state."""
    import inspect

    try:
        sig = inspect.signature(factory)
        return "random_state" in sig.parameters
    except (ValueError, TypeError):
        return False


def fmt(val):
    """Format a float for table display."""
    if np.isnan(val):
        return "  ---"
    return f"{val:+.4f}" if val < 0 else f"{val:.4f}"


def print_table(rows, col_widths, headers):
    """Print a formatted table."""
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += h.ljust(w)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = ""
        for cell, w in zip(row, col_widths):
            line += str(cell).ljust(w)
        print(line)


def run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring):
    """Run default + quick + full for a single algorithm, with progress output."""
    sys.stdout.write(f"  {algo_name}... ")
    sys.stdout.flush()

    # Default
    try:
        def_cv, def_test, def_time = score_default(factory, X_train, y_train, X_test, y_test, scoring, CV_FOLDS)
    except Exception as e:
        def_cv, def_test, def_time = float("nan"), float("nan"), 0
        sys.stdout.write(f"[default failed: {e}] ")

    # OptuML quick
    try:
        quick_cv, quick_test, quick_time, _ = score_optuml(algo_name, X_train, y_train, X_test, y_test, scoring, CV_FOLDS, QUICK_TRIALS)
    except Exception as e:
        quick_cv, quick_test, quick_time = float("nan"), float("nan"), 0
        sys.stdout.write(f"[quick failed: {e}] ")

    # OptuML full
    try:
        full_cv, full_test, full_time, _ = score_optuml(algo_name, X_train, y_train, X_test, y_test, scoring, CV_FOLDS, FULL_TRIALS)
    except Exception as e:
        full_cv, full_test, full_time = float("nan"), float("nan"), 0
        sys.stdout.write(f"[full failed: {e}] ")

    total_time = def_time + quick_time + full_time
    print(f"done ({total_time:.1f}s)")

    return [
        algo_name,
        fmt(def_cv),
        fmt(def_test),
        fmt(quick_cv),
        fmt(quick_test),
        fmt(full_cv),
        fmt(full_test),
        f"{total_time:.1f}",
    ]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_classification_benchmark():
    """Run benchmark on classification datasets."""
    print("=" * 115)
    print("CLASSIFICATION BENCHMARK")
    print("=" * 115)

    for ds_name, loader in CLASSIFICATION_DATASETS.items():
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        scoring = "accuracy"

        print(f"\n--- {ds_name} (n={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))}) ---")
        print(f"    Scoring: {scoring}, CV folds: {CV_FOLDS}, " f"timeout: {PER_ALGO_TIMEOUT}s per run\n")

        rows = []
        for algo_name, factory in CLASSIFIERS.items():
            row = run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring)
            rows.append(row)

        print()
        headers = ["Algorithm", "Default CV", "Default Test", "Quick CV", "Quick Test", "Full CV", "Full Test", "Time(s)"]
        col_widths = [26, 13, 14, 13, 13, 13, 13, 10]
        print_table(rows, col_widths, headers)


def run_regression_benchmark():
    """Run benchmark on regression datasets."""
    print("\n" + "=" * 115)
    print("REGRESSION BENCHMARK")
    print("=" * 115)

    for ds_name, loader in REGRESSION_DATASETS.items():
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        scoring = "r2"

        print(f"\n--- {ds_name} (n={len(X)}, features={X.shape[1]}) ---")
        print(f"    Scoring: {scoring}, CV folds: {CV_FOLDS}, " f"timeout: {PER_ALGO_TIMEOUT}s per run\n")

        rows = []
        for algo_name, factory in REGRESSORS.items():
            row = run_algo(algo_name, factory, X_train, y_train, X_test, y_test, scoring)
            rows.append(row)

        print()
        headers = ["Algorithm", "Default CV", "Default Test", "Quick CV", "Quick Test", "Full CV", "Full Test", "Time(s)"]
        col_widths = [26, 13, 14, 13, 13, 13, 13, 10]
        print_table(rows, col_widths, headers)


def print_summary_legend():
    """Print explanation of columns."""
    print("\n" + "-" * 115)
    print("Legend:")
    print("  Default CV/Test   = scikit-learn with default hyperparameters")
    print(f"  Quick CV/Test     = OptuML with {QUICK_TRIALS} trials (timeout {PER_ALGO_TIMEOUT}s)")
    print(f"  Full CV/Test      = OptuML with {FULL_TRIALS} trials (timeout {PER_ALGO_TIMEOUT}s)")
    print("  CV                = mean cross-validation score on training set")
    print("  Test              = score on held-out test set")
    print("  Time(s)           = total wall time (default + quick + full)")


if __name__ == "__main__":
    run_classification_benchmark()
    run_regression_benchmark()
    print_summary_legend()
