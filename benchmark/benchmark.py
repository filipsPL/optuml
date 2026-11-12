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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

# Default-parameter estimator classes
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
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
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, BayesianRidge, HuberRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression

# Optional dependencies (only imported when the package is installed; optuml's own
# *_AVAILABLE flags decide which algorithms are supported, so mirror them here).
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    pass

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

from optuml import Optimizer
from optuml.optuml import (
    ClassifierOptimizer,
    RegressorOptimizer,
    RoughSetClassifier,
    FRNNClassifier,
    FROVOCOClassifier,
    FRNNRegressor,
    _make_ebm,
    _make_imodels,
)

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

# The benchmarked algorithms are whatever OptuML supports in this environment
# (optional packages appear only when installed), so new algorithms are picked up
# automatically.
CLASSIFIER_NAMES = list(ClassifierOptimizer._get_supported_algorithms())
REGRESSOR_NAMES = list(RegressorOptimizer._get_supported_algorithms())

# name -> factory(random_state) building the *default-hyperparameter* estimator:
# library defaults plus the same fixed, non-tuned settings OptuML uses (iteration
# caps, verbosity off, EBM's _EBM_FIXED, ...), so only the tuned values differ.
DEFAULT_FACTORIES = {
    # --- classifiers ---
    "SVC": lambda rs: SVC(),
    "KNeighborsClassifier": lambda rs: KNeighborsClassifier(),
    "RandomForestClassifier": lambda rs: RandomForestClassifier(random_state=rs),
    "ExtraTreesClassifier": lambda rs: ExtraTreesClassifier(random_state=rs),
    "AdaBoostClassifier": lambda rs: AdaBoostClassifier(random_state=rs),
    "GradientBoostingClassifier": lambda rs: GradientBoostingClassifier(random_state=rs),
    "HistGradientBoostingClassifier": lambda rs: HistGradientBoostingClassifier(random_state=rs),
    "MLPClassifier": lambda rs: MLPClassifier(max_iter=1000, early_stopping=True, random_state=rs),
    "GaussianNB": lambda rs: GaussianNB(),
    "QDA": lambda rs: QDA(),
    "LDA": lambda rs: LDA(),
    "GaussianProcessClassifier": lambda rs: GaussianProcessClassifier(random_state=rs),
    "LinearSVC": lambda rs: LinearSVC(max_iter=5000, random_state=rs),
    "LogisticRegression": lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
    "RidgeClassifier": lambda rs: RidgeClassifier(),
    "DecisionTreeClassifier": lambda rs: DecisionTreeClassifier(random_state=rs),
    "SGDClassifier": lambda rs: SGDClassifier(max_iter=1000, tol=1e-3, random_state=rs),
    "RoughSetClassifier": lambda rs: RoughSetClassifier(),
    "CatBoostClassifier": lambda rs: CatBoostClassifier(verbose=False, allow_writing_files=False, random_state=rs),
    "XGBClassifier": lambda rs: XGBClassifier(eval_metric="logloss", verbosity=0, random_state=rs),
    "LGBMClassifier": lambda rs: LGBMClassifier(verbosity=-1, random_state=rs),
    "FRNNClassifier": lambda rs: FRNNClassifier(),
    "FROVOCOClassifier": lambda rs: FROVOCOClassifier(),
    "ExplainableBoostingClassifier": lambda rs: _make_ebm("classifier", rs),
    "RuleFitClassifier": lambda rs: _make_imodels("RuleFitClassifier", rs),
    "FIGSClassifier": lambda rs: _make_imodels("FIGSClassifier", rs),
    # --- regressors ---
    "SVR": lambda rs: SVR(),
    "KNeighborsRegressor": lambda rs: KNeighborsRegressor(),
    "RandomForestRegressor": lambda rs: RandomForestRegressor(random_state=rs),
    "ExtraTreesRegressor": lambda rs: ExtraTreesRegressor(random_state=rs),
    "AdaBoostRegressor": lambda rs: AdaBoostRegressor(random_state=rs),
    "GradientBoostingRegressor": lambda rs: GradientBoostingRegressor(random_state=rs),
    "HistGradientBoostingRegressor": lambda rs: HistGradientBoostingRegressor(random_state=rs),
    "MLPRegressor": lambda rs: MLPRegressor(max_iter=1000, early_stopping=True, random_state=rs),
    "LinearRegression": lambda rs: LinearRegression(),
    "Ridge": lambda rs: Ridge(),
    "Lasso": lambda rs: Lasso(max_iter=5000),
    "ElasticNet": lambda rs: ElasticNet(max_iter=5000),
    "DecisionTreeRegressor": lambda rs: DecisionTreeRegressor(random_state=rs),
    "SGDRegressor": lambda rs: SGDRegressor(max_iter=1000, tol=1e-3, random_state=rs),
    "GaussianProcessRegressor": lambda rs: GaussianProcessRegressor(random_state=rs),
    "KernelRidge": lambda rs: KernelRidge(),
    "PLSRegression": lambda rs: PLSRegression(max_iter=1000),
    "BayesianRidge": lambda rs: BayesianRidge(),
    "HuberRegressor": lambda rs: HuberRegressor(max_iter=1000),
    "LinearSVR": lambda rs: LinearSVR(dual=True, max_iter=5000, random_state=rs),
    "CatBoostRegressor": lambda rs: CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=rs),
    "XGBRegressor": lambda rs: XGBRegressor(verbosity=0, random_state=rs),
    "LGBMRegressor": lambda rs: LGBMRegressor(verbosity=-1, random_state=rs),
    "FRNNRegressor": lambda rs: FRNNRegressor(),
    "ExplainableBoostingRegressor": lambda rs: _make_ebm("regressor", rs),
    "RuleFitRegressor": lambda rs: _make_imodels("RuleFitRegressor", rs),
    "FIGSRegressor": lambda rs: _make_imodels("FIGSRegressor", rs),
}


def check_factories():
    """Fail loudly if a supported algorithm has no default-estimator factory."""
    missing = [n for n in CLASSIFIER_NAMES + REGRESSOR_NAMES if n not in DEFAULT_FACTORIES]
    if missing:
        raise RuntimeError(
            "benchmark.py has no default-estimator factory for supported algorithm(s): "
            + ", ".join(missing)
            + ". Add them to DEFAULT_FACTORIES so they are not silently skipped."
        )


check_factories()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cv(task):
    """Same shuffled, seeded splitter OptuML builds for an int ``cv``."""
    cls = StratifiedKFold if task == "classification" else KFold
    return cls(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def score_default(algo_name, X_train, y_train, X_test, y_test, scoring, task):
    """Evaluate an algorithm with default hyperparameters.

    The default estimator is wrapped with OptuML's own scaling logic (StandardScaler
    pipeline and target standardization where applicable), so the comparison
    against OptuML measures hyperparameter tuning only, not preprocessing.
    """
    model = DEFAULT_FACTORIES[algo_name](RANDOM_STATE)
    inner = Optimizer(algorithm=algo_name, random_state=RANDOM_STATE)._optimizer
    model = inner._wrap_model(model, X_train)
    cv = _make_cv(task)
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


def run_algo(algo_name, task, X_train, y_train, X_test, y_test, scoring):
    """Run default + quick + full for a single algorithm, with progress output."""
    sys.stdout.write(f"  {algo_name}... ")
    sys.stdout.flush()

    # Default
    try:
        def_cv, def_test, def_time = score_default(algo_name, X_train, y_train, X_test, y_test, scoring, task)
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

        for algo_name in CLASSIFIER_NAMES:
            record = run_algo(algo_name, "classification", X_train, y_train, X_test, y_test, scoring)
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

        for algo_name in REGRESSOR_NAMES:
            record = run_algo(algo_name, "regression", X_train, y_train, X_test, y_test, scoring)
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
