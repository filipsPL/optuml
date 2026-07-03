# test_optimizer.py

import pytest
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, log_loss
from optuml import Optimizer, AlgorithmBenchmark

# Suppress warnings for cleaner test output
import warnings
warnings.filterwarnings('ignore')


@pytest.fixture(scope="module")
def classification_data():
    """Fixture for classification dataset."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def classification_data_binary():
    """Fixture for classification dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)



@pytest.fixture(scope="module")
def regression_data():
    """Fixture for regression dataset."""
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_optimizer_initialization():
    """Test initialization of the Optimizer with supported and unsupported algorithms."""
    # Test with a supported algorithm
    optimizer = Optimizer(algorithm="SVC")
    assert optimizer.algorithm == "SVC"
    
    # Test with an unsupported algorithm
    with pytest.raises(ValueError):
        optimizer = Optimizer(algorithm="UnsupportedAlgorithm")


def test_optimizer_fit_predict_score_classification(classification_data):
    """Test fitting, predicting, and scoring for a classification task."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(
        algorithm="RandomForestClassifier",
        n_trials=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    score = optimizer.score(X_test, y_test)
    
    assert isinstance(predictions, np.ndarray)
    assert 0 <= accuracy <= 1
    assert accuracy == pytest.approx(score)


def test_optimizer_fit_predict_score_regression(regression_data):
    """Test fitting, predicting, and scoring for a regression task."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(
        algorithm="RandomForestRegressor",
        n_trials=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    r2 = r2_score(y_test, predictions)
    score = optimizer.score(X_test, y_test)
    
    assert isinstance(predictions, np.ndarray)
    assert -np.inf < r2 <= 1
    assert r2 == pytest.approx(score)


def test_predict_proba_supported_classifier(classification_data):
    """Test predict_proba method on a classifier that supports it."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(
        algorithm="RandomForestClassifier",
        n_trials=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    proba_predictions = optimizer.predict_proba(X_test)
    
    assert isinstance(proba_predictions, np.ndarray)
    assert proba_predictions.shape == (len(X_test), len(np.unique(y_train)))
    assert np.all(proba_predictions >= 0) and np.all(proba_predictions <= 1)
    assert np.allclose(proba_predictions.sum(axis=1), 1)


def test_predict_proba_supported_classifier_auc(classification_data_binary):
    """Test predict_proba method on a classifier that supports it."""
    X_train, X_test, y_train, y_test = classification_data_binary
    optimizer = Optimizer(
        algorithm="KNeighborsClassifier",
        n_trials=5,
        random_state=42,
        scoring="roc_auc"
    )
    optimizer.fit(X_train, y_train)
    proba_predictions = optimizer.predict_proba(X_test)
    
    assert isinstance(proba_predictions, np.ndarray)
    assert proba_predictions.shape == (len(X_test), len(np.unique(y_train)))
    assert np.all(proba_predictions >= 0) and np.all(proba_predictions <= 1)
    assert np.allclose(proba_predictions.sum(axis=1), 1)



def test_predict_proba_unsupported_estimator(regression_data):
    """Test predict_proba method on an estimator that does not support it."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(
        algorithm="RandomForestRegressor",
        n_trials=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    
    with pytest.raises(AttributeError):
        optimizer.predict_proba(X_test)


def test_optimizer_best_params_attribute(classification_data):
    """Test that best_params_ attribute is set after fitting."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(
        algorithm="SVC",
        n_trials=5,
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    
    assert optimizer.best_params_ is not None
    assert isinstance(optimizer.best_params_, dict)
    assert "C" in optimizer.best_params_
    assert "kernel" in optimizer.best_params_
    # gamma is only present when kernel != "linear"
    if optimizer.best_params_["kernel"] != "linear":
        assert "gamma" in optimizer.best_params_


# def test_optimizer_exception_handling(classification_data):
#     """Test that the optimizer raises an exception when all trials fail."""
#     X_train, X_test, y_train, y_test = classification_data

#     # Introduce invalid data to cause a cross-validation failure
#     X_train_invalid = X_train.copy()
#     X_train_invalid[:, 0] = np.nan  # Introduce NaNs

#     optimizer = Optimizer(
#         algorithm="RandomForestClassifier",
#         n_trials=5,
#         random_state=42,
#         verbose=True
#     )

#     with pytest.raises(ValueError):
#         optimizer.fit(X_train_invalid, y_train)


def test_optimizer_with_different_scoring(classification_data):
    """Test optimizer with a custom scoring method."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(
        algorithm="KNeighborsClassifier",
        n_trials=5,
        scoring="f1_macro",
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    assert isinstance(predictions, np.ndarray)
    assert 0 <= accuracy <= 1


def test_optimizer_timeout(classification_data):
    """Test that the optimizer respects the timeout parameter."""
    pytest.importorskip("xgboost")
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(
        algorithm="XGBClassifier",
        n_trials=1000,  # Large number of trials
        timeout=5,      # Short timeout
        random_state=42
    )
    optimizer.fit(X_train, y_train)

    # Primary intent: the timeout must stop the study well before all 1000 trials run.
    assert optimizer.n_trials_completed_ < 1000
    # Wall-clock should be close to the timeout. The bound is generous because Optuna
    # only checks the timeout between trials, so one in-flight trial can overrun it
    # (and the exact overrun depends on machine load).
    assert optimizer.study_time_ <= 15


def test_optimizer_direction_minimize(regression_data):
    """Test that the optimizer can minimize a metric."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(
        algorithm="SVR",
        n_trials=5,
        direction="minimize",
        scoring="neg_mean_squared_error",
        random_state=42
    )
    optimizer.fit(X_train, y_train)
    
    # Ensure that best_params_ is set
    assert optimizer.best_params_ is not None
    # Since direction is 'minimize', the best_value should be negative
    # (as neg_mean_squared_error returns negative values)
    # We can check that the best score is less than zero
    assert optimizer.best_estimator_ is not None


def test_optimizer_invalid_cv(classification_data):
    """Test that the optimizer raises an error with invalid cv parameter."""
    X_train, X_test, y_train, y_test = classification_data

    # optimizer = Optimizer(
    #     algorithm="SVC",
    #     n_trials=5,
    #     cv=-1,  # Invalid cv value
    #     random_state=42
    # )

    with pytest.raises(ValueError, match="cv must be at least 2"):
        optimizer = Optimizer(
            algorithm="SVC",
            n_trials=5,
            cv=-1,  # Invalid cv value
            random_state=42
        )


# ---------------------------------------------------------------------------
# New algorithm tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", [
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "RidgeClassifier",
])
def test_new_classifiers(algorithm, classification_data):
    """Test that newly added classifiers fit, predict, and score correctly."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(algorithm=algorithm, n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    score = optimizer.score(X_test, y_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert 0 <= score <= 1
    assert optimizer.best_params_ is not None
    assert optimizer.best_estimator_ is not None


@pytest.mark.parametrize("algorithm", [
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
])
def test_new_regressors(algorithm, regression_data):
    """Test that newly added regressors fit, predict, and score correctly."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(algorithm=algorithm, n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    score = optimizer.score(X_test, y_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert optimizer.best_params_ is not None
    assert optimizer.best_estimator_ is not None


def test_lgbm_classifier(classification_data):
    """Test LGBMClassifier (skipped if lightgbm not installed)."""
    pytest.importorskip("lightgbm")
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(algorithm="LGBMClassifier", n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert 0 <= optimizer.score(X_test, y_test) <= 1


def test_lgbm_regressor(regression_data):
    """Test LGBMRegressor (skipped if lightgbm not installed)."""
    pytest.importorskip("lightgbm")
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(algorithm="LGBMRegressor", n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    predictions = optimizer.predict(X_test)
    assert isinstance(predictions, np.ndarray)


def test_extra_trees_predict_proba(classification_data):
    """ExtraTreesClassifier supports predict_proba."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(algorithm="ExtraTreesClassifier", n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    proba = optimizer.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.allclose(proba.sum(axis=1), 1)


def test_ridge_classifier_no_predict_proba(classification_data):
    """RidgeClassifier does not support predict_proba."""
    X_train, X_test, y_train, y_test = classification_data
    optimizer = Optimizer(algorithm="RidgeClassifier", n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    with pytest.raises(AttributeError):
        optimizer.predict_proba(X_test)


def test_linear_regression_single_trial(regression_data):
    """LinearRegression should run exactly 1 Optuna trial (no hyperparameters)."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(
        algorithm="LinearRegression", n_trials=50, random_state=42
    )
    optimizer.fit(X_train, y_train)
    assert optimizer.n_trials_completed_ == 1
    assert optimizer.best_params_ == {}


def test_hist_gradient_boosting_max_depth_none(regression_data):
    """HistGradientBoostingRegressor: max_depth=None must not appear in best_estimator_ params."""
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(
        algorithm="HistGradientBoostingRegressor", n_trials=5, random_state=42
    )
    optimizer.fit(X_train, y_train)
    estimator_params = optimizer.best_estimator_.get_params()
    # If max_depth_none was True, max_depth should be None in the final estimator
    assert "max_depth_none" not in estimator_params


# ---------------------------------------------------------------------------
# Pipeline compatibility tests
# ---------------------------------------------------------------------------

def test_pipeline_clone_preserves_params():
    """sklearn.base.clone() must preserve all Optimizer params including cv_timeout."""
    from sklearn.base import clone
    opt = Optimizer(algorithm="SVC", n_trials=5, cv_timeout=60, random_state=7)
    cloned = clone(opt)
    assert cloned.get_params() == opt.get_params()
    assert cloned._optimizer.cv_timeout == opt.cv_timeout
    assert cloned._optimizer.random_state == opt.random_state


def test_pipeline_fit_predict_classification(classification_data):
    """Optimizer fits and predicts correctly when wrapped in a Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = classification_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("opt", Optimizer(algorithm="GaussianNB", n_trials=3, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    score = pipe.score(X_test, y_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert 0 <= score <= 1


def test_pipeline_fit_predict_regression(regression_data):
    """Optimizer fits and predicts correctly when wrapped in a Pipeline (regression)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = regression_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("opt", Optimizer(algorithm="Ridge", n_trials=3, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)


def test_pipeline_set_params():
    """Pipeline.set_params must propagate parameters into the nested Optimizer."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("opt", Optimizer(algorithm="SVC", n_trials=10, random_state=0)),
    ])
    pipe.set_params(opt__n_trials=3, opt__random_state=99)

    assert pipe.named_steps["opt"].n_trials == 3
    assert pipe.named_steps["opt"].random_state == 99
    assert pipe.named_steps["opt"]._optimizer.n_trials == 3
    assert pipe.named_steps["opt"]._optimizer.random_state == 99


def test_cross_val_score_with_optimizer(classification_data):
    """cross_val_score must work with Optimizer (clones estimator internally)."""
    from sklearn.model_selection import cross_val_score

    X_train, X_test, y_train, y_test = classification_data
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    opt = Optimizer(algorithm="GaussianNB", n_trials=3, random_state=42)
    scores = cross_val_score(opt, X, y, cv=2)

    assert len(scores) == 2
    assert all(0 <= s <= 1 for s in scores)


# ---------------------------------------------------------------------------
# AlgorithmBenchmark tests
# ---------------------------------------------------------------------------

_BENCHMARK_CLASSIFIERS = ["GaussianNB", "KNeighborsClassifier"]
_BENCHMARK_REGRESSORS = ["Ridge", "KNeighborsRegressor"]


def test_benchmark_invalid_task():
    with pytest.raises(ValueError, match="task must be"):
        AlgorithmBenchmark(task="clustering")


def test_benchmark_invalid_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithms"):
        AlgorithmBenchmark(task="classification", algorithms=["Ridge"])


def test_benchmark_classification_fit_and_attributes(classification_data):
    """fit() populates all expected attributes for a classification benchmark."""
    X_train, X_test, y_train, y_test = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
        include_dummy=False,
    )
    bench.fit(X_train, y_train)

    assert hasattr(bench, "results_")
    assert hasattr(bench, "best_algorithm_")
    assert hasattr(bench, "best_score_")
    assert hasattr(bench, "best_estimator_")
    assert hasattr(bench, "best_params_")
    assert hasattr(bench, "optimizers_")

    assert bench.best_algorithm_ in _BENCHMARK_CLASSIFIERS
    assert 0 <= bench.best_score_ <= 1
    assert set(bench.optimizers_.keys()) == set(_BENCHMARK_CLASSIFIERS)
    assert len(bench.results_) == len(_BENCHMARK_CLASSIFIERS)


def test_benchmark_regression_fit_and_attributes(regression_data):
    """fit() populates all expected attributes for a regression benchmark."""
    X_train, X_test, y_train, y_test = regression_data

    bench = AlgorithmBenchmark(
        task="regression",
        algorithms=_BENCHMARK_REGRESSORS,
        n_trials=3,
        random_state=42,
    )
    bench.fit(X_train, y_train)

    assert bench.best_algorithm_ in _BENCHMARK_REGRESSORS
    assert bench.best_estimator_ is not None


def test_benchmark_best_estimator_can_predict(classification_data):
    """best_estimator_ from benchmark can predict on held-out data."""
    X_train, X_test, y_train, y_test = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
    )
    bench.fit(X_train, y_train)
    preds = bench.best_estimator_.predict(X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y_test)


def test_benchmark_summary_returns_sorted_results(classification_data):
    """summary() returns results sorted by score descending."""
    X_train, X_test, y_train, y_test = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
    )
    bench.fit(X_train, y_train)
    summary = bench.summary()

    # Works whether pandas is available or not
    try:
        import pandas as pd
        assert isinstance(summary, pd.DataFrame)
        scores = summary["best_score"].tolist()
    except ImportError:
        assert isinstance(summary, list)
        scores = [r["best_score"] for r in summary]

    # NaNs (failures) are pushed to the end; valid scores are descending
    valid_scores = [s for s in scores if s == s]
    assert valid_scores == sorted(valid_scores, reverse=True)


def test_benchmark_summary_before_fit_raises():
    bench = AlgorithmBenchmark(task="classification", algorithms=_BENCHMARK_CLASSIFIERS)
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        bench.summary()


def test_benchmark_result_fields(classification_data):
    """Each entry in results_ has the expected keys (dummy entries have extra internal keys)."""
    X_train, _, y_train, _ = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
        include_dummy=False,
    )
    bench.fit(X_train, y_train)

    expected_keys = {"algorithm", "best_score", "best_params", "n_trials_completed",
                     "fit_time", "error", "optimizer"}
    for result in bench.results_:
        assert set(result.keys()) == expected_keys
        assert result["fit_time"] > 0
        assert result["error"] is None
        assert result["optimizer"] is not None


def test_benchmark_dummy_classification(classification_data):
    """Dummy classifier appears in results_, sets dummy_score_, never wins best_algorithm_."""
    X_train, X_test, y_train, y_test = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
        include_dummy=True,
    )
    bench.fit(X_train, y_train)

    dummy_results = [r for r in bench.results_ if r.get("_is_dummy")]
    assert len(dummy_results) == 1
    assert "DummyClassifier" in dummy_results[0]["algorithm"]
    assert 0 <= dummy_results[0]["best_score"] <= 1

    assert hasattr(bench, "dummy_score_")
    assert hasattr(bench, "dummy_estimator_")
    assert bench.dummy_estimator_ is not None
    assert "DummyClassifier" not in bench.best_algorithm_

    # dummy must not appear in optimizers_
    assert all("Dummy" not in k for k in bench.optimizers_)


def test_benchmark_dummy_regression(regression_data):
    """Dummy regressor appears in results_ and sets dummy_score_."""
    X_train, _, y_train, _ = regression_data

    bench = AlgorithmBenchmark(
        task="regression",
        algorithms=_BENCHMARK_REGRESSORS,
        n_trials=3,
        random_state=42,
        include_dummy=True,
    )
    bench.fit(X_train, y_train)

    dummy_results = [r for r in bench.results_ if r.get("_is_dummy")]
    assert len(dummy_results) == 1
    assert "DummyRegressor" in dummy_results[0]["algorithm"]

    assert hasattr(bench, "dummy_score_")
    assert "DummyRegressor" not in bench.best_algorithm_


def test_benchmark_dummy_in_summary(classification_data):
    """summary() includes the dummy row with is_dummy=True."""
    X_train, _, y_train, _ = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        random_state=42,
        include_dummy=True,
    )
    bench.fit(X_train, y_train)
    summary = bench.summary()

    try:
        import pandas as pd
        dummy_rows = summary[summary["is_dummy"] == True]
        assert len(dummy_rows) == 1
        total_rows = len(summary)
    except ImportError:
        dummy_rows = [r for r in summary if r["is_dummy"]]
        assert len(dummy_rows) == 1
        total_rows = len(summary)

    assert total_rows == len(_BENCHMARK_CLASSIFIERS) + 1


def test_benchmark_include_dummy_false(classification_data):
    """include_dummy=False produces no dummy entry and no dummy_score_ attribute."""
    X_train, _, y_train, _ = classification_data

    bench = AlgorithmBenchmark(
        task="classification",
        algorithms=_BENCHMARK_CLASSIFIERS,
        n_trials=3,
        include_dummy=False,
    )
    bench.fit(X_train, y_train)

    assert not any(r.get("_is_dummy") for r in bench.results_)
    assert not hasattr(bench, "dummy_score_")
    assert len(bench.results_) == len(_BENCHMARK_CLASSIFIERS)


# ---------------------------------------------------------------------------
# Feature scaling (#3)
# ---------------------------------------------------------------------------

def test_scale_auto_wraps_scale_sensitive(classification_data):
    """scale='auto' wraps scale-sensitive algorithms (SVC) in a StandardScaler pipeline."""
    from sklearn.pipeline import Pipeline
    X_train, _, y_train, _ = classification_data
    opt = Optimizer(algorithm="SVC", n_trials=3, random_state=42).fit(X_train, y_train)
    assert isinstance(opt.best_estimator_, Pipeline)
    assert "scaler" in opt.best_estimator_.named_steps


def test_scale_auto_leaves_trees_unscaled(classification_data):
    """scale='auto' does NOT wrap scale-invariant algorithms (RandomForest)."""
    from sklearn.pipeline import Pipeline
    X_train, _, y_train, _ = classification_data
    opt = Optimizer(algorithm="RandomForestClassifier", n_trials=3, random_state=42).fit(X_train, y_train)
    assert not isinstance(opt.best_estimator_, Pipeline)


def test_scale_false_disables_scaling(classification_data):
    """scale=False keeps even scale-sensitive algorithms unwrapped."""
    from sklearn.pipeline import Pipeline
    X_train, _, y_train, _ = classification_data
    opt = Optimizer(algorithm="SVC", n_trials=3, random_state=42, scale=False).fit(X_train, y_train)
    assert not isinstance(opt.best_estimator_, Pipeline)


def test_scale_true_forces_scaling(regression_data):
    """scale=True wraps an otherwise scale-invariant algorithm."""
    from sklearn.pipeline import Pipeline
    X_train, _, y_train, _ = regression_data
    opt = Optimizer(algorithm="DecisionTreeRegressor", n_trials=3, random_state=42, scale=True).fit(X_train, y_train)
    assert isinstance(opt.best_estimator_, Pipeline)


def test_scale_invalid_value_raises():
    with pytest.raises(ValueError, match="scale must be"):
        Optimizer(algorithm="SVC", scale="sometimes")


def test_scale_param_is_cloned():
    """scale must round-trip through get_params/clone (sklearn compatibility)."""
    from sklearn.base import clone
    opt = Optimizer(algorithm="SVC", scale=False, random_state=1)
    assert opt.get_params()["scale"] is False
    assert clone(opt).get_params()["scale"] is False


# ---------------------------------------------------------------------------
# Seeded, shuffled cross-validation (#5)
# ---------------------------------------------------------------------------

def test_cv_splitter_is_shuffled_and_seeded():
    """int cv builds a shuffled, random_state-seeded stratified/plain KFold."""
    from optuml.optuml import ClassifierOptimizer, RegressorOptimizer
    from sklearn.model_selection import StratifiedKFold, KFold
    cv_c = ClassifierOptimizer(algorithm="SVC", cv=5, random_state=42)._make_cv()
    cv_r = RegressorOptimizer(algorithm="Ridge", cv=5, random_state=42)._make_cv()
    assert isinstance(cv_c, StratifiedKFold) and cv_c.shuffle and cv_c.random_state == 42
    assert isinstance(cv_r, KFold) and cv_r.shuffle and cv_r.random_state == 42


def test_same_seed_is_reproducible(regression_data):
    """Same random_state -> identical best_score_ (folds are seeded)."""
    X_train, _, y_train, _ = regression_data
    a = Optimizer(algorithm="Ridge", n_trials=6, random_state=1).fit(X_train, y_train).best_score_
    b = Optimizer(algorithm="Ridge", n_trials=6, random_state=1).fit(X_train, y_train).best_score_
    assert a == pytest.approx(b)


def test_custom_cv_splitter_accepted(classification_data):
    """A cross-validation splitter object passed as cv is used as-is."""
    from sklearn.model_selection import StratifiedKFold
    X_train, _, y_train, _ = classification_data
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    opt = Optimizer(algorithm="GaussianNB", n_trials=3, cv=splitter, random_state=0).fit(X_train, y_train)
    assert 0 <= opt.best_score_ <= 1


# ---------------------------------------------------------------------------
# Nested cross-validation (#4)
# ---------------------------------------------------------------------------

def test_nested_score_returns_per_fold_scores(classification_data):
    """nested_score returns one score per outer fold and a sane mean."""
    X_train, _, y_train, _ = classification_data
    opt = Optimizer(algorithm="GaussianNB", n_trials=3, random_state=42)
    scores = opt.nested_score(X_train, y_train, outer_cv=3)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)


def test_nested_score_does_not_require_fit(regression_data):
    """nested_score works on an unfitted optimizer (it refits per fold internally)."""
    X_train, _, y_train, _ = regression_data
    opt = Optimizer(algorithm="Ridge", n_trials=3, random_state=42)
    scores = opt.nested_score(X_train, y_train, outer_cv=3)
    assert len(scores) == 3
    # optimizer itself remains unfitted
    assert not hasattr(opt, "best_estimator_")


# ---------------------------------------------------------------------------
# XGBoost arbitrary-label support (#7)
# ---------------------------------------------------------------------------

def test_xgb_accepts_string_labels(classification_data):
    """XGBClassifier must handle arbitrary (string) labels like other classifiers."""
    pytest.importorskip("xgboost")
    X_train, X_test, y_train, y_test = classification_data
    y_train_str = np.array(["setosa", "versicolor", "virginica"])[y_train]
    opt = Optimizer(algorithm="XGBClassifier", n_trials=3, random_state=0)
    opt.fit(X_train, y_train_str)
    preds = opt.predict(X_test)
    # predictions are returned in the original label space
    assert set(np.unique(preds)).issubset({"setosa", "versicolor", "virginica"})
    assert set(opt.classes_) == {"setosa", "versicolor", "virginica"}
    proba = opt.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1)


def test_xgb_accepts_noncontiguous_int_labels(classification_data):
    """XGBClassifier must handle non-contiguous integer labels (e.g. {1,2,3})."""
    pytest.importorskip("xgboost")
    X_train, X_test, y_train, y_test = classification_data
    opt = Optimizer(algorithm="XGBClassifier", n_trials=3, random_state=0)
    opt.fit(X_train, y_train + 1)  # labels {1,2,3}
    preds = opt.predict(X_test)
    assert set(np.unique(preds)).issubset({1, 2, 3})
    score = opt.score(X_test, y_test + 1)
    assert 0 <= score <= 1
