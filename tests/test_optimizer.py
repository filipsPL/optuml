# test_optimizer.py

import pytest
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, log_loss
from optuml import Optimizer

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

    # The study should stop before completing all trials due to timeout
    assert optimizer.study_time_ <= 6  # Allowing a small buffer


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
