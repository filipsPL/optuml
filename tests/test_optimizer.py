import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from optuml import Optimizer

@pytest.fixture
def binary_classification_data():
    """Generate a synthetic binary classification dataset."""
    X, y = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.mark.parametrize("algorithm", ['SVM', 'kNN', 'RandomForest', 'CatBoost', 'XGBoost', 'LogisticRegression', 'DecisionTree'])
def test_optimizer_accuracy(binary_classification_data, algorithm):
    """Test accuracy optimization for each supported algorithm."""
    X_train, X_test, y_train, y_test = binary_classification_data

    # Initialize Optimizer with accuracy optimization
    optimizer = Optimizer(algorithm=algorithm, direction="maximize", verbose=False, n_trials=10, cv=2, scoring="accuracy", random_state=42)
    
    # Fit the model
    optimizer.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = optimizer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Ensure that accuracy is above a reasonable threshold (e.g., 0.7 for this dataset)
    assert accuracy > 0.7, f"Accuracy for {algorithm} is too low: {accuracy}"

    # Check if best_params_ is populated
    assert optimizer.best_params_ is not None, f"Best parameters for {algorithm} were not found"

    # Check if optimization time was recorded
    assert optimizer.optimization_time() > 0, f"Optimization time was not recorded for {algorithm}"

@pytest.mark.parametrize("algorithm", ['SVM', 'RandomForest', 'CatBoost', 'XGBoost'])
def test_optimizer_roc_auc(binary_classification_data, algorithm):
    """Test ROC AUC optimization for selected algorithms that support predict_proba."""
    X_train, X_test, y_train, y_test = binary_classification_data

    # Initialize Optimizer with ROC AUC optimization
    optimizer = Optimizer(algorithm=algorithm, direction="maximize", verbose=False, n_trials=10, cv=2, scoring="roc_auc", random_state=42)
    
    # Fit the model
    optimizer.fit(X_train, y_train)

    # Predict probabilities and calculate ROC AUC
    y_proba = optimizer.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    # Ensure that ROC AUC is above a reasonable threshold (e.g., 0.7 for this dataset)
    assert roc_auc > 0.7, f"ROC AUC for {algorithm} is too low: {roc_auc}"

    # Check if best_params_ is populated
    assert optimizer.best_params_ is not None, f"Best parameters for {algorithm} were not found"

    # Check if optimization time was recorded
    assert optimizer.optimization_time() > 0, f"Optimization time was not recorded for {algorithm}"

def test_optimizer_direction_minimize(binary_classification_data):
    """Test the 'minimize' direction in the optimizer for accuracy."""
    X_train, X_test, y_train, y_test = binary_classification_data

    # Initialize Optimizer to minimize accuracy (artificial test, not practical)
    optimizer = Optimizer(algorithm="SVM", direction="minimize", verbose=False, n_trials=10, cv=2, scoring="accuracy", random_state=42)
    
    # Fit the model
    optimizer.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = optimizer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Check that the minimization didn't yield absurdly high accuracy (should be reasonable)
    assert accuracy < 1.0, f"Minimization should not result in perfect accuracy: {accuracy}"

# def test_optimizer_verbose_flag(binary_classification_data, capsys):
#     """Test that verbose flag controls Optuna logging output."""
#     X_train, X_test, y_train, y_test = binary_classification_data

#     # Initialize Optimizer with verbose=True
#     optimizer = Optimizer(algorithm="SVM", direction="maximize", verbose=True, n_trials=5, cv=3, scoring="accuracy", random_state=42)
    
#     # Fit the model
#     optimizer.fit(X_train, y_train)

#     # Capture the stdout output
#     captured = capsys.readouterr()

#     # Check that the output contains Optuna log info (verbose=True should produce output)
#     assert "Trial" in captured.out, "Verbose output should include trial information but it was missing."

# def test_optimizer_silent_mode(binary_classification_data, capsys):
#     """Test that the optimizer runs silently when verbose=False."""
#     X_train, X_test, y_train, y_test = binary_classification_data

#     # Initialize Optimizer with verbose=False
#     optimizer = Optimizer(algorithm="SVM", direction="maximize", verbose=False, n_trials=5, cv=3, scoring="accuracy", random_state=42)
    
#     # Fit the model
#     optimizer.fit(X_train, y_train)

#     # Capture the stdout output
#     captured = capsys.readouterr()

#     # Check that the output does not contain Optuna log info (verbose=False should produce no output)
#     assert "Trial" not in captured.out, "Verbose output should be suppressed but it was displayed."
