import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from optuml import Optimizer  # Assuming the module is named `optuml.py`

# Load dataset for classification
@pytest.fixture
def classification_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Load dataset for regression
@pytest.fixture
def regression_data():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# # Test SVM classification
# def test_svm_classification(classification_data):
#     X_train, X_test, y_train, y_test = classification_data
#     optimizer = Optimizer(algorithm="SVM", task="classification", n_trials=5, random_state=42)
#     optimizer.fit(X_train, y_train)
    
#     assert optimizer.best_params_ is not None
#     assert optimizer.best_estimator_ is not None
#     assert optimizer.score(X_test, y_test) > 0.5  # Reasonable accuracy check

# Test RandomForest regression
def test_randomforest_regression(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    optimizer = Optimizer(algorithm="RandomForest", task="regression", n_trials=5, random_state=42)
    optimizer.fit(X_train, y_train)
    
    assert optimizer.best_params_ is not None
    assert optimizer.best_estimator_ is not None
    assert optimizer.score(X_test, y_test) > 0.5  # Reasonable R2 score check

# # Test AdaBoost classification
# def test_adaboost_classification(classification_data):
#     X_train, X_test, y_train, y_test = classification_data
#     optimizer = Optimizer(algorithm="AdaBoost", task="classification", n_trials=5, random_state=42)
#     optimizer.fit(X_train, y_train)
    
#     assert optimizer.best_params_ is not None
#     assert optimizer.best_estimator_ is not None
#     assert optimizer.score(X_test, y_test) > 0.5  # Reasonable accuracy check

# # Test MLP regression
# def test_mlp_regression(regression_data):
#     X_train, X_test, y_train, y_test = regression_data
#     optimizer = Optimizer(algorithm="MLP", task="regression", n_trials=5, random_state=42)
#     optimizer.fit(X_train, y_train)
    
#     assert optimizer.best_params_ is not None
#     assert optimizer.best_estimator_ is not None
#     assert optimizer.score(X_test, y_test) > 0.5  # Reasonable R2 score check

# # Test Naive Bayes classification
# def test_naivebayes_classification(classification_data):
#     X_train, X_test, y_train, y_test = classification_data
#     optimizer = Optimizer(algorithm="NaiveBayes", task="classification", n_trials=5, random_state=42)
#     optimizer.fit(X_train, y_train)
    
#     assert optimizer.best_params_ is not None  # For Naive Bayes, there are no tunable parameters
#     assert optimizer.best_estimator_ is not None
#     assert optimizer.score(X_test, y_test) > 0.5  # Reasonable accuracy check

# # Test QDA classification
# def test_qda_classification(classification_data):
#     X_train, X_test, y_train, y_test = classification_data
#     optimizer = Optimizer(algorithm="QDA", task="classification", n_trials=5, random_state=42)
#     optimizer.fit(X_train, y_train)
    
#     assert optimizer.best_params_ is not None
#     assert optimizer.best_estimator_ is not None
#     assert optimizer.score(X_test, y_test) > 0.5  # Reasonable accuracy check
