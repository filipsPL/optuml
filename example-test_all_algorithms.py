import warnings
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from optuml import Optimizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the Optimizer class (assuming it's defined as above)
# from optimizer_module import Optimizer  # Uncomment this line if Optimizer is in a separate module

# List of classifiers and regressors
classifiers = [
    "SVC", "KNeighborsClassifier", "RandomForestClassifier",
    "AdaBoostClassifier", "MLPClassifier", "NaiveBayes",
    "QDA", "CatBoostClassifier", "XGBClassifier"
]

regressors = [
    "SVR", "KNeighborsRegressor", "RandomForestRegressor",
    "AdaBoostRegressor", "MLPRegressor", "CatBoostRegressor",
    "XGBRegressor"
]

# Load datasets
iris = load_iris()
X_classification = iris.data
y_classification = iris.target

diabetes = load_diabetes()
X_regression = diabetes.data
y_regression = diabetes.target

# Split datasets into training and testing sets
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

print("=== Classification Results on Iris Dataset ===\n")

for clf_name in classifiers:
    print(f"Optimizing {clf_name}...")
    optimizer = Optimizer(
        algorithm=clf_name,
        n_trials=20,  # Using 20 trials for demonstration; increase as needed
        verbose=False,
        random_state=42,
        show_progress_bar=True
    )
    optimizer.fit(Xc_train, yc_train)
    predictions = optimizer.predict(Xc_test)
    accuracy = accuracy_score(yc_test, predictions)
    print(f"Best Parameters for {clf_name}: {optimizer.best_params_}")
    print(f"Test Accuracy: {accuracy:.4f}\n")

print("=== Regression Results on Diabetes Dataset ===\n")

for reg_name in regressors:
    print(f"Optimizing {reg_name}...")
    optimizer = Optimizer(
        algorithm=reg_name,
        n_trials=20,  # Using 20 trials for demonstration; increase as needed
        verbose=False,
        show_progress_bar=True,
        random_state=42
    )
    optimizer.fit(Xr_train, yr_train)
    predictions = optimizer.predict(Xr_test)
    r2 = r2_score(yr_test, predictions)
    print(f"Best Parameters for {reg_name}: {optimizer.best_params_}")
    print(f"Test R^2 Score: {r2:.4f}\n")
