from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from optuml import Optimizer

# Load dataset for classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize Optimizer for SVM classifier
optimizer = Optimizer(algorithm="SVC", direction="maximize", n_trials=100, cv=3, scoring="accuracy", random_state=42, verbose=True)

# Fit the model with optimization
optimizer.fit(X_train, y_train)

# Evaluate the model
accuracy = optimizer.score(X_test, y_test)
print(f"Best parameters found: {optimizer.best_params_}")
print(f"Test set accuracy: {accuracy:.4f}")




# Load dataset for regression
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Initialize Optimizer for RandomForest regressor
optimizer = Optimizer(algorithm="SVR", direction="maximize", n_trials=100, cv=3, scoring="r2", random_state=42, verbose=True)

# Fit the model with optimization
optimizer.fit(X_train, y_train)

# Evaluate the model
r2_score = optimizer.score(X_test, y_test)
print(f"Best parameters found: {optimizer.best_params_}")
print(f"Test set R² score: {r2_score:.4f}")
