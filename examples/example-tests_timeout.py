from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from optuml import Optimizer


print("*** Classification ***")
# Load dataset for classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize Optimizer for SVM classifier
optimizer = Optimizer(algorithm="SVC", direction="maximize", n_trials=20, cv=30, scoring="accuracy", random_state=42, show_progress_bar=True, cv_timeout=1)

# Fit the model with optimization
optimizer.fit(X_train, y_train)

# Evaluate the model
accuracy = optimizer.score(X_test, y_test)
print(f"Best parameters found: {optimizer.best_params_}")
print(f"Test set accuracy: {accuracy:.4f}")
