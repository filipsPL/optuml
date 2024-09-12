from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuml import Optimizer
import numpy as np

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Convert the Iris dataset to a binary classification problem (class 0 vs. class 1)
X = X[y != 2]
y = y[y != 2]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of available algorithms to use
algorithms = ['SVM', 'kNN', 'RandomForest', 'CatBoost', 'XGBoost', 'LogisticRegression', 'DecisionTree']

# Iterate through each algorithm, optimize and evaluate
for algorithm in algorithms:
    print(f"Running optimization for: {algorithm}")
    
    # Instantiate the Optimizer with the current algorithm
    optimizer = Optimizer(algorithm=algorithm, direction="maximize", verbosity=False, n_trials=50, timeout=300, cv=3, scoring="accuracy", random_state=42)

    # Fit the optimizer to the training data
    optimizer.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = optimizer.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {algorithm}: {accuracy}")

    # Print the best hyperparameters found during optimization
    print(f"Best Hyperparameters for {algorithm}:\n", optimizer.best_params_)

    # Print the total time taken for optimization
    opt_time = optimizer.optimization_time()
    print(f"Optimization Time for {algorithm}: {opt_time:.2f} seconds\n")

    # Separate line for visual clarity in output
    print("="*60)
