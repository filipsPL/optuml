from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_auc_score
from optuml import Optimizer
import numpy as np

# Load the Diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Binarize the target (e.g., split into two classes based on median)
y_binarized = Binarizer(threshold=np.median(y)).fit_transform(y.reshape(-1, 1)).ravel()

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.3, random_state=42)

# Instantiate the Optimizer and optimize for roc_auc
optimizer = Optimizer(algorithm="SVM", n_trials=50, timeout=300, cv=3, scoring="roc_auc", random_state=42, verbose=True)
optimizer.fit(X_train, y_train)

# Predict probabilities and calculate ROC AUC
y_proba = optimizer.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

print("The frst 10 probabilities:", y_proba[:10])

# Print the best hyperparameters
print(f"Best Hyperparameters: {optimizer.best_params_}")

# print AUROC
print(f"ROC AUC Score: {roc_auc}")
