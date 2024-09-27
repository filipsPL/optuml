from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from optuml import Optimizer
import time

from catboost import CatBoostClassifier

time_start = time.time()


from sklearn.datasets import make_classification

fingerprints, activities = make_classification(n_samples=50, n_features=7, random_state=42)

# ------------------------------------- #

clf = Optimizer(algorithm='KNeighborsClassifier')

probabilities = cross_val_predict(clf, fingerprints, activities, cv=2, method='predict_proba', verbose=100)
print(probabilities[:30])

process_time = time.time() - time_start
print(f'process time: {process_time/60} minutes')


# ------------------------------------- #

clf = Optimizer(algorithm='CatBoostClassifier')

probabilities = cross_val_predict(clf, fingerprints, activities, cv=2, method='predict_proba', verbose=100)
print(probabilities[:30])

process_time = time.time() - time_start
print(f'process time: {process_time/60} minutes')


