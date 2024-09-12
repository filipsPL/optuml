import optuna
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import time
import warnings

class Optimizer(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithm="SVM", direction="maximize", verbose=False, n_trials=100, timeout=None, cv=5, scoring="accuracy", random_state=None):
        """
        Initializes the optimizer with the following parameters:
        :param algorithm: Machine learning algorithm to optimize. Options are 'SVM', 'kNN', 'RandomForest', 'CatBoost', 'XGBoost', 'LogisticRegression', 'DecisionTree'.
        :param direction: Optimization direction, either 'maximize' (default) or 'minimize'.
        :param verbose: If True, enables Optuna verbose logging.
        :param n_trials: Number of trials for Optuna optimization (default 100)
        :param timeout: Maximum time allowed for optimization (in seconds)
        :param cv: Number of cross-validation folds (default 5)
        :param scoring: Scoring method for cross-validation (default "accuracy")
        :param random_state: Random state for reproducibility (default None)
        """
        self.algorithm = algorithm
        self.direction = direction
        self.verbosity = verbose
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_estimator_ = None
        self.study_time_ = None

        # Set Optuna logging verbosity
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Suppress XGBoost warnings about label encoder
        warnings.filterwarnings('ignore', category=UserWarning, message="`use_label_encoder` is deprecated")

    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""
        if self.algorithm == "SVM":
            C = trial.suggest_float("C", 1e-5, 1e2, log=True)
            gamma = trial.suggest_float("gamma", 1e-5, 1e1, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            if kernel == "poly":
                degree = trial.suggest_int("degree", 2, 5)
            else:
                degree = 3
            model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, random_state=self.random_state, probability=True)

        elif self.algorithm == "kNN":
            n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

        elif self.algorithm == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           random_state=self.random_state)

        elif self.algorithm == "CatBoost":
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
            depth = trial.suggest_int("depth", 4, 10)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True)
            model = CatBoostClassifier(learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
                                       verbose=0, random_state=self.random_state)

        elif self.algorithm == "XGBoost":
            eta = trial.suggest_float("eta", 1e-5, 1.0, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 32)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            model = XGBClassifier(eta=eta, max_depth=max_depth, subsample=subsample,
                                  colsample_bytree=colsample_bytree, random_state=self.random_state, use_label_encoder=False)

        elif self.algorithm == "LogisticRegression":
            C = trial.suggest_float("C", 1e-5, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs"])
            model = LogisticRegression(C=C, solver=solver, random_state=self.random_state)

        elif self.algorithm == "DecisionTree":
            max_depth = trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, random_state=self.random_state)

        # Perform cross-validation and return the mean score
        return cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring).mean()

    def fit(self, X, y):
        """Fit the chosen ML model with hyperparameter optimization"""
        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))

        start_time = time.time()
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials, timeout=self.timeout)
        self.study_time_ = time.time() - start_time

        self.best_params_ = study.best_params

        # Choose the best estimator based on the algorithm
        if self.algorithm == "SVM":
            self.best_estimator_ = SVC(**self.best_params_, random_state=self.random_state, probability=True)
        elif self.algorithm == "kNN":
            self.best_estimator_ = KNeighborsClassifier(**self.best_params_)
        elif self.algorithm == "RandomForest":
            self.best_estimator_ = RandomForestClassifier(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "CatBoost":
            self.best_estimator_ = CatBoostClassifier(**self.best_params_, verbose=0, random_state=self.random_state)
        elif self.algorithm == "XGBoost":
            self.best_estimator_ = XGBClassifier(**self.best_params_, random_state=self.random_state, use_label_encoder=False)
        elif self.algorithm == "LogisticRegression":
            self.best_estimator_ = LogisticRegression(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "DecisionTree":
            self.best_estimator_ = DecisionTreeClassifier(**self.best_params_, random_state=self.random_state)

        self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """Make predictions using the best estimator"""
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using the best estimator."""
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("This model does not support probability estimates.")

    def score(self, X, y):
        """Return the score of the model on the test data based on the selected scoring method"""
        return self.best_estimator_.score(X, y)

    def optimization_time(self):
        """Return the time taken for the optimization process"""
        return self.study_time_
