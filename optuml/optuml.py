import optuna

from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

import time
import warnings
from wrapt_timeout_decorator import timeout

class Optimizer(BaseEstimator):

    SUPPORTED_ALGORITHMS = [
        "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor", "RandomForestClassifier", "RandomForestRegressor",
        "AdaBoostClassifier", "AdaBoostRegressor", "MLPClassifier", "MLPRegressor", "GaussianNB", "QDA", "CatBoostClassifier",
        "CatBoostRegressor", "XGBClassifier", "XGBRegressor"
    ]

    def __init__(self,
                 algorithm="SVC",
                 direction="maximize",
                 verbose=False,
                 show_progress_bar=False,
                 n_trials=100,
                 timeout=None,
                 cv=5,
                 scoring=None,
                 cv_timeout=120,
                 random_state=None):
        """
        Initializes the optimizer with the following parameters:
        :param algorithm: Machine learning algorithm to optimize (e.g., 'SVC', 'RandomForestRegressor', etc.).
        :param direction: Optimization direction, either 'maximize' (default) or 'minimize'.
        :param verbose: If True, enables Optuna verbose logging.
        :param show_progress_bar: If True, shows the progress bar during optimization.
        :param n_trials: Number of trials for Optuna optimization (default 100).
        :param timeout: Maximum time allowed for optimization (in seconds).
        :param cv: Number of cross-validation folds (default 5).
        :param scoring: Scoring method for cross-validation (default "accuracy" for classifiers and "r2" for regressors).
        :param random_state: Random state for reproducibility (default None).
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} is not supported.")
        self.algorithm = algorithm
        self.direction = direction
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.timeout_duration = cv_timeout
        self.random_state = random_state
        self.best_params_ = None
        self.best_estimator_ = None
        self.study_time_ = None

        # Set default scoring method based on algorithm type
        if scoring is None:
            if self.algorithm in [
                    "SVC", "KNeighborsClassifier", "RandomForestClassifier", "AdaBoostClassifier", "MLPClassifier", "GaussianNB", "QDA",
                    "CatBoostClassifier", "XGBClassifier"
            ]:
                self.scoring = "accuracy"
            else:
                self.scoring = "r2"
        else:
            self.scoring = scoring

        # Set Optuna logging verbosity
        if isinstance(verbose, bool):
            optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        elif isinstance(verbose, int):
            optuna.logging.set_verbosity(verbose)

        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)

    # def _cross_val_with_timeout(self, model, X, y, cv, scoring):
    #     @timeout(dec_timeout=self.timeout_duration, use_signals=True, timeout_exception=optuna.TrialPruned)
    #     def _wrapped_cross_val():
    #         # time.sleep(30) # for testing timeout
    #         return cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
    #     return _wrapped_cross_val()

    def _cross_val_with_timeout(self, model, X, y, cv, scoring):
        @timeout(dec_timeout=self.timeout_duration, use_signals=True, timeout_exception=optuna.TrialPruned)
        def _wrapped_cross_val():
            return cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score='raise')  # error_score='raise' ensures errors are caught

        try:
            return _wrapped_cross_val()
        except optuna.TrialPruned:
            # Inform the user about the timeout and return NaN for the trial
            if self.verbose:
                print(f"Cross-validation for {self.algorithm} model timed out after {self.timeout_duration} seconds.")
            return float('nan')  # Return NaN to indicate the trial failed due to timeout



    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""

        # Define the model based on the algorithm
        if self.algorithm == "SVC":
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=self.random_state, probability=True)

        elif self.algorithm == "SVR":
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            model = SVR(C=C, gamma=gamma, kernel=kernel)

        elif self.algorithm == "KNeighborsClassifier":
            n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

        elif self.algorithm == "KNeighborsRegressor":
            n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

        elif self.algorithm == "RandomForestClassifier":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           random_state=self.random_state)

        elif self.algorithm == "RandomForestRegressor":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            model = RandomForestRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=self.random_state)

        elif self.algorithm == "AdaBoostClassifier":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME', random_state=self.random_state)

        elif self.algorithm == "AdaBoostRegressor":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=self.random_state)

        elif self.algorithm == "MLPClassifier":
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 100)])
            activation = trial.suggest_categorical("activation", ["tanh", "relu"])
            solver = trial.suggest_categorical("solver", ["adam", "sgd"])
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                  activation=activation,
                                  solver=solver,
                                  random_state=self.random_state)

        elif self.algorithm == "MLPRegressor":
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 100)])
            activation = trial.suggest_categorical("activation", ["tanh", "relu"])
            solver = trial.suggest_categorical("solver", ["adam", "sgd"])
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 activation=activation,
                                 solver=solver,
                                 random_state=self.random_state)

        elif self.algorithm == "GaussianNB":
            model = GaussianNB()

        elif self.algorithm == "QDA":
            reg_param = trial.suggest_float("reg_param", 0.0, 1.0)
            model = QDA(reg_param=reg_param)

        elif self.algorithm == "CatBoostClassifier":
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            iterations = trial.suggest_int("iterations", 100, 1000)
            model = CatBoostClassifier(depth=depth,
                                       learning_rate=learning_rate,
                                       l2_leaf_reg=l2_leaf_reg,
                                       iterations=iterations,
                                       random_state=self.random_state,
                                       verbose=False)

        elif self.algorithm == "CatBoostRegressor":
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            iterations = trial.suggest_int("iterations", 100, 1000)
            model = CatBoostRegressor(depth=depth,
                                      learning_rate=learning_rate,
                                      l2_leaf_reg=l2_leaf_reg,
                                      iterations=iterations,
                                      random_state=self.random_state,
                                      verbose=False)

        elif self.algorithm == "XGBClassifier":
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            model = XGBClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  random_state=self.random_state,
                                  use_label_encoder=False,
                                  eval_metric='logloss')

        elif self.algorithm == "XGBRegressor":
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            model = XGBRegressor(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 subsample=subsample,
                                 colsample_bytree=colsample_bytree,
                                 random_state=self.random_state)

        else:
            raise ValueError(f"Algorithm {self.algorithm} is not supported.")

        # Perform cross-validation and return the mean score
        try:
            return self._cross_val_with_timeout(model, X, y, cv=self.cv, scoring=self.scoring).mean()
        except optuna.TrialPruned:
            # Handle timeout exception specifically
            if self.verbose:
                print(f"Trial was pruned due to timeout for {self.algorithm} model.")
            return float('nan')
        except Exception as e:
            # Handle other exceptions during cross-validation
            if self.verbose:
                print(f"Trial failed with exception: {e}")
            return float('nan')



    def fit(self, X, y):
        """Fit the chosen ML model with hyperparameter optimization."""
        start_time = time.time()  # Start timing the optimization process
        study = optuna.create_study(direction=self.direction)
        study.optimize(lambda trial: self._objective(trial, X, y),
                       n_trials=self.n_trials,
                       timeout=self.timeout,
                       catch=(TimeoutError,), # A study continues to run even when a trial raises one of the exceptions specified in this argument.
                       show_progress_bar=self.show_progress_bar)
        end_time = time.time()  # End timing the optimization process

        self.study_time_ = end_time - start_time  # Manually calculate the time taken for optimization

        if len(study.trials) == 0 or study.best_trial is None:
            # No successful trials
            if self.verbose:
                print("No successful trials. Optimization failed.")
            self.best_params_ = None
            self.best_estimator_ = None
            return self

        self.best_params_ = study.best_params

        # Set the best estimator based on the algorithm
        if self.algorithm == "SVC":
            self.best_estimator_ = SVC(**self.best_params_, random_state=self.random_state, probability=True)
        elif self.algorithm == "SVR":
            self.best_estimator_ = SVR(**self.best_params_)
        elif self.algorithm == "KNeighborsClassifier":
            self.best_estimator_ = KNeighborsClassifier(**self.best_params_)
        elif self.algorithm == "KNeighborsRegressor":
            self.best_estimator_ = KNeighborsRegressor(**self.best_params_)
        elif self.algorithm == "RandomForestClassifier":
            self.best_estimator_ = RandomForestClassifier(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "RandomForestRegressor":
            self.best_estimator_ = RandomForestRegressor(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "AdaBoostClassifier":
            self.best_estimator_ = AdaBoostClassifier(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "AdaBoostRegressor":
            self.best_estimator_ = AdaBoostRegressor(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "MLPClassifier":
            self.best_estimator_ = MLPClassifier(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "MLPRegressor":
            self.best_estimator_ = MLPRegressor(**self.best_params_, random_state=self.random_state)
        elif self.algorithm == "GaussianNB":
            self.best_estimator_ = GaussianNB()  # No parameters for Naive Bayes
        elif self.algorithm == "QDA":
            self.best_estimator_ = QDA(**self.best_params_)
        elif self.algorithm == "CatBoostClassifier":
            self.best_estimator_ = CatBoostClassifier(**self.best_params_, random_state=self.random_state, verbose=False)
        elif self.algorithm == "CatBoostRegressor":
            self.best_estimator_ = CatBoostRegressor(**self.best_params_, random_state=self.random_state, verbose=False)
        elif self.algorithm == "XGBClassifier":
            self.best_estimator_ = XGBClassifier(**self.best_params_,
                                                 random_state=self.random_state,
                                                 use_label_encoder=False,
                                                 eval_metric='logloss')
        elif self.algorithm == "XGBRegressor":
            self.best_estimator_ = XGBRegressor(**self.best_params_, random_state=self.random_state)
        else:
            raise ValueError(f"Algorithm {self.algorithm} is not supported.")

        # Fit the best estimator on the full dataset
        self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """Make predictions using the best estimator"""
        if self.best_estimator_ is None:
            raise AttributeError("Estimator has not been fitted yet.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Get probability estimates using the best estimator"""
        if self.best_estimator_ is None:
            raise AttributeError("Estimator has not been fitted yet.")
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError(f"{self.algorithm} is not a classifier, hence does not support probability predictions.")


    def score(self, X, y):
        """Return the score of the model on the test data based on the selected scoring method"""
        if self.best_estimator_ is None:
            raise AttributeError("Estimator has not been fitted yet.")
        return self.best_estimator_.score(X, y)
