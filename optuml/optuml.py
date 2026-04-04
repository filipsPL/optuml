"""
OptuML: Optuna-based Machine Learning Model Optimizer
A scikit-learn compatible optimizer for automatic hyperparameter tuning
"""

import optuna
import numpy as np
import time
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Sklearn imports
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels

# Model imports
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Optional imports for CatBoost and XGBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def _make_logistic_regression(**kwargs):
    """Create LogisticRegression with sklearn version compatibility.

    sklearn >= 1.8 deprecated 'penalty' in favor of 'l1_ratio' + 'C'.
    Older sklearn requires 'penalty' and only accepts 'l1_ratio' with
    penalty='elasticnet'.
    """
    try:
        # sklearn >= 1.8: use l1_ratio directly, no penalty param
        return LogisticRegression(**kwargs)
    except TypeError:
        # sklearn < 1.8: translate l1_ratio back to penalty
        l1_ratio = kwargs.pop("l1_ratio", 0.0)
        if l1_ratio == 0.0:
            kwargs["penalty"] = "l2"
        elif l1_ratio == 1.0:
            kwargs["penalty"] = "l1"
        else:
            kwargs["penalty"] = "elasticnet"
            kwargs["l1_ratio"] = l1_ratio
        return LogisticRegression(**kwargs)


class OptimizerBase(BaseEstimator):
    """Base class for Optuna-based optimizers"""

    def __init__(
        self,
        algorithm: str,
        direction: str = "maximize",
        verbose: Union[bool, int] = False,
        show_progress_bar: bool = False,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        cv: int = 5,
        scoring: Optional[str] = None,
        cv_timeout: float = 120,
        random_state: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        algorithm : str
            Machine learning algorithm to optimize
        direction : str, default='maximize'
            Optimization direction ('maximize' or 'minimize')
        verbose : bool or int, default=False
            Verbosity level for Optuna logging
        show_progress_bar : bool, default=False
            Whether to show progress bar during optimization
        n_trials : int, default=100
            Number of trials for optimization
        timeout : float or None, default=None
            Maximum time allowed for optimization (seconds)
        cv : int, default=5
            Number of cross-validation folds
        scoring : str or None, default=None
            Scoring method for cross-validation
        cv_timeout : float, default=120
            Timeout for a single CV evaluation (seconds)
        random_state : int or None, default=None
            Random state for reproducibility
        early_stopping_patience : int or None, default=None
            Number of trials without improvement before stopping
        n_jobs : int, default=1
            Number of parallel jobs for cross-validation
        """
        self.algorithm = algorithm
        self.direction = direction
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.cv_timeout = cv_timeout
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters"""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.cv <= 1:
            raise ValueError("cv must be at least 2")
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if self.cv_timeout <= 0:
            raise ValueError("cv_timeout must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")

    def _get_optuna_verbosity_level(self):
        """Resolve verbose setting to an Optuna logging level."""
        if isinstance(self.verbose, bool):
            return optuna.logging.INFO if self.verbose else optuna.logging.WARNING
        if isinstance(self.verbose, int):
            return self.verbose
        return optuna.logging.WARNING

    def _fit_best_estimator(self, X, y):
        """Fit best_estimator_ on full data, suppressing known LightGBM/sklearn warnings."""
        if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
            # LightGBM always stores auto-generated feature names (e.g. 'Column_0') even
            # for numpy input, causing sklearn's feature-name validation warning at predict
            # time. This is a known LightGBM/sklearn integration issue that cannot be fixed
            # from outside (feature_names_in_ is a read-only property on LightGBM classes).
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_.fit(X, y)

    def _cross_val_with_timeout(self, model, X, y):
        """
        Perform cross-validation with timeout protection.

        Parameters
        ----------
        model : estimator
            The model to evaluate
        X : array-like
            Features
        y : array-like
            Target values

        Returns
        -------
        scores : array
            Cross-validation scores
        """
        def _run_cv():
            import warnings as _warnings
            with _warnings.catch_warnings():
                if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
                    _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs, error_score="raise")

        # Use ThreadPoolExecutor for timeout (works on all platforms)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_cv)
            try:
                scores = future.result(timeout=self.cv_timeout)
                return scores
            except FutureTimeoutError:
                if self.verbose:
                    print(f"Cross-validation timed out after {self.cv_timeout} seconds.")
                raise optuna.TrialPruned(f"CV timed out after {self.cv_timeout}s")
            except Exception as e:
                if self.verbose:
                    print(f"Cross-validation failed: {type(e).__name__}: {e}")
                raise optuna.TrialPruned(f"CV failed: {type(e).__name__}: {e}")

    def _get_early_stopping_callback(self):
        """Create early stopping callback for Optuna"""
        if self.early_stopping_patience is None:
            return None

        class EarlyStoppingCallback:
            def __init__(self, patience, direction):
                self.patience = patience
                self.direction = direction
                self.best_score = None
                self.counter = 0

            def __call__(self, study, trial):
                if trial.value is None:  # Pruned trial
                    return

                current_score = trial.value
                if self.best_score is None:
                    self.best_score = current_score
                    self.counter = 0
                else:
                    if self.direction == "maximize":
                        improved = current_score > self.best_score
                    else:
                        improved = current_score < self.best_score

                    if improved:
                        self.best_score = current_score
                        self.counter = 0
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            study.stop()

        return EarlyStoppingCallback(self.early_stopping_patience, self.direction)

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)"""
        return {
            "algorithm": self.algorithm,
            "direction": self.direction,
            "verbose": self.verbose,
            "show_progress_bar": self.show_progress_bar,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "cv": self.cv,
            "scoring": self.scoring,
            "cv_timeout": self.cv_timeout,
            "random_state": self.random_state,
            "early_stopping_patience": self.early_stopping_patience,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")

        # Re-validate parameters
        self._validate_params()

        return self


class ClassifierOptimizer(OptimizerBase, ClassifierMixin):
    """Optimizer for classification algorithms"""

    _estimator_type = "classifier"

    _BASE_ALGORITHMS = (
        "SVC",
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "AdaBoostClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "MLPClassifier",
        "GaussianNB",
        "QDA",
        "LogisticRegression",
        "RidgeClassifier",
        "DecisionTreeClassifier",
        "SGDClassifier",
    )

    @classmethod
    def _get_supported_algorithms(cls):
        """Return supported algorithms including optional ones."""
        algos = list(cls._BASE_ALGORITHMS)
        if CATBOOST_AVAILABLE:
            algos.append("CatBoostClassifier")
        if XGBOOST_AVAILABLE:
            algos.append("XGBClassifier")
        if LIGHTGBM_AVAILABLE:
            algos.append("LGBMClassifier")
        return algos

    def __init__(self, algorithm="SVC", scoring="accuracy", **kwargs):
        """Initialize classifier optimizer with default scoring"""
        supported = self._get_supported_algorithms()
        if algorithm not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

        super().__init__(algorithm=algorithm, scoring=scoring, **kwargs)

    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""

        # Model selection and hyperparameter suggestions
        if self.algorithm == "SVC":
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            if kernel != "linear":
                gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
                model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=self.random_state, probability=True)
            else:
                model = SVC(C=C, kernel=kernel, random_state=self.random_state, probability=True)

        elif self.algorithm == "KNeighborsClassifier":
            n_neighbors = trial.suggest_int("n_neighbors", 1, min(20, len(y) // 2))
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

        elif self.algorithm == "RandomForestClassifier":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "ExtraTreesClassifier":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "AdaBoostClassifier":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 2.0, log=True)
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state,
            )

        elif self.algorithm == "GradientBoostingClassifier":
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "HistGradientBoostingClassifier":
            max_iter = trial.suggest_int("max_iter", 50, 500)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 15)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
            l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)
            model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=self.random_state,
            )

        elif self.algorithm == "MLPClassifier":
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50), (100, 100)])
            activation = trial.suggest_categorical("activation", ["tanh", "relu", "logistic"])
            solver = trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"])
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            mlp_params = dict(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True,
            )
            if solver == "sgd":
                mlp_params["learning_rate"] = trial.suggest_categorical("learning_rate", ["constant", "adaptive"])
            model = MLPClassifier(**mlp_params)

        elif self.algorithm == "GaussianNB":
            var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)
            model = GaussianNB(var_smoothing=var_smoothing)

        elif self.algorithm == "QDA":
            reg_param = trial.suggest_float("reg_param", 0.0, 1.0)
            model = QDA(reg_param=reg_param)

        elif self.algorithm == "LogisticRegression":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["lbfgs", "newton-cg", "saga"])

            # Only saga supports elastic net (l1_ratio between 0 and 1)
            # lbfgs/newton-cg only support l2 (l1_ratio=0)
            if solver == "saga":
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            else:
                l1_ratio = 0.0

            model = _make_logistic_regression(
                C=C,
                l1_ratio=l1_ratio,
                solver=solver,
                random_state=self.random_state,
                max_iter=1000,
            )

        elif self.algorithm == "DecisionTreeClassifier":
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "RidgeClassifier":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = RidgeClassifier(alpha=alpha, random_state=self.random_state)

        elif self.algorithm == "LGBMClassifier" and LIGHTGBM_AVAILABLE:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 127)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            model = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                verbosity=-1,
            )

        elif self.algorithm == "CatBoostClassifier" and CATBOOST_AVAILABLE:
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            iterations = trial.suggest_int("iterations", 100, 1000, step=50)
            border_count = trial.suggest_int("border_count", 32, 255)
            model = CatBoostClassifier(
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                iterations=iterations,
                border_count=border_count,
                random_state=self.random_state,
                verbose=False,
                allow_writing_files=False,
            )

        elif self.algorithm == "XGBClassifier" and XGBOOST_AVAILABLE:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            gamma = trial.suggest_float("gamma", 0, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
            )

        elif self.algorithm == "SGDClassifier":
            # loss: default="hinge" (linear SVM). "log_loss" = logistic regression.
            loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"])
            # penalty: default="l2"
            penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
            # alpha: regularization strength, default=1e-4
            alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
            # l1_ratio: only used when penalty="elasticnet", default=0.15
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.15
            # learning_rate schedule: default="optimal"
            learning_rate_schedule = trial.suggest_categorical("learning_rate", ["optimal", "constant", "invscaling", "adaptive"])
            sgd_params = dict(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                learning_rate=learning_rate_schedule,
                max_iter=1000,
                random_state=self.random_state,
            )
            # eta0 only relevant for schedules other than "optimal"
            if learning_rate_schedule != "optimal":
                sgd_params["eta0"] = trial.suggest_float("eta0", 1e-4, 1.0, log=True)
            model = SGDClassifier(**sgd_params, tol=1e-3)

        else:
            raise ValueError(f"Algorithm {self.algorithm} is not implemented")

        # Perform cross-validation
        scores = self._cross_val_with_timeout(model, X, y)
        return scores.mean()

    def fit(self, X, y):
        """
        Fit the optimizer to find the best hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Store feature names before check_X_y converts DataFrame to numpy
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        # Validate input - handle sklearn version compatibility
        try:
            # Try new parameter name first (sklearn >= 1.6)
            X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], ensure_all_finite=True, ensure_2d=True)
        except TypeError:
            # Fall back to old parameter name (sklearn < 1.6)
            X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], force_all_finite=True, ensure_2d=True)

        # Store feature information
        self.n_features_in_ = X.shape[1]

        # Create and run the study
        start_time = time.time()

        self.study_ = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))

        # Add callbacks
        callbacks = []
        early_stopping_callback = self._get_early_stopping_callback()
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)

        # Apply verbosity locally to avoid polluting other Optuna loggers in the same process.
        _prev_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(self._get_optuna_verbosity_level())
        try:
            self.study_.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=self.show_progress_bar,
                callbacks=callbacks,
                catch=(Exception,),
            )
        finally:
            optuna.logging.set_verbosity(_prev_verbosity)

        self.study_time_ = time.time() - start_time
        self.n_trials_completed_ = len(self.study_.trials)

        # Check if any trials succeeded
        if self.n_trials_completed_ == 0 or self.study_.best_trial is None:
            raise RuntimeError(
                "Optimization failed: No successful trials completed. "
                "Check pruned trial messages via study_.trials for the root cause "
                "(common causes: wrong scoring for task type, cv_timeout too low, bad data)."
            )

        # Store best results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # Create and fit the best estimator
        self._create_best_estimator()
        self._fit_best_estimator(X, y)

        # Set classes_ for classifiers
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        return self

    def _create_best_estimator(self):
        """Create the best estimator with optimal parameters"""
        params = self.best_params_.copy()

        if self.algorithm == "SVC":
            self.best_estimator_ = SVC(**params, random_state=self.random_state, probability=True)
        elif self.algorithm == "KNeighborsClassifier":
            self.best_estimator_ = KNeighborsClassifier(**params)
        elif self.algorithm == "RandomForestClassifier":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = RandomForestClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "ExtraTreesClassifier":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = ExtraTreesClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "AdaBoostClassifier":
            self.best_estimator_ = AdaBoostClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "GradientBoostingClassifier":
            self.best_estimator_ = GradientBoostingClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "HistGradientBoostingClassifier":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = HistGradientBoostingClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "MLPClassifier":
            self.best_estimator_ = MLPClassifier(**params, random_state=self.random_state, max_iter=1000, early_stopping=True)
        elif self.algorithm == "GaussianNB":
            self.best_estimator_ = GaussianNB(**params)
        elif self.algorithm == "QDA":
            self.best_estimator_ = QDA(**params)
        elif self.algorithm == "LogisticRegression":
            self.best_estimator_ = _make_logistic_regression(**params, random_state=self.random_state, max_iter=1000)
        elif self.algorithm == "RidgeClassifier":
            self.best_estimator_ = RidgeClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "DecisionTreeClassifier":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = DecisionTreeClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "CatBoostClassifier" and CATBOOST_AVAILABLE:
            self.best_estimator_ = CatBoostClassifier(**params, random_state=self.random_state, verbose=False, allow_writing_files=False)
        elif self.algorithm == "XGBClassifier" and XGBOOST_AVAILABLE:
            self.best_estimator_ = XGBClassifier(**params, random_state=self.random_state, eval_metric="logloss", verbosity=0)
        elif self.algorithm == "LGBMClassifier" and LIGHTGBM_AVAILABLE:
            self.best_estimator_ = LGBMClassifier(**params, random_state=self.random_state, verbosity=-1)
        elif self.algorithm == "SGDClassifier":
            self.best_estimator_ = SGDClassifier(**params, max_iter=1000, tol=1e-3, random_state=self.random_state)
        else:
            raise ValueError(f"Cannot create estimator: algorithm '{self.algorithm}' is not available.")

    def predict(self, X):
        """Make predictions using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = check_array(X, accept_sparse=["csc", "csr"], ensure_2d=True)
        if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict(X)
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = check_array(X, accept_sparse=["csc", "csr"], ensure_2d=True)

        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(f"{self.algorithm} does not support probability predictions")

        if LIGHTGBM_AVAILABLE and self.algorithm == "LGBMClassifier":
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict_proba(X)
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        """Get decision function values"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = check_array(X, accept_sparse=["csc", "csr"], ensure_2d=True)

        if not hasattr(self.best_estimator_, "decision_function"):
            raise AttributeError(f"{self.algorithm} does not have decision_function")

        return self.best_estimator_.decision_function(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], ensure_2d=True)
        if LIGHTGBM_AVAILABLE and self.algorithm == "LGBMClassifier":
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.score(X, y)
        return self.best_estimator_.score(X, y)


class RegressorOptimizer(OptimizerBase, RegressorMixin):
    """Optimizer for regression algorithms"""

    _estimator_type = "regressor"

    _BASE_ALGORITHMS = (
        "SVR",
        "KNeighborsRegressor",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "AdaBoostRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "MLPRegressor",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "DecisionTreeRegressor",
        "SGDRegressor",
    )

    # Algorithms with no tunable hyperparameters: run only 1 Optuna trial.
    _PARAMETER_FREE_ALGORITHMS = frozenset(["LinearRegression"])

    @classmethod
    def _get_supported_algorithms(cls):
        """Return supported algorithms including optional ones."""
        algos = list(cls._BASE_ALGORITHMS)
        if CATBOOST_AVAILABLE:
            algos.append("CatBoostRegressor")
        if XGBOOST_AVAILABLE:
            algos.append("XGBRegressor")
        if LIGHTGBM_AVAILABLE:
            algos.append("LGBMRegressor")
        return algos

    def __init__(self, algorithm="SVR", scoring="r2", **kwargs):
        """Initialize regressor optimizer with default scoring"""
        supported = self._get_supported_algorithms()
        if algorithm not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

        super().__init__(algorithm=algorithm, scoring=scoring, **kwargs)

    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""

        # Model selection and hyperparameter suggestions
        if self.algorithm == "SVR":
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            if kernel != "linear":
                gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
            else:
                model = SVR(C=C, epsilon=epsilon, kernel=kernel)

        elif self.algorithm == "KNeighborsRegressor":
            n_neighbors = trial.suggest_int("n_neighbors", 1, min(20, len(y) // 2))
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

        elif self.algorithm == "RandomForestRegressor":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "ExtraTreesRegressor":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "AdaBoostRegressor":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 2.0, log=True)
            loss = trial.suggest_categorical("loss", ["linear", "square", "exponential"])
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=self.random_state)

        elif self.algorithm == "GradientBoostingRegressor":
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "HistGradientBoostingRegressor":
            max_iter = trial.suggest_int("max_iter", 50, 500)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 15)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
            l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)
            model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=self.random_state,
            )

        elif self.algorithm == "MLPRegressor":
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50), (100, 100)])
            activation = trial.suggest_categorical("activation", ["tanh", "relu", "logistic"])
            solver = trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"])
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            mlp_params = dict(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True,
            )
            if solver == "sgd":
                mlp_params["learning_rate"] = trial.suggest_categorical("learning_rate", ["constant", "adaptive"])
            model = MLPRegressor(**mlp_params)

        elif self.algorithm == "LinearRegression":
            # LinearRegression has no regularization hyperparameters worth tuning;
            # always use defaults to avoid noise from a near-trivial search space.
            model = LinearRegression()

        elif self.algorithm == "DecisionTreeRegressor":
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = None if max_depth_none else trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"])
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                max_features=max_features,
                random_state=self.random_state,
            )

        elif self.algorithm == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = Ridge(alpha=alpha)

        elif self.algorithm == "Lasso":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            model = Lasso(alpha=alpha, max_iter=5000)

        elif self.algorithm == "ElasticNet":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

        elif self.algorithm == "LGBMRegressor" and LIGHTGBM_AVAILABLE:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 127)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            model = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                verbosity=-1,
            )

        elif self.algorithm == "CatBoostRegressor" and CATBOOST_AVAILABLE:
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            iterations = trial.suggest_int("iterations", 100, 1000, step=50)
            border_count = trial.suggest_int("border_count", 32, 255)
            model = CatBoostRegressor(
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                iterations=iterations,
                border_count=border_count,
                random_state=self.random_state,
                verbose=False,
                allow_writing_files=False,
            )

        elif self.algorithm == "XGBRegressor" and XGBOOST_AVAILABLE:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            gamma = trial.suggest_float("gamma", 0, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                verbosity=0,
            )

        elif self.algorithm == "SGDRegressor":
            # loss: default="squared_error"
            loss = trial.suggest_categorical("loss", ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
            # penalty: default="l2"
            penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
            # alpha: regularization strength, default=1e-4
            alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
            # l1_ratio: only used when penalty="elasticnet", default=0.15
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.15
            # epsilon: only relevant for huber / epsilon_insensitive losses, default=0.1
            epsilon = trial.suggest_float("epsilon", 1e-4, 1.0, log=True) if loss in ("huber", "epsilon_insensitive", "squared_epsilon_insensitive") else 0.1
            # learning_rate schedule: default="invscaling"
            learning_rate_schedule = trial.suggest_categorical("learning_rate", ["invscaling", "optimal", "constant", "adaptive"])
            sgd_params = dict(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                epsilon=epsilon,
                learning_rate=learning_rate_schedule,
                max_iter=1000,
                random_state=self.random_state,
            )
            # eta0 only relevant for schedules other than "optimal"
            if learning_rate_schedule != "optimal":
                sgd_params["eta0"] = trial.suggest_float("eta0", 1e-4, 1.0, log=True)
            model = SGDRegressor(**sgd_params, tol=1e-3)

        else:
            raise ValueError(f"Algorithm {self.algorithm} is not implemented")

        # Perform cross-validation
        scores = self._cross_val_with_timeout(model, X, y)
        return scores.mean()

    def fit(self, X, y):
        """
        Fit the optimizer to find the best hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Store feature names before check_X_y converts DataFrame to numpy
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        # Validate input - handle sklearn version compatibility
        try:
            # Try new parameter name first (sklearn >= 1.6)
            X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], ensure_all_finite=True, ensure_2d=True, y_numeric=True)
        except TypeError:
            # Fall back to old parameter name (sklearn < 1.6)
            X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], force_all_finite=True, ensure_2d=True, y_numeric=True)

        # Store feature information
        self.n_features_in_ = X.shape[1]

        # Create and run the study
        start_time = time.time()

        self.study_ = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))

        # Add callbacks
        callbacks = []
        early_stopping_callback = self._get_early_stopping_callback()
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)

        # Parameter-free algorithms have nothing to optimize: cap to 1 trial.
        n_trials = 1 if self.algorithm in self._PARAMETER_FREE_ALGORITHMS else self.n_trials

        # Apply verbosity locally to avoid polluting other Optuna loggers in the same process.
        _prev_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(self._get_optuna_verbosity_level())
        try:
            self.study_.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=n_trials,
                timeout=self.timeout,
                show_progress_bar=self.show_progress_bar,
                callbacks=callbacks,
                catch=(Exception,),
            )
        finally:
            optuna.logging.set_verbosity(_prev_verbosity)

        self.study_time_ = time.time() - start_time
        self.n_trials_completed_ = len(self.study_.trials)

        # Check if any trials succeeded
        if self.n_trials_completed_ == 0 or self.study_.best_trial is None:
            raise RuntimeError(
                "Optimization failed: No successful trials completed. "
                "Check pruned trial messages via study_.trials for the root cause "
                "(common causes: wrong scoring for task type, cv_timeout too low, bad data)."
            )

        # Store best results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # Create and fit the best estimator
        self._create_best_estimator()
        self._fit_best_estimator(X, y)

        # Set output dimensions
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        return self

    def _create_best_estimator(self):
        """Create the best estimator with optimal parameters"""
        params = self.best_params_.copy()

        if self.algorithm == "SVR":
            self.best_estimator_ = SVR(**params)
        elif self.algorithm == "KNeighborsRegressor":
            self.best_estimator_ = KNeighborsRegressor(**params)
        elif self.algorithm == "RandomForestRegressor":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = RandomForestRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "ExtraTreesRegressor":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = ExtraTreesRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "AdaBoostRegressor":
            self.best_estimator_ = AdaBoostRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "GradientBoostingRegressor":
            self.best_estimator_ = GradientBoostingRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "HistGradientBoostingRegressor":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = HistGradientBoostingRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "MLPRegressor":
            self.best_estimator_ = MLPRegressor(**params, random_state=self.random_state, max_iter=1000, early_stopping=True)
        elif self.algorithm == "LinearRegression":
            self.best_estimator_ = LinearRegression(**params)
        elif self.algorithm == "Ridge":
            self.best_estimator_ = Ridge(**params)
        elif self.algorithm == "Lasso":
            self.best_estimator_ = Lasso(**params, max_iter=5000)
        elif self.algorithm == "ElasticNet":
            self.best_estimator_ = ElasticNet(**params, max_iter=5000)
        elif self.algorithm == "DecisionTreeRegressor":
            max_depth_none = params.pop("max_depth_none", False)
            if max_depth_none:
                params.pop("max_depth", None)
            self.best_estimator_ = DecisionTreeRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "CatBoostRegressor" and CATBOOST_AVAILABLE:
            self.best_estimator_ = CatBoostRegressor(**params, random_state=self.random_state, verbose=False, allow_writing_files=False)
        elif self.algorithm == "XGBRegressor" and XGBOOST_AVAILABLE:
            self.best_estimator_ = XGBRegressor(**params, random_state=self.random_state, verbosity=0)
        elif self.algorithm == "LGBMRegressor" and LIGHTGBM_AVAILABLE:
            self.best_estimator_ = LGBMRegressor(**params, random_state=self.random_state, verbosity=-1)
        elif self.algorithm == "SGDRegressor":
            self.best_estimator_ = SGDRegressor(**params, max_iter=1000, tol=1e-3, random_state=self.random_state)
        else:
            raise ValueError(f"Cannot create estimator: algorithm '{self.algorithm}' is not available.")

    def predict(self, X):
        """Make predictions using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X = check_array(X, accept_sparse=["csc", "csr"], ensure_2d=True)
        if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict(X)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction"""
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X, y = check_X_y(X, y, accept_sparse=["csc", "csr"], ensure_2d=True, y_numeric=True)
        if LIGHTGBM_AVAILABLE and self.algorithm == "LGBMRegressor":
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.score(X, y)
        return self.best_estimator_.score(X, y)


class Optimizer(BaseEstimator):
    """
    Universal optimizer that automatically selects between classifier and regressor.

    This is a convenience wrapper that maintains backward compatibility with the
    original API while providing proper separation between classifiers and regressors.
    """

    @staticmethod
    def _get_supported_algorithms():
        """Return all supported algorithms including optional ones."""
        return ClassifierOptimizer._get_supported_algorithms() + RegressorOptimizer._get_supported_algorithms()

    @staticmethod
    def _get_classifier_algorithms():
        """Return supported classifier algorithms."""
        return ClassifierOptimizer._get_supported_algorithms()

    def __init__(
        self,
        algorithm="SVC",
        direction="maximize",
        verbose=False,
        show_progress_bar=False,
        n_trials=100,
        timeout=None,
        cv=5,
        scoring=None,
        cv_timeout=120,
        random_state=None,
        early_stopping_patience=None,
        n_jobs=1,
    ):
        """
        Initialize the universal optimizer.

        Parameters
        ----------
        algorithm : str, default='SVC'
            Machine learning algorithm to optimize
        direction : str, default='maximize'
            Optimization direction ('maximize' or 'minimize')
        verbose : bool or int, default=False
            Verbosity level
        show_progress_bar : bool, default=False
            Whether to show progress bar
        n_trials : int, default=100
            Number of optimization trials
        timeout : float or None, default=None
            Maximum time for optimization
        cv : int, default=5
            Number of CV folds
        scoring : str or None, default=None
            Scoring method (defaults to 'accuracy' for classifiers, 'r2' for regressors)
        cv_timeout : float, default=120
            Timeout for single CV evaluation
        random_state : int or None, default=None
            Random state for reproducibility
        early_stopping_patience : int or None, default=None
            Patience for early stopping
        n_jobs : int, default=1
            Number of parallel jobs
        """
        supported = self._get_supported_algorithms()
        if algorithm not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

        self.algorithm = algorithm

        # Auto-select scoring if not provided
        if scoring is None:
            if algorithm in self._get_classifier_algorithms():
                scoring = "accuracy"
            else:
                scoring = "r2"

        # Create the appropriate optimizer
        if algorithm in self._get_classifier_algorithms():
            self._optimizer = ClassifierOptimizer(
                algorithm=algorithm,
                direction=direction,
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                n_trials=n_trials,
                timeout=timeout,
                cv=cv,
                scoring=scoring,
                cv_timeout=cv_timeout,
                random_state=random_state,
                early_stopping_patience=early_stopping_patience,
                n_jobs=n_jobs,
            )
            self._estimator_type = "classifier"
        else:
            self._optimizer = RegressorOptimizer(
                algorithm=algorithm,
                direction=direction,
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                n_trials=n_trials,
                timeout=timeout,
                cv=cv,
                scoring=scoring,
                cv_timeout=cv_timeout,
                random_state=random_state,
                early_stopping_patience=early_stopping_patience,
                n_jobs=n_jobs,
            )
            self._estimator_type = "regressor"

        # Store all parameters for get_params
        self.direction = direction
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.cv_timeout = cv_timeout
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the optimizer"""
        self._optimizer.fit(X, y)

        # Mirror all fitted attributes
        self.best_params_ = self._optimizer.best_params_
        self.best_estimator_ = self._optimizer.best_estimator_
        self.best_score_ = self._optimizer.best_score_
        self.study_time_ = self._optimizer.study_time_
        self.study_ = self._optimizer.study_
        self.n_trials_completed_ = self._optimizer.n_trials_completed_
        self.n_features_in_ = self._optimizer.n_features_in_
        self.n_outputs_ = self._optimizer.n_outputs_

        if hasattr(self._optimizer, "classes_"):
            self.classes_ = self._optimizer.classes_
            self.n_classes_ = self._optimizer.n_classes_

        if hasattr(self._optimizer, "feature_names_in_"):
            self.feature_names_in_ = self._optimizer.feature_names_in_

        return self

    def predict(self, X):
        """Make predictions"""
        return self._optimizer.predict(X)

    def predict_proba(self, X):
        """Get probability estimates (classifiers only)"""
        if self._estimator_type != "classifier":
            raise AttributeError(f"{self.algorithm} does not support probability predictions. " "This method is only available for classifiers.")
        return self._optimizer.predict_proba(X)

    def decision_function(self, X):
        """Get decision function values (some classifiers only)"""
        if self._estimator_type != "classifier":
            raise AttributeError(f"{self.algorithm} does not have decision_function. " "This method is only available for some classifiers.")
        return self._optimizer.decision_function(X)

    def score(self, X, y):
        """Return the score of the model on the test data"""
        return self._optimizer.score(X, y)

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            "algorithm": self.algorithm,
            "direction": self.direction,
            "verbose": self.verbose,
            "show_progress_bar": self.show_progress_bar,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "cv": self.cv,
            "scoring": self.scoring,
            "cv_timeout": self.cv_timeout,
            "random_state": self.random_state,
            "early_stopping_patience": self.early_stopping_patience,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        """Set parameters for this estimator"""
        if "algorithm" in params and params["algorithm"] != self.algorithm:
            # Changing algorithm requires recreating the internal optimizer.
            # Merge current stored params with the caller's overrides so that
            # previously set values (n_trials, cv, etc.) are preserved.
            # Reset scoring to None so auto-selection fires for the new algorithm
            # unless the caller explicitly provided a scoring value.
            current = self.get_params()
            current.pop("scoring")  # let __init__ auto-select unless overridden
            current.update(params)
            self.__init__(**current)
        else:
            # Update existing parameters on both this object and the internal optimizer.
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Invalid parameter '{key}' for estimator {type(self).__name__}.")
                if hasattr(self._optimizer, key):
                    setattr(self._optimizer, key, value)

            # Re-validate parameters in the internal optimizer.
            self._optimizer._validate_params()

        return self
