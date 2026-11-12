"""
OptuML: Optuna-based Machine Learning Model Optimizer
A scikit-learn compatible optimizer for automatic hyperparameter tuning
"""

import optuna
import numpy as np
import time
import warnings
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Sklearn imports
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.metaestimators import available_if

# Model imports
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, HuberRegressor
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

# interpret (EBM) and imodels (RuleFit, FIGS) are only probed here and imported lazily
# where used: imodels pulls in matplotlib/mlxtend, too heavy for `import optuml`.
import importlib.util as _importlib_util

INTERPRET_AVAILABLE = _importlib_util.find_spec("interpret") is not None
IMODELS_AVAILABLE = _importlib_util.find_spec("imodels") is not None

try:
    import frlearn  # noqa: F401  (fuzzy-rough-learn, optional)

    FRLEARN_AVAILABLE = True
except ImportError:
    FRLEARN_AVAILABLE = False


def _make_logistic_regression(**kwargs):
    """Create LogisticRegression with sklearn version compatibility.

    sklearn >= 1.8 deprecated ``penalty`` (default ``"deprecated"``) in favor of
    ``l1_ratio`` + ``C``. Older sklearn only honours ``l1_ratio`` when
    ``penalty="elasticnet"``; passing ``l1_ratio`` alone is accepted but silently
    ignored (penalty stays "l2"), so the API generation is detected from the
    default value of ``penalty`` rather than by catching a TypeError.
    """
    if LogisticRegression().penalty == "deprecated":
        # sklearn >= 1.8: l1_ratio is used directly, no penalty param
        return LogisticRegression(**kwargs)
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


def _fitted_estimator_has(method):
    """available_if check: does the fitted ``best_estimator_`` expose ``method``?

    Before ``fit`` the answer is unknown, so the method is reported as available.
    This makes ``hasattr(opt, "predict_proba")`` truthful after fitting, so sklearn
    scorers (roc_auc, ...) and meta-estimators pick a method that actually exists.
    """
    def check(self):
        est = getattr(self, "best_estimator_", None)
        return True if est is None else hasattr(est, method)
    return check


def _facade_has(method):
    """available_if check for the ``Optimizer`` facade (classifiers only)."""
    def check(self):
        if self._estimator_type != "classifier":
            return False
        return _fitted_estimator_has(method)(self._optimizer)
    return check


class _LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so it accepts arbitrary class labels.

    XGBoost's sklearn classifier requires targets encoded as ``0..n_classes-1`` and
    raises on string or non-contiguous integer labels. This wrapper label-encodes
    ``y`` on ``fit`` and inverse-transforms predictions, so ``XGBClassifier`` behaves
    like every other classifier (which accept arbitrary labels). It is a plain
    ``BaseEstimator`` (single ``estimator`` param), so ``sklearn.clone`` works and it
    can be used inside ``cross_val_score``. ``decision_function`` is intentionally not
    defined (XGBoost has none), so ``hasattr(...)`` reports it correctly.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        self.estimator.fit(X, self._label_encoder.transform(y))
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return self._label_encoder.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        return self.estimator.predict_proba(X)


# ---------------------------------------------------------------------------
# fuzzy-rough-learn adapters
# ---------------------------------------------------------------------------
# frlearn does not follow the sklearn API: ``model = FRNN(...)(X, y)`` builds a
# model and ``model(X)`` returns per-class membership scores. These thin adapters
# give it fit/predict/predict_proba plus get_params, so clone, cross_val_score and
# pipelines work. frlearn range-normalises features internally (RangeNormaliser),
# so these algorithms are not in _SCALE_SENSITIVE_ALGORITHMS.

_FRLEARN_DISSIMILARITIES = ("boscovich", "euclidean", "chebyshev")
_FRLEARN_OWA_WEIGHTS = ("linear", "exponential", "reciprocally_linear")


def _frlearn_owa_weights(name):
    """Map an OWA weight-type name to an frlearn weights object."""
    from frlearn.weights import LinearWeights, ExponentialWeights, ReciprocallyLinearWeights
    return {"linear": LinearWeights, "exponential": ExponentialWeights,
            "reciprocally_linear": ReciprocallyLinearWeights}[name]()


def _dense(X, name):
    from scipy.sparse import issparse
    if issparse(X):
        raise TypeError(f"{name} (fuzzy-rough-learn) requires dense input.")
    return np.asarray(X, dtype=float)


class _FRClassifierBase(ClassifierMixin, BaseEstimator):
    """Shared fit/predict for frlearn multiclass classifiers.

    ``predict_proba`` returns the fuzzy-rough membership scores normalised to sum
    to 1 per row. They rank well (fine for roc_auc) but are not calibrated
    probabilities, so log-loss / Brier on them should be read with care.
    """

    def _build(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X, y):
        X = _dense(X, type(self).__name__)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.model_ = self._build()(X, y_idx)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        scores = np.asarray(self.model_(_dense(X, type(self).__name__)), dtype=float)
        scores = np.clip(scores, 0.0, None)
        total = scores.sum(axis=1, keepdims=True)
        n_classes = len(self.classes_)
        return np.where(total > 0, scores / np.where(total > 0, total, 1.0), 1.0 / n_classes)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class FRNNClassifier(_FRClassifierBase):
    """Fuzzy Rough Nearest Neighbour classifier (frlearn ``FRNN``), sklearn API.

    Defaults mirror frlearn: ``upper_k = lower_k = 20`` (frlearn clips k to the
    class size), linear OWA weights, Boscovich (Manhattan) dissimilarity.
    """

    def __init__(self, upper_k=20, lower_k=20, weights="linear", dissimilarity="boscovich"):
        self.upper_k = upper_k
        self.lower_k = lower_k
        self.weights = weights
        self.dissimilarity = dissimilarity

    def _build(self):
        from frlearn.classifiers import FRNN
        w = _frlearn_owa_weights(self.weights)
        return FRNN(upper_weights=w, upper_k=self.upper_k, lower_weights=_frlearn_owa_weights(self.weights),
                    lower_k=self.lower_k, dissimilarity=self.dissimilarity)


class FROVOCOClassifier(_FRClassifierBase):
    """Fuzzy Rough OVO COmbination classifier (frlearn ``FROVOCO``), sklearn API.

    Designed for imbalanced multiclass data (IFROWANN weighting above
    ``ir_threshold``). Defaults mirror frlearn: ``balanced_k=16``,
    ``ir_threshold=9``, Boscovich dissimilarity.
    """

    def __init__(self, balanced_k=16, ir_threshold=9.0, dissimilarity="boscovich"):
        self.balanced_k = balanced_k
        self.ir_threshold = ir_threshold
        self.dissimilarity = dissimilarity

    def _build(self):
        from frlearn.classifiers import FROVOCO
        return FROVOCO(balanced_k=self.balanced_k, ir_threshold=self.ir_threshold,
                       dissimilarity=self.dissimilarity)


class FRNNRegressor(RegressorMixin, BaseEstimator):
    """Fuzzy Rough Nearest Neighbour regressor (frlearn ``FRNN``), sklearn API.

    Defaults mirror frlearn: ``k=10`` neighbours, Chebyshev dissimilarity.
    Cost grows quickly with ``k``, so the search keeps it small.
    """

    def __init__(self, k=10, dissimilarity="chebyshev"):
        self.k = k
        self.dissimilarity = dissimilarity

    def fit(self, X, y):
        from frlearn.regressors import FRNN
        X = _dense(X, type(self).__name__)
        self.n_features_in_ = X.shape[1]
        self.model_ = FRNN(k=self.k, dissimilarity=self.dissimilarity)(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_(_dense(X, type(self).__name__)), dtype=float)


# ---------------------------------------------------------------------------
# Classic (Pawlak) rough-set classifier
# ---------------------------------------------------------------------------

class RoughSetClassifier(ClassifierMixin, BaseEstimator):
    """Classic rough-set rule classifier (Pawlak, 1982), sklearn API.

    Pipeline, all fit on the training data only:

    1. **Discretisation** - each feature is binned with ``KBinsDiscretizer``
       (``n_bins``, ``strategy``); two objects are *indiscernible* on a set of
       attributes when they fall in the same bins.
    2. **Reduct** - greedy QuickReduct: attributes are added one at a time, each
       time the one that most increases the dependency degree
       ``gamma(B) = |POS_B| / |U|`` (fraction of objects whose indiscernibility
       class is pure, i.e. lies in the lower approximation of its decision
       class), until ``gamma(B) >= (1 - reduct_epsilon) * gamma(all)``.
       ``reduct_epsilon > 0`` gives an *approximate reduct* (more robust to noise).
    3. **Rules** - each indiscernibility class over the reduct is a rule
       ``IF a1 in bin AND ... THEN decision``. Pure classes give *certain* rules
       (lower approximation); mixed classes give *possible* rules (boundary
       region) that predict the class distribution; they are used only if
       ``use_possible_rules``. Rules with support below ``min_support`` are
       dropped.
    4. **Prediction** - an object is matched against the rules over the full
       reduct; if nothing matches, the last-added reduct attribute is dropped and
       matching is retried (back-off), down to the class prior.

    Attributes
    ----------
    reduct_ : ndarray of int
        Selected feature indices, in order of selection (most informative first).
    dependency_ : float
        Dependency degree of the reduct on the training data.
    dependency_full_ : float
        Dependency degree of all attributes (consistency of the discretised data).
    rules_ : list of dict
        Certain and possible rules over the full reduct, see :meth:`describe_rules`.

    References
    ----------
    Pawlak Z. (1982) Rough sets. Int. J. Computer and Information Sciences 11, 341-356.
    https://en.wikipedia.org/wiki/Rough_set
    """

    def __init__(self, n_bins=5, strategy="quantile", reduct_epsilon=0.0,
                 min_support=1, use_possible_rules=True):
        self.n_bins = n_bins
        self.strategy = strategy
        self.reduct_epsilon = reduct_epsilon
        self.min_support = min_support
        self.use_possible_rules = use_possible_rules

    # -- helpers -----------------------------------------------------------
    def _make_discretizer(self):
        from sklearn.preprocessing import KBinsDiscretizer
        kw = dict(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
        available = KBinsDiscretizer().get_params()
        if "subsample" in available:
            kw["subsample"] = None  # deterministic bins, no random subsampling
        if "quantile_method" in available:
            kw["quantile_method"] = "averaged_inverted_cdf"  # sklearn >= 1.7 default-to-be
        return KBinsDiscretizer(**kw)

    @staticmethod
    def _groups(Xd, attrs):
        """Indiscernibility classes over ``attrs``: group id per row."""
        if len(attrs) == 0:
            return np.zeros(len(Xd), dtype=int)
        return np.unique(Xd[:, attrs], axis=0, return_inverse=True)[1].ravel()

    @staticmethod
    def _dependency(groups, y):
        """gamma = fraction of rows in pure indiscernibility classes (positive region)."""
        pairs = np.unique(np.stack([groups, y], axis=1), axis=0)
        n_dec = np.bincount(pairs[:, 0])  # distinct decisions per group
        pure = n_dec[groups] == 1
        return float(pure.mean())

    def _quick_reduct(self, Xd, y):
        n_features = Xd.shape[1]
        all_attrs = list(range(n_features))
        self.dependency_full_ = self._dependency(self._groups(Xd, all_attrs), y)
        target = (1.0 - self.reduct_epsilon) * self.dependency_full_
        reduct, gamma = [], self._dependency(self._groups(Xd, []), y)
        while gamma < target - 1e-12 and len(reduct) < n_features:
            best_a, best_g = None, -1.0
            for a in all_attrs:
                if a in reduct:
                    continue
                g = self._dependency(self._groups(Xd, reduct + [a]), y)
                if g > best_g:
                    best_a, best_g = a, g
            reduct.append(best_a)
            gamma = best_g
        return reduct, gamma

    def _rule_table(self, Xd, y, attrs):
        """Map condition tuple -> class-count vector, for rules over ``attrs``."""
        n_classes = len(self.classes_)
        table = {}
        keys = [tuple(r) for r in Xd[:, attrs].astype(int)] if attrs else [()] * len(Xd)
        for key, label in zip(keys, y):
            counts = table.get(key)
            if counts is None:
                counts = table[key] = np.zeros(n_classes)
            counts[label] += 1
        return table

    def _rule_ok(self, counts):
        if counts.sum() < self.min_support:
            return False
        return self.use_possible_rules or np.count_nonzero(counts) == 1

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if not 0.0 <= self.reduct_epsilon < 1.0:
            raise ValueError("reduct_epsilon must be in [0, 1)")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]

        import warnings as _warnings
        with _warnings.catch_warnings():
            # constant / low-cardinality features: sklearn warns and merges bins
            _warnings.simplefilter("ignore", UserWarning)
            self.discretizer_ = self._make_discretizer().fit(X)
        Xd = self.discretizer_.transform(X).astype(int)

        reduct, self.dependency_ = self._quick_reduct(Xd, y_idx)
        self.reduct_ = np.array(reduct, dtype=int)
        # Back-off tables: level k uses the first k reduct attributes (k=0 is the prior).
        self._tables = [self._rule_table(Xd, y_idx, reduct[:k]) for k in range(len(reduct) + 1)]

        full = self._tables[-1]
        self.rules_ = [
            {"conditions": dict(zip(reduct, key)),
             "class": np.asarray(self.classes_[int(np.argmax(c))]).item(),
             "support": int(c.sum()),
             "confidence": float(c.max() / c.sum()),
             "certain": bool(np.count_nonzero(c) == 1)}
            for key, c in full.items() if self._rule_ok(c)
        ]
        self.rules_.sort(key=lambda r: (not r["certain"], -r["support"]))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "discretizer_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        Xd = self.discretizer_.transform(X).astype(int)
        reduct = list(self.reduct_)
        prior = self._tables[0][()]
        proba = np.empty((len(X), len(self.classes_)))
        for i, row in enumerate(Xd):
            counts = prior
            for k in range(len(reduct), 0, -1):
                c = self._tables[k].get(tuple(row[reduct[:k]]))
                if c is not None and self._rule_ok(c):
                    counts = c
                    break
            proba[i] = counts / counts.sum()
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def describe_rules(self, feature_names=None, max_rules=None, certain_only=False):
        """Human-readable decision rules over the reduct, strongest first."""
        check_is_fitted(self, "rules_")
        names = feature_names if feature_names is not None else getattr(self, "feature_names_in_", None)
        lines = []
        for rule in self.rules_:
            if certain_only and not rule["certain"]:
                continue
            conds = []
            for f, b in rule["conditions"].items():
                edges = self.discretizer_.bin_edges_[f]
                lo = "-inf" if b == 0 else f"{edges[b]:.4g}"
                hi = "inf" if b >= len(edges) - 2 else f"{edges[b + 1]:.4g}"
                fname = names[f] if names is not None else f"x{f}"
                conds.append(f"{fname} in [{lo}, {hi})")
            kind = "certain" if rule["certain"] else "possible"
            lines.append(
                f"IF {' AND '.join(conds) or 'TRUE'} THEN class={rule['class']!r} "
                f"[{kind}, support={rule['support']}, confidence={rule['confidence']:.2f}]"
            )
            if max_rules is not None and len(lines) >= max_rules:
                break
        return lines


# ---------------------------------------------------------------------------
# Interpretable models: EBM (interpret) and RuleFit / FIGS (imodels)
# ---------------------------------------------------------------------------
# EBM settings fixed for tuning cost: pairwise interaction detection (default
# interactions="3x") costs ~10-50x per fit, so OptuML searches main-effects-only
# EBMs (a pure GAM: every feature's effect is one plottable curve). outer_bags is
# reduced from 14 to 4 and n_jobs=1 so OptuML, not EBM, controls parallelism.
_EBM_FIXED = dict(interactions=0, outer_bags=4, n_jobs=1)


def _make_ebm(kind, random_state, **params):
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    cls = ExplainableBoostingClassifier if kind == "classifier" else ExplainableBoostingRegressor
    if random_state is not None:
        params["random_state"] = random_state
    return cls(**_EBM_FIXED, **params)


def _suggest_ebm(trial):
    """EBM search space; defaults (max_bins=1024, max_leaves=2, min_samples_leaf=4,
    learning_rate 0.015 clf / 0.04 reg) are all inside."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        max_bins=trial.suggest_int("max_bins", 32, 1024, log=True),
        max_leaves=trial.suggest_int("max_leaves", 2, 4),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 20),
    )


def _make_imodels(name, random_state, **params):
    import imodels
    model = getattr(imodels, name)(random_state=random_state, **params)
    if name == "RuleFitClassifier":
        # imodels' RuleFit is binary-only ("does not yet support multiclass").
        # One-vs-rest gives one rule set per class; for two classes sklearn's
        # OneVsRestClassifier fits a single estimator, so binary use is unchanged.
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)
    return model


def _suggest_rulefit(trial):
    """RuleFit search space; defaults (n_estimators=100, tree_size=4, max_rules=30,
    include_linear=True) are all inside."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        tree_size=trial.suggest_int("tree_size", 2, 8),
        max_rules=trial.suggest_int("max_rules", 10, 60),
        include_linear=trial.suggest_categorical("include_linear", [True, False]),
    )


def _suggest_figs(trial):
    """FIGS search space; defaults (max_rules=12, max_trees=None, max_depth=None,
    min_impurity_decrease=0) are all inside (None via sentinels)."""
    params = dict(
        max_rules=trial.suggest_int("max_rules", 2, 40),
        min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.0, 0.05),
    )
    params["max_trees"] = None if trial.suggest_categorical("max_trees_none", [True, False]) \
        else trial.suggest_int("max_trees", 1, 5)
    params["max_depth"] = None if trial.suggest_categorical("max_depth_none", [True, False]) \
        else trial.suggest_int("max_depth", 1, 6)
    return params


def _figs_params(params):
    """Collapse FIGS sentinels in best_params_ back into estimator kwargs."""
    params = dict(params)
    for key in ("max_trees", "max_depth"):
        if params.pop(f"{key}_none", False):
            params.pop(key, None)
            params[key] = None
    return params


# Algorithms whose performance depends materially on feature scaling (distance-based,
# gradient-based, or regularized-linear models). Tree ensembles, naive Bayes, QDA and
# plain OLS are scale-invariant for prediction and are intentionally excluded, so
# scale="auto" leaves them untouched.
_SCALE_SENSITIVE_ALGORITHMS = frozenset({
    # classifiers
    "SVC", "KNeighborsClassifier", "MLPClassifier", "LogisticRegression",
    "RidgeClassifier", "SGDClassifier", "GaussianProcessClassifier", "LinearSVC",
    # regressors
    "SVR", "KNeighborsRegressor", "MLPRegressor", "Ridge", "Lasso", "ElasticNet",
    "SGDRegressor", "GaussianProcessRegressor", "KernelRidge", "BayesianRidge",
    "HuberRegressor", "LinearSVR",
})


def _make_gp_kernel(kind, nu=None):
    """Build a Gaussian-process kernel from a string name (kernel objects never go into best_params_).

    The kernel hyperparameters (amplitude, length scale, ...) are fit internally by
    marginal-likelihood maximisation, so only the kernel *family* is searched.
    """
    if kind == "rbf":
        return ConstantKernel(1.0) * RBF(1.0)
    if kind == "matern":
        return ConstantKernel(1.0) * Matern(1.0, nu=nu)
    if kind == "rational_quadratic":
        return ConstantKernel(1.0) * RationalQuadratic()
    raise ValueError(f"Unknown GP kernel kind: {kind!r}")


def _decode_hidden_layers(s):
    """Decode a string like ``"100-50"`` into a tuple of ints ``(100, 50)``.

    Optuna does not support tuple categorical choices, so MLP architectures are
    suggested as strings. Tuples/lists are passed through as tuples.
    """
    if isinstance(s, (tuple, list)):
        return tuple(int(v) for v in s)
    return tuple(int(v) for v in str(s).split("-"))


# Algorithms that handle missing values natively; for these, NaN in X is allowed
# through validation (inf is still rejected). y must always be finite.
_NAN_TOLERANT_ALGORITHMS = frozenset({
    "ExplainableBoostingClassifier", "ExplainableBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
})

# Regressors whose hyperparameter ranges (epsilon, C, learning rates) assume a
# unit-scale target; with scale on, y is standardized via TransformedTargetRegressor.
_TARGET_SCALE_ALGORITHMS = frozenset({"SVR", "MLPRegressor", "SGDRegressor", "LinearSVR"})


def _nested_cv_score(estimator, X, y, scoring, outer_cv, random_state, stratify):
    """Unbiased generalization estimate via nested cross-validation.

    Clones ``estimator`` (an unfitted optimizer) and evaluates it with
    ``cross_val_score``: each outer fold triggers a full, independent
    hyperparameter search on the outer-training data and scores on the held-out
    outer fold. The hyperparameters are therefore never selected on the data they
    are scored on, so the result is not contaminated by the search (unlike
    ``best_score_``). Returns the array of per-fold scores.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

    if isinstance(outer_cv, int):
        splitter_cls = StratifiedKFold if stratify else KFold
        outer_cv = splitter_cls(n_splits=outer_cv, shuffle=True, random_state=random_state)

    return cross_val_score(
        clone(estimator), X, y, cv=outer_cv, scoring=scoring, n_jobs=1, error_score="raise"
    )


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
        scale: Union[bool, str] = "auto",
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        algorithm : str
            Machine learning algorithm to optimize
        direction : str, default='maximize'
            Optimization direction ('maximize' or 'minimize'). Keep 'maximize'
            for all of sklearn's built-in scoring strings — they follow the
            higher-is-better convention, including the ``neg_*`` error metrics
            (e.g. 'neg_mean_squared_error'). Only use 'minimize' with a custom
            lower-is-better objective.
        verbose : bool or int, default=False
            Verbosity level for Optuna logging
        show_progress_bar : bool, default=False
            Whether to show progress bar during optimization
        n_trials : int, default=100
            Number of trials for optimization
        timeout : float or None, default=None
            Maximum time allowed for optimization (seconds)
        cv : int or cross-validation splitter, default=5
            Number of cross-validation folds. An int builds a shuffled,
            ``random_state``-seeded StratifiedKFold (classification) or KFold
            (regression) so folds are reproducible and not biased by row order.
            A splitter object is used as-is.
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
        scale : bool or {'auto'}, default='auto'
            Feature scaling policy. ``'auto'`` wraps only scale-sensitive
            algorithms (SVM, KNN, MLP, regularized-linear models) in a
            StandardScaler, leaving scale-invariant models (trees, naive Bayes,
            OLS) untouched. ``True`` always scales, ``False`` never scales. The
            scaler is fit inside each CV fold (and on the final training data),
            so no information leaks from validation folds.
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
        self.scale = scale

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters"""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if isinstance(self.cv, int) and self.cv <= 1:
            raise ValueError("cv must be at least 2")
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if self.cv_timeout <= 0:
            raise ValueError("cv_timeout must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.scale not in (True, False, "auto"):
            raise ValueError("scale must be True, False, or 'auto'")

    def _make_cv(self):
        """Build the CV splitter used for every trial.

        An int ``cv`` becomes a shuffled, ``random_state``-seeded splitter
        (StratifiedKFold for classification, KFold for regression) so folds are
        reproducible and not biased by row order. A splitter object passed as
        ``cv`` is returned unchanged.
        """
        if not isinstance(self.cv, int):
            return self.cv
        from sklearn.model_selection import StratifiedKFold, KFold
        # With random_state=None the seed is drawn once per fit() (see _init_cv_seed),
        # so every trial is scored on the same folds and trials stay comparable.
        seed = getattr(self, "_cv_seed_", self.random_state)
        if getattr(self, "_estimator_type", None) == "classifier":
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=seed)
        return KFold(n_splits=self.cv, shuffle=True, random_state=seed)

    def _init_cv_seed(self):
        """Fix the CV shuffling seed for this fit (random if random_state is None)."""
        if self.random_state is None:
            self._cv_seed_ = int(np.random.randint(0, 2**31 - 1))
        else:
            self._cv_seed_ = self.random_state

    def _finite_arg(self):
        """Value for sklearn's finite-check kwarg: allow NaN for NaN-native models."""
        return "allow-nan" if self.algorithm in _NAN_TOLERANT_ALGORITHMS else True

    def _check_X(self, X):
        """check_array with NaN policy and sklearn-version-compatible kwarg name."""
        kw = dict(accept_sparse=["csc", "csr"], ensure_2d=True)
        try:
            X = check_array(X, ensure_all_finite=self._finite_arg(), **kw)
        except TypeError:  # sklearn < 1.6
            X = check_array(X, force_all_finite=self._finite_arg(), **kw)
        self._check_n_features(X)
        return X

    def _check_n_features(self, X):
        """Reject inputs whose column count differs from the one seen in fit()."""
        expected = getattr(self, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                f"{expected} features as input."
            )

    def _check_Xy(self, X, y, **extra):
        """check_X_y with NaN policy and sklearn-version-compatible kwarg name."""
        kw = dict(accept_sparse=["csc", "csr"], ensure_2d=True, **extra)
        try:
            return check_X_y(X, y, ensure_all_finite=self._finite_arg(), **kw)
        except TypeError:  # sklearn < 1.6
            return check_X_y(X, y, force_all_finite=self._finite_arg(), **kw)

    def _should_scale(self):
        """Whether the current algorithm should be wrapped in a StandardScaler."""
        if self.scale is True:
            return True
        if self.scale is False:
            return False
        return self.algorithm in _SCALE_SENSITIVE_ALGORITHMS

    def _wrap_model(self, model, X):
        """Wrap ``model`` in a StandardScaler->model Pipeline when scaling applies.

        The scaler is a pipeline step, so cross-validation fits it on the training
        fold only (no leakage). For sparse input, centering is disabled
        (``with_mean=False``) to preserve sparsity.
        """
        if not self._should_scale():
            return model
        if (getattr(self, "_estimator_type", None) == "regressor"
                and self.algorithm in _TARGET_SCALE_ALGORITHMS):
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.preprocessing import StandardScaler as _YScaler
            model = TransformedTargetRegressor(regressor=model, transformer=_YScaler())
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import issparse
        scaler = StandardScaler(with_mean=not issparse(X))
        return Pipeline([("scaler", scaler), ("model", model)])

    def nested_score(self, X, y, outer_cv=5):
        """Unbiased generalization estimate via nested cross-validation.

        ``best_score_`` is the maximum inner-CV score over all trials and is
        therefore an optimistic estimate of generalization (the hyperparameters
        are chosen on the same folds they are scored on). ``nested_score`` instead
        refits the entire optimizer independently on each outer training fold and
        scores on the held-out fold, giving an estimate that is not contaminated
        by the search.

        This is expensive: it runs ``outer_cv`` full optimizations. Does not
        require the optimizer to be fitted first. Returns the array of per-fold
        scores; take ``.mean()`` for a point estimate.
        """
        stratify = getattr(self, "_estimator_type", None) == "classifier"
        return _nested_cv_score(self, X, y, self.scoring, outer_cv, self.random_state, stratify)

    def _get_optuna_verbosity_level(self):
        """Resolve verbose setting to an Optuna logging level."""
        if isinstance(self.verbose, bool):
            return optuna.logging.INFO if self.verbose else optuna.logging.WARNING
        if isinstance(self.verbose, int):
            return self.verbose
        return optuna.logging.WARNING

    def _make_sampler(self):
        """Multivariate, grouped TPE sampler (``group=True`` is experimental: warning silenced)."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            return optuna.samplers.TPESampler(seed=self.random_state, multivariate=True, group=True)

    @staticmethod
    def _resolve_kernel_ridge_gamma(params):
        """Collapse KernelRidge's two-step gamma ("default"/"value" + ``gamma_value``) into ``gamma``.

        "default" maps to ``gamma=None`` (sklearn default). Mutates and returns ``params``.
        """
        gamma_value = params.pop("gamma_value", None)
        if "gamma" in params:
            params["gamma"] = gamma_value if params["gamma"] == "value" else None
        return params

    @staticmethod
    def _resolve_svm_gamma(params):
        """Convert the two-step gamma encoding back into a single SVC/SVR ``gamma`` arg.

        The objective suggests a categorical ``gamma`` in {"scale", "auto", "value"} so the
        sklearn defaults ("scale"/"auto") stay reachable; when "value" is chosen a numeric
        ``gamma_value`` is suggested. Here we collapse that pair into the single ``gamma``
        keyword sklearn expects. Mutates and returns ``params``.
        """
        gamma_value = params.pop("gamma_value", None)
        if params.get("gamma") == "value":
            params["gamma"] = gamma_value
        return params

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
        # Apply feature scaling (inside the pipeline, so it is fit per-fold) and build
        # the shuffled, seeded CV splitter.
        model = self._wrap_model(model, X)
        cv = self._make_cv()

        def _run_cv():
            import warnings as _warnings
            with _warnings.catch_warnings():
                if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
                    _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs, error_score="raise")

        # ThreadPoolExecutor gives cross-platform timeouts. We deliberately do NOT use a
        # `with` block: its __exit__ calls shutdown(wait=True), which would block until a
        # slow CV finished and defeat the timeout. Instead we shut down with wait=False so
        # control returns to the caller the moment cv_timeout fires. A timed-out worker
        # thread keeps running in the background (Python threads cannot be killed) but no
        # longer holds up the optimization.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_run_cv)
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
        finally:
            executor.shutdown(wait=False)

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
            "scale": self.scale,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)"""
        valid = self.get_params()
        for key, value in params.items():
            if key in valid:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")

        # Re-validate parameters
        self._validate_params()

        return self


class ClassifierOptimizer(ClassifierMixin, OptimizerBase):
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
        "LDA",
        "GaussianProcessClassifier",
        "LinearSVC",
        "LogisticRegression",
        "RidgeClassifier",
        "DecisionTreeClassifier",
        "SGDClassifier",
        "RoughSetClassifier",
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
        if FRLEARN_AVAILABLE:
            algos.extend(["FRNNClassifier", "FROVOCOClassifier"])
        if INTERPRET_AVAILABLE:
            algos.append("ExplainableBoostingClassifier")
        if IMODELS_AVAILABLE:
            algos.extend(["RuleFitClassifier", "FIGSClassifier"])
        return algos

    def __init__(self, algorithm="SVC", scoring="accuracy", **kwargs):
        """Initialize classifier optimizer with default scoring"""
        supported = self._get_supported_algorithms()
        if algorithm not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

        super().__init__(algorithm=algorithm, scoring=scoring, **kwargs)

    def _needs_proba(self):
        """True when the scoring metric needs probabilities (SVC then needs probability=True)."""
        return isinstance(self.scoring, str) and any(
            k in self.scoring for k in ("roc_auc", "log_loss", "average_precision", "brier")
        )

    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""

        # Model selection and hyperparameter suggestions
        if self.algorithm == "SVC":
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            if kernel != "linear":
                # Keep the sklearn defaults ("scale"/"auto") reachable, plus a tunable float.
                gamma_mode = trial.suggest_categorical("gamma", ["scale", "auto", "value"])
                gamma = trial.suggest_float("gamma_value", 1e-4, 1e1, log=True) if gamma_mode == "value" else gamma_mode
                model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=self.random_state, probability=self._needs_proba())
            else:
                model = SVC(C=C, kernel=kernel, random_state=self.random_state, probability=self._needs_proba())

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
            l2_zero = trial.suggest_categorical("l2_zero", [True, False])
            l2_regularization = 0.0 if l2_zero else trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)
            model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=self.random_state,
            )

        elif self.algorithm == "MLPClassifier":
            hidden_layer_sizes = _decode_hidden_layers(trial.suggest_categorical("hidden_layer_sizes", ["50", "100", "50-50", "100-50", "100-100"]))
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

        elif self.algorithm == "LDA":
            solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
            lda_params = {"solver": solver}
            if solver != "svd":
                mode = trial.suggest_categorical("shrinkage_mode", ["none", "auto", "value"])
                if mode == "auto":
                    lda_params["shrinkage"] = "auto"
                elif mode == "value":
                    lda_params["shrinkage"] = trial.suggest_float("shrinkage", 0.0, 1.0)
            model = LDA(**lda_params)

        elif self.algorithm == "GaussianProcessClassifier":
            kind = trial.suggest_categorical("kernel", ["rbf", "matern", "rational_quadratic"])
            nu = trial.suggest_categorical("nu", [0.5, 1.5, 2.5]) if kind == "matern" else None
            n_restarts = trial.suggest_int("n_restarts_optimizer", 0, 3)
            model = GaussianProcessClassifier(
                kernel=_make_gp_kernel(kind, nu),
                n_restarts_optimizer=n_restarts,
                random_state=self.random_state,
            )

        elif self.algorithm == "LinearSVC":
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            loss = trial.suggest_categorical("loss", ["squared_hinge", "hinge"])
            penalty = trial.suggest_categorical("penalty", ["l2", "l1"]) if loss == "squared_hinge" else "l2"
            model = LinearSVC(
                C=C, loss=loss, penalty=penalty, dual=(penalty != "l1"),
                max_iter=5000, random_state=self.random_state,
            )

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
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = -1 if max_depth_none else trial.suggest_int("max_depth", 2, 20)
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
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample_freq=1,  # subsample is ignored by LightGBM unless freq > 0
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
            max_depth = trial.suggest_int("max_depth", 2, 12)
            min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 1e2, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            gamma = trial.suggest_float("gamma", 0, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            # Wrap in a label encoder: XGBoost requires labels 0..n-1 and errors on
            # string / non-contiguous integer labels, unlike the other classifiers.
            model = _LabelEncodingClassifier(XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
            ))

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

        elif self.algorithm == "ExplainableBoostingClassifier" and INTERPRET_AVAILABLE:
            model = _make_ebm("classifier", self.random_state, **_suggest_ebm(trial))

        elif self.algorithm == "RuleFitClassifier" and IMODELS_AVAILABLE:
            model = _make_imodels("RuleFitClassifier", self.random_state, **_suggest_rulefit(trial))

        elif self.algorithm == "FIGSClassifier" and IMODELS_AVAILABLE:
            model = _make_imodels("FIGSClassifier", self.random_state, **_suggest_figs(trial))

        elif self.algorithm == "RoughSetClassifier":
            # defaults: n_bins=5, quantile bins, exact reduct, min_support=1, possible rules on
            model = RoughSetClassifier(
                n_bins=trial.suggest_int("n_bins", 2, 10),
                strategy=trial.suggest_categorical("strategy", ["quantile", "uniform", "kmeans"]),
                reduct_epsilon=trial.suggest_float("reduct_epsilon", 0.0, 0.2),
                min_support=trial.suggest_int("min_support", 1, 10),
                use_possible_rules=trial.suggest_categorical("use_possible_rules", [True, False]),
            )

        elif self.algorithm == "FRNNClassifier" and FRLEARN_AVAILABLE:
            # frlearn default: upper_k = lower_k = 20, linear OWA weights, boscovich
            model = FRNNClassifier(
                upper_k=trial.suggest_int("upper_k", 1, 50),
                lower_k=trial.suggest_int("lower_k", 1, 50),
                weights=trial.suggest_categorical("weights", list(_FRLEARN_OWA_WEIGHTS)),
                dissimilarity=trial.suggest_categorical("dissimilarity", list(_FRLEARN_DISSIMILARITIES)),
            )

        elif self.algorithm == "FROVOCOClassifier" and FRLEARN_AVAILABLE:
            # frlearn default: balanced_k=16, ir_threshold=9, boscovich
            model = FROVOCOClassifier(
                balanced_k=trial.suggest_int("balanced_k", 1, 50),
                ir_threshold=trial.suggest_float("ir_threshold", 1.0, 30.0, log=True),
                dissimilarity=trial.suggest_categorical("dissimilarity", list(_FRLEARN_DISSIMILARITIES)),
            )

        else:
            raise ValueError(f"Algorithm {self.algorithm} is not implemented")

        # Perform cross-validation
        scores = self._cross_val_with_timeout(model, X, y)
        trial.set_user_attr("cv_std", float(scores.std()))
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

        Notes
        -----
        ``best_score_`` is the best inner-CV score found across all trials and is an
        optimistically biased estimate of generalization (the hyperparameters are
        selected on the same folds they are scored on). For an unbiased estimate, use
        ``nested_score(X, y)``.
        """
        # Store feature names before check_X_y converts DataFrame to numpy
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        # Validate input (NaN allowed only for NaN-native algorithms)
        X, y = self._check_Xy(X, y)

        # Store feature information
        self.n_features_in_ = X.shape[1]
        self._init_cv_seed()

        # Create and run the study
        start_time = time.time()

        self.study_ = optuna.create_study(direction=self.direction, sampler=self._make_sampler())

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
        # Count only COMPLETE trials. Pruned/failed trials must not count here: the
        # attribute is named *completed*, and `study_.best_trial` raises ValueError
        # (rather than returning None) when no trial completed, so we must gate on this
        # before touching best_trial / best_params below.
        self.n_trials_completed_ = len(
            self.study_.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        )

        # Check if any trials succeeded
        if self.n_trials_completed_ == 0:
            raise RuntimeError(
                "Optimization failed: No successful trials completed. "
                "Check pruned trial messages via study_.trials for the root cause "
                "(common causes: wrong scoring for task type, cv_timeout too low, bad data)."
            )

        # Store best results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        self.best_score_std_ = self.study_.best_trial.user_attrs.get("cv_std", float("nan"))

        # Create and fit the best estimator (wrapped in the same scaler used during CV,
        # so predictions are made on identically-scaled features).
        self._create_best_estimator()
        self.best_estimator_ = self._wrap_model(self.best_estimator_, X)
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
            params = self._resolve_svm_gamma(params)
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
            if params.pop("l2_zero", False):
                params.pop("l2_regularization", None)
                params["l2_regularization"] = 0.0
            self.best_estimator_ = HistGradientBoostingClassifier(**params, random_state=self.random_state)
        elif self.algorithm == "MLPClassifier":
            params["hidden_layer_sizes"] = _decode_hidden_layers(params["hidden_layer_sizes"])
            self.best_estimator_ = MLPClassifier(**params, random_state=self.random_state, max_iter=1000, early_stopping=True)
        elif self.algorithm == "GaussianNB":
            self.best_estimator_ = GaussianNB(**params)
        elif self.algorithm == "QDA":
            self.best_estimator_ = QDA(**params)
        elif self.algorithm == "LDA":
            mode = params.pop("shrinkage_mode", None)
            if mode == "none":
                params.pop("shrinkage", None)
            elif mode == "auto":
                params["shrinkage"] = "auto"
            self.best_estimator_ = LDA(**params)
        elif self.algorithm == "GaussianProcessClassifier":
            kind = params.pop("kernel")
            nu = params.pop("nu", None)
            self.best_estimator_ = GaussianProcessClassifier(
                kernel=_make_gp_kernel(kind, nu), **params, random_state=self.random_state
            )
        elif self.algorithm == "LinearSVC":
            params.setdefault("penalty", "l2")
            self.best_estimator_ = LinearSVC(
                **params, dual=(params["penalty"] != "l1"), max_iter=5000, random_state=self.random_state
            )
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
            self.best_estimator_ = _LabelEncodingClassifier(
                XGBClassifier(**params, random_state=self.random_state, eval_metric="logloss", verbosity=0)
            )
        elif self.algorithm == "LGBMClassifier" and LIGHTGBM_AVAILABLE:
            if params.pop("max_depth_none", False):
                params.pop("max_depth", None)
                params["max_depth"] = -1
            self.best_estimator_ = LGBMClassifier(**params, subsample_freq=1, random_state=self.random_state, verbosity=-1)
        elif self.algorithm == "SGDClassifier":
            self.best_estimator_ = SGDClassifier(**params, max_iter=1000, tol=1e-3, random_state=self.random_state)
        elif self.algorithm == "ExplainableBoostingClassifier" and INTERPRET_AVAILABLE:
            self.best_estimator_ = _make_ebm("classifier", self.random_state, **params)
        elif self.algorithm == "RuleFitClassifier" and IMODELS_AVAILABLE:
            self.best_estimator_ = _make_imodels("RuleFitClassifier", self.random_state, **params)
        elif self.algorithm == "FIGSClassifier" and IMODELS_AVAILABLE:
            self.best_estimator_ = _make_imodels("FIGSClassifier", self.random_state, **_figs_params(params))
        elif self.algorithm == "RoughSetClassifier":
            self.best_estimator_ = RoughSetClassifier(**params)
        elif self.algorithm == "FRNNClassifier" and FRLEARN_AVAILABLE:
            self.best_estimator_ = FRNNClassifier(**params)
        elif self.algorithm == "FROVOCOClassifier" and FRLEARN_AVAILABLE:
            self.best_estimator_ = FROVOCOClassifier(**params)
        else:
            raise ValueError(f"Cannot create estimator: algorithm '{self.algorithm}' is not available.")

    def predict(self, X):
        """Make predictions using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = self._check_X(X)
        if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict(X)
        return self.best_estimator_.predict(X)

    @available_if(_fitted_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = self._check_X(X)

        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(f"{self.algorithm} does not support probability predictions")

        if LIGHTGBM_AVAILABLE and self.algorithm == "LGBMClassifier":
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict_proba(X)
        return self.best_estimator_.predict_proba(X)

    @available_if(_fitted_estimator_has("decision_function"))
    def decision_function(self, X):
        """Get decision function values"""
        check_is_fitted(self, ["best_estimator_", "classes_", "n_features_in_"])
        X = self._check_X(X)

        if not hasattr(self.best_estimator_, "decision_function"):
            raise AttributeError(f"{self.algorithm} does not have decision_function")

        return self.best_estimator_.decision_function(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Note: like any sklearn classifier, ``score`` always returns accuracy,
        regardless of the ``scoring`` metric used to *optimize* hyperparameters.
        To evaluate with a different metric, use ``sklearn.metrics`` on
        ``predict``/``predict_proba`` output.
        """
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X, y = self._check_Xy(X, y)
        self._check_n_features(X)
        if LIGHTGBM_AVAILABLE and self.algorithm == "LGBMClassifier":
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.score(X, y)
        return self.best_estimator_.score(X, y)


class RegressorOptimizer(RegressorMixin, OptimizerBase):
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
        "GaussianProcessRegressor",
        "KernelRidge",
        "PLSRegression",
        "BayesianRidge",
        "HuberRegressor",
        "LinearSVR",
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
        if FRLEARN_AVAILABLE:
            algos.append("FRNNRegressor")
        if INTERPRET_AVAILABLE:
            algos.append("ExplainableBoostingRegressor")
        if IMODELS_AVAILABLE:
            algos.extend(["RuleFitRegressor", "FIGSRegressor"])
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
                # Keep the sklearn defaults ("scale"/"auto") reachable, plus a tunable float.
                gamma_mode = trial.suggest_categorical("gamma", ["scale", "auto", "value"])
                gamma = trial.suggest_float("gamma_value", 1e-4, 1e1, log=True) if gamma_mode == "value" else gamma_mode
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
            l2_zero = trial.suggest_categorical("l2_zero", [True, False])
            l2_regularization = 0.0 if l2_zero else trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)
            model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=self.random_state,
            )

        elif self.algorithm == "MLPRegressor":
            hidden_layer_sizes = _decode_hidden_layers(trial.suggest_categorical("hidden_layer_sizes", ["50", "100", "50-50", "100-50", "100-100"]))
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

        elif self.algorithm == "GaussianProcessRegressor":
            kind = trial.suggest_categorical("kernel", ["rbf", "matern", "rational_quadratic"])
            nu = trial.suggest_categorical("nu", [0.5, 1.5, 2.5]) if kind == "matern" else None
            alpha = trial.suggest_float("alpha", 1e-10, 1e0, log=True)
            normalize_y = trial.suggest_categorical("normalize_y", [False, True])
            n_restarts = trial.suggest_int("n_restarts_optimizer", 0, 3)
            model = GaussianProcessRegressor(
                kernel=_make_gp_kernel(kind, nu), alpha=alpha, normalize_y=normalize_y,
                n_restarts_optimizer=n_restarts, random_state=self.random_state,
            )

        elif self.algorithm == "KernelRidge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "laplacian", "sigmoid"])
            kr_params = {"alpha": alpha, "kernel": kernel}
            if kernel != "linear":
                gamma_mode = trial.suggest_categorical("gamma", ["default", "value"])
                if gamma_mode == "value":
                    kr_params["gamma"] = trial.suggest_float("gamma_value", 1e-4, 1e1, log=True)
                if kernel == "poly":
                    kr_params["degree"] = trial.suggest_int("degree", 2, 5)
            model = KernelRidge(**kr_params)

        elif self.algorithm == "PLSRegression":
            n_components = trial.suggest_int("n_components", 1, max(1, min(20, X.shape[1])))
            model = PLSRegression(n_components=n_components, max_iter=1000)

        elif self.algorithm == "BayesianRidge":
            model = BayesianRidge(
                alpha_1=trial.suggest_float("alpha_1", 1e-8, 1e-4, log=True),
                alpha_2=trial.suggest_float("alpha_2", 1e-8, 1e-4, log=True),
                lambda_1=trial.suggest_float("lambda_1", 1e-8, 1e-4, log=True),
                lambda_2=trial.suggest_float("lambda_2", 1e-8, 1e-4, log=True),
            )

        elif self.algorithm == "HuberRegressor":
            epsilon = trial.suggest_float("epsilon", 1.0, 3.0)
            alpha = trial.suggest_float("alpha", 1e-6, 1e0, log=True)
            model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)

        elif self.algorithm == "LinearSVR":
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            loss = trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"])
            epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
            model = LinearSVR(C=C, loss=loss, epsilon=epsilon, dual=True, max_iter=5000, random_state=self.random_state)

        elif self.algorithm == "LGBMRegressor" and LIGHTGBM_AVAILABLE:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth_none = trial.suggest_categorical("max_depth_none", [True, False])
            max_depth = -1 if max_depth_none else trial.suggest_int("max_depth", 2, 20)
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
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample_freq=1,  # subsample is ignored by LightGBM unless freq > 0
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
            max_depth = trial.suggest_int("max_depth", 2, 12)
            min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 1e2, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            gamma = trial.suggest_float("gamma", 0, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
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

        elif self.algorithm == "ExplainableBoostingRegressor" and INTERPRET_AVAILABLE:
            model = _make_ebm("regressor", self.random_state, **_suggest_ebm(trial))

        elif self.algorithm == "RuleFitRegressor" and IMODELS_AVAILABLE:
            model = _make_imodels("RuleFitRegressor", self.random_state, **_suggest_rulefit(trial))

        elif self.algorithm == "FIGSRegressor" and IMODELS_AVAILABLE:
            model = _make_imodels("FIGSRegressor", self.random_state, **_suggest_figs(trial))

        elif self.algorithm == "FRNNRegressor" and FRLEARN_AVAILABLE:
            # frlearn default: k=10, chebyshev
            model = FRNNRegressor(
                k=trial.suggest_int("k", 1, 30),
                dissimilarity=trial.suggest_categorical("dissimilarity", list(_FRLEARN_DISSIMILARITIES)),
            )

        else:
            raise ValueError(f"Algorithm {self.algorithm} is not implemented")

        # Perform cross-validation
        scores = self._cross_val_with_timeout(model, X, y)
        trial.set_user_attr("cv_std", float(scores.std()))
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

        Notes
        -----
        ``best_score_`` is the best inner-CV score found across all trials and is an
        optimistically biased estimate of generalization (the hyperparameters are
        selected on the same folds they are scored on). For an unbiased estimate, use
        ``nested_score(X, y)``.
        """
        # Store feature names before check_X_y converts DataFrame to numpy
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        # Validate input (NaN allowed only for NaN-native algorithms)
        X, y = self._check_Xy(X, y, y_numeric=True)

        # Store feature information
        self.n_features_in_ = X.shape[1]
        self._init_cv_seed()

        # Create and run the study
        start_time = time.time()

        self.study_ = optuna.create_study(direction=self.direction, sampler=self._make_sampler())

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
        # Count only COMPLETE trials. Pruned/failed trials must not count here: the
        # attribute is named *completed*, and `study_.best_trial` raises ValueError
        # (rather than returning None) when no trial completed, so we must gate on this
        # before touching best_trial / best_params below.
        self.n_trials_completed_ = len(
            self.study_.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        )

        # Check if any trials succeeded
        if self.n_trials_completed_ == 0:
            raise RuntimeError(
                "Optimization failed: No successful trials completed. "
                "Check pruned trial messages via study_.trials for the root cause "
                "(common causes: wrong scoring for task type, cv_timeout too low, bad data)."
            )

        # Store best results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        self.best_score_std_ = self.study_.best_trial.user_attrs.get("cv_std", float("nan"))

        # Create and fit the best estimator (wrapped in the same scaler used during CV,
        # so predictions are made on identically-scaled features).
        self._create_best_estimator()
        self.best_estimator_ = self._wrap_model(self.best_estimator_, X)
        self._fit_best_estimator(X, y)

        # Set output dimensions
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        return self

    def _create_best_estimator(self):
        """Create the best estimator with optimal parameters"""
        params = self.best_params_.copy()

        if self.algorithm == "SVR":
            params = self._resolve_svm_gamma(params)
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
            if params.pop("l2_zero", False):
                params.pop("l2_regularization", None)
                params["l2_regularization"] = 0.0
            self.best_estimator_ = HistGradientBoostingRegressor(**params, random_state=self.random_state)
        elif self.algorithm == "MLPRegressor":
            params["hidden_layer_sizes"] = _decode_hidden_layers(params["hidden_layer_sizes"])
            self.best_estimator_ = MLPRegressor(**params, random_state=self.random_state, max_iter=1000, early_stopping=True)
        elif self.algorithm == "GaussianProcessRegressor":
            kind = params.pop("kernel")
            nu = params.pop("nu", None)
            self.best_estimator_ = GaussianProcessRegressor(
                kernel=_make_gp_kernel(kind, nu), **params, random_state=self.random_state
            )
        elif self.algorithm == "KernelRidge":
            params = self._resolve_kernel_ridge_gamma(params)
            self.best_estimator_ = KernelRidge(**params)
        elif self.algorithm == "PLSRegression":
            self.best_estimator_ = PLSRegression(**params, max_iter=1000)
        elif self.algorithm == "BayesianRidge":
            self.best_estimator_ = BayesianRidge(**params)
        elif self.algorithm == "HuberRegressor":
            self.best_estimator_ = HuberRegressor(**params, max_iter=1000)
        elif self.algorithm == "LinearSVR":
            self.best_estimator_ = LinearSVR(**params, dual=True, max_iter=5000, random_state=self.random_state)
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
            if params.pop("max_depth_none", False):
                params.pop("max_depth", None)
                params["max_depth"] = -1
            self.best_estimator_ = LGBMRegressor(**params, subsample_freq=1, random_state=self.random_state, verbosity=-1)
        elif self.algorithm == "SGDRegressor":
            self.best_estimator_ = SGDRegressor(**params, max_iter=1000, tol=1e-3, random_state=self.random_state)
        elif self.algorithm == "ExplainableBoostingRegressor" and INTERPRET_AVAILABLE:
            self.best_estimator_ = _make_ebm("regressor", self.random_state, **params)
        elif self.algorithm == "RuleFitRegressor" and IMODELS_AVAILABLE:
            self.best_estimator_ = _make_imodels("RuleFitRegressor", self.random_state, **params)
        elif self.algorithm == "FIGSRegressor" and IMODELS_AVAILABLE:
            self.best_estimator_ = _make_imodels("FIGSRegressor", self.random_state, **_figs_params(params))
        elif self.algorithm == "FRNNRegressor" and FRLEARN_AVAILABLE:
            self.best_estimator_ = FRNNRegressor(**params)
        else:
            raise ValueError(f"Cannot create estimator: algorithm '{self.algorithm}' is not available.")

    def predict(self, X):
        """Make predictions using the best estimator"""
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X = self._check_X(X)
        if LIGHTGBM_AVAILABLE and self.algorithm in ("LGBMClassifier", "LGBMRegressor"):
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                return self.best_estimator_.predict(X)
        pred = self.best_estimator_.predict(X)
        # PLSRegression (sklearn < 1.3) returns (n, 1) for 1-D y.
        if pred.ndim == 2 and pred.shape[1] == 1 and self.n_outputs_ == 1:
            pred = pred.ravel()
        return pred

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Note: like any sklearn regressor, ``score`` always returns R^2,
        regardless of the ``scoring`` metric used to *optimize* hyperparameters.
        To evaluate with a different metric, use ``sklearn.metrics`` on
        ``predict`` output.
        """
        check_is_fitted(self, ["best_estimator_", "n_features_in_"])
        X, y = self._check_Xy(X, y, y_numeric=True)
        self._check_n_features(X)
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
        scale="auto",
    ):
        """
        Initialize the universal optimizer.

        Parameters
        ----------
        algorithm : str, default='SVC'
            Machine learning algorithm to optimize
        direction : str, default='maximize'
            Optimization direction ('maximize' or 'minimize'). Keep 'maximize'
            for all of sklearn's built-in scoring strings — they follow the
            higher-is-better convention, including the ``neg_*`` error metrics
            (e.g. 'neg_mean_squared_error'). Only use 'minimize' with a custom
            lower-is-better objective.
        verbose : bool or int, default=False
            Verbosity level
        show_progress_bar : bool, default=False
            Whether to show progress bar
        n_trials : int, default=100
            Number of optimization trials
        timeout : float or None, default=None
            Maximum time for optimization
        cv : int or cross-validation splitter, default=5
            Number of CV folds. An int builds a shuffled, seeded splitter; a
            splitter object is used as-is.
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
        scale : bool or {'auto'}, default='auto'
            Feature scaling policy. ``'auto'`` wraps only scale-sensitive
            algorithms (SVM, KNN, MLP, regularized-linear models) in a
            StandardScaler fit per CV fold; ``True`` always scales, ``False``
            never scales.
        """
        supported = self._get_supported_algorithms()
        if algorithm not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

        self.algorithm = algorithm

        # Auto-select scoring for the inner optimizer. ``self.scoring`` keeps the raw
        # user value (possibly None) so get_params/clone/set_params round-trip and
        # an auto-selected metric is re-resolved when ``algorithm`` changes.
        raw_scoring = scoring
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
                scale=scale,
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
                scale=scale,
            )
            self._estimator_type = "regressor"

        # Store all parameters for get_params
        self.direction = direction
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = raw_scoring
        self.cv_timeout = cv_timeout
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.scale = scale

    def __sklearn_tags__(self):
        """Expose the estimator type of the wrapped optimizer (sklearn >= 1.6).

        sklearn >= 1.6 reads the type from tags, not from ``_estimator_type``, so
        without this ``is_classifier(Optimizer("SVC"))`` is False and scorers such
        as ``roc_auc`` / stratified CV are not applied correctly.
        """
        tags = super().__sklearn_tags__()
        inner = self._optimizer.__sklearn_tags__()
        tags.estimator_type = inner.estimator_type
        tags.classifier_tags = inner.classifier_tags
        tags.regressor_tags = inner.regressor_tags
        return tags

    def fit(self, X, y):
        """Fit the optimizer"""
        self._optimizer.fit(X, y)

        # Mirror all fitted attributes
        self.best_params_ = self._optimizer.best_params_
        self.best_estimator_ = self._optimizer.best_estimator_
        self.best_score_ = self._optimizer.best_score_
        self.best_score_std_ = self._optimizer.best_score_std_
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

    @available_if(_facade_has("predict_proba"))
    def predict_proba(self, X):
        """Get probability estimates (classifiers only)"""
        return self._optimizer.predict_proba(X)

    @available_if(_facade_has("decision_function"))
    def decision_function(self, X):
        """Get decision function values (some classifiers only)"""
        return self._optimizer.decision_function(X)

    def score(self, X, y):
        """Return the score of the model on the test data"""
        return self._optimizer.score(X, y)

    def nested_score(self, X, y, outer_cv=5):
        """Unbiased generalization estimate via nested cross-validation.

        ``best_score_`` is optimistically biased because the hyperparameters are
        chosen on the same folds they are scored on. ``nested_score`` refits the
        whole optimizer independently on each outer training fold and scores on the
        held-out fold, so the search never sees the data it is judged on. Runs
        ``outer_cv`` full optimizations (expensive) and returns the per-fold scores.
        """
        stratify = self._estimator_type == "classifier"
        return _nested_cv_score(self, X, y, self.scoring, outer_cv, self.random_state, stratify)

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
            "scale": self.scale,
        }

    def set_params(self, **params):
        """Set parameters for this estimator"""
        if "algorithm" in params and params["algorithm"] != self.algorithm:
            # Changing algorithm requires recreating the internal optimizer.
            # Merge current stored params with the caller's overrides so that
            # previously set values (n_trials, cv, etc.) are preserved.
            # Reset scoring to None so auto-selection fires for the new algorithm
            # unless the caller explicitly provided a scoring value.
            current = self.get_params()  # scoring is the raw user value (None = auto)
            current.update(params)
            self.__init__(**current)
        else:
            # Update existing parameters on both this object and the internal optimizer.
            valid = self.get_params()
            for key, value in params.items():
                if key not in valid:
                    raise ValueError(f"Invalid parameter '{key}' for estimator {type(self).__name__}.")
                setattr(self, key, value)
                if key == "scoring" and value is None:
                    value = "accuracy" if self._estimator_type == "classifier" else "r2"
                setattr(self._optimizer, key, value)

            # Re-validate parameters in the internal optimizer.
            self._optimizer._validate_params()

        return self


class AlgorithmBenchmark:
    """
    Benchmark all (or a subset of) supported algorithms on a dataset.

    Runs an independent Optimizer for each algorithm, collects results, and
    exposes the best-performing estimator.  Not a scikit-learn estimator —
    intentionally outside the BaseEstimator contract so there are no
    clone/Pipeline compatibility constraints.

    Parameters
    ----------
    task : str
        Either ``"classification"`` or ``"regression"``.
    algorithms : list of str or ``"all"``, default ``"all"``
        Algorithms to benchmark.  Pass a list to restrict the comparison.
    direction : str, default ``"maximize"``
        Optuna optimization direction.
    n_trials : int, default 50
        Number of Optuna trials per algorithm.
    timeout : float or None, default None
        Per-algorithm wall-clock timeout for the Optuna study (seconds).
    cv : int, default 5
        Cross-validation folds.
    scoring : str or None, default None
        Scoring metric.  Defaults to ``"accuracy"`` for classification and
        ``"r2"`` for regression when ``None``.
    cv_timeout : float, default 120
        Timeout for a single CV evaluation (seconds).
    random_state : int or None, default None
        Random state passed to every Optimizer.
    early_stopping_patience : int or None, default None
        Early stopping patience passed to every Optimizer.
    n_jobs : int, default 1
        Parallel jobs for CV inside each Optimizer.
    n_jobs_algorithms : int, default 1
        Number of algorithms to run in parallel.  Uses ``joblib.Parallel``.
        Set to ``-1`` to use all available cores.
    verbose : bool or int, default False
        Verbosity level forwarded to each Optimizer.
    include_dummy : bool, default True
        Whether to include a dummy baseline (``DummyClassifier`` with
        ``strategy="most_frequent"`` for classification, ``DummyRegressor``
        with ``strategy="mean"`` for regression).  The dummy is evaluated with
        the same CV setup but bypasses Optuna entirely.  It is always excluded
        from ``best_algorithm_`` / ``best_estimator_`` selection so it cannot
        "win", but appears in ``results_`` and ``summary()`` for reference.
    scale : bool or {'auto'}, default 'auto'
        Feature scaling policy forwarded to every Optimizer.  ``'auto'`` scales
        only scale-sensitive algorithms, so tree ensembles and scale-invariant
        models are compared on equal footing with SVM/KNN/MLP/linear models.
    """

    def __init__(
        self,
        task: str,
        algorithms="all",
        direction: str = "maximize",
        n_trials: int = 50,
        timeout: Optional[float] = None,
        cv: int = 5,
        scoring: Optional[str] = None,
        cv_timeout: float = 120,
        random_state: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        n_jobs: int = 1,
        n_jobs_algorithms: int = 1,
        verbose: Union[bool, int] = False,
        include_dummy: bool = True,
        scale: Union[bool, str] = "auto",
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        if algorithms == "all":
            if task == "classification":
                algorithms = list(ClassifierOptimizer._get_supported_algorithms())
            else:
                algorithms = list(RegressorOptimizer._get_supported_algorithms())
        else:
            supported = (
                ClassifierOptimizer._get_supported_algorithms()
                if task == "classification"
                else RegressorOptimizer._get_supported_algorithms()
            )
            unknown = [a for a in algorithms if a not in supported]
            if unknown:
                raise ValueError(
                    f"Unknown algorithms for task='{task}': {unknown}. "
                    f"Supported: {list(supported)}"
                )

        self.task = task
        self.algorithms = algorithms
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.cv_timeout = cv_timeout
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.n_jobs_algorithms = n_jobs_algorithms
        self.verbose = verbose
        self.include_dummy = include_dummy
        self.scale = scale

    def _run_dummy(self, X, y):
        """Evaluate the dummy baseline via cross-validation (no Optuna)."""
        import time as _time
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        if self.task == "classification":
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy="most_frequent",
                                    random_state=self.random_state)
            label = "DummyClassifier(most_frequent)"
        else:
            from sklearn.dummy import DummyRegressor
            dummy = DummyRegressor(strategy="mean")
            label = "DummyRegressor(mean)"

        # Use the same shuffled, seeded splitter the Optimizers use, so the dummy is
        # scored on identical folds. A splitter object passed as cv is used as-is.
        if isinstance(self.cv, int):
            splitter_cls = StratifiedKFold if self.task == "classification" else KFold
            cv = splitter_cls(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv

        scoring = self.scoring or ("accuracy" if self.task == "classification" else "r2")
        t0 = _time.monotonic()
        try:
            scores = cross_val_score(dummy, X, y, cv=cv, scoring=scoring,
                                     n_jobs=self.n_jobs)
            dummy.fit(X, y)
            elapsed = _time.monotonic() - t0
            return {
                "algorithm": label,
                "best_score": float(scores.mean()),
                "best_params": {},
                "n_trials_completed": 1,
                "fit_time": elapsed,
                "error": None,
                "optimizer": None,
                "_dummy_estimator": dummy,
                "_is_dummy": True,
            }
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            return {
                "algorithm": label,
                "best_score": float("nan"),
                "best_params": {},
                "n_trials_completed": 0,
                "fit_time": elapsed,
                "error": str(exc),
                "optimizer": None,
                "_dummy_estimator": None,
                "_is_dummy": True,
            }

    def _run_one(self, algorithm, X, y):
        """Fit a single Optimizer and return a result dict."""
        import time as _time
        opt = Optimizer(
            algorithm=algorithm,
            direction=self.direction,
            verbose=self.verbose,
            n_trials=self.n_trials,
            timeout=self.timeout,
            cv=self.cv,
            scoring=self.scoring,
            cv_timeout=self.cv_timeout,
            random_state=self.random_state,
            early_stopping_patience=self.early_stopping_patience,
            n_jobs=self.n_jobs,
            scale=self.scale,
        )
        t0 = _time.monotonic()
        try:
            opt.fit(X, y)
            elapsed = _time.monotonic() - t0
            return {
                "algorithm": algorithm,
                "best_score": opt.best_score_,
                "best_params": opt.best_params_,
                "n_trials_completed": opt.n_trials_completed_,
                "fit_time": elapsed,
                "error": None,
                "optimizer": opt,
            }
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            return {
                "algorithm": algorithm,
                "best_score": float("nan"),
                "best_params": {},
                "n_trials_completed": 0,
                "fit_time": elapsed,
                "error": str(exc),
                "optimizer": None,
            }

    def fit(self, X, y):
        """
        Run the benchmark on data ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=self.n_jobs_algorithms)(
            delayed(self._run_one)(alg, X, y) for alg in self.algorithms
        )

        if self.include_dummy:
            results.append(self._run_dummy(X, y))

        self.results_ = results

        # Pick the best among real (non-dummy) succeeded runs
        succeeded = [r for r in results
                     if r["error"] is None and not r.get("_is_dummy", False)]
        if not succeeded:
            raise RuntimeError("All algorithms failed. Check errors in results_.")

        if self.direction == "maximize":
            best = max(succeeded, key=lambda r: r["best_score"])
        else:
            best = min(succeeded, key=lambda r: r["best_score"])

        self.best_algorithm_ = best["algorithm"]
        self.best_score_ = best["best_score"]
        self.best_estimator_ = best["optimizer"].best_estimator_
        self.best_params_ = best["optimizer"].best_params_
        self.optimizers_ = {
            r["algorithm"]: r["optimizer"]
            for r in results
            if not r.get("_is_dummy", False)
        }

        dummy_result = next((r for r in results if r.get("_is_dummy")), None)
        if dummy_result:
            self.dummy_score_ = dummy_result["best_score"]
            self.dummy_estimator_ = dummy_result.get("_dummy_estimator")

        return self

    def summary(self):
        """
        Return benchmark results sorted by score.

        Returns a ``pandas.DataFrame`` if pandas is available, otherwise a
        list of dicts (keys: algorithm, best_score, n_trials_completed,
        fit_time, error).
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("Call fit() before summary().")

        rows = [
            {
                "algorithm": r["algorithm"],
                "best_score": r["best_score"],
                "n_trials_completed": r["n_trials_completed"],
                "fit_time": r["fit_time"],
                "error": r["error"],
                "is_dummy": r.get("_is_dummy", False),
            }
            for r in self.results_
        ]

        ascending = self.direction == "minimize"
        rows.sort(key=lambda r: (r["best_score"] != r["best_score"], r["best_score"]),
                  reverse=not ascending)

        try:
            import pandas as pd
            return pd.DataFrame(rows).reset_index(drop=True)
        except ImportError:
            return rows
