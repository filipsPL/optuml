# OptuML code review — investigation report

**Reviewer:** Claude (Opus)
**Date:** 2026-07-03
**Scope:** `optuml/optuml.py`, `tests/test_optimizer.py`, `README.md`, packaging. Focus on
inconsistencies, bugs, scientific/methodological flaws, and improvements.

Findings are ordered most-severe first. Items **#1**, **#2**, and **#6** were fixed in the same
change that produced this report (see *Status* lines).

---

## Confirmed bugs

### 1. `cv_timeout` does not actually bound per-trial wall-clock time
**Location:** `optuml.py` — `OptimizerBase._cross_val_with_timeout` (~L178–216)
**Status: FIXED.**

The timeout used `with ThreadPoolExecutor(max_workers=1) as executor:`. When
`future.result(timeout=...)` raised `TimeoutError`, the code raised `optuna.TrialPruned` — but
exiting the `with` block calls `executor.shutdown(wait=True)`, which **blocks until the slow CV
finishes anyway**.

Reproduction (1 s timeout on a 4 s task):

```
[1.00s] TimeoutError -> raising TrialPruned
[4.00s] control returned to caller     # timeout did NOT free the caller early
```

So a pathologically slow trial still ran to completion; `cv_timeout` only affected whether the
trial was labeled *pruned* vs *completed*, not how long it took. This contradicts README
("Per-trial timeout… Will stop at timeout even if trials remain").

**Fix applied:** create the executor without a `with` block and call `shutdown(wait=False)` in a
`finally`, so control returns to the caller as soon as the timeout fires. Timed-out worker threads
keep running in the background (Python threads cannot be killed — the documented trade-off in
CLAUDE.md), but they no longer hold up optimization. A fuller solution (killable
`ProcessPoolExecutor`) is noted as future work.

### 2. When all trials fail, `fit()` raises a confusing `ValueError` instead of the intended `RuntimeError`
**Location:** `optuml.py` — `ClassifierOptimizer.fit` (~L638–646) and `RegressorOptimizer.fit`
(~L1110–1118)
**Status: FIXED.**

The guard was `if self.n_trials_completed_ == 0 or self.study_.best_trial is None:`. Two problems
made the curated error message dead code:

- `n_trials_completed_ = len(self.study_.trials)` counted **all** trials (pruned + failed +
  complete), so it was rarely 0. The attribute name was a misnomer.
- `study_.best_trial` **raises** `ValueError("No trials are completed yet.")` when no trial
  completed; it never returns `None`. The `or` short-circuited into that exception.

Reproduction (forcing every trial to prune) yielded `ValueError: No trials are completed yet.`
instead of the intended `RuntimeError` with the "check pruned trial messages…" guidance.

**Fix applied:** count only `COMPLETE` trials
(`study_.get_trials(states=(optuna.trial.TrialState.COMPLETE,))`), store that as
`n_trials_completed_` (now matching its name), and raise the `RuntimeError` when it is 0 — without
ever touching `best_trial` in the failure path.

---

## Scientific / methodological flaws

### 3. No internal feature scaling — undermines the stated "fair comparison" goal
**Status: open (documentation / design).**

CV runs on raw `X`. Scale-sensitive methods (SVC/SVR, KNN, MLP, LogisticRegression, SGD,
Ridge/Lasso/ElasticNet) are systematically handicapped versus scale-invariant tree ensembles, so
`AlgorithmBenchmark` produces an *unfair* comparison by construction — directly at odds with the
README's motivation ("each method the best version of itself"). Users can wrap a `StandardScaler`
in a Pipeline, but nothing warns them it is effectively required, and the benchmark cannot do it
per-algorithm. Recommendation: an optional internal scaler (opt-in per algorithm) or a prominent
documentation warning.

### 4. `best_score_` is an optimistically biased estimate, and the bias differs per algorithm
**Status: open (documentation / design).**

`best_score_` is the maximum CV score over up to `n_trials` trials on the *same* folds — the
classic "hyperparameter optimization overfits the validation set" effect. Ranking algorithms by it
in `AlgorithmBenchmark` favors algorithms with larger/looser search spaces (they overfit the CV
more). There is no nested CV. For a tool whose purpose is fair model comparison this is the most
important scientific caveat and is currently undocumented. Recommendation: nested CV, or at minimum
document that `best_score_` is not an unbiased generalization estimate.

### 5. CV folds are neither shuffled nor tied to `random_state`
**Status: open.**

`cross_val_score(model, X, y, cv=self.cv, …)` uses the default splitter with `shuffle=False`.
Classification gets StratifiedKFold, but for both tasks the folds depend on row order and ignore
`random_state`. Data sorted by target/group yields biased folds, and results are not robust to row
permutation despite `random_state` being advertised for reproducibility.

---

## Other issues & improvements

### 6. SVC/SVR default `gamma` was unreachable
**Location:** `optuml.py` SVC objective (~L336) and SVR objective (~L819)
**Status: FIXED.**

`gamma` was only ever suggested as a float in `[1e-4, 1e1]`; the sklearn default `"scale"` (and
`"auto"`) could never be selected. This violated CLAUDE.md's own principle that every search space
must contain the sklearn default.

**Fix applied:** `gamma` is now a categorical (`"scale"`, `"auto"`, `"value"`); when `"value"` is
chosen, a `gamma_value` float in `[1e-4, 1e1]` (log) is suggested. `_create_best_estimator` resolves
the two-step encoding back into a single `gamma` argument (mirroring the `max_depth_none` sentinel
pattern). The categorical key remains `"gamma"`, so `test_optimizer_best_params_attribute` (which
asserts `"gamma" in best_params_` for non-linear kernels) still holds.

### 7. `XGBClassifier` does not label-encode `y`
XGBoost requires labels `0..n-1`; string or non-contiguous integer labels will error. The other
classifiers tolerate arbitrary labels — an inconsistency. Recommendation: `LabelEncoder` inside the
XGB path.

### 8. `_validate_params()` runs inside `__init__`
sklearn convention is that `__init__` only stores params; validation belongs in `fit`.
`check_estimator` would flag this. Low priority since `clone` still round-trips.

### 9. `score()` ignores `scoring`
Classifier `score` always returns accuracy, regressor always R², even when a different metric was
optimized. Standard sklearn behavior, but worth a doc note since it surprises users comparing
`best_score_` to `score()`.

### 10. `direction="minimize"` is a footgun
All sklearn scoring strings are higher-is-better (including `neg_*`), so `maximize` is essentially
always correct; exposing `minimize` invites misuse.

### 11. Repository hygiene
Committed `catboost_info/`, `testy-skasuj/` (scratch dir), `NOTATKI.md`, and a bloated `dist/` with
every historical wheel. `dist/optuml-0.2.7.tar.gz` was already built while the `setup.py` 0.2.7 bump
was still uncommitted — a version inconsistency. Recommendation: `.gitignore` `dist/`,
`catboost_info/`, and scratch dirs.

### 12. Packaging split
`pyproject.toml` is minimal while `setup.py` holds all metadata; `install_requires` pins nothing
despite version-specific shims for sklearn ≥1.6/≥1.8. Recommendation: consolidate into `pyproject`
and add a lower bound (e.g. `scikit-learn>=1.0`).

---

## Suggested follow-up priority
1. **Done:** #1, #2, #6.
2. Address #3/#4 at least via documentation (ideally an optional internal scaler and a nested-CV
   mode) — they go to the core scientific validity of the "fair comparison" pitch.
3. #5 (shuffled, seeded CV splitter) is a small, high-value reproducibility fix.
