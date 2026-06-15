# Mean imputation, distilled

Unconditional mean imputation replaces every missing entry of a column with the mean of that
column's observed entries. It is the simplest possible completion of a table with `NaN`s:
per-column, univariate, deterministic, single imputation, and it deliberately ignores all
inter-feature structure. Fit on the training rows, the same per-column means are replayed on
any table — including unseen test rows — so the completion is one fixed map across train and
test.

## Problem it solves

A table `X` of shape `(n_samples, n_features)` has missing entries (`NaN`), which most
downstream estimators cannot consume (classical regression drops incomplete rows;
gradient-based learners cannot represent `NaN`; distances are undefined with a missing
coordinate). Discarding incomplete rows is not viable — with `d` columns each missing at rate
`p` under MCAR, the fraction of complete rows is `(1 − p)^d` (e.g. `≈ 0.0012` at `d = 30`,
`p = 0.2`) — and it leaves test-time missingness unsolved. The task is a learnable,
out-of-sample-applicable map from a table-with-`NaN`s to a same-shape finite-valued table.

## Key idea

For each column `j`, the best constant guess of a missing entry, using no other variable, is
the column mean. Two derivations give the same answer:

- **Least-squares constant.** `argmin_c Σ_{i observed}(x_ij − c)² = (1/n_obs) Σ_{i observed} x_ij = x̄_j`.
  The mean is the no-covariate optimal predictor of the column.
- **Mean-preserving constant.** Filling `n − n_obs` holes with `c` makes the completed mean
  `(n_obs · x̄_j + (n − n_obs) · c)/n`, which equals `x̄_j` iff `c = x̄_j`. The mean is the
  unique constant that leaves the column mean intact.

## What it distorts (exact, and why it is acceptable here)

Filling holes with the mean has known, derivable costs in an *inferential* setting:

- **Variance is deflated by an exact factor.** Imputed entries sit at `x̄_j`, contributing
  zero squared deviation, so the sum of squared deviations is unchanged at `(n_obs − 1)·s²_obs`
  but is now divided by `n − 1`:
  `s²_completed = (n_obs − 1)/(n − 1) · s²_obs < s²_obs`.
- **Correlations attenuate toward zero.** Imputed points form a flat band at height `x̄_j`
  independent of the other variables, diluting any true tilt and pulling Pearson `r` toward 0.
- **Standard errors are understated.** A single imputation treats the filled value as known
  with certainty and never propagates imputation uncertainty.

These facts make mean imputation a poor tool for estimating data statistics, but a
*supervised* pipeline re-fits the predictor on the completed table and minimizes prediction
risk, not statistical fidelity — a different objective under which the distortions need not
matter.

## Why it works for prediction (and where it fails)

- **Plug-in into a fixed predictor is biased.** With `X ~ U(0,1)`, `Y = X² + ε`, MCAR, the
  optimal prediction when `X` is missing is `E[X²] = 1/3`, but applying the clean-data
  predictor `f⋆(x) = x²` to the filled mean gives `(E[X])² = 1/4`. Averaging does not commute
  with a nonlinear `f⋆` (`E[g(X)] ≠ g(E[X])`).
- **Impute-then-learn is consistent.** Imputing the *same* constant `α` in train and test and
  fitting a universally consistent learner on the completed table converges to the
  Bayes-optimal predictor for inputs-with-holes, under non-informative missingness (MAR with a
  smooth missingness probability). Away from `α`, the learner sees ordinary observed rows; at
  `α` with no missingness probability, it again sees an ordinary observed point; at `α` with
  positive missingness probability, the imputed spike identifies the missing rows and the
  learner fits their conditional mean. The constant need not be a plausible feature value —
  only a stable marker — so the mean works as well as any constant, and re-fitting launders the
  "wrong" value into a correct prediction.
- **Hard requirement:** the same learned means must be used at train and test (otherwise the
  imputed spike moves and the learner cannot recognize it), and `fit` must not use the target.
- **Failure mode:** informative missingness (the fact of being missing shifts `Y`) breaks the
  fixed plug-in story: for `X ~ U(0,1)`, `Y = X` if observed and `Y = 3X` if missing, a
  complete-case predictor gives `1/2` after filling with `E[X]`, while the missing-row optimum
  is `3/2`. A re-fit flexible learner can sometimes use a stable spike as a proxy for the mask,
  but if the mask is predictive and the completed values do not keep it distinguishable, the
  remedy is to add a missingness-indicator column, which is a different method.

## Final algorithm

```
fit(X):                                  # X: (n, d), NaN = missing; no target
    for each column j:
        statistics_[j] = mean of the observed (non-NaN) entries of column j
        if column j is all-NaN: statistics_[j] = 0.0   # keep output finite

transform(X):                            # train or test table with NaNs
    X_imputed = copy(X)
    for each missing cell (i, j):
        X_imputed[i, j] = statistics_[j]
    return X_imputed                      # finite, same shape
```

## Working code

For columns with observed values, this mirrors `sklearn.impute.SimpleImputer(strategy='mean')`:
`fit` stores per-column means over observed entries via the masked-array mean, and `transform`
scatters them into the holes. The all-missing-column `0.0` marker follows the same-shape
contract and sklearn's `keep_empty_features=True` behavior; sklearn's default instead drops
such columns.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """Unconditional mean imputation: replace each missing entry of a column with
    the mean of that column's observed entries. Per-column, deterministic, single
    imputation; the same learned means are replayed on any table."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # c_j = mean over the observed entries of column j (least-squares constant;
        # the unique mean-preserving constant). No target, no test data.
        missing_mask = np.isnan(X)
        mean_masked = np.ma.mean(np.ma.masked_array(X, mask=missing_mask), axis=0)
        self.statistics_ = np.ma.getdata(mean_masked).astype(float)
        # Same-shape contract for all-missing columns: mirror keep_empty_features=True.
        self.statistics_[np.ma.getmaskarray(mean_masked)] = 0.0
        return self

    def transform(self, X):
        X_imputed = np.array(X, dtype=float, copy=True)
        # Write the SAME learned column means into the holes (train or test): a fixed
        # per-column spike the downstream learner re-fits around.
        nan_mask = np.isnan(X_imputed)
        col_idx = np.where(nan_mask)[1]
        X_imputed[nan_mask] = np.take(self.statistics_, col_idx)
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

The per-column loop in `fit`/`transform` is equivalent and matches the task contract:

```python
def transform(self, X):
    X_imputed = X.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X_imputed[:, j])
        X_imputed[mask, j] = self.statistics_[j]
    return X_imputed
```
