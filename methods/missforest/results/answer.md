# MissForest, distilled

MissForest is an iterative, nonparametric imputer for tabular data with missing entries. It
takes the chained-equations (round-robin) skeleton — fill, then repeatedly re-impute each
variable by regressing it on all the others — and uses a **random forest** as the per-variable
predictor. That single choice buys mixed-type handling (continuous and categorical
simultaneously), scale invariance (no standardization), nonlinearities and interactions for
free, no parametric family to specify, and — crucially — **no tuning parameter the user must
set**. The forest's out-of-bag error additionally yields an estimate of the imputation error
without any held-out set or cross-validation.

## Problem it solves

A data matrix `X` of shape `n × p` has missing entries (`NaN`), and downstream analysis needs a
finite-valued completion `X_imp` whose filled values are close to the unknown truth. The hard
regime — real biological/medical data — has all of: mixed variable types, nonlinear relations
and interactions, variables on unequal scales, sometimes `p ≫ n`, and no prior knowledge to
guide model or hyperparameter choices.

## Key idea

Iterative block-coordinate refinement of a complete working matrix:

1. **Initial fill.** Replace every hole with the column mean (numeric) or modal class
   (categorical), so all predictor blocks are defined from the first sweep.
2. **Order.** Sort the variables with missingness in **ascending** order of number of missing
   values: the most-complete variable trains the cleanest forest first, and its improved column
   then serves as a better predictor for the harder variables later in the same sweep.
3. **Sweep (Gauss-Seidel).** For each variable `X_s` with missing entries, in that order:
   - train a random forest with **observed** `X_s` as the response and the **current** values of
     all other columns (at the observed-`s` rows) as predictors;
   - predict the missing `X_s` from the other columns at the missing-`s` rows;
   - overwrite those holes in the working matrix (so the next variable sees the update).
4. **Stop.** Track the sweep-to-sweep change of the imputed matrix. **Stop the first time it
   increases**, and return the iterate from *just before* the increase (the minimum). The change
   descends as the imputation settles, then turns up once it enters the forest-noise jitter, so
   the minimum is the best estimate. A `max_iter` cap is a safety net; on continuous data the
   rule typically fires after ≈5 sweeps.

Why a random forest in the per-variable slot: CART splits handle numeric and categorical
predictors natively and are invariant to monotone rescaling; the tree partition captures
nonlinearities and interactions automatically; there is no distributional family to choose; and
the defaults need no tuning — the contrast with KNNimpute (tuned `k`, required standardization,
continuous-only) and penalized/parametric chained equations (a model per variable, a penalty to
cross-validate).

## Convergence criterion (exact)

Per sweep, on the working matrix `X_new` versus the previous sweep's `X_old`:

- continuous set `N`:  `Δ_N = Σ_{j∈N}(X_new − X_old)² / Σ_{j∈N}(X_new)²`  (relative squared change)
- categorical set `F`: `Δ_F = (# categorical entries changed) / (# missing categorical entries)`

Continue while the relevant `Δ` is still **decreasing**; stop the first sweep it **increases**
and return the previous iterate. With both types present, stop only once **both** `Δ_N` and
`Δ_F` have turned up (continue while either is still decreasing). If `max_iter` is reached first,
return the latest iterate.

## Error estimate (out-of-bag), for free

Each tree's bootstrap sample omits on average `(1 − 1/n)^n → 1/e ≈ 36.8%` of rows, so every row
is out-of-bag for about 36.8% of the trees. Predicting each row from only the trees that did not see it
gives an essentially unbiased internal error estimate (Breiman 2001), as accurate as a same-size
test set. Since a forest is fit per variable, aggregating the OOB errors by type — average OOB
MSE over numeric variables (normalized to NRMSE), average OOB misclassification over factors
(PFC) — estimates the imputation error with **no test set and no cross-validation**.

## Defaults and why

- **`m_try = floor(sqrt(p))`** candidate variables per split. Breiman's classification bound
  `PE* ≤ ρ̄(1 − s²)/s²` and regression bound `PE*(forest) ≤ ρ̄ · PE*(tree)` both expose the same
  tradeoff: keep trees strong, but reduce the average correlation `ρ̄` between their errors. A
  small random feature subset decorrelates the trees at a limited cost to individual-tree
  strength; `sqrt(p)` is the standard near-optimal default. `m_try = 1` forces the split variable
  (no longer a forest) and hurts; `m_try → p` raises `ρ̄` and runtime for little gain.
- **`n_tree ≈ 100`.** By the SLLN the forest error converges as trees are added — more trees never
  overfits, only costs runtime (≈linear in `n_tree`). 100 sits at the precision/cost knee where
  accuracy has flattened.
- **deep, unpruned trees; min node size ≈5 (regression), bootstrap rows.** Deep trees give low
  bias; averaging over the forest absorbs their variance; bootstrap both decorrelates trees and
  provides the OOB estimate.
- **`max_iter = 10`.** A safety cap; the difference-based stop usually fires first.
- **no user tuning parameter, no required standardization.** The headline ergonomic win.

## Algorithm

```
initial fill: every hole <- column mean (numeric) / modal class (categorical)
order <- variables with missingness, sorted by increasing count of missing values
delta_old <- +inf
repeat up to max_iter sweeps:
    X_old <- current imputed matrix
    for s in order:                       # Gauss-Seidel: use the latest values
        fit RF:  observed(X_s) ~ other columns of current matrix at observed-s rows
        predict the missing X_s from other columns at missing-s rows
        overwrite X_s's holes with the predictions
    X_new <- current imputed matrix
    delta_new <- sum((X_new - X_old)^2) / sum(X_new^2)
    if delta_new > delta_old:  return X_old      # iterate just before the rise (the minimum)
    delta_old <- delta_new
return current matrix                            # reached only if the cap fires first
```

## Working code

Faithful continuous-data realization (sklearn). The categorical case is the direct analogue:
modal-class initial fill, a `RandomForestClassifier` per factor variable, and the `Δ_F`
convergence test.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class MissForest(BaseEstimator, TransformerMixin):
    """Iterative random-forest imputation (MissForest).

    Chained-equations round-robin with a random forest as the per-variable
    predictor: complete mean fill, then sweep variables in increasing order of
    missingness, fitting a forest on each variable's observed part against the
    current values of the others and overwriting its holes; stop when the
    sweep-to-sweep change first increases (return the previous iterate).
    """

    def __init__(self, random_state=42, max_iter=10, n_estimators=100):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        self._impute(X)           # learn from observed entries only; no test labels
        return self

    def transform(self, X):
        return self._impute(X)

    def fit_transform(self, X, y=None):
        return self._impute(X)

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        missing = np.isnan(X)                         # original missingness pattern

        # initial complete fill: column means (best no-covariate constant guess)
        col_mean = np.nanmean(X, axis=0)
        X_imp = X.copy()
        for j in range(n_features):
            X_imp[missing[:, j], j] = col_mean[j]

        # variables to impute, sorted by ascending missingness
        miss_count = missing.sum(axis=0)
        order = [j for j in np.argsort(miss_count) if miss_count[j] > 0]
        if not order:
            return X_imp

        mtry = max(1, int(np.floor(np.sqrt(n_features))))  # canonical m_try = floor(sqrt(p))
        X_best = X_imp.copy()
        delta_old = np.inf
        for _ in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in order:                           # Gauss-Seidel sweep
                obs = ~missing[:, j]
                mis = missing[:, j]
                others = [k for k in range(n_features) if k != j]
                if not others:
                    X_imp[mis, j] = np.nanmean(X[obs, j])
                    continue

                X_train = X_imp[obs][:, others]       # current fill of other columns
                y_train = X[obs, j]                   # genuinely observed targets
                X_pred = X_imp[mis][:, others]

                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,   # ~100: error flattened, runtime ~linear
                    max_features=min(mtry, len(others)),
                    min_samples_leaf=5,                # R randomForest regression nodesize default
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imp[mis, j] = rf.predict(X_pred)

            denom = np.sum(X_imp ** 2)
            delta_new = np.sum((X_imp - X_prev) ** 2) / denom if denom > 0 else 0.0
            if delta_new > delta_old:                 # change turned up -> stop
                return X_best                         # the iterate at the minimum
            X_best = X_imp.copy()
            delta_old = delta_new

        return X_best                                 # cap reached: latest iterate
```

Equivalent sklearn harness: `IterativeImputer(estimator=RandomForestRegressor(...),
initial_strategy="mean", imputation_order="ascending", max_iter=...)` supplies the same
round-robin shell. It is not the exact original stop rule: sklearn stops when an infinity-norm
change falls below `tol` and returns the latest iterate, while the method above stops when the
relative squared change rises and returns the previous iterate.
