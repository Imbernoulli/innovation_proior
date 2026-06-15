**Problem.** A standardized `(n, d)` matrix arrives with 20% MCAR holes; downstream tools cannot consume `NaN`, so it must be completed to a finite same-shape matrix. The floor: complete it using nothing about each row, only each column.

**Key idea (the floor).** Fill every hole in column `j` with that column's observed mean. Two derivations land on the same constant: it is the least-squares constant predictor of the column (`argmin_c Σ_obs (x_ij − c)² = x̄_j`), and it is the unique constant that leaves the completed column mean intact. Under MCAR the observed entries are an unbiased sample, so the observed mean is also the squared-error-optimal *no-covariate* fill for the hidden masked cells — exactly the per-cell RMSE I am scored on, restricted to ignoring the rest of the row.

**Step-1 edit.** This is the scaffold default and the baseline keeps it verbatim: `fit` stores per-column means with `np.nanmean`; `transform` scatters the stored mean into each column's holes; `fit_transform` chains them. It is univariate, deterministic, single-valued, and blind to every inter-feature correlation — the floor by construction.

**Why it is weak (and what it forfeits).** Filling the mean adds zero scatter, so the completed variance deflates by exactly `(n_obs − 1)/(n − 1)` (≈0.8 at 20% missing on unit-variance columns), and imputed points form a flat band that pulls correlations toward zero. RMSE suffers most where features are predictable from each other (worst on the wide, correlated Breast Cancer matrix); the downstream `GradientBoosting` is more forgiving because it re-fits on the completed matrix and can treat the per-column spike as its own region.

**Hyperparameters.** None learned beyond the `d` column means. `random_state`/`max_iter` are carried for API compatibility and unused.

**What to watch.** The gap between a poor RMSE (standardized units near/above 1) and respectable downstream scores — that gap is the structure the next rung claims by reading each row's observed coordinates.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """Mean Imputation: replace missing values with column means."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X_imputed = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_imputed[:, j])
            X_imputed[mask, j] = self.statistics_[j]
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def compute_feature_correlations(X):
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr
```
