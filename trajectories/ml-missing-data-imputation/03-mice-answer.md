**Problem (from step 2).** KNN beat the mean fill on the wide correlated matrices (Breast Cancer RMSE 0.994 → 0.586) but *regressed* on narrow continuous California (RMSE 0.933 > 0.928, R² 0.612 < 0.646): with 8 features the co-observed overlap is tiny and a local average over noisy neighbours of a continuous target is worse than the stable column mean. KNN *copies* nearby rows; when no row is close it has nothing.

**Key idea.** Model, don't copy. Regress each incomplete column on all the others and impute the holes. The predictors are themselves incomplete and the dependence is circular, so break it: fill everything with column means, then re-impute each column in turn from the *current* fill of the rest, sweeping repeatedly. Each column's fill re-enters only indirectly through the other columns, so the chain mixes in a handful of sweeps.

**Why.** This is a Gibbs sweep over per-column conditionals. The benchmark scores squared-error RMSE and downstream prediction, not calibrated inference, so the right per-cell fill is the *conditional mean* (minimizes squared error), not a posterior draw — use the point version (`sample_posterior=False`). The per-column engine is `BayesianRidge`, whose ridge strength is set automatically by evidence maximization, so the wide collinear Breast Cancer matrix regularizes itself with no hand-tuned penalty; its posterior-mean prediction is the conditional mean. Ascending visit order imputes the most-reliable columns first.

**Scaffold edit / hyperparameters.** Fill the slot with `IterativeImputer(estimator=BayesianRidge(), max_iter=30, random_state=seed, imputation_order="ascending", initial_strategy="mean", tol=1e-3)`. `max_iter=30` is a generous cap (the `tol=1e-3` scaled-change early stop fires first); `initial_strategy="mean"` is the complete starting table.

**What to watch.** California is the test of the whole move — a linear model using all 8 features jointly should beat the mean fill there (RMSE below 0.928, R² above 0.646), undoing KNN's regression; Breast Cancer should hold or improve. The risk is Wine: 178 rows is a thin per-column fit and a *linear* conditional mean misses non-linear feature relationships a local average could follow, so Wine RMSE may lag KNN's 0.796.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """MICE: Multiple Imputation by Chained Equations.

    Uses sklearn.impute.IterativeImputer with BayesianRidge estimator.
    Reference: van Buuren & Groothuis-Oudshoorn (2011).
    """

    def __init__(self, random_state=42, max_iter=30):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )
        return self._imputer.fit_transform(X)


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
