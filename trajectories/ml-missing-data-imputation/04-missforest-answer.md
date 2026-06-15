**Problem (from step 3).** MICE's chained-equations skeleton fixed California (RMSE 0.928 → 0.872, R² up) and excelled on Breast Cancer (0.414), but its *linear* per-column engine failed on Wine (RMSE 0.943, worse than KNN's 0.796) — 178 rows is a thin fit and a linear conditional mean misses non-linear feature relationships — with wide seed-to-seed swings (California seed 123 hit 1.155).

**Key idea.** Keep the round-robin skeleton, swap only the per-column engine: replace the linear regression with a **random forest**. Each tree splits on the best threshold among a random feature subset over a bootstrap resample; the recursive partition captures non-linear response surfaces and interactions for free, with no parametric family, and averaging over many trees damps the variance that swung the thin linear fit.

**Why.** Same chained structure handles the multivariate circularity: keep a complete working matrix (mean init), sweep columns ascending in missingness, fit a forest on each column's genuinely-observed values against the current fill of the other columns, overwrite its holes, Gauss-Seidel within the sweep. `max_features="sqrt"` decorrelates the trees so averaging helps; 100 trees sits at the accuracy/runtime knee; deep unpruned trees with averaging absorbing the variance. Stop on relative squared change `diff/denom < 1e-4`.

**Scaffold edit / hyperparameters.** A hand-rolled chained-RF loop in `_fit_transform_internal`: `n_estimators=100`, `max_features="sqrt"`, `max_iter=10`, `tol=1e-4`. Note this slot re-runs the full iterative fit on every `transform` call (no frozen model); harmless here since the loop calls `fit_transform` once on the scored matrix. The stop rule is the forward threshold (stop when relative change first drops below `tol`, return latest), not the canonical MissForest "stop when the change turns back up."

**What to watch.** Wine is the test — a forest should beat KNN's 0.796 (RMSE below MICE's 0.943) with a tighter seed spread; California should hold/improve over 0.872 and tighten its seed variance (no 1.155 outlier). The risk is Breast Cancer: MICE's linear engine was near-optimal there (0.414) on near-duplicate linear columns, and a forest's piecewise-constant fits are coarser, so its RMSE may come in slightly above 0.414 — a net win on the geometric-mean score, which is dominated by the weakest dataset (Wine).

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """MissForest: Iterative Random Forest imputation.

    Implements the MissForest algorithm (Stekhoven & Buehlmann, 2012):
    1. Initial imputation with column means
    2. For each iteration:
       a. Sort features by missingness (ascending)
       b. For each feature with missing values:
          - Train RandomForest on observed entries using all other features
          - Predict missing entries
       c. Check convergence (normalized difference < tol)
    3. Return when converged or max_iter reached

    Reference: Bioinformatics 28(1):112-118, 2012.
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = 100
        self.tol = 1e-4

    def fit(self, X, y=None):
        # Store the fitted state by running fit_transform internally
        self._X_fitted = X.copy()
        self._fit_transform_internal(X)
        return self

    def transform(self, X):
        return self._fit_transform_internal(X)

    def fit_transform(self, X, y=None):
        return self._fit_transform_internal(X)

    def _fit_transform_internal(self, X):
        from sklearn.ensemble import RandomForestRegressor

        X_imp = X.copy()
        n_samples, n_features = X_imp.shape

        # Step 1: Initial imputation with column means
        col_means = np.nanmean(X_imp, axis=0)
        for j in range(n_features):
            mask_j = np.isnan(X_imp[:, j])
            X_imp[mask_j, j] = col_means[j]

        # Identify which features have missing values and sort by missingness
        miss_count = np.isnan(X).sum(axis=0)
        features_with_missing = np.where(miss_count > 0)[0]
        # Sort by number of missing values (ascending)
        features_with_missing = features_with_missing[
            np.argsort(miss_count[features_with_missing])
        ]

        if len(features_with_missing) == 0:
            return X_imp

        # Step 2: Iterative imputation
        for iteration in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in features_with_missing:
                # Observed and missing indices for feature j
                obs_mask = ~np.isnan(X[:, j])
                mis_mask = np.isnan(X[:, j])

                if mis_mask.sum() == 0:
                    continue

                # Predictor features (all except j)
                other_features = [k for k in range(n_features) if k != j]
                X_train = X_imp[obs_mask][:, other_features]
                y_train = X[obs_mask, j]  # Use original observed values
                X_pred = X_imp[mis_mask][:, other_features]

                # Train random forest and predict
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features="sqrt",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imp[mis_mask, j] = rf.predict(X_pred)

            # Step 3: Check convergence
            diff = np.sum((X_imp - X_prev) ** 2)
            denom = np.sum(X_imp ** 2)
            if denom > 0 and diff / denom < self.tol:
                break

        return X_imp


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
