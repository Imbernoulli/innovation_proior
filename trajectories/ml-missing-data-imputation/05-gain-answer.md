**Problem (from step 4).** MissForest fixed Wine (RMSE 0.740, beats KNN/MICE) and tightened California (0.669, no outlier seed), a net gmean win, but left two inefficiencies: a small give-back vs MICE on the linear Breast Cancer matrix (0.478 vs 0.414, the forest under-smooths smooth structure) and an order-of-magnitude runtime cost from the exhaustive best-split search per node. The bottleneck shifted from accuracy to the tree estimator itself.

**Key idea.** Keep the chained-equations skeleton; refine the per-column model *within* the tree family: swap the bagged random forest for **extremely randomized trees**. At each node they draw a *random* threshold per candidate feature (no exhaustive search) and grow on the whole sample. The extra randomization further decorrelates the trees — lower ensemble variance and a smoother, finer-grained response surface — while skipping the costly split search.

**Why.** Lower variance steadies the thin small-matrix fits (Wine seeds tighten); the smoother ensemble claws back the Breast Cancer under-smoothing; the random splits make each tree far cheaper. Weaker individual trees are absorbed by the same averaging the forest relied on. Run through `IterativeImputer`'s freeze-and-replay shell (mean init, ascending order, per-feature fit/predict, max-abs-change `tol=1e-3` early stop), targeting the conditional mean — the squared-error-optimal point fill.

**Lineage note.** The slot is named for the generative-adversarial imputation idea (generator fills holes from masked noise; discriminator predicts componentwise which entries were observed; hint mechanism), which *samples* completions from the conditional distribution. That is unfit here: a numpy GAN cannot converge on 178–5,000 standardized rows, and its deliberate sampling scatter inflates squared-error RMSE. This rung keeps the *goal* (model rich non-linear dependence) and uses `ExtraTreesRegressor` in `IterativeImputer`, which converges reliably to the conditional mean on small data.

**Scaffold edit / hyperparameters.** `IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1), max_iter=10, imputation_order="ascending", initial_strategy="mean", tol=1e-3)`.

**What to watch.** Breast Cancer RMSE should fall below MissForest's 0.478 toward MICE's 0.414 (not all the way); Wine holds/improves on 0.740 with tighter seeds; California holds/edges below 0.669, with the downstream R² holding/improving on 0.687. A consistent small improvement across all three plus lower runtime — the give-back dataset partly repaired without surrendering the non-linear gains, lifting the weakest dataset and the gmean.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """Iterative imputation with ExtraTreesRegressor.

    Uses sklearn's IterativeImputer with ExtraTreesRegressor as the
    estimator. ExtraTrees captures non-linear feature dependencies
    (similar to GAIN's goal) but converges reliably. Each feature
    with missing values is modeled as a function of all other features,
    iterated in round-robin until convergence.

    This replaces the original numpy GAIN (GAN) baseline which could
    not converge due to incomplete backpropagation.
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = 100

    def _make_imputer(self):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )
        return IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )

    def fit(self, X, y=None):
        self._imputer = self._make_imputer()
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        self._imputer = self._make_imputer()
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
