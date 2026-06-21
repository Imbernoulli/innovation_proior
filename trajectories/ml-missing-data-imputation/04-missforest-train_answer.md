MICE confirmed two of my three predictions and the third is the reason to keep climbing. California — the matrix KNN regressed on — is fixed: RMSE fell to $0.872$ against the mean fill's $0.928$ and KNN's $0.933$, with R² up to $0.659$, because modelling each column from all eight jointly recovered the signal no handful of close rows carried. Breast Cancer improved spectacularly, RMSE $0.414$ against KNN's $0.586$, since a self-regularizing regression on thirty correlated features is the right tool and the conditional mean of a strongly-predictable column is sharper than a neighbour average. But Wine came in at $0.943$, *worse than KNN's $0.796$* and barely better than the mean fill's $1.033$, with an ugly per-seed spread $\{1.054, 0.872, 0.903\}$ — seed 42 actually above the mean-fill baseline. That is the failure I flagged: 178 rows make each per-column regression a thin fit, roughly 140 observed rows against twelve predictors, and a *linear* conditional mean cannot capture a non-linear feature relationship that KNN's local average could follow by bending to the data. California's per-seed RMSE was also wild, $\{0.782, 1.155, 0.680\}$, another tell that the linear model extrapolates badly in places. So the chained-equations *skeleton* is vindicated — it fixed what KNN broke — but the *linear per-column engine* is the new bottleneck: wrong wherever the feature relationships are non-linear, and unstable seed-to-seed where the fit is thin.

The fix is surgical: keep the round-robin skeleton — initialize complete, sweep the columns imputing each from the current fill of the rest, iterate to a fixpoint — and swap only what plugs into each per-column slot. MICE's structure handled the multivariate circularity correctly; that is not where it failed. What I want in the slot is a predictor that captures non-linearities and interactions on its own, with no parametric family to specify and no scale sensitivity, robust enough on a 178-row matrix not to swing across seeds the way the thin linear fit did. I propose **MissForest** (Stekhoven & Bühlmann, 2012): the same chained-equations round-robin with a **random forest** as the per-column engine. A forest grows many trees, each on a bootstrap resample of the rows, splitting at each node on the best threshold among a random subset of the features, and averages the trees for a regression prediction. Checked against the spec rather than asserted, it earns every box: a CART split asks "is feature $k$ below a threshold," invariant to any monotone rescaling, so no standardization sensitivity; the recursive partition represents non-linear response surfaces and interactions for free — an interaction is just a split on one feature nested inside a split on another, exactly what Wine's linear conditional mean could not do; there is no parametric family to choose; and averaging over many bootstrap trees damps the variance that made the thin linear fit swing.

Making it concrete in the scaffold runs into one snag, and resolving it is most of the method. To regress column $j$ on the others I need four pieces: the observed part of $j$ (training targets), the missing part (what I predict), the other columns at the rows where $j$ is observed (training inputs), and the other columns at the rows where $j$ is missing (the inputs I predict from). The snag is that those other-column blocks are themselves full of holes — the rows where $j$ happens to be observed are not magically complete elsewhere — so I cannot fit a forest on raw inputs containing `NaN`s. The way out is the trick the skeleton already uses: keep a fully completed working matrix at all times. Start by filling every hole with the column mean, so from the first sweep every predictor block is finite. Those initial fills are wrong, but they are only a starting point the iteration refines. Within a sweep, when I impute column $j$, I train the forest with the *genuinely observed* values of $j$ as the target — never my own guesses for $j$ — and the other columns of the current working matrix restricted to the observed-$j$ rows as inputs; then I predict $j$'s holes from the other columns at the missing-$j$ rows; then I overwrite those holes in the working matrix. The next column is imputed using the matrix I just updated — Gauss-Seidel, not Jacobi — so an earlier column's improved fill immediately helps a later column's regression within the same pass, which is why a couple of sweeps already mix structure across all the columns. The visit order is the same ascending-missingness argument as MICE: a forest is only as good as its training set, and the cleanest training set belongs to the column with the fewest holes, so the best-trained forests run first and their improved columns become better predictors for the harder columns later in the sweep.

The forest's own knobs are set by argument, not hand-tuning, or I would have reintroduced the cost I was fleeing. The number of features considered at each split: not all $p$, because the forest's error is the *averaged* tree error and averaging only helps when the trees are decorrelated — if every tree saw all features at every node they would find nearly the same splits, be highly correlated, and averaging would buy almost nothing. Restricting each node to a random subset decorrelates the trees at the cost of each being a touch weaker, and the standard choice capturing "small, decorrelating, not degenerate" is `max_features="sqrt"`, namely $\lfloor\sqrt{p}\rfloor$, well below $p$ and comfortably above $1$. The number of trees: adding trees only tightens the estimate toward its limit and never overfits, so more is weakly better in accuracy with runtime roughly linear in the count, and a hundred sits at the knee where accuracy has flattened but runtime has not yet ballooned. So `n_estimators=100`, `max_features="sqrt"`, trees grown deep and unpruned with averaging absorbing the variance. The convergence test reads the imputation's own dynamics instead of imposing a per-dataset tolerance: the relative, dimensionless quantity is the total squared change in the working matrix from the previous sweep, $\text{diff}/\text{denom} = \sum (X_{\text{imp}} - X_{\text{prev}})^2 / \sum X_{\text{imp}}^2$, and when it drops below `1e-4` the fills have effectively stopped moving and further sweeps only churn forest noise, so I stop, with a generous ten-sweep cap that rarely fires on continuous data. I note this is the simpler *forward* threshold — stop when the relative squared change first falls below `tol`, return the latest matrix — not the canonical MissForest rule that watches the change turn back up and returns the previous iterate; at a hundred trees the noise jitter the turn-up rule guards against is small.

One honesty about the scaffold edit: in this slot `fit` runs the full iterative procedure internally and `transform` runs it *again* on whatever matrix it is handed, so the imputer does not freeze a learned set of per-column forests and replay them — each call re-runs the chained-equations fit on the matrix in front of it. Under this benchmark that is harmless, because the loop calls `fit_transform` once on exactly the corrupted matrix being scored; it does mean this fill is not a freeze-and-replay predictor in the sklearn sense the earlier rungs were, but it matches what the task evaluates. Wine is the test of the move — a forest capturing non-linear relationships should beat KNN's $0.796$, RMSE well below MICE's $0.943$, with a tighter seed spread from averaging — and California should hold or improve over $0.872$ with its seed variance tightening, no more $1.155$ outlier, because the forest does not extrapolate the way the linear model did. The open risk is Breast Cancer: MICE's linear engine was already near-optimal there at $0.414$, because thirty near-duplicate columns are a regime where a regularized linear conditional mean is close to optimal and a forest's piecewise-constant predictions are *coarser* than a smooth linear fit on genuinely linear structure — so MissForest's Breast Cancer RMSE may come in slightly above $0.414$, trading a little accuracy on the linear matrix for the large gains on the non-linear ones. That is still a net win on the geometric-mean score, which is dominated by the weakest dataset, and Wine was the weak link. If even the forest leaves a non-linear matrix wanting, the next move is a different tree family in the same slot — which is exactly where the strongest rung goes.

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
