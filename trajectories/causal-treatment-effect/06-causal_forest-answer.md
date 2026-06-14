**Problem (from step 5).** The R-learner reached "orthogonalization at low variance" — best ATE errors,
no clear loss anywhere — but its *single* gradient-boosted final stage on the heavy-tailed ratio target
$\tilde Y/\tilde T$ still fought variance on the small/mid sets (`jobs_synth` PEHE 441 lost to the
S-learner's 396; `ihdp_synth` 0.813 only tied it). Keep R's residualization; replace the final stage
with a lower-variance, locally-adaptive learner.

**Key idea.** A forest is an *adaptive kernel*: it learns which directions matter and pools the right
neighbors — exactly what a global GBT lacks in $p=50$ dimensions. Run R's exact Robinson residualization
($\hat m,\hat e$ cross-fit, $\tilde Y,\tilde T$, the same `safe_T` guard, 95th-pct clip, $\tilde T^2$
weight), then fit the residualized R-loss target with a **random forest** (500 trees) instead of one
boosted regressor. Confounding-robustness from the residualization; lower-variance heterogeneity from the
forest average.

**Why it works.** 500 deep (`min_samples_leaf=5`), decorrelated (`max_features="sqrt"`) trees average
into a smoother estimate of the noisy effect target than a single regressor — the variance reduction the
small/mid sets needed — while the depth captures sharper heterogeneity. It is structurally "the R-learner
with a forest final stage."

**Harness reality (load-bearing).** The fill tries `econml.dml.CausalForestDML`, but **econml is not
installed here**, so the `ImportError` *fallback* runs and produced the measured numbers: manual DML with
`KFold(3)` (not 5), the R-loss pseudo-outcome, and a `RandomForestRegressor` final stage. The GRF
apparatus (honest splits, effect-seeking split criterion, bootstrap-of-little-bags intervals;
`inference=False` anyway) is the method's natural form but is *not* exposed by this run path.

**What to watch.** The forest final stage should win the two sets R lost/tied — `ihdp_synth` below 0.81
(~0.77) and `jobs_synth` below 441 (~360) — even ceding `acic_synth` back to R by a hair (~0.50 vs
0.421), giving the lowest variance-adjusted PEHE ceiling of any method.

```python
class CATEEstimator(BaseCATEEstimator):
    """Causal Forest (via econml CausalForestDML).

    Combines double machine learning (DML) for debiasing with
    generalized random forests for heterogeneous effect estimation.

    Steps:
    1. Cross-fit nuisance models: E[Y|X] and E[T|X]
    2. Compute residuals: Y_res = Y - E[Y|X], T_res = T - E[T|X]
    3. Fit a causal forest on residualized outcomes

    Falls back to a pure-sklearn implementation if econml is unavailable.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._use_econml = True
        try:
            from econml.dml import CausalForestDML
            self._cf = CausalForestDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=self._seed,
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=self._seed + 1,
                ),
                n_estimators=500,
                min_samples_leaf=5,
                max_depth=None,
                honest=True,
                inference=False,
                random_state=self._seed + 2,
                cv=3,
            )
        except ImportError:
            self._use_econml = False
            # Fallback: manual residualization + random forest
            self._model_y = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=self._seed,
            )
            self._model_t = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=self._seed + 1,
            )
            self._cate_model = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5,
                max_features="sqrt", random_state=self._seed + 2,
            )

    def fit(self, X, T, Y):
        if self._use_econml:
            self._cf.fit(Y, T, X=X)
        else:
            # Manual DML: cross-fit residuals
            kf = KFold(n_splits=3, shuffle=True, random_state=self._seed)
            Y_res = np.zeros_like(Y)
            T_res = np.zeros_like(T, dtype=float)

            for train_idx, val_idx in kf.split(X):
                my = clone(self._model_y).fit(X[train_idx], Y[train_idx])
                mt = clone(self._model_t).fit(X[train_idx], T[train_idx])
                Y_res[val_idx] = Y[val_idx] - my.predict(X[val_idx])
                T_res[val_idx] = T[val_idx] - mt.predict_proba(X[val_idx])[:, 1]

            # R-Learner-style pseudo-outcome with stable divisor + sample
            # weighting so small |T_res| doesn't explode the fit.
            safe_T = np.where(np.abs(T_res) > 0.01, T_res, np.sign(T_res) * 0.01 + 1e-8)
            pseudo = Y_res / safe_T
            weights = T_res ** 2
            q = np.percentile(np.abs(pseudo), 95)
            pseudo = np.clip(pseudo, -q, q)
            self._cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        if self._use_econml:
            return self._cf.effect(X).flatten()
        else:
            return self._cate_model.predict(X)
```
