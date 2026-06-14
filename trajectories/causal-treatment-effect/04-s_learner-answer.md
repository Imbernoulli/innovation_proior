**Problem (from step 3).** DR-learner won where it had data and hard surfaces (`acic_synth` PEHE 0.501)
but lost the small dataset to a plain T-learner (`ihdp_synth` 1.383 > 1.14): every correction layer —
propensity inversion, augmentation, cross-fitting — is a layer of variance, and at $n=747$ the variance
dominates. The next move is *down* the complexity axis: the lowest-variance pooled estimator.

**Key idea.** Pool all rows into one surface $\mu(x,w)=E[Y\mid X=x,T=w]$ with the treatment flag as an
ordinary feature, and read $\hat\tau(x)=\hat\mu(x,1)-\hat\mu(x,0)$ by toggling it. The "S" is single. The
shared baseline is learned *once* with all $N$ rows and cancels exactly in the toggle; where the effect
is simple the learner declines to split on the flag and reports near-zero — a bias *toward the truth*
on these simple-effect DGPs.

**Why it works.** No thin-arm double-estimation (unlike T), no $1/\hat e$ residual correction (unlike
DR) — just one low-variance pooled fit. Cost: a regularized learner can attenuate a *real but weak*
effect to zero, so where the covariates dominate (`acic_synth`, $p=50$) it should lose to
orthogonalization. The S-learner and DR-learner are complements, not a strict ordering.

**Hyperparameters.** One `GradientBoostingRegressor` (200 trees, depth 4, leaf floor 20,
`subsample=0.8`), seeded from `os.environ["SEED"]`. Depth $>1$ is load-bearing — a covariate split
beneath a flag split *is* heterogeneity.

**What to watch.** Bet on complementarity / a crossing: S-learner should win `ihdp_synth` (PEHE below
1.0, pooling the shared baseline) and beat DR's ATE error on `jobs_synth`, but lose `acic_synth` to DR's
orthogonalization. The fix the next rung needs: keep orthogonalization (to win `acic_synth`) at low
variance (to win the small sets) — residualize once, fit the effect directly.

```python
class CATEEstimator(BaseCATEEstimator):
    """S-Learner: single model approach to CATE estimation.

    Fits a single outcome model mu(X, T) on the combined data, then
    estimates CATE as mu(X, 1) - mu(X, 0).
    Uses GradientBoostingRegressor as the base learner for flexibility.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed,
        )

    def fit(self, X, T, Y):
        n, p = X.shape
        XT = np.column_stack([X, T.reshape(-1, 1)])
        self._model.fit(XT, Y)
        return self

    def predict(self, X):
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])
        X0 = np.column_stack([X, np.zeros((n, 1))])
        return self._model.predict(X1) - self._model.predict(X0)
```
