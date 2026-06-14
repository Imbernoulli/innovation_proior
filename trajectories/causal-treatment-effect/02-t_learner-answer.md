**Problem (from step 1).** IPW's pseudo-outcome divides by $\hat e(1-\hat e)$, so its variance is
uncontrolled — `jobs_synth` PEHE blew up to 5183.8 (earnings-scale $Y$ over near-clip $\hat e$). A
conditionally-unbiased signal is useless if its variance is unbounded; the fix is to put outcome models
back in and stop reweighting raw outcomes.

**Key idea.** Under unconfoundedness each response surface is a plain within-arm regression:
$\mu_0(x)=E[Y\mid X,T=0]$ on the control rows, $\mu_1(x)=E[Y\mid X,T=1]$ on the treated rows, and
$\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)$. Two models, one per arm — the "T".

**Why it works.** No division by $\hat e$, so no weighting blow-up; and because the model *is* the
treatment (separate fits per arm), the effect can never be attenuated to zero the way a pooled S-learner
shrinks a weak $T$ feature. Cost: the arms are fit in isolation, so $\hat\tau$ inherits the
*response-surface* rate (not the often-faster effect rate), and two independent errors corrupt the
difference where $\tau$ is simpler than $\mu_w$.

**Hyperparameters.** Two independent `GradientBoostingRegressor`s (200 trees, depth 4, leaf floor 20,
`subsample=0.8`), seeded `seed` and `seed+1` from `os.environ["SEED"]` so the arms' stochastic fits are
decorrelated. Depth $>1$ lets each surface capture the DGP interactions; the leaf floor guards the
difference against leaf noise.

**What to watch.** Removing the $1/\hat e$ division should drop `jobs_synth` PEHE by more than an order
of magnitude (into the hundreds). Smallest gain expected on `acic_synth` ($p=50$, hard isolated
surfaces, thin treated regions) — the response-rate ceiling and lack of orthogonalization bite there,
which is what the residualized rungs should beat.

```python
class CATEEstimator(BaseCATEEstimator):
    """T-Learner: two separate models for treated and control groups.

    Fits mu0(X) on control data and mu1(X) on treated data, then
    estimates CATE as mu1(X) - mu0(X).
    Uses GradientBoostingRegressor for both models.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._model0 = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed,
        )
        self._model1 = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed + 1,
        )

    def fit(self, X, T, Y):
        mask0 = T == 0
        mask1 = T == 1
        self._model0.fit(X[mask0], Y[mask0])
        self._model1.fit(X[mask1], Y[mask1])
        return self

    def predict(self, X):
        return self._model1.predict(X) - self._model0.predict(X)
```
