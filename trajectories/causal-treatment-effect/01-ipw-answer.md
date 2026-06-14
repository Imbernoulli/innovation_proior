**Problem.** Estimate $\tau(x)=E[Y(1)-Y(0)\mid X=x]$ from confounded observational data. The naive
treated-minus-control gap mixes the effect with who-gets-treated; the scaffold's placeholder Ridge
S-learner is linear and shrinks the lone treatment coefficient, so it captures neither the nonlinear
response nor the heterogeneity.

**Key idea.** Adjust through the *assignment* side, not the outcome. The propensity $e(x)=P(T=1\mid X)$
is the coarsest balancing score ($X\perp T\mid e(X)$), collapsing high-dimensional confounding into one
scalar. Treat the treated arm as a biased sample drawn with inclusion probability $e(X)$; the
Horvitz-Thompson argument *forces* inverse-probability weighting, and the per-unit pseudo-outcome
$\psi=T\,Y/e(X)-(1-T)\,Y/(1-e(X))$ has $E[\psi\mid X=x]=\tau(x)$. So fit $\hat e$ by a probability
classifier, form $\psi$, and regress $X\to\psi$ — the conditional mean *is* $\hat\tau(x)$.

**Why it works.** The $e(X)$ in the weight exactly cancels the $e(X)$ over-representation of treated
units, so $\psi$ is conditionally unbiased for $\tau$ under unconfoundedness with *no* outcome model;
regressing $\psi$ on $X$ smooths the per-unit noise into a function and gives the ATE as its mean.

**Hyperparameters.** Propensity: `GradientBoostingClassifier` (200 trees, depth 3) for `predict_proba`
on the nonlinear DGP propensities. Clip $\hat e$ to $[0.05,0.95]$ (matching the DGP's own overlap clip)
so every IPW weight is $\le 20$. Outcome regressor on $\psi$: `GradientBoostingRegressor` (200 trees,
depth 4). Seeds split via `os.environ["SEED"]` (`seed`, `seed+1`).

**What to watch.** IPW should be the weakest rung: $\psi$ divides by $\hat e(1-\hat e)$, so its variance
is largest at extreme propensities even after clipping, and on `jobs_synth` (earnings-scale outcomes)
one inflated weight makes PEHE catastrophic. The fix at the next rung is to put an outcome model back in
so the predictable part of $Y$ is subtracted before weighting.

```python
class CATEEstimator(BaseCATEEstimator):
    """IPW-based CATE estimator with propensity score weighting.

    1. Estimate propensity score e(X) = P(T=1|X) via logistic regression.
    2. Construct IPW pseudo-outcomes: Y_ipw = T*Y/e(X) - (1-T)*Y/(1-e(X)).
    3. Fit a regression model on X -> Y_ipw for CATE estimation.

    Clips propensity scores to [0.05, 0.95] for stability.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._prop_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )
        self._outcome_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed + 1,
        )

    def fit(self, X, T, Y):
        # Estimate propensity scores
        self._prop_model.fit(X, T)
        e_hat = self._prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # IPW pseudo-outcomes
        pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)

        # Fit outcome model on pseudo-outcomes
        self._outcome_model.fit(X, pseudo_outcome)
        return self

    def predict(self, X):
        return self._outcome_model.predict(X)
```
