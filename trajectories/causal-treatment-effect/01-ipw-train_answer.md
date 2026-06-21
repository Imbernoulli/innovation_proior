The scaffold hands me a placeholder S-learner — one `Ridge` on $[X\mid T]$, read off by toggling the treatment flag — and on these DGPs that floor is honest about its own poverty. The model is linear where the data-generating processes are built from sines, ReLUs, quadratics, and confounder-shared interactions; worse, it treats the treatment column as one regularized coefficient among $p+1$, so the single number it fits for $T$ is the *average* shift and its $\hat\tau(x)$ comes out essentially flat. It will recover the ATE roughly and the heterogeneity not at all. Rather than reach immediately for the strongest machinery I know, I want to climb from the weakest *idea*, and the weakest honest idea is to adjust through the side of the problem the regression placeholder completely ignores: the assignment mechanism.

What actually contaminates the naive treated-minus-control gap tells me what object to build. I want $E[Y(1)-Y(0)]$, a contrast between something I see and something I never see for the same unit, but the reflex comparison $E[Y\mid T=1]-E[Y\mid T=0]$ is not the effect, because treatment was assigned out in the world and the units who got treated differ systematically in $X$ from those who did not. In every one of these DGPs the propensity $e(x)=P(T=1\mid X=x)$ is an explicit nonlinear function of the very covariates that drive the response — in `ihdp_synth` the logit even carries $0.25\,x_3 x_5$, a term shared with $\tau$ — so part of the raw gap is the covariate difference between arms, not the treatment.

I propose to estimate $\tau(x)$ by *inverse-propensity weighting*: collapse the confounding into the propensity and invert it, with no outcome model anywhere. The pivot is that $e(x)$ is a *balancing score*, and the coarsest one. Among units sharing the same propensity, treatment is balanced with respect to $X$: since $P(T=1\mid X)=e(X)$ depends on $X$ only through $e(X)$ itself, $X\perp T\mid e(X)$, so conditioning on this single scalar is enough to break the confounding no matter how high-dimensional $X$ is. That is the lever the regression placeholder leaves on the table — the whole high-dimensional confounding problem collapses into one number per unit.

How to *use* $e(X)$ is where the construction is forced rather than chosen. The coarse routes — stratify into propensity bins and difference within each, or match each treated unit to a control of similar score — are blunt, piecewise-constant, and hand me a number rather than the smooth $\tau(x)$ that PEHE demands. The clean route comes from reading the confounding as a sampling problem: the treated group is a biased sample of the population, a unit with covariates $x$ included with probability $e(x)$, so high-$e$ units are over-represented and low-$e$ under-represented. Horvitz and Thompson (1952) settled exactly this for survey sampling. To estimate a population total from a sample drawn with unequal inclusion probabilities $P(u_i)$, the only unbiased linear estimator $\hat S=\sum_{i\in\text{sample}}\beta_i x_i$ must have $P(u_i)\beta_i=1$ term by term — taking the expectation over which elements land in the sample gives $\sum_i P(u_i)\beta_i X_i$, and for this to equal $S$ for *every* configuration of the unknown values forces $\beta_i=1/P(u_i)$. There is no freedom: each sampled element is up-weighted by the reciprocal of its inclusion probability, standing in for the similar ones that were not sampled. Inverse-probability weighting is not a heuristic here; it is forced by unbiasedness.

The carry-over is exact, and I check the cancellation lands without a stray factor. Weighting each treated unit by $1/e(X)$, consider $E[\,T\,Y/e(X)\mid X\,]$; the factor $T$ selects treated units where $Y=Y(1)$, so it is $(1/e(X))\,E[T\,Y(1)\mid X]$, and unconfoundedness gives $E[T\,Y(1)\mid X]=E[T\mid X]\,E[Y(1)\mid X]=e(X)\mu_1(X)$, so the $e(X)$ cancels and $E[T\,Y/e(X)\mid X]=\mu_1(X)$. A control unit is "included" with probability $1-e(X)$, so weighting controls by $1/(1-e(X))$ recovers $\mu_0$, and

$$\tau=E\!\left[\frac{T\,Y}{e(X)}-\frac{(1-T)\,Y}{1-e(X)}\right].$$

The over-representation of high-$e$ units is undone precisely by dividing by $e(X)$. The whole burden has moved onto one scalar I model as a probability, rather than onto two high-dimensional response surfaces.

The same identity pays off a second time, pointwise, which is what lets me deliver a *function* without binning anything. The conditional computation above never integrated out $X$, so the per-unit pseudo-outcome

$$\psi_i=\frac{T_i Y_i}{e(X_i)}-\frac{(1-T_i)Y_i}{1-e(X_i)}$$

satisfies $E[\psi\mid X=x]=\mu_1(x)-\mu_0(x)=\tau(x)$ — a noisy but *conditionally-unbiased label* for $\tau(x)$. So I compute $\psi_i$ for every unit from the data and $\hat e$, then regress $\psi$ on $X$ with a flexible regressor: the regression's conditional-mean fit estimates $E[\psi\mid X]=\tau(x)$, and its smoothing tames the per-unit noise. The unobservable per-unit effect has become an ordinary regression with an observable, pointwise-unbiased target, and the ATE falls out as the sample mean of $\hat\tau(x)$.

Two design choices carry weight. First, $e(x)$ is not handed to me, so I estimate it — predicting binary $T$ from $X$ is a plain classification problem. Rosenbaum and Rubin's (1983) default is logistic, but these propensities are built from products and quadratics, so I use a `GradientBoostingClassifier` with `predict_proba` (the calibrated probability, not the hard label); the harness's `LogisticRegression` would underfit the same nonlinearity the placeholder Ridge does. Second, the variance danger that makes raw IPW fragile is already visible in the Horvitz-Thompson variance, whose diagonal carries a $1/P(u_i)$ factor that blows up as the inclusion probability nears zero. Here a treated unit with tiny $\hat e$ — one that "should almost never have been treated" but was — gets weight $1/\hat e$, and a single such unit swings the whole average; symmetrically as $\hat e\to 1$ on the control side. The DGPs clip the true $e$ to $[0.05,0.95]$, but a finite-sample $\hat e$ can still stray to a whisker of the boundary, so I clip $\hat e$ into $[0.05,0.95]$ before inverting, capping every weight at $20$. This trades a sliver of bias for a large variance reduction, and the trade is sensible because the extreme-$\hat e$ units are exactly where the data are thinnest and least trustworthy; the $0.05$ floor matching the DGP's own overlap clip is the principled choice.

The pieces line up against the worries: confounding is handled by the $e(X)$ cancellation that makes $\psi$ conditionally unbiased under unconfoundedness, with no response surface to get right; heterogeneity by regressing $\psi$ on $X$; exploding weights by the clip. The one genuine vulnerability is the single point of failure — everything rests on $\hat e$ being reasonable, and there is no second model to catch it if the propensity is badly wrong. And even when $\hat e$ is decent, $\psi$ divides by $\hat e(1-\hat e)$, so its variance is largest exactly at extreme propensities even after clipping. I expect this to be the weakest rung, worst PEHE on every dataset, and *catastrophic* on `jobs_synth`: its outcomes live on the earnings scale, so a single inflated weight turns into a pseudo-outcome in the tens of thousands and the regression that should smooth $\psi$ instead chases the spikes, putting its PEHE in the thousands. That diagnosis — the signal is unbiased but the variance is uncontrolled because there is no outcome model to subtract off the predictable part of $Y$ before weighting — is exactly what points the next rung toward modeling the outcome surfaces directly.

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
