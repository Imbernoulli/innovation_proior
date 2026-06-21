IPW's numbers confirmed its diagnosis in the most quantitative way possible: on `jobs_synth` its PEHE came back at **5183.8** with an ATE error of **843.3**, exactly the weighting-variance blow-up I had predicted — $\psi=T\,Y/\hat e-(1-T)\,Y/(1-\hat e)$ divides an earnings-scale $Y$ by an $\hat e$ that, even clipped to $0.05$, lets a single near-clip unit inject a pseudo-outcome of order tens of thousands. On the order-one datasets the damage was bounded but still the worst on the board (`ihdp_synth` **2.12**, `acic_synth` **1.30**), because the $1/\hat e(1-\hat e)$ factor inflates the variance of $\psi$ everywhere, and there was no outcome model subtracting off the predictable bulk of $Y$ before the weighting hit. A conditionally-unbiased signal is worthless if its variance is uncontrolled. The obvious next move is to flip the burden: stop reweighting raw outcomes, model the outcome surfaces directly, and difference them.

The identification is the same one IPW used, exploited from the other side. Under unconfoundedness, conditioning on $X$ makes $T$ independent of the potential outcomes, so the observable within-arm regression equals the counterfactual response surface: $E[Y\mid X=x,T=w]=E[Y(w)\mid X=x]=\mu_w(x)$. Then $\tau(x)=\mu_1(x)-\mu_0(x)$ is a difference of two ordinary conditional expectations, each estimable by a plain regression on its own arm's observed outcomes — no counterfactual ever appears inside either regression. This is the polar opposite of IPW: IPW touched no response surface and paid for it in variance; this route touches nothing but response surfaces.

I propose the *T-learner*: fit one regressor on the control rows $\{(X_i,Y_i):T_i=0\}$ to get $\hat\mu_0$, fit a separate regressor on the treated rows $\{(X_i,Y_i):T_i=1\}$ to get $\hat\mu_1$, and set $\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)$. Two models, hence the name. The causal content lives entirely in the act of splitting by arm and subtracting; the estimation of each surface is handed to whatever off-the-shelf supervised learner I trust. This directly addresses the failure mode I just watched in another respect too: the treatment can never be accidentally dropped, because *which model I use is the treatment*. $\hat\mu_1$ lives entirely on treated data and $\hat\mu_0$ entirely on control data, so unlike a pooled model that hands $T$ to a flexible learner as one feature among $p$ strong covariates — where a greedy tree on `acic_synth` ($p=50$) can descend to its leaves never once splitting on the weak $T$ feature, making the implied effect exactly zero — the two-arm split forces the treatment to matter structurally. That is the right reflex right after IPW: where IPW had to force the treatment to matter through the weighting, the two-model split makes it matter through the architecture.

But the same isolation that is the strength is the weakness, and I want it characterized precisely. Each of the two regressions is tuned to fit *its own response surface as well as possible*; neither has any incentive to make the *difference* good, and they never see each other. The regime that hurts is when $\mu_0$ and $\mu_1$ each carry a big, complex shared baseline dependence on $X$ but the treatment shifts the outcome by a comparatively simple amount, so $\tau=\mu_1-\mu_0$ is simpler than either surface. These DGPs are built exactly this way: in `ihdp_synth` the base surface carries $\exp(0.8x_0+0.5x_1)$ and products while $\tau$ is a lower-order polynomial-plus-sine; in `jobs_synth` $\mu_0$ runs to thousands of dollars with several interactions while $\tau$ is a smoother earnings increment. The two-model route spends all its modeling capacity chasing the complicated baseline in each arm separately, fits each with its own independent error, then subtracts two complicated, independently-noisy fits to recover a quantity that was simpler all along. The shared structure that would have cancelled in the difference is estimated twice, independently, and two uncorrelated errors corrupt the cancellation — so even where the true $\tau$ is smooth, $\hat\mu_1-\hat\mu_0$ can show a spurious bump. It gets worse under imbalance: wherever the treated arm is locally thin — the low-$\hat e$ regions where IPW's weights exploded — $\hat\mu_1$ is fit on few points and its local error dominates the difference. The two methods fail in the *same regions* for opposite reasons, IPW because $1/\hat e$ blows up the weight, the T-learner because the treated arm is data-starved.

There is a unifying frame worth naming because it places this rung against its neighbors. Ask where a tree ensemble is allowed to split on $T$: in a pooled single model the ordinary loss decides, so the split can happen anywhere or nowhere; in the two-model route the split on $T$ is *forced at the root* — separating treated from control is the first thing that happens, and the two subtrees are exactly the two arm-models. So the T-learner is the pooled model with the constraint "split on $T$ first," the member of the family that maximally privileges the treatment. That makes it the right move right after IPW's collapse, but it pays the no-pooling, response-surface-rate price, and with no orthogonalization its error is *first-order* in the response-surface error. Concretely, writing $\hat\tau-\tau=(\hat\mu_1-\mu_1)-(\hat\mu_0-\mu_0)$ and using $(a-b)^2\le 2a^2+2b^2$, the effect error is bounded by twice the sum of the two arm errors, and a change-of-measure argument (overlap supplying the bounded $e_{\max}/e_{\min}$ factor) lands $\mathrm{EMSE}(\hat\tau)=O(n^{-a_\mu}+m^{-a_\mu})$, where $n,m$ are the treated and control counts and $a_\mu$ is the *response-surface* minimax rate — never the (possibly faster) effect rate $a_\tau$. On DGPs where $\tau$ is simpler than $\mu_w$, that is exactly the wrong rate to be paying, and there is no knob inside the recipe to fix it because the arms never talk. That is the limitation this rung hands forward.

For the scaffold edit, both arms use 200 trees at depth 4. Depth $>1$ is load-bearing so each tree can capture the low-order interactions in the response surfaces; the leaf floor of 20 plus row subsampling (`subsample=0.8`) regularize each surface against leaf-level noise, which matters doubly because the final quantity is a *difference* of two predictions. The clean statement of the T-learner would clone a single base learner for each arm, but here I instantiate *two distinct* `GradientBoostingRegressor` objects seeded `seed` and `seed+1` from `os.environ["SEED"]`: functionally the same two-arm difference, but the per-arm seeds make the two arm-models independently reproducible rather than identically-seeded clones, which is the honest thing when the arms have different sizes and the stochastic subsampling should not be correlated across them. `fit` masks the rows by arm and fits each model; `predict` returns $\hat\mu_1(X)-\hat\mu_0(X)$.

I expect removing the $1/\hat e$ division to remove the catastrophe — `jobs_synth` PEHE should drop from **5183.8** by more than an order of magnitude into the hundreds, and its ATE error from **843.3** to the tens, because the estimate is now a difference of bounded regression predictions. `ihdp_synth` PEHE should fall toward order one. The *smallest* relative gain should come on `acic_synth`, the hardest case, where the T-learner trades IPW's weighting variance for the cost of fitting two hard 50-dimensional surfaces in isolation; it should beat IPW's **1.30** but may itself be among the weaker surface methods there, exactly because the no-pooling, response-surface-rate ceiling bites hardest when the surfaces are complex and the arms locally thin. That is the expectation I most want the next rung to falsify in its favor: the diagnosis I hand forward is not "model the outcomes" — the T-learner already does — but "the two surfaces are estimated in isolation, at the response rate, with first-order sensitivity to their error; tie them together and orthogonalize."

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
