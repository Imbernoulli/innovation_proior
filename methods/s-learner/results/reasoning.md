Let me start from what I actually have in my hands and what I actually want. I have a pile of units, each with a covariate vector x, a binary flag w telling me whether it got the treatment, and one outcome y. I want, for a new unit at covariate value x, the number tau(x) = E[Y(1) - Y(0) | X = x]: how much better off this kind of unit is under treatment than under control. The cruel part is that for every single unit in my data I see exactly one of the two outcomes — y = Y(w) for the w it actually got — and the other one is gone forever. So I can never form the per-unit difference Y_i(1) - Y_i(0) for even one row of my data. I'm trying to estimate a difference of two things when, for any given unit, I only ever observe one of them.

Before I even reach for a model, let me get clear on what is and isn't recoverable, because if I aim at the wrong target I'll chase noise forever. The unit-level effect D_i = Y_i(1) - Y_i(0) — could I ever pin that down? Picture the simplest world: x is uniform on [0,1], treatment is a fair coin independent of x, and the control outcome is Rademacher, Y(0) = +1 or -1 with equal probability, independent of everything. Now write down two different worlds for the treated outcome. In world one, Y(1) = Y(0) exactly, so D_i = 0 for everyone. In world two, Y(1) = -Y(0), so D_i is +2 or -2 for everyone, never zero. In both worlds, what do I actually observe? A unit shows me either its Y(0) or its Y(1); marginally each is ±1 with probability one-half, and w is an independent coin — so the joint distribution of (y, x, w) is identical across the two worlds. The observed data cannot tell them apart, yet D_i is uniformly zero in one and uniformly ±2 in the other. So no estimator built on observed data can be consistent for D_i. The individual effect is simply not identifiable. That settles the target: I must aim not at D_i but at its conditional mean, tau(x) = E[D | X = x], which in this example equals zero in both worlds — consistent, recoverable.

Is giving up on D_i and settling for tau(x) actually a loss? Let me check by writing the mean-squared error of a prediction tau_hat(x) for a fresh unit at this covariate value; once I condition on the fitted estimator, the remaining randomness is the unit's potential-outcome residual:

  E[(D_i - tau_hat(x))^2 | X_i = x, tau_hat] = E[(D_i - tau(x) + tau(x) - tau_hat(x))^2 | X_i = x, tau_hat].

Expand the square. The cross term is 2·(tau(x) - tau_hat(x))·E[D_i - tau(x) | X_i = x]; since tau(x) is exactly E[D_i | X_i = x], the last expectation is zero. I'm left with

  E[(D_i - tau_hat(x))^2 | X_i = x, tau_hat] = E[(D_i - tau(x))^2 | X_i = x] + (tau(x) - tau_hat(x))^2.

The first piece is the variance of the individual effect around its conditional mean — irreducible, I can do nothing about it. The second is exactly the squared error of estimating tau(x). So the estimator that minimizes the MSE for the unobservable individual effect is the very same estimator that minimizes the MSE for the CATE. Targeting tau(x) costs me nothing I could have had. Good. My metric is the expected mean-squared error of the effect surface, EMSE = E[(tau(X) - tau_hat(X))^2], and that's what I'll keep my eye on.

Now, can I even write tau(x) in terms of things I can regress on observed data? Define the two response surfaces mu_0(x) = E[Y(0) | X = x] and mu_1(x) = E[Y(1) | X = x]; then tau(x) = mu_1(x) - mu_0(x) by linearity. These are still counterfactual — they're conditional means of potential outcomes. The bridge I need is an assumption about how treatment got assigned. If treatment assignment, conditional on the covariates, is unrelated to the potential outcomes — ignorability, (eps(0), eps(1)) ⟂ W | X, all the confounders are in x — then conditioning on W = w doesn't distort the potential-outcome distribution at fixed x, so

  E[Y^obs | X = x, W = w] = E[Y(w) | X = x, W = w] = E[Y(w) | X = x] = mu_w(x).

That's the whole game in one line: under ignorability, the *observable* regression of the outcome on covariates within each treatment value equals the *counterfactual* response surface. I also need overlap, 0 < e_min < e(x) < e_max < 1, so that both arms actually appear at every x and these conditional means are estimable everywhere. With those two assumptions, tau(x) = E[Y^obs | x, W=1] - E[Y^obs | x, W=0] — a difference of two ordinary conditional expectations, each of which is just a regression. Suddenly this is a problem I have a hundred tools for.

So which tool? The thing I most want to avoid is inventing a bespoke causal regularizer from scratch. I have random forests, gradient boosting, BART, neural nets — mature, battle-tested predictors of E[Y | features] that already handle high dimensions, nonlinearity, and overfitting with sensible built-in regularization, and that I trust. They were built to predict outcomes, not effects; there's no counterfactual anywhere in their loss. But the derivation just told me the counterfactual surfaces ARE ordinary conditional expectations. So the meta-move is: don't build a new learner, build a recipe that turns the trusted learner into a CATE estimator. Decompose the effect-estimation problem into sub-regression problems that any black-box supervised algorithm can solve, then recombine. The base learner stays a black box; I only choose how to slice the data and how to combine the fits.

The most direct slicing reads straight off tau(x) = mu_1(x) - mu_0(x): estimate each surface on its own arm. Take all the control rows, regress y on x there to get mu_hat_0; take all the treated rows, regress y on x there to get mu_hat_1; then tau_hat(x) = mu_hat_1(x) - mu_hat_0(x). Two fits, possibly two different learners. This is clean, and it has a comforting property: the treatment can never be accidentally dropped, because which model I use *is* the treatment — mu_hat_1 lives entirely on treated data and mu_hat_0 entirely on control data. Let me also check its error. Using (a - b)^2 <= 2a^2 + 2b^2 on tau_hat - tau = (mu_hat_1 - mu_1) - (mu_hat_0 - mu_0),

  EMSE(tau_hat) <= 2·E[(mu_hat_1(X) - mu_1(X))^2] + 2·E[(mu_hat_0(X) - mu_0(X))^2],

so the effect error is bounded by twice the sum of the two response-fit errors. If each response is estimable at the minimax rate of its family, this inherits that rate. Fine so far.

But stare at what this two-model recipe is actually doing, because something is being wasted. Each of the two regressions is tuned, regularized, cross-validated to fit *its own response surface as well as possible* — mu_0 on its own, mu_1 on its own. Neither model has any incentive to make the *difference* good; they never see each other. And here's the regime that hurts: suppose the two response surfaces mu_0 and mu_1 are each individually wiggly and complicated — they share a big, complex baseline dependence on x — but the treatment shifts the outcome by a simple, nearly constant amount, so tau = mu_1 - mu_0 is simple. The two-model recipe spends all its modeling capacity chasing the complicated baseline in each arm separately, fits each with its own independent error, and then I subtract two complicated, independently-noisy fits to recover a quantity that was simple all along. The shared structure that would have cancelled in the difference is estimated twice, independently, and the cancellation is corrupted by two uncorrelated errors. And it gets worse when the design is unbalanced: control data piles up cheaply from administrative records while treated units are scarce, so n << m. Then mu_hat_1 is fit on a tiny treated sample, its error dominates the bound at rate n^{-a}, and I've thrown away the fact that the *difference* might be far easier to learn than either piece. The two-model recipe simply cannot see that the effect is simpler than the responses, because it never represents the effect at all — only the two responses, separately.

I want a recipe where the model can *share* statistical strength across the two arms and can *decline to model* the treatment dependence where the treatment doesn't matter. So let me not split the data. Pool everything — all N rows, treated and control together — and hand the learner the treatment flag as just one more feature, with no special status, sitting in the feature vector right next to the components of x. Fit a single combined surface

  mu(x, w) := E[Y^obs | X = x, W = w]

on the whole dataset using one base learner. By the ignorability bridge, mu(x, 1) = mu_1(x) and mu(x, 0) = mu_0(x). Then to read off the effect at x, I hold all of x fixed and toggle only the treatment feature, taking the difference:

  tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0).

One model, single, sharing all the data. The "S" is for single. Notice what this buys over the two-model recipe. The single learner now sees both arms at once, so wherever the response depends on x in the same way regardless of treatment — the shared baseline — it models that dependence once, with all N rows of statistical strength, instead of twice with independent errors. And when I toggle w from 0 to 1, the shared baseline that depends only on x is *identical* in both predictions and subtracts out exactly, so the estimated effect is the model's pure response to flipping that one feature. The wasteful double-estimation-then-subtract is gone; the cancellation that I wanted is built into the architecture, because both terms come from the same fitted function evaluated at the same x.

And there's a second thing the single model can do that the two-model recipe structurally cannot: it can choose *not* to use the treatment feature where the treatment is irrelevant. If over some region tau(x) is genuinely zero, a good learner — say a tree ensemble fitting squared error — will simply find that splitting on the treatment flag there reduces the loss not at all, so it won't split on it, and mu_hat(x, 1) - mu_hat(x, 0) comes out to exactly zero. The two-model recipe can't do that: it would fit two separate models that, by sampling noise alone, disagree somewhere, and their difference would be nonzero even where the truth is zero. So in the regime where the effect is small or sparse — which the literature says is common, since the same covariates that drive the baseline outcome cancel in the contrast — the single model is exactly right: it correctly reports zero rather than manufacturing spurious heterogeneity. This is the regime the single-model recipe is best in: simple, near-zero, smooth effects.

There's a unifying picture lurking here that I should pin down, because it tells me precisely what I've traded. Imagine the base learner is a tree ensemble and ask: where in the tree is the model allowed to split on the treatment flag w? In the single-model recipe, the ordinary squared-error criterion decides — the split on w can happen anywhere in any tree, or nowhere at all, wherever it best reduces loss. In the two-model recipe, the split on w is *forced at the very root* — separating treated from control is the first thing that happens, and after that the two subtrees are just the two arm-models; so the two-model recipe is the single model with the constraint "split on w first." And the modified-splitting forest sits at the third extreme, forcing the split on w to be the last thing, right before the leaves. One family, parameterized by where the treatment split is permitted: anywhere (single), first (two-model), last (modified forest). That's a clean way to see what "single model, treatment as a feature" is — it's the *least constrained* member, the one that lets the data decide whether and where the treatment matters.

Now I have to be honest and hunt for where this single-model move breaks, because the very freedom that makes it good — "the learner may decline to use the treatment feature" — is also a knife that can cut the wrong way. The treatment flag is exactly one feature competing against all d components of x for the learner's attention, and every one of my trusted learners is *regularized*: it penalizes complexity, shrinks weak signals, prefers parsimony. So what happens when the treatment's effect on the outcome is real but *weak relative to* how strongly the covariates predict the outcome? The covariates explain most of the variance in y; the treatment flag explains a little; the regularizer, doing its job, downweights or outright ignores the weak feature in favor of the informative ones. Then mu_hat(x, 1) ≈ mu_hat(x, 0), the toggled difference collapses, and tau_hat is biased toward zero. In the extreme, the learner discards the treatment feature entirely and predicts a flat zero effect everywhere. This is not a hypothetical: a random-forest single-model with a hundred thousand trees, on data where the covariates strongly predict the control outcome and the treatment is a comparatively weak predictor, has its trees split on the treatment flag only very rarely — so most trees ignore the treatment and the effect estimate is pulled hard toward zero. The same architecture that *correctly* returns zero when the effect truly is zero will *incorrectly* return near-zero when the effect is real but the regularizer prefers the covariates. That's the failure mode, and it's not a bug I can patch — it's the direct consequence of treating w as one feature among many and letting a complexity-penalizing learner decide its fate. Wall.

So the single-model recipe and the two-model recipe sit at opposite ends of one bias-variance trade. The single model pools data and can shrink the effect to zero — wonderful when the effect is simple or absent, dangerous when the effect is real but the regularizer under-weights the treatment, where it's biased toward zero. The two-model recipe forces the treatment to matter — safe against the bias-to-zero, but it splits the data, can't share strength, can't exploit a simple effect, and overfits the small arm in an unbalanced design. Neither dominates. The right reading is not "one of these is the answer" but "these are two ends of a knob, and which end I want depends on whether the true effect is simpler or more complex than the responses." When the effect is genuinely simple or close to zero — and especially when the treatment assignment has little predictive power for the pooled outcome, so the learner ignores it and correctly returns zero — the single model is the one to reach for. When the effect is complex and there's nothing to be borrowed across arms, the two-model recipe is better. That dichotomy is the honest characterization of the single-model recipe; I won't pretend it's universally best.

Let me make the response-surface ceiling quantitative, because I want to see exactly which smoothness governs it and where overlap enters. The clean bound is for the separate-arm response estimator, where each fitted response has its own sample size; it is the conservative rate warning for this whole response-estimate-then-difference route. Set up the minimax frame: a family of superpopulations P where, conditional on each arm, the response is estimable at some rate. Concretely, say the control-arm distribution of (X, Y(0)) given W=0 lies in a class with minimax rate a_mu, and likewise the treated arm, and write n for the number of treated units and m for the number of control units. From tau_hat(x) - tau(x) = (mu_hat_1(x) - mu_1(x)) - (mu_hat_0(x) - mu_0(x)) and (a-b)^2 <= 2a^2 + 2b^2,

  EMSE <= 2·E[(mu_hat_1(X) - mu_1(X))^2] + 2·E[(mu_hat_0(X) - mu_0(X))^2]
        =: 2A + 2B.

Look at A. The subtlety is that mu_hat_1 is trained only on the *treated* rows, so its error is naturally an expectation over the treated-arm covariate distribution, the law of X given W=1 — but the EMSE I want is over the *marginal* law of X (a fresh test point, drawn from the population, not from the treated subpopulation). I need to change measure from "X given W=1" to "X marginal," and overlap is exactly what makes that change of measure bounded. For any nonnegative g,

  E[g(X) | W = 1] = E[g(X) e(X)] / E[W] = E[g(X) e(X)] / E[e(X)].

The numerator is at least e_min·E[g(X)] and at most e_max·E[g(X)] since e_min < e(x) < e_max; the denominator E[e(X)] = E[W] lies in (e_min, e_max). So

  (e_min/e_max)·E[g(X)] <= E[g(X) | W = 1] <= (e_max/e_min)·E[g(X)],

and symmetrically, dividing through the control arm with 1 - e(x) in place of e(x),

  ((1-e_max)/(1-e_min))·E[g(X)] <= E[g(X) | W = 0] <= ((1-e_min)/(1-e_max))·E[g(X)].

Now condition A on the treatment vector. Given W, the treated rows are i.i.d. from the law of (X, Y) given W=1, so the inner expectation of (mu_hat_1(X) - mu_1(X))^2 over a fresh marginal X can be flipped to an expectation over a treated-distributed test point at the cost of the factor e_max/e_min from the inequality above, and that treated-distributed error is exactly what the minimax rate controls: it is at most C·n^{-a_mu} because mu_hat_1 is fit on the n treated units in their own family. So

  A <= (e_max/e_min)·C·n^{-a_mu},

and by the symmetric control-arm bound

  B <= ((1-e_min)/(1-e_max))·C·m^{-a_mu}.

Putting them together,

  EMSE <= 2C·(e_max/e_min)·n^{-a_mu}
        + 2C·((1-e_min)/(1-e_max))·m^{-a_mu}
        = O(n^{-a_mu} + m^{-a_mu}).

There's the rate, and there's the limitation written in plain algebra: the exponent is a_mu, the smoothness of the *response surfaces*. Not a_tau, the smoothness of the *effect*. A recipe that gets the effect only by estimating response values and subtracting pays the response rate even when the effect is far smoother (a_tau > a_mu); the effect's own structure never directly enters the regression target. When n << m in the arm-split bound, the n^{-a_mu} treated-arm term dominates, and even with m enormous the response-route ceiling is still the response rate in n, never the faster effect rate a_tau. So the very regularity the literature keeps reporting — that tau is often simpler than mu_0 and mu_1 — is exactly the thing this family cannot cash in directly. That's the precise statement of the gap, and I'll leave it sitting there as the thing a better recipe would have to beat; the single-model recipe I'm building is the honest, simplest pooled member of this family, best when the effect really is simple.

So let me commit to the single-model recipe as a clean, complete CATE estimator and write it down, because for the simple/near-zero/smooth-effect regime it's the right tool, it reuses any black-box learner, and it pools all the data. The procedure is exactly three moves: build a feature matrix by appending the treatment flag as one more column to the covariates; fit one regressor on (covariates, flag) -> outcome over the whole dataset; predict at a new x by running it twice, once with the flag pinned to 1 and once pinned to 0, and subtracting. That's it.

What base learner should the single slot hold? I want a flexible nonparametric regressor that can bend the response surface differently in the treated and control regions when the flag is informative, yet shrink the flag's influence when it is not — and that handles tabular covariates with sensible built-in regularization and little hand-tuning. Gradient-boosted trees fit this: an additive ensemble of shallow trees fit to the squared-error residual. The shallow depth (say four) lets each tree capture low-order interactions — including interactions *between the treatment flag and the covariates*, which is precisely the mechanism by which the single model represents heterogeneity: a split on a covariate beneath a split on the flag is exactly "the effect varies with that covariate." A modest learning rate with many trees trades a little fitting speed for a smoother, lower-variance surface; subsampling rows each round adds stochastic regularization; and a floor on leaf size stops any single leaf from chasing a handful of points, which matters because the effect is a *difference* of two predictions and is therefore especially sensitive to leaf-level noise. None of these knobs is causal; they're just the standard recipe for a low-variance nonparametric tabular regressor, and that's the point — the causal content lives entirely in how I slice and toggle, not in the learner.

Let me write the estimator into the fit/predict slot. In fit, I stack the treatment column onto X and fit the single base learner on the pooled data:

```python
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class CATEEstimator:
    """S-learner: a single ('S') outcome model on the pooled data with the treatment
    flag included as an ordinary feature, then CATE read off by toggling that flag.

    Under ignorability and overlap, mu(x, w) = E[Y^obs | X=x, W=w] equals the
    counterfactual response surface mu_w(x), so
        tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0)
    is a CATE estimator. Any supervised regressor can be the base learner; a flexible,
    regularized tree ensemble is the natural choice for tabular covariates.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        # one base learner, shared across both arms (the "single" model)
        self._model = GradientBoostingRegressor(
            n_estimators=200,      # many shallow trees -> smooth, low-variance surface
            max_depth=4,           # depth >1 lets flag x covariate interactions form (heterogeneity)
            learning_rate=0.1,     # modest shrinkage per tree
            min_samples_leaf=20,   # leaf floor: a difference of predictions is leaf-noise sensitive
            subsample=0.8,         # stochastic row subsampling -> extra regularization
            random_state=self._seed,
        )

    def fit(self, X, T, Y):
        # treat the binary treatment T as one more feature, appended to X, on ALL rows
        XT = np.column_stack([X, T.reshape(-1, 1)])   # pooled design: [covariates | treatment flag]
        self._model.fit(XT, Y)                         # fit the single surface mu(x, w) = E[Y|x,w]
        return self

    def predict(self, X):
        n = X.shape[0]
        # hold every covariate fixed; toggle only the treatment flag
        X1 = np.column_stack([X, np.ones((n, 1))])     # everyone "treated":  mu_hat(x, 1)
        X0 = np.column_stack([X, np.zeros((n, 1))])    # everyone "control":  mu_hat(x, 0)
        return self._model.predict(X1) - self._model.predict(X0)   # tau_hat(x) = mu_hat(x,1) - mu_hat(x,0)
```

Let me trace the whole causal chain back. I wanted the effect surface tau(x) but could only ever see one potential outcome per unit, and a quick two-world construction showed the individual effect is unidentifiable, so I retargeted the conditional mean tau(x) — and an MSE decomposition proved that costs nothing, since the best ITE estimator is the best CATE estimator. Ignorability turned the counterfactual response surfaces into ordinary observable regressions, E[Y^obs | x, w] = mu_w(x), with overlap making both arms estimable everywhere; that let me reuse any trusted black-box regressor instead of inventing a causal one. The most direct recipe — two separate arm-models, subtracted — wastes statistical strength: it fits each complicated response independently and corrupts the (often simple) difference with two uncorrelated errors, and it overfits the small arm in an unbalanced design. So instead I pooled all the data into one model with the treatment as an ordinary feature and read the effect off by toggling that feature, which shares strength across arms, cancels the shared baseline exactly, and can correctly return zero where the effect is zero. The price, derived not patched, is that a regularized learner sees the treatment as one weak feature among many and can shrink it toward zero — biasing the effect to zero exactly when the true effect is real but small relative to the covariates' pull. A minimax change-of-measure argument for the arm-split response estimator, with overlap supplying the bounded e_max/e_min factor, pins the response-route ceiling at O(n^{-a_mu} + m^{-a_mu}) — the *response* smoothness, never the (often greater) *effect* smoothness. Within that response-estimate-then-toggle family the single-model recipe is the least-constrained, simplest pooled member, ideal for simple or near-zero effects, and it drops into the fit/predict slot as: append the flag, fit one regressor on the pool, predict twice and subtract.
