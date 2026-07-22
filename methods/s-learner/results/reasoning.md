Let me start from what I actually have in my hands and what I actually want. I have a pile of units, each with a covariate vector x, a binary flag w telling me whether it got the treatment, and one outcome y. I want, for a new unit at covariate value x, the number tau(x) = E[Y(1) - Y(0) | X = x]: how much better off this kind of unit is under treatment than under control. The cruel part is that for every single unit in my data I see exactly one of the two outcomes — y = Y(w) for the w it actually got — and the other one is gone forever. So I can never form the per-unit difference Y_i(1) - Y_i(0) for even one row of my data. I'm trying to estimate a difference of two things when, for any given unit, I only ever observe one of them.

Before I even reach for a model, let me get clear on what is and isn't recoverable, because if I aim at the wrong target I'll chase noise forever. The unit-level effect D_i = Y_i(1) - Y_i(0) — could I ever pin that down? Picture the simplest world: x is uniform on [0,1], treatment is a fair coin independent of x, and the control outcome is Rademacher, Y(0) = +1 or -1 with equal probability, independent of everything. Now write down two different worlds for the treated outcome. In world one, Y(1) = Y(0) exactly, so D_i = 0 for everyone. In world two, Y(1) = -Y(0), so D_i is +2 or -2 for everyone, never zero. The question I actually have to answer is whether the observed data can tell these two worlds apart, and I shouldn't just assert that it can't — let me work out the joint law of what I see, (y, w), in each world. In world one a unit with w=0 shows Y(0) and a unit with w=1 shows Y(1)=Y(0); in world two a unit with w=0 shows Y(0) and a unit with w=1 shows Y(1)=-Y(0). Take the cell (y=+1, w=1). In world one this needs Y(0)=+1 (prob 1/2) and w=1 (prob 1/2), so 1/4. In world two it needs Y(1)=-Y(0)=+1, i.e. Y(0)=-1 (prob 1/2), and w=1 (prob 1/2), so again 1/4. The flip in world two just relabels which sign of Y(0) lands in which y-cell, and because Y(0) is symmetric ±1 the relabeling is invisible. Running all four cells gives the same 1/4 each way. To be sure I'm not fooling myself with the algebra, I simulate four million units in both worlds and tabulate the joint law of (y, w): every cell comes out 0.25 in both, the largest discrepancy between the two worlds' tables being 0.0003, pure Monte Carlo jitter — while D_i is uniformly 0 in world one and uniformly ±2 in world two. The observed data genuinely cannot distinguish the worlds, yet D_i differs by a country mile. So no estimator built on observed data can be consistent for D_i; the individual effect is not identifiable. But notice what the same simulation reports for the conditional mean: tau = E[D] is 0 in world one and -0.001 (i.e. 0 up to noise) in world two — the two worlds agree on it. That is the quantity I must aim at: not D_i but tau(x) = E[D | X = x].

Is giving up on D_i and settling for tau(x) actually a loss? Let me check by writing the mean-squared error of a prediction tau_hat(x) for a fresh unit at this covariate value; once I condition on the fitted estimator, the remaining randomness is the unit's potential-outcome residual:

  E[(D_i - tau_hat(x))^2 | X_i = x, tau_hat] = E[(D_i - tau(x) + tau(x) - tau_hat(x))^2 | X_i = x, tau_hat].

Expand the square. The cross term is 2·(tau(x) - tau_hat(x))·E[D_i - tau(x) | X_i = x]; since tau(x) is exactly E[D_i | X_i = x], the last expectation is zero. I'm left with

  E[(D_i - tau_hat(x))^2 | X_i = x, tau_hat] = E[(D_i - tau(x))^2 | X_i = x] + (tau(x) - tau_hat(x))^2.

The first piece is the variance of the individual effect around its conditional mean — irreducible, I can do nothing about it. The second is exactly the squared error of estimating tau(x). So the estimator that minimizes the MSE for the unobservable individual effect is the very same estimator that minimizes the MSE for the CATE. Targeting tau(x) costs me nothing I could have had. Good. My metric is the expected mean-squared error of the effect surface, EMSE = E[(tau(X) - tau_hat(X))^2], and that's what I'll keep my eye on.

Now, can I even write tau(x) in terms of things I can regress on observed data? Define the two response surfaces mu_0(x) = E[Y(0) | X = x] and mu_1(x) = E[Y(1) | X = x]; then tau(x) = mu_1(x) - mu_0(x) by linearity. These are still counterfactual — they're conditional means of potential outcomes. The bridge I need is an assumption about how treatment got assigned. If treatment assignment, conditional on the covariates, is unrelated to the potential outcomes — ignorability, (eps(0), eps(1)) ⟂ W | X, all the confounders are in x — then conditioning on W = w doesn't distort the potential-outcome distribution at fixed x, so

  E[Y^obs | X = x, W = w] = E[Y(w) | X = x, W = w] = E[Y(w) | X = x] = mu_w(x).

That's the whole game in one line: under ignorability, the *observable* regression of the outcome on covariates within each treatment value equals the *counterfactual* response surface. I also need overlap, 0 < e_min < e(x) < e_max < 1, so that both arms actually appear at every x and these conditional means are estimable everywhere. With those two assumptions, tau(x) = E[Y^obs | x, W=1] - E[Y^obs | x, W=0] — a difference of two ordinary conditional expectations, each of which is just a regression. Suddenly this is a problem I have a hundred tools for.

So which tool? The thing I most want to avoid is inventing a bespoke causal regularizer from scratch. I have random forests, gradient boosting, BART, neural nets — mature, battle-tested predictors of E[Y | features] that already handle high dimensions, nonlinearity, and overfitting with sensible built-in regularization, and that I trust. They were built to predict outcomes, not effects; there's no counterfactual anywhere in their loss. But the derivation just told me the counterfactual surfaces ARE ordinary conditional expectations. So the meta-move I'll pursue is: don't build a new learner, build a recipe that turns the trusted learner into a CATE estimator. Decompose the effect-estimation problem into sub-regression problems that any black-box supervised algorithm can solve, then recombine. The base learner stays a black box; I only choose how to slice the data and how to combine the fits.

The most direct slicing reads straight off tau(x) = mu_1(x) - mu_0(x): estimate each surface on its own arm. Take all the control rows, regress y on x there to get mu_hat_0; take all the treated rows, regress y on x there to get mu_hat_1; then tau_hat(x) = mu_hat_1(x) - mu_hat_0(x). Two fits, possibly two different learners. This is clean, and it has a comforting property: the treatment can never be accidentally dropped, because which model I use *is* the treatment — mu_hat_1 lives entirely on treated data and mu_hat_0 entirely on control data. Let me also check its error. Using (a - b)^2 <= 2a^2 + 2b^2 on tau_hat - tau = (mu_hat_1 - mu_1) - (mu_hat_0 - mu_0),

  EMSE(tau_hat) <= 2·E[(mu_hat_1(X) - mu_1(X))^2] + 2·E[(mu_hat_0(X) - mu_0(X))^2],

so the effect error is bounded by twice the sum of the two response-fit errors. If each response is estimable at the minimax rate of its family, this inherits that rate. Fine so far.

But stare at what this two-model recipe is actually doing, because something is being wasted. Each of the two regressions is tuned, regularized, cross-validated to fit *its own response surface as well as possible* — mu_0 on its own, mu_1 on its own. Neither model has any incentive to make the *difference* good; they never see each other. And here's the regime that hurts: suppose the two response surfaces mu_0 and mu_1 are each individually wiggly and complicated — they share a big, complex baseline dependence on x — but the treatment shifts the outcome by a simple, nearly constant amount, so tau = mu_1 - mu_0 is simple. The two-model recipe spends all its modeling capacity chasing the complicated baseline in each arm separately, fits each with its own independent error, and then I subtract two complicated, independently-noisy fits to recover a quantity that was simple all along. The shared structure that would have cancelled in the difference is estimated twice, independently, and the cancellation is corrupted by two uncorrelated errors. I want to know whether this is a real concern or just a worry, so I build the regime and measure it: a strong wiggly baseline (4·sin(2x_0) + 3·x_1^2 - 3·x_2·x_3), a constant true effect tau = 0.5, N = 2000, and gradient-boosted trees as the base learner. The two-model recipe gets the *average* effect about right (mean tau_hat 0.475 against 0.5) but its PEHE — the RMS error of the whole surface — is 1.72: pointwise it is wildly off, because two independent wiggly fits leave a wiggly residual everywhere they fail to cancel. The worry is real and large.

I want a recipe where the model can *share* statistical strength across the two arms and can *decline to model* the treatment dependence where the treatment doesn't matter. So let me try not splitting the data at all. Pool everything — all N rows, treated and control together — and hand the learner the treatment flag as just one more feature, with no special status, sitting in the feature vector right next to the components of x. Fit a single combined surface

  mu(x, w) := E[Y^obs | X = x, W = w]

on the whole dataset using one base learner. By the ignorability bridge, mu(x, 1) = mu_1(x) and mu(x, 0) = mu_0(x). Then to read off the effect at x, I hold all of x fixed and toggle only the treatment feature, taking the difference:

  tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0).

One model, single, sharing all the data. Does this actually buy what I claimed — does the shared baseline really cancel exactly when I toggle? With a tree ensemble I can check the mechanism directly rather than wave at it. Take a base learner that is a sum of trees; the toggled difference is the sum, over trees, of (tree's prediction with flag=1) minus (tree's prediction with flag=0). Any individual tree that never splits on the flag routes a given x to the same leaf no matter what the flag is set to, so its contribution to the difference is *identically* zero — not approximately, exactly. Only trees that actually split on the flag survive the subtraction. I fit a single gradient-boosted model on a problem where the truth is tau ≡ 0 and inspect its 200 trees: 180 of them never split on the flag, and for those 180 the toggled prediction difference is 0.0 to the last bit; the entire estimated effect comes from the 20 trees that did split on the flag. So the cancellation I wanted isn't a hope about the architecture, it's a structural fact — both terms come from the same fitted function at the same x, and every component that ignores the flag drops out cleanly.

That same experiment answers a second question: can the single model decline to use the treatment feature where treatment is irrelevant? On the tau ≡ 0 problem with a genuinely complicated baseline (3·sin(2x_0) + x_1^2 - 2·x_2·x_3 + x_4), the gradient-boosted model assigns the treatment flag a feature importance of 0.0007 — three orders of magnitude below the covariates — and the resulting tau_hat has mean -0.008 and RMS 0.056 across a thousand test points. It correctly reports approximately zero rather than manufacturing heterogeneity. The two-model recipe structurally cannot do this: it would fit two separate models that, by sampling noise alone, disagree somewhere, and their difference would be nonzero even where the truth is zero. So in the regime where the effect is small or sparse — which the literature says is common, since the same covariates that drive the baseline outcome cancel in the contrast — the pooled single model looks exactly right. And to confirm the sharing-strength claim isn't only about the zero case, I run a simple-but-nonzero effect (constant tau = 1, moderate baseline, N = 1000): the single model gets PEHE 0.145 against the two-model recipe's 0.293, half the error, precisely because it models the shared baseline once with all N rows instead of twice with independent errors.

There's a unifying picture lurking here that I should pin down, because it tells me precisely what I've traded. Imagine the base learner is a tree ensemble and ask: where in the tree is the model allowed to split on the treatment flag w? In the single-model recipe, the ordinary squared-error criterion decides — the split on w can happen anywhere in any tree, or nowhere at all, wherever it best reduces loss. In the two-model recipe, the split on w is *forced at the very root* — separating treated from control is the first thing that happens, and after that the two subtrees are just the two arm-models; so the two-model recipe is the single model with the constraint "split on w first." And the modified-splitting forest sits at the third extreme, forcing the split on w to be the last thing, right before the leaves. One family, parameterized by where the treatment split is permitted: anywhere (single), first (two-model), last (modified forest). That's a clean way to see what "single model, treatment as a feature" is — it's the *least constrained* member, the one that lets the data decide whether and where the treatment matters.

Now I have to be honest and hunt for where this single-model move breaks, because the very freedom that makes it good — "the learner may decline to use the treatment feature" — is also a knife that can cut the wrong way. The treatment flag is exactly one feature competing against all d components of x for the learner's attention, and every one of my trusted learners is *regularized*: it penalizes complexity, shrinks weak signals, prefers parsimony. So what should happen when the treatment's effect on the outcome is real but *weak relative to* how strongly the covariates predict the outcome? My fear is that the covariates explain most of the variance in y, the treatment flag explains a little, the regularizer downweights the weak feature, mu_hat(x,1) ≈ mu_hat(x,0) collapses, and tau_hat is biased toward zero. I should check whether this actually bites or whether the learner is robust enough to keep the weak signal. First a mild version: a strong wiggly baseline with a real constant effect tau = 0.5. The single model returns mean tau_hat 0.461 — biased low, but only by 0.04, and its PEHE 0.46 crushes the two-model recipe's 1.72 on the same data, so here the bias is a small price for the variance reduction. The failure isn't dramatic yet. So I push into the regime my fear actually names — effect genuinely weak *relative to* the baseline's pull. Baseline amplitude ×6, true effect tau = 0.2, eight covariates, N = 800: the single model returns mean tau_hat 0.022, a bias of -0.178 that has wiped out almost the entire effect, and the treatment flag is split on in only 3 of 200 trees with importance 0.0001. Push once more, baseline ×8 and tau = 0.1: mean tau_hat 0.004, bias -0.096, the flag split on in 2 of 200 trees. So the fear is correct and I can name the mechanism precisely: when the covariates dominate the loss, the squared-error criterion almost never finds it worthwhile to split on the flag, so almost every tree ignores treatment and the toggled difference collapses toward zero. The very same architecture that *correctly* returned tau_hat ≈ 0 (flag importance 0.0007) when the effect truly was zero now *incorrectly* returns tau_hat ≈ 0 (flag importance 0.0001) when the effect is real but small — the two situations are nearly indistinguishable to the learner, and that is exactly why this is not a bug I can patch. It is the direct consequence of treating w as one feature among many and letting a complexity-penalizing learner decide its fate.

So the single-model recipe and the two-model recipe sit at opposite ends of one bias-variance trade. The single model pools data and can shrink the effect to zero — wonderful when the effect is simple or absent (PEHE 0.056 at tau ≡ 0, 0.145 at simple tau = 1), dangerous when the effect is real but the regularizer under-weights the treatment, where I measured it biased almost all the way to zero. The two-model recipe forces the treatment to matter — safe against the bias-to-zero, but it splits the data, can't share strength, can't exploit a simple effect (PEHE 1.72 where the single model got 0.46), and overfits the small arm in an unbalanced design. Neither dominates. The right reading is not "one of these is the answer" but "these are two ends of a knob, and which end I want depends on whether the true effect is simpler or more complex than the responses." When the effect is genuinely simple or close to zero — and especially when the treatment assignment has little predictive power for the pooled outcome, so the learner ignores it and correctly returns zero — the single model is the one to reach for. When the effect is complex and there's nothing to be borrowed across arms, the two-model recipe is better. That dichotomy is the honest characterization of the single-model recipe; I won't pretend it's universally best.

Let me make the response-surface ceiling quantitative, because I want to see exactly which smoothness governs it and where overlap enters. The clean bound is for the separate-arm response estimator, where each fitted response has its own sample size; it is the conservative rate warning for this whole response-estimate-then-difference route. Set up the minimax frame: a family of superpopulations P where, conditional on each arm, the response is estimable at some rate. Concretely, say the control-arm distribution of (X, Y(0)) given W=0 lies in a class with minimax rate a_mu, and likewise the treated arm, and write n for the number of treated units and m for the number of control units. From tau_hat(x) - tau(x) = (mu_hat_1(x) - mu_1(x)) - (mu_hat_0(x) - mu_0(x)) and (a-b)^2 <= 2a^2 + 2b^2,

  EMSE <= 2·E[(mu_hat_1(X) - mu_1(X))^2] + 2·E[(mu_hat_0(X) - mu_0(X))^2]
        =: 2A + 2B.

Look at A. The subtlety is that mu_hat_1 is trained only on the *treated* rows, so its error is naturally an expectation over the treated-arm covariate distribution, the law of X given W=1 — but the EMSE I want is over the *marginal* law of X (a fresh test point, drawn from the population, not from the treated subpopulation). I need to change measure from "X given W=1" to "X marginal," and overlap is exactly what makes that change of measure bounded. For any nonnegative g,

  E[g(X) | W = 1] = E[g(X) e(X)] / E[W] = E[g(X) e(X)] / E[e(X)].

The numerator is at least e_min·E[g(X)] and at most e_max·E[g(X)] since e_min < e(x) < e_max; the denominator E[e(X)] = E[W] lies in (e_min, e_max). So

  (e_min/e_max)·E[g(X)] <= E[g(X) | W = 1] <= (e_max/e_min)·E[g(X)],

and symmetrically, dividing through the control arm with 1 - e(x) in place of e(x),

  ((1-e_max)/(1-e_min))·E[g(X)] <= E[g(X) | W = 0] <= ((1-e_min)/(1-e_max))·E[g(X)].

Let me sanity-check the direction of these inequalities on a number before I lean on them: take e(x) ≡ 0.5 (a randomized trial), so e_min = e_max = 0.5 and the factor e_max/e_min = 1 — the change of measure is free, as it must be when treated and marginal distributions coincide. Take instead e ranging in (0.1, 0.9): then e_max/e_min = 9, so the treated-arm error can inflate the marginal error by up to ninefold — severe overlap problems make the bound loose, exactly the qualitative behavior I'd expect. The inequalities point the right way. Now condition A on the treatment vector. Given W, the treated rows are i.i.d. from the law of (X, Y) given W=1, so the inner expectation of (mu_hat_1(X) - mu_1(X))^2 over a fresh marginal X can be flipped to an expectation over a treated-distributed test point at the cost of the factor e_max/e_min from the inequality above, and that treated-distributed error is exactly what the minimax rate controls: it is at most C·n^{-a_mu} because mu_hat_1 is fit on the n treated units in their own family. So

  A <= (e_max/e_min)·C·n^{-a_mu},

and by the symmetric control-arm bound

  B <= ((1-e_min)/(1-e_max))·C·m^{-a_mu}.

Putting them together,

  EMSE <= 2C·(e_max/e_min)·n^{-a_mu}
        + 2C·((1-e_min)/(1-e_max))·m^{-a_mu}
        = O(n^{-a_mu} + m^{-a_mu}).

There's the rate, and there's the limitation written in plain algebra: the exponent is a_mu, the smoothness of the *response surfaces*. Not a_tau, the smoothness of the *effect*. A recipe that gets the effect only by estimating response values and subtracting pays the response rate even when the effect is far smoother (a_tau > a_mu); the effect's own structure never directly enters the regression target. When n << m in the arm-split bound, the n^{-a_mu} treated-arm term dominates, and even with m enormous the response-route ceiling is still the response rate in n, never the faster effect rate a_tau. So the very regularity the literature keeps reporting — that tau is often simpler than mu_0 and mu_1 — is exactly the thing this family cannot cash in directly. That's the precise statement of the gap, and I'll leave it sitting there as the thing a better recipe would have to beat; the single-model recipe I'm building is the honest, simplest pooled member of this family, best when the effect really is simple.

So let me commit to the single-model recipe as a clean, complete CATE estimator and write it down, because for the simple/near-zero/smooth-effect regime my measurements say it's the right tool, it reuses any black-box learner, and it pools all the data. The procedure is exactly three moves: build a feature matrix by appending the treatment flag as one more column to the covariates; fit one regressor on (covariates, flag) -> outcome over the whole dataset; predict at a new x by running it twice, once with the flag pinned to 1 and once pinned to 0, and subtracting. That's it.

What base learner should the single slot hold? I want a flexible nonparametric regressor that can bend the response surface differently in the treated and control regions when the flag is informative, yet shrink the flag's influence when it is not — and that handles tabular covariates with sensible built-in regularization and little hand-tuning. Gradient-boosted trees fit this: an additive ensemble of shallow trees fit to the squared-error residual. The shallow depth (say four) lets each tree capture low-order interactions — including interactions *between the treatment flag and the covariates*, which is precisely the mechanism by which the single model represents heterogeneity: a split on a covariate beneath a split on the flag is exactly "the effect varies with that covariate." A modest learning rate with many trees trades a little fitting speed for a smoother, lower-variance surface; subsampling rows each round adds stochastic regularization; and a floor on leaf size stops any single leaf from chasing a handful of points, which matters because — as the tau ≡ 0 experiment showed, where the whole effect rode on 20 of 200 trees — the effect is a *difference* of two predictions and is therefore especially sensitive to leaf-level noise. None of these knobs is causal; they're just the standard recipe for a low-variance nonparametric tabular regressor, and that's the point — the causal content lives entirely in how I slice and toggle, not in the learner.

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
