Let me start from what actually goes wrong when I try to read a causal effect off observational data. I have units with covariates X, a binary treatment T, and an outcome Y, and I want to know what the treatment did — formally, each unit carries two potential outcomes, Y(1) if treated and Y(0) if not, and I only ever see the one matching the treatment it got, Y = T·Y(1) + (1−T)·Y(0). The thing I want, E[Y(1) − Y(0)], is a contrast between something I can see and something I can never see for the same unit. The reflex is to compare the average treated outcome to the average control outcome, E[Y|T=1] − E[Y|T=0]. But this is not the effect, because treatment was not assigned by me — it was assigned out in the world by some process I don't control, and the people who got treated are systematically different in X from the people who didn't. So E[Y|T=1] − E[Y|T=0] mixes the effect of the treatment with the effect of *being the kind of unit that tends to get treated*. The treated and control groups have different X distributions, and part of the outcome gap is that X difference, not the treatment. That is the entire problem: confounding.

I need an assumption to make any of this recoverable, and the honest one is that I've measured enough covariates that, within a cell of units sharing the same X, who got treated is as-good-as-random: (Y(0), Y(1)) ⊥ T | X. Unconfoundedness. I also need overlap, 0 < Pr(T=1|X=x) < 1 for every x — every kind of unit must have had a real chance of either treatment, otherwise there's a region of X where I only ever see one arm and there's simply nothing to contrast. Granting those two, the conditional response surfaces become identified: under unconfoundedness E[Y(w)|X=x] = E[Y|T=w, X=x] =: μ_w(x), because conditioning on X makes T independent of the potential outcomes, so the observed conditional mean in arm w *is* the potential-outcome mean. And then the effect is τ(x) = μ_1(x) − μ_0(x), with the average τ = E_X[μ_1(X) − μ_0(X)].

So one route is staring at me: estimate μ_1 and μ_0 from the data — fit a regression of Y on X in each arm — difference them, average over the empirical distribution of X. That's the obvious thing, and it's what the regression baselines do. Fit μ on the pooled data with T as a feature and read off the treatment coefficient; or fit two separate surfaces, one per arm, and difference. But I don't like where the entire burden lands. Both of those put *all* of the work on getting a high-dimensional, nonlinear function μ_w(x) exactly right. When X is 50-dimensional and the response surface is nonlinear with interactions — which is exactly the regime I care about — that's where flexible learners are least trustworthy. Two independently-fit surfaces, each separately regularized, can carry systematically different biases, and the *difference* of two biased surfaces is spurious heterogeneity that looks like a treatment effect but isn't. And the single-model version has a worse failure: if T is one weak feature among many strong covariates, a regularized or tree-based learner barely splits on it and shrinks the estimated effect toward zero. I'm trading confounding bias for misspecification bias, and I have no second line of defense if the outcome model is wrong.

What bothers me more is that the regression route never *uses* the structure that the assignment side hands me for free. Let me think about what's special here. The thing that contaminates the comparison is that treatment depends on X. So define the object that captures exactly that dependence: e(x) = Pr(T=1 | X=x) = E[T | X=x], the probability a unit with covariates x ends up treated. This is one number per unit — a scalar — no matter how high-dimensional X is. And there's a result I should lean on hard: if I condition on this scalar e(X), is that enough to break the confounding? Conditional on e(X), is the distribution of X the same in the treated and control groups? Let me check the claim X ⊥ T | e(X). It's enough to show Pr(T=1 | X, e(X)) doesn't depend on X beyond e(X). But Pr(T=1 | X) = e(X) by definition, and e(X) is a function of X, so Pr(T=1 | X, e(X)) = Pr(T=1 | X) = e(X), which depends on X only through e(X). So yes — among units with the same propensity score, treatment is balanced with respect to X; e(X) is a balancing score. And it's the *coarsest* one: any function b(x) is a balancing score exactly when e(x) is a function of it, e(x) = f(b(x)), because if b(x) were coarser than e (so two points with e(x_1) ≠ e(x_2) shared a b value) then at that b value T and X would not be independent. So e(X) is the most economical summary that still removes the bias. And crucially, if unconfoundedness holds given the full X, it holds given e(X) too: (Y(0), Y(1)) ⊥ T | e(X). I can collapse adjustment for the whole high-dimensional X down to adjustment for one number. That's a big deal — the confounding lives in a one-dimensional object.

Now, how do I actually *use* e(X)? The coarse ways are already on the table and I find them unsatisfying. I could bin units into strata of similar e(X) and difference treated-minus-control within each stratum, then average across strata — that's subclassification, and it's the direct reading of "within a balancing-score cell the contrast is unbiased, so average over the score." But it's piecewise-constant and crude: residual imbalance survives inside each bin, the number and edges of the bins are arbitrary, and I get a number, not a smooth τ(x). Matching on e(X) has the same flavor — pair each treated unit with a control of similar score — and throws away unmatched units and depends on caliper and match count. Both feel like blunt instruments for what should be a clean reweighting. Wall: I have a scalar e(X) that provably suffices, but binning and matching use it coarsely and don't give me the function τ(x) I want.

Let me back up and think about what the confounding actually *is*, mechanically, in terms of e(X). The treated group is a sample of the population, but a *biased* sample: a unit with covariates x is in the treated group with probability e(x), so high-e units are over-represented among the treated and low-e units under-represented. The control group is the mirror image, over-representing low-e units. So the treated sample is a non-representative draw from the population, drawn with a known (well, estimable) inclusion probability e(x) that varies across units.

And that is a problem I've seen solved cleanly, in a completely different subfield. Survey sampling. Horvitz and Thompson worked out exactly this: how to estimate a population quantity from a sample drawn with *unequal* selection probabilities. Let me reconstruct their argument, because it's the key. Suppose I want the population total S = Σ_{i=1}^N X_i over a finite universe of N elements, but I only get to observe a sample drawn without replacement, where element u_i has some probability P(u_i) of being included in the sample. The raw sample sum Σ_{i in sample} x_i is biased — it over-counts elements with high inclusion probability. I want a linear estimator that's unbiased whatever the unknown X values are. Consider the class Ŝ = Σ_{i in sample} β_i x_i, where each sampled element gets a fixed weight β_i. Its expectation, taking expectation over which elements land in the sample, is E[Ŝ] = Σ_{i=1}^N P(u_i) β_i X_i, because element i contributes β_i X_i exactly when it's included, which happens with probability P(u_i). For this to equal S = Σ X_i for *every* possible configuration of the unknown X's, I need P(u_i) β_i X_i to match X_i term by term, i.e. P(u_i) β_i = 1, forcing β_i = 1/P(u_i). There's no freedom: the only unbiased linear estimator in this class is

  Ŝ = Σ_{i in sample} x_i / P(u_i).

Weight each sampled element by the reciprocal of its inclusion probability. That's the whole idea — inverse-probability weighting — and it's not a heuristic, it's *forced* by the unbiasedness requirement. An element that was unlikely to be sampled but got in stands in for all the similar elements that didn't, so it gets up-weighted by exactly 1/P(u_i) to repair the sampling bias.

The connection writes itself once I line up the analogy. In my causal problem, the "population" is everyone, and the treated units are a sample drawn with inclusion probability e(X). So to estimate the average of Y(1) over the whole population using only the treated units (the ones for whom I observe Y(1)), I should weight each treated unit by 1/e(X). Let me make that precise rather than just hand-wave the analogy, because I want to be sure the conditional expectation actually lands on E[Y(1)] and not something off by a factor. Consider E[ T·Y / e(X) ]. The factor T picks out treated units, where Y = Y(1). Condition on X first:

  E[ T·Y / e(X) | X ] = E[ T·Y(1) / e(X) | X ] = (1/e(X)) · E[ T·Y(1) | X ].

Now use unconfoundedness — given X, T is independent of Y(1) — so E[T·Y(1)|X] = E[T|X]·E[Y(1)|X] = e(X)·μ_1(X). The e(X) cancels:

  E[ T·Y / e(X) | X ] = e(X)·μ_1(X) / e(X) = μ_1(X).

Take the outer expectation over X: E[ T·Y/e(X) ] = E[μ_1(X)] = E[Y(1)]. So inverse-propensity weighting of the treated outcomes recovers the population mean of Y(1) exactly, in expectation. The cancellation of e(X) is the entire trick: the over-representation of high-e units in the treated group (the e(X) that would otherwise bias me) is exactly undone by dividing by e(X). Symmetrically, for the controls: a control unit is "included" with probability 1 − e(X) (the probability of *not* being treated), so weighting control outcomes by 1/(1−e(X)) recovers E[Y(0)]. Same algebra: E[(1−T)Y/(1−e(X)) | X] = (1/(1−e(X)))·E[(1−T)Y(0)|X] = (1/(1−e(X)))·(1−e(X))·μ_0(X) = μ_0(X), and averaging gives E[Y(0)]. So

  τ = E[Y(1) − Y(0)] = E[ T·Y / e(X) − (1−T)·Y / (1−e(X)) ].

The effect is the expectation of a single per-unit quantity that only involves the outcome and the propensity — no outcome model at all. The propensity score, which the balancing-score theory said was sufficient, turns out to be sufficient in this very concrete operational sense: divide by it and the confounding cancels. Replace the expectation by its sample average and I have the estimator

  τ̂ = (1/N) Σ_i [ T_i Y_i / e(X_i) − (1−T_i) Y_i / (1−e(X_i)) ].

This is the Horvitz–Thompson estimator carried over to treatment effects. I never modeled μ_w(x); the only thing I need right is e(x). That's a real shift in where the burden sits — onto a one-dimensional object I'm modeling as a probability, rather than onto two high-dimensional response surfaces.

But e(X) is not handed to me; in observational data the assignment mechanism is unknown, so I have to estimate it. Fitting e(x) is a plain classification problem — predict the binary T from X — and that's exactly what I have flexible classifiers for. Rosenbaum and Rubin recommend a logit model, which is the natural calibrated-probability choice, and noting that I want good probabilities and not just a hard classification, I can also reach for a gradient-boosted classifier when the assignment is nonlinear with interactions, since the synthetic processes I face have propensities built from products and quadratics of the covariates. I want predict_proba, the probability, not predict, the label. So: fit ê(x) by a probability classifier, then plug ê into the weighting.

Now I have to confront the thing that makes inverse-probability weighting dangerous, and it's already visible in the Horvitz–Thompson variance. Go back to the survey derivation and ask what V(Ŝ) looks like. The variance of Ŝ = Σ x_i/P(u_i) has terms like Σ X_i²·(1 − P(u_i))/P(u_i) plus cross terms — the diagonal piece carries a 1/P(u_i) factor. As P(u_i) → 0, that term blows up: an element that was extremely unlikely to be sampled, if it happens to get in, carries a large weight 1/P(u_i), and a single such unit can dominate the estimate and inflate its variance without bound. Translated to my setting: if some treated unit has a tiny ê(X) — a unit that "should almost never have been treated" but was — its weight 1/ê(X) explodes, and one rare unit swings the whole average. The same on the control side as ê(X) → 1, where 1/(1−ê(X)) blows up. This is precisely the place where the overlap assumption was doing real work: Horvitz–Thompson required P(u_i) > 0 for every element, and the variance formula is only finite when the inclusion probabilities stay away from zero. Empirically, even when overlap holds in theory, an estimated ê can stray to within a whisker of 0 or 1 in finite samples, and then the estimator is at the mercy of a few enormous weights. Wall: the estimator is unbiased but its variance can be catastrophic in exactly the low-overlap regions, and a single near-zero ê can destabilize it.

The fix has to bound the weights. The cleanest is to clip the propensity score into a safe interval before inverting it: ê ← min(max(ê, c), 1−c) for some small floor c. Clipping caps every weight at 1/c (and at 1/c on the control side after the upper clip), so no single unit can blow up. This trades a small bias — I'm slightly mis-stating ê for the handful of extreme units — for a large variance reduction, and it's a sensible trade because the extreme-ê units are exactly the ones where the data are thinnest and least trustworthy anyway; I'd rather pull them toward the interior than let them dominate. What floor? A common, well-tested choice is c = 0.05, clipping to [0.05, 0.95], so the maximum inverse-propensity weight is 1/0.05 = 20. It's a bias–variance knob, and 0.05 keeps the variance tame without distorting the bulk of the sample.

There's a second, subtler instability worth addressing even after clipping. Look at the treated weights in the population average: in expectation N^{-1}Σ_i T_i/e(X_i) should be 1, since E[T/e(X)] = E[E[T|X]/e(X)] = E[e(X)/e(X)] = 1, so the inverse-probability treated weights are "supposed to" sum to N. The control side has the same property, E[(1−T)/(1−e(X))] = 1. But that only holds in expectation; in any given sample the realized treated or control weight total can land off from N, and that deviation feeds straight into the estimate as noise. I can kill that source of variance by *self-normalizing*: instead of dividing by N, divide each arm's weighted outcome sum by its own realized sum of weights. For the treated arm, Σ T_i Y_i/ê(X_i) divided by Σ T_i/ê(X_i); for the control arm, Σ (1−T_i) Y_i/(1−ê(X_i)) divided by Σ (1−T_i)/(1−ê(X_i)):

  τ̂_norm = [ Σ_i T_i Y_i / ê(X_i) ] / [ Σ_i T_i / ê(X_i) ]
          − [ Σ_i (1−T_i) Y_i / (1−ê(X_i)) ] / [ Σ_i (1−T_i) / (1−ê(X_i)) ].

This is the self-normalized — Hájek — form. It replaces the raw weights by normalized arm-specific weights that sum to exactly one within each arm, so each arm's estimate is a genuine weighted average of observed outcomes, which the raw Horvitz–Thompson form is not guaranteed to be. It's location-and-scale stable, and it usually has noticeably lower variance. And there's a real efficiency payoff hiding here: when ê is estimated *nonparametrically* rather than known, this normalized weighting estimator attains the semiparametric efficiency bound for the average treatment effect — weighting by the inverse of an estimated propensity is more efficient than weighting by the true propensity, because the estimation of ê implicitly soaks up exactly the right amount of the residual variation. That's the surprising part: I'd have guessed knowing the true e(X) could only help, but for the variance of this estimator, plugging in a flexible estimate ê is what makes it efficient. So even though clipping is my safety valve, normalization is the move that makes the scalar estimator stable and asymptotically efficient under the same regularity conditions.

So far I've built an estimator of the scalar τ. But the harder target is τ(x), the heterogeneous effect across the covariate space, and I want a smooth function, not a stratified number. The Horvitz–Thompson identity pays off a second time, in a *pointwise* form. Look again at the per-unit quantity inside the expectation,

  ψ_i = T_i Y_i / e(X_i) − (1−T_i) Y_i / (1−e(X_i)).

I showed E[ψ] = τ. But the same conditional computation I did above never integrated out X — it stopped at E[ψ | X = x]. Redo it conditionally: E[ T·Y/e(X) | X=x ] = μ_1(x) and E[ (1−T)Y/(1−e(X)) | X=x ] = μ_0(x), so

  E[ ψ | X = x ] = μ_1(x) − μ_0(x) = τ(x).

The IPW pseudo-outcome is, in conditional expectation, exactly the conditional treatment effect. That means ψ is a *noisy but conditionally-unbiased label* for τ(x). So I can compute ψ_i for every unit from the data and the fitted ê, and then regress ψ on X with any flexible regressor: the regression's conditional-mean fit estimates E[ψ|X] = τ(x). I've turned an unobservable target — the per-unit effect — into a regression with an observable, pointwise-unbiased pseudo-outcome. The CATE just falls out as the conditional mean of the IPW pseudo-outcome, and averaging its predictions over the sample recovers the ATE, consistent with the scalar estimator. This is the clean way to get a smooth τ(x) out of inverse-probability weighting without binning anything: weight to remove confounding (constructing ψ), then let a regressor smooth.

The pieces now line up against the failure modes I was worried about. Confounding is handled because the e(X) cancellation makes ψ conditionally unbiased for τ(x) under unconfoundedness — I never have to get a response surface right. Heterogeneity is handled because regressing the pseudo-outcome on X recovers the whole function τ(x), with the regressor's flexibility setting how much heterogeneity I can capture. Overlap and extreme weights are handled by clipping ê into [0.05, 0.95], capping every weight. Residual weight noise is damped by normalization for the scalar estimate and softened in the pseudo-outcome by the clip. The one genuine vulnerability left is that everything rests on ê being reasonable — if the propensity model is badly wrong, the cancellation is imperfect and ψ is biased. That's the price of the route: I move the modeling burden from two high-dimensional outcome surfaces onto one propensity model. For problems where assignment is easier to model than the outcome — or where I just want a method that uses the assignment mechanism the regression baselines ignore — that's the right trade, and it's a single point of failure I can at least diagnose by checking the fitted ê and its overlap.

Now let me write the estimator into the fixed harness — fit a propensity classifier, clip, form the IPW pseudo-outcome, regress X onto it; predict returns the regressor's prediction, which is τ̂(x). Two models, both off-the-shelf sklearn, no outcome surface per arm:

```python
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class CATEEstimator(BaseCATEEstimator):
    """Inverse-propensity-weighting CATE estimator.

    1. Fit the propensity model e(X) = P(T=1|X) (a probability classifier).
    2. Clip e into [0.05, 0.95] for overlap / weight stability.
    3. Form the Horvitz-Thompson IPW pseudo-outcome
         psi = T*Y/e(X) - (1-T)*Y/(1-e(X)),   with  E[psi | X=x] = tau(x).
    4. Regress X -> psi; the conditional mean of psi is tau(x).
    """

    def __init__(self):
        seed = int(os.environ.get("SEED", "42"))
        # propensity model: predict the binary treatment, want calibrated proba
        self._prop_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=seed,
        )
        # CATE model: regress the IPW pseudo-outcome on X
        self._outcome_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=seed + 1,
        )

    def fit(self, X, T, Y):
        # e(X) = P(T=1|X); predict_proba gives the probability, not the label
        self._prop_model.fit(X, T)
        e_hat = self._prop_model.predict_proba(X)[:, 1]
        # clip into [0.05, 0.95]: caps every IPW weight at 20, repairing overlap
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Horvitz-Thompson IPW pseudo-outcome; E[psi | X=x] = tau(x)
        pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)

        # the conditional mean of the pseudo-outcome is tau(x)
        self._outcome_model.fit(X, pseudo_outcome)
        return self

    def predict(self, X):
        # the regression's conditional mean estimates E[psi | X] = tau(x)
        return self._outcome_model.predict(X)
```

The same fitted propensity also gives the population-level estimator the pointwise one is built from — the classical IPW / Hájek average treatment effect, useful when I just want the scalar ATE rather than the surface:

```python
def ipw_ate(X, T, Y, prop_model, clip=0.05, normalized=True):
    """Horvitz-Thompson / Hájek inverse-propensity-weighted ATE."""
    prop_model.fit(X, T)
    e = np.clip(prop_model.predict_proba(X)[:, 1], clip, 1 - clip)
    w1 = T / e                 # treated inclusion prob is e(X)
    w0 = (1 - T) / (1 - e)     # control "inclusion prob" is 1 - e(X)
    if normalized:             # Hájek: divide each arm by its own weight sum
        return np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)
    return np.mean(w1 * Y - w0 * Y)   # raw Horvitz-Thompson
```

I end with one modeling burden instead of two. The naive treated-minus-control difference is confounded because treatment depends on X, and the regression route dumps the whole correction onto high-dimensional outcome surfaces while ignoring the assignment mechanism. Unconfoundedness plus overlap identify the effect, and the propensity score e(X) — the coarsest balancing score — collapses the confounding into one scalar, but binning and matching on it are blunt. Treating the observed treated arm as a non-representative sample drawn with inclusion probability e(X) brings in the Horvitz–Thompson argument, whose unbiasedness condition forces inverse-inclusion-probability weighting. Once I carry that over, weighting outcomes by 1/e(X) on the treated and 1/(1−e(X)) on the controls makes the confounding cancel exactly in expectation, giving τ = E[TY/e(X) − (1−T)Y/(1−e(X))] with no outcome model. I estimate e by a probability classifier, clip ê to [0.05, 0.95] so every inverse-propensity weight is at most 20, and use Hájek normalization when I want the self-normalized scalar average. Reading the same identity conditionally gives the IPW pseudo-outcome ψ with E[ψ|X=x] = τ(x), so regressing ψ on X turns the unobservable conditional effect into a regression target and gives a smooth τ̂(x) whose sample average is the corresponding ATE.
