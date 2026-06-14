# Context: estimating heterogeneous treatment effects with generic machine learning (circa 2016-2017)

## Research question

We observe `n` i.i.d. observational records `(X_i, W_i, Y_i)`: covariates `X_i`, a binary
treatment `W_i in {0,1}`, and an outcome `Y_i`. Under the potential-outcomes model
`{Y_i(0), Y_i(1)}` with `Y_i = Y_i(W_i)`, the target is the conditional average treatment
effect (CATE)

```
tau*(x) = E[Y(1) - Y(0) | X = x].
```

The data are observational, not randomized, so treatment assignment depends on the
covariates. We assume unconfoundedness — `{Y(0), Y(1)} ⫫ W | X` — which makes `tau*`
identified, and overlap — `eta < e*(x) < 1 - eta` for the propensity `e*(x) = P(W=1|X=x)` —
so that both arms are represented everywhere in covariate space. Two distinct difficulties
have to be paid for simultaneously. First, **confounding**: because `W` correlates with `X`,
any comparison of treated and control outcomes mixes the causal effect with differences in
the covariate distributions of the two groups; a method that does not control for this
estimates a biased effect. Second, **heterogeneity with flexible models**: `tau*(.)` can be
an arbitrary nonlinear function of `X`, so we want to bring modern, high-capacity machine
learning — boosting, neural networks, penalized regression — to bear on estimating it, yet
those methods regularize, and naive uses of them inject systematic bias into the effect
estimate (made precise below).

What a satisfactory solution would have to achieve, beyond just "low error on a benchmark":
(1) be usable with *any* off-the-shelf machine-learning method framed as loss minimization,
not a bespoke algorithm per learner; (2) be *robust* to the inevitable inaccuracy in the
estimated treatment-assignment and outcome models — small errors there should not blow up the
effect estimate; (3) come with a *formal guarantee* tying the accuracy of the effect estimate
to the complexity of `tau*(.)` itself rather than to the (possibly much harder) complexity of
the confounding structure; and (4) let practitioners tune the effect model by ordinary
cross-validation. Existing methods each hit a subset of these; none hits all four. Closing
that gap is the problem.

## Background

The field state at this time. There is a burst of activity on adapting machine learning to
treatment-effect estimation: lasso-based methods (Imai & Ratkovic 2013), recursive
partitioning and causal trees (Athey & Imbens 2015; Su et al. 2009), Bayesian additive
regression trees (Hill 2011; Chipman et al. 2010), random forests (Wager & Athey 2017), boosting
(Powers et al. 2018), and neural networks (Shalit et al. 2017). The literature has not settled
on how machine learning *should* be adapted: most of these methods are bespoke structural
modifications of one particular learner, justified mainly by simulation, and with no error
bound proving they isolate the causal signal better than a plain regression would.

The load-bearing concepts in the field are these.

**Partialling-out / residualization in semiparametrics.** The oldest relevant idea is the
Frisch-Waugh-Lovell theorem (1930s) for linear regression: the coefficient on one regressor
equals the coefficient from regressing the *residual* of the outcome (after projecting out the
other regressors) on the *residual* of that regressor. Robinson (1988) lifted this to the
**partially linear model** `E(Y | X, Z) = beta'X + theta(Z)`, where `theta` is an unknown
nonparametric nuisance and `beta` is a finite-dimensional parameter of interest. His key
observation: the model implies

```
Y - E(Y | Z) = beta'(X - E(X | Z)) + U,   E(U | X, Z) = 0,
```

so one can estimate `E(Y|Z)` and `E(X|Z)` by nonparametric (kernel) regression, form the two
residuals, and recover `beta` by no-intercept OLS of one residual on the other. Robinson
proved this gives `sqrt(n)`-consistent, asymptotically normal `beta` *even though the kernel
nuisances themselves converge slower than `sqrt(n)`*. The crucial qualitative fact: the target
`beta` can be estimated at a fast rate while the nuisances are estimated at a slow rate. The
target there is a constant slope vector; the nonparametric piece is the thing partialled out.

**Doubly-robust / orthogonal-moment estimation and cross-fitting.** A semiparametrics program
(van der Laan & Robins; Newey 1994; Robins 2004; Chernozhukov et al. 2018) studies estimation
of a low-dimensional target parameter `theta0` in the presence of nuisance functions `eta0`
that cannot be estimated at `sqrt(n)`. Two ingredients make `sqrt(n)`-rate inference on
`theta0` possible. (i) A **Neyman-orthogonal** estimating equation: a moment condition whose
sensitivity to the nuisance, evaluated at the truth, is first-order zero (its directional
derivative in the nuisance direction vanishes), so that first-order errors in the estimated
nuisance do not propagate into the target. (ii) **Cross-fitting**: estimate the nuisance on
data folds *other* than the one on which the moment is evaluated, so the nuisance estimate is
statistically independent of the held-out observations; this removes the own-observation
overfitting bias and lets one dispense with empirical-process (Donsker) restrictions on the
nuisance class, which is exactly what one needs when the nuisance is fit by an arbitrary
high-capacity learner. The canonical worked example of this whole program *is* the partially
linear model — Robinson's residual-on-residual recast as an orthogonal moment with cross-fit
nuisances. The diagnostic fact established by this literature: naively plugging an ML estimate
of `eta0` into an estimating equation for `theta0` is *not* `sqrt(n)`-consistent, because
regularization bias and overfitting in `eta0` leak into `theta0`; the orthogonal-plus-cross-fit
construction is what removes that leak.

**A diagnostic about regularization bias in effect estimation.** Consider the
high-dimensional linear model `Y_i(w) = X_i' beta*_(w) + eps_i(w)`. Fit two separate lassos,
one per arm, and difference the coefficients to get the effect. Because each `beta_(w)` is
independently shrunk toward zero, their difference `beta_(1) - beta_(0)` can be regularized
*away from zero even when the true effect is zero everywhere* — the two shrinkages do not
cancel, and the problem is acute when the arms have very different sample sizes. This is an
observed pathology of arm-wise modeling in meta-learner examples, and it is
why "model each arm and subtract" is fragile: the regularization meant to help prediction in
each arm actively corrupts the difference.

## Baselines

The prior methods a new CATE estimator would be measured against and would react to. Each turns
a generic predictor into an effect estimator.

**S-learner.** Fit a single regression `f-hat(x, w) = E[Y | X=x, W=w]` that takes the treatment
as just another input feature, then set `tau-hat(x) = f-hat(x, 1) - f-hat(x, 0)`. Core appeal:
one model, trivially uses any learner. *Limitation:* because `W` is one feature among many, a
regularized learner can shrink its influence or split on it rarely, so the fitted `f-hat`
barely depends on `w`; the estimated effect is then biased toward zero, and the learner gives
no special status to the quantity we actually care about.

**T-learner.** Fit the two arm response surfaces separately,
`mu-hat_(w)(x) = E[Y | X=x, W=w]` for `w in {0,1}` on the treated and control subsamples, and
set `tau-hat(x) = mu-hat_(1)(x) - mu-hat_(0)(x)`. Core idea: let each arm have its own model.
*Limitation:* the two models are trained independently and separately regularized, so their
difference is unstable and inherits the spurious-structure pathology above; with the lasso it
can regularize the effect away from zero even when the truth is null, and the instability is
worst under arm imbalance. It also models the full response surface in each arm — including all
the confounding structure — even when the effect itself is simple.

**X-learner.** A two-stage repair of the T-learner from the meta-learner line. Fit `mu-hat_(w)`;
then on the treated units form imputed effects `D_i = Y_i - mu-hat_(0)^{(-i)}(X_i)` and regress
them on `X` to get `tau-hat_(1)`; symmetrically get `tau-hat_(0)` from controls; combine
`tau-hat(x) = (1 - e-hat(x)) tau-hat_(1)(x) + e-hat(x) tau-hat_(0)(x)`. Core idea: regress on
imputed individual effects, weighting the two estimates by the propensity. *Limitation:* the
effect estimate inherits the errors of `mu-hat_(0), mu-hat_(1)` to *first order* — a small
perturbation of those arm models of size `o(n^{-1/4})` shifts `tau-hat` by the same order — so
its accuracy is tied to how well the (possibly complex) arm surfaces are estimated, not to the
complexity of the effect alone. It is robust only in the extreme regime where one arm vastly
outnumbers the other.

**U-learner / transformed-outcome regression.** Use the algebraic identity that, with the
marginal mean `m*(x) = E[Y|X=x]` and propensity `e*(x)`,
`E[(Y - m*(X)) / (W - e*(X)) | X = x] = tau*(x)`, and simply regress the transformed outcome
`U_i = (Y_i - m-hat(X_i)) / (W_i - e-hat(X_i))` on `X` with any learner. Related propensity-
weighting transforms (Athey & Imbens 2015; Tian et al. 2014) regress
`Y_i (W_i - e*) / (e*(1-e*))` on `X`. Core idea: one off-the-shelf regression of a constructed
target. *Limitation:* the divisor `W_i - e*(X_i)` is near zero wherever the propensity
approaches 0 or 1, so the transformed outcome has enormous variance there; the estimator is
unstable and, empirically, pays a large error for it.

**Causal-structure learners (causal forest, causal boosting, BART-style outcome modeling).** Bespoke
modifications of a specific learner — splitting criteria that target effect heterogeneity
(Wager & Athey 2017; Powers et al. 2018), or Bayesian regression trees used for the response
surface (Hill 2011). Core idea: rebuild the learner internals to chase the
effect. *Limitation:* each is tied to one learner and must accomplish *both* jobs at once —
controlling confounding and expressing the effect — inside its own machinery, so it cannot be
swapped for a different learner, cannot be tuned by ordinary cross-validation, and comes (with
exceptions) without an error bound certifying it isolates the causal component.

## Evaluation settings

The natural yardsticks already in use, with known ground-truth effects so error in the *effect*
can be measured directly.

- **Semi-synthetic field-experiment data.** Take a real randomized get-out-the-vote study
  (Arceneaux et al. 2006) where the measured effect is essentially nil, keep the real covariate
  and propensity structure, and *spike in* a known synthetic effect function `tau*(.)` (here by
  strategically flipping a fraction of the binary outcomes). `d = 11` covariates; binary `Y` and
  `W`. Because `tau*` is set by the analyst, the oracle test-set mean-squared error
  `(1/n_test) sum (tau-hat(X_i) - tau*(X_i))^2` is computable.
- **Fully synthetic designs.** Draw `X_i ~ P_d`, `W_i | X_i ~ Bernoulli(e*(X_i))`,
  `eps_i ~ N(0,1)`, and `Y_i = b*(X_i) + (W_i - 0.5) tau*(X_i) + sigma eps_i`, for several
  configurations chosen to stress different difficulties: hard confounding with an easy effect
  (a Friedman-type baseline `b*`, a trimmed-sinusoid propensity, `tau*(x) = (x_1 + x_2)/2`); a
  randomized trial `e* = 1/2` where no confounding control is needed; an easy propensity with a
  hard baseline and strong confounding but a constant effect `tau* = 1`; and unrelated treated/
  control arms. Sweep sample size `n`, dimension `d`, and noise `sigma`.
- **Metrics and protocol.** Test-set mean-squared error of the estimated effect against `tau*`
  (precision in estimating heterogeneous effects), and `|mean(tau-hat) - ATE_true|` for the
  average effect; reported relative to an oracle that knows the nuisance functions. Nuisance
  models and the effect model are each tuned by `k`-fold cross-validation (typically 5 or 10
  folds), with results aggregated over repetitions to reward stability across splits.
- **Implementations.** `glmnet` for penalized regression, weighted kernel ridge, and `XGBoost`
  for boosting, so that "use a generic loss-minimizer" can be realized concretely.

## Code framework

The estimator plugs into a generic CATE harness with a `fit(X, W, Y)` / `predict(X)` interface.
The available pieces are standard supervised learners for regression and classification
(gradient boosting, penalized regression), `KFold` for splitting data into folds, and the data
pipeline. The unresolved slot is the rule that turns those generic predictive components into
a confounding-adjusted heterogeneous-effect estimator.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """Estimate tau(x) = E[Y(1) - Y(0) | X = x] from observational (X, W, Y),
    under unconfoundedness and overlap, reusing generic supervised learners."""

    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed

    def _make_regressor(self):
        # a generic, well-tuned predictive regressor (already exists)
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed,
        )

    def _make_classifier(self):
        # a generic, well-tuned predictive classifier (already exists)
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed + 1,
        )

    def fit(self, X, W, Y):
        X, W, Y = np.asarray(X), np.asarray(W), np.asarray(Y)
        # Generic held-out predictions for Y from X and for W from X are easy.
        # A direct arm-wise difference is the fragile route; the open slot is
        # how to combine predictive components without inheriting that fragility.
        # TODO: the estimator core.
        raise NotImplementedError

    def predict(self, X):
        # TODO: return the per-row effect estimate from the object fitted above.
        raise NotImplementedError
```
