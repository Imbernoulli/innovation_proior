# Context: heterogeneous treatment effect estimation from observational data

## Research question

We observe `n` i.i.d. records `(X_i, W_i, Y_i)`, with covariates `X_i in R^p`, a binary
treatment `W_i in {0,1}`, and an outcome `Y_i in R`. In the potential-outcomes framework each
unit has a treated and untreated outcome `(Y_i(1), Y_i(0))`, only one of which is ever observed:
`Y_i = Y_i(W_i)`. The quantity we want is not a population average but a *function* — the
conditional average treatment effect

```
tau(x) = E[ Y_i(1) - Y_i(0) | X_i = x ],
```

how the causal effect of the treatment varies across individuals. Estimating `tau(x)` well is
what powers personalized medicine, targeted policy, and customized recommendations.

Three things make this hard at once. First, **confounding**: in observational data treatment is
not randomized; the propensity `e(x) = P(W_i = 1 | X_i = x)` depends on `x`, so a naive
treated-minus-control comparison conflates the treatment effect with differences in who gets
treated. Identification rests on **unconfoundedness**, `{Y_i(0), Y_i(1)} _|_ W_i | X_i`
(Rosenbaum & Rubin 1983): conditional on enough covariates, assignment is as good as random.
Second, **heterogeneity**: `tau(x)` can vary across the covariate space in complex, nonlinear
ways, and *which* covariates drive the variation differs from one scientific question to the
next, so a method must adaptively discover the relevant structure rather than have it specified
in advance. Third, **inference without ground truth**: because only one potential outcome per
unit is ever seen, there is no held-out test set on which to measure `tau`-error — so a usable
estimator must come with a *tractable sampling distribution* (consistency and asymptotic
normality) that lets a practitioner form confidence intervals and test hypotheses directly. A
solution must deliver nonparametric flexibility in high dimension, adapt its notion of
similarity to heterogeneity in `tau`, stay robust to confounding, and admit valid asymptotic
inference.

## Background

The field state rests on several lines of work and several pain points.

**Random forests as adaptive nearest neighbors.** Random forests (Breiman 2001) are an ensemble
of CART trees grown by greedy recursive axis-aligned partitioning on bootstrap/subsample draws
with random split-variable selection; the regression prediction is the average of the per-tree
leaf means, `mu_hat(x) = B^{-1} sum_b mu_hat_b(x)`. A second, equivalent reading of a regression
forest is as an *adaptive kernel*: a tree contributes weight `1/|L_b(x)|` to each training point
in the same leaf `L_b(x)` as `x`, and averaging over trees gives data-driven similarity weights

```
alpha_i(x) = (1/B) sum_b  1{X_i in L_b(x)} / |L_b(x)|,    sum_i alpha_i(x) = 1.
```

For the conditional mean these two views coincide: `sum_i alpha_i(x)(Y_i - mu_hat(x)) = 0` if
and only if `mu_hat(x)` is the average of tree leaf means. This adaptive-neighbor view was
advanced for survival analysis (Hothorn et al. 2004) and quantile estimation (Meinshausen 2006),
and underlies several theoretical analyses (Lin & Jeon 2006). The forest's advantage over a
fixed kernel is that it *learns* which directions in covariate space matter, sidestepping the
curse of dimensionality that cripples kernel methods past two or three covariates.

**Local estimating equations / local maximum likelihood.** A broad classical tradition (Stone
1977; Tibshirani & Hastie 1987; Staniswalis 1989; Newey 1994; Fan, Farmen & Gijbels 1998)
estimates a parameter that varies with covariates, `theta(x)`, by a *local* moment condition.
Many statistical targets — conditional means, quantiles, average partial effects, regression
slopes — are characterized by an estimating equation

```
E[ psi_{theta(x), nu(x)}(O_i) | X_i = x ] = 0,
```

where `psi` is a score function, `O_i` is the observable, and `nu(x)` is an optional nuisance
parameter. Given similarity weights `alpha_i(x)` (classically a kernel `K_h(X_i - x)`), one
solves the plug-in equation `sum_i alpha_i(x) psi_{theta, nu}(O_i) = 0`. Elegant, general, and
asymptotically well understood — but kernel weights suffer badly from the curse of
dimensionality, so the approach is essentially unusable beyond a handful of covariates.

**Orthogonalization / partialling-out.** Robinson (1988) studied the partially linear model
`Y_i = W_i beta + g(X_i) + eps_i`. Writing the conditional means `m(x) = E[Y_i | X_i = x]` and
`e(x) = E[W_i | X_i = x]`, subtracting them gives the residual-on-residual identity

```
Y_i - m(X_i) = (W_i - e(X_i)) beta + eps_i,      E[eps_i | X_i] = 0,
```

and the moment built on the residualized treatment yields a sqrt-n efficient estimator for a
constant `beta`. Such orthogonalized moments are first-order insensitive to errors in the
nuisance functions `m` and `e` — a Neyman-orthogonality property. The recent
debiased/double machine-learning program (Chernozhukov et al. 2016/2018) makes this systematic:
estimate the nuisances `m, e` with *any* flexible ML method, **cross-fit** them (fit on one fold,
predict on another) to break the dependence between nuisance-estimation error and the residuals,
then estimate the low-dimensional target from the residualized moment. Provided the nuisances
are learned at `o(n^{-1/4})` rates, their errors do not contaminate the target's first-order
asymptotics. This whole line, however, has been developed to estimate *constant* low-dimensional
parameters (an average effect, a single coefficient), not a heterogeneous function `tau(x)`.

**Honest sample-splitting.** A recurring diagnostic about adaptive partitioning is that using the
*same* data both to choose splits and to estimate within-leaf values introduces an upward,
selection-driven bias — a leaf chosen because it looked extreme estimates an extreme value. The
observed cure (Biau 2012; Denil, Matheson & de Freitas 2014; and in the treatment-effect setting
Athey & Imbens 2016) is to *split the sample*: use one part of the data to place the splits and a
disjoint part to estimate the quantity in the resulting leaves. This decouples split selection
from estimation and is what makes centered, valid confidence intervals possible.

**Observed limitations of existing heterogeneous-effect methods.** Several diagnostic facts about
prior tree-based causal methods are already established. Methods that build
separate outcome models for treated and control arms and difference them (e.g. via BART, Hill
2011) entangle two distinct jobs — modeling baseline outcome variation and modeling the effect —
in a single fit. Methods that adapt CART's loss to treatment effects are tied to the binary-effect
case and depend on having an unbiased model-free estimate of the leaf MSE. And it is documented
that a forest whose splits chase outcome/propensity structure "spends" splits on variation that
is irrelevant to — or actively confounds — the treatment-effect signal.

## Baselines

The prior methods a new estimator would be measured against and reacts to.

**S-learner / T-learner / IPW.** The simplest plug-ins. S-learner fits one outcome model
`mu_hat(x, w) = E[Y | X = x, W = w]` and reports `tau_hat(x) = mu_hat(x,1) - mu_hat(x,0)`;
T-learner fits two separate models, one per arm, and differences them. Inverse-propensity
weighting reweights observations by `1/e(x)` (treated) and `1/(1-e(x))` (control) to emulate a
randomized comparison. **Gap:** S/T-learners model the full response surfaces — high-variance,
nonlinear objects — when only their *difference* is wanted, and a flexible learner will pour
capacity into baseline variation that cancels in `tau`; IPW is unbiased under unconfoundedness
but its variance explodes when `e(x)` approaches 0 or 1, and it carries no model of how the
effect varies with `x`.

**Kernel local estimating equations** (Stone 1977; Newey 1994; Fan et al. 1998). Solve
`sum_i K_h(X_i - x) psi_{theta, nu}(O_i) = 0` with a fixed kernel and bandwidth `h`. **Gap:**
the fixed, isotropic kernel weights neighbors by raw covariate distance, so in more than a few
dimensions almost no training points are "near" `x` and the estimator's effective sample size
collapses — the curse of dimensionality.

**Causal trees with honest splitting** (Athey & Imbens 2016). A single CART-style tree whose
splitting criterion targets *treatment-effect heterogeneity* rather than outcome fit, estimated
with an honest train/estimate split and a `Cp`-style overfit penalty (Mallows 1973) that corrects
for the extra sampling variance of small leaves. **Gap:** a single tree is high-variance and
discontinuous; the construction is specific to the binary-treatment MSE, for which an unbiased
model-free leaf-error estimate happens to exist — when the target parameter is identified only
through a moment condition, no such unbiased, model-free estimate of the leaf criterion is
available to minimize directly.

**Forest-based treatment-effect estimators** (Wager & Athey 2018). A forest of honest causal
trees with two distinct splitting strategies. *Procedure 1* grows each tree to split on
treatment-effect heterogeneity (an extension of the CART rule to the effect), then estimates the
effect in each leaf; it is established that this is sensitive to changes in the treatment-effect
function but does poorly under strong confounding. *Procedure 2* obtains the neighborhood from a
classification forest trained on the treatment assignments `W_i` (a propensity tree); it is
established to be robust to confounding but insensitive to treatment-effect heterogeneity. The
forest is shown to be pointwise consistent and asymptotically Gaussian, with variance estimable
by the infinitesimal jackknife, enabling confidence intervals. **Gap (noted in that work's own
discussion):** the two procedures have complementary strengths but the hard-coded
nature of each makes it difficult to reconcile them — the practitioner is forced to pick one and
thereby sacrifice either heterogeneity sensitivity or confounding robustness; and each tree
computes its own leaf effect rather than contributing to a shared adaptive neighborhood, and the
exact heterogeneity-loss split is computationally heavy.

## Evaluation settings

The natural yardsticks for a heterogeneous-effect estimator, all pre-existing.

- **Synthetic data-generating processes with known `tau`.** Because real observational data never
  reveal `Y_i(1) - Y_i(0)`, evaluation uses simulators where the ground-truth effect is known.
  A standard family draws `X_i ~ U([0,1]^p)`, `W_i | X_i ~ Bernoulli(e(X_i))`, and
  `Y_i | X_i, W_i ~ N(m(X_i) + (W_i - 0.5) tau(X_i), 1)`, and toggles three regimes: pure
  heterogeneity (`m = 0`, `e = 0.5`, `tau` varying), pure confounding (`tau = 0` but `e(x)` and
  `m(x)` varying with `x`), and both at once. Covariate dimension `p` and sample size `n` are
  swept; irrelevant covariates are added to stress dimension adaptivity.
- **Metrics.** Precision in estimation of heterogeneous effects,
  `PEHE = sqrt( mean_x (tau_hat(x) - tau(x))^2 )`, and the average-effect error
  `|mean_x tau_hat(x) - ATE_true|`. Both lower is better.
- **Protocol.** Stability across resampling is part of the bar: estimates are evaluated over
  repeated train/test splits and seeds so the method is judged on its sampling behavior rather
  than on one realization, and coverage of the asymptotic confidence intervals (does a nominal
  95% interval contain `tau(x)` about 95% of the time?) is itself a reported diagnostic.

## Code framework

The available estimator harness already has the data pipeline for `(X, W, Y)`, off-the-shelf
supervised learners for conditional expectations, a cross-fitting splitter, and a base
tree-ensemble that returns leaf memberships. What remains open is the rule that turns these
pieces into a heterogeneous-effect estimator.

```python
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold


class CATEEstimator:
    """Estimate the heterogeneous effect tau(x) = E[Y(1) - Y(0) | X = x] from
    observational (X, W, Y) under unconfoundedness. Fit/predict harness; the
    effect-estimation rule is left open."""

    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        # off-the-shelf learners for any conditional expectations we may need
        self.model_y = GradientBoostingRegressor(random_state=seed)       # E[Y | X]
        self.model_w = GradientBoostingClassifier(random_state=seed + 1)  # E[W | X]
        # TODO: estimator state

    def _cross_fit(self, X, target, model):
        """Cross-fitted out-of-fold predictions of E[target | X] (an existing
        primitive: fit on one fold, predict on the held-out fold)."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        pred = np.zeros(len(target), dtype=float)
        for tr, va in kf.split(X):
            m = clone(model).fit(X[tr], target[tr])
            pred[va] = (m.predict_proba(X[va])[:, 1]
                        if hasattr(m, "predict_proba") else m.predict(X[va]))
        return pred

    def fit(self, X, W, Y):
        # TODO: use the available primitives to estimate the heterogeneous effect.
        raise NotImplementedError

    def predict(self, X):
        # TODO: return tau_hat(x) for each row of X.
        raise NotImplementedError
```

The cross-fitting and supervised learners are generic; the empty `fit`/`predict` slot is where
the heterogeneous-effect estimator will live.
