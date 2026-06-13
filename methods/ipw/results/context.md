## Research question

We have observational data on `N` units: a vector of pretreatment covariates `X_i`, a binary
treatment indicator `T_i in {0, 1}`, and a scalar outcome `Y_i`. We did **not** run the
experiment — treatment was assigned by some unknown process out in the world, not by us
flipping a coin. We want the causal effect of the treatment on the outcome. In the
potential-outcome framing each unit has two latent outcomes, `Y_i(1)` (what its outcome would
be if treated) and `Y_i(0)` (if untreated), and we observe only the one that matches the
treatment it actually received: `Y_i = T_i * Y_i(1) + (1 - T_i) * Y_i(0)`. The targets are the
average treatment effect `tau = E[Y(1) - Y(0)]` and, more ambitiously, the conditional
(heterogeneous) effect `tau(x) = E[Y(1) - Y(0) | X = x]` — how the effect varies across the
covariate space.

The structural difficulty is that we never see both potential outcomes for the same unit, so
the effect is a contrast between an observed quantity and a counterfactual. The obvious
fix — compare the average outcome of the treated to the average outcome of the untreated,
`E[Y | T = 1] - E[Y | T = 0]` — is biased whenever the treatment was not randomized, because
the treated and the untreated differ systematically in `X`, and some of the outcome gap is
that `X` difference rather than the treatment. A solution has to disentangle the effect of the
treatment from the effect of being the *kind of unit that tends to get treated*, using only
the covariates we observed, ideally without committing to a rigid parametric model of how `Y`
depends on `X` (which would just trade confounding bias for misspecification bias).

## Background

The field that studies this is causal inference under *selection on observables*. A few
load-bearing pieces of theory and prior practice are in place.

**Potential outcomes and the fundamental problem.** The notation
`Y_i(1), Y_i(0)` with `Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)` formalizes causal effects as
contrasts of potential outcomes (developed by Neyman 1923 for randomized experiments and by
Rubin 1974, 1977, 1978 for observational ones). The fundamental problem of causal inference
is that exactly one of the two is ever observed per unit, so the individual effect
`Y_i(1) - Y_i(0)` is never directly seen; only averages are recoverable, and only under
assumptions. The stability assumption (SUTVA) — one unit's outcome is unaffected by others'
treatments, and there is a single version of each treatment — is maintained throughout.

**What randomization buys, and why observational data lacks it.** In a randomized experiment,
treatment assignment `T` is independent of the potential outcomes, so the treated and control
groups are exchangeable and the simple difference in means is unbiased. In observational data
this independence fails: the same covariates that drive who gets treated may also drive the
outcome (a *confounder*), so the groups are not comparable. The diagnostic fact that motivates
everything below is concrete: in a nonrandomized study the units exposed to one treatment
generally differ systematically in `X` from those exposed to the other, so the naive
treated-minus-control difference confounds the treatment effect with the covariate imbalance.

**The identifying assumptions.** Two conditions make the effect recoverable from observational
data (Rosenbaum & Rubin 1983; Rubin 1978). *Unconfoundedness* (also called ignorability,
selection on observables, or conditional independence): conditional on the covariates,
treatment is independent of the potential outcomes,

```
(Y(0), Y(1)) ⊥ T | X.
```

This says that within a cell of units sharing the same `X`, who got treated is as-good-as
random. *Overlap* (positivity): every unit has a genuine chance of either treatment,

```
0 < Pr(T = 1 | X = x) < 1   for all x in the support.
```

Together these identify the conditional response surfaces `mu_w(x) = E[Y | T = w, X = x]`,
since under unconfoundedness `E[Y(w) | X = x] = E[Y | T = w, X = x]`, and hence the effect
`tau(x) = mu_1(x) - mu_0(x)` and its average `tau = E_X[mu_1(X) - mu_0(X)]`. Overlap is not a
technicality: where `Pr(T = 1 | X = x)` approaches 0 or 1 the data contain essentially only
one treatment arm at that `x`, so the local contrast is unestimable.

**The propensity score and the balancing-score result (Rosenbaum & Rubin 1983).** Define the
*propensity score* `e(x) = Pr(T = 1 | X = x) = E[T | X = x]`, the conditional probability of
receiving treatment. A *balancing score* `b(x)` is any function of `x` such that
`X ⊥ T | b(x)`. Rosenbaum and Rubin proved a chain of results about it. The propensity score
is a balancing score (their Theorem 1: `X ⊥ T | e(X)`), and in fact it is the *coarsest* one —
a function `b(x)` is a balancing score iff it is finer than `e(x)`, i.e. `e(x) = f(b(x))` for
some `f` (Theorem 2); `x` itself is the finest. If treatment is unconfounded (strongly
ignorable) given `x`, then it is also unconfounded given any balancing score `b(x)`
(Theorem 3), in particular given the scalar `e(X)`. This collapses the conditioning from the
full high-dimensional `X` down to a single number: adjusting for the scalar `e(X)` removes the
same bias as adjusting for all of `X`. Their Theorem 4 makes the consequence explicit — under
strong ignorability the expected treated-minus-control difference *at a fixed value of a
balancing score*, averaged over the distribution of that score, equals the average treatment
effect,

```
E_{b(X)}[ E(Y | b(X), T = 1) - E(Y | b(X), T = 0) ] = E(Y(1) - Y(0)).
```

In practice `e(x)` is unknown and must be modeled; Rosenbaum and Rubin recommend a logit model
(Cox 1970), noting that raw subclassification on `X` fails in any non-trivial dimension
because the empirical assignment proportion is 0 or 1 in almost every covariate cell.

**The survey-sampling heritage.** Long before treatment effects, survey statistics solved a
formally identical problem: estimate a population total from a sample drawn with *unequal*
selection probabilities. When elements enter a sample with known but heterogeneous inclusion
probabilities `P(u_i)`, a representative-by-construction sample is not available, and the raw
sample sum is biased toward whichever elements were over-sampled. Horvitz and Thompson (1952)
worked out the theory of unbiased estimation under arbitrary selection probabilities — the
estimator, its sampling variance, and an unbiased estimator of that variance — for one-stage
designs, with the requirement that every element have positive inclusion probability. Their
linear-unbiasedness argument is exact: if a sampled element `u_i` is weighted by `beta_i`, then
unbiasedness of `sum_{i in sample} beta_i x_i` for `sum_i x_i`, for every possible population
vector, forces `P(u_i) beta_i = 1` term by term, hence `beta_i = 1 / P(u_i)`. That body of
theory about reweighting a non-representative sample by its selection probabilities is sitting
in the background, in a different subfield, waiting to be connected to the observational-causal
problem where the "selection probability" into the treated group is exactly `e(X)`.

## Baselines

These are standard estimators for the unconfounded-observational problem, each with its
concrete form and the gap it leaves.

**S-learner (single model / regression with a treatment indicator).** Fit one regression
`mu(x, t) = E[Y | X = x, T = t]` on the pooled data, treating `T` as just another feature, and
estimate `tau_hat(x) = mu(x, 1) - mu(x, 0)`. Concretely one might fit `Y ~ alpha + beta'X +
gamma T` and read off `gamma`, or use a flexible learner on the augmented feature vector
`[X, T]`. Core idea: model the response surface directly and difference it. **Gap:** the single
model can shrink the treatment indicator's influence toward zero — when `T` is one weak
feature among many strong covariates, a regularized or tree-based learner may barely split on
it, biasing the estimated effect toward zero; and a misspecified `mu` produces a biased effect
with no second line of defense.

**T-learner (separate outcome models per arm).** Fit two regressions,
`mu_1(x) = E[Y | X = x, T = 1]` on the treated and `mu_0(x) = E[Y | X = x, T = 0]` on the
controls, and difference them: `tau_hat(x) = mu_1(x) - mu_0(x)`. The population effect is the
average of the differenced predictions over the empirical covariate distribution,
`(1/N) sum_i [mu_1_hat(X_i) - mu_0_hat(X_i)]`. Core idea: let each arm have its own response
surface. **Gap:** each model is fit on only its own arm's data, so in regions where one arm is
sparse (exactly the low-overlap regions) that arm's surface is extrapolated, and two
independently-fit, separately-regularized surfaces can have systematically different bias whose
difference is spurious heterogeneity; the procedure depends entirely on getting both outcome
surfaces right and never uses the assignment mechanism.

**Subclassification / stratification on the propensity score.** Estimate `e(x)`, partition
units into strata of similar `e(x)` (e.g. quintiles), take the treated-minus-control mean
difference within each stratum, and average across strata weighted by stratum size. Core idea:
within a propensity stratum the groups are approximately balanced (the balancing-score
property), so the within-stratum contrast is approximately unbiased; this is the direct
operationalization of Rosenbaum & Rubin's Theorem 4. **Gap:** it is a coarse, piecewise-constant
use of `e(x)` — residual within-stratum imbalance remains, the number and placement of strata
are arbitrary, and it does not naturally yield a smooth `tau(x)` over the covariate space.

**Propensity-score matching.** For each treated unit find one or more control units with
similar `e(x)` and contrast their outcomes; average over matches. Core idea: rebuild a
balanced comparison set by pairing on the scalar score. **Gap:** discards unmatched units,
introduces dependence on the match count and caliper, and the estimate is a sum of local
contrasts rather than a single closed-form quantity; like stratification it gives a number, not
a function `tau(x)`.

**Outcome-model averaging without the assignment mechanism, generally.** All of the regression
routes above (S- and T-learner) share one structural commitment: they put the entire burden of
removing confounding on correctly estimating one or two high-dimensional outcome surfaces
`mu_w(x)`. When those surfaces are nonlinear and the covariate dimension is high, this is
exactly where flexible learners are least reliable, and there is no use made of the comparatively
low-dimensional object — the one-dimensional `e(x)` — that the balancing-score theory says is
sufficient to remove the bias. The prior art either models the outcome while ignoring `e`, or uses
`e` only through coarse grouping or pair construction, so the remaining gap is variance- and
bias-prone adjustment in the high-overlap and low-overlap regions where a smooth heterogeneity
estimate is still desired.

## Evaluation settings

The natural yardsticks for a CATE estimator on synthetic observational data:

- **Synthetic benchmark families with known ground-truth effects.** Three task-local
  data-generating processes, each producing covariates `X`, an observational binary treatment
  `T` assigned by a nonlinear, interaction-laden propensity model, an outcome `Y` from a
  nonlinear response surface, and the true per-unit effect `tau` and true `ATE`: an IHDP-flavored
  small-sample process (`n approx 747`, `p approx 25`, nonlinear effects); a Jobs/LaLonde-flavored
  process (`n approx 2000`, `p approx 10`, economic/earnings outcomes); and an ACIC-flavored
  high-dimensional process (`n approx 4000`, `p approx 50`, correlated covariates, complex
  confounding). In every one, the propensity depends on the same variables that drive the
  heterogeneous effect (strong confounding), with treatment probabilities kept away from 0 and 1
  so both arms remain represented across the covariate space. These are inspired by the IHDP /
  Jobs / ACIC families but are not the official datasets.

- **Metrics (both lower-is-better).** *PEHE* (Precision in Estimation of Heterogeneous
  Effects), `sqrt( mean( (tau_hat - tau_true)^2 ) )`, which scores the whole `tau(x)` surface;
  and *ATE error*, `| mean(tau_hat) - ATE_true |`, which scores only the average.

- **Protocol.** 5-fold cross-fitting (fit on `K - 1` folds, predict the held-out fold)
  repeated over 10 random seeds, so an estimator is rewarded for stability across train/test
  splits rather than for fitting one realization. The estimator interface is `fit(X, T, Y) ->
  self` and `predict(X) -> tau_hat` of shape `(n,)`.

## Code framework

The estimator plugs into an existing scikit-learn-based harness. The data-generating
processes, the cross-fitting evaluation, the PEHE / ATE-error metrics, and the
`fit`/`predict` interface are all fixed; the one open slot is the body of the `CATEEstimator`
class — how it learns from `(X, T, Y)` and how it produces `tau_hat(x)`. Nothing about that
strategy is settled here, so the substrate is only the generic machinery that already exists:
numpy/scipy, and scikit-learn's regressors and classifiers (linear models, gradient-boosted
trees, random forests, neural nets) with their standard `fit` / `predict` / `predict_proba`
methods. The single empty slot is the estimator itself.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# ... other sklearn regressors/classifiers available


class BaseCATEEstimator:
    """Fixed interface: fit on observational (X, T, Y); predict per-unit effects."""

    def fit(self, X, T, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class CATEEstimator(BaseCATEEstimator):
    """Estimator implementation slot.

    fit(X, T, Y): X is (n, p) covariates, T is (n,) binary treatment, Y is (n,)
    observed outcomes. predict(X): return tau_hat of shape (n,), the estimated
    per-unit treatment effect E[Y(1) - Y(0) | X = x].
    """

    def __init__(self):
        # TODO: models and state for an estimator implementation.
        pass

    def fit(self, X, T, Y):
        # TODO: estimation strategy mapping observational (X, T, Y)
        #       into an estimate of tau(x).
        pass

    def predict(self, X):
        # TODO: return the per-unit treatment-effect estimates tau_hat.
        pass


# existing cross-fitting evaluation the estimator plugs into
def evaluate_estimator(estimator, X, T, Y, tau_true, ate_true, n_splits=5, seed=42):
    from sklearn.model_selection import KFold
    import copy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tau_hat = np.zeros(len(X))
    for tr, te in kf.split(X):
        est = copy.deepcopy(estimator)
        est.fit(X[tr], T[tr], Y[tr])
        tau_hat[te] = est.predict(X[te])
    pehe = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    ate_err = np.abs(np.mean(tau_hat) - ate_true)
    return {"PEHE": pehe, "ATE_error": ate_err}
```

The harness supplies `(X, T, Y)` to `fit` and asks `predict` for the per-unit effects; the
`__init__` / `fit` / `predict` bodies are where the estimator will live.
