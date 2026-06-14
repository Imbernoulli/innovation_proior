# Context: heterogeneous treatment-effect estimation from observational data

## Research question

We observe an i.i.d. sample of `Z = (X, A, Y)`: covariates `X` in `R^d`, a binary treatment or
exposure `A` in `{0,1}`, and a real-valued outcome `Y`. The target is the **conditional average
treatment effect** (CATE),

```
tau(x) = E(Y^1 - Y^0 | X = x),
```

the expected difference in outcomes had the units with covariates `X = x` been treated versus not.
Under the standard causal assumptions — no unmeasured confounding (`{Y^0, Y^1} ⟂ A | X`),
consistency, and positivity/overlap (`epsilon <= pi(x) <= 1 - epsilon` with probability one for
the propensity `pi(x) = P(A=1|X=x)`) — this counterfactual contrast is identified from the observed
data as a difference of two regression functions,

```
tau(x) = mu_1(x) - mu_0(x),    mu_a(x) = E(Y | X = x, A = a).
```

The problem is hard for three reasons that interact. (1) **Confounding**: because `A` depends on
`X`, naive treated-minus-control comparisons are biased, and any honest estimator must adjust for
`X` either through the outcome models `mu_a` or through the propensity `pi`. (2) **Heterogeneity**:
`tau(x)` can vary across the covariate space in complex, nonlinear ways, so we need a flexible,
nonparametric estimate of the *whole function*, not a single scalar. (3) **Differential
complexity**: the individual response surfaces `mu_a` are often complicated and hard to estimate,
yet their *difference* `tau` can be far simpler — even constant or zero — and we want an estimator
that can exploit that simplicity rather than inheriting the complexity of `mu_1` and `mu_0`.

What a good solution must achieve, beyond accuracy: it should be **flexible** (usable with any
modern machine-learning regressor for the various sub-models), **robust to confounding**, and
ideally enjoy a guarantee that it stays accurate even when the components used to fight confounding
are themselves estimated only crudely — in particular, a guarantee that the final CATE error does
not simply equal the (potentially slow) error of estimating `mu_a` or `pi`. The flagship pain point
is the gap between what we can estimate well (a scalar summary of the effect) and what we want (the
full heterogeneous function), under nuisances that machine learning estimates at slower-than-`sqrt(n)`
rates.

## Background

**The identification and the nuisances.** Everything downstream is built from three nuisance
functions of `X`: the propensity `pi(x) = P(A=1|X=x)`, the two arm-specific outcome regressions
`mu_a(x) = E(Y|X=x,A=a)`, and sometimes the marginal outcome regression
`eta(x) = E(Y|X=x) = pi(x) mu_1(x) + (1-pi(x)) mu_0(x)`. The CATE is the contrast
`mu_1 - mu_0`; the average treatment effect (ATE) is its mean, `E{tau(X)}`.

**A diagnostic phenomenon that drives everything.** Consider a one-dimensional design where the
treatment is strongly confounded — say `pi(x) = 0.5 + 0.4 sign(x)`, so the left half of the
covariate space is mostly untreated and the right half mostly treated — while the two response
surfaces are *equal*, `mu_1 = mu_0`, and are an awkward, non-smooth piecewise-polynomial function.
The true CATE is then exactly the constant zero, the simplest possible function. But estimate each
surface separately: where treated units are scarce (the left), `mu_1` is fit from little data and
oversmooths; where controls are scarce (the right), `mu_0` undersmooths. Their difference is
therefore a complicated, spurious bump — large error for a target that is identically zero. This is
not a quirk of one dataset; it is the generic consequence of estimating two hard surfaces under
confounding and subtracting them, and it is the empirical fact any method here must answer to: the
difficulty of `tau` can be *much* lower than the difficulty of the `mu_a`, and an estimator built
by differencing the `mu_a` cannot see that.

**Semiparametric efficiency theory for the average.** For the *scalar* ATE there is a mature,
load-bearing theory. The functional `psi(P) = E{E(Y|X,A=1)}` (and the analogous control mean) is
pathwise differentiable, with a von Mises / influence-function expansion

```
psi(Pbar) - psi(P) = ∫ phi(z; Pbar) d(Pbar - P)(z) + R_2(Pbar, P),
```

where `phi` is the mean-zero efficient influence function and `R_2` is a **second-order remainder**
— it depends only on *products or squares* of the differences between distributions. For the
treated mean the influence function and remainder are known explicitly (Robins & Rotnitzky 1995;
Hahn 1998):

```
phi(Z; P) = 1(A=1)/pi(X) {Y - mu_1(X)} + mu_1(X) - psi,
R_2(Pbar, P) = ∫ {1/pibar(x) - 1/pi(x)} {mu_1(x) - mubar_1(x)} pi(x) dP(x).
```

The remainder being a *product* of the propensity error and the outcome error is the crucial
structural fact: each individual error can be large (slow nonparametric rate) yet the product can
be small. This is the source of the robustness property people want — and it is stated, in the
classical theory, for the *scalar* functional.

**Regularization bias and why plug-in machine learning fails.** Chernozhukov, Chetverikov, Demirer,
Duflo, Hansen, Newey & Robins (2018) made the practical mechanism precise. If you estimate a
nuisance `g_0` with a regularized machine learner (lasso, forests, boosting, nets — all of which
trade variance for bias) and plug it into an estimating equation for the target, the scaled error
picks up a term like `(1/sqrt(n)) sum_i m_0(X_i){g_0(X_i) - ghat_0(X_i)}` whose summands do *not*
have mean zero; with a nuisance rate `n^{-phi_g}` and `phi_g < 1/2`, this term is of order
`sqrt(n) n^{-phi_g} -> infinity`, so the naive plug-in is not even `sqrt(n)`-consistent. Two
ingredients fix it. **Orthogonalization**: reformulate the estimating equation so it is *Neyman
orthogonal* — first-order insensitive to nuisance perturbations — which makes the leading bias the
*product* `(mhat_0 - m_0)(ghat_0 - g_0)`, of order `sqrt(n) n^{-(phi_m + phi_g)}`, which can vanish
even when each nuisance converges slowly. **Cross-fitting**: estimate the nuisances on one part of
the data and evaluate the target on an independent part, then swap and average; this kills the
empirical-process / own-observation coupling that would otherwise reintroduce a non-vanishing
remainder, with no Donsker-class restriction on the (highly complex) ML estimators. All of this,
again, is developed for `sqrt(n)`-estimable *scalar* parameters.

**Residualization / partialling out.** A second classical idea targets the effect directly rather
than through the outcome levels. If we write `m(x) = E(Y|X=x)` for the marginal outcome regression,
then under unconfoundedness the Robinson (1988) decomposition holds:

```
Y - m(X) = {A - pi(X)} tau(X) + epsilon,   E(epsilon | X, A) = 0,
```

so the outcome residual equals the treatment residual times the CATE plus noise. This isolates the
causal signal: nuisances `m` and `pi` are *partialled out* of `Y` and `A`, and what remains is
governed by `tau`. It is the basis of the partially-linear-model machinery and of g-estimation.

**Smoothness and the achievable yardstick.** In nonparametric terms, if a function is `s`-smooth
(Hölder class `H(s)`) on `R^d`, a minimax-optimal regression estimates it at the pointwise rate
`n^{-1/(2 + d/s)}`; if it is `s`-sparse, at roughly `sqrt(s log d / n)`. An *oracle* who could
observe the true individual treatment contrast `Y^1 - Y^0` and regress it on `X` would attain the
rate set by the smoothness/sparsity of `tau` *alone*. That oracle rate is the benchmark a good CATE
estimator should aspire to match — converging at the complexity of `tau`, not of the harder `mu_a`.

## Baselines

These are the prior CATE estimators a new method is measured against, with the gap each leaves open.

**S-learner (single model).** Fit one regression `mu(x, a)` of `Y` on `(X, A)` jointly, then set
`tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0)`. Simple and data-efficient, but the learner is free to
treat `A` as just another weak feature; regularization shrinks the fitted `A`-dependence, biasing
`tau` toward zero and often washing out genuine heterogeneity. **Gap:** the treatment can be
under-weighted by the very regularization that makes the model work, and there is no mechanism that
ties the estimator's error to the (possibly low) complexity of `tau` rather than that of `mu`.

**T-learner (two models / plug-in).** Fit `mu_1` on the treated and `mu_0` on the controls
separately, and difference them: `tau_hat = mu_1_hat - mu_0_hat` (Künzel, Sekhon, Bickel & Yu 2019
name this the T-learner). This is the natural plug-in for `tau = mu_1 - mu_0`. **Gap:** it is
exactly the differencing estimator the diagnostic phenomenon above indicts. Each arm is fit only on
its own (possibly scarce, confounding-skewed) sub-sample, the two fits oversmooth/undersmooth in
opposite regions, and the difference inherits the *larger* of the two surface errors. When `tau` is
much simpler than `mu_0, mu_1`, the plug-in cannot exploit that simplicity — its error is the
nuisance error, with no second-order cancellation.

**Inverse-probability weighting (IPW) of a transformed outcome.** With a known or estimated
propensity, the weighted transform `(A - pi(X)) Y / {pi(X)(1 - pi(X))}` has conditional mean
exactly `tau(x)` (Athey & Imbens-style weighting; Tian et al.). So one can regress this transformed
outcome on `X` and, up to constants, behave like an oracle with access to the counterfactual
contrast — adapting to the smoothness of `tau`. **Gap:** it is *singly* robust (consistent only if
`pi` is correct) and, more painfully, high-variance: dividing by `pi(1 - pi)` explodes wherever the
propensity approaches 0 or 1, which is precisely where overlap is weakest and the data are most
informative. It throws away the outcome models entirely, so it pays full variance for the weighting.

**X-learner.** Künzel et al. (2019) impute pseudo-effects: on the treated units form
`D_i = Y_i - mu_0_hat(X_i)` and regress to get `tau_1_hat`; symmetrically on the controls form
`mu_1_hat(X_i) - Y_i` to get `tau_0_hat`; then blend with the propensity,
`tau_hat(x) = {1 - pi_hat(x)} tau_1_hat(x) + pi_hat(x) tau_0_hat(x)`. This is more efficient than
the T-learner when one arm is much larger, and it does smooth the effect rather than the levels.
**Gap:** it is not doubly robust — the imputation `Y - mu_0_hat` carries the full first-order error
of `mu_0_hat`, so the estimator inherits a first-order (not product) dependence on nuisance error,
and there is no guarantee the CATE error decouples from the nuisance rate.

**R-learner / double-residual regression.** Building on the Robinson decomposition, Nie & Wager
(2021) form an objective from estimated residuals (the "R-loss"),

```
tau_hat = argmin_tau  (1/n) sum_i [ {Y_i - m_hat(X_i)} - {A_i - pi_hat(X_i)} tau(X_i) ]^2 + penalty,
```

minimized over any flexible function class, with the nuisances `m, pi` cross-fit. The construction
is Neyman-orthogonal, so the leading nuisance dependence is second-order, and it adapts to the
complexity of `tau`. **Gap:** the orthogonality here is of the loss, and the published oracle
guarantees require both nuisances to be estimated at `n^{-1/4}` or faster — the same demanding,
ATE-like rate condition — leaving open whether a CATE estimator can be oracle-efficient under
*weaker* conditions when `tau` is smoother than the nuisances, and whether a doubly-robust (rather
than merely orthogonal) error bound is available so that different nuisances may converge at
different rates.

## Evaluation settings

The natural yardsticks are synthetic data-generating processes with a *known* ground-truth effect,
because in real observational data the counterfactual contrast is never observed, so error against
the truth cannot be measured directly.

- **A low-dimensional, strongly-confounded smooth-CATE design.** Covariates `X` uniform on
  `[-1, 1]`; propensity a sharp step in `sign(x)` (heavy confounding); response surfaces an awkward
  non-smooth piecewise polynomial that is *equal* across arms, so the true CATE is constant (zero).
  This is the design that exposes plug-in differencing. Nuisances and second-stage fits via
  smoothing splines.
- **A high-dimensional sparse design.** `X ~ N(0, I_d)` with `d` large (e.g. 500); a logistic
  propensity and logistic outcome surfaces depending on a sparse subset of `alpha` and `beta`
  coordinates respectively, normalized so propensities stay in a moderate overlap range; CATE again
  zero (or simple). Nuisances and second-stage fits via cross-validated lasso. This stresses
  high-dimensional confounding and sparse estimation.
- **Mid-scale tabular designs** with hundreds to a few thousand observations and tens of covariates,
  with nonlinear effects and economic-style outcomes, fit with flexible learners (gradient boosting,
  forests).

Protocol that matters for stability of the conclusions: repeated simulation (many seeds / many
independent test draws), with `K`-fold sample splitting (e.g. 5 folds), and averaging across
splits — so the estimator is judged on its stability across train/test partitions rather than on a
single lucky realization. The error metrics are the **PEHE** (precision in estimation of
heterogeneous effects, the root-mean-square of `tau_hat(x) - tau(x)` over the test covariates) and
the **ATE error** `|mean(tau_hat) - ATE_true|`; both lower-is-better.

## Code framework

The harness already exists: data come as arrays `(X, A, Y)`, scikit-learn-style regressors and
classifiers are available for sub-models, `KFold` provides sample splitting, and numpy/scipy supply
the arithmetic. What is *not* settled is how the sub-model predictions should be combined before the
last regression. So the substrate is a generic split-sample scaffold: fit whatever auxiliary
quantities the estimator needs out-of-fold, create a one-dimensional training target, and fit a
regressor from covariates to that target.

```python
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """Generic two-stage CATE estimator. Stage 1 fits flexible sub-models with
    sample splitting (out-of-fold predictions only). Stage 2 maps X to the
    target constructed from those held-out predictions."""

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))

    # flexible learners available for whatever sub-models the method needs
    def _make_regressor(self):
        return GradientBoostingRegressor(random_state=self._seed)

    def _make_classifier(self):
        return GradientBoostingClassifier(random_state=self._seed)

    def fit(self, X, T, Y):
        n = len(Y)
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)

        for train_idx, val_idx in kf.split(X):
            # Stage 1: fit sub-models on train_idx, predict on the held-out val_idx
            # so each unit's sub-model predictions are independent of that unit.
            # TODO: the auxiliary quantities we will compute from (X, T, Y).
            pass

        # TODO: from the held-out sub-model predictions, form the target the final
        #       regressor will be trained on.
        target = np.zeros(n)  # placeholder

        # Stage 2: regress X -> target to obtain the effect function.
        self._final = self._make_regressor()
        self._final.fit(X, target)
        return self

    def predict(self, X):
        return self._final.predict(X)
```

The empty slot is the construction of the held-out quantities and the scalar target before the final
regression.
