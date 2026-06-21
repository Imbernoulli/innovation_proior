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

Three features characterize the setting. (1) **Confounding**: `A` depends on `X`, so an estimator
adjusts for `X` either through the outcome models `mu_a` or through the propensity `pi`.
(2) **Heterogeneity**: `tau(x)` can vary across the covariate space in complex, nonlinear ways, so
the target is a flexible, nonparametric estimate of the *whole function*, not a single scalar.
(3) **Differential complexity**: the individual response surfaces `mu_a` can be complicated, yet
their *difference* `tau` can be far simpler — even constant or zero.

We want a CATE estimator that is **flexible** (usable with any modern machine-learning regressor for
the various sub-models) and **robust to confounding**, in a regime where the nuisance functions are
estimated by machine learning at slower-than-`sqrt(n)` rates.

## Background

**The identification and the nuisances.** Everything downstream is built from three nuisance
functions of `X`: the propensity `pi(x) = P(A=1|X=x)`, the two arm-specific outcome regressions
`mu_a(x) = E(Y|X=x,A=a)`, and sometimes the marginal outcome regression
`eta(x) = E(Y|X=x) = pi(x) mu_1(x) + (1-pi(x)) mu_0(x)`. The CATE is the contrast
`mu_1 - mu_0`; the average treatment effect (ATE) is its mean, `E{tau(X)}`.

**A reference design.** Consider a one-dimensional setup where the treatment is strongly confounded
— say `pi(x) = 0.5 + 0.4 sign(x)`, so the left half of the covariate space is mostly untreated and
the right half mostly treated — while the two response surfaces are *equal*, `mu_1 = mu_0`, and are
an awkward, non-smooth piecewise-polynomial function. The true CATE is then exactly the constant
zero, the simplest possible function, while the individual surfaces are difficult to estimate.
The difficulty of `tau` can be *much* lower than the difficulty of the `mu_a`.

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

The remainder is a *product* of the propensity error and the outcome error: each individual error
can be large (slow nonparametric rate) yet the product can be small. This is stated, in the
classical theory, for the *scalar* functional.

**Regularization bias and plug-in machine learning.** Chernozhukov, Chetverikov, Demirer, Duflo,
Hansen, Newey & Robins (2018) made the practical mechanism precise. If a nuisance `g_0` is estimated
with a regularized machine learner (lasso, forests, boosting, nets — all of which trade variance for
bias) and plugged into an estimating equation for the target, the scaled error picks up a term like
`(1/sqrt(n)) sum_i m_0(X_i){g_0(X_i) - ghat_0(X_i)}` whose summands do *not* have mean zero; with a
nuisance rate `n^{-phi_g}` and `phi_g < 1/2`, this term is of order `sqrt(n) n^{-phi_g} -> infinity`,
so the naive plug-in is not `sqrt(n)`-consistent. Two ingredients address this.
**Orthogonalization**: reformulate the estimating equation so it is *Neyman orthogonal* — first-order
insensitive to nuisance perturbations — which makes the leading bias the *product*
`(mhat_0 - m_0)(ghat_0 - g_0)`, of order `sqrt(n) n^{-(phi_m + phi_g)}`, which can vanish even when
each nuisance converges slowly. **Cross-fitting**: estimate the nuisances on one part of the data and
evaluate the target on an independent part, then swap and average; this removes the empirical-process
/ own-observation coupling that would otherwise reintroduce a non-vanishing remainder, with no
Donsker-class restriction on the (highly complex) ML estimators. All of this is developed for
`sqrt(n)`-estimable *scalar* parameters.

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
rate set by the smoothness/sparsity of `tau` *alone* — the complexity of `tau`, not of the `mu_a`.
That oracle rate is the natural benchmark for a CATE estimator.

## Baselines

These are the prior CATE estimators a new method is measured against.

**S-learner (single model).** Fit one regression `mu(x, a)` of `Y` on `(X, A)` jointly, then set
`tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0)`. Simple and data-efficient; the learner treats `A` as one
feature among many, and regularization acts on the fitted `A`-dependence.

**T-learner (two models / plug-in).** Fit `mu_1` on the treated and `mu_0` on the controls
separately, and difference them: `tau_hat = mu_1_hat - mu_0_hat` (Künzel, Sekhon, Bickel & Yu 2019
name this the T-learner). This is the natural plug-in for `tau = mu_1 - mu_0`: each arm is fit on its
own sub-sample, and the two fitted surfaces are subtracted.

**Inverse-probability weighting (IPW) of a transformed outcome.** With a known or estimated
propensity, the weighted transform `(A - pi(X)) Y / {pi(X)(1 - pi(X))}` has conditional mean
exactly `tau(x)` (Athey & Imbens-style weighting; Tian et al.). One regresses this transformed
outcome on `X`, behaving up to constants like an oracle with access to the counterfactual contrast
and adapting to the smoothness of `tau`. It is consistent when `pi` is correct, and uses the
propensity alone, weighting `Y` by `1/{pi(1 - pi)}`.

**X-learner.** Künzel et al. (2019) impute pseudo-effects: on the treated units form
`D_i = Y_i - mu_0_hat(X_i)` and regress to get `tau_1_hat`; symmetrically on the controls form
`mu_1_hat(X_i) - Y_i` to get `tau_0_hat`; then blend with the propensity,
`tau_hat(x) = {1 - pi_hat(x)} tau_1_hat(x) + pi_hat(x) tau_0_hat(x)`. It is more efficient than the
T-learner when one arm is much larger, and it smooths the effect rather than the levels.

**R-learner / double-residual regression.** Building on the Robinson decomposition, Nie & Wager
(2021) form an objective from estimated residuals (the "R-loss"),

```
tau_hat = argmin_tau  (1/n) sum_i [ {Y_i - m_hat(X_i)} - {A_i - pi_hat(X_i)} tau(X_i) ]^2 + penalty,
```

minimized over any flexible function class, with the nuisances `m, pi` cross-fit. The construction
is Neyman-orthogonal, so the leading nuisance dependence is second-order, and it adapts to the
complexity of `tau`. The published oracle guarantees require both nuisances to be estimated at
`n^{-1/4}` or faster.

## Evaluation settings

The natural yardsticks are synthetic data-generating processes with a *known* ground-truth effect,
because in real observational data the counterfactual contrast is never observed, so error against
the truth cannot be measured directly.

- **A low-dimensional, strongly-confounded smooth-CATE design.** Covariates `X` uniform on
  `[-1, 1]`; propensity a sharp step in `sign(x)` (heavy confounding); response surfaces an awkward
  non-smooth piecewise polynomial that is *equal* across arms, so the true CATE is constant (zero).
  Nuisances and second-stage fits via smoothing splines.
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
