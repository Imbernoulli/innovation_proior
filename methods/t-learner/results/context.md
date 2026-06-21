# Context: estimating heterogeneous treatment effects from observational data

## Research question

We have a sample of units, and for each we record a covariate vector `X in R^d`, a binary
treatment assignment `W in {0,1}`, and a scalar outcome `Y`. We want to know, for a *new* unit
with covariates `x`, how much the treatment would change its outcome — not on average over the
whole population, but specifically for units that look like `x`. Formally the quantity of
interest is the Conditional Average Treatment Effect

```
tau(x) = E[ Y(1) - Y(0) | X = x ],
```

where `Y(1)` and `Y(0)` are the two potential outcomes of a unit (its outcome if treated and if
left untreated). This is the function a policymaker, clinician, or platform needs to decide
*whom* to treat, as opposed to *whether* the treatment works on average.

Two facts make this hard. First, the *fundamental problem of causal inference*: for any single
unit we only ever observe one of the two potential outcomes — the one corresponding to the
treatment it actually received, `Y^obs = Y(W)` — never both, so the unit-level difference
`D = Y(1) - Y(0)` is never directly seen and, without strong assumptions, is not even
identifiable. Second, in observational data the treatment is not randomized: which units get
treated depends on their covariates, so a naive comparison of treated and untreated outcomes
confounds the treatment effect with the differences between who-gets-treated and who-doesn't.
On top of these, `tau` can vary across the covariate space in arbitrarily complex, nonlinear
ways, and we do not know its functional form in advance.

The question is how to estimate `tau(x)` from observational `(X, W, Y^obs)` triples using
flexible supervised regression methods.

## Background

**The potential-outcome framework.** Following Neyman (1923) and Rubin (1974), we posit a
super-population distribution `P` from which `(Y_i(0), Y_i(1), X_i, W_i)` are drawn i.i.d. The
*response surfaces* are the conditional means of each potential outcome,

```
mu_0(x) = E[Y(0) | X = x],     mu_1(x) = E[Y(1) | X = x],
```

and `tau(x) = mu_1(x) - mu_0(x)` whenever both means are identified. The Average Treatment
Effect `ATE = E[Y(1) - Y(0)]` is the population average of `tau`. We write `Y^obs_i = Y_i(W_i)`
for the single outcome we actually observe.

**Why the unit effect is unidentifiable, and why the conditional mean is the right target.**
Consider a one-dimensional `X ~ Unif[0,1]`, an independent fair-coin treatment `W ~ Bern(0.5)`,
and a Rademacher control outcome `P(Y(0)=1) = P(Y(0)=-1) = 0.5`. Compare two data-generating
processes: in the first `Y(1) = Y(0)` (so every unit effect `D_i = 0`); in the second
`Y(1) = -Y(0)` (so every `D_i in {-2, 2}`). The two processes induce *exactly the same
distribution* on the observable triple `(Y^obs, X, W)`, yet have different unit effects, so no
estimator built from observed data can recover `D_i`. The conditional mean, however, is `tau(x)
= 0` in both, and is consistently estimable. There is also a clean reason the conditional mean
is what one should aim at even if one cares about the unit effect: for any estimator `tauhat`,

```
E[(D_i - tauhat(x))^2 | X_i = x]
  = E[(D_i - tau(x))^2 | X_i = x] + E[(tau(x) - tauhat(x))^2],
```

where the second expectation is over the randomness of the fitted estimator, independent of the
fresh target unit. The first term does not depend on the estimator, so minimizing the
mean-squared error for the unit effect is the same as minimizing it for the CATE.

**Confounding and the propensity score.** The treatment indicator is generated as
`W ~ Bern(e(X))`, where `e(x) = P(W = 1 | X = x)` is the *propensity score* (Rosenbaum & Rubin
1983). When `e` depends on `X`, treated and control groups have different covariate
distributions, and raw outcome differences are biased. Two standard conditions tame this. The
first is *ignorability* (unconfoundedness): conditional on the covariates, the potential
outcomes are independent of treatment assignment,

```
( eps(0), eps(1) ) ⟂ W | X,         where  Y(w) = mu_w(X) + eps(w).
```

The second is *overlap* (positivity): there exist `e_min, e_max` with
`0 < e_min < e(x) < e_max < 1` for all `x` in the support. Ignorability says that within a
covariate cell the treated and control units are exchangeable; overlap says every covariate
cell contains some of both, so both response surfaces are identified everywhere. Under
ignorability the observed conditional means coincide with the potential-outcome means within
each arm: `E[Y^obs | X = x, W = 1] = mu_1(x)` and `E[Y^obs | X = x, W = 0] = mu_0(x)`.

**Minimax nonparametric regression — the rate language.** For ordinary regression `mu(x) =
E[Y|X=x]` from `N` i.i.d. samples, performance is measured by the Expected Mean Squared Error
`EMSE(P, muhat) = E[(muhat(X) - mu(X))^2]`. For a single fixed `P` there is always a trivial
zero-error estimator (`muhat = mu`), so one studies the worst case over a *family* `F` of
distributions (Stone 1982; Gyorfi et al. 2006; Tsybakov 2009). Write `F in S(a)` if `F` admits
an estimator with `sup_{P in F} EMSE(P, muhat_N) <= C N^{-a}`. Parametric families (e.g. linear
`Y = beta X + eps` with OLS) sit in `S(1)`, the `N^{-1}` parametric rate. Lipschitz-continuous
regression functions over `[0,1]^d` sit at the slower rate `N^{-2/(2+d)}` — for `d = 1` that is
`N^{-2/3}` — and this is attained by `k`-nearest-neighbors or Nadaraya-Watson with the right
bandwidth. A concrete handle used throughout: for a `k`-NN estimator of a `L`-Lipschitz `mu` on
`Unif([0,1]^d)` with noise variance `sigma^2`,

```
E[ || muhat_N - mu ||^2 ] <= sigma^2 / k  +  c L^2 (k/N)^{2/d},
```

a bias-variance split (variance `~ sigma^2/k` from averaging `k` neighbors, bias `~ L^2 (k/N)^{2/d}`
from the neighbors' distance), minimized at `k ~ N^{2/(2+d)}` to give the `N^{-2/(2+d)}` rate.

## Baselines

**Single pooled response model (the "S" approach), with BART or regression trees (Hill 2011;
Green & Kern 2012; Athey & Imbens 2016).** Treat the treatment indicator as one more feature and
fit a single regression of `Y^obs` on `(X, W)`,

```
muhat(x, w) ~ E[Y^obs | X = x, W = w],     tauhat_S(x) = muhat(x, 1) - muhat(x, 0).
```

One model, all the data; the learner decides on its own how much the outcome depends on `W`.

**Virtual-twins / predicted-twin differencing (Foster, Taylor & Ruberg 2011; Foster 2013).**
Fit a model (originally a random forest, also studied with linear regression) and, for each
unit, predict the outcome of its hypothetical treated "twin" and control "twin," then difference
them per unit and regress or threshold that difference to find subgroups with enhanced effect.
This makes the heterogeneous-effect question concrete as a difference of predicted potential
outcomes.

**Recursive partitioning for causal effects / causal trees (Athey & Imbens 2016).** Build a
single tree whose leaves are chosen to capture treatment-effect heterogeneity: modify the CART
splitting criterion so it targets the mean-squared error of the *effect* within a leaf rather
than of the outcome, and use *honest* estimation — one subsample to choose the partition, an
independent subsample to estimate the leaf-level effects — so the leaf effect estimates are
unbiased and admit valid confidence intervals even with many covariates. The leaf effect is the
treated-minus-control mean within the leaf.

**Transformed-outcome regression (Athey & Imbens 2016; Tian et al. 2014).** Build a single
re-weighted/transformed response `Y^*` (using the propensity score) whose conditional mean is
`tau(x)`, then regress `Y^*` on `X` directly.

## Evaluation settings

The natural yardsticks are of two kinds. *Theoretically*, the minimax convergence-rate
framework above: place the data-generating process in a family with known regression rate (e.g.
linear response surfaces in `S(1)`, Lipschitz response surfaces at `N^{-2/(2+d)}`) and ask at
what rate an estimator's EMSE for `tau` decays in the per-arm sample sizes `n` (treated) and `m`
(control). *Empirically*, simulation studies with a known ground-truth `tau`, designed to span
the regimes that stress different estimators: a zero treatment effect everywhere (so pooling
should help); response surfaces with little common structure and a complex effect (so pooling can
be costly); designs with and without confounding; and balanced versus highly unbalanced arm sizes.
Base learners used inside the meta-procedures include random forests, BART, regression trees, and
linear models. Metrics are the root-mean-squared error of the estimated effect against the true
effect over a test sample (the precision in estimating heterogeneous effects), and the absolute
error of the implied average effect. Real field experiments serve as application settings: a
get-out-the-vote social-pressure mailing experiment (Gerber, Green & Larimer 2008) and a
door-to-door canvassing experiment on reducing prejudice (Broockman & Kalla 2016), where control
records vastly outnumber treated ones.

## Code framework

The estimator must expose the standard supervised-learning contract specialized to causal data:
a `fit` that consumes observational covariates, binary treatment, and outcome, and a `predict`
that returns one estimated treatment effect per row. The substrate that already exists is a
library of off-the-shelf regressors (`scikit-learn` estimators, gradient boosting, forests),
`numpy`/`scipy` for array handling, and the boolean-masking idiom for splitting a sample by a
condition. What is *not* settled is how to turn the three observational arrays into a
conditional-effect prediction — that mapping is exactly the slot to design.

```python
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor


def make_base_regressor(seed):
    """A generic off-the-shelf regressor; any sklearn-style estimator would do."""
    return GradientBoostingRegressor(random_state=seed)


class CATEEstimator:
    """Estimate per-row conditional treatment effects tau_hat(x) from observational
    (X, T, Y). T is a binary treatment indicator. Any supervised regressor
    is available as a building block; how they are combined into a CATE
    estimate is the open design."""

    def __init__(self, base_regressor=None):
        # TODO: the components we will define here.
        pass

    def fit(self, X, T, Y):
        """Learn from observational covariates X, binary treatment T, outcome Y."""
        # TODO: the estimation procedure we will design — how to use the
        #       observed (X, T, Y) to learn something that yields tau_hat.
        return self

    def predict(self, X):
        """Return predicted conditional treatment effects tau_hat for each row of X."""
        # TODO: produce the conditional effect from whatever fit() learned.
        pass
```
