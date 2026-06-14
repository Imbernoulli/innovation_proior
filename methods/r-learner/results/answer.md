# R-learner, distilled

The R-learner estimates the conditional average treatment effect (CATE)
`tau*(x) = E[Y(1) - Y(0) | X = x]` from observational data by a two-step,
loss-minimization procedure. It generalizes Robinson's partially-linear-model
residual-on-residual estimator so the constant slope becomes a function `tau(x)`,
and it cleanly separates two jobs: the *loss* removes confounding, while the
*learner* chosen to minimize the loss expresses the effect heterogeneity. The name
("R") is for **R**obinson and for **r**esidualization.

## Problem it solves

Estimate a flexible, nonlinear `tau*(.)` from observational `(X, W, Y)` (covariates,
binary treatment, outcome) under unconfoundedness `{Y(0),Y(1)} ⫫ W | X` and overlap
`eta < e*(x) < 1-eta`, using any generic machine-learning loss-minimizer, robustly to
errors in the estimated nuisance models, and with a formal guarantee that the effect
estimate's rate depends only on the complexity of `tau*` — not on the complexity of the
confounding.

## Key idea

Let `m*(x) = E[Y|X=x]` be the marginal outcome mean and `e*(x) = P(W=1|X=x)` the
propensity. Under unconfoundedness, `E[Y|X,W] = mu_0*(X) + W tau*(X)` and
`m*(x) = mu_0*(x) + e*(x) tau*(x)`, so

```
E[Y - m*(X) | X, W] = (W - e*(X)) tau*(X).
```

Equivalently, with a conditionally mean-zero residual,

```
Y_i - m*(X_i) = (W_i - e*(X_i)) tau*(X_i) + eps_i,   E[eps_i | X_i, W_i] = 0.
```

This is **Robinson's decomposition** with the constant slope promoted to a function. The
CATE is then the population least-squares projection of the outcome residual on the
treatment residual (the **R-loss**):

```
tau*(.) = argmin_tau  E[ ( (Y - m*(X)) - (W - e*(X)) tau(X) )^2 ].
```

*Why this is exact:* writing `(Y - m*) - (W - e*)tau = (W - e*)(tau* - tau) + eps` and
using `E[eps | X, W] = 0`, the loss equals
`E[(W - e*)^2 (tau - tau*)^2] + E[eps^2]`, whose first term is a nonnegative
overlap-weighted squared distance minimized uniquely at `tau = tau*`. Since
`E[(W-e*)^2|X]=e*(X)(1-e*(X))`, overlap makes the weight strictly positive; a loose
coupling is `(1-eta)^{-2}R(tau) < E[(tau-tau*)^2] < eta^{-2}R(tau)`, where
`R(tau)=L(tau)-L(tau*)`.

## The estimator

Two steps with **cross-fitting** over `Q` folds (typically 5 or 10):

1. Fit nuisance predictors `m-hat(x) = E[Y|X]` and `e-hat(x) = P(W=1|X)` with any
   predictive learners, tuned for accuracy. Use held-out (cross-fit) predictions:
   `m-hat^{(-q(i))}, e-hat^{(-q(i))}`.
2. Minimize the plug-in R-loss with a regularizer:

```
tau-hat = argmin_tau  (1/n) sum_i [ (Y_i - m-hat^{(-q(i))}(X_i)) - (W_i - e-hat^{(-q(i))}(X_i)) tau(X_i) ]^2
                     + Lambda_n(tau).
```

**Weighted-regression form.** Factoring `W-tilde_i = W_i - e-hat(X_i)` out of the square,

```
[ (Y_i - m-hat) - W-tilde_i tau(X_i) ]^2 = W-tilde_i^2 [ (Y_i - m-hat)/W-tilde_i - tau(X_i) ]^2,
```

so the R-loss is a weighted least-squares regression of the **pseudo-outcome**
`(Y_i - m-hat)/W-tilde_i` on `X_i` with **sample weight** `W-tilde_i^2`. Any
sample-weight-aware learner (boosting, ridge, neural net, weighted forest) minimizes it
in one call. This is the U-learner's transformed-outcome regression with the variance-
correct weight restored: `W-tilde^2` downweights exactly the low-overlap points where the
`1/W-tilde` pseudo-outcome has high variance.

## Why it works — quasi-oracle property

Use `Delta_m=m* - m-hat` and `Delta_e=e* - e-hat`. Expanding the feasible-minus-oracle
regret against a reference `tau_ref` leaves five terms:

- `-2/n sum Delta_m Delta_e (tau - tau_ref)`: a **product** term, bounded by
  `(RMSE of m-hat)·(RMSE of e-hat)·||tau - tau_ref||_inf` — second order.
- `1/n sum Delta_e^2 (tau^2 - tau_ref^2)`: a squared-propensity-error term, also second order
  under the bounded-`tau`/RKHS-cap assumptions.
- `-2/n sum (Y-m*) Delta_e (tau-tau_ref)`,
  `-2/n sum (W-e*) Delta_m (tau-tau_ref)`, and
  `+2/n sum (W-e*) Delta_e (tau^2-tau_ref^2)`: single-error cross terms. With cross-fitting,
  the nuisance estimate is fixed relative to the held-out fold, so the conditional means
  factor onto `E[Y-m*(X)|X]=0` or `E[W-e*(X)|X]=0`. They are centered empirical-process
  fluctuations controlled by concentration and chaining, not raw `O(a_n)` bias terms.

Net: `|R-hat_n(tau;c) - R-tilde_n(tau;c)| <= 0.125 R(tau;c) + o(rho_n(c))`. Plugged into the
isomorphic-coordinate-projection ERM bound (Bartlett-type), penalized kernel regression on
the feasible R-loss attains the **same** rate as the oracle that knows the nuisances:

```
R(tau-hat) = Otilde_P( n^{-(1 - 2 alpha)/(p + (1 - 2 alpha))} ),
```

for an RKHS with eigenvalue decay `sigma_j ~ j^{-1/p}` and smoothness `alpha` of `tau*`,
provided each nuisance is `o(n^{-kappa})` with `kappa > 1/4` and overlap holds. As
`alpha, p -> 0` this recovers the classical fourth-root-consistency threshold for
`sqrt(n)` semiparametric inference. The X-learner does not generally have this property: an
`o(n^{-1/4})` shift of its arm models shifts `tau-hat` by the same first order, so its
nuisance dependence is uncancelled.

## Practical knobs and why

- **Cross-fitting (5-10 folds):** out-of-sample residuals; needed both to avoid
  own-observation overfitting and to make the first-order cross terms mean-zero.
- **Clip propensity to `[eta, 1-eta]` (e.g. 0.05/0.95):** enforce practical overlap; the
  pseudo-outcome and the regret-to-error coupling degrade as `e -> 0` or `1`.
- **Weight `(W - e-hat)^2`:** the loss's own weighting; cancels the `1/W-tilde` variance
  blowup of the pseudo-outcome.
- **Bound/regularize the final `tau` learner:** the theory caps `||tau||_inf <= 2M` to rule
  out pathological minimizers; practical implementations enforce the analogue through the
  final learner and its regularization without changing the R-loss.
- **Strong predictive learners for nuisances, generic learner for `tau`:** nuisances only
  need `o(n^{-1/4})` predictive accuracy; the `tau`-model need only minimize the R-loss
  (tunable by cross-validating on the R-loss), with no need to audit it for confounding.

## Working code

A minimal Python rendition of the canonical weighted-regression implementation:

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """R-learner (Robinson decomposition). Cross-fit m(X)=E[Y|X] and e(X)=P(W=1|X),
    residualize, then minimize the R-loss as a weighted regression of the
    pseudo-outcome (Y - m_hat)/(W - e_hat) with weight (W - e_hat)^2."""

    def __init__(self, n_folds=5, seed=42, eta=0.05):
        self.n_folds = n_folds
        self.seed = seed
        self.eta = eta

    def _make_regressor(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed,
        )

    def _make_classifier(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed + 1,
        )

    def fit(self, X, W, Y):
        X, W, Y = np.asarray(X), np.asarray(W), np.asarray(Y)
        n = len(Y)

        # Cross-fit nuisances: held-out predictions for out-of-sample residuals.
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)
        for tr, va in kf.split(X):
            my = self._make_regressor(); my.fit(X[tr], Y[tr])
            m_hat[va] = my.predict(X[va])
            mw = self._make_classifier(); mw.fit(X[tr], W[tr])
            e_hat[va] = mw.predict_proba(X[va])[:, 1]

        e_hat = np.clip(e_hat, self.eta, 1 - self.eta)

        Y_tilde = Y - m_hat                        # outcome residual using m_hat
        W_tilde = W - e_hat                        # treatment residual using e_hat

        weights = W_tilde ** 2                     # exact R-loss weights
        pseudo = Y_tilde / W_tilde                 # exact weighted-regression target

        # Minimize the R-loss as a generic weighted regression.
        self._cate_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed + 2,
        )
        self._cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        return self._cate_model.predict(np.asarray(X))
```

Equivalent canonical forms in the `rlearner` R package are `rboost`, which uses
`y_tilde/w_tilde` with weights `w_tilde^2` in weighted XGBoost, and `rlasso`, which
reparameterizes the same loss as a linear weighted least squares problem with design
`(W - e-hat)·[1, X]`.
