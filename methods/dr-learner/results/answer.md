# DR-Learner, distilled

The DR-Learner (Doubly Robust Learner) estimates the conditional average treatment effect
`tau(x) = E(Y^1 - Y^0 | X = x)` from observational data in two stages. Stage one cross-fits three
flexible nuisance models — the two outcome regressions `mu_a(x) = E(Y|X,A=a)` and the propensity
`pi(x) = P(A=1|X)` — and uses them to build a per-unit **doubly-robust (augmented-IPW)
pseudo-outcome** whose conditional mean is `tau(x)`. Stage two simply **regresses that pseudo-outcome
on `X`**. It is the function-valued analog of the AIPW estimator of the average treatment effect:
where AIPW *averages* the efficient influence function to get an efficient ATE, the DR-Learner
*regresses* it on covariates to get the CATE.

## Problem it solves

Estimating a heterogeneous treatment effect under confounding, where (a) naive
treated-minus-control comparisons are biased, (b) the effect `tau` may be far simpler (smoother /
sparser) than the individual response surfaces `mu_a`, so differencing the surfaces (the T-learner)
inherits their error, and (c) the nuisance functions can only be estimated at slow nonparametric
rates. The goal is a method-agnostic CATE estimator that converges at the complexity of `tau`
itself and stays consistent if *either* the outcome models or the propensity is correct.

## Key idea

Form the per-unit pseudo-outcome (with `mu_A = mu_1` if `A=1` else `mu_0`):

```
phi_hat(Z) = (A - pi_hat(X)) / [pi_hat(X)(1 - pi_hat(X))] · {Y - mu_A_hat(X)} + mu_1_hat(X) - mu_0_hat(X)
           = mu_1_hat(X) - mu_0_hat(X)
             + A (Y - mu_1_hat(X)) / pi_hat(X)
             - (1 - A) (Y - mu_0_hat(X)) / (1 - pi_hat(X)).
```

(The two written forms are algebraically identical for binary `A`.) It is a regression-difference
`mu_1 - mu_0` augmented by an inverse-probability-weighted correction on each arm's residual.

- **Oracle unbiasedness.** With true nuisances, `E(phi | X=x) = tau(x)`: the correction term has
  conditional mean zero on each arm, leaving `mu_1 - mu_0 = tau`.
- **Double robustness via product bias.** With *estimated* nuisances, the conditional bias of the
  pseudo-outcome is

  ```
  b_hat(x) = E(phi_hat - phi | nuisances, X=x)
           = sum_{a=0}^{1} { (pi_hat(x) - pi(x)) / [a·pi_hat(x) + (1-a)(1 - pi_hat(x))] } · (mu_a_hat(x) - mu_a(x)),
  ```

  a per-arm **product** of the propensity error and the outcome error. Hence `b_hat ≡ 0` if
  `pi_hat = pi` (any `mu_a_hat`) or if `mu_a_hat = mu_a` (any `pi_hat`). This is the CATE-level echo
  of the AIPW second-order remainder for the ATE.
- **Cross-fitting is required**, not optional: it makes the nuisances independent of the units they
  are evaluated on, so `b_hat` is the true second-stage bias and the empirical-process contamination
  has conditional mean zero — with no Donsker/complexity restriction on the (flexible) learners.
- **Regress, don't difference.** Because `phi_hat` already targets `tau`, the final stage is a plain
  MSE regression of `phi_hat` on `X`; it recovers `tau` at `tau`'s own complexity.

## Algorithm

Let `(D_1, D_2)` be two independent samples (or `K` folds).

1. **Nuisance training (Stage 1).** On `D_1`, fit `pi_hat`, and fit `mu_0_hat`, `mu_1_hat`
   separately on the control and treated arms. Any flexible learner is allowed.
2. **Pseudo-outcome regression (Stage 2).** On `D_2`, form `phi_hat(Z)` from the `D_1`-trained
   nuisances and regress it on `X`: `tau_hat(x) = Ehat_n{phi_hat(Z) | X=x}`.
3. **Cross-fitting.** Swap the roles of `D_1, D_2` (or rotate over `K` folds), and average the
   resulting estimators.

## Guarantees

Let `tau_tilde(x) = Ehat_n{phi(Z)|X=x}` be the oracle that regresses the *true* pseudo-outcome,
with risk `R_n*(x)`. If the second-stage regressor is a **stable** linear smoother (local
polynomials, series, kernels, smoothing splines, k-NN, kernel ridge, forests-as-smoothers) and the
nuisances are consistent, then

```
tau_hat(x) - tau_tilde(x) = Ehat_n{b_hat(X) | X=x} + o_P(R_n*(x)),
```

so the DR-Learner is **oracle efficient** when the *smoothed* product-bias is `o_P(R_n*)`. Under
Hölder smoothness — `pi` is `alpha`-smooth, `mu_a` is `beta`-smooth, `tau` is `gamma`-smooth, with
a minimax-optimal smoother whose weights satisfy `sum_i |w_i| = O_P(1)` —

```
tau_hat(x) - tau(x) = O_P( n^{-1/(2 + d/gamma)} + n^{-(1/(2 + d/alpha) + 1/(2 + d/beta))} ),
```

the oracle (CATE-complexity) rate plus a product penalty. Oracle efficiency holds iff

```
sqrt(alpha · beta) >= (d/2) / sqrt( 1 + (d/gamma)(1 + d/(2·s_bar)) ),    s_bar = 2/(1/alpha + 1/beta).
```

As `gamma -> infinity` this recovers the ATE condition `sqrt(alpha·beta) >= d/2`; for finite `gamma`
the bar is *lowered*, because the oracle CATE rate is slower than root-`n`, so the nuisances may be
rougher than the ATE would tolerate and the estimator still hits the oracle.

## Practical guards

- **Clip the propensity** into `[eps, 1 - eps]` (e.g. `eps = 0.05`) to enforce overlap and keep the
  inverse weights bounded — a small local bias for bounded variance.
- **Winsorize the pseudo-outcome** at a high quantile of `|phi_hat|` (e.g. 97th percentile) so heavy
  residual tails near weak overlap do not dominate the second-stage fit.
- **Learners.** Flexible nuisance models (gradient boosting / forests / lasso); a *shallower, more
  regularized* final regressor, since `tau` is bet to be the simplest function in play.

## Relation to prior methods

- **T-learner (plug-in)** = the DR-Learner with the augmentation correction dropped (`mu_1 - mu_0`
  only): first-order in the outcome error, not doubly robust.
- **IPW pseudo-outcome regression** = the DR-Learner with the outcome models dropped
  (`mu_a_hat ≡ 0`): `(A - pi)Y / [pi(1-pi)]` has conditional mean `tau` but is singly robust and
  high-variance near weak overlap; the `mu_a` augmentation subtracts the predictable part of `Y`
  before weighting, reducing that variance.
- **X-learner** smooths effects rather than levels but is not doubly robust (its imputations carry
  the full first-order error of an outcome model).
- **R-learner / double-residual regression** is Neyman-orthogonal but its published oracle
  guarantees require both nuisances at `n^{-1/4}`; the DR-Learner reaches the oracle under the weaker
  (lowered) bar above and ends in a plain MSE regression rather than an `(A - pi)`-weighted least
  squares.

## Working code

```python
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """Cross-fitted AIPW pseudo-outcome regressed on X.

    Stage 1 cross-fits nuisance models:
        mu0(x) = E[Y|X, T=0], mu1(x) = E[Y|X, T=1]   (outcome models)
        e(x)   = P(T=1|X)                            (propensity)
    Stage 2 forms the doubly-robust pseudo-outcome and regresses it on X:
        phi = mu1 - mu0 + T(Y - mu1)/e - (1-T)(Y - mu0)/(1-e)
    Consistent if either the outcome models or the propensity model is correct.
    """

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))

    def _make_model_y(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )

    def _make_model_t(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
        )

    def _make_cate_model(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )

    def fit(self, X, T, Y):
        n = len(Y)

        # --- Stage 1: cross-fit nuisances (train out-of-fold, predict in-fold) ---
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            mask0 = T[train_idx] == 0
            mask1 = T[train_idx] == 1

            m0 = self._make_model_y()
            m1 = self._make_model_y()

            if mask0.sum() > 5:
                m0.fit(X[train_idx[mask0]], Y[train_idx[mask0]])
                mu0_hat[val_idx] = m0.predict(X[val_idx])
            else:
                mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()

            if mask1.sum() > 5:
                m1.fit(X[train_idx[mask1]], Y[train_idx[mask1]])
                mu1_hat[val_idx] = m1.predict(X[val_idx])
            else:
                mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()

            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # enforce overlap so the inverse weights stay bounded
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # --- Stage 2: doubly-robust pseudo-outcome, then regress on X ---
        pseudo = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e_hat
            - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
        )

        # winsorize extreme pseudo-outcomes
        q = np.percentile(np.abs(pseudo), 97)
        pseudo = np.clip(pseudo, -q, q)

        self._cate_model = self._make_cate_model()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X):
        return self._cate_model.predict(X)
```
