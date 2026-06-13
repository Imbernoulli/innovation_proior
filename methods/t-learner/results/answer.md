# T-learner, distilled

The T-learner ("T" = two) is a meta-algorithm for estimating the Conditional Average Treatment
Effect (CATE) that turns any supervised regression method into a CATE estimator. It fits two
separate outcome models — one on the control arm, one on the treated arm — and estimates the
effect as their difference. The causal content is entirely in splitting by treatment arm and
subtracting; the per-arm estimation is delegated to any off-the-shelf learner.

## Problem it solves

From observational data `(X_i, W_i, Y^obs_i)` — covariates, binary treatment, single observed
outcome (`Y^obs = Y(W)`) — estimate `tau(x) = E[Y(1) - Y(0) | X = x]`, the effect of the
treatment specific to units with covariates `x`. The per-unit effect `D = Y(1) - Y(0)` is never
observed and is not identifiable; for a fresh target unit, the conditional mean `tau` is the
MSE-optimal predictor of `D`, with the remaining error irreducible.

## Assumptions

- **Ignorability / unconfoundedness:** `(eps(0), eps(1)) ⟂ W | X` where `Y(w) = mu_w(X) + eps(w)`.
- **Overlap / positivity:** `0 < e_min < e(x) < e_max < 1`, `e(x) = P(W=1 | X=x)`.

Under these, the observed within-arm conditional means coincide with the potential-outcome
surfaces: `E[Y^obs | X=x, W=1] = mu_1(x)` and `E[Y^obs | X=x, W=0] = mu_0(x)`.

## Key idea

Estimate the two response surfaces separately, with any (possibly different) base learner per
arm, then difference:

```
mu0_hat = regress Y on X over control rows  {(X_i, Y_i): W_i = 0}
mu1_hat = regress Y on X over treated rows  {(X_i, Y_i): W_i = 1}
tau_hat_T(x) = mu1_hat(x) - mu0_hat(x)
```

- **Why two models, not one pooled model with W as a feature (the S-approach).** A single model
  given `W` as just another covariate may downweight or ignore it — a forest can fail to split on
  a weakly-predictive `W`, shrinking the estimated effect toward 0. Separating the arms prevents
  that particular collapse. Cost: no pooling, so shared
  structure across arms must be relearned in each arm, and the full sampling cost is paid per arm
  — worst in unbalanced designs (one arm much smaller).
- **Match base-learner complexity to each arm's own sample size**, and use a different learner per
  arm if the surfaces differ in complexity. This freedom is the point of a meta-algorithm.

## Convergence rate

Place the data-generating distribution in `S(a_mu, a_tau)`: ignorability holds, each response
surface family is in `S(a_mu)` (minimax regression rate `N^{-a_mu}`), and the effect-imputation
families are in `S(a_tau)`. Then with the minimax base learner in each arm,

```
EMSE(tau_hat_T) <= 2C [ (1 - e_min)/(1 - e_max) + e_max/e_min ] ( n^{-a_mu} + m^{-a_mu} ),
```

with `n` = #treated, `m` = #control. Proof:

1. `(a-b)^2 <= 2a^2 + 2b^2` gives
   `(tau_hat - tau)^2 <= 2(mu1_hat - mu1)^2 + 2(mu0_hat - mu0)^2`; define `A` and `B` as the
   treated- and control-surface EMSEs before this outer factor 2, so `EMSE(tau_hat) <= 2A + 2B`.
2. Conditioning on `sum_i W_i = n` makes rows dependent; condition instead on the full assignment
   vector `W = w`. Then the treated tuples `(X_i, Y_i)_{W_i=1}` are i.i.d. from `P_1 = law of
   (X,Y) | W=1` (independence lemma + identical-distribution lemma, both from mutual independence
   of the original i.i.d. sample).
3. `mu1_hat` uses only the `n` i.i.d. treated rows, so its EMSE under `P_1`'s covariate law is
   `O(n^{-a_mu})`.
4. Change of measure (uses overlap): for positive `g`,
   `(e_min/e_max) E[g(X)] <= E[g(X)|W=1] <= (e_max/e_min) E[g(X)]` (and the `1-e` analogue for
   `W=0`), since `E[g(X)|W=1] = E[g(X)e(X)]/E[W]` and both `e(X)` and `E[W]=E[e(X)]` lie between
   `e_min` and `e_max`. Converts the
   error from `P_1`'s covariate law to the target marginal at the cost of the propensity-ratio
   constant.
5. Hence `A <= (e_max/e_min) C n^{-a_mu}` and
   `B <= ((1-e_min)/(1-e_max)) C m^{-a_mu}`. Therefore
   `EMSE(tau_hat) <= 2C(e_max/e_min)n^{-a_mu} + 2C((1-e_min)/(1-e_max))m^{-a_mu}`, which is bounded
   by the displayed clean constant times `n^{-a_mu} + m^{-a_mu}`.

Concrete instance: Lipschitz response surfaces in `d` dimensions give `a_mu = 2/(2+d)`; with
`k`-NN base learners (`k_1 ~ n^{2/(2+d)}`, `k_0 ~ m^{2/(2+d)}`), using the bias-variance split
`E||mu_hat - mu||^2 <= sigma^2/k + c L^2 (k/N)^{2/d}`, the T-learner attains the minimax-optimal
`O(n^{-2/(2+d)} + m^{-2/(2+d)})`.

**Ceiling.** The general guarantee is the response-surface rate `a_mu` in each arm, not the
possibly faster effect rate `a_tau`: because the two arms are estimated in total isolation, the
procedure has no general mechanism for exploiting a `tau` that is simpler than the surfaces (e.g.
complicated surfaces with a constant or linear difference). This is most visible when one arm is
much larger than the other.

## Working code

Fills the `CATEEstimator` contract (`fit(X, T, Y) -> self`, `predict(X) -> tau_hat`) with two
per-arm regressors and a difference. A gradient-boosting regressor is used as a concrete base
learner, but any sklearn-style regressor drops in (a different one per arm is allowed).

```python
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor


def make_base_regressor(seed):
    return GradientBoostingRegressor(random_state=seed)


class CATEEstimator:
    """T-learner: two outcome models, CATE = mu1_hat - mu0_hat."""

    def __init__(self, base_regressor=None):
        self._base_regressor = (
            make_base_regressor(seed=42) if base_regressor is None else base_regressor
        )
        self._models = {}

    def fit(self, X, T, Y):
        X, T, Y = np.asarray(X), np.asarray(T), np.asarray(Y)
        mask0 = T == 0                      # control rows -> mu0
        mask1 = T == 1                      # treated rows -> mu1
        self._models[0] = clone(self._base_regressor)
        self._models[1] = clone(self._base_regressor)
        self._models[0].fit(X[mask0], Y[mask0])
        self._models[1].fit(X[mask1], Y[mask1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self._models[1].predict(X) - self._models[0].predict(X)
```

Library form (e.g. causalml / EconML): deep-copy or clone a base regressor into separate control
and treatment models, fit each on its arm, and return `model_t.predict(X) - model_c.predict(X)`.
