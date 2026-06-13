# S-learner, distilled

The S-learner ("single") is a meta-algorithm for estimating the Conditional Average
Treatment Effect (CATE) `tau(x) = E[Y(1) - Y(0) | X = x]`. It fits **one** off-the-shelf
supervised regressor on the **pooled** data, treating the binary treatment indicator `W`
as just another feature alongside the covariates `X`, then reads the effect off by toggling
that one feature:

```
mu_hat(x, w) = a single model of E[Y^obs | X = x, W = w], fit on all N rows;
tau_hat_S(x) = mu_hat(x, 1) - mu_hat(x, 0).
```

The base learner is arbitrary (random forest, BART, gradient boosting, neural net, linear);
the causal content is entirely in *pooling and toggling*, not in the learner.

## Problem it solves

From observational data `(X_i, W_i, Y_i)` — covariates, binary treatment, one observed
outcome per unit — recover the whole effect surface `tau(x)`, reusing the mature, trusted
toolbox of supervised regression instead of building a bespoke causal estimator. Two
obstacles: only one of each unit's two potential outcomes is ever seen (the
fundamental problem of causal inference), and treatment assignment may depend on `X`
(confounding).

## Why this is valid (identification)

- **Potential outcomes** (Neyman–Rubin): `Y_i^obs = Y_i(W_i)`; `mu_w(x) = E[Y(w)|X=x]`.
- The individual effect `D_i = Y_i(1) - Y_i(0)` is **not identifiable** (two DGPs can match
  the observed-data law yet differ in `D_i`); the achievable target is `tau(x) = E[D|X=x]`.
  For a fresh-unit prediction, conditioning on the fitted estimator gives
  `E[(D_i - tau_hat(x))^2 | X_i=x, tau_hat] = E[(D_i - tau(x))^2 | X_i=x] + (tau(x) - tau_hat(x))^2`,
  so the best ITE estimator is the best CATE estimator. Metric of record:
  `EMSE = E[(tau(X) - tau_hat(X))^2]`.
- **Ignorability** `(Y(0),Y(1)) ⟂ W | X` (Rosenbaum & Rubin 1983) gives the bridge
  `E[Y^obs | X=x, W=w] = mu_w(x)` — the observable conditional mean equals the
  counterfactual response surface. **Overlap** `0 < e_min < e(x) < e_max < 1` makes both
  arms estimable everywhere. Hence `tau(x) = mu(x,1) - mu(x,0)`.

## Key idea

Do **not** split the data into arms. Pool all rows, hand the learner the treatment flag as
an ordinary feature, fit one surface `mu_hat(x,w)`, and difference predictions across the
flag. Versus the two-model recipe (fit `mu_0`, `mu_1` separately and subtract — the
T-learner of Foster 2013 / Athey–Imbens 2015), the single model:

- **shares statistical strength** across arms — the shared baseline dependence on `x` is
  modeled once with all `N` rows, and **cancels exactly** when the flag is toggled at fixed
  `x`, instead of being estimated twice with independent errors and corrupted in the
  difference;
- **can correctly report zero** where `tau(x) = 0`: a loss-driven learner simply declines
  to split on the flag there, rather than manufacturing spurious heterogeneity from
  sampling noise.

A unifying view (tree base learner): the recipes differ only in *where the split on `W` is
allowed* — **anywhere** (S-learner, loss decides), **at the root** (T-learner, forced
first), **before the leaves** (modified-splitting forest). The S-learner is the
least-constrained member.

## Main limitation (failure mode)

The flag is one feature competing with all of `X`, and every base learner is regularized.
When the treatment effect is **real but weak relative to** the covariates' predictive power,
the regularizer down-weights or **discards** the flag, so `mu_hat(x,1) ≈ mu_hat(x,0)` and
`tau_hat` is **biased toward zero** (in the extreme, a flat zero). So the same property that
makes the S-learner *correct* when the effect is genuinely zero makes it *biased* when the
effect is real but small. Best regime: simple / sparse / near-zero / smooth effects; worst:
complex, strong-but-under-weighted effects (where the two-model recipe is preferable). The
two recipes are opposite ends of one bias–variance knob; neither dominates.

## Convergence rate

The clean response-surface rate is obtained for the arm-split two-model estimator and gives
the ceiling to remember for estimators that get the effect by estimating response values and
subtracting. For families with arm-wise minimax response rate `a_mu`, `n` treated and `m`
control units, and overlap `e_min < e(x) < e_max`:

```
tau_hat - tau = (mu_hat_1 - mu_1) - (mu_hat_0 - mu_0),  and (a-b)^2 <= 2a^2 + 2b^2  =>
EMSE <= 2 E[(mu_hat_1 - mu_1)^2] + 2 E[(mu_hat_0 - mu_0)^2].
```

A change of measure from the arm distribution to the marginal, bounded by overlap —
`(e_min/e_max) E[g(X)] <= E[g(X)|W=1] <= (e_max/e_min) E[g(X)]` (and the symmetric
control-arm bound with `1 - e(x)`) — turns each arm-error into the minimax rate:

```
EMSE <= 2C (e_max/e_min) n^{-a_mu}
      + 2C ((1-e_min)/(1-e_max)) m^{-a_mu}
      = O(n^{-a_mu} + m^{-a_mu}).
```

The exponent is `a_mu`, the **response** smoothness — never `a_tau`, the (often greater)
**effect** smoothness. So this family pays the response rate even when the effect is much
simpler than the responses, and in the unbalanced case `n << m` it is stuck at `n^{-a_mu}`
in the small arm. That structural ceiling — being unable to exploit a smoother effect — is
the gap a difference-of-responses estimator leaves open.

## Working code

Filling the `fit`/`predict` slot of the CATE harness with the three S-learner moves —
append the flag, fit one pooled model, predict twice and subtract:

```python
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class CATEEstimator:
    """S-learner: a single outcome model on the pooled data with the treatment flag as an
    ordinary feature; CATE read off by toggling the flag.  tau_hat(x) = mu_hat(x,1) - mu_hat(x,0)."""

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._model = GradientBoostingRegressor(
            n_estimators=200,      # many shallow trees -> smooth, low-variance surface
            max_depth=4,           # depth > 1 lets flag x covariate interactions form (heterogeneity)
            learning_rate=0.1,     # modest per-tree shrinkage
            min_samples_leaf=20,   # leaf floor: the effect is a difference of predictions, leaf-noise sensitive
            subsample=0.8,         # stochastic row subsampling -> extra regularization
            random_state=self._seed,
        )

    def fit(self, X, T, Y):
        XT = np.column_stack([X, T.reshape(-1, 1)])   # pooled design: [covariates | treatment flag]
        self._model.fit(XT, Y)                         # single surface mu(x, w) = E[Y^obs | x, w]
        return self

    def predict(self, X):
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])     # toggle flag to 1: mu_hat(x, 1)
        X0 = np.column_stack([X, np.zeros((n, 1))])    # toggle flag to 0: mu_hat(x, 0)
        return self._model.predict(X1) - self._model.predict(X0)   # tau_hat(x)
```

The same single-model-with-treatment-indicator construction underlies the standard library
implementations: for binary treatment, `causalml`'s `BaseSRegressor` prepends a 0/1
treatment column to the pooled `X`, fits the model once, predicts with all-zero and all-one
treatment columns, and subtracts; `EconML`'s `SLearner` fits one `overall_model` on `X`
concatenated with encoded treatment indicators and subtracts the control prediction from
the treatment-arm prediction.
With BART as the base learner this is the response-surface estimator of Hill (2011);
"single" is what generalizes it to any supervised regressor.
