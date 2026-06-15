# Gradient Boosting, distilled

Gradient boosting builds an additive model `F(x) = F_0 + sum_m v * h_m(x)` as **steepest
descent in function space**: at each round it fits a base learner (a small regression tree) by
least squares to the **negative gradient of the loss** evaluated at the current model's
predictions (the "pseudo-residuals"), then sets each leaf by the loss-specific constant update
for that region (an exact one-dimensional line search where available, a Newton step for losses
with no closed form), and folds the result in shrunk by a learning rate. This turns
forward-stagewise additive modeling for pointwise losses into a sequence of ordinary
least-squares tree fits plus low-dimensional line searches — the one thing the base-learner
library already does well.

## Problem it solves

Fit a forward-stagewise additive expansion `F(x) = sum_m beta_m h(x; a_m)` minimizing an
arbitrary differentiable loss `L(y, F)`. The per-stage subproblem
`argmin_{beta,a} sum_i L(y_i, F_{m-1}(x_i) + beta h(x_i; a))` has a convenient form only for
half squared error ("fit the residuals") and exponential loss (AdaBoost). Gradient boosting gives a
single recipe that works for squared error, absolute error, Huber, and binomial/multinomial
deviance alike, inheriting the robustness and interpretability of regression trees.

## Key idea

Treat the function values `{F(x_i)}` as the parameters and descend the empirical loss in that
N-dimensional space. The negative gradient at each point,

```
y~_i = -[ dL(y_i, F(x_i)) / dF(x_i) ]_{F = F_{m-1}},
```

is the best local descent direction but is defined only at the data. Generalize it by fitting
the base learner to it by least squares (the tree most aligned with `-g_m`), then size the step
on the true loss:

```
a_m   = argmin_{a,beta} sum_i ( y~_i - beta h(x_i; a) )^2          # find a descent direction (LS)
rho_m = argmin_rho sum_i L(y_i, F_{m-1}(x_i) + rho h(x_i; a_m))    # size the step on real L
F_m   = F_{m-1} + rho_m h(x; a_m)
```

For half squared error `y~_i` is the residual `y_i - F_{m-1}(x_i)`, so this reduces to classic
residual fitting. For absolute error `y~_i = sign(y_i - F_{m-1}(x_i))` and the line search is a
weighted median — robust by construction.

**Per-leaf (TreeBoost) refinement.** A `J`-leaf tree is a sum of disjoint-region indicators, so
instead of one global `rho_m`, give each leaf its own optimal constant; disjointness makes this
`J` independent 1-D line searches:

```
gamma_jm = argmin_gamma sum_{x_i in R_jm} L(y_i, F_{m-1}(x_i) + gamma).
```

The least-squares fit now only *finds the regions*; the per-leaf line search *sets the values*.

## Controls

- **Number of trees `M`** — natural early-stopping regularizer, chosen on held-out data.
- **Shrinkage / learning rate `v in (0,1)`**: `F_m = F_{m-1} + v * sum_j gamma_jm 1(x in R_jm)`.
  Each tree is a noisy greedy estimate of the descent direction; small steps average many of
  them (variance reduction), so `v` is the step-size regularizer at the cost of needing larger
  `M` (`v` and `M` trade off). Incremental shrinkage changes the path because each later
  pseudo-response is computed after the earlier shrunken steps; globally shrinking the finished
  model does not do that.
- **Tree size `J`**: a `J`-leaf tree captures interactions up to order `min(J-1, n)`; the
  additive ensemble's interaction order is capped by the single-tree cap. So `J` tracks the
  target's dominant interaction order — small trees usually suffice, large trees rarely needed.
- **Influence trimming**: an optional induction shortcut that ignores points contributing
  negligibly to the current update (for deviance, those with curvature
  `|y~_i|(2-|y~_i|)` near zero — confidently classified), with the retained influence chosen as
  a computational tolerance.

## Concrete loss instances

- **Half squared error (LS regression)**: `y~_i = y_i - F_{m-1}(x_i)`; per-leaf `gamma_jm` = mean
  residual in the leaf (= the least-squares tree's own leaf value, so the leaf update is the
  identity). `F_0` = mean of `y`.
- **Absolute error (LAD regression)**: `y~_i = sign(y_i - F_{m-1}(x_i))`; per-leaf `gamma_jm` =
  median of the leaf's residuals. `F_0` = median of `y`.
- **Huber-M regression**: `y~_i = (y_i - F_{m-1})` if `|y_i - F_{m-1}| <= delta_m`, else
  `delta_m * sign(y_i - F_{m-1})`, with `delta_m` = a quantile of `|residuals|`. In a leaf,
  with residuals `r_i` and `r_j = median_{i in R_j} r_i`, the one-step robust-location update is
  `gamma_jm = r_j + (1/N_j) * sum_{i in R_j} sign(r_i - r_j) * min(delta_m, |r_i - r_j|)`.
- **Binomial deviance (two-class), `y in {-1,1}`**, `L = log(1 + e^{-2yF})`:
  `y~_i = 2 y_i / (1 + e^{2 y_i F_{m-1}(x_i)})`. No closed-form line search, so each leaf gets a
  single Newton step from `gamma = 0`:

  ```
  gamma_jm ~= ( sum_{i in R_jm} y~_i ) / ( sum_{i in R_jm} |y~_i| (2 - |y~_i|) ),
  ```

  numerator = summed negative gradient, denominator = summed second derivative (curvature),
  with `|y~_i|(2-|y~_i|)` the per-point curvature in this `{-1,1}`, factor-2 logit convention.
  Equivalently, in the `{0,1}` convention with `p = sigmoid(F)` and pseudo-residual `r = y - p`,
  the negative gradient is `r` and the curvature is `p(1-p) = (y-r)(1-(y-r))`, so the leaf step
  is `sum(r) / sum(p(1-p))` (the form used in the code below).
  `F_0 = 0.5 log[(1+ybar)/(1-ybar)]` in the factor-2 convention, equivalently `log[ybar/(1-ybar)]`
  in the `{0,1}` convention; probabilities `p_+ = sigmoid(F_M)`. Putting the
  first-order pseudo-response in the least-squares fit and the curvature only in the summed
  per-leaf denominator avoids dividing by a vanishing per-point `p(1-p)`, the instability of
  fitting a full Newton target directly.
- **Multiclass deviance**: symmetric multinomial logit, `K` trees per round, per-leaf diagonal-
  Hessian Newton step `gamma_jkm = ((K-1)/K) * sum y~_{ik} / sum |y~_{ik}|(1 - |y~_{ik}|)`.

## Generic algorithm

```
F_0(x) = argmin_rho sum_i L(y_i, rho)
for m = 1..M:
    y~_i  = -[ dL(y_i, F(x_i)) / dF(x_i) ]_{F = F_{m-1}}          # pseudo-residuals (neg gradient)
    fit J-leaf regression tree to {(x_i, y~_i)} by least squares  # regions R_jm
    for each leaf j:
        gamma_jm = argmin_gamma sum_{x_i in R_jm} L(y_i, F_{m-1}(x_i) + gamma)
    F_m(x) = F_{m-1}(x) + v * sum_j gamma_jm 1(x in R_jm)         # shrinkage v
```

## Working code

The implementation structure is the standard tree-ensemble loop: per stage, fit a
`DecisionTreeRegressor` to the negative gradient, then overwrite each leaf value with the
loss-specific update (identity for squared error; `sum(neg_g)/sum(prob*(1-prob))` with
`prob = y - neg_g` for the binomial loss), and add
`learning_rate * leaf_value` to the running raw predictions.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _safe_divide(numerator, denominator):
    if abs(denominator) <= 1e-150:
        return 0.0
    return numerator / denominator


class GradientBoostingMachine:
    """Gradient boosting: fit trees to negative gradients (pseudo-residuals),
    set each leaf by the loss-specific update (exact for squared error, Newton for deviance), shrink by lr.
        F(x) = init + lr * sum_m (leaf values of tree m)
    task="regression" uses squared error; task="classification" uses binomial deviance.
    """

    def __init__(self, task="regression", n_rounds=200, max_depth=3, learning_rate=0.1):
        self.task = task
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.init_ = 0.0

    def _init_prediction(self, y):
        # F_0 = argmin_rho sum_i L(y_i, rho).
        if self.task == "regression":
            return float(np.mean(y))                 # mean for squared error
        p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))   # y in {0,1}
        return float(np.log(p / (1.0 - p)))          # logit init (p = sigmoid(F_0))

    def _pseudo_residuals(self, y, F):
        # y~_i = -dL/dF at F_{m-1}.
        if self.task == "regression":
            return y - F                             # squared error: negative gradient = residual
        return y - _sigmoid(F)                        # deviance: y - p on the probability scale

    def _leaf_value(self, resid_leaf, y_leaf):
        # gamma_jm = argmin_gamma sum_{leaf} L(y, F + gamma).
        if self.task == "regression":
            return float(np.mean(resid_leaf))        # squared error: mean residual (= LS leaf value)
        p = y_leaf - resid_leaf                                          # p = sigmoid(F) = y - neg_grad
        num = float(np.sum(resid_leaf))                                  # summed negative gradient
        den = float(np.sum(p * (1.0 - p)))                              # summed curvature p(1-p)
        return _safe_divide(num, den)                # single Newton step with safe zero update

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.init_ = self._init_prediction(y)
        F = np.full(len(y), self.init_)
        for _ in range(self.n_rounds):
            resid = self._pseudo_residuals(y, F)                 # negative gradient at current F
            tree = DecisionTreeRegressor(max_depth=self.max_depth, criterion="friedman_mse")
            tree.fit(X, resid)                                   # least-squares fit -> finds regions
            leaves = tree.apply(X)
            update = np.zeros(len(y))
            for leaf in np.unique(leaves):
                idx = np.where(leaves == leaf)[0]
                gamma = self._leaf_value(resid[idx], y[idx])     # per-leaf exact / Newton update
                tree.tree_.value[leaf, 0, 0] = gamma             # overwrite LS value with loss-specific update
                update[idx] = gamma
            F += self.learning_rate * update                     # shrunk stagewise update
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        F = np.full(X.shape[0], self.init_)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)            # leaf values already set to gamma
        return F

    def predict(self, X):
        F = self.decision_function(X)
        if self.task == "regression":
            return F
        return (_sigmoid(F) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])
```

## Relation to prior methods

- **LS-Boost / matching pursuit** is gradient boosting with half squared-error loss: the negative
  gradient is exactly the residual, so it reduces to iteratively fitting the residuals.
- **AdaBoost** corresponds to the exponential loss `e^{-yF}`; gradient boosting replaces its
  bespoke reweighting with the universal negative-gradient-fit-and-line-search loop and lets
  the gentler binomial deviance be used instead, for robustness to noisy labels.
- **LogitBoost / Newton boosting** fits a full Newton (IRLS) working response
  `z = (y* - p)/(p(1-p))`; gradient boosting fits the bounded first-order pseudo-response and
  defers the second derivative to the summed per-leaf denominator, which is the numerically
  stable way to get the same Newton behavior.
