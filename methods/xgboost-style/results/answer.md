# XGBoost-style second-order tree boosting, distilled

XGBoost-style boosting trains an additive tree ensemble by fitting each new tree to a
regularized second-order approximation of the loss. At round `t`, with current predictions
`yhat_i^(t-1)`, compute

```
g_i = d l(y_i, yhat_i^(t-1)) / d yhat
h_i = d^2 l(y_i, yhat_i^(t-1)) / d yhat^2
```

and minimize

```
Ltilde^(t) = sum_i [g_i f_t(x_i) + 0.5 h_i f_t(x_i)^2] + gamma T + 0.5 lambda sum_j w_j^2,
f_t(x) = w_{q(x)}.
```

For a fixed tree structure, with leaf instance set `I_j`,
`G_j = sum_{i in I_j} g_i` and `H_j = sum_{i in I_j} h_i`, the objective separates as

```
Ltilde^(t) = sum_j [G_j w_j + 0.5 (H_j + lambda) w_j^2] + gamma T.
```

The optimal leaf and structure score are therefore

```
w_j* = -G_j / (H_j + lambda)
score(q) = -0.5 sum_j G_j^2 / (H_j + lambda) + gamma T
```

and the gain from splitting a node into left/right children is

```
gain = 0.5 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma.
```

Keep a split only when `gain > 0`. The same formula chooses the split and sets the leaves.
`gamma` is the extra-leaf cost; `lambda` shrinks leaf scores and stabilizes low-curvature leaves.

## Loss derivatives

| Loss | `g_i` | `h_i` | Leaf value |
| --- | --- | --- | --- |
| squared error `0.5(yhat-y)^2` | `yhat_i - y_i` | `1` | `sum(y_i-yhat_i)/(n_j+lambda)` |
| logistic log loss, `p=sigmoid(margin)` | `p_i - y_i` | `max(p_i(1-p_i), eps)` | `-sum(p_i-y_i)/(sum p_i(1-p_i)+lambda)` |

Completing the square gives the Newton working-response form:

```
g_i f_i + 0.5 h_i f_i^2 = 0.5 h_i (f_i - (-g_i/h_i))^2 + constant.
```

So a weighted-tree scaffold should fit target `z_i = -g_i/h_i` with sample weight `h_i`.
Fitting bare `-g_i` with uniform weights and using Hessians only in a scalar multiplier is not
the canonical XGBoost tree; it is a first-order/hybrid approximation.

## Fixed-harness adapter

If the surrounding benchmark insists on the four-stub interface in `context.md`, the closest
faithful adapter is to carry Hessians as `sample_weights` and fit Newton responses as targets.
The scalar `alpha` below is only a damped line search along the fitted tree; exact per-leaf
`lambda` regularization and XGBoost split gain require the custom tree in the next section.

```python
import numpy as np


class BoostingStrategy:
    """Newton-response adapter for a fixed sklearn-style tree harness."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.global_l2 = config.get("reg_lambda", 1.0)
        self.eps = 1e-16
        self._raw_scores = None

    def init_weights(self, n_samples):
        self._raw_scores = np.zeros(n_samples)
        if self.task_type == "classification":
            return np.full(n_samples, 0.25)        # h = p(1-p) at p=0.5
        return np.ones(n_samples)                  # squared loss h = 1

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # The harness calls tree.fit(..., sample_weight=sample_weights) immediately
        # after this method, so keep sample_weights synchronized with the current Hessian.
        if self.task_type == "regression":
            sample_weights[:] = 1.0
            return y - current_predictions          # z = -g/h, with h = 1

        self._raw_scores = current_predictions.copy()
        p = self._sigmoid(self._raw_scores)
        h = np.maximum(p * (1.0 - p), self.eps)
        sample_weights[:] = h
        return (y - p) / h                          # z = -g/h

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        t = learner.predict(X)
        numerator = np.sum(sample_weights * pseudo_targets * t)      # sum(-g_i t_i)
        denominator = np.sum(sample_weights * t * t) + self.global_l2
        if denominator <= 0:
            return 0.0
        return max(numerator / denominator, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        self._raw_scores += self.learning_rate * alpha * learner.predict(X)
        if self.task_type == "classification":
            p = self._sigmoid(self._raw_scores)
            return np.maximum(p * (1.0 - p), self.eps)
        return np.ones_like(sample_weights)
```

## Canonical `(G,H)` tree core

This is the faithful L2-only tree-builder core matching `dmlc/xgboost` when L1 and max-delta are
disabled: `CalcWeight = -G/(H+lambda)` and `CalcGain = G^2/(H+lambda)`.

```python
import numpy as np


def grad_hess(y, margin, task):
    if task == "regression":
        return margin - y, np.ones_like(y)
    p = 1.0 / (1.0 + np.exp(-np.clip(margin, -500, 500)))
    return p - y, np.maximum(p * (1.0 - p), 1e-16)


def calc_weight(G, H, lam):
    return -G / (H + lam)


def calc_gain(G, H, lam):
    return (G * G) / (H + lam)


def split_gain(GL, HL, GR, HR, lam, gamma):
    G, H = GL + GR, HL + HR
    return 0.5 * (calc_gain(GL, HL, lam) + calc_gain(GR, HR, lam) - calc_gain(G, H, lam)) - gamma


def best_split(X, g, h, rows, lam=1.0, gamma=0.0, min_child_h=1.0):
    G, H = g[rows].sum(), h[rows].sum()
    parent = calc_gain(G, H, lam)
    best = None
    best_gain = 0.0

    for feat in range(X.shape[1]):
        col = X[rows, feat]
        present = ~np.isnan(col)
        present_rows = rows[present]
        missing_rows = rows[~present]
        if len(present_rows) < 2:
            continue

        order = np.argsort(col[present], kind="stable")
        present_rows = present_rows[order]
        values = col[present][order]
        csg = np.cumsum(g[present_rows])
        csh = np.cumsum(h[present_rows])
        G_present, H_present = csg[-1], csh[-1]

        for s in range(1, len(present_rows)):
            if values[s] == values[s - 1]:
                continue
            threshold = 0.5 * (values[s] + values[s - 1])

            # Missing values go right: left child is the observed prefix.
            GL, HL = csg[s - 1], csh[s - 1]
            GR, HR = G - GL, H - HL
            if HL >= min_child_h and HR >= min_child_h:
                gain = 0.5 * (calc_gain(GL, HL, lam) + calc_gain(GR, HR, lam) - parent) - gamma
                if gain > best_gain:
                    best_gain = gain
                    right = np.concatenate([present_rows[s:], missing_rows])
                    best = (gain, feat, threshold, False, present_rows[:s], right)

            # Missing values go left: right child is the observed suffix.
            GR, HR = G_present - csg[s - 1], H_present - csh[s - 1]
            GL, HL = G - GR, H - HR
            if HL >= min_child_h and HR >= min_child_h:
                gain = 0.5 * (calc_gain(GL, HL, lam) + calc_gain(GR, HR, lam) - parent) - gamma
                if gain > best_gain:
                    best_gain = gain
                    left = np.concatenate([present_rows[:s], missing_rows])
                    best = (gain, feat, threshold, True, left, present_rows[s:])

    return best            # None means no positive regularized gain
```

Each leaf created by this split search receives `calc_weight(G_leaf, H_leaf, lambda)`, and the
ensemble update is `F_t(x) = F_{t-1}(x) + eta * tree_t(x)`.
