On structured tabular data — spam, click-through, fraud, ranking, physics event classification — the learner that reliably wins is a boosted ensemble of shallow decision trees: build trees one at a time, each correcting the residual error of the ensemble so far, and sum their outputs as $\hat y_i = \sum_{k=1}^K f_k(x_i)$, where each $f_k$ is a regression tree with structure $q$ mapping an example to a leaf and a vector $w$ of leaf scores so that $f(x) = w_{q(x)}$. The forward-stagewise shape is settled: earlier trees are frozen, each round adds one new tree, and the only real question is *how* a round should choose that tree. What is not settled is a single strategy for the three decisions a round must make — what target the next tree fits, how its contribution is weighted, and how instances are re-weighted between rounds — that works for both classification and regression under any reasonable differentiable loss, that sets every leaf as well as possible, and that controls overfitting from inside the round rather than only by early stopping.

The existing recipes each fail one of those tests. AdaBoost, for binary labels in $\{-1,+1\}$, fits a weak classifier to weighted data, reads off $\alpha = \tfrac12\log((1-\mathrm{err})/\mathrm{err})$ from the weighted error, and re-weights by $w_i \leftarrow w_i\exp(-\alpha\,y_i h_t(x_i))$. That coefficient and that multiplicative reweighting are not generic decorations: they are exactly the stagewise minimization of the exponential loss $\sum_i \exp(-y_i F(x_i))$ written operationally. It is elegant for binary classification, but the original-label target and the exponential reweighting do not transfer to regression or to an arbitrary loss, and the exponential loss punishes mislabeled points exponentially. Gradient boosting is the general starting point: treat the predictions $F(x_i)$ as the coordinates being optimized, compute the first derivative $g_i = \partial l(y_i, F(x_i))/\partial F(x_i)$, and fit a tree to the pseudo-residuals $-g_i$, which for squared error is just the ordinary residual. But once the tree structure is fixed, Friedman re-optimizes each leaf by a *separate* one-dimensional problem $\gamma_j = \arg\min_\gamma \sum_{i\in I_j} l(y_i, F_{t-1}(x_i)+\gamma)$ that has no closed form for a general loss — a median for absolute deviation, an inner Newton solve for logistic loss. So the criterion that *chooses the split* is least squares on $-g_i$, while the criterion that *sets the leaf* is the original loss: two patched-together objects the structure search never reconciles. LogitBoost shows what is missing — for logistic loss it fits a working response $(y-p)/(p(1-p))$ weighted by $p(1-p)$, a Newton step using both derivatives — but it is derived for specific losses, folds the curvature into per-instance working responses rather than a single loss-agnostic objective, and carries no explicit penalty on the leaf scores.

I propose XGBoost-style second-order tree boosting: fit each new tree to a *regularized second-order approximation of the loss*, so that one objective simultaneously scores the split and sets the leaf, uses the loss curvature as a first-class quantity, and folds regularization directly into the round. The derivation closes cleanly. At round $t$, add one tree $f_t$ to the frozen model and expand the loss to second order in the increment, using the first and second derivatives $g_i = \partial l/\partial \hat y^{(t-1)}$ and $h_i = \partial^2 l/\partial \hat y^{(t-1)2}$:
$$l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat y_i^{(t-1)}) + g_i\,f_t(x_i) + \tfrac12 h_i\,f_t(x_i)^2.$$
The constant term drops, so the round minimizes $\tilde L^{(t)} = \sum_i [g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2] + \Omega(f_t)$, with the complexity charged inside the same objective by
$$\Omega(f) = \gamma T + \tfrac12 \lambda \sum_j w_j^2.$$
The $\gamma T$ term charges for extra leaves; the L2 term shrinks leaf scores and stabilizes Newton steps in leaves whose total curvature is small. Setting $\gamma=\lambda=0$ recovers the bare second-order objective, so this is an extension, not a different problem. Now, because every example in leaf $j$ receives the same value $w_j$, define $G_j = \sum_{i\in I_j} g_i$ and $H_j = \sum_{i\in I_j} h_i$ and the objective separates leaf by leaf into independent quadratics:
$$\tilde L^{(t)} = \sum_j \big[G_j w_j + \tfrac12 (H_j + \lambda) w_j^2\big] + \gamma T.$$
Differentiating each leaf's quadratic, $G_j + (H_j+\lambda)w_j = 0$, gives the optimal leaf value in closed form,
$$w_j^* = -\frac{G_j}{H_j + \lambda},$$
which is precisely a damped Newton step: the Hessian sum is the *denominator*, not a later correction, and $\lambda$ keeps a large gradient sum from producing a wild score where curvature is thin. Substituting the optimum back, each leaf contributes $-\tfrac12 G_j^2/(H_j+\lambda)$, so a fixed structure has the score
$$\tilde L^{(t)}(q) = -\frac12 \sum_j \frac{G_j^2}{H_j + \lambda} + \gamma T.$$
This is the impurity analogue I wanted: the same expression that sets the leaf values also scores the tree structure, so the split criterion and the leaf criterion are finally one object. Comparing a parent leaf with its two children gives the split gain
$$\text{gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma,$$
and I keep a split only when this is positive — a pruning rule that falls out of the objective rather than being bolted on, because a nonpositive gain means the loss reduction did not pay for the added leaf. Completing the square on a single example, $g_i f_i + \tfrac12 h_i f_i^2 = \tfrac12 h_i (f_i - (-g_i/h_i))^2 + \text{const}$, shows the equivalent weighted-least-squares form: the Newton target is $z_i = -g_i/h_i$ and the per-example weight is $h_i$. This is the point to keep straight — fitting bare $-g_i$ with uniform weights and using $h_i$ only in a single scalar multiplier is a first-order or hybrid approximation, not the canonical second-order tree; the exact method either fits $z_i$ with Hessian weights or, more faithfully, grows the tree directly from the aggregated $(G,H)$ and writes each leaf as $-G/(H+\lambda)$.

The loss derivatives instantiate this and check the signs. For squared error $l = \tfrac12(\hat y - y)^2$ we get $g_i = \hat y_i - y_i$ and $h_i = 1$, so the leaf value is the regularized mean residual $\sum_{i\in I_j}(y_i-\hat y_i)/(n_j+\lambda)$, which at $\lambda=0$ is exactly ordinary gradient boosting — constant curvature is *why* first-order and Newton boosting coincide there. For binary logistic loss with raw margin $F$ and $p = \sigma(F)$, $g_i = p_i - y_i$ and $h_i = p_i(1-p_i)$ floored away from zero, giving the working response $(y_i-p_i)/(p_i(1-p_i))$ weighted by $p_i(1-p_i)$, exactly the LogitBoost form recovered as a special case. The intuition is not "confident points do not matter": confidently *wrong* points carry a large gradient and tiny curvature, which is precisely why $\lambda$, the Hessian floor, and a minimum-child-Hessian rule are load-bearing rather than cosmetic. The method's boundary is honest too — if the loss has no useful second derivative, as with least-absolute-deviation or quantile loss, the quadratic term cannot set a meaningful Newton scale, and one should fall back to first-order gradient boosting with a line search or a loss-specific leaf solve. This is the clean answer for twice-differentiable losses with usable curvature.

Mapped onto a fixed four-stub harness that fits a weighted sklearn-style tree, the faithful adapter carries Hessians as the sample weights and fits Newton responses as the targets: the initial weights are the Hessians at the initial predictions (ones for squared error, $0.25$ for logistic at zero margin), the per-round target is $-g/h$, and the next round's weights are the new Hessians after the margin update — never AdaBoost-style exponential weights. The scalar $\alpha = \sum_i h_i z_i t_i / (\sum_i h_i t_i^2 + \lambda_{\text{global}})$ is only a damped line search along the whole fitted direction; it does not reproduce the per-leaf $+\lambda$ in $-G/(H+\lambda)$ or the canonical split gain. For that exactness the tree learner itself must aggregate gradients and Hessians, score splits by the gain formula, and write the leaf values directly. Both pieces follow: the harness adapter, and beneath it the canonical $(G,H)$ tree core matching `dmlc/xgboost` with L1 and max-delta disabled, where `CalcWeight` is $-G/(H+\lambda)$ and `CalcGain` is $G^2/(H+\lambda)$.

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
