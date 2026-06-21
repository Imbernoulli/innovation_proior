Gradient boosting did what I predicted. On regression it crushed AdaBoost where the reweighting was bleeding: California Housing fell from 0.6466 to **0.5002** (a 23% drop, per-seed spread tightening to 0.5113 / 0.4922 / 0.4970) and diabetes from 60.50 to **58.82** — removing the reweighting and fitting residuals on the uniform distribution was the right diagnosis. On classification it gave a little back, also as predicted: breast-cancer accuracy slipped from 0.9708 to **0.9620**. That slip is the tell. In the classification branch I sized each tree by one Newton ratio $\sum_i g_i\,\text{pred}_i / \sum_i p_i(1-p_i)\,\text{pred}_i^2$ — a single scalar for the *whole* tree. But the curvature $p(1-p)$ is *per example* and varies enormously: near zero for a confident point, near $\tfrac14$ for a maximally-uncertain one. Collapsing all of it into one scalar means a leaf full of uncertain points and a leaf full of confident points get scaled by the same factor, when the right Newton step is leaf-specific — large where curvature is small and the gradient is real, small where it is large. The question for this rung is how to get the per-example second-order information *into* the fit, not just into a single scalar.

I propose the **second-order, XGBoost-style** round. At round $t$, add one tree $f_t$ to the frozen model and minimize $\sum_i L(y_i, \hat y_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$. Expand the loss to second order in the increment,

$$L(y_i, \hat y_i + f_i) \approx L(y_i,\hat y_i) + g_i f_i + \tfrac12 h_i f_i^2,$$

with $g_i, h_i$ the first and second derivatives of the loss at the current prediction. Dropping the constant first term, the round's objective is $\sum_i[g_i f_i + \tfrac12 h_i f_i^2] + \Omega(f_t)$ — the generalization of gradient boosting, which kept only the $g_i$ term and fit $-g_i$ by least squares. Completing the square on a single point,

$$g_i f_i + \tfrac12 h_i f_i^2 = \tfrac12 h_i\big(f_i - (-g_i/h_i)\big)^2 + \text{const},$$

shows what the curvature buys: the exact second-order round is a *weighted* least-squares fit to the Newton working response $z_i = -g_i/h_i$ with per-example weight $h_i$ — not "fit $-g_i$ and adjust the step later," but fit the working response weighted by curvature. For log-loss, $g_i=p_i-y_i$ and $h_i=p_i(1-p_i)$, so $z_i=(y_i-p_i)/(p_i(1-p_i))$ weighted by $p_i(1-p_i)$ — the LogitBoost / IRLS working response. For squared error $g_i=F_i-y_i$, $h_i=1$, so $z_i$ is the plain residual with unit weight, which is exactly why first- and second-order coincide there.

Putting regularization *inside* the same objective is what makes the curvature safe to use. A regression tree is $f(x)=w_{q(x)}$ with leaves $j$ and leaf scores $w_j$; the natural cost is $\Omega(f)=\gamma T + \tfrac12\lambda\sum_j w_j^2$, where $\gamma T$ charges for extra leaves and the L2 term shrinks leaf scores and *stabilizes Newton steps in low-curvature leaves*. With leaf statistics $G_j=\sum_{i\in j} g_i$, $H_j=\sum_{i\in j} h_i$, the objective separates into independent one-variable quadratics $\sum_j[G_j w_j + \tfrac12(H_j+\lambda)w_j^2] + \gamma T$, so each leaf has a closed-form optimum

$$w_j^\* = -\frac{G_j}{H_j+\lambda},$$

with structure score $-\tfrac12\sum_j G_j^2/(H_j+\lambda) + \gamma T$ and split gain $\tfrac12[\,G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - G^2/(H+\lambda)\,] - \gamma$. The Hessian is not a later correction — it is the *denominator* of the leaf value, and $\lambda$ prevents a large gradient sum in a low-curvature leaf from producing a wild score. The same $(G,H)$ expression scores the split *and* sets the leaf — one object, not two patched-together steps.

That is the faithful second-order method, and I have to be blunt about what this harness exposes, because the faithful artifact requires building the tree from the $(G,H)$ gain and writing $-G_j/(H_j+\lambda)$ into every leaf — and the four-method contract does none of that. The base learner is the fixed sklearn `DecisionTree`: I do not control its split criterion (so the $(G,H)$ gain is unavailable) and I never see or write its leaf values (so $-G_j/(H_j+\lambda)$ is unavailable). The per-example Hessian *weighting* is the one piece I might hope to slip in through `sample_weights`, but the loop sets the weights from `update_weights` and the targets from `compute_targets`, and the classification target is already $y-\sigma(\text{raw})$ — the bare $-g$, because the true Newton target $-g/h=(y-p)/(p(1-p))$ would divide by a vanishing curvature and blow up on confident points, the exact LogitBoost instability. So the honest reduction this rung ships is *not* the canonical $(G,H)$ tree, nor even the Hessian-weighted Newton fit: it is the same first-order tree as gradient boosting, with the second-order information again confined to a *global* scalar — but now with two changes that distinguish it from the previous rung.

Both changes are what $+\lambda$ and the regression line search give me inside the global-step world. First, the global step carries the L2 regularizer in its denominator,

$$\alpha = \frac{\sum_i g_i\,\text{pred}_i}{\sum_i h_i\,\text{pred}_i^2 + \lambda},\qquad \lambda = 1.0,$$

the global analogue of $w_j^\*=-G_j/(H_j+\lambda)$ — the same regularized-Newton denominator, aggregated over the whole tree instead of per leaf. It damps the step when the summed curvature is small, precisely when the unregularized gradient-boosting step (with $+10^{-10}$) could overshoot. Second — the real delta from the previous rung — I apply the *same line search to regression*, where gradient boosting just used $\alpha=1.0$. For squared error $h_i=1$, so $\alpha=\sum_i \text{pseudo}_i\,\text{pred}_i / (\sum_i \text{pred}_i^2 + \lambda)$: a regularized least-squares step length along the fitted tree. The least-squares tree fits the residual, so $\sum(\text{pseudo}\cdot\text{pred})/\sum(\text{pred}^2)$ is the ordinary least-squares scale of the tree onto the residual — essentially 1 when the tree fits well — and the $+1.0$ shrinks it slightly, giving an $\alpha$ near but a touch under 1, a mild extra shrinkage on top of the loop's learning rate. That L2-damped step is the lever that should buy a small extra accuracy on the regression tasks. The classification branch is gradient boosting's Newton ratio with $+\lambda$ swapped for $+10^{-10}$ in the denominator — a more conservative global step.

The design choices carry over: depth-3 trees (still weak, still the noisy greedy direction estimate I want to average), $\text{lr}=0.1$, 200 rounds, the `_raw_scores` margin tracking for classification, uniform weights throughout, since this is a residual/gradient method, not a reweighting one. The only genuinely new constant is $\lambda=1.0$ in every step's denominator. Neither branch recovers the *per-leaf* Newton step — that is the part the harness structurally cannot express, and I name it plainly rather than oversell a global scalar as the canonical second-order tree.

The claims are deliberately modest, because the harness has reduced the second-order idea to a global-step refinement, not a regime change. On regression I expect a tie-to-slight-improvement: the L2-damped line search should hold diabetes around 58.8 and California around 0.50, possibly nudging diabetes a hair below 58.82 since the regularized step is a touch better calibrated than the fixed $\alpha=1$ — but the diabetes seed-456 outlier (67.55) is residual-structure-hard and will stay high regardless, so the diabetes mean will barely move. On classification the honest expectation is a near-tie around 0.9620, *not* a recovery to AdaBoost's 0.9708, since the conservative $+\lambda$ step is still a global scalar that cannot recover the per-example second-order signal — I would not be surprised if a more damped step ends up marginally lower on one seed by underfitting the few hard cancer cases the bespoke vote handled. The genuine second-order win lives in the $(G,H)$ tree with per-leaf $-G_j/(H_j+\lambda)$ and gain-based splits — exactly the win this fixed-tree contract does not let me collect.

```python
class BoostingStrategy:
    """XGBoost-style: second-order Newton boosting with regularization."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        # L2 regularization on leaf weights (lambda in XGBoost)
        self.reg_lambda = 1.0
        # Track raw scores for gradient/Hessian computation
        self._raw_scores = None

    def init_weights(self, n_samples):
        self._raw_scores = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "regression":
            # Negative gradient of squared error = residuals
            return y - current_predictions
        else:
            # Negative gradient of log-loss
            probs = self._sigmoid(self._raw_scores)
            return y - probs

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        preds = learner.predict(X)
        if self.task_type == "regression":
            # Newton step: sum(gradient * pred) / (sum(hessian * pred^2) + lambda)
            # For squared error: gradient = residual, hessian = 1
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(preds ** 2) + self.reg_lambda
            alpha = numerator / denominator
            return max(alpha, 0.0)
        else:
            # For log-loss: hessian = p*(1-p)
            probs = self._sigmoid(self._raw_scores)
            hessians = probs * (1.0 - probs)
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(hessians * preds ** 2) + self.reg_lambda
            alpha = numerator / denominator
            return max(alpha, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # XGBoost uses second-order info, not sample reweighting.
        # Update raw scores for next round's gradient computation.
        preds = learner.predict(X)
        self._raw_scores += self.learning_rate * alpha * preds
        # Weights stay uniform — boosting signal is in the pseudo-residuals
        return sample_weights
```
