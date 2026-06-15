**Problem (from step 2).** Gradient boosting fixed regression (California 0.5002, diabetes 58.82) but
gave back classification (0.9620 vs AdaBoost's 0.9708). Its global scalar step uses the per-example
curvature `p(1-p)` only to set one step length for the whole tree, collapsing a quantity that varies
from near-zero (confident points) to `1/4` (uncertain points) into a single factor.

**Key idea (second-order / XGBoost-style).** Expand the per-round loss to second order:
`g_i f_i + 0.5 h_i f_i^2`. Completing the square gives the Newton working response `z_i = -g_i/h_i`
weighted by curvature `h_i`; with leaf statistics `G_j, H_j` and L2 `lambda`, the regularized objective
yields the closed-form leaf value `w_j* = -G_j/(H_j+lambda)`, structure score `-0.5 sum G_j^2/(H_j+lambda)`,
and split gain `0.5[G_L^2/(H_L+lambda)+G_R^2/(H_R+lambda)-G^2/(H+lambda)] - gamma`. The Hessian is the
denominator of the leaf, not an afterthought; `lambda` stabilizes low-curvature leaves.

**This-harness specifics (not the paper).** The canonical `(G,H)` tree is **not expressible**: the fixed
sklearn tree exposes neither its split criterion (no gain-based splits) nor its leaf values (no
`-G_j/(H_j+lambda)`), and the Newton target `-g/h` divides by vanishing curvature on confident points
(the LogitBoost instability), so the classification target stays the bare `-g = y - sigmoid(F)`. What
this rung actually adds over gradient boosting is therefore confined to the **global** step: (1) carry
the L2 regularizer in the step denominator, `alpha = sum(g*pred)/(sum(h*pred^2) + reg_lambda)` with
`reg_lambda = 1.0` (the global analogue of `+lambda`); (2) apply that line search to **regression too**
(`h=1`), where gradient boosting used `alpha=1.0`. Per-leaf Newton and structure-aware gain are the
parts the contract cannot collect.

**Hyperparameters.** Fixed depth-3 tree, 200 rounds, `learning_rate = 0.1`, `reg_lambda = 1.0`; sigmoid
clipped to `[-500,500]`; `alpha` clamped non-negative; uniform weights throughout.

**What to watch.** Modest, by construction (a global-step refinement, not a regime change): regression
holds or nudges down (diabetes ~58.7, California ~0.50; the seed-456 diabetes outlier stays hard);
classification near 0.9620, **not** a recovery to 0.9708, since the global scalar cannot recover the
per-example second-order signal. The real second-order win lives in the `(G,H)` tree this fixed-tree
contract does not let me build.

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
