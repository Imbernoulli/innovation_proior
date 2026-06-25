**Problem (from step 1).** AdaBoost's regression weak spot (California 0.6466, diabetes 60.50) was
self-inflicted: its reweighting concentrated mass on the worst residuals, and a depth-3 tree chasing
those points injected variance the no-median additive accumulator could not average away. The fix is to
stop reweighting and choose, principled, what each tree fits — under a single recipe that also serves
classification with a non-exponential loss.

**Key idea (gradient boosting).** Build an additive model `F = F_0 + sum_m alpha_m*lr*h_m` as steepest
descent in function space. At each round fit the depth-3 tree by least squares to the **negative
gradient** of the loss at the current predictions (the pseudo-residuals), and size the step on the true
loss. For squared error the negative gradient is the residual `y - F`, so it is the classic uniform-
distribution residual loop; for log-loss it is `y - sigmoid(F)` on the probability scale.

**Why it works.** Fitting residuals on the **uniform** distribution removes the variance AdaBoost's
reweighting added, so regression should improve; the negative-gradient view makes one loop serve any
differentiable loss; shrinkage (`lr=0.1`) averages many noisy greedy directions over 200 rounds.

**This-harness specifics.** TreeBoost's per-leaf Newton step is **not expressible** —
the strategy never sees the tree's leaves (no `tree.tree_.value` access). So the per-leaf step is
reduced to a single **global** scalar `alpha`: regression uses `alpha = 1.0` (the LS tree already fits
the residual mean per leaf; shrinkage comes from the loop's `alpha*lr` accumulation), classification
uses a global Newton line search `alpha = sum(g*pred)/sum(p(1-p)*pred^2)` along the fitted tree. The
margin is tracked in `_raw_scores`; classification targets are continuous, so the loop routes the tree
to its continuous accumulating head. Sample weights stay **uniform** (no reweighting).

**Hyperparameters.** Fixed depth-3 tree, 200 rounds, `learning_rate = 0.1`; sigmoid clipped to
`[-500, 500]`; Newton denominator floored with `1e-10`; `alpha` clamped non-negative.

**What to watch.** Regression should drop clearly (California well below 0.6, diabetes toward the high-
50s); the diabetes seed-456 outlier may stay hard regardless. Classification may give back a little
(continuous-head log-loss vs the bespoke signed vote). A slip there names the next rung: the global
scalar step leaves per-example Hessian information on the table.

```python
class BoostingStrategy:
    """Gradient Boosting: negative gradient (pseudo-residual) fitting."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        # Track raw scores for logistic gradient computation
        self._raw_scores = None

    def init_weights(self, n_samples):
        # Gradient boosting uses uniform weights (no reweighting);
        # the key insight is fitting to pseudo-residuals instead.
        self._raw_scores = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "regression":
            # Negative gradient of squared error = residuals
            return y - current_predictions
        else:
            # Negative gradient of log-loss (logistic)
            # For log-loss: -dL/dF = y - sigmoid(F)
            probs = self._sigmoid(self._raw_scores)
            return y - probs

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        if self.task_type == "regression":
            # Standard gradient boosting: alpha=1, shrinkage via learning_rate in ensemble
            return 1.0
        else:
            # For classification: use line search on log-loss
            preds = learner.predict(X)
            # Approximate optimal step size via Newton step
            probs = self._sigmoid(self._raw_scores)
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(probs * (1 - probs) * preds ** 2) + 1e-10
            alpha = numerator / denominator
            return max(alpha, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # Gradient boosting doesn't reweight samples; it fits to pseudo-residuals.
        # But we update raw scores for classification gradient computation.
        if self.task_type == "classification":
            preds = learner.predict(X)
            self._raw_scores += self.learning_rate * alpha * preds
        # Weights stay uniform
        return sample_weights
```
