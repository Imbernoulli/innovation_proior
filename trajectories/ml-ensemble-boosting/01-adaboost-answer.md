**Problem.** The inert default fill gives every tree the same vote, never moves the sample
distribution, and fits raw labels forever — 200 identical trees on the same problem. The floor that is
still *boosting* is to aim the weak learner: reweight the training examples toward the cases the
committee currently gets wrong, and combine the trees by a confidence-weighted vote.

**Key idea (AdaBoost).** Maintain a sample distribution. Each round, fit the depth-3 tree on it,
*measure* the tree's weighted error `eps`, and let that one number set both the reweighting strength
and the vote weight — strong rounds (small `eps`) shift the distribution hard and vote loud, weak
rounds (`eps` near `1/2`) barely move it and barely count. The constants are not chosen by hand:
minimizing the training-error bound gives `beta = eps/(1-eps)` and coefficient `0.5*log((1-eps)/eps)`.

**Why it works.** Multiplicative reweighting `w_i *= exp(alpha*1[wrong])` demotes solved examples and
keeps the hard ones heavy, forcing each new tree onto the current failures, so the trees' errors point
in different directions and the weighted vote beats any single tree; the measured-error coefficient
makes the whole thing adaptive — no edge needs to be known in advance.

**This-harness specifics (not the paper).** Classification returns the **raw labels** so the fixed loop
routes the tree to its *discrete signed-vote* head (`alpha*(2*pred-1)`); the harness folds shrinkage
into the coefficient, `alpha = learning_rate * 0.5*log((1-eps)/eps)`, since the discrete head has no
separate `learning_rate`. Regression has **no weighted median** here (the fixed accumulator is a
weighted sum of residual predictions), so the regression branch fits **residuals** with `alpha = 1.0`
(shrinkage via the loop's `alpha*learning_rate`) and keeps only AdaBoost.R2's hard-example
*reweighting* — `beta = avg_loss/(1-avg_loss)`, `w_i *= beta^(1-L_i)` over max-normalized errors `L_i`.

**Hyperparameters.** Fixed depth-3 tree, 200 rounds, `learning_rate = 0.1`; `weighted_err` clipped to
`[1e-10, 1-1e-10]`; weights renormalized each round.

**What to watch.** Strong on breast-cancer classification (home territory of the signed adaptive vote);
the regression weak spot is the worry — reweighting concentrates mass on the worst residuals while a
depth-3 tree chases them, injecting variance that a residual-only method (uniform weights) would
avoid. That is the failure the next rung removes.

```python
class BoostingStrategy:
    """AdaBoost: exponential loss reweighting (classification) / AdaBoost.R2 (regression)."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "classification":
            # AdaBoost fits on original labels (not residuals)
            return y
        else:
            # Regression: fit on negative gradient (residuals) so that the
            # fixed ensemble_predict accumulation (mean + sum alpha*lr*pred)
            # works correctly.
            return y - current_predictions

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        if self.task_type == "classification":
            preds = learner.predict(X)
            incorrect = (preds != y).astype(float)
            weighted_err = np.dot(sample_weights, incorrect) / sample_weights.sum()
            weighted_err = np.clip(weighted_err, 1e-10, 1.0 - 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - weighted_err) / weighted_err)
            return alpha
        else:
            # Regression: use alpha=1.0; shrinkage is applied by the fixed
            # ensemble_predict via learning_rate.  Sample reweighting in
            # update_weights handles the AdaBoost.R2 emphasis on hard examples.
            return 1.0

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        preds = learner.predict(X)
        if self.task_type == "classification":
            incorrect = (preds != y).astype(float)
            # w_i *= exp(alpha * I(wrong))
            sample_weights = sample_weights * np.exp(alpha * incorrect)
        else:
            # AdaBoost.R2-style: reduce weight on well-predicted samples
            # pseudo_targets are residuals; compare learner predictions to them
            errors = np.abs(preds - pseudo_targets)
            max_err = errors.max()
            if max_err > 0:
                errors = errors / max_err  # normalize to [0, 1]
            avg_loss = np.dot(sample_weights, errors)
            avg_loss = np.clip(avg_loss, 1e-10, 1.0 - 1e-10)
            beta = avg_loss / (1.0 - avg_loss)
            # Decrease weight for well-predicted samples
            sample_weights = sample_weights * np.power(beta, 1.0 - errors)
        # Normalize
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights
```
