AdaBoost's numbers split exactly along the line I worried about. On breast-cancer classification it was the strongest rung I will run — mean accuracy 0.9708 — because the signed adaptive vote, fed by reweighting toward the few ambiguous cancer cases, is precisely what adaptive boosting is good at. But on the two regression tasks it bled: diabetes mean RMSE 60.50 and California Housing 0.6466. The failure is structural, not a tuning miss. On regression I ran AdaBoost's *reweighting* on top of a residual-fitting additive model whose combiner is a plain weighted sum, because this harness has no weighted median. The reweighting keeps concentrating the sample distribution onto the worst-fit points, a depth-3 tree fit to residuals on that sharply peaked distribution chases the noisy tail, and nothing in the loop averages the injected variance away. The fix is therefore not a better reweighting — it is *no reweighting at all* on regression, plus a principled choice of what each tree should fit.

I propose **gradient boosting**: stop thinking about reweighting examples and instead build an additive expansion $F(x)=F_0+\sum_m \beta_m\,h(x;a_m)$ fit greedily, one tree at a time, to minimize a differentiable loss. At stage $m$ I hold $F_{m-1}$ fixed and want $\arg\min_{\beta,a}\sum_i L\!\big(y_i,\,F_{m-1}(x_i)+\beta\,h(x_i;a)\big)$. For half squared error this is lovely — adding $\beta h$ means fitting it to the residual $y_i-F_{m-1}(x_i)$ by least squares, the one operation the depth-3 tree does — and it needs *no* sample reweighting: every example carries equal weight, the tree fits the residual on the uniform distribution, and the variance AdaBoost injected on California simply never enters. But the losses I actually need are not all squared error: classification is log-loss, and for log-loss the stage $\arg\min$ over $(\beta,a)$ has no convenient form. I do not want a zoo of bespoke procedures (exponential loss gives AdaBoost, which I just ran); I want a single recipe that turns *any* differentiable loss's stage subproblem into the one operation the tree does well.

The move is to change what I differentiate. Treat $F$, evaluated at the $N$ training points, as its own free parameter — just $N$ numbers — and descend the empirical loss in that $N$-dimensional space. The negative-gradient component at point $i$ is

$$-g_i = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}},$$

which always exists. For squared error it is the residual $y_i-F_{m-1}(x_i)$, recovering the classic loop; for log-loss with $p=\sigma(F)$ it is $y_i-p_i$ on the probability scale. So the negative gradient is a vector of *pseudo-responses*, one per point, for any differentiable loss. The catch is that this gradient is defined only at the $N$ data points — a list with no values anywhere else — and a model has to predict at new $x$. The rescue is to fit the member of the base class whose values at the training points are as parallel as possible to $-g_m$, and "most parallel, free to rescale" is least squares: minimize $\sum_i (-g_m(x_i)-\beta\,h(x_i))^2$. That is exactly `tree.fit(X, pseudo_targets, sample_weight)`. The unsolvable stage subproblem is replaced by "compute the one-line derivative, then least-squares-fit a tree to it," and for squared error the pseudo-response *is* the residual, so it reduces to the classic loop while the same loop now also runs for log-loss.

Landing this in *this* harness forces one honest reduction. The full TreeBoost refinement gives each leaf its own optimal constant by an independent per-leaf line search on the real loss, overwriting the tree's least-squares leaf values — strictly better, because the least-squares fit *finds the regions* and the per-leaf step *sets the values*. But this strategy never sees the tree's leaves; the four-method contract hands me `learner.predict(X)` and nothing else, so I cannot read `tree.tree_.value` or reassign leaf constants. The per-leaf Newton step is simply not expressible. The next-best thing the contract exposes is a *global* scalar line search: the least-squares tree already carries leaf values equal to the mean residual per leaf, and I size the whole tree's contribution with one scalar $\alpha$ chosen on the real loss, with the fixed loop folding in $\alpha\cdot\text{lr}\cdot\text{tree.predict}(X)$. The harness thus reduces TreeBoost's per-leaf Newton to "least-squares direction, one global Newton-style step length."

Concretely, two branches. **Regression** is the clean fix for AdaBoost's weak spot: `init_weights` is uniform and *stays* uniform — `update_weights` returns the weights untouched, because gradient boosting does not reweight. `compute_targets` returns $y-\text{current\_predictions}$, the negative gradient of squared error; `compute_learner_weight` returns $\alpha=1.0$, since for squared error the optimal global step is essentially the identity (the least-squares tree already fits the residual mean per leaf) and the loop's $\alpha\cdot\text{lr}$ accumulation supplies the shrinkage. No exponential reweighting, no $\beta$, no max-normalized loss — every example contributes equally to every fit, so the California variance is gone by construction. **Classification** is where the global step does real work, because squared error is the wrong criterion and I am committed to log-loss. I track the raw ensemble margin in `_raw_scores` (zeros at init). `compute_targets` returns $y-\sigma(\text{\_raw\_scores})$ — the log-loss negative gradient, a bounded pseudo-response in $(-1,1)$ that shrinks toward zero for confidently-classified points, exactly the influence structure I want. Because that target is continuous, the loop routes the tree to its *continuous* accumulating head, not AdaBoost's signed-vote head. Expanding log-loss to second order in the increment $\rho\,\text{pred}$, the optimal step is the Newton ratio

$$\alpha = \frac{\sum_i \text{pseudo}_i\,\text{pred}_i}{\sum_i p_i(1-p_i)\,\text{pred}_i^2},$$

gradient–tree alignment over summed curvature $p(1-p)$ weighted by $\text{pred}^2$; I floor the denominator with $10^{-10}$ and clamp $\alpha\ge0$ so a tree pointing the wrong way cannot get a negative weight that corrupts the margin. `update_weights` then advances $\text{\_raw\_scores} \mathrel{+}= \text{lr}\cdot\alpha\cdot\text{pred}$ and leaves the sample weights uniform.

The shallow-tree and shrinkage choices now have a cleaner justification than before. Depth 3 keeps each tree a *noisy, greedy* estimate of the descent direction — the best least-squares-aligned tree to a sample-based negative gradient, which has variance and can latch onto idiosyncrasies of this training sample. Taking the full step would commit hard to each noisy direction; $\text{lr}=0.1$ means each tree contributes only a fraction, so the ensemble averages over many slightly different noisy directions before moving far — variance reduction by averaging, the same reason a small learning rate stabilizes any noisy gradient descent, with 200 rounds giving room for it to pay off. This is also *why* it should beat AdaBoost on regression: AdaBoost added variance by reweighting onto the tail, where gradient boosting removes it by fitting on the uniform distribution and shrinking.

The falsifiable claims, reading AdaBoost's measured shape: on regression I expect a clear improvement, with California the sharpest test — its 0.6466 was inflated by reweighting variance, so it should fall well below 0.6, and diabetes should drop from 60.50 toward the high-50s. The diabetes seed-456 outlier is the riskier one; a single seed whose residual structure is genuinely hard could stay high regardless. On classification I expect to give a little back: trading AdaBoost's bespoke exponential-loss vote for a general continuous-head log-loss model sized by a *global* step is a different, smoother object, so breast-cancer accuracy may slip slightly below 0.97. If the pattern is regression clearly up and classification slightly down, the next rung is already named — the global scalar step leaves per-example Hessian information on the table that a per-example second-order weighting would recover.

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
