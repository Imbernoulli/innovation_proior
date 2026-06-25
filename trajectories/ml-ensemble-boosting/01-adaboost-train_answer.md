The scaffold I start from is inert: it hands every tree the same vote, never moves the sample distribution, and fits the raw labels on every round. Run through the fixed loop that is 200 *identical* trees on the same problem — no committee, just one tree copied. The cheapest thing the four-method contract can express that is still genuinely boosting is to *aim* the weak learner. A depth-3 tree that is only a little better than chance, called repeatedly on the same distribution, keeps finding the same cheap regularity and keeps falling down on the same hard pocket; a majority over correlated mistakes is still a mistake. The leverage is that the weak-learning promise is distribution-free — an edge on *every* distribution, not just the natural one — so I am allowed to be adversarial toward my own committee: after collecting some trees, put the sample mass exactly where they currently fail, and the next tree is still obligated to find an edge there.

I propose **AdaBoost**: maintain a sample distribution, fit the tree on it, *measure* the tree's weighted error, and let that single number drive both how hard to shift the distribution and how loudly the tree votes. Keeping a weight $w_i$ per example and normalizing to a distribution $p_i$, the weighted error of the returned $h_t$ is

$$\varepsilon_t = \sum_i p_i\,\mathbf{1}[h_t(x_i)\neq y_i].$$

The reweighting is multiplicative, $w_i \leftarrow w_i\,\beta_t^{\,1-\mathbf{1}[\text{wrong}]}$, because multiplicative decay makes the total-weight bookkeeping factor cleanly across rounds. The direction is the load-bearing check: a *correct* example gets exponent $1$ and is multiplied by $\beta_t<1$, so it shrinks (demote the solved case); a *wrong* example gets exponent $0$ and is multiplied by $\beta_t^0=1$, so it holds (keep the hard case heavy). That is exactly the pressure forcing the next tree onto the current failures, and it is what makes the trees' errors point in different directions so the weighted vote can beat any single tree.

The constants are not guessed — the training-error analysis dictates them. Tracking the total weight $W_t$, the convexity of $\beta^x$ on $[0,1]$ bounds $W_{t+1}\le W_t\,[1-(1-\beta_t)(1-\varepsilon_t)]$; lower-bounding the surviving weight of any final-vote mistake and squeezing the two handles together means minimizing each per-round factor $[\varepsilon+(1-\varepsilon)\beta]/\sqrt{\beta}$, whose stationarity $(1-\varepsilon)\beta=\varepsilon$ gives

$$\beta_t = \frac{\varepsilon_t}{1-\varepsilon_t},\qquad \text{minimized factor } 2\sqrt{\varepsilon_t(1-\varepsilon_t)}.$$

The training error is then bounded by $\prod_t 2\sqrt{\varepsilon_t(1-\varepsilon_t)} = \exp(-2\sum_t \gamma_t^2)$ writing $\varepsilon_t=\tfrac12-\gamma_t$ — exponential amplification that never needs the edge in advance and banks the *actual* squared edges, the slack the fixed-schedule predecessors (Schapire 1990; the flat-majority construction) threw away by committing to one worst-case $\gamma$ up front. The equivalent signed $\{-1,+1\}$ form gives the same algorithm with vote coefficient $\alpha_t=\tfrac12\log\!\frac{1-\varepsilon_t}{\varepsilon_t}$; the half is the convention, since the signed margin spans width $2$ while the indicator spans width $1$. A strong round (small $\varepsilon$) thus shifts the distribution hard and votes loud, while a round near $\varepsilon=\tfrac12$ barely moves anything and barely counts.

Landing this in *this* harness rather than the textbook forces two deviations I must respect. For **classification**, the fixed loop's discrete head already accumulates $\alpha\,(2\,\text{pred}-1)$ and thresholds at zero — exactly the signed weighted vote — *provided* `compute_targets` hands the tree integer labels so the loop routes it to that discrete head (it tests `np.array_equal(pt, pt.astype(int))`). So I return the raw $y$, not residuals. The vote weight is the measured-error coefficient $\tfrac12\log\frac{1-\varepsilon}{\varepsilon}$, but the harness deliberately folds the shrinkage *into* $\alpha$: it writes $\alpha = \text{lr}\cdot\tfrac12\log\frac{1-\varepsilon}{\varepsilon}$. This is not the standard discrete-AdaBoost coefficient (which carries no learning rate); it is this task's choice, because the discrete head adds $\alpha(2\,\text{pred}-1)$ with no separate `learning_rate` multiplier, so the only place shrinkage can enter the vote is inside $\alpha$. The reweighting then uses the same coefficient, $w_i \leftarrow w_i\exp(\alpha\,\mathbf{1}[\text{wrong}])$ — the one-sided $\{0,1\}$ form that promotes the wrong examples and leaves the right ones, equivalent after renormalization to the signed $\exp(-\alpha\,y\,h)$ update.

For **regression** the harness diverges from AdaBoost.R2 in a way I cannot paper over. The textbook cousin bounds a per-example loss to $[0,1]$, reweights by $\beta=\overline{L}/(1-\overline{L})$, and *combines the learners by a weighted median* — because a mean is fragile, one wild learner dragging the ensemble arbitrarily far. But this harness's fixed `ensemble_predict` has no median: for regression it keeps a `MeanPredictor` as the first model and then accumulates $\alpha\cdot\text{lr}\cdot\text{tree.predict}(X)$, a weighted *sum* of residual predictions. The median combiner R2 relies on is simply not exposed; returning raw labels and trying to vote would make the additive accumulator sum whole-label predictions and diverge. So the faithful move is to make the regression branch a residual-fitting stagewise model and keep only the *reweighting* half of R2: `compute_targets` returns $y-\text{current\_predictions}$ (a continuous target the loop routes to its additive head); `compute_learner_weight` returns $\alpha=1.0$, since shrinkage is already supplied by the loop's $\alpha\cdot\text{lr}$ accumulation; and `update_weights` carries R2's emphasis on hard examples — normalize the absolute prediction errors to $[0,1]$ by their max, form $\overline{L}=\sum_i p_i L_i$, set $\beta=\overline{L}/(1-\overline{L})$, and update $w_i \leftarrow w_i\,\beta^{\,1-L_i}$ so well-predicted samples shrink and badly-predicted ones hold, then renormalize.

A few design choices the algebra does not force but I am relying on. The tree stays at depth 3: it only has to clear better-than-chance on each reweighted problem, which a shallow tree almost always does, and it must stay *weak* so each round contributes a little and later rounds still have an edge to find — a fully grown tree would drive $\varepsilon_t$ toward zero, take an enormous $\alpha$, and leave no edge for anyone else. Shrinkage $\text{lr}=0.1$ over 200 rounds gives the ensemble room to accumulate, and the $[10^{-10}, 1-10^{-10}]$ clip on the weighted error keeps the log finite when a round is perfect or hopeless.

I expect this to be strong on breast-cancer classification — its home territory, where reweighting toward the few ambiguous cancer cases and the discrete signed vote are exactly what adaptive boosting does well, with no residual-accumulation noise. The regression tasks are the worry, and the reason is structural: I am running AdaBoost's reweighting on top of a residual-fitting additive model whose combiner is a plain weighted sum, not the robust median R2 was designed around. The reweighting keeps piling mass onto the hardest-to-fit points, and a depth-3 tree fit to residuals on a distribution sharply peaked at the worst residuals chases exactly those points — on a noisy target that is how you overfit the tail, and nothing in the loop averages that injected variance away. If the numbers show strong classification and clearly worse regression than a residual-only alternative, the diagnosis for the next rung is already written: drop the reweighting entirely on regression and fit the negative gradient on the *uniform* distribution.

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
