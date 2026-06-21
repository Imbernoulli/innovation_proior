# Context: forward-stagewise tree ensembles for general losses (circa 2014-2016)

## Research question

On structured, tabular data — spam, ad click-through, fraud, web-search ranking, physics event
classification — the learner that most often wins is an ensemble of shallow decision trees grown
by boosting: build trees one at a time, each correcting what the ensemble so far gets wrong, and
sum their outputs. The accuracy is real and repeatable. But the recipe that produces it is not one
clean procedure; it is a family of loosely related strategies that each differ in three concrete
choices a boosting round has to make:

1. **What target does the next tree fit?** The original labels, the current errors, the negative
   gradient of the loss, or something else built from the loss?
2. **How is each tree's contribution weighted before it is added?** A coefficient read off the
   weighted error, a fixed step shrunk by a learning rate, or a separately optimized scalar — and,
   for a regression tree, is that scalar shared by the whole tree or chosen per leaf?
3. **How are training instances re-weighted between rounds?** Up-weight the ones the ensemble still
   gets wrong, or leave the weights uniform and let the targets carry the signal?

These three knobs are not independent: a target choice forces a particular leaf-value choice, which
forces a particular reweighting. Different settings give AdaBoost, gradient boosting, and the
statistical "additive logistic regression" reading of boosting, and they do not all behave the same
across loss functions — some are tied to classification, some to squared-error regression, some need
an extra one-dimensional optimization that has no closed form for a general loss. The problem is to
find a strategy for these three knobs that works for both classification and regression under any
reasonable differentiable loss. The weak learner is fixed: a shallow regression tree (here depth 3).
The contribution to find is the strategy, not the tree learner.

## Background

The field already agrees on the *shape* of the answer. A tree ensemble predicts
`ŷ_i = Σ_{k=1}^K f_k(x_i)`, where each `f_k` is a regression tree: a structure `q` mapping an
example to a leaf index, and a vector `w` of continuous leaf scores, so `f(x) = w_{q(x)}`. It is
trained *additively* — the model at round `t` is the model at round `t-1` plus one new tree — because
the parameters are tree structures and leaf scores, not points in a Euclidean space, so ordinary
gradient descent on the parameters does not apply. This forward-stagewise view is settled (Friedman,
Hastie & Tibshirani 2000; Friedman 2001): earlier trees are frozen, each new tree is chosen to most
reduce the training loss, and the only question is *how* to choose it.

Two facts about decision-tree learning are load-bearing here. First, a tree is grown greedily: start
from a single leaf and repeatedly pick the feature and threshold whose split most improves a quality
score, recursing until a depth or gain limit. The whole inner loop of boosting is therefore a
*scoring function for candidate splits* — give the tree learner a good score and it will find good
trees. For a plain regression tree the score is variance reduction / squared-error impurity; what
score a *boosting* round should use is exactly one of the open choices. Second, once a tree's
structure is fixed, its leaf scores are free real numbers chosen to minimize the round's objective —
and for squared error that optimum is just the mean of the leaf's targets, but for a general loss it
is whatever value minimizes the loss summed over the leaf's instances.

A second settled idea is that the natural per-instance signal for "what the next tree should fix" is
the *derivative of the loss at the current prediction*. If `l(ŷ, y)` is the per-example loss and
`ŷ^{(t-1)}` the current prediction, then the first derivative `g_i = ∂l/∂ŷ^{(t-1)}` points in the
direction that locally increases the loss, so its negative is the locally optimal direction to nudge
the prediction. Friedman's function-space view treats `{F(x_i)}` as the parameters and `-g_i` as the
steepest-descent step in that space. This first-order quantity is the backbone of gradient boosting.
A further fact, established in the statistical reading of boosting, is that the loss also has a
*second* derivative `h_i = ∂²l/∂ŷ^{(t-1)²}` — its local curvature — and that for some losses this
curvature varies strongly from instance to instance (it is constant only for squared error).

A third settled idea, observed and then explained: boosting overfits if left unchecked, and two
cheap regularizers reliably help. Shrinkage (Friedman 2002) scales each newly added tree by a small
factor `η` before adding it, so no single tree dominates and later trees retain room to correct;
empirically `η ≈ 0.1` with many rounds beats `η = 1` with few. Column (feature) subsampling, borrowed
from Random Forests (Breiman 2001), considers only a random subset of features per tree (or per
split) and is reported to curb overfitting at least as well as row subsampling. Both are knobs on
*how much* of each tree to trust, not on the per-round objective itself.

## Baselines

These are the prior strategies a new boosting strategy is measured against and reacts to.

**AdaBoost (Freund & Schapire 1997).** For binary labels `y ∈ {-1, +1}`, keep a weight `w_i` on each
training instance (initially uniform). Each round, fit a weak classifier `h_t` to the *weighted*
   data, measure its weighted error `err = Σ_i w_i 1[y_i ≠ h_t(x_i)] / Σ_i w_i`, give it coefficient
`α_t = ½ log((1-err)/err)`, and update each weight by
`w_i ← w_i exp(-α_t y_i h_t(x_i))`, then renormalize. Equivalently, relative to a correctly
classified point, a misclassified point receives an `exp(2α_t)` multiplier. The final prediction is
`sign(Σ_t α_t h_t)`. Friedman, Hastie & Tibshirani (2000) showed this is exactly stagewise fitting of
an additive model under the exponential loss `Σ_i exp(-y_i F(x_i))`: `α_t` and the reweighting both
fall out of minimizing that loss.

**Gradient boosting (Friedman 2001).** Treat the predictions as parameters and do steepest descent
in function space. Each round, compute the pseudo-residual `ỹ_i = -g_i = -∂l(y_i, F(x_i))/∂F(x_i)` at
the current model, fit a regression tree to the `ỹ_i` by least squares (so the tree's *structure* is
found by ordinary variance-reduction on the negative gradients), then choose the tree's contribution
by a line search `ρ = argmin_ρ Σ_i l(y_i, F_{m-1}(x_i) + ρ h(x_i))`, and add `η · ρ · h`. For a
regression tree Friedman refines this: rather than one shared `ρ`, re-optimize *each leaf* `j`
separately, `γ_j = argmin_γ Σ_{i ∈ leaf j} l(y_i, F_{m-1}(x_i) + γ)`, because the leaves partition the
instances and their updates are independent. Sample weights stay uniform; all the boosting signal is
in the `ỹ_i`. This is general — any differentiable loss gives a pseudo-residual — and for squared
error it reduces to the familiar "fit the residuals."

**LogitBoost / the second-order statistical reading (Friedman, Hastie & Tibshirani 2000).** For the
binomial log-likelihood, the same authors derived an *adaptive Newton* algorithm: at the current
probabilities `p_i`, form a working response `z_i = (y_i^* - p_i)/(p_i(1-p_i))` (with `y^* ∈ {0,1}`)
and per-instance weights `u_i = p_i(1-p_i)`, fit the next function by *weighted* least squares of `z`
on `x`, and add half of it. This is a Newton step on the log-likelihood — it uses both the first
derivative (through `y^*-p`) and the second derivative (through `p(1-p)`), and the weighted-least-
squares fit is the data-version of `F ← F - H^{-1}s`. They also showed AdaBoost itself is the Newton
step for the exponential loss.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Breast Cancer Wisconsin** — binary classification, 569 instances, 30 continuous features; metric
  test accuracy (higher is better). A small, clean, separable-ish medical dataset.
- **Diabetes** — regression, 442 instances, 10 features; metric test RMSE (lower is better). A small
  regression benchmark with modest signal.
- **California Housing** — regression, 20,640 instances, 8 features; metric test RMSE. A larger,
  noisier regression task.
- Protocol: a fixed 80/20 train/test split; a fixed number of boosting rounds (200); the weak learner
  fixed to a depth-3 decision tree; a global learning-rate / shrinkage of 0.1. The strategy under
  test owns only the four per-round decisions (initial weights, pseudo-targets, learner weight,
  sample reweighting); the tree fitting and the outer loop are held constant. Both a classification
  loss and a squared-error regression loss must be handled by the *same* strategy.

## Code framework

The strategy plugs into a fixed boosting harness. The harness owns the outer loop: it initializes
sample weights, then for each round computes per-instance pseudo-targets, fits one depth-3 regression
tree to those targets (weighted by the sample weights), asks the strategy for a scalar `α` for that
tree, folds `η · α · tree.predict(x)` into the running predictions, and asks the strategy for the next
round's sample weights. What target each tree fits, how `α` is set, and how the weights evolve are
exactly the open choices — so the substrate exposes only generic, already-existing primitives
(`numpy`, a `DecisionTreeRegressor`, the current predictions, the round index, and a config with the
task type and learning rate), and leaves the four decisions as empty stubs.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BoostingStrategy:
    """The four per-round decisions of a boosting strategy. The outer loop, the
    depth-3 tree, and the shrinkage are fixed by the harness; only these stubs
    define a particular strategy."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]          # "classification" | "regression"
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        # TODO: any per-round state the strategy needs to carry across rounds.

    def init_weights(self, n_samples):
        # Initial sample weights (sum to 1).
        # TODO
        pass

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # The per-instance pseudo-targets the next tree will fit.
        # TODO
        pass

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # The scalar alpha for the just-fitted tree.
        # TODO
        pass

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # Sample weights for the next round.
        # TODO
        pass


# fixed harness the strategy plugs into
def boost(X, y, strategy, n_rounds, learning_rate):
    n = X.shape[0]
    sample_weights = strategy.init_weights(n)
    predictions = np.zeros(n)
    learners, alphas = [], []
    for t in range(n_rounds):
        targets = strategy.compute_targets(y, predictions, sample_weights, t)
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, targets, sample_weight=sample_weights)
        alpha = strategy.compute_learner_weight(tree, X, y, targets, sample_weights, t)
        predictions = predictions + learning_rate * alpha * tree.predict(X)
        sample_weights = strategy.update_weights(sample_weights, tree, X, y,
                                                 targets, alpha, t)
        learners.append(tree); alphas.append(alpha)
    return learners, alphas
```

The four `# TODO` stubs are the strategy; everything else is the fixed harness.
