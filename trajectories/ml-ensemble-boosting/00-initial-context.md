## Research question

Boosting builds a predictor as a sum of weak learners fit one after another, each round trying to
correct what the previous rounds left wrong. With the weak learner *fixed* — a shallow
`DecisionTree(max_depth=3)` — and the prediction-aggregation, training loop, and evaluation all frozen,
the single thing being designed is the **boosting strategy**: how sample weights are initialized, what
pseudo-target each new tree fits, how each tree's contribution (its `alpha`) is set, and how the sample
weights are updated for the next round. One strategy must serve both a binary classification task and
two regression tasks, because the harness runs the same `BoostingStrategy` on all three. Everything
else — the tree, the learning-rate shrinkage, the score accumulation — is substrate.

## Prior art before the first rung (the boosting lineage)

The first rung reacts to the line of additive-model methods that precede it; the fixed substrate below
is the shape they all converged to (sequential weak learners, a per-round contribution, a sample
distribution).

- **Weighted majority / multiplicative weights (Littlestone & Warmuth, 1994).** Maintain a weight per
  expert, multiply down the ones that erred, predict by the weighted vote; the total-weight bookkeeping
  factors cleanly across rounds. The template for "reweight what's wrong," but it combines a *fixed*
  pool of experts — it does not manufacture new hypotheses aimed at the current failures. Gap: no
  mechanism to generate the next weak learner where the committee is currently weakest.
- **Boosting by filtering / majority-of-three (Schapire, 1990).** Proves weak learnability implies
  strong learnability by running the weak learner on three manufactured distributions and recursing on a
  majority-of-three. Establishes that "aim the weak learner at the hard cases" works, but the machine is
  a rigid recursive circuit pinned to a worst-case error level, and a round that comes back unusually
  strong cannot be cashed in. Gap: rigid, non-adaptive, cannot exploit easy rounds.
- **Forward stagewise additive modeling (the classical least-squares residual loop).** Fit
  `F(x) = sum_m beta_m h(x; a_m)` greedily, one term at a time, each term fit to the current residual by
  least squares. Beautiful for squared error, where "fit the residuals" *is* the stage subproblem; but
  for any other loss the per-stage `argmin` over `(beta, a)` has no convenient form, so each loss needs
  its own bespoke procedure. Gap: not a single recipe across losses.

## The fixed substrate

A stagewise tree-boosting loop in `scikit-learn/custom_boosting.py` is frozen and must not be touched.
It (1) calls `init_weights(n)` once; (2) for each of `n_rounds = 200` rounds, asks the strategy for
`compute_targets`, fits a `DecisionTree(max_depth=3)` on `(X, pseudo_targets, sample_weights)`, asks
for `compute_learner_weight` (`alpha`) and then `update_weights`; (3) renormalizes the weights, clips
them positive, and folds the new tree into a running raw-score accumulator. The accumulator is the
load-bearing detail every strategy must respect:

- **Regression** keeps a `MeanPredictor(y_train.mean())` as the first model, then accumulates
  `alpha * learning_rate * tree.predict(X)` per round; the final prediction is that raw score.
- **Classification** routes a tree whose pseudo-targets are integers to a *discrete* head — a signed
  majority vote `alpha * (2*pred - 1)` thresholded at zero — and a tree whose pseudo-targets are
  continuous to a *continuous* head accumulating `alpha * learning_rate * pred`, also thresholded at
  zero. The loop decides discrete-vs-continuous automatically by `np.array_equal(pt, pt.astype(int))`.

So the strategy never writes leaf values and never sees the tree's splits; the only levers are the four
methods. Available in the FIXED section: `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`,
`sklearn.model_selection`. The strategy is constructed with a `config` dict carrying `task_type`,
`n_rounds`, `learning_rate` (`0.1`), `n_samples`, `n_features`, `dataset`, `seed`.

## The editable interface

Exactly one region is editable — the `BoostingStrategy` class (lines 147-256 of `custom_boosting.py`).
Every method on the ladder is a fill of this same four-method contract: `init_weights(n_samples)` (the
starting sample distribution), `compute_targets(y, current_predictions, sample_weights, round_idx)`
(what the next tree fits), `compute_learner_weight(learner, X, y, pseudo_targets, sample_weights,
round_idx)` (the tree's `alpha`), and `update_weights(sample_weights, learner, X, y, pseudo_targets,
alpha, round_idx)` (the next round's sample weights).

The starting point is the scaffold default: **uniform weights, fit the raw labels, alpha = 1, never
reweight** — the inert strategy the loop ships with. Each method replaces exactly these four bodies and
nothing else.

```python
# EDITABLE region of custom_boosting.py (lines 147-256) — default fill (inert strategy)
class BoostingStrategy:
    """Sample weighting and update strategy for gradient boosting.

    The fixed loop calls init_weights() once, then per round: compute_targets ->
    fit DecisionTree(max_depth=3) on (X, pseudo_targets, sample_weights) ->
    compute_learner_weight -> update_weights.  config carries task_type, n_rounds,
    learning_rate, n_samples, n_features, dataset, seed.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        return np.ones(n_samples) / n_samples                # uniform distribution

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        return y                                             # fit the raw labels

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        return 1.0                                           # every tree counts the same

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        return sample_weights                                # no reweighting
```

## Evaluation settings

Three datasets spanning one classification and two regression problems, each over three seeds
{42, 123, 456}, with a single fixed configuration: 200 boosting rounds, `DecisionTree(max_depth=3)`,
`learning_rate = 0.1`, 80/20 train/test split, standardized features.

- **Breast Cancer Wisconsin** — binary classification, 569 samples, 30 features → metric
  `test_accuracy_breast_cancer` (**higher is better**).
- **Diabetes** — regression, 442 samples, 10 features → metric `test_rmse_diabetes` (**lower is
  better**).
- **California Housing** — regression, 20,640 samples, 8 features → metric
  `test_rmse_california_housing` (**lower is better**).
