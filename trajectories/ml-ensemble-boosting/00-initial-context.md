## Research question

Boosting builds a predictor as a sum of weak learners fit one after another, each round trying to correct what the previous rounds left wrong. With the weak learner fixed as a shallow `DecisionTree(max_depth=3)` and the training loop, aggregation, and evaluation frozen, the only design space is the **boosting strategy**: how sample weights start, what pseudo-target each new tree fits, how each tree's contribution (`alpha`) is set, and how sample weights are updated for the next round. The same strategy must work for one binary classification task and two regression tasks, because the harness runs it unchanged on all three.

## Prior art / Background / Baselines

- **Weighted majority / multiplicative weights.** Maintain a weight per expert, multiply down the ones that err, and predict by weighted vote.
- **Boosting by filtering / majority-of-three.** Run the weak learner on filtered distributions and combine the results to show that weak learnability implies strong learnability. The construction is a recursive circuit with fixed error thresholds.
- **Forward stagewise additive modeling.** Fit an additive model greedily, one term at a time, by least-squares against the current residual. This gives a clean stage-wise update for squared error.

## Fixed substrate / Code framework

The boosting loop in `scikit-learn/custom_boosting.py` is frozen. It calls `init_weights(n)` once, then for `n_rounds = 200` rounds calls `compute_targets`, fits a `DecisionTree(max_depth=3)` on `(X, pseudo_targets, sample_weights)`, calls `compute_learner_weight` for `alpha`, and calls `update_weights`. Weights are then renormalized and clipped positive, and the new tree is folded into a running raw-score accumulator.

- **Regression** starts from a `MeanPredictor(y_train.mean())` and accumulates `alpha * learning_rate * tree.predict(X)`.
- **Classification** uses a discrete head (`alpha * (2*pred - 1)` thresholded at zero) when pseudo-targets are integers and a continuous head (accumulating `alpha * learning_rate * pred`) when they are continuous. The loop selects automatically by checking `np.array_equal(pt, pt.astype(int))`.

The strategy never writes leaf values and never sees the tree's splits. Available imports in the fixed section are `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`, and `sklearn.model_selection`. The strategy receives a `config` dict with `task_type`, `n_rounds`, `learning_rate` (`0.1`), `n_samples`, `n_features`, `dataset`, and `seed`.

## Editable interface

Only the `BoostingStrategy` class is editable. It implements four methods:

- `init_weights(n_samples)` — the starting sample distribution.
- `compute_targets(y, current_predictions, sample_weights, round_idx)` — what the next tree fits.
- `compute_learner_weight(learner, X, y, pseudo_targets, sample_weights, round_idx)` — the tree's `alpha`.
- `update_weights(sample_weights, learner, X, y, pseudo_targets, alpha, round_idx)` — the next round's sample weights.

The default fill is the inert strategy below. Each method replaces exactly these four bodies and nothing else.

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

Three datasets over seeds {42, 123, 456}, with a single configuration: 200 boosting rounds, `DecisionTree(max_depth=3)`, `learning_rate = 0.1`, 80/20 train/test split, standardized features.

- **Breast Cancer Wisconsin** — binary classification, 569 samples, 30 features → `test_accuracy_breast_cancer` (higher is better).
- **Diabetes** — regression, 442 samples, 10 features → `test_rmse_diabetes` (lower is better).
- **California Housing** — regression, 20,640 samples, 8 features → `test_rmse_california_housing` (lower is better).
