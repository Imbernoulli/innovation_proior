# Context: turning weak learners into accurate predictors (circa 1996)

## Research question

In PAC learning, a *strong* learner can achieve arbitrarily small error given enough data; a *weak* learner only guarantees error slightly below random guessing on every distribution. Kearns and Valiant asked whether the two notions are equivalent: can a black-box weak learner be called repeatedly and its outputs combined to drive error to zero? The setting is how to call a weak learner over a sequence of rounds and combine its hypotheses into one accurate predictor.

## Background

**Weak versus strong learning.** Strong learning demands error at most any chosen `epsilon`; weak learning fixes a single modest accuracy such as `1/2 - gamma`.

**Distributions over the training set.** A weak learner has an edge on *every* distribution, so one can present it with distributions that emphasize whatever the current committee gets wrong, rather than re-running it on the same distribution each round.

**Reweighting mechanisms.** Two standard ways to show the learner a chosen distribution are *boosting by filtering*, which uses rejection sampling, and *boosting by reweighting*, which keeps a fixed training set and passes example weights (or a resample drawn from those weights). A design choice is how to update the distribution from round to round.

**Multiplicative-weight updates.** The online allocation / experts literature provides a template: maintain weights over options, decay the ones that perform badly, and bound total loss via a potential on the sum of weights.

**Weak learners in practice.** The base learner is deliberately simple and high-bias: a decision stump or a shallow decision tree. Such rules are cheap and usually beat chance on reweighted data.

## Baselines

**Schapire (1990) three-distribution construction.** It proves weak implies strong by running the learner on the original distribution, then on a distribution where the first hypothesis is neutralized, then on one where the first two disagree, and taking a majority vote of the three resulting hypotheses. The final predictor is a recursive majority circuit, with each sub-call charged at a worst-case error level.

**Freund (1995) boosting by majority.** It flattens the recursion into a single majority vote over many sequentially generated hypotheses, using a binomial-tail weighting schedule that is near the information-theoretic minimum. The schedule is computed from a fixed edge `gamma` supplied before the run, and every hypothesis counts equally in the final vote.

**Breiman (1996) bagging.** It trains predictors on independent bootstrap resamples of the data and aggregates them by voting or averaging. Diversity comes from random resampling, and every member receives equal weight.

## Evaluation settings

- UCI classification benchmarks, evaluated by test-set error under cross-validation or repeated train/test splits, with decision stumps or shallow trees as the base learner.
- Binary classification with labels in `{-1, +1}` or `{0, 1}`; training error of the combined predictor and per-round weighted error of each weak hypothesis are tracked across rounds.
- Regression benchmarks with a continuous target, evaluated by RMSE or MAE on a fixed test split, using shallow regression trees.
- Protocol: base learner, number of rounds, and data split are held fixed across methods; each combiner receives the same weighted training set.

## Code framework

The boosting harness already exists. An outer loop fits a fresh weak learner on the current weighted training set each round and then asks a strategy object for the learner's coefficient and the next round's weights. Everything specific to the combiner lives behind the strategy interface below.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BoostingStrategy:
    """The per-round decisions of a sequential ensemble. Everything here is the
    open design: how examples are weighted, what each weak learner fits, how much
    each learner counts in the final vote, and how the weights move for next round."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]          # "classification" or "regression"
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        # Starting distribution over the training examples (should sum to 1).
        # TODO: the initial weighting we will choose.
        pass

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # What the next weak learner is asked to fit.
        # TODO: the per-round target we will design.
        pass

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # Whether the just-fitted learner is usable, and how much it counts if it is.
        # TODO: the acceptance test and coefficient we will design.
        pass

    def update_weights(self, sample_weights, learner, X, y,
                       pseudo_targets, learner_weight, round_idx):
        # The distribution over examples for the next round.
        # TODO: the reweighting rule we will design.
        pass


# fixed harness the strategy plugs into
def fit_ensemble(X, y, strategy, make_weak_learner):
    sample_weights = strategy.init_weights(len(y))
    learners, learner_weights = [], []
    predictions = np.zeros(len(y))                     # current ensemble output on train
    for t in range(strategy.n_rounds):
        targets = strategy.compute_targets(y, predictions, sample_weights, t)
        learner = make_weak_learner()                  # a shallow tree
        learner.fit(X, targets, sample_weight=sample_weights)
        learner_weight = strategy.compute_learner_weight(learner, X, y, targets,
                                                         sample_weights, t)
        if learner_weight is None:
            break
        learners.append(learner); learner_weights.append(learner_weight)
        sample_weights = strategy.update_weights(sample_weights, learner, X, y,
                                                 targets, learner_weight, t)
        predictions = ensemble_predict(learners, learner_weights, X, strategy)  # aggregate so far
    return learners, learner_weights
```

The harness supplies the fitted learner, the current weights, the labels, and the round index; the four stubs are where the combiner's decisions go.
