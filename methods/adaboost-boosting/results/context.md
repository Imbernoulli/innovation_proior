# Context: Boosting Weak Predictions

## Research Question

A learning algorithm is available as a black box. On any distribution over labeled examples, it can return a rule whose error is slightly below one half. The rule is only barely useful: it beats random guessing, but by a margin that may be small. The goal is much stronger. Given a target error epsilon, can repeated access to this weak learner produce a single classifier with error at most epsilon, using only polynomially many examples, calls, and computation?

This is the hypothesis-boosting problem behind the weak-versus-strong learnability question. Strong learnability immediately implies weak learnability. The hard direction is the converse: whether a small predictive advantage on every distribution is already enough to obtain arbitrarily high accuracy.

The black-box condition matters. The outer procedure should not depend on the internal structure of the weak learner. It can choose the distributions on which the weak learner is trained, observe the returned hypotheses, and combine them. A useful solution should also avoid requiring a reliable pre-run estimate of the weak learner's exact edge, because that edge can vary from one induced distribution to the next.

## Background

In the PAC setting, examples are drawn from an unknown distribution and labeled by an unknown target concept. A strong learner must, for arbitrary epsilon and delta, output with probability at least `1-delta` a hypothesis with error at most epsilon. A weak learner relaxes only the accuracy requirement: it needs only to output a hypothesis whose error is bounded below `1/2` by an inverse-polynomial amount.

Reliability amplification is not the central difficulty. Confidence can be increased by independent runs and validation. Accuracy amplification is harder, because running the same weak learner many times on the original distribution can reproduce the same mistakes. The hard region of the distribution may be missed by every hypothesis, so a plain majority vote over same-distribution outputs need not improve the error.

The useful lever is distributional control. If earlier hypotheses already handle some examples, later calls should spend less effort on them and more effort on the examples that remain difficult. The challenge is to make that redistribution precise while keeping the weak learner as a black box.

## Prior Baselines

The first general solution uses recursive filtering. It calls the weak learner on the original distribution, then on a filtered distribution where the first hypothesis has no advantage, then on a distribution supported by disagreements between the first two hypotheses. The majority of the three hypotheses has smaller error, and recursion drives the error down. This proves the equivalence of weak and strong learning, but the final hypothesis is a recursive majority circuit and the analysis is governed by worst-case subcall errors.

A later flat construction replaces the recursive circuit by one majority gate over many weak hypotheses. It derives a nearly optimal weighting strategy from a majority-vote game and achieves strong round-complexity guarantees. Its weakness is that it plans around a fixed edge parameter known before the run. If one weak hypothesis is much better than the worst-case edge and another is barely better than chance, the unweighted final majority cannot fully use that difference.

A separate online-allocation line maintains weights over experts, predicts by weighted vote, and multiplicatively changes weights after losses are observed. It supplies a potential-based way to analyze adaptive sequences. Its objects are experts and trials, not examples and hypotheses, so it does not directly solve the boosting problem, but the mathematical shape is suggestive.

## Evaluation Setting

The theoretical yardstick is empirical or distributional classification error, together with the number of weak-learner calls needed to reach a target epsilon. The desired guarantee has the form: if every generated distribution admits a weak hypothesis with error at most `1/2 - gamma`, then the combined classifier's error should shrink exponentially in the number of rounds.

For a batch implementation, the input is a fixed labeled training set. The outer procedure maintains a distribution over training examples, calls a weighted weak learner, measures weighted error, updates the distribution, and returns a combined classifier. Standard weak learners include decision stumps and shallow decision trees, because they are cheap and can be trained under sample weights.

Generalization is evaluated by comparing training error with held-out or population error. A direct complexity analysis treats the final classifier as a thresholded linear combination of `T` base hypotheses, so the size of `T` matters. A more refined analysis may also need to account for the confidence of the final vote on each example.

## Code Framework

```python
import numpy as np


class WeakLearner:
    def fit(self, X, y, sample_weight):
        # Return a classifier trained under the supplied distribution.
        return self

    def predict(self, X):
        raise NotImplementedError


def weighted_error(h, X, y, w):
    return np.sum(w * (h.predict(X) != y))


def boost(X, y, rounds, weak_learner_factory):
    n = len(y)
    w = np.full(n, 1.0 / n)
    hypotheses = []
    coefficients = []

    for _ in range(rounds):
        h = weak_learner_factory().fit(X, y, sample_weight=w)
        err = weighted_error(h, X, y, w)

        coefficient = None
        # TODO: choose how much this hypothesis should count from err.
        # TODO: update w so later rounds focus on examples still causing errors.

        hypotheses.append(h)
        coefficients.append(coefficient)

    return hypotheses, coefficients


def predict(hypotheses, coefficients, X):
    # TODO: combine weak hypotheses into a final classifier.
    raise NotImplementedError
```
