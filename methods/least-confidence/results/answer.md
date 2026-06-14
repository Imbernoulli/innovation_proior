# Least Confidence (Uncertainty Sampling)

Least-confidence sampling is a simple pool-based active-learning acquisition rule. Given a
probabilistic classifier trained on the labels gathered so far, it queries the unlabeled examples
the classifier is *least confident* about — the ones with the smallest top-class posterior
probability — sends them to the oracle, retrains, and repeats. It is the canonical instance of
**uncertainty sampling**.

## Problem it solves

Labels are expensive (a human oracle) and unlabeled examples are abundant and cheap. Under a
fixed labeling budget, uniform random sampling wastes labels on examples the classifier already
handles and, for rare classes, returns almost no positives. The goal is to choose which pool
examples to label so the classifier reaches a target effectiveness with as few labels as
possible.

## Key idea

Learning shrinks a version space; only examples in the **region of uncertainty** (where two
still-consistent hypotheses disagree) inform the learner, and a random draw lands there with a
probability that decays to zero. Querying inside that region is the fix, but computing it
exactly needs the whole version space (intractable) and committee approximations need
noise-free realizability and version-space sampling. A single probabilistic classifier already
reports where it is on the fence: its posterior. So use the classifier's own uncertainty about
its prediction as a soft, single-model, rankable surrogate for "inside the region of
uncertainty," and spend the budget on the most uncertain examples.

- **Binary case:** query the examples whose posterior `P(C | x)` is nearest `0.5` (the decision
  boundary under equal misclassification costs). Equivalently, with `p = P(C | x)`,
  `max(p, 1 - p) = 0.5 + |p - 0.5|`, so minimizing top-class confidence is exactly choosing
  the posterior nearest `0.5`.
- **Multi-class generalization (least confidence):** let `ŷ = argmax_y P(y | x)` be the
  predicted class. Score each example by

  ```
  x*_LC = argmax_x [ 1 - P(ŷ | x) ] = argmax_x [ 1 - max_y P(y | x) ],
  ```

  i.e. query the examples with the *smallest* top-class posterior. This equals the model's own
  estimate of the probability it will mislabel `x` (expected 0/1 loss), and reduces to the
  nearest-`0.5` rule when there are two classes.

## Why least confidence (and not margin or entropy)

Least confidence uses only the most probable class, "throwing away" the rest of the
distribution. Two refinements use more of it:

- **Margin** (smallest `P(ŷ1 | x) - P(ŷ2 | x)`) — adds the runner-up class, so it prefers
  examples where the model is torn between two specific classes.
- **Entropy** (largest `-Σ_y P(y | x) log P(y | x)`) — uses the whole distribution, and
  generalizes to structured/multi-label outputs.

In the binary case all three give the same query ordering: they all choose the posterior nearest
`0.5`. They differ only with three or more classes. Entropy is the more natural uncertainty
measure when the loss of interest is log-loss; least confidence and margin are more directly
tied to decision uncertainty under classification (0/1) error. Least confidence is the minimal,
most direct form — it scores nothing but the top-class probability the classifier already exposes,
costs nothing beyond the forward pass, and equals the model's own misclassification probability.

## Algorithm (one round of the pool-based loop)

```
given: classifier trained on the current labeled set, unlabeled pool U, budget n
1. for each x in U: compute posterior P(. | x); confidence(x) = max_y P(y | x)
2. select the n examples in U with the smallest confidence(x)
3. send them to the oracle, add the returned labels to the labeled set
4. retrain the classifier on the enlarged labeled set
repeat until the budget is exhausted
```

The selection is only as good as the current classifier; a deficiency this round tends to be
compensated by the examples it selects next round (the exploratory feedback). A non-trivial
initial classifier matters, so the loop is seeded with a small labeled set (e.g. a few known
positives plus a few random examples) before the first query.

## Working code

The canonical BADGE query strategy. `predict_prob` returns posterior probabilities of shape
`[m, n_classes]`; `.max(1)[0]` is the top-class probability per example; `.sort()[1][:n]` are
the indices of the `n` smallest:

```python
import numpy as np
from .strategy import Strategy


class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled],
                                  np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]                       # confidence in the predicted class
        return idxs_unlabeled[U.sort()[1][:n]]    # n least confident (smallest max-prob)
```

Margin and entropy use the same posterior distribution but change the score: margin sorts by
smallest `P(ŷ1 | x) - P(ŷ2 | x)`, while entropy sorts by largest
`-Σ_y P(y | x) log P(y | x)`. They are useful refinements, not the canonical least-confidence
query rule above.
