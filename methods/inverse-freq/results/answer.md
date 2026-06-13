# Inverse class frequency reweighting, distilled

Inverse class frequency reweighting assigns each class a loss weight inversely proportional to
its training count, `w_c = N / (C · n_c)`, and hands that length-`C` vector to a weighted
cross-entropy. It is the unique importance-sampling weight that turns the standard average-loss
objective — which optimizes performance under the skewed *training* class prior — into an
unbiased estimator of the loss under the *balanced* test prior. It is also exactly scikit-learn's
`class_weight="balanced"` heuristic and the multi-class specialization of the classical
weighted-likelihood prior correction (Manski & Lerman 1977; King & Zeng 2001), and reads
cost-sensitively (Elkan 2001) as "make every class contribute equally to the loss."

## Problem it solves

Class-imbalanced (long-tailed) classification: a few head classes hold most of the training
examples, most tail classes hold very few, and plain cross-entropy — which minimizes the average
loss — is dominated by the head and biased toward predicting frequent classes. The model is
scored on a *balanced* test set where every class counts equally. The objective being minimized
(loss under the training prior `n_c/N`) is not the objective being scored (loss under the uniform
prior `1/C`). The fix must act only through the loss, depend only on the per-class counts, and
leave the data, sampler, model, optimizer, and metric untouched.

## Key idea

Plain cross-entropy estimates `E_{(x,y)~p}[L] = Σ_c p(y=c) · E[L|y=c]`, weighting each class's
conditional loss by the empirical class prior `p(y=c) = n_c/N`. The balanced target is
`E_q[L] = Σ_c (1/C) · E[L|y=c]`. The importance-sampling identity

```
E_q[L] = Σ_c q(c) E[L|c] = Σ_c p(c) · (q(c)/p(c)) · E[L|c] = E_{p}[ (q(y)/p(y)) · L ]
```

makes the per-class importance weight the ratio of target prior to training prior:

```
w_c = q(c) / p(c) = (1/C) / (n_c/N) = N / (C · n_c).
```

Then `R_w(f) = (1/N) Σ_i w_{y_i} L(f(x_i), y_i)` is an **unbiased estimator of the balanced
risk**: `E[R_w] = Σ_c p(c) · (q(c)/p(c)) · E[L|c] = Σ_c q(c) E[L|c] = E_q[L]`. Any weight other
than `q/p` leaves a residual `p(c)` factor and estimates a different objective.

- **Inverse in the count** falls out of the ratio, not as a heuristic: small `n_c` ⇒ large `w_c`.
- **The `N/C` constant** sets the average weight per example to 1: `Σ_c n_c·w_c = Σ_c N/C = N`,
  so the weighted loss is on the same scale as unweighted CE and the effective step size (with a
  fixed learning rate) is unchanged. The bare content is `w_c ∝ 1/n_c`.
- **Equal-contribution reading:** total contribution of class `c` is `n_c·w_c·(mean L) =
  (N/C)·(mean L)`, count-independent — every class contributes equally, the cost-sensitive
  "every class equally important" stance, with cost inversely proportional to frequency.
- **Lineage:** `w_c = q/p` is the multi-class weighted exogenous-sampling MLE
  (`w = (target prior)/(observed prior) = τ/ȳ`) with a uniform target prior; cost-sensitively it
  is Elkan's per-class-weight = cost.

## Normalization

The `N/C` factor is part of the canonical importance ratio, not an arbitrary decoration. With
`R_w(f) = (1/N) Σ_i w_{y_i} L_i`, choosing `w_c = N/(C·n_c)` gives
`E[R_w] = E_q[L]` exactly and sets the total weighted mass to `Σ_c n_c·w_c = N`, so the average
example weight is 1. A different global rescale would preserve the relative class tradeoff but
would estimate a rescaled risk `κ E_q[L]` and change the loss magnitude under a fixed learning
rate. The canonical balanced rule keeps the exact unbiasedness statement and the ordinary
average-loss scale simultaneously.

## Caveat (what it does and does not fix)

`w_c = q/p` corrects the *bias* of the prior, not the *variance* of the per-class loss estimate.
Under extreme imbalance the tail weight grows without bound (`∝ 1/n_c`), so the high-variance
loss estimate from a handful of noisy, near-duplicate tail samples is amplified rather than
repaired — a variance bottleneck, not a bias one. This is the known weakness of pure
inverse-frequency weighting on heavy long-tail data; the canonical inverse-frequency rule remains
the plain `N/(C·n_c)` prior correction.

## Working code

The pure count-to-weight hook, grounded in the canonical "balanced" weighting
(`n_samples / (n_classes · bincount(y))`), filling the open slot in the harness:

```python
import torch


def compute_class_weights(class_counts, num_classes, config):
    """Inverse class frequency weighting: w_c = N / (C * n_c).

    The importance ratio (uniform target prior 1/C) / (empirical class prior n_c/N).
    Inversely proportional to the class count; N/C fixes the per-example average weight to 1.
    Pure (counts only). Returns a length-C tensor for nn.CrossEntropyLoss(weight=...).
    Every class must be present, matching the balanced-weighting assumption p(c) > 0.
    """
    counts = class_counts.float()
    if torch.any(counts <= 0):
        raise ValueError("balanced class weights require every class count to be positive")
    total = counts.sum()                                  # N = total training samples
    weights = total / (num_classes * counts)              # N / (C * n_c)
    return weights
```

Equivalent NumPy / scikit-learn `balanced` form (same rule):

```python
import numpy as np


def compute_class_weights_balanced(y, num_classes):
    """w[c] = n_samples / (n_classes * bincount(y)[c]) = N / (C * n_c)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("balanced class weights require every class to appear in y")
    return counts.sum() / (num_classes * counts)          # length-C inverse-frequency weights
```

The weight depends only on the counts, so it is model- and architecture-agnostic and plugs
directly into the class-weighted cross-entropy without changing anything else in the pipeline.
