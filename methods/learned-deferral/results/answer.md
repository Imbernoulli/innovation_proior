# Learning to Defer to an Expert (consistent surrogate), distilled

Learning to defer trains a single model that, for each input, either predicts the label or
**defers** the decision to a downstream expert (a radiologist, a moderator) who may have side
information the model never sees. The contribution is a convex, classification-consistent
surrogate loss — a cost-sensitive generalization of cross-entropy, `L_CE^α` — that jointly
learns the classifier and the rejector with one extra output unit and one backward pass, and
whose population minimizer is the combined-system Bayes-optimal predict-or-defer rule.

## Problem it solves

A fixed classifier sits in front of an expert. We see only samples of the expert's past
decisions `{(x_i, y_i, m_i)}`. For each `x` decide who answers — model or expert — so the
*combined* system error (or fairness, or expert-time budget) is minimized, with the classifier
free to **adapt**: give up on inputs the expert handles well and specialize where only the model
helps. The decision is made from `x` alone, without the expert's side context.

## Setup and Bayes-optimal rule

Classifier `h: X → Y` (`Y = {1..K}`), rejector `r: X → {0,1}` (`r=1` = defer). System 0-1 loss

```
L_{0-1}(h, r) = E[ I[h(x)≠y]·I[r(x)=0] + I[m≠y]·I[r(x)=1] ].
```

With `η_y(x) = P(Y=y|x)`, the pointwise minimizers are

```
h^B(x) = argmax_y η_y(x)
r^B(x) = I[ max_y η_y(x) ≤ P(Y=M|X=x) ]          # defer iff expert-agreement beats top-class confidence
```

Setting `l_exp ≡ c` (a constant) recovers Chow-style learning with a reject option; the
deferral cost `I[m≠y]` is instance-dependent, which is what makes deferral harder.

## Key idea

Treat **defer** as an extra `(K+1)`-th class and learn `K+1` scores `g_1..g_K, g_⊥` with one
softmax. This turns predict-or-defer into cost-sensitive `(K+1)`-class classification
(`argmin_i E[c(i)|x]`, with `c(i)=I[i≠y]` for classes and `c(⊥)=I[m≠y]`).

**Cost-sensitive surrogate (generalizes cross-entropy).**

```
L̃_CE(g, x, c) = − Σ_{i=1}^{K+1} ( max_j c(j) − c(i) ) · log softmax_i(g)(x)
```

Convex; consistent — its minimizer's argmax is `argmin_i E[c(i)|x]`. With misclassification
costs it reduces to plain cross-entropy (`max_j c(j)=1`, weight `I[i=y]`).

**Deferral surrogate.** Plugging the deferral costs in and taking the expectation collapses to
two terms (`h(x)=argmax_{y∈Y} g_y`, `r(x)=I[ max_{y∈Y} g_y(x) ≤ g_⊥(x) ]`):

```
L_CE(h,r,x,y,m) = − log softmax_y(x)  −  I[m=y] · log softmax_⊥(x),
```

cross-entropy toward the label, plus a *defer-where-the-expert-is-right* term gated by the bit
`I[m=y]`. It needs only the target and that one bit — never the expert's actual label.

**Properties.** `L_CE` is convex in `g`, upper-bounds `L_{0-1}` when the log is measured in
bits (the natural-log version is the same loss up to a positive constant), and is consistent:
its minimizer satisfies `softmax_i* = η_i(x)/(1+P(Y=M|x))` and
`softmax_⊥* = P(Y=M|x)/(1+P(Y=M|x))`, so `argmax_y g_y* = h^B` and
`r* defers ⇔ P(Y=M|x) ≥ max_y η_y(x) = r^B`. This settles the open problem of a consistent
*multiclass* reject surrogate, which the binary hinge-based two-function constructions provably
cannot achieve for `K>2`.

**Adaptivity knob `α ≥ 0`.** Down-weight the target term where the expert is correct so the
model stops spending capacity on inputs it can just defer:

```
L_CE^α(h,r,x,y,m) = −( α·I[m=y] + I[m≠y] ) · log softmax_y  −  I[m=y] · log softmax_⊥.
```

`α=1 ⇒ L_CE` (consistent); `α≠1` is a capacity/adaptivity knob, usually validated with
`α<1` to stop spending model capacity where the expert is already correct.

## Why prior approaches fall short

- **Confidence comparison** (independent task model + expert-correctness model, defer to the
  more confident): consistent over all measurable functions, but the classifier is fit ignoring
  the expert, so under limited capacity it cannot specialize away from the expert's strong
  region (two-subpopulation failure); also needs two models, hurting sample complexity.
- **Mixture-of-experts soft gate**: its optimal gate compares classifier *entropy* `H(h^B(x))`
  against `P(Y≠M|x)` instead of *confidence* `max_y η_y(x)`, so it is **not** classification
  consistent (only realizable-consistent for scaling-closed classes), and empirically collapses
  to never deferring as the classifier loss vanishes; it is also non-convex in `(g,r)`.

## Generalization bound

For empirical minimizers `(ĥ*, r̂*)`, w.p. ≥ 1−δ,

```
L_{0-1}(ĥ*,r̂*) ≤ L_{0-1}(h*,r*) + ℜ_n(H) + ℜ_n(R) + ℜ_{nP(M≠Y)/2}(R)
                  + 2√(log(2/δ)/(2n)) + (P(M≠Y)/2)·exp(−nP(M≠Y)/8).
```

When `P(M≠Y)=0` this recovers the pure rejection-learning bound — so deferral is strictly more
sample-intensive than rejection (the extra rejector-complexity term scales with the expert's
error rate). Sharing one `(K+1)`-output backbone, instead of two networks, keeps the cost down.

## Binary case (instance-dependent cost)

Extending the binary plus-hinge `L_SH = exp((α/2)(r−hy)) + (c+I[m≠y])·exp(−βr)`, consistency
requires `β/α = sqrt((1−c(x))/c(x))`; for `α=1`, `β = sqrt((1−c(x))/c(x))`. The effective cost
`c(x) = c − c·P(M≠Y|x) + P(M≠Y|x)` is instance-dependent, so a constant `β` (which sufficed for
constant-cost rejection) is provably inconsistent here. The `(K+1)`-class cross-entropy handles
this per example via `I[m=y]`.

## Working code

The canonical implementation is a backbone with `K+1` output units (units `0..K-1` are the
class logits, unit `K` is the deferral logit), trained with `L_CE^α`. Defer at inference iff
the deferral unit wins the `(K+1)`-way argmax.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeferralNet(nn.Module):
    """Backbone with K+1 outputs: 0..K-1 are class logits (classifier h), K is the
    deferral logit g_bot (rejector r). One shared network does both jobs."""

    def __init__(self, make_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = make_features()
        self.head = nn.Linear(self.features.out_dim, num_classes + 1)   # +1 deferral unit

    def forward(self, x):
        return self.head(self.features(x))                              # raw (K+1) logits


def deferral_loss(outputs, target, expert_label, num_classes, alpha):
    """L_CE^alpha: build the canonical weights and call reject_CrossEntropyLoss.
       outputs:      (B, K+1) raw logits
       target:       (B,) labels y
       expert_label: (B,) expert decisions m
       alpha:        adaptivity knob (alpha=1 -> consistent L_CE)
    """
    probs = F.softmax(outputs, dim=1)                                  # softmax over Y union {defer}
    expert_correct = (expert_label == target)                         # I[m = y]  (gating bit)
    m  = expert_correct.float()                                        # weight on the deferral term
    m2 = torch.where(expert_correct,                                   # alpha*I[m=y] + I[m!=y]
                     torch.full_like(m, alpha), torch.ones_like(m))
    return reject_CrossEntropyLoss(probs, m, target, m2, num_classes)


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    """Canonical batch loss on softmax probabilities.
       m:  I[expert prediction equals target], weight on the deferral term
       m2: alpha*I[expert prediction equals target] + I[expert prediction differs]
    """
    batch_size = outputs.size(0)
    rc = [n_classes] * batch_size                                      # deferral index K
    eps = 1e-12
    loss = -m * torch.log2(outputs[range(batch_size), rc].clamp_min(eps)) \
           -m2 * torch.log2(outputs[range(batch_size), labels].clamp_min(eps))
    return torch.sum(loss) / batch_size


@torch.no_grad()
def predict_or_defer(outputs, num_classes):
    """Defer iff the deferral unit wins the (K+1)-way argmax; else predict argmax over the K
    class logits. Equivalent to r(x) = I[g_bot(x) >= max_y g_y(x)]."""
    top = outputs.argmax(dim=1)
    defer = top == num_classes
    class_pred = outputs[:, :num_classes].argmax(dim=1)
    return defer, class_pred


def coverage_threshold(outputs, num_classes, coverage):
    """For a coverage target: rank by the defer margin q(x) = g_bot - max_y g_y and threshold
    at the coverage quantile (single trained model gives the whole coverage range)."""
    q = outputs[:, num_classes] - outputs[:, :num_classes].max(dim=1).values
    tau = torch.quantile(q, coverage)             # top (1-coverage) by defer margin get deferred
    return q >= tau                                                    # True = defer


def train(model, data_loader, optimizer, expert_fn, num_classes, alpha):
    model.train()
    for x, target in data_loader:                                     # draw a minibatch
        outputs = model(x)                                            # (K+1) logits
        expert_label = expert_fn(x, target)                          # observed expert decisions
        loss = deferral_loss(outputs, target, expert_label, num_classes, alpha)
        optimizer.zero_grad()
        loss.backward()                                              # backprop through shared backbone
        optimizer.step()
```

Tabular per-batch form (same loss, indexing the deferral class at `num_classes`):

```python
def deferral_loss_tabular(probs, target, expert_correct, num_classes, alpha):
    # probs: (B, K+1) softmax outputs; expert_correct: (B,) bool I[m=y]
    m  = expert_correct.float()
    m2 = torch.where(expert_correct, torch.full_like(m, alpha), torch.ones_like(m))
    return reject_CrossEntropyLoss(probs, m, target, m2, num_classes)
```
