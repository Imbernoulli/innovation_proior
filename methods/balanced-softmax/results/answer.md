# Balanced Softmax, distilled

Balanced Softmax is a drop-in replacement for the ordinary Softmax cross-entropy loss for
long-tailed classification. During training it adds the per-class log-count `log n_j` to each logit
before applying standard cross-entropy; at test time it uses the bare logits. This makes the
network's logits model the *balanced* test posterior while it trains on the imbalanced split — a
train-only, parameter-free correction of the class prior baked into the Softmax.

## Problem it solves

A classifier trained by cross-entropy on a long-tailed split (per-class counts `n_1, ..., n_k`,
`Σ_j n_j = n`) fits the *training* posterior `p̂(y|x) ∝ p(x|y)·(n_y/n)`, but is graded on a
*balanced* test set, scored by top-1 / balanced accuracy, which wants the *balanced* posterior
`p(y|x) ∝ p(x|y)·(1/k)`. The Softmax is a faithful estimator of the wrong posterior: the head-heavy
training prior is baked in and absent at test, so the argmax favors head classes and the tail is
under-predicted.

## Key idea

The training and test posteriors share the class-conditional likelihood `p(x|y=j)` (same images) and
differ only in the class prior. Demanding that the logits `η_j = θ_j^T f(x)` are the standard Softmax
of the *balanced* posterior `φ`, and asking what the *training* posterior `φ̂` is in those same `η`,
gives an exact identity:

```
φ̂_j = n_j e^{η_j} / Σ_{i=1}^k n_i e^{η_i}.
```

The training posterior is the balanced one with the prior (the count) multiplied back inside the
Softmax — additive in log-space. So train against `φ̂`, i.e. minimize

```
l̂(θ) = -log φ̂_y = -log Softmax(η + log n)_y,
```

which is just: add `log n_j` to each logit, then ordinary cross-entropy. At test, drop the shift and
predict `argmax_j η_j`, because `η` now models the balanced posterior.

Why a *log-count* offset (not `1/n`, not `n^{1/4}`): the Bayes-optimal rule for balanced error is
`argmax_y p̂(y|x)/p̂(y) = argmax_y [η_y − log n_y]`, i.e. divide out the training prior, which is
exactly a per-logit log-count shift. Why *additive on logits* (not a scalar loss reweight): the prior
enters the posterior multiplicatively *inside* the
normalizer (`n_i e^{η_i}`), so it must be corrected inside the sum; a class-constant loss weight is
model-blind, can be invariant to the converged max-margin solution on separable data, and produces
abnormally large gradients at severe imbalance. The log-count shift is bounded and grows only
logarithmically with `n_j`, so it avoids that instability, and it reduces to plain cross-entropy when
the data is balanced.

## Two derivations agree on the form

1. **Label shift (exact).** From the exponential-family canonical link `η_j = log(φ_j/φ_k)`, add
   `−log(φ_j/φ̂_j)` to both sides, normalize via `Σ_j φ̂_j = 1`; the shared likelihood and the
   class-independent evidence/`log(n/k)` terms cancel, leaving `φ̂_j ∝ n_j e^{η_j}`. Fixes the
   exponent at `a = 1`.
2. **Margin bound (confirms the form).** The balanced generalization bound
   `err_bal ≲ (1/k) Σ_j [(1/γ_j)√(C/n_j) + (log n)/√n_j]` under a budget `Σ_j γ_j = β`, minimized by
   Cauchy-Schwarz, gives the optimal per-class margin `γ*_j = β n_j^{-1/4}/Σ_i n_i^{-1/4}` (rarer
   classes deserve larger margins). Enforcing it as a shifted Softmax gives `φ̂_j ∝ n_j^{1/4}
   e^{η_j}` — the same family, exponent `1/4`. This route is a relaxation; the exact label-shift
   identity fixes the exponent at `1`.

## Caveats and variants

- **Do not stack with a hand-set class-balanced sampler.** The loss already encodes the full
  rebalancing; combining it with uniform-per-class sampling double-counts. The zero-gradient condition
  first has `(1/n_j)E_{y=j}[f(1-φ̂_j)] - Σ_{i≠j}(1/n_i)E_{y=i}[fφ̂_j]=0`; as `φ̂_y → 1`, the negative-sample
  ratio `φ̂_j/φ_j → n_j/n_i`, and dividing the resulting equation by `n_j` gives an effective `1/n_j^2`
  weighting. Use instance-balanced sampling; if resampling is needed at extreme imbalance, the rate
  must be learned, not set to uniform.
- **Binary-logistic heads (detection/segmentation).** By Bayes' theorem each binary logit is shifted
  by `−log[ ((n/k)/n_j)·((n − n_j)/(n − n/k)) ]`, an extra not-class-`j` prior factor beyond the
  Softmax case; apply to foreground classes only and leave background on the plain sigmoid to avoid a
  flood of false positives.

## Working code

```python
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    """Balanced Softmax cross-entropy: shift each logit by + log(n_class), then
    standard cross-entropy. Trains logits to model the balanced-test posterior."""
    spc = sample_per_class.type_as(logits)               # n_j per class
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)   # [batch, k]
    logits = logits + spc.log()                          # eta_j + log n_j
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class BalancedSoftmax(_Loss):
    """Loss module holding the per-class training counts n_1..n_k."""

    def __init__(self, sample_per_class):
        super().__init__()
        self.sample_per_class = torch.as_tensor(sample_per_class)

    def forward(self, logits, labels, reduction='mean'):
        return balanced_softmax_loss(labels, logits, self.sample_per_class, reduction)


# Test time: NO shift — eta already models the balanced posterior.
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)
```

For a multiple-binary-logistic head, replace the additive shift with the per-class binary offset
`b_j = log[ ((n/k)/n_j)·((n − n_j)/(n − n/k)) ]` applied as `η_j − b_j` to foreground classes only,
then sigmoid binary cross-entropy.
