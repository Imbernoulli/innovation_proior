# Batch-level reward whitening (batch_std), distilled

`batch_std` is a reward-shaping transform that runs **before** the advantage estimator in
policy-gradient LLM fine-tuning. It standardizes (whitens) the batch of per-response reward
scalars — subtract the batch mean, divide by the batch standard deviation — so the signal fed to
the gradient estimator has approximately zero mean and unit spread, every batch, regardless of
the reward source's scale. It ignores per-prompt grouping: one `(mean, std)` pair for the whole
batch. It is the classic RLHF-style whitening of the reward signal.

## Problem it solves

The policy-gradient estimator averages `R · ∇log π` over a small batch of sampled responses. Two
properties of the raw reward numbers hurt it: (1) **high variance** of the Monte-Carlo estimate,
and (2) an **effective learning rate that drifts with the reward's arbitrary scale and offset**.
For a bounded one-signed reward (binary final-answer correctness), the raw signal also has an
**all-same-sign pathology**: every sampled response gets a non-negative weight, so the policy can
only reinforce, never explicitly suppress, a bad response. The transform must fix these without
biasing the gradient direction and without erasing the good-vs-bad signal.

## Key idea

Whiten the per-response rewards across the batch:

```
scores = token_level_scores.sum(-1)          # one scalar per response
scores = (scores - batch_mean) / (batch_std + eps)
```

The two halves have separate, principled justifications:

- **Subtract the batch mean (center).** Subtracting any reference `b` fixed with respect to the
  sampled action leaves the gradient unbiased — `E[(R−b)∇log π] = E[R∇log π]` because
  `E[∇log π] = ∑_a π ∇log π = ∑_a ∇π = ∇∑_a π = ∇1 = 0` — while changing its variance. The
  variance-minimizing constant reference is `b* = E[R||∇log π||²]/E[||∇log π||²]`, a
  squared-score-norm-weighted mean of the reward; dropping the unavailable per-sample weights
  leaves the plain **batch mean** as its cheap, reward-only plug-in. Using the same empirical
  batch mean for every sample is not the exact action-independent-baseline identity, but its
  finite-batch effect on the centered estimator is a positive scale shrinkage rather than a
  direction change. Centering also cures the all-same-sign pathology: with a 0/1 reward,
  below-mean (wrong) responses get a negative weight and are pushed *down*, not merely ignored.
- **Divide by the batch std (rescale).** The signal's magnitude acts as a multiplier on the
  effective learning rate; dividing by the batch standard deviation makes the transformed reward
  invariant to positive reward-unit rescalings — multiplying every reward by `c > 0` scales both
  `R−mean` and `std` by `c`, which cancels in `(R−mean)/std`. A negative `c` flips the sign, as it
  should because it reverses preference order. The `eps` floor only breaks exact invariance when
  the spread is near zero. The per-batch learning-rate tuning is moved out of the hand-set
  constant and made automatic, so the real step size no longer lurches with the batch's
  difficulty mix (the spread of a 0/1 reward depends on the fraction correct).

## Design choices and why

- **Batch-wide statistic, not per-prompt.** A per-prompt mean/std is estimated from only the few
  rollouts per prompt (noisy, the std especially), and the downstream GRPO estimator *already*
  subtracts a per-prompt group mean and divides by a per-prompt group std — applying it again
  upstream is redundant. A single batch-spanning affine whitening is the complementary,
  non-redundant move that sets the global signal scale while leaving per-prompt contrasts for
  GRPO to extract.
- **`+ eps` denominator floor.** The std is zero when all batch scalars are equal (an all-correct
  or all-wrong 0/1 batch); `(R−mean)/(std+eps)` then gives `0`, the honest "no relative signal"
  answer, instead of `0/0`. `eps = 1e-6` sits below any healthy reward spread, so it is negligible
  in normal batches and positive-scale invariance holds.
- **Degenerate single sample** (`bsz <= 1`): no spread / no baseline, pass through unchanged.
- **Population std** (`unbiased=False`, divide by `N`) is the batch_std edit's chosen empirical
  spread. The canonical verl `masked_whiten` defaults to Bessel-corrected `masked_var`; this
  reward-stage edit deliberately uses the population std and handles `bsz <= 1` before computing
  it.
- **Sum-then-scatter and `no_grad`.** `.sum(-1)` recovers the scalar from the "outcome reward at
  the last valid token" layout; the normalized scalar is scattered back to that token and masked,
  preserving the convention for downstream per-token machinery. The transform is a pure data
  operation, so it runs under `torch.no_grad()`.

## Final form (the transform)

```python
import torch
import numpy as np
from typing import Optional


@torch.no_grad()
def normalize_rewards(
    token_level_scores: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,        # (bs, response_length); 1 on valid response tokens
    index: np.ndarray = None,           # (bs,) prompt id — unused: this is a global, not per-prompt, transform
    epsilon: float = 1e-6,
    config: Optional[object] = None,
    **kwargs,
) -> torch.Tensor:
    """batch_std: subtract the batch mean, divide by the batch std + eps (reward whitening)."""
    bsz, seq_len = token_level_scores.shape
    scores = token_level_scores.sum(dim=-1)          # (bs,) one scalar per response

    if bsz <= 1:
        return token_level_scores * response_mask     # single sample: pass through unchanged

    mean = scores.mean()                              # weight-free plug-in baseline (center)
    std = scores.std(unbiased=False)                  # population spread (rescale)

    scores = (scores - mean) / (std + epsilon)       # whiten; +eps floors the zero-std batch

    out = torch.zeros_like(token_level_scores)
    last_idx = (response_mask.long().sum(dim=-1) - 1).clamp(min=0)   # (bs,)
    out[torch.arange(bsz, device=out.device), last_idx] = scores     # scatter to last valid token
    return out * response_mask
```

## Relation to prior approaches

- **Raw / outcome-only** = this transform with the whitening removed: inherits the variance, the
  scale-coupling, and the all-same-sign pathology.
- **Constant baseline subtraction** (Williams 1992; Sutton 1984 reinforcement comparison) = the
  *center* half only, with a hand-set reference; whitening adds the data-driven mean and the
  std-rescale.
- **Per-prompt group normalization** = the same standardize move but with the per-prompt
  statistic; `batch_std` instead uses the whole-batch statistic to stay complementary to (and not
  redundant with) the downstream GRPO per-group normalizer.
- **General whitening primitive.** The same operation, computed over masked valid elements, is the
  standard `masked_whiten`: `whitened = (values − masked_mean) * rsqrt(masked_var + eps)`
  (with optional re-adding of the mean), used widely for reward/advantage whitening in
  PPO/RLHF training stacks. `batch_std` applies this primitive to the per-response reward scalars
  at the reward stage, with population std and a `(std + eps)` floor instead of verl's default
  Bessel-corrected variance and `rsqrt(var + 1e-8)` denominator.
```python
def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean) ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        variance = variance * (mask_sum / (mask_sum - 1))
    return variance


def masked_whiten(values, mask, shift_mean=True):
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)                       # default: Bessel-corrected
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened
```
