**Problem.** The prior rungs (√-inverse, effective number) are both *power-law dampenings of the raw count*, compressed by a √ to survive the skip-free VGG. Effective number stalled on the ratio-50 VGG-16-BN setting (44.82). The real train/test mismatch is a *prior* shift — the training prior `p̂(y=j) = n_j/Σn_i` vs the uniform balanced-test prior — which a softmax bakes into the logits, biasing them by `log(n_j)`. That is a *logarithmic* effect, on a scale no power-law weight matches.

**Key idea (motivated by Balanced Softmax, Ren et al., NeurIPS 2020, arXiv:2007.10740).** The principled correction is an *additive logit shift*: train cross-entropy on `η_j + log(n_j)`, which makes the training posterior `n_j·e^{η_j}/Σ_i n_i·e^{η_i}` match the prior-shifted training distribution, leaving `η` as the unbiased balanced-test score. The head's prior advantage is subtracted from its logit, so capacity goes to the likelihood `p(x|y=j)` instead of re-learning the prior.

**Why (this task's fill — a weighting, NOT the paper's logit shift).** The edit surface exposes only a per-class CE *weight*; it cannot touch logits, so the exact `+log(n_j)` shift is *not implementable here* and the harness omits the mechanism that makes Balanced Softmax unbiased. What transfers is its insight that the head bias is *logarithmic* in frequency (every prior rung was a power law). So express a log-scale weight: `gap[c] = log(n_max) − log(n_c)` (0 at head, `log(ratio)` at tail), `w[c] = 1 + 4·(gap[c]/max_gap)`, a moderate 5:1 head→tail emphasis. No √ damping is needed — the log already lives in a tight range (`log(100) ≈ 4.6`), so it is inherently stable on the skip-free VGG where the power-law rungs were most fragile. Normalize to sum to `num_classes`.

**Hyperparameters.** Log-gap weight scaled linearly from 1 (head) to 5 (rarest); 5:1 cap (vs inverse-freq's ~100×, √-inverse's ~10×); normalization sum-to-`num_classes`; degenerate-input guard returns ones when `max_gap == 0`. Counts ≥ 1, so `log` is finite and `gap ≥ 0`.

```python
# EDITABLE region of pytorch-vision/custom_weighting.py — step 3: balanced-softmax-inspired log-prior
def compute_class_weights(class_counts, num_classes, config):
    """Balanced Softmax-inspired log-prior weighting (Ren et al., NeurIPS 2020).

    Balanced Softmax adjusts logits by subtracting log(pi_c). In this task's
    class-weighting interface, we use weights derived from the log-frequency gap:
    weight[c] = 1 + alpha * (log(n_max) - log(n_c)), alpha chosen so that
    the max/min weight ratio is moderate (~5:1).
    This provides a log-scale reweighting that is gentler than inverse
    frequency but still informed by class prior.
    """
    log_counts = torch.log(class_counts.float())
    log_max = log_counts.max()
    # Log-gap weights: classes further from max count get higher weight
    gap = log_max - log_counts  # 0 for most frequent, log(ratio) for rarest
    # Scale so the rarest class gets weight ~5x the most frequent
    max_gap = gap.max()
    if max_gap > 0:
        weights = 1.0 + 4.0 * (gap / max_gap)
    else:
        weights = torch.ones(num_classes)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights
```
