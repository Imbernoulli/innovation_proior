**Problem.** Long-tail training: plain cross-entropy minimizes a frequency-weighted per-class average (`Σ_c (n[c]/N)·mean_loss[c]`), but the balanced test measures an *unweighted* per-class average. The head classes carry the gradient, the tail decision regions collapse, and balanced-test accuracy on rare classes craters. The first rung is the simplest *principled* correction.

**Key idea.** Re-weight each class's loss so the training objective stops being frequency-weighted. The algebra is direct: to make `w[c]·n[c]` constant across classes, take `w[c] ∝ 1/n[c]` — inverse frequency, `w[c] = N/(C·n[c])`, which exactly cancels the class-frequency term and turns the per-example mean into a per-class mean. Then dampen with a square root, `w ← √w`, collapsing the raw head/tail ratio from `ratio` to `√ratio` (100× → ~10×). Normalize to sum to `num_classes` so the average weight is 1 and the effective learning rate is unchanged.

**Why (this task's fill, not literal `1/n`).** Literal inverse frequency hands the rarest class ~100× the weight of the most frequent; a few tail examples — repeatedly augmented, weighted 100× — drive the gradient and over-fit the tail, and the per-batch loss spikes destabilize architectures without skip connections (VGG-16-BN). The √ dampening keeps the monotone "tail heavier than head" ordering that fixes the metric mismatch while pulling the dynamic range down to ~10×, which is what lets the VGG run hold together. The √ exponent is a heuristic, not a derivation — a single fixed global dampening with no model of how class size maps to needed emphasis — which is exactly the gap the next rung attacks.

**Hyperparameters.** Dampening exponent ½ (square root); normalization: sum-to-`num_classes`. No tunables exposed from `config`; pure function of `class_counts`. Counts are ≥ 1 by construction, so no division-by-zero guard is needed.

```python
# EDITABLE region of pytorch-vision/custom_weighting.py — step 1: inverse frequency (√-dampened)
def compute_class_weights(class_counts, num_classes, config):
    """Inverse frequency weighting.

    weight[c] = total_samples / (num_classes * count[c]).
    Directly proportional to inverse class frequency.
    Smoothed via square-root dampening to prevent training instability
    on architectures without skip connections (e.g. VGG).
    Normalized so weights sum to num_classes.
    """
    total = class_counts.sum().float()
    weights = total / (num_classes * class_counts.float())
    # Square-root dampening: reduces dynamic range while preserving ordering
    # For ratio=100 CIFAR-100: raw ratio ~100x -> dampened ~10x
    weights = torch.sqrt(weights)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights
```
