**Problem.** Inverse frequency treats class value as linear in sample count, so it over-credits the head — it acts as though the 5000th head example is as informative as the 1st. Under augmentation and natural redundancy, the marginal value of a sample *diminishes* as a class grows, so inverse frequency (and its √ damping) miscalibrates exactly on the many-class settings where the head dominates.

**Key idea (effective number of samples, Cui et al., CVPR 2019, arXiv:1901.05555).** Model each class as covering a finite region of `N` distinct prototypes; the expected covered volume after `n` draws is the effective number `E_n = (1 − β^n)/(1 − β)` with `β = (N−1)/N`, a geometric series from the recurrence `E_n = 1 + β·E_{n−1}`. Weight by the inverse effective number, `w[c] ∝ 1/E_{n[c]} = (1 − β)/(1 − β^{n[c]})`. The single parameter `β` interpolates between uniform (`β→0`, `E_n=1`) and inverse frequency (`β→1`, `E_n=n`); it dampens the saturated head automatically while leaving the tail's strong up-weighting intact.

**Why (this task's fill, vs the paper).** Use a fixed `β = 0.9999` — paper-recommended for long-tail CIFAR — applied uniformly across all three settings (it is *not* re-derived per class's `N`, which is unobservable). Departing from the paper's class-balanced loss, this task's fill then applies the *same* square-root dampening as the inverse-frequency rung on top of `1/E_n` (the paper normalizes `1/E_n` directly, no √): the principled curve supplies the head-dampening *shape*, and the √ compresses the residual dynamic range to keep the skip-free VGG-16-BN run stable. Normalize to sum to `num_classes` so the effective learning rate matches the other rungs.

**Hyperparameters.** `β = 0.9999` (fixed, task-local); extra √ dampening on top of `1/E_n`; normalization sum-to-`num_classes`. Counts ≥ 1, so `β^n < 1` strictly and `E_n > 0` — no division-by-zero guard needed.

```python
# EDITABLE region of pytorch-vision/custom_weighting.py — step 2: effective number of samples
def compute_class_weights(class_counts, num_classes, config):
    """Effective number of samples weighting (Cui et al., CVPR 2019).

    E_n = (1 - beta^n) / (1 - beta).
    weight[c] = (1 - beta) / (1 - beta^count[c]).
    Uses beta=0.9999, a task-local value explored in class-balanced losses.
    Smoothed via square-root dampening to prevent training instability
    on architectures without skip connections (e.g. VGG).
    """
    beta = 0.9999
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / effective_num
    # Square-root dampening: reduces dynamic range while preserving ordering
    weights = torch.sqrt(weights)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights
```
