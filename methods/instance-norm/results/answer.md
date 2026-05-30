# Instance Normalization

## Problem

A feed-forward generator trained to stylize images in one pass (the amortized replacement for Gatys' slow per-image optimization) produces images visibly worse than the optimization it imitates, and fails in two telling ways: training on *more* content images makes the results *worse* (few images plus early stopping is best), and the heaviest artifacts cluster on the image border where convolutions are zero-padded. The behavior is that of a network forced to learn a function it represents poorly. The diagnosis: stylized contrast is set by the *style* image and is independent of the content image's contrast, so the generator must discard content contrast — a per-image normalization awkward to build out of convolutions and ReLUs, which the network was overfitting instead of representing cleanly.

## Key idea

Bake the contrast normalization into the architecture by changing the normalization layer. Batch normalization normalizes each channel using statistics pooled over the whole minibatch; instead, normalize each *individual image's* each channel over its spatial extent only — and apply the same computation at test time.

For x ∈ R^{T×C×W×H} (t image index, i channel, j,k spatial):

  Batch norm:    μ_i  = (1/(HWT)) Σ_t Σ_{l,m} x_tilm,   σ_i²  = (1/(HWT)) Σ_t Σ_{l,m} (x_tilm − μ_i)²
  Instance norm: μ_ti = (1/(HW))  Σ_{l,m}   x_tilm,      σ_ti² = (1/(HW))  Σ_{l,m}   (x_tilm − μ_ti)²
  y_tijk = (x_tijk − μ_ti) / √(σ_ti² + ε),  then per-channel affine γ_i, β_i.

The only change from batch norm is dropping the sum over the batch index t: statistics become per-instance.

## Why it works

- **Removes content contrast by construction.** Subtracting each image's own per-channel mean and dividing by its own spatial standard deviation *is* a contrast normalization of that single image — exactly the operation stylization requires the generator to perform, now wired into the architecture instead of learned. With no data-dependent normalization left to fit, the more-data-hurts and overfitting pathologies dissolve.

- **Batch norm cannot do this.** Pooling statistics across the batch means a single image's contrast is only partially removed (mixed with the rest of the batch), and at test time batch norm substitutes fixed training-set running statistics — so a single content image's contrast is never normalized by its own content.

- **Identical at train and test.** Instance norm has no batch dependence and no population statistics; its output is a deterministic function of the single input image, so train and test behave the same. This is essential for single-image inference, where the goal is to normalize *that* image's contrast — not to apply a frozen training-set average.

## Design choices

- **Per-instance statistics** (drop the batch sum): discards the content image's contrast, which stylization must ignore; removes instance-specific mean/contrast shift and simplifies learning.
- **Over spatial (H, W) only, per channel**: contrast is a per-channel spatial-scale property.
- **Per-channel affine γ, β retained**: lets the network rescale the normalized signal, as in batch norm.
- **Same operation at test time, no running statistics**: single-image inference must normalize the actual input; with no batch to pool over, train/test consistency is automatic.
- **ε** for numerical stability of the division.
- Replace batch norm with instance norm **everywhere** in the generator.

## Code

The original realization reuses the fast batch-norm kernel by folding the batch into the channel axis: reshape (N, C, H, W) → (1, N·C, H, W) and batch-normalize, so each (image, channel) slice is normalized over its own (H, W).

```python
import torch
import torch.nn as nn


class InstanceNormalization(nn.Module):
    """Per-(image, channel) spatial normalization, identical at train and test."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))    # gamma
            self.bias = nn.Parameter(torch.zeros(num_features))     # beta

    def forward(self, x):
        n, c, h, w = x.size()
        x_flat = x.contiguous().view(1, n * c, h, w)            # fold batch into channels
        weight = self.weight.repeat(n) if self.affine else None
        bias = self.bias.repeat(n) if self.affine else None
        out = nn.functional.batch_norm(                          # batch size 1, N*C channels
            x_flat, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True, eps=self.eps)  # training=True -> train==test
        return out.view(n, c, h, w)
```

Equivalently, modern frameworks provide this directly as `nn.InstanceNorm2d(C, eps=1e-5, affine=True)` (no running statistics, so it behaves identically in train and eval). In the stylization generator (the Texture Networks multi-scale architecture or Johnson's residual architecture), every batch-norm layer is replaced by this layer and kept active at test time; the generator is trained per fixed style with a pretrained-VGG perceptual loss (Gram-matrix style loss + deep-feature content loss).
