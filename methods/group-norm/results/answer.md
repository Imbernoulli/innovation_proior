# Group Normalization (GN)

## Problem

A feature-normalization layer that normalizes each channel over the mini-batch makes its
accuracy depend on the batch size: the per-channel mean/variance are estimated over the
batch, so they become noisy and biased as the per-device batch shrinks (on ResNet-50 /
ImageNet the validation error climbs from ~23.6% at batch 32 to ~34.7% at batch 2). The
batch dependence also forces frozen running statistics at inference (a train/test
discrepancy), breaks under distribution shift on transfer, and is corrupted by non-i.i.d.
batches. Many vision tasks (detection, segmentation, video) are forced into batch sizes of
1–2 by memory, exactly where the batch-based layer is weakest.

## Key idea

Normalize over a **group of channels within a single sample**, never across the batch.

All of batch / layer / instance / group normalization share one computation — pick a set
S_i of feature positions that share one mean and variance, standardize, then apply a
learnable per-channel affine. With i = (i_N, i_C, i_H, i_W) indexing (N, C, H, W):

    x̂_i = (x_i − μ_i) / σ_i,
    μ_i = (1/m) Σ_{k∈S_i} x_k,   σ_i = sqrt( (1/m) Σ_{k∈S_i} (x_k − μ_i)² + ε ),
    y_i = γ x̂_i + β        (γ, β are per-channel, learnable)

They differ only in S_i:

| method | S_i | reduce over |
|---|---|---|
| Batch Norm | { k : k_C = i_C } | (N, H, W), per channel |
| Layer Norm | { k : k_N = i_N } | (C, H, W), per sample |
| Instance Norm | { k : k_N = i_N, k_C = i_C } | (H, W), per sample-channel |
| **Group Norm** | { k : k_N = i_N, ⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋ } | (C/G, H, W), per sample-group |

GN splits the C channels into G contiguous groups of C/G channels and, for each sample and
each group, pools over that group's channels and all spatial positions. Because S_i always
fixes k_N = i_N, the computation is **independent of batch size and identical at training
and inference** — no running statistics, no freezing. G is a hyper-parameter (default 32);
G = 1 recovers Layer Norm, G = C recovers Instance Norm, so GN interpolates between the two
batch-free extremes. The interior is better than either end: more flexible than one shared
statistic over all channels (Layer Norm is too coarse for unlike conv channels), yet it
exploits cross-channel correlation that a per-channel statistic discards (Instance Norm is
too isolated). Keeping (H, W) in S_i guarantees a large, stable sample for the estimate even
on a single image. The grouping mirrors classical group-wise normalization (HOG/SIFT
orientation histograms, VLAD/Fisher sub-vectors), where correlated coefficients are
normalized together.

## Algorithm

For input x of shape (N, C, H, W), groups G, channels-per-group C/G:
1. Reshape x to (N, G, C/G, H, W) — exposes the group as its own axis (contiguous blocks).
2. Compute mean and (biased) variance over the within-group-channel and spatial axes
   (axes 2,3,4), keeping the N and G axes — one (μ, σ) per (sample, group).
3. Standardize: x̂ = (x − μ)/√(var + ε).
4. Reshape back to (N, C, H, W) and apply the per-channel affine: y = γ x̂ + β
   (γ initialized to 1, β to 0). The same recipe extends to video by including a temporal
   axis T in the spatial reduction.

## Code

```python
import torch
import torch.nn as nn


def group_norm_forward(x, gamma, beta, G, eps=1e-5):
    # x: [N, C, H, W]; gamma, beta: per-channel scale/shift, shape [1, C, 1, 1].
    N, C, H, W = x.shape
    assert C % G == 0
    x = x.reshape(N, G, C // G, H, W)              # expose the group axis
    mean = x.mean(dim=(2, 3, 4), keepdim=True)     # per (sample, group); N and G kept
    var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.reshape(N, C, H, W)
    return x * gamma + beta                         # per-channel affine


class GroupNorm(nn.Module):
    """Structurally identical to torch.nn.GroupNorm: same statistics in train and eval,
    no running buffers, per-channel affine of size num_channels."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))   # gamma
            self.bias = nn.Parameter(torch.zeros(num_channels))    # beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]                       # (H, W) for images, (T, H, W) for video
        x = x.reshape(N, self.num_groups, C // self.num_groups, *spatial)
        dims = tuple(range(2, x.dim()))             # within-group channels + all spatial
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(N, C, *spatial)
        if self.affine:
            shape = [1, C] + [1] * len(spatial)
            x = x * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x
```

Equivalently, in TensorFlow, GN is the same two-line normalization as the batch layer but
with the moments taken over the group/spatial axes after the reshape:

```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)   # BN would use [0, 2, 3]
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
```

## Practical notes

- Default G = 32. The accuracy is a broad plateau over G; both fixing G and fixing the
  channels-per-group are reasonable, with the extremes G = 1 (Layer Norm) and G = C
  (Instance Norm) the worst points.
- γ is typically initialized to 1 and β to 0; setting γ = 0 at the last normalization of a
  residual block makes the block start as identity.
- Drop-in replacement for the batch-based layer in ResNet-style `Conv → Norm → ReLU`
  blocks; unchanged between train and eval; extends to (T, H, W) for 3D/video models.
