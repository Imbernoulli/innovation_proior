# DropBlock, distilled

DropBlock is a structured form of dropout for convolutional networks: instead of zeroing
independent units, it zeros contiguous `block_size × block_size` square regions of each
feature-map channel during training. The point is that convolutional activations are
spatially correlated, so independent-unit dropout removes almost nothing — a dropped
unit's information survives in its neighbors and reaches the next layer. Deleting a whole
contiguous block leaves no surviving correlated neighbors inside the hole, so the
information is genuinely removed and the network is forced to look elsewhere.

## Problem it solves

Dropout regularizes fully connected layers well but is ineffective on convolutional
layers, because spatial correlation lets information flow around individually dropped
units. DropBlock provides structured noise that actually removes information from a
convolutional feature map, is applicable at any convolutional layer, is a no-op at
inference, and is robust to its keep-probability hyperparameter.

## Key idea

- Sample block **centers** ("seeds") as an independent Bernoulli field with per-position
  probability `gamma`, restricted to the valid interior where a full block fits.
- **Expand** each seed into a `block_size × block_size` zero square (implemented as a
  stride-1 max-pool of the seed field, then invert).
- **Apply** the resulting block-structured keep-mask to the activations. The exact
  formulation samples a mask for each feature map/channel; the compact PyTorch
  implementation below samples one spatial mask per example and broadcasts it over
  channels, matching that implementation.
- **Rescale** by the *realized* survival fraction so training-time activation scale is
  normalized by the actual keep-mask and inference uses the plain network.

It is a one-parameter generalization of two existing methods: `block_size = 1` recovers
ordinary dropout, and `block_size =` the full feature map recovers SpatialDropout
(dropping a whole channel).

## Setting gamma

To make a target keep probability `keep_prob` meaningful (each unit dropped with
probability `1 - keep_prob`), `gamma` is derived by matching expected counts. The expected
number of dropped units is

```
gamma * (feat_size - block_size + 1)^2 * block_size^2  ≈  (1 - keep_prob) * feat_size^2,
```

since seeds live in the `(feat_size - block_size + 1)^2` valid-center region and each seed
deletes `block_size^2` units. Solving,

```
gamma = (1 - keep_prob) / block_size^2  *  feat_size^2 / (feat_size - block_size + 1)^2.
```

- The `1 / block_size^2` factor: one seed deletes `block_size^2` units, so seeds must be
  `block_size^2` times sparser to hit a target unit-drop rate.
- The `feat_size^2 / (feat_size - block_size + 1)^2` factor: seeds are confined to the
  smaller valid interior, so their density is scaled up by the full-to-valid area ratio.

This count ignores overlap between blocks, so for a fixed `gamma` it can overestimate how
many units are actually dropped; the per-sample rescale (below) uses the realized keep
fraction rather than the nominal probability. For large feature maps the area ratio `≈ 1`,
so a common simplification is `gamma ≈ (1 - keep_prob) / block_size^2`.

## Scheduled keep_prob

A fixed strong drop rate from the start hurts early learning. Linearly decrease `keep_prob`
from `1` (no dropping) to the target over training — equivalently, linearly increase
`drop_prob` / `gamma` from `0`. This protects early learning and makes the method robust to
the choice of target `keep_prob`.

## Algorithm

```
Input: activations A, block_size, gamma, mode
if mode == Inference: return A
Sample seed mask S: S_{i,j} ~ Bernoulli(gamma)      # only in the valid interior
Expand every drop-seed into dropped-region mask D
Invert to a keep-mask K = 1 - D
Apply:     A = A * K
Normalize: A = A * count(K) / count_ones(K)         # realized-survival rescale
```

Typical settings: `keep_prob` in `[0.75, 0.95]`, `block_size ≈ 7` for small deep maps;
apply per channel, on the convolution branch **and** the skip connection, in the deeper
residual groups, with a constant `block_size` across resolutions; identity at inference.

## Working code

DropBlock as an `nn.Module` faithful to `miguelvr/dropblock`: the seed field is spatial
per example and broadcast over channels, the block expansion is a stride-1 max-pool, the
rescale uses the realized survival fraction, and `gamma` uses the compact approximation
from that implementation.

```python
import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of a conv activation tensor (N, C, H, W).
    Identity at inference; rescales by the realized survival fraction."""

    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob          # 1 - keep_prob (target), scheduled externally
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, "expected (N, C, H, W)"
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self._compute_gamma(x)

        # sample seed mask (block centers), shared across channels in this implementation
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)

        # expand seeds into blocks, invert to a keep-mask
        block_mask = self._compute_block_mask(mask)

        # apply keep-mask, broadcasting over channels
        out = x * block_mask[:, None, :, :]

        # rescale by realized survival fraction, not the nominal drop rate
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:        # symmetric pad overshoots by one for even sizes
            block_mask = block_mask[:, :, :-1, :-1]
        return 1 - block_mask.squeeze(1)

    def _compute_gamma(self, x):
        # seeds per position; one seed deletes block_size^2 units.
        # Exact form also multiplies by feat^2/(feat - block_size + 1)^2 (valid interior);
        # that ratio -> 1 for large maps and is dropped in this simplified form.
        return self.drop_prob / (self.block_size ** 2)
```

Linear schedule that ramps `drop_prob` from `0` to the target over training:

```python
import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    """Linearly increase drop_prob from start_value to stop_value over nr_steps,
    stepped once per training step (start at 0 = no dropping early)."""

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super().__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1
```

The TPU implementation keeps the exact boundary ratio and restricts seeds to valid centers:

```python
def compute_gamma_exact(drop_prob, block_size, feat_size):
    return (drop_prob / block_size ** 2
            * feat_size ** 2 / (feat_size - block_size + 1) ** 2)
```
