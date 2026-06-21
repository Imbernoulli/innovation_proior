The central problem is that dropout, the workhorse noise regularizer for fully connected layers, loses its power when it is placed inside convolutional layers. A convolutional feature map is spatially correlated: adjacent positions are computed by overlapping receptive fields over a smooth input, so neighboring activations carry nearly the same information. If standard dropout zeros a single activation, the next convolution can simply read the same signal from the surviving neighbors, so the dropped unit is effectively bypassed and the thinned sub-network is not thinned at all. SpatialDropout responds by dropping entire channels at once, which does remove information, but its all-or-nothing granularity is too blunt on high-resolution maps and gives no dial between a scalar and a whole channel. Cutout masks a contiguous square, but only on the input image, so it cannot be inserted at arbitrary intermediate layers where the same correlation problem exists. What is needed is a layer that drops contiguous spatial regions at any convolutional stage, removes information at the scale of correlation, and still behaves like an identity at inference.

The method is DropBlock. During training it samples block centers, called seeds, on a coarse grid inside the feature map and expands every seed into a contiguous block_size by block_size square of zeros. The seeds are drawn from an independent Bernoulli field with probability gamma, restricted to positions where a full block fits. Each seed is dilated into a dropped region using a stride-1 max-pool of the seed field, and the result is inverted to produce a keep-mask. The activations are multiplied by that mask and then rescaled by the realized survival fraction, so the expected magnitude is preserved using the actual mask rather than the nominal probability. At inference the layer does nothing, just as ordinary dropout leaves the network unchanged at test time. A block_size of one recovers ordinary dropout, because a single seed drops a single unit; a block_size equal to the feature-map size recovers SpatialDropout, because one seed drops the whole channel. The block_size parameter therefore smoothly interpolates between the two prior methods and lets the practitioner choose how much spatial structure to remove.

The per-position seed probability gamma is chosen so that the expected number of dropped units matches a target keep probability. In expectation, the number of dropped units is gamma times the number of valid center positions times block_size squared, and this is set equal to one minus the keep probability times the total number of spatial positions. Solving gives gamma equal to one minus keep probability divided by block_size squared, scaled by the ratio of the full feature-map area to the valid-center area. That area ratio is one for large maps, so a common compact implementation uses gamma approximately equal to drop_prob divided by block_size squared. Because blocks can overlap near edges and boundaries, the exact dropout rate is not guaranteed by gamma alone, which is why the forward pass rescales by the actual fraction of surviving positions in the sampled mask. To avoid destroying not-yet-learned features early in training, the drop probability is not switched on at full strength from the first step. Instead it is linearly increased from zero to the target over the course of training, protecting early learning and making the final target less sensitive.

In practice DropBlock is applied in the deeper, lower-resolution residual groups, where a fixed block_size covers a larger semantic fraction of the map. It should be placed on the convolution branch and also on the skip connection, because an identity skip would otherwise leak the very information the block tried to remove. Typical settings use a keep probability between 0.75 and 0.95 and a block size around seven for small deep feature maps, with the same block size used across resolutions so it acts more aggressively on the smaller maps.

```python
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class DropBlock2D(nn.Module):
    """Randomly zero contiguous block_size x block_size squares of a conv activation tensor.
    Identity at inference; rescales by the realized survival fraction."""

    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob      # 1 - keep_prob (target), scheduled externally
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, "expected (N, C, H, W)"
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self._compute_gamma(x)

        # Sample seed mask (block centers), one spatial mask per example in this variant.
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)

        # Expand seeds into blocks and invert to obtain a keep-mask.
        block_mask = self._compute_block_mask(mask)

        # Apply keep-mask, broadcasting over channels.
        out = x * block_mask[:, None, :, :]

        # Rescale by the realized survival fraction.
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # Max-pool dilates each seed into a block_size x block_size dropped region.
        block_mask = F.max_pool2d(
            mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            # Symmetric padding overshoots by one for even block sizes.
            block_mask = block_mask[:, :, :-1, :-1]
        return 1 - block_mask.squeeze(1)

    def _compute_gamma(self, x):
        # One seed deletes block_size^2 units; the exact form also multiplies by
        # feat_size^2 / (feat_size - block_size + 1)^2 for the valid interior.
        # That ratio tends to 1 for large maps and is omitted in this compact form.
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    """Linearly ramp drop_prob from start_value to stop_value over nr_steps,
    advancing once per training step."""

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super().__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(
            start=start_value, stop=stop_value, num=int(nr_steps)
        )

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1
```
