# Global Max Pooling (GMP), distilled

Global Max Pooling is a parameter-free global spatial-pooling head for CNN image
classification. It collapses each feature map `X_k` (its `N = H·W` activations, non-negative
after a ReLU) to a single scalar by taking the maximum over the spatial grid:

```
f_k = max_{x ∈ X_k} x,
```

producing a `C`-vector `f = [f_1, ..., f_C]` from a `[B, C, H, W]` tensor. It is a drop-in
alternative to global average pooling: input `[B, C, H, W]`, output `[B, C]`, channel dimension
unchanged, any spatial size (including `1×1`), and no learnable parameters.

## Problem it solves

A CNN classification head is bottlenecked by the single spatial collapse from the conv feature
tensor to a fixed-length vector. The pre-pooling default — flatten the maps and run
fully-connected layers — holds most of the parameters (overfits), fixes the input size, and bakes
spatial layout into weights instead of being translation-invariant. Global pooling removes those
costs, but the standard rule, the average, weights every spatial location equally. Under
*image-level* supervision (the label says only that a class is *present somewhere*) on cluttered
images, the discriminative response is *localized and sparse* — a few strong locations against a
large mostly-dead field — so averaging dilutes that signal against the background and spreads the
learning signal uniformly over the grid, including over background. GMP instead reads off the
single strongest location per channel.

## Key idea

Read the image-level label as Multiple Instance Learning: the image is a **bag**, each spatial
location is an **instance**, the bag is positive iff *some* instance is positive, and instance
labels are unknown. The quantity that decides a bag under that model is the **largest** instance
score, so the right per-channel summary is the **max** over spatial locations, not the average.

- **Forward — tracks peak evidence, not area.** `f_k = max_x x` returns the most-confident
  location's response and ignores how much dead area surrounds it. Two images where the object
  occupies very different fractions of the frame but is equally clearly present get the same high
  score, because both have at least one strongly-firing location; vast background contributes
  nothing as long as it stays below the peak.
- **Backward — argmax router + hard-negative mining.** For `f_k = max_i x_i`, if the winning
  location is unique, `j = argmax_i x_i`, then `∂f_k/∂x_i = 1` when `i = j` and `0` otherwise
  (ties require a subgradient or implementation tie-break). The entire upstream gradient is routed
  to the single highest-scoring location. On a **positive** image this *raises* the score at the
  location the channel already finds most object-like (and, via convolution weight sharing, at all
  similar locations across the dataset); on a **negative** image it *lowers* the highest-scoring
  location — suppressing the most-tempting false alarm, i.e. hard-negative mining. A network given
  only presence/absence labels therefore gets a concrete location hypothesis to update without
  location labels. Average pooling cannot do this: it spreads gradient uniformly, so it has no
  comparable selector.
- **Hard counterpart to soft aggregation.** Earlier softmax-style spatial aggregators (sum of
  exponential units, smooth combination of all locations) recover location through a blended
  response but mix every location into the output and share the error across the whole field. GMP
  commits fully to the single best location, gradient there and nowhere else, which is what yields
  the clean argmax router and the clean hard-negative mining.

## Properties and cost

- **Parameter-free** — a pure reduction; nothing to optimize or overfit at this layer.
- **Size-independent** — reducing over the `H, W` axes returns one number for any spatial size
  (`8×8` down to `1×1`); it is a property of pooling over the spatial axes, not of which reduction
  is chosen. Channel dimension `C` is untouched.
- **Cost.** Committing to one location per channel per image discards every other location that
  also fired (corroborating evidence) and routes gradient to only that one location, so the
  component is at the mercy of a single value: a noisy or spurious peak can hijack it, and the
  channel's other true responses contribute nothing to the score or the gradient. This is the
  opposite failure mode of average pooling (which hears everyone and drowns the signal in
  background); it is the right trade in the image-level, localized-evidence, cluttered regime,
  where diluting the signal is worse than betting on the peak.

## Final form

```
GMP(X)_k = max over the H × W grid of channel k,   no parameters.
```

Implemented by reusing the adaptive max-pool primitive at output size `1×1` (handles any `H, W`,
including `1×1`) and reshaping to `[B, C]`.

## Working code

Filling the global-pooling slot (`[B, C, H, W] -> [B, C]`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPool(nn.Module):
    """Global max pooling: f_k = max over the H x W grid of channel k.
    No parameters. Reads off the single strongest (most-confident) location per
    channel, matching an image-level 'present somewhere' (MIL bag) label."""

    def __init__(self):
        super().__init__()

    def forward(self, x):                       # x: [B, C, H, W], x >= 0 after ReLU
        # global max over the whole spatial grid; adaptive pool handles any H, W (incl. 1x1)
        return F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
```

Equivalent forward-value forms (same `[B, C]` result):

```python
def global_max_pool(x):
    # max reduction over the two spatial axes
    return torch.amax(x, dim=(-2, -1))          # [B, C, H, W] -> [B, C]

# or, explicitly via flatten + max:
def global_max_pool_flat(x):
    return x.flatten(2).max(dim=2).values        # [B, C, H*W] -> [B, C]
```
