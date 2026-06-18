# Sublinear-memory training (gradient checkpointing)

## Problem

Backpropagation needs forward activations later in the reverse pass. A standard `n`-layer training run therefore stores `O(n)` feature maps, and ordinary liveness optimizations only reduce constants because those activation lifetimes overlap the backward pass. The target is exact gradients with sublinear activation memory.

## Method

Store a small set of checkpoint activations and drop the rest. During backward, reload the nearest stored checkpoint, rerun the forward computation for that segment, use the regenerated activations to compute gradients, and then free them.

For `k` equal segments of an `n`-layer chain, peak feature-map memory has two terms:

`cost(k) = max_i cost(segment_i) + O(k) = O(n/k) + O(k)`.

The derivative of `n/k + k` is `-n/k^2 + 1`; the continuous minimizer is `k = sqrt(n)`, giving `2sqrt(n) = O(sqrt(n))` memory. Each forward operation is recomputed at most once, so the one-level scheme adds one extra forward pass. Recursing the same idea gives `g(n)=k+g(n/(k+1))`, hence `g(n)=k log_{k+1} n`; `k=1` gives `O(log n)` memory with logarithmic recomputation overhead.

For general graphs, use a mirror plan: `m(v)=0` keeps node `v`'s output as a boundary, `m(v)=1` drops it and mirrors the node into the backward region, and larger counts encode recursive recomputation. A budgeted greedy planner chooses boundaries by accumulating output sizes until a candidate split crosses budget `B`, then static allocation measures the exact peak for each candidate plan. Cheap ops can be dropped preferentially while expensive ops such as convolutions or matrix multiplies are kept.

## Code

```python
from math import ceil, sqrt

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class CheckpointedDeepNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        n = len(self.layers)
        self.segments = 0 if n == 0 else max(1, min(n, ceil(sqrt(n))))

    def forward(self, x):
        if len(self.layers) == 0:
            return x

        return cp.checkpoint_sequential(
            list(self.layers),
            self.segments,
            x,
            use_reentrant=False,
            preserve_rng_state=True,
        )
```

`checkpoint_sequential` is the faithful modern primitive for the sequential case: it divides the layer list into chunks, checkpoints all chunks except the last, saves each checkpointed chunk's input, and recomputes that chunk during backward. `preserve_rng_state=True` keeps stochastic layers aligned with the original forward; `use_reentrant=False` selects the current recommended PyTorch implementation.
