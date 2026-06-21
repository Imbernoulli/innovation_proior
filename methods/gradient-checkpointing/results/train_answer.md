Training a deep neural network by backpropagation requires every forward activation to stay alive until the backward pass consumes it, so an n-layer network normally carries O(n) activation memory. In convolutional and recurrent models the parameters are often modest compared with the feature maps, which means the activations themselves dominate the memory budget and cap how deep, how wide, or how long a model can be trained on a single device. In-place operations and liveness-based memory sharing help inference drop to nearly O(1), but they only trim constants during training because all forward activations overlap the upcoming backward pass and therefore cannot be freed early. Model parallelism and CPU offloading move the memory elsewhere but do not reduce the total amount, and they add communication overhead. What is needed is a way to compute exact gradients while storing far fewer intermediate feature maps and paying only a modest extra computation cost.

The key observation is that activations do not strictly need to be stored at all. They only need to be available when the corresponding backward operator asks for them. If we keep a sparse set of checkpoint activations and discard everything in between, we can regenerate the discarded activations later by rerunning the forward computation from the nearest checkpoint. This trades a small amount of extra forward work for a large reduction in peak activation memory, and because the recomputation is exact, the gradients remain exact.

The method is gradient checkpointing. During the forward pass we divide the network into segments and save only the input activation to each segment, dropping all internal activations as soon as they are no longer needed for the immediate forward progress. When the backward pass reaches a segment, we reload its saved input, rerun the forward computation through that segment, recompute the internal activations into a small local buffer, backpropagate through them to obtain gradients for that segment, and then free those recomputed activations before moving to the previous segment. Only one segment's worth of recomputed activations lives in memory at any time, plus the small set of stored checkpoints. Because the recomputation uses the same inputs and preserves the original random number generator state, the regenerated activations are bit-identical to the originals and the gradients are exact even when the network contains dropout or other stochastic operations.

For an n-layer chain split into k equal segments, the peak feature-map memory has two competing terms. Storing the k segment boundary activations costs O(k), and the largest segment needs O(n/k) memory for its recomputed activations during backward, giving total memory O(n/k)+O(k). These terms pull in opposite directions: more segments reduce per-segment recomputation memory but increase checkpoint memory, while fewer segments do the reverse. Minimizing this balance gives k roughly sqrt(n), yielding O(sqrt(n)) peak activation memory while adding at most one extra full forward pass over the network. Since the backward pass is already roughly twice as expensive as a forward pass, the extra forward pass is a modest fraction of total training time, often on the order of twenty to thirty percent.

The same idea can be pushed further by applying it recursively. If each segment is itself checkpointed, the memory recursion g(n)=k+g(n/(k+1)) leads to g(n)=k log_{k+1} n. Choosing k=1 at every level stores a single intermediate and splits the chain in two, giving O(log n) activation memory, though the recomputation cost grows to O(n log n). This logarithmic regime is usually too expensive for routine training, but it shows the continuum of trade-offs and can be useful in extreme memory-constrained settings. In practice the one-level sqrt(n) plan is the most common sweet spot, and modern deep learning frameworks provide automatic helpers that choose a reasonable segmentation.

For heterogeneous computation graphs, not every layer deserves the same treatment. A budget-driven planner can mark each node as either kept or dropped, dropping cheap operations such as activation functions, batch normalization, or pooling while keeping expensive ones such as convolutions and large matrix multiplies. This mirror plan assigns a flag to each node: kept nodes act as persistent boundaries, while dropped nodes are mirrored into the backward region of the graph and recomputed there. The planner sweeps the graph in topological order, accumulating output sizes until the running total crosses a memory budget, at which point it marks the current node as a boundary and resets the accumulator. Static allocation can then measure the exact peak memory for each candidate plan, and a small grid search over budgets selects the best configuration. This handles real networks where layer sizes, receptive fields, and recomputation costs vary dramatically.

There is one important correctness subtlety. When a checkpointed region contains stochastic operations such as dropout, the recomputed forward must reproduce exactly the same random masks that were drawn during the original forward. Gradient checkpointing therefore preserves the random number generator state that was active when each checkpointed segment first ran, restores that state during recomputation, and then restores the original state afterward. With this safeguard the recomputed activations match the original ones bit for bit, and the computed gradients are identical to those obtained by storing every activation.

The primitive in PyTorch is checkpoint(function, *inputs), which avoids saving the intermediate tensors that ordinary autograd would retain and instead saves only the inputs needed to rerun function. For a sequential model the higher-level checkpoint_sequential helper splits a list of layers into chunks and checkpoints all chunks except the last one, saving each chunk's input and recomputing that chunk during backward. Setting preserve_rng_state=True keeps stochastic layers aligned with the original forward, and use_reentrant=False selects the current recommended implementation.

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
        # Balance O(n/k) recomputation memory against O(k) checkpoint memory.
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
