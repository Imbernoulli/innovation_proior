# Sublinear-memory training (gradient checkpointing)

## Problem

Backpropagation needs each layer's forward activations to compute its gradients, so the standard training loop stores all `n` layers' activations until the backward pass consumes them — `O(n)` activation memory, linear in depth (or in RNN unrolling length). In conv/recurrent nets these feature maps dominate memory and cap how deep/long a model can be trained. In-place operation and memory sharing (liveness analysis) only reduce this by a constant factor for training, because the activations' lifetimes all overlap the backward pass. Goal: reduce training activation memory below `O(n)` with exact gradients and little extra compute.

## Key idea

Trade computation for memory: **drop** most forward activations and **recompute** them during backprop from a small set of stored checkpoints, instead of storing them all. Cutting an `n`-layer net into `k` segments and keeping only segment-boundary activations gives memory `O(n/k) + O(k)`; choosing `k = √n` minimizes this to `O(√n)`, at the cost of one extra forward pass per minibatch.

## Final method

Split the network into segments. Store only each segment's boundary (input) activation; drop everything inside a segment after the forward pass uses it. In the backward pass, for each segment in reverse: reload its stored input, re-run the forward through that segment to regenerate its internal activations, backprop through the segment, free the regenerated activations.

**Memory–compute trade.** With `k` equal segments of `n/k` layers:
`cost = max_i cost-of-segment(i) + O(k) = O(n/k) + O(k)`.
Minimizing `n/k + k` gives `k = √n` and `cost = 2√n = O(√n)`. Each layer's forward runs at most twice (once originally, once on recompute), so the extra cost is exactly one forward pass; since backward ≈ 2× forward, the measured slowdown is ~30%.

**Recursion to `O(log n)`.** Apply the scheme recursively inside each segment: storing `k` intermediates and recursing on sub-paths of length `n/(k+1)` gives `g(n) = k + g(n/(k+1))`, solving to `g(n) = k·log_{k+1}(n)`. With `k=1`, `g(n) = log_2 n` memory, at `O(n log n)` forward cost.

**General graphs.** A per-node mirror count `m(v)` marks each node as kept (`m=0`, a boundary) or dropped/recomputed (`m≥1`); a dropped node is duplicated ("mirrored") into the backward region. A greedy budget planner sweeps in topological order accumulating output sizes; when the running total exceeds budget `B` at a candidate split, it marks a boundary and resets — searching `B` (around `√(x·y)`) balances the two memory terms for non-uniform layer costs.

**Cheap special case.** Drop low-cost ops, keep expensive ones — e.g. in `Conv-BatchNorm-Activation`, keep the conv output, recompute BN/activation/pooling.

**Correctness.** Recompute must reproduce exact activations, so the RNG state of the original forward is restored before recomputation (so dropout/noise masks match); gradients are then identical to full-storage backprop — no approximation. Composes with in-place and sharing optimizations.

## Code

```python
import torch
import torch.utils.checkpoint as cp

def run_segment(layers, start, end):
    def forward(x):
        for i in range(start, end + 1):
            x = layers[i](x)
        return x
    return forward

def checkpoint_sequential(layers, segments, x):
    n = len(layers)
    seg = n // segments
    end = -1
    for start in range(0, seg * (segments - 1), seg):
        end = start + seg - 1
        # forward runs without storing this segment's activations;
        # backward recomputes them from the saved segment input.
        x = cp.checkpoint(run_segment(layers, start, end), x, use_reentrant=False)
    return run_segment(layers, end + 1, n - 1)(x)   # last segment: no checkpoint

class CheckpointedDeepNet(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.segments = max(1, int(len(layers) ** 0.5))   # k = sqrt(n) -> O(sqrt(n)) memory

    def forward(self, x):
        return checkpoint_sequential(list(self.layers), self.segments, x)

# A single checkpointed block (the primitive checkpoint_sequential is built from):
#   y = cp.checkpoint(block_fn, x, use_reentrant=False)
# forward: run block_fn(x) without saving internals, save only x.
# backward: re-run block_fn(x) to rebuild the local graph, then backprop.
```
