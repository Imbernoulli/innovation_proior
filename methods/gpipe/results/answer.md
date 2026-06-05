# GPipe

## Problem

Train neural networks too large to fit on a single accelerator by partitioning them across devices, in a way that is architecture-agnostic, keeps the accelerators busy, and does not change the optimization the user would run on one device. The first wall is activation memory: caching every layer's forward activations for backprop costs O(N × L) for a mini-batch of N over L layers. Naive sequential model parallelism (cell k on device k) fixes memory but, because of the chain dependency, leaves all but one device idle at any instant.

## Key idea

Split the layer sequence into K consecutive **cells**, one per device (communicating only the activation tensor at cell boundaries). Then split each mini-batch into M **micro-batches** and **pipeline** them through the cells so every device is busy on a different micro-batch. Accumulate the gradients of all M micro-batches and apply **one synchronous update per mini-batch** — making the result identical regardless of K and M. Use **re-materialization** to keep only cell-boundary activations and recompute the rest in the backward pass, cutting peak activation memory.

## Final method

- **Interface:** number of partitions K, number of micro-batches M, and an ordered list of L layers (each with forward f_i, params w_i, optional cost c_i). Cells are formed by grouping consecutive layers to **minimize per-cell cost variance** (balance the pipeline); cell k's forward is F_k = f_j ∘ … ∘ f_i, its backward B_k built by autodiff.
- **Schedule:** forward all M micro-batches through the K cells (pipelined), then backward all M, accumulate gradients, apply a single update.
- **Re-materialization:** store only each cell's boundary input during forward; recompute the cell's internals from it during backward. Peak activation memory **O(N + (L/K)(N/M))** versus O(N × L) naive.
- **Bubble:** fill/drain idle fraction **O((K−1)/(M + K−1))**, negligible once **M ≥ 4K** (helped further because backward recompute can be scheduled early).
- **Communication:** only boundary activation tensors cross devices — light enough to scale without high-speed interconnect.
- **BatchNorm:** statistics computed per micro-batch during training; moving average over the mini-batch kept for evaluation.

## Code

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable):
        self.f, self.w, self.cost = forward, params, cost_fn

class PipelineEngine:
    def __init__(self, layers: List[Layer], K: int, M: int):
        self.K, self.M = K, M
        self.cells = self._balanced_partition(layers, K)   # minimize cost variance
        self.F = [self._compose([l.f for l in cell]) for cell in self.cells]
        self.B = [autodiff_backward(F_k) for F_k in self.F]

    def _balanced_partition(self, layers, K): ...
    def _compose(self, fns): ...

    def forward(self, micro_batches):
        boundary = [[None] * self.M for _ in range(self.K)]
        x = micro_batches
        for k in range(self.K):           # cell k on device k
            for m in range(self.M):       # pipelined micro-batches
                boundary[k][m] = x[m]     # store ONLY cell-input activation
                x[m] = self.F[k](x[m])
        return x, boundary

    def backward(self, d_out, boundary):
        grads = self._zero_like_params()
        d = d_out
        for k in reversed(range(self.K)):
            for m in range(self.M):
                acts = self.F[k].recompute(boundary[k][m])  # re-materialize
                d[m], g = self.B[k](d[m], acts)
                grads = self._accumulate(grads, g)          # sum over micro-batches
        return grads

    def train_step(self, mini_batch, optimizer):
        micro = self._split(mini_batch, self.M)
        out, boundary = self.forward(micro)
        grads = self.backward(self._loss(out), boundary)
        optimizer.apply(grads)            # one synchronous update per mini-batch
```

This is the GPipe library's design: define the model as a sequence of layers, set K and M, and the engine handles cell placement, boundary communication, the pipeline schedule, re-materialization, and balanced partitioning — usable for arbitrary architectures and composable with data parallelism.
