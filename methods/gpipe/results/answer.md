GPipe trains neural networks too large to fit on a single accelerator by partitioning them across devices, in a way that is architecture-agnostic, keeps the accelerators busy, and does not change the optimization the user would run on one device. The first wall is activation memory: caching every layer's forward activations for backprop costs O(N × L) for a mini-batch of N over L layers. Naive sequential model parallelism (cell k on device k) fixes memory but, because of the chain dependency, leaves all but one device idle at any instant.

Split the layer sequence into K consecutive **cells**, one per accelerator (communicating only the activation tensor at cell boundaries). Then split each mini-batch into M **micro-batches** and **pipeline** them through the cells so every accelerator is busy on a different micro-batch. Accumulate the gradients of all M micro-batches and apply **one synchronous update per mini-batch** so there is no weight staleness and the partition count does not change the optimizer step. Use **re-materialization** to keep only cell-boundary activations and recompute the rest in the backward pass, cutting peak activation memory.

The final method is:

- **Interface:** number of partitions K, number of micro-batches M, and an ordered list of L layers (each with forward f_i, params w_i, optional cost c_i). Cells are formed by grouping consecutive layers to **minimize per-cell cost variance** (balance the pipeline); cell k's forward is F_k = f_j ∘ … ∘ f_i, its backward B_k built by autodiff.
- **Schedule:** forward all M micro-batches through the K cells with a wavefront pipeline, then run the reverse wavefront backward, accumulate gradients, and apply a single update.
- **Re-materialization:** store only partition-boundary activations during forward; recompute each cell's internals from its cached boundary activation during backward. Peak activation memory **O(N + (L/K)(N/M))** versus O(N × L) naive.
- **Bubble:** fill/drain idle fraction **O((K−1)/(M + K−1))**, negligible once **M ≥ 4K** (helped further because backward recompute can be scheduled early).
- **Communication:** only boundary activation tensors cross devices — light enough to scale without high-speed interconnect.
- **BatchNorm:** statistics computed per micro-batch during training; moving average over the mini-batch kept for evaluation.

The implementation-shaped schedule is:

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable = None):
        self.f, self.w, self.cost = forward, params, cost_fn

class LayerSequenceTrainer:
    def __init__(self, layers: List[Layer], devices: List[object], M: int):
        self.layers = layers
        self.devices = devices
        self.K = len(devices)
        self.M = M
        self.cells = self.build_layer_groups()   # minimize per-cell cost variance
        self.F = [self._compose([l.f for l in cell]) for cell in self.cells]
        self.B = [autodiff_backward(F_k) for F_k in self.F]

    def build_layer_groups(self): ...
    def _compose(self, fns): ...
    def _on_device(self, k, fn, *args): ...
    def _rematerialize(self, k, boundary_activation): ...

    def run_forward(self, mini_batch):
        micro_batches = self._split(mini_batch, self.M)
        boundary = [[None] * self.M for _ in range(self.K)]
        x = [[None] * self.M for _ in range(self.K + 1)]
        x[0] = list(micro_batches)
        for t in range(self.M + self.K - 1):
            for k in range(self.K):
                m = t - k
                if 0 <= m < self.M:
                    boundary[k][m] = x[k][m]          # cache boundary activation feeding cell k
                    x[k + 1][m] = self._on_device(k, self.F[k], x[k][m])
        self._last_boundary = boundary
        return x[self.K]

    def run_backward(self, d_final):
        grads = self._zero_like_params()
        d = [[None] * self.M for _ in range(self.K + 1)]
        d[self.K] = list(d_final)
        for t in range(self.M + self.K - 1):
            for k in reversed(range(self.K)):
                m = t - (self.K - 1 - k)
                if 0 <= m < self.M:
                    acts = self._rematerialize(k, self._last_boundary[k][m])
                    d[k][m], g = self.B[k](d[k + 1][m], acts)
                    grads = self._accumulate(grads, g)      # sum over micro-batches
        return grads

    def train_step(self, mini_batch, optimizer):
        out = self.run_forward(mini_batch)
        loss = self._loss(out)
        grads = self.run_backward(self._loss_grad(loss))
        optimizer.apply(grads)            # one synchronous update per mini-batch
```

This is the GPipe library's design: define the model as a sequence of layers, set K and M, and the engine handles cell placement, boundary communication, the pipeline schedule, re-materialization, and balanced partitioning — usable for arbitrary architectures and composable with data parallelism.
