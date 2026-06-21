The problem is training neural networks that have grown too large to fit on a single accelerator. The binding constraint is usually not the parameters themselves but the activation memory: during backpropagation, computing a layer's gradient requires both the incoming backward gradient and that layer's cached forward activation. A naive forward over L layers with a mini-batch of size N therefore stores O(N × L) worth of activations, and for deep models this is what overflows device memory first. Existing approaches either require hand-crafted, architecture-specific distributed operators or sacrifice synchronous optimization semantics. Sequential model parallelism—placing consecutive layer groups on consecutive devices—does solve the memory problem and keeps communication light, but the chain dependency means only one accelerator is ever active at a time while the others wait.

The key insight is that idleness, not partitioning, is the real enemy. We can keep the simple vertical split that makes the model fit, but instead of pushing one mini-batch at a time through the chain, we split the mini-batch into smaller micro-batches and pipeline them. After a brief fill phase, every accelerator is busy processing a different micro-batch at a different pipeline stage. To preserve the exact optimization trajectory of single-device training, gradients from all micro-batches are accumulated and only a single weight update is applied at the end of the mini-batch, eliminating weight staleness. Activation memory is further reduced by re-materialization: only the activations at cell boundaries are cached during the forward pass, and each cell's internal activations are recomputed from the cached boundary during the backward pass.

The method is called GPipe. It takes an ordered sequence of layers and two user-specified numbers: K, the number of partitions, and M, the number of micro-batches per mini-batch. The layers are grouped into K consecutive cells, with grouping chosen to balance per-cell compute cost so that no single stage becomes the pipeline bottleneck. Cell k is placed on device k, and its composite forward F_k is the composition of its layers' forward functions. Its backward B_k is built automatically by automatic differentiation. During training, the forward pass pushes micro-batches through the cells in a wavefront schedule; the backward pass runs the reverse wavefront. Because each micro-batch's forward and backward use the same fixed weights, the sum of accumulated micro-batch gradients is identical to the gradient that would have been computed on the unsplit mini-batch, so adding partitions changes throughput without changing optimization.

Re-materialization makes this scalable in memory. Storing only boundary activations reduces the peak activation memory to O(N + (L/K)(N/M)), a dramatic improvement over the naive O(N × L). The cost is one extra forward recomputation per cell during the backward pass. The remaining inefficiency is the pipeline bubble: the K−1 fill and drain steps at the start and end of each mini-batch. The bubble fraction is approximately (K−1)/(M + K−1), so choosing M large relative to K—roughly M ≥ 4K—makes the overhead negligible. Communication is also minimal, since only the activation tensors at the K−1 cell boundaries move between devices; parameters and internal layer activations stay local. This makes GPipe practical even without high-speed interconnects. For batch normalization, statistics are computed per micro-batch during training, with a moving average maintained over the full mini-batch for evaluation.

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable = None):
        self.f, self.w, self.cost = forward, params, cost_fn

def autodiff_backward(forward_fn):
    """Build a backward function for a forward computation."""
    ...

class LayerSequenceTrainer:
    """GPipe: pipeline parallelism with micro-batching and re-materialization.
    Gradients over all micro-batches are accumulated before one synchronous update."""
    def __init__(self, layers: List[Layer], devices: List[object], M: int):
        self.layers = layers
        self.devices = devices
        self.K = len(devices)
        self.M = M
        self.cells = self.build_layer_groups()
        self.F = [self._compose([l.f for l in cell]) for cell in self.cells]
        self.B = [autodiff_backward(F_k) for F_k in self.F]

    def build_layer_groups(self):
        # Group consecutive layers into K cells minimizing variance of per-cell cost.
        ...

    def _compose(self, fns):
        def composed(x):
            for f in fns:
                x = f(x)
            return x
        return composed

    def _on_device(self, k, fn, *args):
        # Run fn on device k; in a real system this wraps device placement.
        return fn(*args)

    def _split(self, batch, M):
        # Split mini-batch into M equal micro-batches.
        ...

    def _zero_like_params(self):
        ...

    def _accumulate(self, grads, g):
        # Add micro-batch gradient g into accumulated grads.
        ...

    def _rematerialize(self, k, boundary_activation):
        # Recompute cell k's internal activations from cached boundary activation.
        return self.F[k](boundary_activation)

    def run_forward(self, mini_batch):
        micro_batches = self._split(mini_batch, self.M)
        boundary = [[None] * self.M for _ in range(self.K)]
        x = [[None] * self.M for _ in range(self.K + 1)]
        x[0] = list(micro_batches)
        for t in range(self.M + self.K - 1):
            for k in range(self.K):
                m = t - k
                if 0 <= m < self.M:
                    boundary[k][m] = x[k][m]
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
                    grads = self._accumulate(grads, g)
        return grads

    def train_step(self, mini_batch, optimizer, compute_loss, compute_loss_grad):
        out = self.run_forward(mini_batch)
        loss = compute_loss(out)
        grads = self.run_backward(compute_loss_grad(loss))
        optimizer.apply(grads)
        return loss
```
