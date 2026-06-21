## Research question

Data-parallel training spends most of its wall-clock time communicating gradients: every step, each worker computes a stochastic gradient and must all-reduce it before the optimizer can move. The single thing being designed is the **gradient compressor**: a lossy map applied to each gradient tensor before it goes on the wire (and its inverse after), so the communicated volume drops by one to three orders of magnitude. Everything else — the model, the data pipeline, the SGD+momentum optimizer, the cosine schedule, the 200-epoch budget — is fixed. The tension is the whole problem: any lossy compression perturbs the descent direction, and a perturbed direction can slow convergence, bias the solution, or stop it converging. A method has to compress aggressively (the target here is **100×**, keep 1%) and still land on a model as good as full-gradient training would.

## Prior art / Background / Baselines

Several families of gradient compressors exist, and each leaves a concrete gap.

- **Full dense gradient (no compression).** All-reduce the entire float32 gradient every step. Convergence is whatever SGD gives, but the communication is `O(d)` per step — the bottleneck the whole task exists to remove. Gap: no compression at all.
- **Sign compression.** Send one bit per coordinate, the sign. Maximally cheap. Gap: the sign operator is biased and magnitude-blind; small consistent coordinates are repeatedly mis-signed, and the expected update can point the wrong way.
- **Magnitude sparsification (top-k / gradient dropping).** Keep only the `k` largest-magnitude coordinates, send their values and indices, and zero the rest. Empirically strong at 99–99.9% drop. Gap: top-k is biased; persistently small coordinates are never transmitted, so naive top-k stalls.
- **Error feedback.** For any biased, contractive compressor, accumulate the compression residual in local memory and add it back before the next compression. The virtual iterate then runs exact SGD, recovering the convergence rate. Gap: it fixes convergence for contractive compressors but still stores a full-dimension residual and leaves the base compressor's message structure unchanged.

## Fixed substrate / Code framework

A single-node benchmark simulates distributed training: it computes gradients normally, then for each parameter applies `compress() → decompress()` (standing in for compress → communicate → decompress) and steps the optimizer on the decompressed gradient. This faithfully measures the effect of compression on convergence without multi-node infrastructure. Frozen and untouchable: the model definitions (ResNet-20/56, VGG-11-BN), CIFAR data loading with standard augmentation (random `32×32` crop pad 4, horizontal flip, per-channel normalize), the training loop, the cosine LR schedule with warmup, and the optimizer — **SGD, momentum 0.9, weight decay `5×10⁻⁴`, base LR `0.1`, 5 warmup epochs, 200 epochs, batch 128**. The loop calls the compressor once per parameter per step, between `loss.backward()` and `optimizer.step()`.

## Editable interface

Exactly one region is editable: the `Compressor` class in `custom_compressor.py`. Any compressor fills this same contract:

- `__init__(self, compress_ratio=0.01)` — fix the target ratio (`0.01` = keep 1% = 100×).
- `compress(self, tensor, name) -> (compressed_tensors: list[Tensor], ctx)` — map a gradient tensor to a small payload that would be communicated; `ctx` stays local. `name` identifies the parameter for per-parameter state.
- `decompress(self, compressed_tensors, ctx) -> Tensor` — rebuild a tensor of the *same shape* as the original input.

The compressor may hold internal state across calls (residuals, warm-start matrices) keyed by `name`. The starting point is the scaffold default: **identity (no compression)** — `compress` clones the tensor and stores its shape, `decompress` views it back. Each method replaces exactly these three method bodies.

```python
class Compressor:
    """Gradient compressor base implementation.

    Default: identity (no compression). Replace with your method."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        # default: ship the full tensor (no compression)
        return [tensor.clone()], tensor.shape

    def decompress(self, compressed_tensors, ctx):
        return compressed_tensors[0].view(ctx)
```

## Evaluation settings

Three settings, all at **100× compression** (`compress_ratio = 0.01`), each over three seeds {42, 123, 456}:

- **ResNet-20 / CIFAR-10** (~0.27M params) — small model, standard benchmark.
- **VGG-11-BN / CIFAR-100** (~9.8M params) — larger model, harder 100-class problem.
- **ResNet-56 / CIFAR-10** (~0.85M params) — deeper model, tests scalability.

Metric: **best test accuracy** (higher is better) per setting, reported per seed and as the mean. All settings share the fixed SGD+momentum + cosine-schedule + 200-epoch recipe above.
