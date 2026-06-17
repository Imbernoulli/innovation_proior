**Problem.** QSGD collapsed on VGG/CIFAR-100 (seeds {69.8, 38.6, 37.9}) because its
clip-then-quantize fill *deletes* gradient energy with no memory, and at fixed `s = 256` it never
even reached 100×. The fix must compress aggressively *and* never permanently throw gradient mass
away.

**Key idea (Top-K sparsification + error feedback).** Gradients are positively skewed, so the `k`
largest-magnitude coordinates hold almost all the energy `‖g‖²`. Keep them (send `k` values +
`k` indices), zero the rest — at `k = ratio·d`, `ratio = 0.01`, that is a genuine 100×. Top-k is
*biased*, so naive use starves persistently-small coordinates (the same deletion that collapsed
QSGD's seeds). **Error feedback** repairs it: keep a per-tensor residual `e`, compress
`p = g + e`, and stash `e ← p − top_k(p)`. Suppressed coordinates accumulate in `e` until they
cross the threshold and are sent in one shot — nothing forgotten, only delayed.

**Why it works.** With the virtual iterate `x̃ = x − e`, error feedback runs *exact* SGD on `x̃`;
top-k is contractive (`δ = k/d`), so the residual stays bounded and the compression quality
enters only the higher-order term of the rate — leading order matches uncompressed SGD. This is
the convergence guarantee QSGD's clip forfeited, recovered via memory rather than unbiasedness.

**This task's fill.** Per-name residual dict (`self.residuals`), keyed by parameter — local,
not communicated. `k = max(1, int(numel·ratio))` floors at 1 so no tensor is permanently silent
(QSGD had no such floor and no memory). Values read directly via `tensor_flat[indices]`; residual
computed inline as `tensor_flat − decompressed_flat`. Payload `[values, indices]`; context
`(numel, shape)`.

**Hyperparameters.** `compress_ratio = 0.01` (100×, `k = 1%` of each tensor); per-name residual
state; no other knobs.

```python
class Compressor:
    """TopK sparsification with error feedback (EF-TopK).

    Keeps the K largest-magnitude gradient elements per tensor.
    Error feedback accumulates the compression error (original - decompressed)
    and adds it to the next gradient before compression, ensuring convergence.
    """

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}

    def compress(self, tensor, name):
        # Error feedback: add accumulated residual
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        k = max(1, int(numel * self.compress_ratio))

        # Select top-k by magnitude
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = tensor_flat[indices]

        # Update residual: store what was NOT communicated
        decompressed_flat = torch.zeros_like(tensor_flat)
        decompressed_flat.scatter_(0, indices, values)
        self.residuals[name] = tensor_flat - decompressed_flat
        self.residuals[name] = self.residuals[name].view(shape)

        return [values, indices], (numel, shape)

    def decompress(self, compressed_tensors, ctx):
        values, indices = compressed_tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
```
