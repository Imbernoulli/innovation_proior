QSGD's numbers came back and the split across settings is the exact failure the unbiased-but-clipped quantizer was set up to have. The two CIFAR-10 settings are fine — ResNet-56 is dead level at $\{94.06, 93.94, 94.23\}$, mean $94.08$; ResNet-20 is mostly fine at $\{92.66, 92.38, 86.78\}$ with one weak seed. But VGG-11-BN/CIFAR-100 is the catastrophe: $\{69.80, 38.57, 37.91\}$, mean $48.76$. Seed 42 trained to $\sim\!70$; the other two *collapsed* to half that. That bimodal, seed-dependent shape is not uniform quantization loss — uniform loss shaves a point off every seed equally. It is the fingerprint of unrepaired deletion: on the unlucky seeds the early large-norm gradients were clipped to unit norm, the clipped mass was thrown on the floor with no memory, and the model dropped into a basin it never climbed out of. So the diagnosis is precise — the floor's weakness is *deleted gradient energy with nothing carrying it forward* — and the fix must compress aggressively *and* never permanently throw gradient mass away.

I propose **EF-TopK — top-$k$ magnitude sparsification with error feedback**. The structural fact QSGD ignored is that gradients are strongly positively skewed: most coordinates are near zero, a few are large, so most of the energy $\lVert g\rVert^2$ lives in a small number of coordinates. Keep the $k$ largest-magnitude coordinates per tensor and zero the rest — send $k$ values and $k$ indices — and at $k = \text{ratio}\cdot d$ with $\text{ratio}=0.01$ that is a genuine 100× on the wire, not QSGD's few×. Magnitude is the importance signal, and the skew is what makes magnitude a *good* one. This is done strictly per tensor, layer-wise: a conv layer's gradient and a big embedding's gradient live on different scales, so "largest magnitude" only means something within a block, and a global top-$k$ would let the large-scale layers eat the whole budget.

But top-$k$ is *biased* — $\mathbb{E}[\text{top}_k(g)] \ne \nabla f$ — so used raw, none of the SGD guarantees apply, and it has its own sharper version of QSGD's deletion disease: a coordinate whose magnitude is persistently small never enters the top-$k$, so its direction is never transmitted and is starved at every step. The cure is to see what starvation actually is. The problem is not that I send only $k$ coordinates now — that would be fine if every coordinate eventually got sent — it is that the $d-k$ dropped coordinates are *discarded* and never come back, even though summed over a hundred steps a consistently-small coordinate's contribution can be large. So do not discard it: keep a per-tensor residual $e$, the running total of everything suppressed so far. Each step, before compressing, add it back — compress $p = g + e$ instead of $g$, send $\text{top}_k(p)$, and stash the leftover that did not make the cut, $e \leftarrow p - \text{top}_k(p)$. A persistently-small coordinate now *accumulates* in $e$ until it finally crosses into the top-$k$ and is sent in one shot. Nothing is forgotten; only delayed. This is **error feedback**, the memory QSGD's clip lacked, made the centerpiece.

This converges, and not just by feeling right. The clean handle is a virtual iterate $\tilde x = x - e$: the residual is an update owed but not yet applied, so subtract it from the real point. With the iterate stepping by the communicated $\Delta = \text{top}_k(p)$ and the residual $e' = p - \Delta$,
$$\tilde x' = x' - e' = (x - \Delta) - (p - \Delta) = x - p = x - (g+e) = (x-e) - g = \tilde x - g,$$
so the virtual iterate runs *exact* plain SGD — the $\Delta$ and the compression cancel perfectly. Error feedback is therefore not a heuristic approximation of SGD; it is honest SGD on a shadow sequence $\tilde x$, and the only gap to the real iterate is $e$. Top-$k$ keeping the $k$ largest is contractive ($\lVert C(x) - x\rVert^2 \le (1-\delta)\lVert x\rVert^2$ with $\delta = k/d$), so $e$ stays bounded, $x \approx \tilde x$, and because $f$ is smooth $\nabla f(x) \approx \nabla f(\tilde x)$; the compression quality $\delta$ enters only the higher-order term of the rate while the leading term matches uncompressed SGD. It is a delayed gradient method, and on a smooth loss delay is cheap.

The fill keeps the residual as a per-name dict keyed by parameter — each tensor its own memory and scale, the layer-wise discipline again. On `compress`, if a residual exists I add it, flatten, and set $k = \max(1, \lfloor\text{numel}\cdot\text{ratio}\rfloor)$; the $\max(1,\cdot)$ floor guarantees even a tiny tensor sends at least one coordinate so nothing is permanently silent — the contrast with QSGD, which had no such floor and no memory at all. I take the top-$k$ by magnitude (`torch.topk(|flat|, k)` for the indices, then read `values = tensor_flat[indices]`), then build the decompressed flat vector by scattering the kept values into a zero buffer and set the residual to the corrected flat tensor minus it — exactly $e \leftarrow p - C(p)$. The payload is `[values, indices]`, the context `(numel, shape)`; `decompress` scatters the values back into a zero vector and views to shape, the zeros being the dropped coordinates sitting in the residual waiting for next step.

Reading QSGD's shape, I expect the CIFAR-10 settings to hold or slightly improve, the weak 86.78 ResNet-20 seed to come up toward the others as the residual stops stranding its deleted mass, and — the decisive test — VGG-11-BN/CIFAR-100 to lose the seed-dependent collapse entirely, pulling the mean from $48.76$ into a healthy $\sim\!70$ with the seed spread shrinking from a $\sim\!32$-point chasm to a couple of points. The price I am consciously paying is that this is a biased compressor whose guarantee leans on the residual staying small, and at 100× the residual carries a lot of lag — so the question for the next rung is whether even error-feedback top-$k$ leaves accuracy slightly below where a denser per-coordinate signal could reach.

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
