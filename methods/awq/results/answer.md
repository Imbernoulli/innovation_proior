# AWQ (Activation-aware Weight Quantization)

**Problem.** On-device, batch-1 LLM generation is memory-bound (arithmetic intensity ~1, weight loading dominates), so the lever is W4A16 — 4-bit weights, 16-bit activations — for ~4× less weight traffic. But round-to-nearest at 3–4 bits loses substantial accuracy. We want a weight-only low-bit quantizer that recovers accuracy with no backpropagation, no per-weight reconstruction regression, minimal calibration dependence, and a hardware-friendly uniform layout.

**Key observations → idea.**
1. LLM weights are not equally important; keeping ~1% of weight channels in FP16 nearly recovers accuracy. The salient channels are identified by **activation magnitude** (the weights that multiply the largest-magnitude input features), *not* by weight magnitude — selecting by weight norm is no better than random.
2. A literal FP16/INT mixed type is hardware-hostile. Instead, protect a salient weight by **scaling**: for $Q(w)=\Delta\,\mathrm{Round}(w/\Delta)$, computing $Q(ws)(x/s)$ leaves the product unchanged but changes the error by a factor $\tfrac{\Delta'}{\Delta}\cdot\tfrac1s$. Since average round error (~0.25) is magnitude-independent and a modest $s$ barely moves the group max ($\Delta'\approx\Delta$), the salient weight's error shrinks by $\sim 1/s$ — at no extra bits.
3. But large $s$ makes the scaled weight dominate the group, raising $\Delta'$, which *amplifies* every non-salient weight's error by $\Delta'/\Delta$. So the scale must balance salient protection against non-salient damage.

**Method — search to scale.** Minimize the layer's real output error over a one-parameter family driven by the per-channel average activation magnitude $\mathbf s_X$:
$$\mathbf s=\mathbf s_X^{\alpha},\qquad \alpha^\*=\arg\min_{\alpha\in[0,1]}\big\lVert Q(\mathbf W\,\mathrm{diag}(\mathbf s))\,(\mathrm{diag}(\mathbf s)^{-1}\mathbf X)-\mathbf W\mathbf X\big\rVert .$$
$\alpha=0$ is plain RTN; $\alpha=1$ protects salient channels most aggressively. Grid-search $\alpha$ (≈20 points) by directly measuring output MSE against the FP16 reference — no gradients. Normalize $\mathbf s$ by the geometric mean of its extremes, and add weight clipping to trim group-range MSE. The scale $\mathrm{diag}(\mathbf s)^{-1}$ on activations folds into the previous layer; the weights stay a uniform group-wise 4-bit tensor.

```python
import torch

@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)          # per-input-channel avg magnitude

@torch.no_grad()
def search_scale(block, fcs, inp, w_bit=4, group_size=128, n_grid=20):
    org_out = block(inp)                                  # FP16 reference output
    x_max = get_act_scale(inp)                            # s_X
    org_w = [fc.weight.data.clone() for fc in fcs]

    best_loss, best_scales = float("inf"), None
    for i in range(n_grid):
        ratio = i / n_grid                                # alpha
        scales = x_max.pow(ratio).clamp(min=1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc, w0 in zip(fcs, org_w):
            fc.weight.data = pseudo_quantize(w0 * scales.view(1, -1), w_bit, group_size)
            fc.weight.data = fc.weight.data / scales.view(1, -1)
        loss = (org_out - block(inp)).float().pow(2).mean().item()
        for fc, w0 in zip(fcs, org_w):
            fc.weight.data = w0
        if loss < best_loss:
            best_loss, best_scales = loss, scales
    return best_scales
```

Because it relies only on average activation magnitude and a 1-D search, AWQ barely depends on the calibration distribution and generalizes across domains/modalities. The 4-bit weights pack into a SIMD/GPU-friendly layout (e.g. interleaved order for cheap unpacking) and are dequantized on the fly inside fused matmul kernels, turning the 4× memory saving into >3× measured on-device speedup over FP16.
