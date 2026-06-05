# SmoothQuant

**Problem.** Quantize large LLMs to W8A8 (INT8 weights *and* INT8 activations) so the linear layers run on hardware INT8 GEMM units for real inference speedups — post-training, no retraining, with negligible accuracy loss. The obstacle: activation outliers concentrated in a few persistent input channels (~100× the typical value) destroy per-tensor activation quantization, and the per-channel activation scaling that would fix them sits on the matmul's contraction axis, which a hardware INT8 GEMM cannot apply (scales are only allowed as an epilogue along the outer dimensions $T$ and $C_o$).

**Key idea.** Don't scale activations at runtime — rebalance offline. For $\mathbf Y = \mathbf X\mathbf W$, insert a per-input-channel factor that shrinks the activations and grows the weights by the same amount, an exact equivalence:
$$\mathbf Y = \big(\mathbf X\,\mathrm{diag}(\mathbf s)^{-1}\big)\big(\mathrm{diag}(\mathbf s)\,\mathbf W\big) = \hat{\mathbf X}\hat{\mathbf W}.$$
Choose $\mathbf s$ to *split* the quantization difficulty between the (easy-to-quantize) weights and the (hard-to-quantize) activations:
$$\mathbf s_j = \frac{\max(|\mathbf X_j|)^{\alpha}}{\max(|\mathbf W_j|)^{1-\alpha}},\qquad \alpha=0.5\ \text{(default)}.$$
At $\alpha=1$ this fully flattens activations but overloads weights; at $\alpha=0$ the reverse; $\alpha=0.5$ makes the smoothed activation and weight maxima of each channel equal (both the geometric mean $\sqrt{\max|\mathbf X_j|\cdot\max|\mathbf W_j|}$), sharing the difficulty. Models with heavier activation outliers (e.g. GLM-130B) use a larger $\alpha\approx0.75$. The activation maxima $\max(|\mathbf X_j|)$ are estimated from a small calibration set (input-dependent); the weight maxima are exact. No search, no gradients, no mixed precision.

The factor $\mathrm{diag}(\mathbf s)^{-1}$ is folded into the preceding LayerNorm (or linear), so the activation smoothing is free at runtime; $\mathrm{diag}(\mathbf s)$ is folded into the following linear weights once, offline. Afterwards, both operands quantize well with the hardware-friendly per-tensor / per-token / per-output-channel scheme, and a plain INT8 GEMM runs the layer.

```python
import torch

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    # act_scales[j] = max|activation| over calibration in input channel j  (C_i,)
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)                        # per-input-channel weight max

    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    ln.weight.div_(s)                                      # fold diag(s)^-1 into prev layer
    ln.bias.div_(s)
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))                      # fold diag(s) into next linears
```

Applied before every quantized linear in a Transformer block (attention QKV/out projections, feed-forward layers); attention batched matmuls also run INT8, while Softmax, LayerNorm, and residual adds stay FP16. The serving path (e.g. FasterTransformer) then runs INT8 GEMM end-to-end, cutting both latency and memory while matching FP16 accuracy up to 175B/176B-scale models.
