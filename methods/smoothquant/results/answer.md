# SmoothQuant

**Problem.** Quantize large LLMs to W8A8 (INT8 weights *and* INT8 activations) so the linear layers run on hardware INT8 GEMM units for real inference speedups — post-training, no retraining, and without a mixed FP16 outlier path. The obstacle: activation outliers concentrated in a few persistent input channels (~100× the typical value) destroy per-tensor activation quantization, and the per-channel activation scaling that would fix them sits on the matmul's contraction axis, which a hardware INT8 GEMM cannot apply (scales are only allowed as an epilogue along the outer dimensions $T$ and $C_o$).

**Key idea.** Don't scale activations at runtime — rebalance offline. For $\mathbf Y = \mathbf X\mathbf W$, insert a per-input-channel factor that shrinks the activations and grows the weights by the same amount, an exact equivalence:
$$\mathbf Y = \big(\mathbf X\,\mathrm{diag}(\mathbf s)^{-1}\big)\big(\mathrm{diag}(\mathbf s)\,\mathbf W\big) = \hat{\mathbf X}\hat{\mathbf W}.$$
Choose $\mathbf s$ to *split* the quantization difficulty between the (easy-to-quantize) weights and the (hard-to-quantize) activations:
$$s_j = \frac{\max(|\mathbf X_j|)^{\alpha}}{\max(|\mathbf W_j|)^{1-\alpha}},\qquad \alpha=0.5\ \text{(default)}.$$
At $\alpha=1$ this fully flattens activations but overloads weights; at $\alpha=0$ the reverse; $\alpha=0.5$ makes the smoothed activation and weight maxima of each channel equal (both the geometric mean $\sqrt{\max|\mathbf X_j|\cdot\max|\mathbf W_j|}$), sharing the difficulty. Models with heavier activation outliers (e.g. GLM-130B) use a larger $\alpha\approx0.75$; suitable $\alpha$ values are chosen with a quick validation-set grid search. The activation maxima $\max(|\mathbf X_j|)$ are estimated from calibration data (512 random Pile sentences in the experiments); the weight maxima are exact. Once $\alpha$ and these statistics are fixed, $\mathbf s$ is closed form: no gradients and no mixed FP16 outlier/decomposition path.

For LayerNorm- or linear-produced inputs, the factor $\mathrm{diag}(\mathbf s)^{-1}$ is folded into the preceding LayerNorm (or linear), so activation smoothing adds no runtime kernel; $\mathrm{diag}(\mathbf s)$ is folded into the following linear weights once, offline. For residual-add inputs, use explicit scaling on the residual branch. Afterwards, both operands quantize well with hardware-friendly scales: per-token or per-tensor activations, and per-tensor or output-channel epilogue-compatible weights, depending on the deployment setting. A plain INT8 GEMM runs the layer.

```python
import torch

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # nn.Linear stores weights as [out_features, in_features], so dim=0
    # gives the per-input-channel max across output channels.
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)                        # per-input-channel

    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    ln.weight.div_(s)                                      # fold diag(s)^-1 into LayerNorm
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s)
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))                      # fold diag(s) into next linears
```

Applied to LayerNorm-fed groups in a Transformer block (for example attention input feeding Q/K/V together and feed-forward input feeding the first MLP projection), then the compute-heavy linears are replaced with W8A8 linears. If the producer is a linear layer instead of a LayerNorm, the reciprocal factor is folded into that producer's output-channel weights and bias by the same equivalence. Attention batched matmuls also run INT8, while Softmax, LayerNorm, and residual adds stay FP16. The serving path (e.g. FasterTransformer) can then use dense INT8 GEMM for the hot operations instead of an FP16 outlier-decomposition path.
