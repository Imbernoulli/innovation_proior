**Problem (from step 3).** Weight-only quantization is nearly lossless, but it only helps memory-bound
batch-1 generation. For compute-bound large-batch serving the *matmul* must run on INT8 tensor cores,
which need both operands in INT8 — i.e. W8A8. Weights are easy at 8-bit, but naive W8A8 destroys large
models (OPT-175B zero-shot collapses to 35.5%, chance level) because activations have persistent
~100×-outlier channels. The fix — per-channel activation scaling — lives on the matmul's *contraction*
axis, which an INT8 GEMM kernel cannot apply (scales are only allowed as an epilogue along T and C_out).

**Key idea.** **SmoothQuant** (Xiao 2022). Don't scale activations at runtime — rebalance offline. For
Y = XW, insert a per-input-channel factor that shrinks activations and grows weights by the same amount,
an exact equivalence: Y = (X·diag(s)⁻¹)(diag(s)·W) = X̂Ŵ. The smoothed activations lose their channel
spikes (now easy to quantize per-tensor) and the weights absorb the magnitude (only modestly harder,
and weights were nearly free). Choose s to *split* the difficulty: s_j = max(|X_j|)^α / max(|W_j|)^(1−α),
α=0.5 by default — making each channel's smoothed activation and weight maxima equal (both the geometric
mean). Models with heavier outliers (GLM-130B) use α≈0.75; α is picked by a quick validation grid
search. diag(s)⁻¹ folds into the preceding LayerNorm and diag(s) into the next linear's weights — both
offline — so at runtime there is no scaling kernel, just a dense INT8 GEMM.

**Why it works.** Per-channel correction is moved off the forbidden contraction axis by baking it into
the parameters. After smoothing, both operands quantize well on hardware-friendly per-tensor/per-token
scales; activation maxima come from a small calibration pass and weight maxima are exact, so s is
closed-form — no gradients, no mixed-precision outlier path.

**Change / code.** The per-channel smoothing factor folded into a LayerNorm and the linears it feeds.

```python
import torch

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)   # per-input-channel max from calibration

    # nn.Linear stores weights [out, in], so dim=0 gives the per-input-channel max over outputs.
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)

    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)   # split difficulty

    ln.weight.div_(s)                                        # fold diag(s)^-1 into LayerNorm (offline)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s)
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))                        # fold diag(s) into next linears (offline)
```

Applied to LayerNorm-fed groups (attention input → Q/K/V; FFN input → first MLP projection); the
compute-heavy linears are then replaced with W8A8 linears running INT8 GEMM, while Softmax, LayerNorm,
and residual adds stay FP16. If the producer is a linear instead of a LayerNorm, the reciprocal folds
into that producer's output-channel weights and bias.

**Target.** Restore OPT-175B W8A8 from naive 35.5% back to within a fraction of a point of FP16's 66.9%
zero-shot average, with the layers running as plain INT8 GEMMs — crossing from weight-only into true
weight-and-activation quantization, and exposing that 4-bit activations will need a sharper instrument.
