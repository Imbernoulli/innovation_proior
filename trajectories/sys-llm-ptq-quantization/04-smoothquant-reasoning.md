Weight-only quantization is in good shape now. AWQ at INT4-g128 on Llama-2-7B sits at 5.60 perplexity,
a tenth off FP16, and even INT3-g128 is at 6.24. I have squeezed the weights about as hard as the
batch-1 generation use case demands. But I have been quietly ignoring half the matmul this whole time.
Every method so far — RTN, GPTQ, AWQ — quantizes W and leaves the activations X in FP16. That is fine
when the bottleneck is reading weights from DRAM, which is the memory-bound, batch-1 regime. It is *not*
fine for the other regime: large-batch serving, where you are throughput-bound and you want the multiply
itself to run on the hardware's integer tensor cores. Those INT8 GEMM units need *both* operands in
INT8. Weight-only quantization buys me nothing there, because the arithmetic still happens in FP16.

So the target shifts: W8A8 — INT8 weights *and* INT8 activations — so the linear layers execute as
dense INT8 GEMMs. The weights I already know how to quantize to 8 bits; 8-bit RTN on weights is nearly
lossless. The whole problem is the activations, and it is a brutal one. When I just apply RTN to the
activations and run W8A8, accuracy on a large model collapses — on OPT-175B the zero-shot average falls
to about 35.5%, which is chance-level; the model is destroyed. I need to understand *why* before I can
fix it, because this is the same outlier failure I flagged back at the RTN floor, and now I have to
actually defeat it.

Here is the structure of the difficulty. LLM activations have a small number of *persistent channels* —
the same input-feature columns across nearly all tokens — whose magnitudes are on the order of 100×
larger than everything else. A per-tensor activation quantizer sets its single step Δ from the global
maximum, which those outlier channels dominate; so Δ is enormous, and every ordinary activation (the
overwhelming majority of the signal) rounds to a handful of levels near zero. The information in the
bulk of the activations is annihilated. The natural fix that everyone reaches for is *per-channel*
activation quantization: a separate Δ per input channel, so the outlier channels get their own coarse
grid and the quiet channels keep a fine one. And it works — simulated per-channel activation
quantization recovers FP16 accuracy.

But there is a wall, and it is a hardware wall, not a statistics one. Look at where the activation
scaling axis lives in the matmul Y = X W, with X of shape [T tokens × C_in] and W of shape [C_in ×
C_out]. A per-channel activation scale is indexed by C_in — the *contraction* axis of the matmul, the
axis being summed over. An INT8 GEMM kernel can only apply dequantization scales as an *epilogue*, along
the output dimensions T and C_out, *after* the sum. It physically cannot apply a different scale per
element of the contraction axis, because by the time the scale would be applied the contraction has
already been summed away. So the granularity that fixes activation outliers (per-channel) is exactly the
granularity the fast kernel forbids, and the granularity the kernel allows (per-token, indexed by T)
does nothing about outliers, because the difficulty is organized by channel, not by token. The field is
stuck between a quantizer that works but won't run and a quantizer that runs but doesn't work.

Let me stop trying to scale activations *at runtime* and ask whether I can rebalance the difficulty
*offline*. The outliers are a property of which channels are large — and I can see those channels from
calibration data ahead of time. The same equivalence-transform trick that protected weight channels in
the previous rung works here, but pointed the other way. For Y = X W, insert a per-input-channel factor
diag(s) that *shrinks* the activations and *grows* the weights by the same amount:

  Y = (X · diag(s)⁻¹) (diag(s) · W) = X̂ Ŵ.

This is an exact identity before quantization. Now the activations X̂ have had their outlier channels
divided down — the 100× spikes are tamed — and the weights Ŵ have absorbed that magnitude. The activations
become *easy* to quantize (no more channel outliers blowing up Δ) and the weights become *harder*, but
weights were nearly free to quantize to begin with, so this is a good trade. Crucially, the per-channel
factor is along C_in for *both* tensors, and for both it can be folded away offline: diag(s)⁻¹ folds into
the preceding LayerNorm (or previous linear) and diag(s) folds into this linear's weights, once, before
serving. So at runtime there is no per-channel scaling kernel at all — just a plain INT8 GEMM. I have
moved the per-channel correction off the forbidden contraction axis by baking it into the parameters.

The only real decision is how much difficulty to migrate — the choice of s. If I push s all the way to
flatten the activations completely, I dump all the difficulty onto the weights and overload *them*; if I
push the other way I am back to broken activations. So I want s to *split* the difficulty between the two
operands. The clean way to express that: for each channel, take

  s_j = max(|X_j|)^α / max(|W_j|)^(1−α),  with α = 0.5 by default.

At α = 0.5 the smoothed activation maximum and the smoothed weight maximum of every channel become equal
— both land at the geometric mean √(max|X_j|·max|W_j|) — so the two operands share the burden evenly. At
α = 1 the activations are fully flattened but the weights are overloaded; at α = 0 the reverse. Models
with heavier activation outliers (GLM-130B is the extreme) want a larger α ≈ 0.75 to push more of the
difficulty onto the still-easy weights, and the right α is picked with a quick validation-set grid
search. The activation maxima come from a calibration pass (a few hundred sentences); the weight maxima
are exact. Once α and those statistics are fixed, s is closed-form — no gradients, no per-weight
reconstruction, no mixed-precision outlier path.

```python
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    weight_scales = torch.cat(                              # per-input-channel weight max
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)
    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    ln.weight.div_(s)                                       # fold diag(s)^-1 into LayerNorm (offline)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s)
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))                       # fold diag(s) into next linears (offline)
```

The bar and the bet. The reference point is the *naive* W8A8, which on OPT-175B gives a zero-shot
average of 35.5% — chance level, a broken model — against an FP16 of 66.9%. The whole question is
whether migrating the activation outliers into the weights offline lets both operands quantize to INT8
on hardware-friendly per-tensor / per-token scales. My claim is that after smoothing, the activations no
longer have channel spikes that destroy their per-tensor grid, the weights only modestly harder than
before still quantize cleanly, and the resulting W8A8 model runs as a plain INT8 GEMM with essentially no
accuracy loss. I am betting the OPT-175B zero-shot average comes back up to within a fraction of a point
of the 66.9% FP16 — call it ~66.8%. The risk is the split: if α is wrong for a given model I overload one
side or the other, which is exactly why α is grid-searched per model. If it holds, I have crossed from
weight-only into true weight-and-activation quantization, and the linear layers finally run in integer
arithmetic. But this is INT8. The same per-tensor scales that survive 8-bit activations will *not*
survive 4-bit activations — the outliers I migrated are smaller, not gone — so the moment I try W4A4, the
problem comes back, and a sharper instrument than offline rescaling will be needed.
