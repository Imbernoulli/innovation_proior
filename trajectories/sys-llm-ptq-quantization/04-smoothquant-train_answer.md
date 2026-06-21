Weight-only quantization is in good shape now: AWQ at INT4-g128 on Llama-2-7B sits at 5.60 perplexity, a tenth off FP16, and even INT3-g128 is at 6.24. I have squeezed the weights about as hard as the batch-1 generation use case demands. But every method so far — RTN, GPTQ, AWQ — quantizes $W$ and leaves the activations $X$ in FP16, which is fine only in the memory-bound, batch-1 regime where the bottleneck is reading weights from DRAM. It is *not* fine for large-batch serving, where you are throughput-bound and want the multiply itself to run on the hardware's INT8 tensor cores — and those units need *both* operands in INT8. Weight-only quantization buys nothing there, because the arithmetic still happens in FP16.

So the target shifts to W8A8 — INT8 weights *and* INT8 activations — so the linear layers execute as dense INT8 GEMMs. The weights are easy: 8-bit RTN on weights is nearly lossless. The whole problem is the activations, and it is brutal. Naive RTN on the activations under W8A8 collapses accuracy on a large model — OPT-175B zero-shot average falls to about 35.5%, chance level, the model destroyed. The cause is the same outlier failure flagged back at the RTN floor: LLM activations have a small number of *persistent channels* — the same input-feature columns across nearly all tokens — whose magnitudes are $\sim 100\times$ larger than everything else. A per-tensor activation quantizer sets its single step $\Delta$ from the global maximum, which those outliers dominate, so $\Delta$ is enormous and every ordinary activation rounds to a handful of levels near zero. The natural fix is *per-channel* activation quantization — a separate $\Delta$ per input channel — and in simulation it recovers FP16 accuracy. But there is a hardware wall. For $Y = XW$ with $X$ of shape $[T \times C_{\text{in}}]$ and $W$ of shape $[C_{\text{in}} \times C_{\text{out}}]$, a per-channel activation scale is indexed by $C_{\text{in}}$ — the *contraction* axis being summed over. An INT8 GEMM kernel can only apply dequantization scales as an *epilogue*, along the output dimensions $T$ and $C_{\text{out}}$, after the sum; it physically cannot apply a different scale per element of the contraction axis, because by then the contraction is already summed away. So the granularity that fixes outliers is exactly the one the fast kernel forbids, and the granularity the kernel allows (per-token, indexed by $T$) does nothing about outliers organized by channel.

The method is **SmoothQuant**, and the move is to stop scaling activations *at runtime* and rebalance the difficulty *offline*. The outliers are a property of which channels are large, and those channels are visible from calibration data ahead of time. The same equivalence transform that protected weight channels works here, pointed the other way: for $Y = XW$, insert a per-input-channel factor $\mathrm{diag}(s)$ that *shrinks* the activations and *grows* the weights by the same amount,
$$Y = \big(X \cdot \mathrm{diag}(s)^{-1}\big)\,\big(\mathrm{diag}(s) \cdot W\big) = \hat X \hat W,$$
an exact identity before quantization. The smoothed activations $\hat X$ have had their outlier channels divided down — the $100\times$ spikes tamed — and the weights $\hat W$ absorb that magnitude. The activations become *easy* to quantize (no channel outliers blowing up $\Delta$) and the weights become *harder*, but weights were nearly free to begin with, so it is a good trade. Crucially, the per-channel factor is along $C_{\text{in}}$ for *both* tensors and folds away offline: $\mathrm{diag}(s)^{-1}$ into the preceding LayerNorm (or previous linear) and $\mathrm{diag}(s)$ into this linear's weights, once, before serving. At runtime there is no per-channel scaling kernel at all — just a plain INT8 GEMM. I have moved the per-channel correction off the forbidden contraction axis by baking it into the parameters.

The only real decision is how much difficulty to migrate — the choice of $s$. Push $s$ to flatten the activations completely and I dump all the difficulty onto the weights and overload them; push the other way and I am back to broken activations. I want $s$ to *split* the difficulty between the two operands, and the clean way to express that is, for each channel,
$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}, \qquad \alpha = 0.5 \text{ by default}.$$
At $\alpha = 0.5$ the smoothed activation maximum and the smoothed weight maximum of every channel become equal — both land at the geometric mean $\sqrt{\max|X_j|\cdot\max|W_j|}$ — so the two operands share the burden evenly. At $\alpha = 1$ the activations are fully flattened but the weights overloaded; at $\alpha = 0$ the reverse. Models with heavier activation outliers (GLM-130B is the extreme) want a larger $\alpha \approx 0.75$ to push more onto the still-easy weights, and $\alpha$ is picked with a quick validation-set grid search. The activation maxima come from a calibration pass of a few hundred sentences; the weight maxima are exact. Once $\alpha$ and those statistics are fixed, $s$ is closed-form — no gradients, no per-weight reconstruction, no mixed-precision outlier path.

The reference point is the *naive* W8A8, which on OPT-175B gives a chance-level 35.5% against an FP16 of 66.9%. The bet is that migrating the activation outliers into the weights offline lets both operands quantize to INT8 on hardware-friendly per-tensor / per-token scales: after smoothing the activations have no channel spikes that destroy their per-tensor grid, the weights only modestly harder still quantize cleanly, and the W8A8 model runs as a plain INT8 GEMM with essentially no accuracy loss — bringing the OPT-175B zero-shot average back to within a fraction of a point of FP16, call it $\sim 66.8\%$. This crosses from weight-only into true weight-and-activation quantization, with the linear layers finally in integer arithmetic. But this is INT8. The same per-tensor scales that survive 8-bit activations will *not* survive 4-bit: the outliers I migrated are smaller, not gone, so the moment I try W4A4 the problem returns, and a sharper instrument than offline rescaling will be needed.

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
