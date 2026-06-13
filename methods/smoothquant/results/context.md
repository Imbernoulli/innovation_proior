# Research question

To actually *speed up* large-language-model inference with quantization — not just shrink storage — both the weights *and* the activations of the big linear layers must be in 8-bit integers (W8A8), so the multiply itself runs on hardware INT8 GEMM units (NVIDIA tensor cores, Intel/Qualcomm integer kernels). Weight-only quantization halves storage but still does the arithmetic in FP16. The problem: activations of large LLMs are extremely hard to quantize because of outliers, and the quantized model must work *post-training* — no retraining, no fine-tuning, a small calibration pass — while preserving accuracy and mapping cleanly onto standard INT8 GEMM kernels (which is itself a hard constraint).

# Background

**Integer quantization and granularity.** Symmetric uniform INT8 quantization maps a float tensor by $\bar{\mathbf X} = \lceil \mathbf X/\Delta\rfloor$ with step $\Delta = \max(|\mathbf X|)/(2^{N-1}-1)$, $N=8$. (Asymmetric distributions add a zero-point.) The granularity of $\Delta$ matters: *per-tensor* (one step for the whole matrix), *per-token* (one step per row of the activation, i.e. per token), *per-channel* (for activations, one step per input feature column; for weights, one step per output channel), or *group-wise* (per group of channels). For a Transformer linear layer $\mathbf Y = \mathbf X\mathbf W$ with $\mathbf X\in\mathbb R^{T\times C_i}$, $\mathbf W\in\mathbb R^{C_i\times C_o}$, finer granularity gives more accuracy but not all granularities map onto efficient kernels.

**The hardware constraint that drives everything.** A hardware INT8 GEMM executes a tight sequence of tensor-core MMA instructions and cannot tolerate lower-throughput operations (scale conversions, CUDA-core FMAs) inserted *inside* the inner accumulation. Therefore the dequantization scales can only be applied along the matmul's *outer* dimensions — the token dimension $T$ of the activation and the output-channel dimension $C_o$ of the weight — as an epilogue after the integer accumulation finishes:
$$\mathbf Y = \mathrm{diag}(\boldsymbol\Delta_{\mathbf X})\cdot(\bar{\mathbf X}^{\text{INT8}}\bar{\mathbf W}^{\text{INT8}})\cdot\mathrm{diag}(\boldsymbol\Delta_{\mathbf W}).$$
A per-*token* activation scale ($T$ is an outer dim) is fine. A per-*output-channel* weight scale ($C_o$ is an outer dim) is fine. But a per-*input-channel* activation scale would sit on the contraction dimension $C_i$, *inside* the sum, and cannot be applied as an epilogue — so it breaks the fast kernel.

**Diagnostic findings about LLM activations.** Visualizing the input activations and weights of a linear layer with large quantization error reveals a consistent structure:
- *Weights are easy.* Their distribution is flat and uniform; INT8 (even INT4) weight-only quantization barely dents accuracy.
- *Activation outliers are the problem.* A few activation values are ~100× larger than the typical value. Under per-tensor quantization, the outlier sets the max and therefore $\Delta$: a non-outlier channel with max magnitude $m_i$, against a tensor max $m$, gets only $2^8\cdot m_i/m$ *effective* quantization levels — often just 2–3 for normal channels, so they are nearly destroyed.
- *Outliers live in fixed channels.* The large values concentrate in a small fraction of the *input channels*, and they persist there across all tokens. Variance *across channels* for a given token is large; variance *within a channel* across tokens is small (an outlier channel is consistently large).
- Consequently, *simulated per-channel activation quantization* (different $\Delta$ per input channel) recovers FP16 accuracy, while *per-token* quantization helps little — because the difficulty is organized by channel, not by token. But per-channel activation quantization is exactly the granularity the hardware kernel forbids (it's on the contraction axis).

So the field is stuck: the granularity that would fix activation quantization (per-channel) is unusable on the fast path, and the usable granularity (per-token) doesn't address the outliers.

# Baselines

**LLM.int8().** Keeps the outlier feature dimensions in FP16 and runs the rest in INT8 via mixed-precision decomposition. Preserves accuracy, but the mixed-precision/decomposition path and per-feature handling are hard to map onto a single dense INT8 GEMM, and in practice it does not yield a latency speedup (runtime is often comparable to or slower than FP16). Limitation: lossless but not faster.

**Per-tensor / per-token INT8 (W8A8).** The hardware-friendly schemes. Per-token activation quantization is what prior W8A8 works use. Limitation: cannot confine outliers (they're per-channel), so accuracy degrades on large models — only marginally better than per-tensor.

**ZeroQuant / nuQmm — custom quantized matmul paths.** ZeroQuant uses per-token dynamic activation quantization with group-wise weight quantization; nuQmm also relies on specialized quantized matrix-multiplication kernels. More accurate than naive W8A8 at smaller scales, but the difficulty of channel-fixed activation outliers is not removed, only worked around with extra granularity or custom kernels.

**Outlier-suppression / per-embedding-group work.** Earlier work confirmed per-channel-style activation scaling helps, but did not give a general post-training recipe that keeps the fast kernel and works up to 100B+ scale.

# Evaluation settings

- **Models.** OPT (up to 175B), BLOOM (up to 176B), GLM-130B (noted to have a larger ~30% outlier fraction), and MT-NLG 530B for scale stress testing.
- **Quantization scheme studied.** W8A8 — INT8 weights *and* INT8 activations — to exercise INT8 GEMM kernels; compute-heavy ops (linear layers, attention batched matmuls) in INT8, lightweight ops (Softmax, LayerNorm, residual adds) left in FP16. Both static (calibrated offline) and dynamic (runtime range) activation quantization, at per-tensor / per-token granularity.
- **Calibration.** Use 512 random Pile sentences to estimate activation-channel ranges for smoothing and, for static settings, activation quantization step sizes; choose suitable $\alpha$ values with a quick grid search on a Pile validation subset.
- **Metrics.** Zero-shot accuracy on standard LM benchmarks (LAMBADA, HellaSwag, PIQA, WinoGrande, etc.) and language-modeling perplexity (WikiText), measured against the FP16 baseline; plus inference latency and memory under a serving framework (FasterTransformer).

# Code framework

The primitives that already exist: per-channel max-magnitude statistics over a calibration pass, a uniform INT8 quantizer with selectable granularity, an INT8 GEMM, and a Transformer block whose linear layers are usually preceded by a LayerNorm or another linear layer. There is still an unresolved offline preprocessing step to run before the layers are replaced with W8A8.

```python
import torch

@torch.no_grad()
def collect_act_scales(model, calib_loader):
    # per-input-channel max|activation| at each quantized linear layer's input
    scales = {}
    # TODO: hook each target linear's input, accumulate max over abs along the token dim
    return scales

def quantize_per_tensor_int8(x):
    delta = x.abs().max() / 127.0
    return torch.clamp(torch.round(x / delta), -127, 127), delta

@torch.no_grad()
def prepare_block(prev_layer, fcs, act_scales):
    """Offline preprocessing for a producer (LayerNorm/linear) and the linears it feeds, run before W8A8 replacement."""
    # TODO
    pass

# driver: run calibration to fill act_scales, then for each block call prepare_block,
# then replace the linear layers with INT8 (W8A8) linears and run INT8 GEMM.
```
