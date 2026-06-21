# Research question

On-device and single-user LLM inference (batch size 1, e.g. a chatbot on a laptop or an edge GPU) is dominated by the *generation* phase, which produces one token at a time and is severely *memory-bound*: at batch 1 the arithmetic intensity is ~1 FLOP/byte, far below the hardware roofline, and weight loading dominates the memory traffic. Reducing weight traffic means storing weights in low-bit integers (W4A16: 4-bit weights, 16-bit activations), cutting weight traffic ~4×.

How can we quantize LLM weights to 3–4 bits, post-hoc and without retraining, while keeping accuracy close to FP16 and producing a layout that turns the theoretical 4× memory saving into real on-device speedup?

# Background

**Weight-only quantization and where the time goes.** For on-device generation, FLOPs are fixed and small; the bottleneck is reading the weights from DRAM. A roofline analysis on an edge GPU (e.g. RTX 4090: ~165 TFLOPS, ~1 TB/s) shows any workload with arithmetic intensity below ~165 is memory-bound, and batch-1 generation sits at ~1. Quantizing weights to 4-bit raises arithmetic intensity ~4× and the peak ~4×. Weight-only 4-bit quantization (W4A16) is the lever here — distinct from W8A8 activation quantization, which targets compute-bound, large-batch serving.

**Uniform weight quantization.** A group/block of weights $\mathbf w$ is quantized by $Q(\mathbf w) = \Delta\cdot\mathrm{Round}(\mathbf w/\Delta)$ with $\Delta = \max(|\mathbf w|)/2^{N-1}$ for $N$ bits in the symmetric error analysis. The absolute rounding residual per element is roughly uniform on $[0,0.5]$, averaging ~0.25. A practical implementation can use the same round-to-nearest idea with an affine per-group scale and zero point, e.g. scale $(\max-\min)/(2^N-1)$ over groups of columns.

**Diagnostic findings about weight importance.** The weights of an LLM are *not equally important*:
- Keeping a small fraction (~0.1–1%) of weight channels in FP16 while quantizing the rest to INT3 nearly recovers FP16 accuracy.
- *Which* channels matter is the key observation. Selecting the FP16-kept channels by *weight* magnitude / $L_2$-norm gives essentially no benefit over random selection. But selecting them by **activation magnitude** — keeping the weight channels that multiply the largest-magnitude input features — gives a large accuracy improvement at the same 0.1–1% budget.
- Interpretation: input features with larger magnitude are more important to the output; the weights that process them are *salient*. Saliency is determined by the *activation* distribution, not the weight distribution.

**The activation-outlier context.** As in other LLM-quantization work, the input activations have a few persistent high-magnitude channels. Here those high-magnitude activation channels mark which *weight* channels are salient.

# Baselines

**Round-to-nearest (RTN).** Quantize every weight independently to its nearest grid value (with group-wise $\Delta$). Scalable and hardware-friendly, treating all weights as equally important.

**Mixed-precision (keep 1% salient weights in FP16).** Keep the activation-selected salient weight channels in FP16 and quantize the rest. Recovers most of the accuracy at ~0.1% extra bits, using a mixed FP16/INT data type with irregular memory access.

**GPTQ.** Second-order, output-reconstruction weight quantization: minimizes $\lVert\mathbf{WX}-\widehat{\mathbf W}\mathbf X\rVert^2$ using inverse-Hessian error compensation, a column sweep, and a Cholesky reformulation. Highly accurate at 3–4 bit. It performs a calibration-set *regression* (error feedback fit to calibration activations), and its low-bit kernels were specialized to particular models/settings.

**QAT / straight-through / learned-step methods.** Recover accuracy by optimizing through the non-differentiable rounding with approximate gradients, using backpropagation.

# Evaluation settings

- **Models.** OPT and LLaMA families, with instruction-tuned and multimodal variants as stress tests; the analysis ablations use OPT-6.7B.
- **Quantization.** Weight-only, INT3 and INT4, with group-wise quantization (group size 128 is the standard setting); activations stay FP16 (W4A16 / W3A16).
- **Calibration.** A small calibration set from the pre-training data, used to cache layer inputs, measure *per-channel average activation magnitude*, and produce reference FP16 outputs; it is not used for backpropagation or per-weight reconstruction regression.
- **Metrics.** Language-modeling perplexity (WikiText-2) and zero-shot accuracy, compared against the FP16 baseline, RTN, the 1%-FP16 mixed-precision oracle, and GPTQ. Plus, for the system side, on-device latency/throughput (tokens/s) and memory on edge GPUs and CPUs (roofline and weight-packing analysis).

# Code framework

The primitives that already exist: a per-channel average-activation-magnitude statistic over calibration, a group-wise uniform weight quantizer, FP16 linear layers, and a per-block forward that can produce a reference FP16 output. The routine that decides how to quantize a block's weights is the empty slot.

```python
import torch

@torch.no_grad()
def get_act_scale(x):
    # per-input-channel average magnitude of the activation
    return x.abs().view(-1, x.shape[-1]).mean(0)

def pseudo_quantize(w, n_bit=4, group_size=128, zero_point=True):
    # group-wise quantize-dequantize: round-to-nearest on a per-group grid
    ...

@torch.no_grad()
def quantize_block(block, fcs, inp):
    """Given a block's linear layers and a calibration input, produce the
    INT3/INT4 weight-only quantized block: no backprop, no per-weight
    regression."""
    org_out = block(inp)                          # FP16 reference output
    # TODO: the contribution goes here.
    pass

# driver: for each block, cache its calibration input, call quantize_block,
# then store the result group-wise-quantized to INT3/INT4 (W4A16).
```
