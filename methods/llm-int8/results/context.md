# Research question

Large Transformer language models are increasingly used but barely fit in memory: in models at and beyond 6.7B parameters, the feed-forward and attention-projection matrix multiplications hold about 95% of the parameters and 65–85% of the compute. Storing those weights in 16-bit forces large multi-GPU deployments. The obvious lever is to store and multiply them in 8-bit integers, halving the memory.

The precise problem: can we take an *already-trained* 16/32-bit Transformer — up to 175B parameters — convert its linear-layer weights to Int8, run 8-bit matrix multiplications at inference, and get *zero* degradation in predictive performance, immediately, with no retraining or post-quantization tuning? Prior 8-bit work degraded accuracy, needed extra tuning, and had only ever been validated below 350M parameters. What happens between 350M and 175B was unknown, and a no-degradation method at that scale was an open challenge.

# Background

**Integer quantization of a matrix multiply.** An 8-bit integer holds values in $[-127, 127]$. To map a real tensor into that range we scale by a constant and round.

*Absmax (symmetric) quantization.* Scale by $127$ divided by the tensor's largest magnitude:
$$\mathbf X_{i8} = \Big\lfloor \frac{127}{\lVert\mathbf X_{f16}\rVert_\infty}\,\mathbf X_{f16}\Big\rceil = \lfloor s_x\,\mathbf X_{f16}\rceil .$$
A single scale per tensor.

*Zeropoint (asymmetric) quantization.* Shift and scale so the *full* $[-127,127]$ range is used even for an asymmetric distribution (e.g. post-ReLU values that are all $\ge 0$, where absmax wastes the entire negative half):
$$nd_x = \frac{2\cdot 127}{\max(\mathbf X)-\min(\mathbf X)},\qquad zp_x = \big\lfloor \mathbf X\cdot\min(\mathbf X)\big\rceil,\qquad \mathbf X_{i8} = \lfloor nd_x\,\mathbf X_{f16}\rceil .$$
Higher precision for asymmetric data, but using the zeropoint inside a matmul needs a special fused instruction; without it, the product $(\mathbf A_{i8}+zp_a)(\mathbf B_{i8}+zp_b)$ unrolls into four terms with Int16/32 cross-terms, which is slow on GPUs/TPUs. That practical cost is why absmax dominates in practice.

*The matmul with scales.* For hidden states $\mathbf X_{f16}\in\mathbb R^{s\times h}$ and weights $\mathbf W_{f16}\in\mathbb R^{h\times o}$, an 8-bit multiply with 16-bit I/O is $\mathbf X_{f16}\mathbf W_{f16}\approx \tfrac{1}{c_x c_w}\mathbf C_{i32} = S_{f16}\cdot\mathbf A_{i8}\mathbf B_{i8}$, where the integer accumulation $\mathbf C_{i32}$ is dequantized by the product of the (per-tensor) scaling constants.

**The failure of a single scale.** A single scale per tensor means one large-magnitude entry sets the range for everything; all the small values then collapse into a few quantization bins, and most bins go empty. The standard mitigation is more scales per tensor — block-wise or row-wise quantization (one scale per row of the activation matrix) — so an outlier's damage is confined to its block/row rather than the whole tensor.

**Diagnostic finding: emergent outlier features.** Empirically, as Transformers scale, a small set of *feature dimensions* (columns of the hidden state $\mathbf X\in\mathbb R^{s\times h}$) develop extreme magnitudes — up to ~20× larger than other dimensions. These outliers are *systematic*: in a 6.7B model about 150,000 outliers appear per 2048-token sequence, yet they are concentrated in only ~6 of the thousands of feature dimensions, and the same dimensions light up across essentially all layers. They emerge with a sharp phase shift: at ~6.7B parameters the fraction of layers carrying these outliers jumps from ~65% to 100% and the fraction of affected sequence positions from ~35% to ~75% — exactly where 8-bit quantization starts failing. Measured against perplexity rather than parameter count, the emergence is smooth and monotonic, suggesting it tracks model quality, not size alone. The outliers are almost always *one-sided* (purely positive or purely negative), which is why an asymmetric (zeropoint) quantizer handles them better than a symmetric one. And they are causally important: zeroing the ~6–7 outlier dimensions cuts the mean top-1 attention softmax probability from ~40% to ~20% and inflates validation perplexity by 600–1000%, whereas zeroing the same number of *random* dimensions changes top-1 probability by <0.3% and perplexity by ~0.1%.

# Baselines

**Per-tensor absmax / zeropoint.** One scale (or one scale + zeropoint) per matrix. Simple, hardware-friendly (absmax), but a single outlier ruins the precision of the whole tensor. Holds up to ~2.7B; fails as scale grows.

**Row-wise quantization.** A separate absmax scale per row of the activation matrix (and per the weight). Confines an outlier's effect to its row. Better, but outliers live in *columns* (feature dimensions), so per-row scaling still mixes outlier and normal values within each row.

**ZeroQuant / nuQmm (parallel large-model work).** Group-wise quantization — even finer granularity than per-row — with custom CUDA kernels. Targets throughput and footprint; validated up to 2.7B (nuQmm) and 20B (ZeroQuant). Complementary to a no-degradation goal but does not address the column-wise outlier structure head-on.

**Prior outlier studies.** Earlier work tied large-magnitude features in BERT-family models to LayerNorm and to token-frequency effects, but had not connected outlier emergence to autoregressive model scale, nor shown that handling them is the key to lossless large-scale quantization.

# Evaluation settings

- **Models.** Dense autoregressive Transformers from 125M to 13B for the analysis and perplexity study (trained in fairseq on Books, English Wikipedia, CC-News, OpenWebText, CC-Stories, English CC100), plus OPT up to 175B and BLOOM-176B; analysis cross-checked on GPT-2 (OpenAI), fairseq Meta models, and GPT-J (Tensorflow-Mesh) to rule out framework artifacts.
- **Metric.** Language-modeling perplexity on the C4 validation set (a robust, quantization-sensitive measure) for comparing quantization schemes; zero-shot accuracy on WinoGrande, HellaSwag, PIQA, and LAMBADA via the standard LM-evaluation-harness for the scaling-trend study. The object of interest is the *trend* across scales, not any single point.
- **Scope.** Quantize only the feed-forward and attention-projection linear layers; the attention function itself (no parameters) stays in 16-bit. Hardware: Int8 tensor cores on NVIDIA GPUs (the only widely available 8-bit type; FP8 unsupported at the time).

# Code framework

The primitives that already exist: a per-tensor absmax quantizer, an Int8 GEMM that accumulates in Int32, and FP16 linear layers in a Transformer. The replacement for the linear layer's forward pass is the empty slot.

```python
import torch

def absmax_quantize(x, dim=None):
    # scale by 127 / max|x| (per-tensor if dim is None), round to Int8
    scale = 127.0 / x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    q = torch.clamp(torch.round(x * scale), -127, 127).to(torch.int8)
    return q, scale

def int8_matmul(a_i8, b_i8):
    # Int8 x Int8 -> Int32 accumulation
    return torch.matmul(a_i8.to(torch.int32), b_i8.to(torch.int32))

class Int8Linear(torch.nn.Module):
    """Drop-in for an FP16 linear layer: store the weight in Int8, run the matmul
    in Int8, return an FP16 result that should match the FP16 layer's output."""
    def __init__(self, weight_f16):
        super().__init__()
        # TODO: how to store / scale the weight so the 8-bit forward is lossless
        pass

    def forward(self, x_f16):
        # TODO: the contribution — an 8-bit linear-layer forward that does not
        #       degrade accuracy at any scale, given what we know about the
        #       hidden states feeding into it
        pass
```
