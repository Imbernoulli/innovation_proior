# Research question

Large Transformer language models are increasingly useful but barely fit in memory. At and beyond 6.7B parameters, the feed-forward and attention-projection matrix multiplications hold about 95% of the parameters and 65-85% of the compute. Storing those weights in 16-bit forces large multi-GPU deployments. The obvious lever is to store the linear-layer weights in 8-bit integers and run the large products on Int8 tensor cores.

The precise problem: can an already-trained 16/32-bit Transformer, up to 175B parameters, have its linear-layer weights converted to Int8 and be used immediately at inference with no retraining and no post-quantization tuning? Prior 8-bit Transformer work had been studied below 350M parameters. What happens from 350M to 175B is unknown.

# Background

**Integer quantization of a matrix multiply.** An 8-bit signed integer uses the range $[-127,127]$. A real tensor is mapped into that range by choosing a scale, dividing by it, and rounding.

*Absmax symmetric quantization.* With a dequantization scale $c_x=\lVert\mathbf X_{f16}\rVert_\infty/127$,
$$\mathbf X_{i8}=\left\lfloor \mathbf X_{f16}/c_x\right\rceil.$$
Equivalently, this is multiplication by $s_x=127/\lVert\mathbf X_{f16}\rVert_\infty$. One scale covers an entire tensor, making it fast and hardware-friendly.

*Zeropoint asymmetric quantization.* For an asymmetric distribution, use the full dynamic range with
$$nd_x=\frac{2\cdot127}{\max(\mathbf X)-\min(\mathbf X)},\qquad \mathbf X_{i8}=\left\lfloor nd_x\,\mathbf X\right\rceil,$$
and carry a separate integer shift $zp_x$ for the operation. The shift moves the minimum and maximum toward the two ends of $[-127,127]$, which suits one-sided distributions such as post-ReLU activations. The matmul then expands as $(A_{i8}+zp_a)(B_{i8}+zp_b)$, which includes Int16/32 cross-terms unless a special fused instruction exists. Symmetric absmax avoids those cross-terms.

*The matmul with scales.* For hidden states $\mathbf X_{f16}\in\mathbb R^{s\times h}$ and weights $\mathbf W_{f16}\in\mathbb R^{h\times o}$, quantize to $\mathbf A_{i8}$ and $\mathbf B_{i8}$, accumulate $\mathbf A_{i8}\mathbf B_{i8}$ in Int32, then multiply by the dequantization scales. With one scale per tensor, $\mathbf X_{f16}\mathbf W_{f16}\approx (c_xc_w)\mathbf C_{i32}$.

**More scales per product.** A matrix multiplication is a grid of inner products. For output $(i,k)$, row $i$ of $\mathbf X$ is dotted with column $k$ of $\mathbf W$. This allows one absmax scale for each row of $\mathbf X$ and one for each column of $\mathbf W$:
$$\mathbf C_{f16}\approx \mathbf C_{i32}\odot(\mathbf c_x\otimes\mathbf c_w),\qquad \mathbf c_x\in\mathbb R^s,\ \mathbf c_w\in\mathbb R^o.$$
This gives every inner product its own row/column scale pair while still permitting a single Int8 GEMM followed by an output rescale.

**Diagnostic finding: emergent outlier features.** As Transformers scale, a small set of feature dimensions, the columns of $\mathbf X\in\mathbb R^{s\times h}$, develop extreme magnitudes up to about 20x larger than other dimensions. These outliers are systematic: in a 6.7B model about 150,000 outlier values appear per 2048-token sequence, yet they are concentrated in about 6 feature dimensions, and the same dimensions light up across essentially all layers. For analysis, a recurring outlier feature has magnitude at least 6.0, appears in the same feature dimension in at least 25% of layers, and appears in at least 6% of sequence positions. They emerge with a sharp parameter-scale shift: around 6.7B parameters, the fraction of layers carrying these outliers rises from about 65% to 100%, and affected sequence positions rise from about 35% to about 75%. Measured against perplexity rather than parameter count, the emergence is smooth and monotonic, suggesting it tracks model quality rather than size alone. The outliers are usually one-sided, which is why asymmetric zeropoint quantization handles them better than symmetric absmax. They are also load-bearing: zeroing the outlier dimensions cuts the mean top-1 attention softmax probability from about 40% to about 20% and inflates validation perplexity by 600-1000%, whereas zeroing the same number of random dimensions changes top-1 probability by at most about 0.3% and perplexity by about 0.1%.

# Baselines

**Per-tensor absmax / zeropoint.** One scale, or one scale plus a zeropoint, per matrix. Simple and hardware-friendly for absmax.

**Row-wise quantization.** A separate absmax scale per row of the activation matrix, with a corresponding scale on the weight side. This confines scaling to individual token rows.

**Row/column inner-product scaling.** One scale for each activation row and one for each weight column. This is the natural granularity for a GEMM because each output is one row-column inner product. It preserves a single Int8 GEMM followed by an output rescale.

**ZeroQuant / nuQmm.** Parallel large-model work uses group-wise quantization, an even finer normalization granularity than row/column scaling, with custom CUDA kernels. These methods target throughput and footprint and validate larger models than earlier BERT-scale work.

**Prior outlier studies.** Earlier work tied large-magnitude features in BERT-family models to LayerNorm, positional structure, anisotropy, and token-frequency effects.

# Evaluation settings

- **Models.** Dense autoregressive Transformers from 125M to 13B for the scale and outlier analysis, trained in fairseq on Books, English Wikipedia, CC-News, OpenWebText, CC-Stories, and English CC100; OPT up to 175B and BLOOM-176B for large-scale inference; GPT-2, fairseq Meta models, and GPT-J for framework cross-checks.
- **Metrics.** C4 validation perplexity for quantization degradation; zero-shot accuracy on WinoGrande, HellaSwag, PIQA, and LAMBADA through the standard language-model evaluation harness for scaling trends.
- **Scope.** Quantize the feed-forward and attention-projection linear layers. The attention function itself has no parameters and stays in 16-bit. The relevant hardware path is NVIDIA Int8 tensor cores; FP8 was not widely supported on GPUs/TPUs at the time.

# Code framework

The starting point is a Transformer linear layer, an absmax quantizer that can operate along a chosen dimension, and an Int8 GEMM with Int32 accumulation.

```python
import torch

def absmax_quantize(x, dim, eps=1e-8):
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=eps) / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale

def int8_matmul(a_i8, b_i8):
    return torch.matmul(a_i8.to(torch.int32), b_i8.to(torch.int32))

class Int8Linear(torch.nn.Module):
    def __init__(self, weight_f16):
        super().__init__()
        # TODO: store the weight and any auxiliary state the chosen scheme needs.
        pass

    def forward(self, x_f16):
        # TODO: compute and return the linear output.
        pass
```
