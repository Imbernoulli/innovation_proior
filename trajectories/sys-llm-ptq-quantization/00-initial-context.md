## Research question

A pretrained LLaMA / Llama-2 model (7B–70B) is already trained and fixed. The task is to compress it for serving by quantizing its FP16 tensors to low-bit integers after training, with no retraining and no gradient update to the original weights. Quality is measured by WikiText-2 perplexity (lower is better) and by zero-shot commonsense-reasoning accuracy averaged over standard tasks (higher is better). The only free variable is the quantization method: how weights and activations are mapped to a uniform integer grid and what preprocessing is permitted before that mapping.

At most we have a small calibration set (a few hundred sequences) and a few hours of single-GPU compute to fit quantization parameters. The useful regimes split into two axes:

- **Weight-only quantization (WxA16).** Weights are quantized to 3–4 bits while activations stay in FP16. This cuts memory footprint and speeds up batch-1 generation, which is memory-bandwidth-bound.
- **Weight-and-activation quantization (WxAx).** Both operands are quantized so matmuls run on integer tensor cores. This speeds up compute-bound large-batch serving.

## Prior art / Background / Baselines

The simplest baseline and the starting point is **round-to-nearest (RTN)**: pick a per-channel or per-group scale from the maximum magnitude, divide, round, clamp. Several methods build from or around this foundation.

- **Round-to-nearest (RTN).** Maps each tensor to a uniform symmetric grid using a scale from the maximum absolute value. At 8 bits, performance is strong. At lower bits, per-channel granularity uses one scale per output channel, while group-wise (g128) granularity applies one scale per contiguous block of 128 weights for finer resolution. On LLaMA-7B 3-bit per-channel, WikiText-2 perplexity is **25.54** versus an FP16 reference of **5.68**; on Llama-2-7B W4A4 perplexity is on the order of **2×10³**.

- **GPTQ.** Accumulates the layer input second-moment matrix during calibration and quantizes columns one at a time, feeding the rounding residual forward through the inverse Hessian. On LLaMA-7B 3-bit per-channel it reports perplexity **8.07**; FP16 reference is **5.68**.

- **AWQ.** Identifies salient weight channels by activation magnitude, scales them up before rounding, and searches the scaling factor against layer output error. On Llama-2-7B INT3-g128 it reports perplexity **6.24** versus RTN-g128 **6.66** and FP16 **~5.47**.

- **SmoothQuant.** Migrates activation outliers into the weight matrix offline with a per-channel equivalence transform so that W8A8 quantization survives. On OPT-175B W8A8 it achieves zero-shot average **66.8%**, close to FP16 **66.9%**; on Llama-2-7B W4A4 it reports perplexity **83.12**.

- **QuaRot.** Applies computationally invariant rotations to spread activation outliers before quantizing weights, activations, and KV cache to 4 bits. On Llama-2-7B W4A4KV4 it reports perplexity **6.10** versus FP16 **5.47**, with similar reported gaps at 13B (**5.40** vs **4.88**) and 70B (**3.79** vs **3.32**).

## Fixed substrate / Code framework

The substrate is a fixed pretrained Transformer LLM. Quantization is uniform integer quantization on a per-channel or group-of-128 (g128) grid. A small calibration set is used only to fit scales, Hessians, or rotations; the model weights are never updated. Evaluation is WikiText-2 perplexity at sequence length 2048 and a fixed zero-shot reasoning suite. The bit-width and grouping are reported with every number, because per-channel and g128 results are not interchangeable.

The harness loads the model, records the FP16 reference, then walks the transformer blocks one at a time. For each linear layer it collects calibration inputs, calls the quantizer, writes the quantized-dequantized weight back in place, and propagates the quantized block's outputs to the next block. The quantizer sees one linear layer at a time.

## Editable interface

The editable region is the `WeightQuantizer` class (and any helpers it calls). It must implement `configure(bits, perchannel, sym)`, `find_params(x)`, and `quantize(x)`, returning a same-shape, same-dtype tensor that is the fake-quantized weight. The default fill below is plain RTN: the scale is `max(|x|) / (2^(bits-1) - 1)`, and the forward pass is round-then-dequantize.

```python
import torch


def sym_quant_dequant(x, scale, maxq):
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return scale * q


class WeightQuantizer(torch.nn.Module):
    """Default fill: symmetric per-channel round-to-nearest."""

    def configure(self, bits, perchannel=True, sym=True):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.maxq = 2 ** (bits - 1) - 1

    def find_params(self, x):
        x = x.flatten(1) if self.perchannel else x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=x.device)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        xmax = torch.maximum(xmin.abs(), xmax).clamp(min=1e-5)
        self.scale = (xmax / self.maxq).reshape([-1] + [1] * (x.dim() - 1))

    def quantize(self, x):
        return sym_quant_dequant(x, self.scale, self.maxq)
```

Group-wise g128 quantization is obtained by reshaping each row into contiguous blocks of 128 columns and running the same routine independently per block.

## Evaluation settings

Primary metric is WikiText-2 perplexity at the stated weight/activation bit-width; secondary metric is zero-shot accuracy where the source reports it. Every number is taken from the named method's own paper table or repository README, not re-run here. Numbers are labeled with exact bit-width and grouping (per-channel vs g128, weight-only WxA16 vs weight+activation WxAx), and they are compared only within matched settings. Weight-only numbers are comparable among themselves at the same bit-width and grouping; W8A8 numbers are read from the SmoothQuant OPT-175B setting; W4A4KV4 numbers are read from the same comparison table at the same setting.
