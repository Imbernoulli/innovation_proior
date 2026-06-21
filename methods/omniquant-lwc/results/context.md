# Context

## Research question

Quantize a large language model's weights (and, in the full method, activations) to very low bit-widths
— 2, 3, 4 bits — without the cost of full quantization-aware training, which for a billion-parameter
LLM means backpropagating the task loss through the whole network for thousands of steps on large data.
The competing regime is post-training quantization (PTQ): cheap, data-light, gradient-free. The
question: can a PTQ-grade budget (a small calibration set, a short per-block optimization, no
end-to-end backprop through the full model) recover near-QAT accuracy at INT2/INT3/INT4? The piece in
focus here is the *weight*-side mechanism: how to set the per-group clipping range of the quantizer so
that the rounded weights keep the model's output close to full precision.

## Background

A uniform affine quantizer maps a real tensor `W` to integer codes and back via a scale `h` and a
zero-point `z`:

```
W_q = clamp( round(W / h) + z, 0, 2^N - 1 ),     W_dequant = (W_q - z) * h.
```

The clipping/dynamic range is what determines `h`: from per-group extremes `xmax`, `xmin`,
`h = (xmax - xmin) / (2^N - 1)` (asymmetric) or, for symmetric signed quantization,
`h = max(|xmax|, |xmin|) / (2^{N-1} - 1)`. Standard min-max PTQ sets `xmax`/`xmin` to the literal
per-group extremes — a no-clip cover.

Load-bearing prior concepts:
- **Round-to-nearest / min-max PTQ.** Cover the extremes, round once.
- **Straight-through estimator (Bengio et al., 2013).** Treat `round` as the identity on the backward
  pass so a gradient can reach any continuous parameter feeding the quantizer (here, the clipping
  range). This is what makes the clipping range *learnable* without differentiating through the round.
- **Quantization-aware training (QAT).** End-to-end backprop of the task loss through the quantized
  network recovers low-bit accuracy, requiring full-model gradients, large data, and many steps.
- **Block-wise / layer-wise reconstruction PTQ (AdaRound; BRECQ; GPTQ).** Optimize quantization to
  minimize a *local* reconstruction error per layer or block on a small calibration set, instead of the
  global task loss — the regime this method operates in.

## Baselines

- **RTN / min-max PTQ.** Fixed clipping at the literal extremes; no calibration of the range.
- **GPTQ (Frantar et al., 2022).** Second-order, error-compensating weight rounding per layer; does not
  learn a clipping range.
- **AWQ (Lin et al., 2023).** Activation-aware per-channel scaling to protect salient weights; a
  closed-form transform, not a learned clip.
- **Fixed-clip / percentile PTQ.** Clip at a hand-set percentile of the weight magnitude.

## Evaluation settings

- **Models.** LLaMA / LLaMA-2 family (7B–70B) and OPT, quantized weight-only (W2/W3/W4, per-group) and
  weight-activation.
- **Calibration.** A small set of 128 sequences (e.g. from C4), used for a short *per-block* optimization
  — no end-to-end backprop through the full model.
- **Protocol.** Each transformer block is quantized in turn; the clipping factors (and, in the full
  method, equivalent-transformation parameters) are optimized to minimize the block's output
  reconstruction error vs. the full-precision block, with the straight-through estimator carrying the
  gradient through the round. A handful of epochs per block.
- **Metric.** WikiText-2 / C4 perplexity (held out) and zero-shot accuracy, vs. full precision, across
  bit-widths and model sizes.

## Code framework

The primitives that already exist: an autodiff framework; per-group uniform affine quantize/dequantize
(scale/zero-point from clipping extremes, clamp + round); a straight-through `round_ste`; a small
calibration loader; and a per-block reconstruction loop that, for each block, runs forward on the
calibration inputs, compares quantized-block output to full-precision-block output (an MSE), and steps
an optimizer over whatever quantization parameters are registered as learnable. The open question is
how the per-group clipping range is parameterized so it can be *learned*.

```python
import torch, torch.nn as nn

CLIPMIN = 1e-5

def round_ste(x):
    # round in forward, straight-through (gradient = 1) in backward
    return (x.round() - x).detach() + x

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits, group_size, shape, symmetric=False, lwc=False):
        super().__init__()
        self.n_bits = n_bits
        self.qmin, self.qmax = 0, 2 ** n_bits - 1
        self.group_size = group_size
        self.symmetric = symmetric
        self.lwc = lwc
        # TODO: if lwc, register learnable per-group clip parameters

    def calibration(self, x):
        # per-group extremes
        xmax = x.amax(dim=-1, keepdim=True)
        xmin = x.amin(dim=-1, keepdim=True)
        # TODO: if lwc, adjust xmax/xmin using learnable parameters
        # scale / zero-point from (possibly clipped) extremes
        ...

    def fake_quant(self, x, scale, round_zero_point):
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        return x_dequant.mul(scale)

    def forward(self, x):
        # TODO: compute (possibly learned-clip) scale/zero-point, then fake_quant
        ...
```
