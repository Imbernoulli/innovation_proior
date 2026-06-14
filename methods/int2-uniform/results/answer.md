# BitLinear with a 2-bit uniform-symmetric weight grid (int2-uniform), distilled

`BitLinear` is a drop-in replacement for `nn.Linear` that trains a Transformer whose linear-layer
weights are low-bit during every forward pass. It keeps a full-precision **latent** weight as the
optimizer's accumulator, but in the forward pass the operands actually multiplied are a
**quantized weight grid** and **int8 activations**, joined to the latent parameters by a
**straight-through estimator** so gradients still reach the latent weights. The `int2_uniform`
instantiation uses a **2-bit, 4-level uniform-symmetric weight grid** `{-1, -1/3, +1/3, +1}` with
a per-tensor **absmean** scale, and **8-bit absmax** activations.

## Problem it solves

Most of an LLM's inference cost is the matrix multiplications in its linear projections: the
weights dominate memory and DRAM bandwidth, and the floating-point multiplies dominate energy.
Making the forward weights tiny integers shrinks all three at once (a weight×activation becomes a
signed integer accumulate). Post-training quantization collapses below ~4 bits because a
float-trained model has no good low-bit neighbor, so the quantization must be experienced during
training — but a quantizer is piecewise-constant, so its true gradient is zero and naive
back-propagation stalls. The method makes 2-bit-weight pretraining trainable and stable.

## Key idea

1. **Latent weight + STE (BinaryConnect 2015; Bengio 2013).** Keep a full-precision `weight`
   the optimizer updates; quantize it on the fly in the forward pass; pass the gradient through
   the quantizer as if it were the identity. The latent weight is the high-resolution
   accumulator SGD needs; the discrete weight is what the matmul uses.
2. **Saturating forward + STE backward (BNN 2016).** The forward clamp pins values beyond the
   grid to the extreme level, which is what stops a latent weight from running off to infinity
   in the regime where pushing it further changes nothing. The detached-difference idiom
   `q + (q − w_n).detach()` gives forward `= q` (with the clamp inside `q`) and backward
   `= identity`; the strict BNN gradient mask `1_{|w_n|≤1}` is the optional add-on that also
   zeroes the backward gradient outside the range.
3. **L2-optimal per-tensor scale = absmean (XNOR-Net 2016).** Minimizing `‖W − s·grid‖²` over
   `s>0` for a sign grid gives `s* = (1/n)‖W‖₁ = mean(|W|)`. It is cheap, outlier-robust, and
   places the typical weight at the grid's `±1`. (Absmax would let one outlier crush all other
   weights toward a single level.)
4. **2-bit uniform-symmetric weight grid `{-1, -1/3, +1/3, +1}`.** Exactly `log2 4 = 2` bits —
   finer magnitude resolution than ternary's `log2 3 ≈ 1.58` bits at the same integer-add cost.
   Symmetric → no zero-point to store/add (the int matmul stays a pure signed accumulate);
   uniform → trivial round-to-nearest with a constant, bounded worst-case step error; fixed (not
   learned) for stability. Round-to-nearest minimizes per-element squared error on a fixed grid.
   The grid has no exact zero, so it trades ternary's explicit "off"/feature-filtering level for
   two genuine bits of magnitude resolution.
5. **8-bit absmax activations (LLM.int8 convention 2022).** Weights are far easier to quantize
   than activations (the contribution is low-bit *weights*), so activations stay at int8.
   Symmetric absmax (`s_x = max(|x|)/127`) never clips the extreme activation — and a clipped
   activation is destroyed, whereas a clipped weight is just a misplaced grid assignment.
6. **Output rescale + variance control.** `y = s · s_x · (grid_weightᵀ · int8_act)`: the inner
   product runs on the fixed-grid weights and int8 activation codes, and the two learned
   per-tensor scales `s`, `s_x` come out and multiply the output, not every element. The
   variance estimate is `Var(y) = n · E[w̃²] E[x̃²] = n · s² E[g²] · E[x̃²]` (`g` a grid value);
   `n · s² E[g²]` is order one under standard init with the absmean scale, so a LayerNorm before
   the activation quant (making `E[x̃²]≈1`) keeps the quantized forward at order-one variance
   across depth.

## Final quantizers

Bit-width = `log2(#levels)`: binary `{-1,+1}` = 1 bit; ternary `{-1,0,+1}` = `log2 3 ≈ 1.58`
bits; the 4-level grid = 2 bits.

**Weight (2-bit uniform-symmetric, absmean scale, STE):**
```
s    = mean(|W|)                            # absmean (XNOR-Net L2-optimal scale)
w_n  = W / s                                # normalize: typical |w_n| ~ 1
code = clip( round( (w_n + 1) * 1.5 ), 0, 3 )      # nearest of 4 codes; clamp = saturation
q    = -1 + code * (2/3)                    # dequantized grid value in {-1,-1/3,+1/3,+1}
W̃   = w_n + (q - w_n).detach()             # STE: forward = q, backward = identity
forward weight = s * W̃
```
(`code = round((w_n+1)·1.5)` because `level(code) = -1 + code·(2/3)`, so the nearest code is
`round((w_n − (−1))/(2/3))`. Boundaries land at the level midpoints: `±2/3`, `0`.)

**Activation (8-bit symmetric absmax, STE):**
```
Q_b  = 127
s_x  = max(|x|) / Q_b                                         # output (dequant) scale
x_n  = x / s_x                                                # in [-Q_b, Q_b]
x_q  = x_n + ( round(clip(x_n, -Q_b, Q_b)) - x_n ).detach()   # STE: forward int, backward identity
forward activation = x_q,  with output scale s_x
```

**Layer output:** `y = (grid_weightᵀ · x_q) · (s · s_x) + bias` — the inner product runs on the
fixed-grid weights and int8 activation codes; the two per-tensor scales `s`, `s_x` dequantize the
output once (not per element); bias is added in full precision.

## Working code

Faithful to the canonical BitLinear (latent FP weight; absmean weight quant; absmax int8
activation; detached-difference STE), instantiated at the 2-bit uniform-symmetric grid. Fills the
`weight_quant` / `activation_quant` / `forward` slots of the `nn.Linear`-drop-in scaffold.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(weight):
    """2-bit uniform-symmetric weight quantization onto {-1, -1/3, +1/3, +1}
    with a per-tensor absmean scale and a straight-through estimator.
    Returns (quantized_weight, scale) with quantized_weight * scale ~= weight."""
    scale = weight.detach().abs().mean().clamp(min=1e-5)     # absmean (L2-optimal scale)
    w_n = weight / scale                                     # typical |w_n| ~ 1
    # round to nearest of 4 uniform levels: codes 0..3 -> -1 + code*(2/3)
    code = ((w_n + 1.0) * 1.5).round().clamp(0, 3)
    q = -1.0 + code * (2.0 / 3.0)                            # grid value in [-1, 1]
    w_q = w_n + (q - w_n).detach()                           # STE: forward q, backward identity
    return w_q, scale


def activation_quant(x):
    """Symmetric absmax 8-bit activation quantization with STE.
    Returns (quantized_x, scale) with quantized_x * scale ~= x."""
    Qb = 127.0
    scale = x.detach().abs().max().clamp(min=1e-5) / Qb      # absmax dequant scale
    x_n = x / scale
    x_int = x_n.round().clamp(-Qb, Qb)                       # int8-range values
    x_q = x_n + (x_int - x_n).detach()                       # STE
    return x_q, scale


class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear. Keeps a full-precision latent weight,
    but multiplies a 2-bit weight grid against int8 activations in the forward pass.
    Same quantization path in train and eval."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))   # latent FP weight
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight)            # grid weights + absmean scale
        x_q, x_scale = activation_quant(x)                  # int8 activations + absmax scale
        out = F.linear(x_q, w_q, None)                      # low-bit inner product in the target kernel
        out = out * (w_scale * x_scale)                     # dequantize by the two operand scales
        if self.bias is not None:
            out = out + self.bias                           # bias in full precision
        return out
```

## Relation to prior methods

- **BinaryConnect / BNN:** latent FP weight + on-the-fly sign binarization + (saturating) STE.
  This keeps that training recipe but replaces the bare `{-1,+1}` grid with a scaled 4-level
  uniform-symmetric grid.
- **XNOR-Net:** supplies the L2-optimal scale `mean(|W|)`; here it is the per-tensor unit for the
  multi-level grid (the same scale that is optimal for the sign grid).
- **Ternary `{-1,0,+1}` (1.58-bit):** the round-to-nearest-with-absmean mechanism is the same;
  the 2-bit grid drops the explicit zero and adds a level to reach 4 = 2 bits, buying magnitude
  resolution at the cost of the "off" state.
- **LLM.int8 absmax:** the activation path — symmetric, no zero-point, scale set by the max so
  the extreme activation is never clipped.
