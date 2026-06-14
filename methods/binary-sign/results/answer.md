# BitNet / BitLinear (binary sign), distilled

BitNet replaces every floating-point linear projection in a Transformer with `BitLinear`, a
drop-in layer whose forward weights are 1-bit (`{−1, +1}`) and whose activations are 8-bit.
Weights are binarized by the sign function with a per-tensor absmean scale; activations are
quantized with absmax; both non-differentiable steps are trained through the straight-through
estimator (STE); a high-precision latent weight, binarized on the fly each forward pass,
accumulates the optimizer updates and is discarded at inference. A sub-layer normalization
(SubLN) placed right before activation quantization keeps the output variance at order one, and
a large learning rate makes the 1-bit weights reorganize fast. The result is a linear layer
whose core matmul is integer addition rather than floating-point multiply-accumulate.

## Problem it solves

Serving a large language model is bottlenecked by streaming FP16 weights from DRAM and by the
energy of FP16 matmuls in the attention/FFN projections. Post-training quantization collapses at
very low bit-width (1-bit weight-only PTQ of a trained LLM gives astronomically bad perplexity)
because the model never learned the coarse grid; quantization-aware training still optimizes a
float parameter. BitLinear instead makes the forward weight genuinely 1-bit on every step, from
scratch, while keeping the network trainable, stable in a deep stack, and designed to preserve
predictable language-model scaling behavior.

## Key idea

1. **Sign + absmean is the l2-optimal 1-bit summary of a weight.** Minimizing
   `J(B, α) = ‖W − αB‖²` over `B ∈ {−1, +1}ⁿ`, `α > 0`: since `BᵀB = n` and `WᵀW` are constant,
   `J = α²n − 2α·WᵀB + const`, so the optimal binary tensor maximizes `WᵀB`, giving
   `B* = sign(W)` up to arbitrary zero ties; and `∂J/∂α = 0` gives
   `α* = WᵀB*/n = (1/n)Σ|Wᵢ| = mean(|W|)` (absmean). Both the direction (sign) and the magnitude
   (absmean scale `β`) drop out of one least-squares problem. The implemented layer then centers
   `W` before the sign (`sign(W − ᾱ)`) to make both symbols of the 1-bit code carry mass, while
   retaining `β = mean(|W|)` as the global scale.
2. **STE makes `sign` trainable.** `sign` has zero derivative a.e.; in the backward pass treat
   it (and the activation `Clip`/`round`) as the identity. This is the deterministic shadow of an
   unbiased stochastic-binary-neuron gradient estimator, `ĝ = (h − sign(a))·L`.
3. **Latent high-precision weight.** A binary weight cannot accumulate SGD's tiny noisy steps
   (a small step rarely flips a sign). Keep a float latent weight that the optimizer updates and
   binarize it on the fly; only its expected value needs precision. Discarded at inference.
4. **Asymmetric W1A8.** Weights binarize cleanly (flat distributions); activations are harder
   and carry persistent outlier channels, so quantize them to 8-bit absmax, not 1-bit.
5. **SubLN for variance.** `Var(y) = n·β²·E[x̃²]`; the absmean scale makes `n·β²` order one
   under standard initialization, so the remaining quantity to control is the input second
   moment. Normalizing immediately before activation quantization forces `E[LN(x)²] = 1`, so
   `Var(y) ≈ 1`, matching the full-precision layer's scale.
6. **Large learning rate.** Small latent updates fail to flip bits (a bias worst early in
   training); a large LR flips them fast, with the low-bit forward path and SubLN providing the
   stability margin needed to try that more aggressive update.

## Final form

The full BitLinear forward, with weight scale `β = (1/nm)‖W‖₁`, activation absmax
`γ = ‖LN(x)‖_∞`, and half-width `Q_b = 2^{b−1}` (`Q_b = 128` for 8-bit, stored as signed int8
`[-128, 127]`):

```
W̃ = sign(W − ᾱ),          ᾱ = (1/nm) Σ Wᵢⱼ              # 1-bit weight, zero-centered
x̃ = Clip(LN(x)·Q_b/γ, −Q_b+ε, Q_b−ε),  γ = ‖LN(x)‖_∞    # 8-bit absmax activation
y = W̃ · x̃ · (β·γ / Q_b)                                 # matmul, then undo both scales
```

Per-tensor statistics during training, per-token at inference; group quantization/normalization
along the partition dimension keeps model parallelism communication-free. STE on `sign`, `Clip`,
and `round`. Mixed precision: weights/activations low-bit in the forward pass, gradients and
optimizer states high precision.

## Working code

Faithful to the canonical BitLinear pattern (binary `{−1,+1}` weights with absmean scale, absmax
8-bit activations, STE via `z + (quant(z) − z).detach()`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(weight):
    """1-bit weight: sign(W - mean(W)) with absmean scale beta = mean(|W|).
    Returns (binary_weight, beta) with STE to `weight`."""
    beta = weight.abs().mean()                           # beta = (1/n)||W||_1
    e = weight.mean()                                    # per-tensor mean for centering
    w_bin = torch.where(weight > e, torch.ones_like(weight), -torch.ones_like(weight))
    w_ste = weight + (w_bin - weight).detach()           # forward +-1, backward identity
    return w_ste, beta


def activation_quant(x):
    """8-bit absmax activation quantization. Returns (int8-valued_x, gamma/Qb); STE to `x`."""
    Qb = 128.0                                           # 2^(8-1)
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_int = (x * (Qb / gamma)).round().clamp(-Qb, Qb - 1)  # signed int8 range [-128, 127]
    x_ste = x + (x_int - x).detach()                     # forward int8-valued, backward identity
    return x_ste, gamma / Qb


class BitLinear(nn.Module):
    """1-bit-weight, 8-bit-activation drop-in for nn.Linear. self.weight is the latent
    high-precision weight the optimizer updates; it is binarized every forward pass."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)   # SubLN before quant
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.norm(x)                                 # normalize -> Var(y) ~ 1
        x_int, x_scale = activation_quant(x)             # int8-valued activations
        w_bin, w_scale = weight_quant(self.weight)       # 1-bit sign weights
        out = F.linear(x_int, w_bin) * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```

## Notes on the W1A8 layer interface

Some training implementations dequantize the operands before `F.linear` for convenience, using
`F.linear(x_int * x_scale, w_bin * w_scale)`. The factorized BitLinear form above is the same
mathematics but keeps the intended low-bit matmul explicit:
`F.linear(x_int, w_bin) * (w_scale * x_scale)`.
