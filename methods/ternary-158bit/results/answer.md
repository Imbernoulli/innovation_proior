# BitNet b1.58 (ternary 1.58-bit linear), distilled

BitNet b1.58 replaces every `nn.Linear` in a LLaMA-style decoder's attention and SwiGLU
projections with a `BitLinear` whose forward weights are ternary `{-1, 0, +1}` and whose
activations are 8-bit. A high-precision latent weight accumulates the optimizer's updates
and is quantized on the fly; the straight-through estimator carries the gradient back
through the non-differentiable quantizer. The matmul against a ternary weight tensor is
mostly integer addition/subtraction, with weight and activation dequantization scales
applied around the low-bit kernel. Three discrete values carry `log₂(3) ≈ 1.58` bits per
weight — hence "1.58-bit."

## Problem it solves

Inference cost of an LLM is dominated by the linear projections: FP16 floating-point
multiplies in the matmuls and the DRAM→SRAM bandwidth to load the weights. Post-training
quantization can only snap a trained float model onto a grid near its float values and
collapses below ~4 bits (1-bit PTQ is catastrophic). BitNet b1.58 instead trains the model
with natively low-bit weights from scratch (quantization-aware), so the optimizer searches
for a solution that is good under the low-bit forward pass rather than snapping a finished
float solution onto a coarse grid.

## Key idea

- **Ternary weights via absmean RoundClip.** Scale the weight matrix by its mean absolute
  value, then round each entry to the nearest of `{-1, 0, +1}` (clipping the tails):

  ```
  W̃ = RoundClip(W / (γ + ε), -1, 1),    RoundClip(x, a, b) = max(a, min(b, round(x))),
  γ = (1 / nm) Σ_{ij} |W_{ij}|.
  ```

  The absmean scale is the L2-optimal scale for a signed quantizer (XNOR-Net: minimizing
  `||W - α·sign(W)||²` gives `α* = mean(|W|)`). Round-then-clip snaps a normalized weight to
  `0` when `|W| < γ/2`, so the small (weak) weights are **gated to zero** — explicit feature
  filtering — and the survivors become `±1`. Larger `γ` ⇒ more zeros (a sparsity dial);
  absmean is the neutral parameter-free setting and can make the `{-1, 0, +1}` distribution
  nearly uniform.

- **Why ternary, symmetric.** The added `0` gives an off switch and capacity that pure
  binary `{-1, +1}` lacks (binary is worse, especially at smaller sizes). The set is
  symmetric so the matmul is a balanced signed add/subtract; `{0, 1}` is unstable (exploding
  gradients, early divergence). Richer sets like `{-2, -1, 0, 1}` give no demonstrated gain
  over ternary — stop at `{-1, 0, +1}` (Occam).

- **8-bit per-token absmax activations, symmetric.** The paper notation uses
  `Qb = 2^{b-1}` (`Qb = 128` for `b = 8`); the official code uses scale `127 / max|x|`,
  rounds, clamps to the actual int8 range `[-128, 127]`, and dequantizes by the inverse:

  ```
  x̃ = Clip( x · 127 / max|x|, -128, 127 )  (per token),   then divide back by the scale.
  ```

  Per-token (not per-tensor) confines an outlier feature's damage to its own row; symmetric
  `[-Qb, Qb]` maps zero to zero with no zero-point offset to track (simpler than the
  asymmetric `[0, Qb]` pre-nonlinearity shift used in the 1-bit predecessor).

- **Straight-through estimator + latent weights.** Forward uses the quantized value, backward
  treats the quantizer as the identity: `x_q = x + (quant(x) - x).detach()`. A full-precision
  latent weight accumulates the tiny updates and is quantized on the fly each forward; it is
  discarded at inference. Without it, SGD steps never flip a discrete weight.

- **Variance-preserving normalization.** In the binary ancestor derivation, with i.i.d.
  assumptions an output coordinate has
  `Var(y) = n·E[w̃²]·E[x̃²] = n·β²·E[x̃²] ≈ E[x̃²]`, where `β = mean(|W|)` and
  `n·β² ≈ 1` at the intended scale. Full-precision initialization keeps `Var(y)` near
  `1`, which aids stability; to match it, normalize the activations before quantizing.
  Implemented as an RMSNorm placed inside the sublayer, fused into BitLinear so the
  surrounding block drops its own pre-projection norm.

- **LLaMA-alike backbone.** RMSNorm, SwiGLU, rotary embeddings, no biases — for drop-in
  compatibility with the open-source serving ecosystem. Only the parametric projections are
  quantized; embeddings, norms, and residuals stay high precision (the embeddings must
  produce high-precision sampling probabilities).

## Training recipe

- Adam, betas `(0.9, 0.95)`, short warmup; sequence length ~2048.
- **Large, two-stage learning rate.** A small latent update rarely flips a ternary weight
  (worst at the start), so a large LR is needed to drive flips; the discrete model tolerates
  an LR that diverges its full-precision twin. The low-bit loss curve is S-shaped (the big
  drop comes late), so use a high peak LR in the first half, then decay midway to a lower
  rate for the second half.
- **Two-stage weight decay (`0.1 → 0`).** A latent weight's magnitude is a confidence score
  for its discrete value; weight decay shrinks magnitudes ⇒ lower confidence ⇒ more flips.
  Use `0.1` (LLaMA recipe) early for useful churn, then disable it in the second half so the
  ternary weights commit and the model converges.

## Working code (training)

The canonical training-time `BitLinear`. `self.weight` (from `nn.Linear`) is the latent FP
weight; the quantizers return dequantized values so the tensor entering `F.linear` is already
on scale; the `.detach()` wrappers are the straight-through estimator.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Per-token absmax quantization to 8 bits (symmetric, no zero-point).
    x: (..., d). Scale each token by 127 / max|x|, round into the int8 range, divide back."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """Ternary {-1, 0, +1} absmean quantization.
    scale = 1 / mean|W|; round to nearest of {-1,0,1}, clip, divide back.
    A weight with |w| < mean|W| / 2 rounds to 0 (feature filtering)."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def rmsnorm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class BitLinear(nn.Linear):
    """Ternary-weight, 8-bit-activation linear, applied identically in train and eval.
    Built on nn.Linear so self.weight is the high-precision latent parameter.
    RMSNorm is fused in, so the surrounding LLaMA block removes its pre-projection norm."""

    def forward(self, x):
        w = self.weight
        x_norm = rmsnorm(x)
        # Straight-through estimator: forward = quantized, backward = identity
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
```

To build the model from a LLaMA LLM: replace every `nn.Linear` in attention and the SwiGLU
feed-forward with `BitLinear`, and remove the RMSNorm before attention and the feed-forward
(BitLinear has it built in).

## Working code (inference)

At inference the weights are offline-quantized once to ternary, the STE is dropped, the
RMSNorm is fused with the activation quantizer, and `F.linear` is replaced by a low-bit
kernel that does the integer add/subtract and applies the dequant scales afterward.

```python
import torch
import torch.nn as nn


def activation_norm_quant(x, eps=1e-5):
    """Fused RMSNorm + per-token 8-bit absmax quantization (kernel-fusable).
    Returns the int8 activations and the per-token dequant scale."""
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)   # RMSNorm
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale


class BitLinear(nn.Linear):
    """Inference BitLinear. self.weight holds ternary values; self.weight_scale = 1 / mean|W|."""

    def forward(self, x):
        w = self.weight                 # ternary {-1, 0, +1}
        w_scale = self.weight_scale     # = 1 / mean|W| from offline quantization
        x_quant, x_scale = activation_norm_quant(x)
        y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale   # integer add/sub matmul + dequant
        return y
```

## Relation to prior methods

- **Sign + absmean binary (XNOR-Net):** ternary RoundClip generalizes the binary case;
  `γ = mean(|W|)` is the same L2-optimal absmean scale, now used as the rounding scale that
  also defines the zero-threshold `γ/2`.
- **Absmax / per-token activation quantization (LLM.int8()):** reused directly for the 8-bit
  activations; per-token to survive outlier features, but symmetric (no zero-point).
- **Centralized binarization (BiT):** the prior 1-bit BitNet centralized weights before the
  sign; ternary instead lets the explicit `0` provide the off switch and the capacity gain.
- **Straight-through estimator (Bengio et al. 2013) + latent weights (BinaryConnect):** the
  training machinery that makes quantization-aware training of a discrete forward op work.
- **Sub-LayerNorm (Magneto):** the normalization placement that preserves the matmul output
  variance at ≈ 1; realized here as a fused RMSNorm.
- **The original 1-bit BitNet (`{-1, +1}`):** ternary `{-1, 0, +1}` keeps the low-bit
  computation paradigm while adding an explicit off state; symmetric per-token activation
  scaling replaces the asymmetric `[0, Qb]` pre-nonlinearity shift.
