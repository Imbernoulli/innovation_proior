# Squared ReLU (ReLU²), distilled

Squared ReLU is a pointwise activation for the Transformer feed-forward network: rectify, then
square. It replaces the FFN's usual ReLU/GELU/Swish with `act(z) = max(z, 0)²`, adding no
parameters and changing no dimension, yet it improves language-model sample efficiency — a
parameter-free activation-level change that lowers compute-to-quality.

## Problem it solves

The FFN dominates a Transformer block's parameter and compute budget, and its middle activation is
the one knob that is free to change (no new weights, negligible FLOPs). ReLU, GELU and Swish are all
asymptotically linear, so they differ only near the origin — which is why GELU barely beats ReLU on
LM perplexity. The goal is a drop-in activation that raises sample efficiency enough to cut total
training compute, without the extra weight matrix and width-shrink that the gated (GLU-variant)
FFNs require.

## Key idea

Use a **rectified quadratic**:

```
act(z) = max(z, 0)² = relu(z)²,     applied between the FFN's two matmuls
FFN(x, W_fc, W_proj) = ( relu(x W_fcᵀ)² ) W_projᵀ
```

Three things make it work, each derivable rather than assumed:

- **It is the GLU benefit with one matrix instead of two.** A ReGLU hidden unit is
  `max(uᵀx, 0)·(vᵀx)`. Tie the two weight matrices (`u = v`) and it becomes `max(uᵀx, 0)·(uᵀx)`,
  which equals `(uᵀx)²` when `uᵀx > 0` and `0` otherwise — i.e. `relu(uᵀx)²`. So squared ReLU is
  ReGLU with tied weights, applied to the pre-activation the up-projection already produces. The
  second multiplicative factor is the up-projection's own output, so no third matrix `V` and no
  `2/3` inner-width shrink are needed.
- **Super-linear asymptotics and a magnitude-aware gradient.** As `z → ∞`, `relu(z)² ~ z²`, unlike
  the asymptotically-linear ReLU/GELU/Swish. Its derivative is `d/dz[relu(z)²] = 2·relu(z)` (the
  `1[z>0]` indicator is subsumed by `relu`), which grows linearly and is unbounded above — strongly
  firing units receive proportionally larger gradients, where GELU/Swish derivatives saturate near 1.
- **Minimal degree-2 multiplicative interaction.** `relu(z)²` is the leanest rectified higher-order
  term — the same second-order/multiplicative structure that ReGLU/GEGLU and the `x³` term in the
  tanh-approximate GELU carry — captured with a single self-multiplication and no parameters.

## Design choices and why

- **Power 2 (not 3, 4, …):** the minimal super-linear rectified polynomial; its gradient grows only
  linearly (`2·relu(z)`), where a power-`n` unit has gradient `n·relu(z)^{n-1}` that grows faster and
  overflows sooner in low precision (bf16). Rectified powers are sharper for higher `n`, but 2 is the
  safe, smallest step away from linear.
- **Rectify, not plain `x²`:** `x²` is even and non-monotonic, firing identically for `+z` and `−z`,
  which discards the sign of the pre-activation and collapses half the representation. `relu(z)²` is
  monotone non-decreasing — it keeps ReLU's "off" behavior for negatives and only curves the active
  half upward.
- **No third matrix / no retune:** the activation is a literal drop-in into the existing two-matrix
  FFN, parameter-matched by construction, tested with the same learning-rate schedule.

## Backward pass

With row-batched `x ∈ R^{N×d}`, `W_fc ∈ R^{4d×d}`, `W_proj ∈ R^{d×4d}`,
`h = x W_fcᵀ ∈ R^{N×4d}`, `r = relu(h)`, `a = r²`, `y = a W_projᵀ ∈ R^{N×d}`,
and upstream `g = ∂L/∂y ∈ R^{N×d}`:

```
∂L/∂a      = g @ W_proj                         # (N,d) @ (d,4d) -> (N,4d)
∂L/∂h      = 2·relu(h) ⊙ (∂L/∂a)                # elementwise, (N,4d)
∂L/∂W_proj = gᵀ @ a                              # (d,N) @ (N,4d) -> (d,4d)
∂L/∂W_fc   = (∂L/∂h)ᵀ @ x                        # (4d,N) @ (N,d) -> (4d,d)
∂L/∂x      = (∂L/∂h) @ W_fc                      # (N,4d) @ (4d,d) -> (N,d)
```

The derivative identity is exact at the origin as well: `relu(0) = 0`, so the chosen subgradient is
`0`. The torch baseline saves both `h` and `relu(h)` in the forward; `relu(h)` supplies the
`2·relu(h)` factor and the activation value for `∂L/∂W_proj`.

## Canonical form

The activation itself, as used in the FFN (mesh-tensorflow's primitive, `square(relu(x))`; the
annotated PyTorch form is identical):

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    """act(z) = max(z, 0)^2."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return x * x
```

## Working code — squared-ReLU FFN with a hand-written backward

A pure-PyTorch FFN core that uses squared ReLU and computes the backward by hand, mirroring the
MLS-Bench `relu_sq_torch` baseline: it saves the pre-activation and `relu(h)`, then applies the
analytic `2·relu(h)` factor in backward. Drop-in for a GELU FFN; faster, no extra parameters.

```python
import torch
from torch.nn import functional as F


def fused_mlp_forward(x, w_fc, w_proj):
    """FFN core: up-project -> squared ReLU -> down-project, with a custom backward.

    x:      (B*T, n_embd)
    w_fc:   (4*n_embd, n_embd)
    w_proj: (n_embd, 4*n_embd)
    returns (B*T, n_embd)
    """

    class ReLUSquaredMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_fc, w_proj):
            h = x @ w_fc.t()              # pre-activation z
            relu_h = F.relu(h)           # r = relu(z)
            act = relu_h * relu_h        # a = relu(z)^2
            out = act @ w_proj.t()
            ctx.save_for_backward(x, w_fc, w_proj, h, relu_h)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, w_fc, w_proj, h, relu_h = ctx.saved_tensors
            dtype = grad_output.dtype
            d_act = grad_output @ w_proj.to(dtype)        # dL/da = g @ W_proj
            d_h = 2 * relu_h.to(dtype) * d_act            # d/dz[relu(z)^2] = 2*relu(z)
            act_sq = (relu_h * relu_h).to(dtype)          # a = relu(z)^2
            grad_w_proj = grad_output.t() @ act_sq        # dL/dW_proj = g^T @ a
            grad_w_fc = d_h.t() @ x.to(dtype)             # dL/dW_fc   = d_h^T @ x
            grad_x = d_h @ w_fc.to(dtype)                 # dL/dx      = d_h @ W_fc
            return grad_x, grad_w_fc, grad_w_proj

    return ReLUSquaredMLP.apply(x, w_fc, w_proj)
```

## Relation to prior FFN activations

- **ReLU / GELU / Swish FFN:** asymptotically linear; squared ReLU keeps ReLU's sign-gate but makes
  the active half super-linear, with an unbounded-above (magnitude-aware) gradient.
- **ReGLU / GEGLU / SwiGLU (GLU variants):** multiplicative gates with a third weight matrix `V` and
  a `2/3` inner-width shrink to stay parameter-matched. Squared ReLU = ReGLU with `V = W` (tied),
  recovering the multiplicative benefit from the up-projection alone — same second-order interaction,
  one matrix, no shrink.
- **Rectified polynomials (Krotov & Hopfield 2016):** `F(z) = zⁿ` for `z ≥ 0`; squared ReLU is the
  `n = 2` case used as an activation — the gentlest super-linear rung.
