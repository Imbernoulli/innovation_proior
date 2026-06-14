# ReZero (residual with zero initialization), distilled

ReZero is a one-line change to the residual connection of a deep network: gate each residual
transformation with a single learnable scalar `α` for that layer and initialize it to zero,

  x_{i+1} = x_i + α_i · F(x_i),   α_i = 0 at initialization.

At initialization every block is exactly the identity, so the input-output Jacobian is the identity
matrix, all of its singular values are 1, and the network trivially satisfies *dynamical isometry* —
for any `F`, including ones (ReLU, self-attention) whose own Jacobians have vanishing singular
values. The scalars are trainable, so layers come online during training. The resulting Transformer
block needs no LayerNorm or learning-rate warm-up for initial signal propagation.

## Problem it solves

Depth gives a network exponentially more expressive power but does not come for free: a per-layer
signal/gradient gain `r` compounds to `r^L` over `L` layers, so unless `r ≈ 1` is held at every
layer, signals and gradients vanish or explode exponentially and the optimizer fails (a deeper plain
network reaches *higher* training error than a shallower one). The goal is a single, cheap,
architecture-agnostic residual rule that keeps the per-layer gain at one at initialization and
removes the careful-init / normalization / warm-up scaffolding the field uses to make deep training
work.

## Key idea and why it works

- **Identity at init via one zero-initialized scalar.** The sharp form of "gain ≈ 1 at every layer"
  is dynamical isometry: *all* singular values of `J_io = ∂x_L/∂x_0` near 1 (a strictly stronger
  condition than the mean squared singular value `χ ≈ 1`, which can hide a ruinously spread
  spectrum). Setting `α_i = 0` makes each block `x_{i+1} = x_i`, so `J_io = I` and every singular
  value is exactly 1 — exact dynamical isometry, independent of `F`. This is why one scalar suffices:
  you don't need `F` to be isometric, only the block to be the identity, and `α = 0` annihilates
  `F`'s entire output at init.
- **Why zero and not one.** `α = 1` is the vanilla residual and reproduces the `r^L` pathology.
  Toy model: `L` single-neuron layers sharing weight `w` and scalar `α` give
  `x_L = (1 + αw)^L x_0`, `J_io = (1 + αw)^L`. At `α = 1, w ≈ 1` the gradient
  `w ← w − λ L α x_0 (1+αw)^{L−1} ∂_x C|_{x=x_L}` carries a `(1+w)^{L−1}` factor, forcing a
  learning rate `λ ∝ L^{−1}(1+w)^{−(L−1)}` that is *exponentially small in depth*. At `α = 0`:
  `J_io = 1` (signal preserved), the weight gradient is zero (it has an explicit `α` factor, so the
  deep stack gets no ill-conditioned first step), but `α`'s own update
  `α ← −λ L w x_0 ∂_x C|_{x=x_0}` is finite and evaluated at the well-conditioned input `x_L = x_0`.
  So `α` wakes up first on a clean gradient,
  grows just enough to keep `1+αw` near 1, and routes the trajectory around the poorly-conditioned
  `α ≈ 1` region. The remaining explicit depth factor is `L`, so the stable learning-rate scale is
  polynomial, such as `1/L`, rather than exponentially small. Zero is the unique init giving exact
  identity and well-conditioned first-step dynamics.
- **Replaces normalization, removes warm-up.** In a Transformer, `α` replaces LayerNorm rather than
  supplementing it: LayerNorm's job (control signal scale) is subsumed, and LayerNorm itself
  contributes `2n` vanishing singular values per layer that fight isometry. Warm-up existed only to
  survive the large init-time gradients of the Post-LN Transformer; identity-at-init makes those
  gradients well-behaved, so warm-up is unnecessary.

## Where it sits relative to prior residual schemes

- **ResNet** `σ(x + F(x))`: identity shortcut but the branch is full-strength at init (block is not
  the identity); output variance still compounds with depth.
- **Highway / Gated ResNet** `C·x + T·F(x)`: Highway uses data-dependent gates with their own
  weights and compute; Gated ResNet reduces the transform gate to a single scalar but keeps the
  sigmoid transform/carry split, so a finite bias can lean toward carrying without initializing to
  an exact identity.
- **zero-γ**: zero-init a trailing norm layer's scale to start the block at identity — but applies
  only when such a norm exists and zero-inits a whole channel vector, not one scalar.
- **FixUp**: normalization-free, but an elaborate depth-aware recipe (zero-init last branch layer,
  scale branch weights by `L^{−1/(2m−2)}`, scalar multiplier initialized at *one*).

ReZero is the minimal point in this sequence: one learnable scalar per residual layer, initialized to zero,
input-independent, architecture-agnostic, making the block an exact identity at init.

## Practical notes

- **One `α` per layer, shared across both sublayers** of a Transformer block (attention and FFN); a
  single shared scalar already zeroes the whole block at init.
- **Residual weights are gains:** give them a gentle, small/steady learning rate under aggressive
  schedules, because a large step in one scalar changes a whole layer's contribution.
- **Diagnostics to watch:** `|α_i|` should grow from zero while staying modest, and the `J_io`
  singular-value histogram should remain concentrated near one rather than collapsing toward zero or
  spreading wildly.

## Final algorithm

```
for each residual block i with residual branch F_i:
    alpha_i  <- 0                      # learnable scalar, zero-initialized
forward:
    x_{i+1} = x_i + alpha_i * F_i(x_i) # identity at init; alpha grows in training
for a Transformer layer:
    use the same alpha_i for self-attention and feed-forward sublayer outputs
training:
    use a gentle learning rate for alpha under aggressive schedules
```

## Working code

Canonical residual-with-zero-initialization Transformer encoder layer — one `resweight`
(initialized to 0) per layer, shared across self-attention and feed-forward, no LayerNorm:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


class RZTXEncoderLayer(nn.Module):
    """ReZero Transformer encoder layer: each sublayer output is scaled by a
    single per-layer learnable scalar `resweight`, initialized to zero, so the
    layer is the identity at init (exact dynamical isometry). No LayerNorm."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))   # alpha, zero-initialized
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # x_{i+1} = x_i + alpha * self_attn(x_i)
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0]
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)
        # x_{i+1} = x_i + alpha * FFN(x_i)   (same shared resweight)
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src
```
