# GeGLU, distilled

GeGLU is a gated replacement for the Transformer's position-wise feed-forward (FFN)
sublayer. Instead of one up-projection through one pointwise nonlinearity, it forms the
hidden representation as the elementwise product of two independent linear projections of the
input — one passed through GELU (the gate), one carried linearly (the value) — then projects
back down. It is one member of a family of GLU variants (ReGLU, GEGLU, SwiGLU, Bilinear)
obtained by choosing the gate's activation.

## Problem it solves

The FFN sublayer holds the bulk of a Transformer's parameters and compute, yet forms every
hidden unit from a *single* learned linear view of the input bent by a *fixed* pointwise
function. The goal is to make this layer fit the language-modeling objective better at
**matched** parameters and FLOPs, with a drop-in change confined to the FFN (same
`(batch, length, d_model)` contract; no change to attention, normalization, data, optimizer,
or evaluation).

## Key idea

Every standard activation is already a "value × gate": `ReLU(z)=z·1[z>0]`, `GELU(z)=z·Φ(z)`,
`Swish_β(z)=z·σ(βz)` — but the value and the gate are tied to the same projection `xW1`.
Untie them. Use two up-projections, a gate projection `W` and a value projection `V`, and form

```
FFN_GEGLU(x) = ( GELU(xW) ⊗ (xV) ) W2
```

The scalar activation derivatives used in the reasoning are:

```
GELU'(z) = Φ(z) + z·φ(z),        φ(z) = exp(−z²/2)/√(2π)
Swish_β'(z) = σ(βz) + βz·σ(βz)(1−σ(βz))
             = β·Swish_β(z) + σ(βz)(1−β·Swish_β(z))
```

- **Multiplicative interaction.** Each hidden unit multiplies two learned views of `x`.
  In the un-activated limit `(xW)⊗(xV)`, the "Bilinear" variant is exactly a degree-2
  multiplicative coupling; with GELU on the gate, the unit keeps that two-view product structure
  while adding smooth input-dependent modulation.
- **Clean gradient path.** The value path is carried *linearly*. For the simplified GLU unit
  `X⊗σ(X)`,
  `∇[X⊗σ(X)] = ∇X⊗σ(X) + X⊗σ'(X)∇X`; the first term multiplies the upstream gradient by the gate
  value, not by an activation derivative. Contrast the both-paths-nonlinear "gated tanh unit"
  `∇[tanh(X)⊗σ(X)] = tanh'(X)∇X⊗σ(X) + σ'(X)∇X⊗tanh(X)`, where both paths carry saturating
  derivative factors (`0≤tanh'≤1`, `0≤σ'≤¼`) and there is no derivative-free content path. So
  the nonlinearity goes on the gate; the value stays linear.
- **GELU gate.** Using GELU on the gate keeps consistency with the FFN activation being
  improved. Since `GELU(z)=zΦ(z)`, the gate activation is smooth, grows above 1 for sufficiently
  positive `z`, and becomes slightly negative for some negative `z`; unlike σ's pure `(0,1)`
  attenuation, it can amplify and softly sign-flip the value path.

## Matched-budget sizing (the 2/3 / 8/3 rule)

Three matrices replace two, so the hidden width must shrink to keep the budget fixed. With
baseline hidden `d_ff` (two matrices, `2·d·d_ff` params) and gated hidden `d_ff'` (three
matrices, `3·d·d_ff'` params):

```
3·d·d_ff' = 2·d·d_ff   ⇒   d_ff' = (2/3)·d_ff.
```

FLOPs scale identically (three `d×d_ff'` matmuls = two `d×d_ff` matmuls). With the standard
4× expansion `d_ff = 4d`, this gives `d_ff' = (8/3)·d ≈ 2.667·d` (e.g. `d_ff=3072 → 2048` at
`d=768`). Biases are omitted; in a GPT-style implementation the exact target `(8/3)·d` is
rounded up to a multiple of 64 for matmul-friendly shapes, an explicit sub-1% implementation
nudge at common widths.

## Final layer

GPT-style implementation (gate `w1`, value `w3`, down-projection `c_proj`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """GeGLU feed-forward sublayer: ( GELU(xW) ⊗ (xV) ) W2, at 8/3 hidden width
    so 3 matrices match the budget of the baseline 2-matrix FFN at 4x expansion."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)            # (2/3) * 4d
        hidden_dim = ((hidden_dim + 63) // 64) * 64        # round up to a multiple of 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)      # gate proj W
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)      # value proj V
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down proj W2
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                                   # (B, T, n_embd) -> (B, T, n_embd)
        # GeGLU: GELU(xW) * (xV), then project back
        return self.dropout(self.c_proj(F.gelu(self.w1(x)) * self.w3(x)))
```

LabML-style generic gated feed-forward module (choose the gate activation to get GLU / ReGLU /
GEGLU / SwiGLU / Bilinear):

```python
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise FFN with optional gated hidden layer."""

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.0,
        activation=nn.GELU(),
        is_gated=True,
        bias1=False,
        bias2=False,
        bias_gate=False,
    ):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)       # gate path W
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)       # output path W2
        self.act = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)  # value path V
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.layer1(x))
        if self.is_gated:
            h = h * self.linear_v(x)                        # f(xW) ⊗ xV
        return self.layer2(self.dropout(h))                 # ... W2
```

To size `d_ff` for a matched-budget comparison against a baseline FFN of width `d_ff_base`,
set `d_ff = (2/3) * d_ff_base` (i.e. `8/3 * d_model` when the baseline uses `4 * d_model`).

Mesh TensorFlow expresses the same hidden product with `dense_product`:

```python
hidden_channels = mtf.Dimension("d_ff", hidden_size)
h = mtf.layers.dense_product(
    x,
    reduced_dims=x.shape.dims[-1:],
    new_dims=hidden_channels,
    activation_functions=["gelu", "linear"],
    use_bias=False,
    name="wi",
)
y = mtf.layers.dense(
    h,
    x.shape.dims[-1],
    reduced_dims=h.shape.dims[-1:],
    activation=None,
    use_bias=False,
    name="wo",
)
```

## The GLU-variant family

| variant   | gate activation | hidden form              |
|-----------|-----------------|--------------------------|
| Bilinear  | identity        | `(xW) ⊗ (xV)`            |
| GLU       | sigmoid σ       | `σ(xW) ⊗ (xV)`           |
| ReGLU     | ReLU            | `max(0, xW) ⊗ (xV)`      |
| GEGLU     | GELU            | `GELU(xW) ⊗ (xV)`        |
| SwiGLU    | Swish₁ = SiLU   | `Swish₁(xW) ⊗ (xV)`      |

All use three matrices `(W, V, W2)` and are sized to the matched 2/3 hidden width when
compared against a two-matrix FFN, with optional round-up to a hardware-friendly multiple.
