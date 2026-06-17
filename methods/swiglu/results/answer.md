# SwiGLU, distilled

SwiGLU is a gated replacement for the Transformer's position-wise feed-forward (FFN)
sublayer. Instead of one up-projection through one pointwise nonlinearity, it forms the
hidden representation as the elementwise product of two independent linear projections of the
input — one passed through Swish₁ = SiLU = `x·σ(x)` (the gate), one carried linearly (the
value) — then projects back down. It is one member of a family of GLU variants (GLU, ReGLU,
GEGLU, SwiGLU, Bilinear) obtained by choosing the gate's activation.

## Problem it solves

The FFN sublayer holds the bulk of a Transformer's parameters and compute, yet forms every
hidden unit from a *single* learned linear view of the input bent by a *fixed* pointwise
function. The goal is to make this layer fit the language-modeling objective better at
**matched** parameters and FLOPs, with a drop-in change confined to the FFN (same
`(batch, length, d_model)` contract; no change to attention, normalization, data, optimizer,
or evaluation).

## Key idea

Every standard activation is already a "value × gate of itself": `ReLU(z)=z·1[z>0]`,
`GELU(z)=z·Φ(z)`, `Swish_β(z)=z·σ(βz)` — but the value and the gate are tied to the *same*
projection `xW1`. Untie them. Use two up-projections, a gate projection `W` and a value
projection `V`, and form

```
FFN_SwiGLU(x) = ( Swish_1(xW) ⊗ (xV) ) W2,        Swish_1(z) = z·σ(z) = SiLU(z)
```

The scalar activation derivatives that the reasoning uses:

```
Swish_β'(z) = σ(βz) + βz·σ(βz)(1−σ(βz))
            = β·Swish_β(z) + σ(βz)(1 − β·Swish_β(z))
GELU'(z)    = Φ(z) + z·φ(z),        φ(z) = exp(−z²/2)/√(2π)
```

- **Multiplicative interaction.** Each hidden unit multiplies two learned views of `x`. In the
  un-activated limit `(xW)⊗(xV)`, the "Bilinear" variant is exactly a degree-2 multiplicative
  coupling; with Swish₁ on the gate, the unit keeps that two-view product structure while
  adding smooth, input-dependent modulation. Multiplication (not division) is the safe
  coupling: an activation search found that divisive units explode when the denominator nears
  zero, whereas the product of two finite linear maps is always finite.
- **Clean gradient path.** The value path is carried *linearly*. For the simplified GLU unit
  `X⊗σ(X)`, `∇[X⊗σ(X)] = ∇X⊗σ(X) + X⊗σ'(X)∇X`; the first term multiplies the upstream gradient
  by the gate *value*, not by an activation derivative. Contrast the both-paths-nonlinear
  "gated tanh unit" `∇[tanh(X)⊗σ(X)] = tanh'(X)∇X⊗σ(X) + σ'(X)∇X⊗tanh(X)`, where both paths
  carry saturating derivative factors (`0≤tanh'≤1`, `0≤σ'≤¼`) and there is no derivative-free
  content path. So the nonlinearity goes on the gate; the value stays linear.
- **Swish gate.** Swish₁ is unbounded above, smooth, and non-monotonic (a small sub-zero
  "bump"). As a gate it can pass content at greater-than-unit gain (amplify) and softly
  suppress or sign-flip on the negative bump — strictly richer than σ's pure `(0,1)`
  attenuation. Swish₁ is the `β=1` Swish from an automated activation search (and coincides
  with the SiLU proposed independently for RL); the winning searched functions all had the
  `b(x, g(x))` "raw value recombined with a gate of itself" structure. At `β=1`, its derivative
  has magnitude below 1 for inputs below roughly `1.25`, so it lacks ReLU's exact-1 gradient
  flow on that region — but that is irrelevant here, because the gradient highway runs through
  the *linear* value path, not through the gate's derivative.

## Matched-budget sizing (the 2/3 / 8/3 rule)

Three matrices replace two, so the hidden width must shrink to keep the budget fixed. With
baseline hidden `d_ff` (two matrices, `2·d·d_ff` params) and gated hidden `d_ff'` (three
matrices, `3·d·d_ff'` params):

```
3·d·d_ff' = 2·d·d_ff   ⇒   d_ff' = (2/3)·d_ff.
```

FLOPs scale identically (three `d×d_ff'` matmuls = two `d×d_ff` matmuls). With the standard 4×
expansion `d_ff = 4d`, this gives `d_ff' = (8/3)·d ≈ 2.667·d` (e.g. `d_ff=3072 → 2048` at
`d=768`); the parameter counts match exactly (`2·d·4d = 8d² = 3·d·(8/3)d`). Biases are
omitted. If a host implementation rounds `d_ff'` to a hardware-friendly multiple, that is an
explicit implementation choice and its small budget delta should be recorded separately from the
paper formula.

## Final layer

Canonical bias-free implementation (gate `wi_0`, value `wi_1`, down-projection `wo`; dropout,
when used, is applied to the hidden product before `wo`, matching the Mesh/T5 feed-forward
layer):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: ( Swish_1(xW) ⊗ (xV) ) W2.

    Pass d_ff = (2/3) * d_ff_base; for T5-base, d_ff_base=3072 -> d_ff=2048.
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)  # gate projection W
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)  # value projection V
        self.dropout = nn.Dropout(dropout)
        self.wo = nn.Linear(d_ff, d_model, bias=False)    # down projection W2

    def forward(self, x):                                  # (B, T, d_model) -> (B, T, d_model)
        # F.silu(z) == z * sigmoid(z) == Swish_1(z).
        h = F.silu(self.wi_0(x)) * self.wi_1(x)
        return self.wo(self.dropout(h))
```

LabML-style generic gated feed-forward module (choose the gate activation to get GLU / ReGLU /
GEGLU / SwiGLU / Bilinear; pass `activation=nn.SiLU()` for SwiGLU):

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
        activation=nn.SiLU(),          # Swish_1 gate -> SwiGLU
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
            h = h * self.linear_v(x)                        # Swish_1(xW) ⊗ xV
        return self.layer2(self.dropout(h))                 # ... W2
```

To size `d_ff` for a matched-budget comparison against a baseline FFN of width `d_ff_base`,
set `d_ff = (2/3) * d_ff_base` (i.e. `8/3 * d_model` when the baseline uses `4 * d_model`).
If `2*d_ff_base` is not divisible by 3, choose and document an explicit rounding rule rather than
silently flooring.

Mesh TensorFlow expresses the same hidden product with `dense_product`, choosing the gate
activation per factor (`["swish", "linear"]` ⇒ SwiGLU):

```python
hidden_channels = mtf.Dimension("d_ff", hidden_size)
h = mtf.layers.dense_product(
    x,
    reduced_dims=x.shape.dims[-1:],
    new_dims=hidden_channels,
    activation_functions=["swish", "linear"],   # gate = Swish_1, value = linear
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

| variant   | gate activation          | hidden form              |
|-----------|--------------------------|--------------------------|
| Bilinear  | identity                 | `(xW) ⊗ (xV)`            |
| GLU       | sigmoid σ                | `σ(xW) ⊗ (xV)`           |
| ReGLU     | ReLU                     | `max(0, xW) ⊗ (xV)`      |
| GEGLU     | GELU                     | `GELU(xW) ⊗ (xV)`        |
| SwiGLU    | Swish₁ = SiLU = `x·σ(x)` | `Swish₁(xW) ⊗ (xV)`      |

All use three matrices `(W, V, W2)` and are sized to the matched 2/3 hidden width when
compared against a two-matrix FFN, with any hardware-friendly rounding documented explicitly.
