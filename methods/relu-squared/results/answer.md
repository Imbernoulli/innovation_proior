# Squared ReLU (ReLU²), distilled

Squared ReLU replaces the Transformer feed-forward network's pointwise nonlinearity with
`act(z) = max(0, z)^2` — rectify, then square. It is a drop-in, parameter-free change to the
FFN sublayer: the two-matrix `Linear -> act -> Linear` sandwich and the hidden width are
untouched, and only the activation between the projections changes. The official Mesh
TensorFlow/Primer configuration sets `DenseReluDense.activation = ["squared_relu"]`, and
Mesh TensorFlow defines `squared_relu(x)` as `square(relu(x))`; the equivalent PyTorch
expression is `F.relu(x).square()`.

## Problem it solves

Lower the validation loss / improve the sample efficiency of autoregressive (decoder-only)
language-model pretraining with a strictly modular, feed-forward-only intervention: keep the
`(B, T, d) -> (B, T, d)` shape, do not modify attention / normalization / dataset / optimizer
/ evaluation, add no parameters, and require no re-tuning so it can be dropped into an existing
codebase.

## Key idea

The FFN's only nonlinearity is its pointwise activation, and the activations that displaced
ReLU — GELU `z·Phi(z)` and Swish `z·sigmoid(beta z)` — are all smooth near zero but
**asymptotically linear** on the positive side (`act(z) -> z` as `z -> +inf`). They tacitly
assume the positive branch should be ~linear. Squared ReLU breaks that assumption with a
**quadratic positive branch** while keeping ReLU's negative-killing sparsity:

- `act(z) = 0` for `z <= 0` (off units stay off; sparsity preserved);
- `act(z) = z^2` for `z > 0`, so `act(z)/z = z -> +inf` — a genuinely different, faster-than-
  linear growth regime. Strongly activated units dominate quadratically, sharpening the layer.
- The derivative is `2z` for `z > 0` and `0` for `z < 0`; it is continuous at the origin (no
  hard ReLU kink in the slope) and never saturates.

It must be **rectify then square**, not a raw `z^2`: the bare square is even, loses the sign,
and turns "feature absent" (`z < 0`) into a large output — destroying the gating. So
`max(0, z)^2` is exactly the degree-2 member of the rectified-polynomial activation family
`max(0, z)^p` (motivated by Krotov & Hopfield's rectified-polynomial memory/activation
duality), the minimal step beyond linear.

## Why it is the tied-ReGLU case

`max(0, z)^2 = max(0, z) · z` for all `z` (for `z > 0`: `z·z = z^2`; for `z <= 0`:
`0·z = 0`). Therefore squared ReLU applied after an affine projection `z = xW + b` **is
ReGLU with its gate and value affine maps tied** (same `W`, and the same bias when biases
are present):

```
ReGLU(x) = max(0, xW + b) ⊗ (xV + c)
set V = W and c = b:
max(0, xW + b) ⊗ (xW + b) = (ReLU(xW + b))^2.
```

So squared ReLU is the tied-map, single-hidden-projection special case of ReGLU. It keeps
the GLU-like multiplicative structure — a real degree-2 product term, using two copies of
the same pre-activation — while avoiding an independent value projection:

- **Two matrices, not three.** ReGLU/GEGLU/SwiGLU need a gate projection `W`, a value
  projection `V`, and the down-projection; Shazeer's T5-base comparison reduces `d_ff`
  from 3072 to 2048 (a 2/3 factor) to match parameter and operation counts. Tying the
  gate/value maps keeps two matrices and the ordinary FFN width.
- **No extra parameters, no new hyperparameter, no width adjustment.**
- **Simpler and cheaper per element:** no `erf`/`exp`/`sigmoid`, just a rectify and a square.

Degree 2 (not 3, 4, ...) is the choice because higher rectified polynomials grow faster but are
numerically meaner (overflow on large activations, underflow on small ones, especially in
low precision through a deep stack), and only degree 2 has the clean single-product
tied-ReGLU interpretation.

## Final form

```
FFN(x) = ( max(0, x W1 + b1) )^2 W2 + b2,
W1: d -> d_ff,   W2: d_ff -> d,   usually d_ff = 4d.
```

## Working code

Filling the FFN's activation slot; this is the PyTorch equivalent of the official
`square(relu(x))` activation in a two-matrix MLP. Bias and dropout follow the host model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Transformer feed-forward sublayer with a squared-ReLU (ReLU squared) activation.

    Up-project to 4x width, apply max(0, z)^2 pointwise, project back down.
    Two matrices and the full 4x hidden width -- no extra parameters versus a
    ReLU/GELU FFN. Equivalent to ReGLU with tied gate/value affine maps.
    Maps (B, T, n_embd) -> (B, T, n_embd).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)            # z = x W1 : up-projection to the 4x hidden state
        x = F.relu(x).square()      # max(0, z)^2  (rectify keeps sparsity, then square)
        x = self.c_proj(x)          # down-projection back to n_embd
        x = self.dropout(x)
        return x
```

## Relation to prior FFN nonlinearities

- **ReLU** `max(0, z)`: positive branch is the identity (piecewise linear, hard kink). Squared
  ReLU keeps the rectification but makes the positive branch quadratic and smooth-sloped.
- **GELU** `z·Phi(z)`, **Swish** `z·sigmoid(beta z)`: smooth near zero but asymptotically
  linear; squared ReLU changes the growth regime instead of just the near-zero shape, and
  needs no transcendental.
- **ReGLU / GEGLU / SwiGLU**: three-matrix gated FFNs with a width cut when matched for
  parameters and operations. Squared ReLU is the tied-affine, two-matrix special case of
  ReGLU — less expressive than independent gate/value maps, but with the multiplicative
  degree-2 term and no extra parameters.
