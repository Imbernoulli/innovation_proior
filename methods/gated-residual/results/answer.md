# ReZero, distilled

ReZero (residual with zero initialization) gates every residual branch with a single
learnable scalar `alpha`, initialized to zero, and leaves the skip path unmodulated:

```
x_{i+1} = x_i + alpha_i * F(x_i),    alpha_i = 0 at initialization.
```

One scalar per residual block (for a Transformer, one shared across a layer's two sublayers).
For dimension-preserving residual layers, initialization makes every block the exact identity,
so the whole stack is the identity map: its input-output Jacobian is `J = I`, all singular
values are exactly 1, and the network trivially satisfies *dynamical isometry* — for *any*
branch `F`, including ReLU and self-attention, because `F` is multiplied by zero. The scalar
leaves zero on the first optimization step (it receives gradient `F(x)`), after which the
branch weights begin to learn, so the network dynamically grows its effective depth instead of
fighting a bad initialization.

## Problem it solves

Training very deep residual-style networks — fully connected stacks, convolutional ResNets, and
Transformers — with fast convergence and no architecture-specific machinery. At depth `L`, a
per-layer perturbation factor `r ≠ 1` scales forward signals and backward gradients by `r^L`,
so they vanish or explode. The strong condition for efficient deep training is *dynamical
isometry* (all singular values of the input-output Jacobian near 1), which for ReLU networks and
self-attention is provably unattainable by any weight-initialization scheme. On the residual
paths that preserve dimension, ReZero attains it exactly by switching every branch off at init.

## Key idea and why each choice

- **A single scalar on the branch.** Minimal: one parameter per block, negligible compute, and
  its only job — controlling whether the block contributes — is data-independent, so it needs no
  weight tensor (Highway's input-dependent gate) and no per-channel parameters (zero-`gamma`).
- **No gate function wrapping it.** `alpha` is used raw, free to take any real value (small,
  order-one, even negative); a clamp (as in scalar-gated ResNets) only removes expressivity,
  and the init-to-identity job is already done by setting the raw value to zero.
- **Initialize to exactly 0, not 1 and not `1/sqrt(d)`.** Zero gives the exact identity map and
  exact dynamical isometry, is identical across every depth and architecture (no need to know
  the block count `d`), and avoids "unlearning" a fully-active random branch. Starting at 1 puts
  the layer on the ill-conditioned ridge of a standard residual block; `1/sqrt(d)` is
  depth-dependent and only near-identity.
- **Skip path stays unmodulated.** Following the pre-activation-ResNet lesson that the cleanest
  gradient flow comes from an unmodulated shortcut, nothing — no activation, norm, or gate —
  sits on the skip; the scalar modulates only the branch. Projection shortcuts remain only the
  usual shape-matching exception.
- **Works with or without normalization.** `alpha` is standalone, unlike zero-`gamma` /
  SkipInit, which presuppose a normalization layer. Batch norm's regularization can be kept
  alongside it if desired; it is not required for convergence.

## Why zero does not freeze learning (the gradients)

For a block `x_{i+1} = x_i + alpha F(x_i)`:

- Gradient to the branch weights: `∂L/∂W = (∂L/∂x_{i+1}) · alpha · ∂F/∂W`, which is 0 at
  `alpha = 0` — the branch is silent on step one.
- Gradient to the scalar: `∂L/∂alpha = (∂L/∂x_{i+1}) · F(x_i)`, which is nonzero (the random
  branch evaluated on the data, with a clean upstream gradient because the rest of the net is
  the identity). So `alpha` moves off zero immediately, and from the next step the branch
  weights receive gradient `∝ alpha`.

## Toy model (why 0 beats 1)

`L` single-neuron layers sharing weight `w` and gate `alpha`, no bias:
`x_{i+1} = (1 + alpha w) x_i`, so `x_L = (1 + alpha w)^L x_0` and `J = (1 + alpha w)^L`.

- `alpha = 1, w ≈ 1`: `J = 2^L` (exploding). The weight update
  `w ← w − lambda L alpha x_0 (1 + alpha w)^{L-1} C'(x_L)` carries `(1+w)^{L-1} ≈ 2^{L-1}`, so a
  stable step needs `lambda ∝ L^{-1}(1+w)^{-(L-1)}` — a learning rate *exponentially small in
  depth*.
- `alpha = 0`: `J = 1` for any `w`. The branch weight gets no first-step gradient (factor
  `alpha = 0`), but `alpha` does: `∂x_L/∂alpha = L w x_0 (1 + alpha w)^{L-1} = L w x_0` at
  `alpha = 0`, free of any exponential factor. Gradient descent converges with a learning rate
  *polynomial in `L`*, the gate stepping off zero into a well-conditioned region rather than
  onto the ill-conditioned ridge at `alpha ≈ 1`.

## Transformer form

Replace LayerNorm and gate each layer with one shared zero-initialized scalar:

```
x_{i+1} = x_i + alpha_i * sublayer(x_i),    sublayer in {self-attention, feed-forward},
```

`alpha_i = 0` at init. At initialization `J = I` exactly despite the LayerNorm/softmax
pathologies, because the sublayers are off; as `alpha_i` grows the model regains full
expressivity from a well-conditioned start, without relying on learning-rate warm-up, LayerNorm,
or auxiliary losses for the initialization problem.

## Practical note

Because `alpha` sits in front of the whole branch, the loss is roughly linear in `alpha` near
init and a too-large step on it destabilizes training. Under an aggressive (one-cycle,
large-peak) schedule, hold the residual weights `alpha` at a small constant learning rate
(around 0.1) while the rest of the network rides the full schedule. With an ordinary step-down
or cosine schedule this is not needed.

## Working code

CIFAR ResNet basic block (one zero-initialized scalar on the branch, unmodulated skip):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """ReZero residual block: out = shortcut(x) + alpha * F(x), alpha init 0."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()                 # identity skip when shapes match
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(              # 1x1 only to match dimensions
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        self.resweight = nn.Parameter(torch.zeros(1))   # alpha, initialized at zero

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))           # F(x): residual branch
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + self.resweight * out   # x_{i+1} = x_i + alpha * F(x)
        return out
```

ReZero Transformer encoder layer (no LayerNorm; one scalar shared across both sublayers):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RZTXEncoderLayer(nn.Module):
    """ReZero Transformer encoder layer: x_{i+1} = x_i + alpha * sublayer(x_i)."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        self.resweight = nn.Parameter(torch.zeros(1))   # alpha, init 0, shared

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)                         # x + alpha * attn(x)
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)                         # x + alpha * ffn(x)
        return src
```
