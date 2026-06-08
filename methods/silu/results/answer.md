# SiLU / Swish — the method, distilled

**Problem.** Hand-designed ReLU replacements give *inconsistent* gains across models/datasets, so ReLU
stays the default. Rather than hand-design another curve from a property wishlist, automatically search
the space of *scalar* activations (one scalar in, one out — a true drop-in for ReLU), select by a
validation signal, then read off what shape that signal prefers.

**Key idea.** Define a structured search space: build a scalar activation by composing core units
`b(u₁,u₂)`, where two unary-transformed scalar sources are combined by one binary primitive. The unary
library includes `x, x², σ, tanh, eˣ, max(x,0), βx, …`; the binary library includes `+, ·, −, max,
x₁·x₂, …`. Enumerate small spaces; for large ones use an RNN controller trained with PPO
(reward = validation accuracy of a cheap child network — ResNet-20/CIFAR-10), parallelized over workers.
The selected self-gated activation is

`f(x) = x · σ(βx)`  (**Swish**; `β` constant or per-channel trainable),

a self-gated unit. `β=1` recovers the earlier SiLU/SiL `x·σ(x)`.

**Why it works.** Search trends favor simple functions (1–2 core units), a clean raw-input path
`b(x, g(x))` (ReLU = `max(x,0)` fits it), and avoidance of unstable division. ReLU can also be written as
`x·1(x>0)`; replacing the hard gate with a sigmoid gives a one-core-unit realization with `u₁=x`,
`u₂=σ(βx)`, and `b(a,b)=a·b`. For positive `β`, `β→∞` implies `σ(βx)→1(x>0)` and Swish → ReLU;
`β→0` implies `σ(βx)→1/2` and Swish → `x/2`. It is unbounded above, bounded below, smooth, and
non-monotonic because the negative input side is multiplied by a positive gate that shrinks gradually
toward zero. With `s=σ(βx)` and `f=xs`, the derivative is
`f′(x)=s+βxs(1−s)=βf+σ(βx)(1−βf)`, so the unit keeps a non-saturating positive tail without requiring
ReLU's exact unit positive slope.

**Hyperparameters / practical notes.** `β=1` (Swish-1 / SiLU) is the simple default; trainable
per-channel `β` can be initialized at 1.0. The forward pass is `x*sigmoid(beta*x)`. If BatchNorm precedes
it, keep the BN learnable *scale* on because this activation is not ReLU-style positive homogeneous.
Treat the learning rate as a parameter to re-tune rather than blindly reusing the ReLU setting.

```python
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish / SiLU: x * sigmoid(beta * x)."""

    def __init__(self, num_channels=None, trainable_beta=False, beta_init=1.0):
        super().__init__()
        if trainable_beta:
            shape = (num_channels,) if num_channels is not None else (1,)
            self.beta = nn.Parameter(torch.full(shape, float(beta_init)))
            self.trainable = True
        else:
            self.register_buffer("beta", torch.tensor(float(beta_init)))
            self.trainable = False

    def forward(self, x):
        if self.trainable and self.beta.numel() > 1:
            shape = [1] * x.dim()
            shape[1] = -1
            b = self.beta.view(*shape)
        else:
            b = self.beta
        return x * torch.sigmoid(b * x)


class SiLU(nn.Module):
    """SiLU == Swish-1: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)
```
