# Mish — the method, distilled

**Problem.** Smooth, self-gated, non-monotonic units (`x·σ(βx)`, `x·Φ(x)`) are strong ReLU alternatives
on deep nets. Find a single parameter-free pointwise drop-in that keeps the family's good properties
(smooth everywhere, small preserved negative region, unbounded above, bounded below) and has a
mechanistic account of its gradient behavior.

**Key idea.** Hold the good properties fixed and search only the gate `h` in `f(x)=x·h(x)`. Softplus is
a smooth positive ReLU-like inner map; `tanh` turns it into a bounded gate with the right endpoints,
yielding

`f(x) = x·tanh(softplus(x)) = x·tanh(ln(1+eˣ))`  (**Mish**).

**Why it works.** Properties: `C∞`-smooth (no kink), `f→x` as `x→+∞` (identity, unbounded above, no
positive saturation), `f→0` as `x→−∞`, with a shallow non-monotonic negative bump (min ≈ −0.31, range
`[≈−0.31,∞)`) that preserves a little negative signal and removes the dying-ReLU precondition. The
mechanism is in the first derivative: with `s=softplus(x)`, `s′=σ(x)`,

`f′(x) = tanh(s) + x·sech²(s)·σ(x).`

For `x≠0`, this can be rewritten exactly as

`f′(x) = Δ(x)·swish(x) + f(x)/x,  Δ(x)=sech²(softplus(x)),  swish(x)=x·σ(x).`

At `x=0`, `f(x)/x` is supplied by the continuous extension `tanh(softplus(0))=3/5`, so the rewritten
form matches the direct derivative everywhere. The `Δ(x)` term is a smooth input-dependent multiplier
on the Swish-shaped component of the derivative, while `f(x)/x` supplies the gate term. This supports a
preconditioning analogy: the activation derivative gives a smooth local rescaling of gradient flow, not
a hard ReLU kink or a fixed negative-side leak.

**Hyperparameters.** None — parameter-free. Drop in for ReLU/Swish with everything else fixed. Slightly
costlier than Swish (softplus+tanh vs. one sigmoid), the price of the smoother gradient.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish: x * tanh(softplus(x))."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
```
