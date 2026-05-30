# Gumbel-Softmax

## Problem

Train stochastic neural networks with **categorical latent variables** by gradient descent. With a
categorical node `z ~ Categorical(π)`, `π = π(θ)`, and cost `f(z)`, we minimize
`L(θ) = E_z[f(z)]` and need `∇_θ L`. But sampling a categorical is `one_hot(argmax(...))`, which is
non-differentiable, so backprop cannot reach `θ`. The reparameterization trick that powers Gaussian
VAEs (`z = μ + σ·ε`) has no discrete analogue, and the unbiased score-function/REINFORCE estimator
`E_z[f(z)∇_θ log p_θ(z)]` is too high-variance.

## Key idea

Sample the categorical via the **Gumbel-Max** trick, then **relax the argmax to a softmax** with a
temperature `τ`, yielding a reparameterized, differentiable sample on the simplex.

Gumbel-Max (exact): with `g_i ~ Gumbel(0,1)` i.i.d. (`g_i = -log(-log u_i)`, `u_i ~ U(0,1)`),

```
z = one_hot( argmax_i [ log π_i + g_i ] )  ~  Categorical(π).
```

This is exact: writing `x_i = log π_i + g_i` (Gumbel with location `log π_i`, so
`P(x_i ≤ t) = exp(-π_i e^{-t})`),

```
P(argmax = k) = ∫ π_k e^{-t} exp(-(Σ_i π_i)e^{-t}) dt = π_k / Σ_i π_i.
```

All randomness is in the parameter-free `g_i`; `θ` enters only through the smooth `log π_i`. The
sole non-differentiable operation is the `argmax`. Replace it by a temperature-`τ` softmax:

```
y_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ),   i = 1, ..., k.
```

`y ∈ Δ^{k-1}` is smooth in `π` for `τ > 0`, so `∂y/∂π` exists and backprop flows. This is the
Gumbel-Softmax / Concrete distribution, with closed-form density

```
p_{π,τ}(y) = Γ(k) τ^{k-1} (Σ_i π_i / y_i^τ)^{-k} ∏_i (π_i / y_i^{τ+1}).
```

## Temperature and the bias-variance tradeoff

- `τ → 0`: `y → one_hot(argmax)`, exact categorical samples (low bias), but gradient variance
  can become high because the Jacobian `(1/τ)(diag(y)-yyᵀ)` is spiky near decision boundaries.
- `τ` large: `y → uniform` (1/k), smooth low-variance gradients, but biased (ignores `π` in the
  limit).
- Practice: **anneal** `τ` from high to a small positive floor, e.g. `τ = max(0.5, exp(-r·t))`.
  A learned `τ` acts as an adaptive entropy regularizer.

## Straight-Through Gumbel-Softmax

When a hard discrete value is required (discrete actions, quantization): use `one_hot(argmax y)` in
the forward pass and the soft Jacobian in the backward pass, `∇_θ z ≈ ∇_θ y`. Implemented as
`z = y_hard - y_soft.detach() + y_soft` (value is one-hot; gradient is the soft sample's). Lower
variance than Bernoulli Straight-Through because `y` is a sample-dependent differentiable proxy of
`z`, avoiding the forward/backward mismatch.

## Code

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Differentiable (approximate) categorical sample.

    logits[..., k]: unnormalized log-probabilities (log pi up to a constant).
    Returns a sample on the simplex; if hard=True, a one-hot vector with the
    soft sample's gradient (straight-through).
    """
    # g_i ~ Gumbel(0,1): -log(-log u) == -log(e), e ~ Exp(1).
    g = -torch.empty_like(logits).exponential_().log()
    scores = (logits + g) / tau                 # (log pi_i + g_i) / tau
    y_soft = scores.softmax(dim)                # relaxed sample on Δ^{k-1}

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class CategoricalLatentVAE(nn.Module):
    def __init__(self, x_dim, n_vars, n_classes, hidden=256):
        super().__init__()
        self.n_vars, self.n_classes = n_vars, n_classes
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_vars * n_classes))
        self.decoder = nn.Sequential(
            nn.Linear(n_vars * n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim))

    def forward(self, x, tau, hard=False):
        logits = self.encoder(x).view(-1, self.n_vars, self.n_classes)
        z = gumbel_softmax(logits, tau=tau, hard=hard)
        x_logits = self.decoder(z.view(x.size(0), -1))
        return x_logits, logits


def elbo(x, x_logits, logits):
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="none").sum(-1)
    q = F.softmax(logits, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    k = logits.size(-1)
    kl = (q * (log_q + math.log(k))).sum(-1).sum(-1)  # KL(q || uniform)
    return (recon + kl).mean()


def train(model, optimizer, loader, n_steps, r=1e-4, tau_min=0.5):
    step = 0
    for x in loader:
        tau = max(tau_min, math.exp(-r * step))  # anneal temperature
        optimizer.zero_grad()
        x_logits, logits = model(x, tau=tau)
        loss = elbo(x, x_logits, logits)
        loss.backward()      # gradient flows through gumbel_softmax into the encoder
        optimizer.step()
        step += 1
        if step >= n_steps:
            break
```

The gradient flows from the loss, through the decoder, through the differentiable
`gumbel_softmax` sample, into the encoder logits — with no stochastic node blocking it. Set
`hard=True` when a true one-hot is required downstream.
