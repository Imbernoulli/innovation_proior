# Glow: Generative Flow with Invertible 1×1 Convolutions

## Problem

Model the density of high-dimensional natural images so that the **exact** log-likelihood is computable, the **exact** latent of any image is recoverable, and both training and sampling run in parallel — and make it good enough to synthesize realistic high-resolution images. Autoregressive models sample serially; VAEs only bound the likelihood and infer latents approximately; adversarial models have no encoder and no likelihood. A normalizing flow — a bijection `x = g(z)`, `z = f(x)` between data and a simple latent — delivers all of these in principle; Glow makes it competitive on likelihood and scalable to high-resolution synthesis.

## Key idea

Define the model by change of variables and compose many simple invertible layers:

```
log p(x) = log p(z) + Σ_i log|det(dh_i / dh_{i-1})|,   z = f(x).
```

Each **step of flow** is three sublayers in sequence:

1. **Actnorm** — per-channel scale `s` and bias `b`, **data-dependently initialized** (zero mean, unit variance on the first minibatch) and then batch-independent. Replaces batch normalization, which fails at the batch-size-1 per-PU regime forced by high-resolution images. Log-det `= h·w·Σ log|s|`.
2. **Invertible 1×1 convolution** — apply one learned invertible `c×c` matrix `W` to the channel vector at every spatial position. This generalizes the fixed channel permutation of prior flows (a permutation is just `W = P`) to a *learned* channel mixing. Log-det `= h·w·log|det W|`. With the **LU parameterization** `W = P·L·(U + diag(s))` (P fixed, L unit-diagonal lower, U zero-diagonal upper), `log|det W| = Σ log|s|`, reducing the cost from `O(c³)` to `O(c)`.
3. **Affine coupling** — split on channels, `xa, xb = split(x)`; `(log s, t) = NN(xb)`; `ya = s ⊙ xa + t`, `yb = xb`. Invertible without inverting NN; log-det `= Σ log|s|`. Channel split only — the 1×1 conv already mixes channels, so RealNVP's checkerboard masking is dropped.

Steps are stacked `K` per level over `L` levels, with **squeeze** (`h×w×c → h/2×w/2×4c`) and **multi-scale** factoring-out of half the channels (as Gaussian latents) at each level.

## Final algorithm / objective

Dequantize `x̃ = x + u`, `u ~ U(0, 1/n_bins)`. Run `f` accumulating each sublayer's log-determinant; score the latents under Gaussian priors. Train by minimizing, in bits per dimension,

```
−log p(x̃) = −[ log p(z) + logdet − log(n_bins)·D ],
bits/dim   = −log p(x̃) / (log 2 · D).
```

The three components, their inverses, and their log-determinants (tensor shape `h×w×c`):

| Component | Forward | Reverse | Log-det |
|---|---|---|---|
| Actnorm | `y = s ⊙ x + b` | `x = (y − b)/s` | `h·w·Σ log|s|` |
| Invertible 1×1 conv | `y_{i,j} = W x_{i,j}` | `x_{i,j} = W⁻¹ y_{i,j}` | `h·w·log|det W|` = `h·w·Σ log|s|` (LU) |
| Affine coupling | `ya = s⊙xa + t, yb = xb` | `xa = (ya − t)/s, xb = yb` | `Σ log|s|` |

Optimizer: Adam, `α = 0.001`. Last conv of each coupling net is zero-initialized so each step starts as identity. Sampling can use temperature `T`: `z ~ N(0, T²I)`, giving `p_{θ,T}(x) ∝ p_θ(x)^{T²}`.

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def squeeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * factor * factor, h // factor, w // factor)


def unsqueeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c // factor ** 2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // factor ** 2, h * factor, w * factor)


class ActNorm(nn.Module):
    """Per-channel affine, data-dependent init, then batch-independent."""
    def __init__(self, channels):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def _init(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.b.data = -mean / (std + 1e-6)
            self.log_s.data = -torch.log(std + 1e-6)
            self.initialized.fill_(1)

    def forward(self, x, logdet):
        if self.initialized.item() == 0:
            self._init(x)
        h, w = x.shape[2], x.shape[3]
        y = torch.exp(self.log_s) * x + self.b
        return y, logdet + h * w * self.log_s.sum()

    def reverse(self, y):
        return (y - self.b) * torch.exp(-self.log_s)


class InvConv1x1(nn.Module):
    """Learned invertible channel mixing; LU-parameterized log|det W| = sum(log|s|)."""
    def __init__(self, channels):
        super().__init__()
        w0 = torch.linalg.qr(torch.randn(channels, channels))[0]
        P, L, U = torch.linalg.lu(w0)
        self.register_buffer("P", P)
        s = torch.diag(U)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.register_buffer("l_mask", torch.tril(torch.ones_like(L), -1))
        self.register_buffer("eye", torch.eye(channels))

    def _w(self):
        L = self.L * self.l_mask + self.eye
        U = self.U * self.l_mask.t() + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x, logdet):
        h, w = x.shape[2], x.shape[3]
        W = self._w()
        y = F.conv2d(x, W.view(*W.shape, 1, 1))
        return y, logdet + h * w * self.log_s.sum()

    def reverse(self, y):
        W_inv = torch.inverse(self._w())
        return F.conv2d(y, W_inv.view(*W_inv.shape, 1, 1))


class CouplingNN(nn.Module):
    def __init__(self, in_ch, out_ch, width=512):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 1)
        self.c3 = nn.Conv2d(width, out_ch, 3, padding=1)
        nn.init.zeros_(self.c3.weight); nn.init.zeros_(self.c3.bias)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        return self.c3(x)


class AffineCoupling(nn.Module):
    def __init__(self, channels, width=512):
        super().__init__()
        self.net = CouplingNN(channels // 2, channels, width)

    def forward(self, x, logdet):
        xa, xb = x.chunk(2, dim=1)
        log_s, t = self.net(xb).chunk(2, dim=1)
        s = torch.sigmoid(log_s + 2.)
        ya = s * xa + t
        return torch.cat([ya, xb], dim=1), logdet + torch.log(s).flatten(1).sum(1)

    def reverse(self, y):
        ya, yb = y.chunk(2, dim=1)
        log_s, t = self.net(yb).chunk(2, dim=1)
        s = torch.sigmoid(log_s + 2.)
        xa = (ya - t) / s
        return torch.cat([xa, yb], dim=1)


class FlowStep(nn.Module):
    def __init__(self, channels, width=512):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConv1x1(channels)
        self.coupling = AffineCoupling(channels, width)

    def forward(self, x, logdet):
        x, logdet = self.actnorm(x, logdet)
        x, logdet = self.invconv(x, logdet)
        x, logdet = self.coupling(x, logdet)
        return x, logdet

    def reverse(self, y):
        y = self.coupling.reverse(y)
        y = self.invconv.reverse(y)
        return self.actnorm.reverse(y)


def gaussian_logp(z, mean, log_sd):
    return -0.5 * (math.log(2 * math.pi) + 2 * log_sd
                   + (z - mean) ** 2 / torch.exp(2 * log_sd))


class Glow(nn.Module):
    def __init__(self, in_ch=3, K=32, L=3, width=512):
        super().__init__()
        self.L = L
        self.levels = nn.ModuleList()
        self.priors = nn.ModuleList()
        c = in_ch * 4
        for i in range(L):
            self.levels.append(nn.ModuleList(
                [FlowStep(c, width) for _ in range(K)]))
            if i < L - 1:
                self.priors.append(nn.Conv2d(c // 2, c, 3, padding=1))
                c = c * 2
            else:
                self.top_prior = nn.Conv2d(c, 2 * c, 3, padding=1)

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        log_p = torch.zeros_like(logdet)
        z = squeeze2d(x)
        for i in range(self.L):
            for step in self.levels[i]:
                z, logdet = step(z, logdet)
            if i < self.L - 1:
                z1, z2 = z.chunk(2, dim=1)
                mean, log_sd = self.priors[i](z1).chunk(2, dim=1)
                log_p = log_p + gaussian_logp(z2, mean, log_sd).flatten(1).sum(1)
                z = squeeze2d(z1)
        mean, log_sd = self.top_prior(torch.zeros_like(z)).chunk(2, dim=1)
        log_p = log_p + gaussian_logp(z, mean, log_sd).flatten(1).sum(1)
        return z, logdet, log_p


def loss_bits_per_dim(logdet, log_p, n_bins, n_pixels):
    ll = log_p + logdet - math.log(n_bins) * n_pixels
    return (-ll / (math.log(2) * n_pixels)).mean()
```
