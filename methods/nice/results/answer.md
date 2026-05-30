# NICE: Non-linear Independent Components Estimation

**Problem.** Learn the density of complex, high-dimensional continuous data (images) with an
exact tractable log-likelihood, exact and direct latent inference, easy unbiased sampling, and a
meaningful latent representation — all at once. The families that give an exact likelihood are
either intractable to train (partition functions) or sequential to sample (autoregressive); the
families that sample easily give up either the exact likelihood (GAN) or the exact, deterministic
encoder (VAE).

**Key idea.** Model the data as a deterministic, invertible transformation `h = f(x)` of a fixed
*factorial* prior. The change-of-variables formula gives the exact likelihood,

  log p_X(x) = log p_H(f(x)) + log |det(∂f(x)/∂x)|,

and with a factorial prior the first term is a sum Σ_d log p_{H_d}(f_d(x)). The log-determinant
term penalizes contraction and rewards volume expansion at the data points, preventing the
trivial likelihood inflation a contracting preprocessing would otherwise allow. Inference is
h = f(x); sampling is h ~ p_H, x = f^{-1}(h). The only obstruction is the O(D³) Jacobian
determinant, which is removed by designing f so its Jacobian is **triangular**.

**Coupling layer.** Partition the coordinates into (I_1, I_2). Freeze the first block; transform
the second by an invertible coupling law g conditioned on an arbitrary function m of the first:

  y_{I_1} = x_{I_1},   y_{I_2} = g(x_{I_2} ; m(x_{I_1})).

The Jacobian is block-lower-triangular with an identity top-left block, so its determinant is
det(∂y_{I_2}/∂x_{I_2}) and the conditioner m never appears in it (m can be a deep ReLU MLP). The
inverse is x_{I_1} = y_{I_1}, x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1})), and m is never inverted.

**Additive coupling.** Take g(a; b) = a + b:

  y_{I_2} = x_{I_2} + m(x_{I_1}),   inverse   x_{I_2} = y_{I_2} − m(y_{I_1}).

The bottom-right Jacobian block is the identity, so det = 1 (zero log-det) and the inverse is a
subtraction. Chosen over multiplicative/affine laws for numerical stability: with ReLU m the
transform is piecewise linear. Compose layers, alternating which block is frozen; **at least three
layers** are needed for every dimension to influence every other (four are used in practice).

**Diagonal scaling.** Additive couplings are volume-preserving (det ≡ 1), which cannot shape a
density. Restore volume change with a single diagonal top layer, parametrized exponentially:

  h = exp(s) ⊙ h^{(top)},   contributing   Σ_i s_i   to the log-likelihood.

The full criterion is

  log p_X(x) = Σ_i log p_{H_i}(f_i(x)) + Σ_i s_i.

The prior term pushes the scales down; the log-det term Σ_i s_i forbids any scale collapsing to
zero. The equilibrium scales σ_d = S_dd^{-1} act as a nonlinear PCA spectrum, exposing which
latent dimensions carry variation (a learned manifold).

**Prior.** Factorial, from a standard family. Logistic per coordinate,
log p_{H_d}(h) = −softplus(h) − softplus(−h), is preferred because its score lies in (−1, 1) —
a better-behaved gradient than the Gaussian's unbounded −h. Gaussian,
log p_{H_d}(h) = −½(h² + log 2π), is also used.

**Training.** Dequantize discrete pixels (add uniform noise of one quantization bin, rescale into
a bounded box) so the continuous likelihood is well posed; maximize the exact log-likelihood with
Adam (lr 1e-3). Comparable bits-per-dimension adds D·log(256) to the negative log-likelihood and
divides by D·log 2.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Coupling(nn.Module):
    """Additive coupling: freeze one parity, shift the other by an MLP of it. det = 1."""
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        super().__init__()
        self.mask_config = mask_config
        self.in_block = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
            for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, reverse=False):
        B, W = x.size()
        x = x.reshape(B, W // 2, 2)
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]
        out = self.in_block(off)
        for block in self.mid_block:
            out = block(out)
        shift = self.out_block(out)
        on = on - shift if reverse else on + shift
        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape(B, W)


class Scaling(nn.Module):
    """Diagonal volume change: h = exp(s) * x; log|det| = sum(s)."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x, reverse=False):
        log_det_J = torch.sum(self.scale)
        x = x * torch.exp(-self.scale) if reverse else x * torch.exp(self.scale)
        return x, log_det_J


class NICE(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config):
        super().__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.coupling = nn.ModuleList([
            Coupling(in_out_dim, mid_dim, hidden, (mask_config + i) % 2)
            for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def f(self, x):                       # data -> latent
        for layer in self.coupling:
            x = layer(x)
        return self.scaling(x)            # (z, log_det_J)

    def g(self, z):                       # latent -> data (exact inverse)
        x, _ = self.scaling(z, reverse=True)
        for layer in reversed(self.coupling):
            x = layer(x, reverse=True)
        return x

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        return torch.sum(self.prior.log_prob(z), dim=1) + log_det_J

    def sample(self, n):
        z = self.prior.sample((n, self.in_out_dim))
        return self.g(z)

    def forward(self, x):
        return self.log_prob(x)


class StandardLogistic(torch.distributions.Distribution):
    def log_prob(self, z):
        return -(F.softplus(z) + F.softplus(-z))

    def sample(self, size):
        u = torch.rand(size)
        return torch.log(u) - torch.log1p(-u)


# training
flow = NICE(prior=StandardLogistic(), coupling=4, in_out_dim=784,
            mid_dim=1000, hidden=5, mask_config=1)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, betas=(0.9, 0.01), eps=1e-4)
for x in dataloader:                      # x dequantized into a continuous box
    loss = -flow.log_prob(x).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**Architecture used in practice.** Four additive coupling layers on an odd/even partition, each
conditioner a deep ReLU MLP with linear outputs (five hidden layers of 1000 for MNIST, four of
5000 for TFD, four of 2000 for SVHN and CIFAR-10), an exponentiated diagonal scaling on top, a
logistic prior (Gaussian for TFD), and linear preprocessing per dataset (none / approximate
whitening / exact ZCA). The same change-of-variables identity, applied to the augmented pair
(x, ε), reproduces the variational autoencoder as a two-affine-coupling NICE model with a Gaussian
prior, with the SGVB/ELBO objective falling out of the joint likelihood of (x, ε).
