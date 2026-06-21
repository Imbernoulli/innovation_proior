I want an unsupervised density model for high-dimensional continuous data such as images, and I want it to do several things at once that usually come from separate, incompatible model families: train by exact maximum likelihood, perform exact data-to-code inference, sample cheaply by ancestral draws, and keep a latent representation of the same dimensionality as the data. Each existing route gives me some of this and forfeits the rest. Undirected models like RBMs and DBMs are flexible but carry an intractable partition function, so training and sampling lean on MCMC and likelihood evaluation reduces to approximations such as annealed importance sampling. Variational autoencoders sample quickly and learn a recognition model, but the encoder is stochastic and approximate and the objective is only a lower bound on the likelihood, not the likelihood. Autoregressive estimators give an exact tractable likelihood through a triangular dependency structure, but sampling is sequential in the dimension count and the coordinate ordering is baked into the model. Adversarial generators sample in parallel from simple noise but supply neither a tractable likelihood nor an exact encoder. The goal — exact likelihood, exact inference, and easy sampling in a single deterministic object — is not met by any of these.

The only way I see to get all three simultaneously is to make the encoder itself a change of variables. I propose NICE, Non-linear Independent Components Estimation. Let $h = f(x)$ be a smooth bijection between spaces of equal dimension and place a simple factorial prior $p_H(h) = \prod_d p_{H_d}(h_d)$ on the latent. Then inference is exactly $f(x)$, sampling is to draw $h \sim p_H$ and apply $f^{-1}$, and the density is forced by conservation of probability mass,
$$p_X(x) = p_H(f(x))\,\bigl|\det \tfrac{\partial f(x)}{\partial x}\bigr|, \qquad \log p_X(x) = \sum_i \log p_{H_i}(f_i(x)) + \log\bigl|\det \tfrac{\partial f(x)}{\partial x}\bigr|.$$
The determinant term is not bookkeeping; it has the right sign in the data-to-latent direction. If $f$ contracts a data neighborhood, $|\det \partial f/\partial x|$ is small and the likelihood pays a negative log-volume price; if it expands around high-density data the term is positive. So the model cannot win by merely shrinking the data before scoring it under the prior. This is also exactly the representation objective I wanted: twist the data into coordinates that look independent under a simple prior, where with a factorial prior the first term is just a coordinate-wise sum $\sum_i \log p_{H_i}(f_i(x))$.

The obstruction is that for a general neural network in $D$ dimensions the Jacobian determinant costs $O(D^3)$ and is numerically fragile. I need a map as expressive as a neural network whose determinant is readable without forming a dense matrix. Triangular Jacobians solve the determinant — their determinant is the product of diagonal entries — but I do not want to force every weight matrix to be triangular, because that is far too restrictive. The trick is to get a triangular Jacobian structurally. I split the coordinates into two blocks $I_1$ and $I_2$, leave the first block untouched, and transform the second using a function of the first,
$$y_{I_1} = x_{I_1}, \qquad y_{I_2} = g\bigl(x_{I_2}; m(x_{I_1})\bigr).$$
Here $g$ need only be invertible in its first argument once the second is fixed, and the conditioner $m$ can be an arbitrarily deep neural network because I never invert it. The Jacobian has block form with the identity in the top-left, $\partial y_{I_2}/\partial x_{I_1}$ in the off-diagonal, and $\partial y_{I_2}/\partial x_{I_2}$ in the bottom-right, so $\det(\partial y/\partial x) = \det(\partial y_{I_2}/\partial x_{I_2})$ and the entire derivative of $m$ drops out of the determinant. The inverse is equally direct: $x_{I_1} = y_{I_1}$ and $x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1}))$. That is the crucial design shape — arbitrary capacity in the conditioner, a triangular Jacobian, and no inversion of the conditioner.

For the coupling law I choose the additive case as the most stable option,
$$y_{I_2} = x_{I_2} + m(x_{I_1}), \qquad x_{I_2} = y_{I_2} - m(y_{I_1}).$$
Its bottom-right block is the identity, so the layer determinant is exactly $1$ and its log-determinant is $0$. Multiplicative and affine variants live in the same framework — $y_{I_2} = x_{I_2}\odot b$ gives $\log|\det| = \sum_j \log|b_j|$, and $y_{I_2} = x_{I_2}\odot b_1 + b_2$ gives $\sum_j \log|(b_1)_j|$, each requiring nonzero scale entries — but learned multiplicative scales are numerically awkward, whereas with a rectified conditioner the additive layer stays piecewise linear and stable, so I take additive. One additive layer copies half the coordinates through; a second with the roles exchanged updates the other half. Checking the dependency graph, influence cannot pass fully across both blocks in both directions until at least three alternating layers, and I use four as a conservative practical choice. Each layer has log-determinant zero, so the whole additive stack is volume-preserving.

That volume preservation is the next problem, because density modeling needs local contraction and expansion. Rather than sacrifice the stable additive layers, I put all the volume change into a separate top layer, a diagonal positive scaling applied to the hidden state $u$ after the additive stack,
$$h_i = \exp(s_i)\, u_i,$$
invertible by multiplying by $\exp(-s_i)$ and with log-determinant exactly $\sum_i s_i$. The full criterion for the additive stack plus top scaling is therefore
$$\log p_X(x) = \sum_i \log p_{H_i}(f_i(x)) + \sum_i s_i.$$
The sign matters: in the encoder direction the diagonal map multiplies volume by $\prod_i \exp(s_i)$, so the log-determinant enters as $+\sum_i s_i$, and the two terms oppose each other — the prior pulls codes toward high-density regions while the scaling term prevents all scales from collapsing to zero. The inverse scales $\sigma_i = \exp(-s_i)$ read as a nonlinear analogue of a component spectrum: large $\sigma_i$ means the data varies along that latent coordinate, small $\sigma_i$ means the model has suppressed it.

For the prior I only need independent coordinates from a standard family. A standard Gaussian gives $\log p(h_i) = -\tfrac12(h_i^2 + \log 2\pi)$ with score $-h_i$, which grows without bound; a standard logistic gives $\log p(h_i) = -\mathrm{softplus}(h_i) - \mathrm{softplus}(-h_i)$ whose score $\mathrm{sigmoid}(-h_i) - \mathrm{sigmoid}(h_i)$ is bounded in $[-1,1]$ and is therefore easier to optimize. The construction recovers familiar special cases as a sanity check: an affine map $z = Lx + b$ with lower-triangular $L$ and a Gaussian prior has log-determinant $\sum_i \log|L_{ii}|$, so whitening appears as the linear special case of the same change-of-variables likelihood. There is also a consistency tie to the VAE — reparameterizing a recognition model $z = g_\phi(\epsilon; x)$ with $\epsilon \sim N(0,I)$, defining the standardized residual $\xi = (x - f_\theta(z))/\sigma$, and applying change of variables on the pair recovers the sampled SGVB objective once the auxiliary noise density is subtracted, the residual scaling contributing the $-D_X\log\sigma$ term and the recognition Jacobian becoming $-\log q_\phi(z\mid x)$ — confirming the same machinery while NICE's own inference stays deterministic and exact.

Finally, because a continuous density would otherwise exploit discrete pixels by placing arbitrarily narrow spikes on the grid, I dequantize: a normalized 8-bit value $k/255$ becomes $(k+u)/256$ with $u \sim \mathrm{Uniform}(0,1)$, and for data stored in $[-1,1]$ the bin width is $2/256 = 1/128$. When reporting bits per dimension I add the $D\log 256$ discrete-level constant to the negative log-likelihood and divide by $D\log 2$, including any constant log-determinant from fixed linear preprocessing. The full recipe is then: dequantize, optionally apply fixed linear preprocessing, split the coordinates into two equal halves, stack additive coupling layers that alternate which half is updated, append the exponentiated diagonal scale, and train by exact maximum likelihood; encoding is the forward transform with its accumulated log-determinant, and sampling draws from the factorial prior and applies the inverse layers in reverse order. The hard nonlinear functions appear only as conditioners, so the determinant and inverse stay trivial while the transformation remains highly nonlinear.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardLogistic:
    def log_prob(self, z):
        return -(F.softplus(z) + F.softplus(-z))

    def sample(self, shape, device=None):
        u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
        return torch.log(u) - torch.log1p(-u)


class StandardNormal:
    def log_prob(self, z):
        return -0.5 * (z.square() + torch.log(torch.tensor(2.0 * torch.pi, device=z.device)))

    def sample(self, shape, device=None):
        return torch.randn(shape, device=device)


class AdditiveCoupling(nn.Module):
    def __init__(self, dim, conditioner, split=None, reverse_permute=True):
        super().__init__()
        self.dim = dim
        self.split = dim // 2 if split is None else split
        self.conditioner = conditioner
        perm = torch.arange(dim - 1, -1, -1) if reverse_permute else torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x, inverse=False):
        if inverse:
            y = x[:, self.inv_perm]
            y1, y2 = y[:, :self.split], y[:, self.split:]
            x2 = y2 - self.conditioner(y1)
            return torch.cat([y1, x2], dim=1)

        x1, x2 = x[:, :self.split], x[:, self.split:]
        y2 = x2 + self.conditioner(x1)
        y = torch.cat([x1, y2], dim=1)
        return y[:, self.perm]

    def forward_with_logdet(self, x):
        return self.forward(x), x.new_zeros(())


class TileReordering(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        perm = torch.cat([torch.arange(0, dim, 2), torch.arange(1, dim, 2)])
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x, inverse=False):
        return x[:, self.inv_perm if inverse else self.perm]


class DiagonalScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x, inverse=False):
        sign = -1.0 if inverse else 1.0
        y = x * torch.exp(sign * self.log_scale)
        logdet = sign * self.log_scale.sum()
        return y, logdet


class NICE(nn.Module):
    def __init__(self, dim, conditioners, prior, tile_reorder=False):
        super().__init__()
        self.dim = dim
        self.prior = prior
        self.reorder = TileReordering(dim) if tile_reorder else None
        self.layers = nn.ModuleList([
            AdditiveCoupling(dim, conditioner) for conditioner in conditioners
        ])
        self.scale = DiagonalScale(dim)

    def f(self, x):
        if self.reorder is not None:
            x = self.reorder(x)
        logdet = x.new_zeros(())
        for layer in self.layers:
            x, layer_logdet = layer.forward_with_logdet(x)
            logdet = logdet + layer_logdet
        z, scale_logdet = self.scale(x)
        return z, logdet + scale_logdet

    def g(self, z):
        x, _ = self.scale(z, inverse=True)
        for layer in reversed(self.layers):
            x = layer(x, inverse=True)
        if self.reorder is not None:
            x = self.reorder(x, inverse=True)
        return x

    def log_prob(self, x):
        z, logdet = self.f(x)
        return self.prior.log_prob(z).sum(dim=1) + logdet

    def sample(self, n, device=None):
        z = self.prior.sample((n, self.dim), device=device)
        return self.g(z)
```

```python
def dequantize(x, low=0.0, high=1.0, n_values=256):
    y = (x - low) / (high - low)
    y = (y * (n_values - 1) + torch.rand_like(y)) / n_values
    return y * (high - low) + low
```
