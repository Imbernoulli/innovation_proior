# NICE: Non-linear Independent Components Estimation

NICE is an exact-likelihood normalizing-flow model with a deterministic encoder `h = f(x)`, a factorial prior on `h`, additive coupling layers, and a final positive diagonal scaling layer.

## Density

For an invertible data-to-latent map `h = f(x)` with equal input and latent dimension,

```text
p_X(x) = p_H(f(x)) |det df(x)/dx|,
log p_X(x) = sum_i log p_{H_i}(f_i(x)) + log |det df(x)/dx|.
```

Sampling is exact:

```text
h ~ p_H,    x = f^{-1}(h).
```

## Coupling Layer

For a partition `(I_1, I_2)`,

```text
y_{I_1} = x_{I_1}
y_{I_2} = g(x_{I_2}; m(x_{I_1})).
```

The Jacobian is block triangular, so

```text
det(dy/dx) = det(dy_{I_2}/dx_{I_2}).
```

The conditioner `m` is never inverted:

```text
x_{I_1} = y_{I_1}
x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1})).
```

Layer cases:

```text
Additive:       y_2 = x_2 + m(x_1),          logdet = 0.
Multiplicative: y_2 = x_2 * b(x_1),          logdet = sum_j log |b_j|.
Affine:         y_2 = x_2 * b_1(x_1) + b_2,  logdet = sum_j log |(b_1)_j|.
```

NICE chooses the additive case for stability with ReLU conditioners. Since additive layers are volume-preserving, it appends

```text
h = exp(s) * u,
logdet = sum_i s_i.
```

The full criterion is therefore

```text
log p_X(x) = sum_i log p_{H_i}(f_i(x)) + sum_i s_i.
```

## Priors

```text
Gaussian: log p(h_i) = -0.5 * (h_i^2 + log(2*pi)).
Logistic: log p(h_i) = -softplus(h_i) - softplus(-h_i).
```

The logistic score is bounded, while the Gaussian score is `-h_i`.

## Reference-Faithful Implementation Sketch

The first-author repository is Theano/Pylearn2. This PyTorch sketch matches its core semantics: split first half / second half, add the conditioner output to the second half, reverse-permute after each coupling layer, invert by undoing the permutation and subtracting, and use a Homothety-style top scaling whose log-determinant is `sum(log_scale)`.

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

Dequantization matching the reference corruptor for `corruption_level = 1`:

```python
def dequantize(x, low=0.0, high=1.0, n_values=256):
    y = (x - low) / (high - low)
    y = (y * (n_values - 1) + torch.rand_like(y)) / n_values
    return y * (high - low) + low
```

For `[0, 1]` 8-bit data this is `(k + u) / 256`; for `[-1, 1]` data the bin width is `1/128`.

## Reference-Code Caveat

The public first-author repo implements the layer/objective above in `pylearn2/models/{mlp.py,nice.py}` and YAML experiment files. The paper text reports Adam with `lr = 1e-3`, `momentum = 0.9`, `beta2 = 0.01`, `lambda = 1`, and `epsilon = 1e-4`; the public YAMLs instead use Pylearn2 `SGD` with a custom `RMSPropMomentum`, learning-rate decay, and a momentum adjustor. Therefore optimizer claims should be attributed separately: the mathematical method is the exact likelihood above; the canonical released code is the Theano/Pylearn2 implementation with `RMSPropMomentum`.
