# Inverse Autoregressive Flow (IAF)

**Problem.** In a VAE the variational bound L = log p(x) − KL(q(z|x) ‖ p(z|x)) is loose by exactly the
KL from the approximate posterior to the true one. Affordable posteriors are diagonal Gaussians, which
cannot represent the correlations of a real posterior. We want a far more flexible q(z|x) that is still
cheap to evaluate, cheap to sample, parallel across dimensions, and — unlike planar/radial flows —
**scales to high-dimensional latent spaces**.

**Key idea.** Build a normalizing flow from the *inverse* of an autoregressive transformation. An
autoregressive model's *sampling* recursion y_i = μ_i(y_{1:i-1}) + σ_i(y_{1:i-1})·ε_i is sequential
(cost ∝ D), but its inverse

  ε_i = (y_i − μ_i(y)) / σ_i(y)

is **fully parallel** (every ε_i depends only on the given y, not on the other ε's) and has a
**triangular Jacobian** with diagonal 1/σ_i, so

  log |det(∂ε/∂y)| = Σ_i −log σ_i(y).

Flexible, parallel, simple log-determinant — exactly the flow step needed for high-dimensional posteriors.

**The flow.** The encoder emits an initial Gaussian's parameters μ_0, σ_0 and a context vector h fed to
every step. Initialize z_0 = μ_0 + σ_0 ⊙ ε (ε ~ N(0,I)), then apply T steps, each a different
autoregressive net of (z_{t-1}, h):

  z_t = μ_t + σ_t ⊙ z_{t-1}.

Each step is autoregressive in z_{t-1}, so its Jacobian is triangular with σ_t on the diagonal. The exact
density of the final iterate is

  log q(z_T | x) = − Σ_i ( ½ ε_i² + ½ log 2π + Σ_{t=0}^{T} log σ_{t,i} ).

**Numerically stable step (the implemented form).** Let the net output two unconstrained vectors m_t, s_t
and use an LSTM-style gated blend with σ_t = sigmoid(s_t):

  σ_t = sigmoid(s_t),   z_t = σ_t ⊙ z_{t-1} + (1 − σ_t) ⊙ m_t,

which is the general step with μ_t = (1 − σ_t) ⊙ m_t, so the density formula is unchanged. The sigmoid
bounds the scale in (0, 1) for stability; initializing s_t with a positive "forget-gate bias" makes σ_t
start near 1, so the flow begins as the identity and learns deviations from a factorized Gaussian.
Reversing the variable order between steps is volume-preserving and mixes dependencies both ways.

**Special case and prior.** A single *linear* IAF step z = L(x)·y with L lower-triangular (the inverse
Cholesky factor) turns a factorized Gaussian into an arbitrary full-covariance Gaussian posterior. On the
generative side, pairing IAF with an **autoregressive prior** p(z_{1:L}) = p(z_L) ∏_{l<L} p(z_l | z_{l+1:L})
makes the true posterior more flexible and thus easier for the IAF posterior to match, tightening the bound
without reducing the generative model's flexibility.

```python
import math
import torch
import torch.nn as nn


class AutoregressiveNN(nn.Module):
    """Masked net: (m, s) per coordinate, output i depending only on z_{1:i-1}; plus a context."""
    def __init__(self, dim, hidden, context_dim):
        super().__init__()
        self.made = MADE(dim, hidden, n_out=2 * dim)        # degree-masked feedforward net
        self.context = nn.Linear(context_dim, 2 * dim)

    def forward(self, z, h):
        m, s = torch.chunk(self.made(z) + self.context(h), 2, dim=1)
        return m, s


class IAFStep(nn.Module):
    def __init__(self, dim, hidden, context_dim, forget_bias=1.5):
        super().__init__()
        self.net = AutoregressiveNN(dim, hidden, context_dim)
        self.forget_bias = forget_bias

    def forward(self, z, h):
        m, s = self.net(z, h)
        sigma = torch.sigmoid(s + self.forget_bias)         # starts near 1 -> identity step
        z = sigma * z + (1.0 - sigma) * m
        log_det = -torch.log(sigma + 1e-8).sum(dim=1)       # -sum_i log sigma_i
        return z, log_det


class IAFPosterior(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, n_steps, hidden):
        super().__init__()
        self.encoder = Encoder(x_dim, z_dim, context_dim)   # -> (mu0, log_sigma0, h)
        self.steps = nn.ModuleList(
            [IAFStep(z_dim, hidden, context_dim) for _ in range(n_steps)])

    def sample_and_log_prob(self, x):
        mu0, log_sigma0, h = self.encoder(x)
        eps = torch.randn_like(mu0)
        z = mu0 + torch.exp(log_sigma0) * eps
        log_q = -(0.5 * eps ** 2 + 0.5 * math.log(2 * math.pi) + log_sigma0).sum(dim=1)
        for i, step in enumerate(self.steps):
            z = z.flip(dims=(1,)) if i % 2 == 1 else z
            z, log_det = step(z, h)
            log_q = log_q + log_det
        return z, log_q


def elbo(decoder, prior, x, z, log_q):
    return (decoder.log_prob(x, z) + prior.log_prob(z) - log_q).mean()

# optimize -elbo(...) with Adam
```

**Setup used in practice.** Improve VAEs by parameterizing each layer's posterior with an IAF. On MNIST,
a convolutional ResNet VAE with a layer of Gaussian stochastic units and a 2-layer MADE per IAF step, ELU
nonlinearities, weight normalization with data-dependent initialization, Adam; expressiveness varied by
chain depth and width. On CIFAR-10, a deep ResNet VAE with many stochastic layers, masked-PixelCNN
autoregressive nets (0/1/2 hidden layers) and an autoregressive prior; measured by the variational lower
bound, an importance-sampled marginal-likelihood estimate, and bits per dimension, with synthesis far
faster than a fully sequential autoregressive image model.
