# Masked Autoregressive Flow (MAF)

**Problem.** Build a neural density estimator that is both flexible and tractable, and that returns
the *exact* density p(x) of any externally provided datapoint in a single fast GPU pass — the regime
that matters for likelihood-free inference, learned priors, and importance proposals. A single masked
autoregressive model with simple conditionals is too rigid and order-sensitive; flows built for
variational inference (IAF) can only score their own samples cheaply.

**Key idea.** An autoregressive model with Gaussian conditionals, viewed as a generator, is a
differentiable transform x = f(u) of internal random numbers u ~ N(0, I) — and it is invertible with a
triangular Jacobian, so it *is* a normalizing flow. Improve a single such model by **stacking**: model
the random numbers of each layer with the next layer, gaining multimodality from composition while
keeping each layer a cheap invertible affine recursion.

**One layer.** With per-coordinate conditional N(x_i; μ_i, (exp α_i)²), μ_i = f_{μ_i}(x_{1:i-1}),
α_i = f_{α_i}(x_{1:i-1}):

  generate (u → x):   x_i = u_i·exp(α_i) + μ_i,
  invert  (x → u):    u_i = (x_i − μ_i)·exp(−α_i)    (one masked pass, all i at once).

The Jacobian of x → u is triangular with diagonal ∂u_i/∂x_i = exp(−α_i), so

  log |det(∂f^{-1}/∂x)| = −Σ_i α_i,

and the exact density follows from the change-of-variables formula. The conditioners {f_{μ_i}, f_{α_i}}
are a Gaussian **MADE**: a masked feedforward net that emits all (μ_i, α_i) in one pass while respecting
the autoregressive property (degree-based binary masks, M[j,k] = 1{deg_next[j] ≥ deg_prev[k]}).

**Stacking.** Compose K such layers M_1, …, M_K; M_2 models the random numbers of M_1, and so on, with
the final random numbers declared standard normal. The log-det of the stack is the sum of layer log-dets.
Use the dataset order for the first layer and **reverse** the order each successive layer. Insert an
invertible **batch-normalization** layer between layers to keep the deep stack trainable:

  x = (u − β) ⊙ exp(−γ) ⊙ (v + ε)^{1/2} + m,   |det(∂f^{-1}/∂x)| = exp( Σ_i [γ_i − ½ log(v_i + ε)] ),

with γ exponentiated to force a positive scale and simplify the log-det.

**Relationships.**
- *IAF* uses the identical recursion but with conditioners reading the random numbers u_{1:i-1} instead
  of the data x_{1:i-1}. This flips the cost: MAF scores any x in one pass but samples in D sequential
  passes; IAF samples in one pass but scores external data in D passes. MAF is for density estimation;
  IAF is for variational inference (scoring its own samples). Training a MAF by maximum likelihood
  (minimizing KL(π_x ‖ p_x)) equals minimizing KL(p_u ‖ π_u) — exactly the variational objective of an
  implicit IAF with base density π_x and transform f^{-1}.
- *Real NVP* coupling layer = a MAF (or IAF) layer restricted so that a fixed prefix is copied
  (μ_i = α_i = 0 for i ≤ d) and the rest is transformed as a function of only that prefix. MAF/IAF are
  strictly more flexible (every coordinate scaled/shifted by *all* previous ones); Real NVP's advantage is
  one-pass *both* sampling and scoring.

**Conditional MAF.** For p(x | y), augment every MADE's inputs with y (dropping no connection from y);
y becomes an extra input to every layer.

```python
import math
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__(n_in, n_out, bias)
        self.mask = None
    def set_mask(self, mask):
        self.mask = mask
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """Gaussian MADE: one pass -> (mu, alpha) per coordinate, autoregressive by construction."""
    def __init__(self, n_in, hidden_dims, random_order=False, seed=None):
        super().__init__()
        np.random.seed(seed)
        self.n_in, self.n_out, self.hidden_dims = n_in, 2 * n_in, hidden_dims
        dims = [n_in, *hidden_dims, self.n_out]
        layers = []
        for i in range(len(dims) - 2):
            layers += [MaskedLinear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [MaskedLinear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self._make_masks(random_order)

    def _make_masks(self, random_order):
        L, D = len(self.hidden_dims), self.n_in
        deg = {0: permutation(D) if random_order else np.arange(D)}
        for l in range(L):
            deg[l + 1] = randint(low=deg[l].min(), high=D - 1, size=self.hidden_dims[l])
        deg[L + 1] = deg[0]
        masks = []
        for i in range(len(deg) - 1):
            M = (deg[i + 1][:, None] >= deg[i][None, :]).astype(int)
            masks.append(torch.tensor(M, dtype=torch.float32))
        masks[-1] = torch.cat((masks[-1], masks[-1]), dim=0)
        it = iter(masks)
        for m in self.net.modules():
            if isinstance(m, MaskedLinear):
                m.set_mask(next(it))

    def forward(self, x):
        return self.net(x.float())


class MAFLayer(nn.Module):
    def __init__(self, dim, hidden_dims, reverse):
        super().__init__()
        self.dim, self.made, self.reverse = dim, MADE(dim, hidden_dims), reverse

    def forward(self, x):                                 # x -> u (density, one pass)
        mu, logp = torch.chunk(self.made(x), 2, dim=1)    # 0.5*logp = -alpha
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)            # -sum_i alpha_i
        return u, log_det

    def backward(self, u):                                # u -> x (sampling, D sequential steps)
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for i in range(self.dim):
            mu, logp = torch.chunk(self.made(x), 2, dim=1)
            x[:, i] = mu[:, i] + u[:, i] * torch.exp(torch.clamp(-0.5 * logp[:, i], max=10))
        return x


class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        m, v = x.mean(0), x.var(0)
        u = (x - m) / torch.sqrt(v + self.eps) * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v + self.eps))
        return u, log_det


class MAF(nn.Module):
    def __init__(self, dim, n_layers, hidden_dims, use_reverse=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def log_prob(self, x):
        u, log_det = self.forward(x)
        base = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return base + log_det


# training
maf = MAF(dim=784, n_layers=5, hidden_dims=[1024])
optimizer = torch.optim.Adam(maf.parameters(), lr=1e-4, weight_decay=1e-6)
for x in dataloader:
    loss = -maf.log_prob(x).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**Setup used in practice.** MADE/MADE-MoG and each MAF layer are masked feedforward nets (ReLU; tanh for
GAS), 1–2 hidden layers chosen by validation; MAF with 5 or 10 layers and a standard-Gaussian base (MAF MoG
puts a 10-component MADE-MoG as the base). Adam, minibatch 100, step 10⁻⁴ for the deep flows (10⁻³ for a
single MADE), ℓ₂ coefficient 10⁻⁶, early stopping after 30 epochs without validation improvement. Images
are dequantized, rescaled to [0, 1], and modeled in logit space.
