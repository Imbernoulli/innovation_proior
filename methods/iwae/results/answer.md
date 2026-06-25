# Importance Weighted Autoencoder (IWAE)

## Problem

A variational autoencoder trains a generative network `p(x,h)` and a recognition network `q(h|x)` by maximizing the evidence lower bound `L(x) = E_{q(h|x)}[log p(x,h)/q(h|x)]`. This bound is an expectation under `q` of a single log-importance-weight, so it penalizes *every* posterior sample that fails to explain `x`. The objective therefore encourages the generative model to learn posteriors that are approximately factorial and feed-forward-predictable -- the only posteriors a simple `q` can match -- which wastes modeling capacity: many latent dimensions become inactive and representations are overly simple.

## Key idea

Replace the single importance weight inside the log with a `k`-sample average. Define the importance weights `w_i = p(x,h_i)/q(h_i|x)` for `h_1,…,h_k ∼ q(h|x)` drawn independently, and the bound

  `L_k(x) = E_{h_1,…,h_k ∼ q}[ log (1/k) Σ_{i=1}^k w_i ]`.

`k = 1` is the standard VAE bound. Because a batch of `k` samples can pool its weight before the log, a few high-weight explanations can dominate the average instead of requiring every draw from `q` to be good. That relaxes the pressure on `q` and frees the model to keep complex posteriors. The architecture is identical to the VAE's; only the training objective changes.

## Properties (the bound ladder)

For all `k`, `log p(x) >= L_{k+1} >= L_k`, and `L_k -> log p(x)` as `k -> infinity` under bounded, positive-weight regularity for `w = p(x,h)/q(h|x)`. The guarantee is non-strict: equality can occur when the random weights leave no Jensen slack (for example, if the weights are constant almost surely).

1. **Lower bound.** `E[(1/k)Σ_i w_i] = p(x)` (each `w_i` is unbiased for `p(x)`), so by Jensen (`log` concave) `L_k = E[log (1/k Σ w_i)] ≤ log E[(1/k)Σ w_i] = log p(x)`.

2. **Monotone in `k`.** For a uniformly random `m`-subset `I ⊂ {1,…,k}` (`m ≤ k`), `E_I[(1/m)Σ_{j} w_{i_j}] = (1/k)Σ_{i} w_i` (each index appears with probability `m/k`). Then `L_k = E[log E_I[(1/m)Σ_j w_{i_j}]] ≥ E[E_I[log (1/m)Σ_j w_{i_j}]] = L_m` (Jensen over `I`; an `m`-subset of i.i.d. draws is again `m` i.i.d. draws). Hence `L_{k+1} ≥ L_k`.

3. **Consistency.** `M_k = (1/k)Σ_i w_i -> E_q[w] = p(x)` a.s. by the SLLN; with the bounded positive-weight assumption, the log of the average converges in expectation to `log p(x)`.

4. **Variance is tame.** Plain importance sampling of `p(x)` can have huge variance, but the estimator is the *log* of the average. For a positive unbiased `Ẑ` of `Z`, Markov on `Ẑ/Z` gives `Pr(log Ẑ > log Z + b) ≤ e^{-b}`, which bounds the mean absolute deviation of `log Ẑ` by `2 + 2δ`, with `δ = log Z − E[log Ẑ]` the bound gap.

## Gradient

Reparameterize `h_i = h(ε_i, x, θ)`, `ε_i ∼ N(0,I)`. Then

  `∇_θ L_k = E_{ε_1,…,ε_k}[ Σ_i w̃_i ∇_θ log w_i ]`,   where   `w̃_i = w_i / Σ_j w_j`

are the **self-normalized** importance weights (derivation: `∇_θ log (1/k Σ_i w_i) = (Σ_i ∇_θ w_i)/(Σ_j w_j) = Σ_i w̃_i ∇_θ log w_i`, using `∇w_i = w_i ∇ log w_i`). For `k = 1`, `w̃_1 = 1` and this is the VAE update. Each `∇_θ log w_i = ∇_θ log p(x,h_i) − ∇_θ log q(h_i|x)`: a reconstruction/autoencoding term plus a spread-out term, averaged with weight proportional to each sample's importance weight — hence "importance weighted autoencoder."

Implementation: maximize the surrogate `Σ_i w̃_i log w_i` with `w̃_i` **detached** (stop-gradient); its value is not `L_k`, but its gradient is exactly `Σ_i w̃_i ∇_θ log w_i`. Compute weights in the log domain (subtract the max log-weight before exponentiating) for stability. The held-out log-likelihood estimate is the actual bound estimate `L_k = logmeanexp_i(log w_i)`. Cost scales linearly in `k`; a stochastic variant (sample one index proportional to `w̃_i`, backprop only it) needs `k` forward passes and one backward pass.

## Code

One stochastic layer, binary (Bernoulli) observations.

```python
import numpy as np
import torch
import torch.nn as nn

LOG2PI = float(np.log(2 * np.pi))


class GaussianBlock(nn.Module):
    """Two tanh layers -> (mu, sigma) of a diagonal Gaussian; exp keeps sigma > 0."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.body(x)
        return self.fc_mu(h), torch.exp(self.fc_logsigma(h))


class IWAE(nn.Module):
    def __init__(self, dim_latent, dim_obs):
        super().__init__()
        self.encoder = GaussianBlock(dim_obs, 200, dim_latent)
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, dim_obs), nn.Sigmoid())

    def encode(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps, mu, sigma, eps     # h = mu + sigma * eps

    def log_weights(self, x):
        # x: (k, batch, dim_obs)
        h, mu, sigma, eps = self.encode(x)
        log_q = torch.sum(-0.5 * eps ** 2 - torch.log(sigma) - 0.5 * LOG2PI, -1)   # log q(h|x)
        log_prior = torch.sum(-0.5 * h ** 2 - 0.5 * LOG2PI, -1)                    # log p(h)
        p = self.decoder(h)
        log_lik = torch.sum(x * torch.log(p) + (1 - x) * torch.log(1 - p), -1)     # log p(x|h)
        return log_prior + log_lik - log_q                                         # log w_i, (k, batch)

    def objective(self, x):
        # Training surrogate, not the numerical value of L_k.
        log_w = self.log_weights(x)
        log_w_stable = log_w - torch.max(log_w, 0, keepdim=True)[0]
        w = torch.exp(log_w_stable)
        w_tilde = (w / torch.sum(w, 0, keepdim=True)).detach()    # self-normalized, stop-gradient
        return torch.mean(torch.sum(w_tilde * log_w, 0))          # grad == grad L_k

    def log_likelihood_estimate(self, x):
        log_w = self.log_weights(x)
        m = torch.max(log_w, 0, keepdim=True)[0]
        return torch.mean(m.squeeze(0) + torch.log(torch.mean(torch.exp(log_w - m), 0)))  # logmeanexp


def train_step(model, x, optimizer, k):
    x = x.unsqueeze(0).expand(k, *x.shape)         # k samples per example
    optimizer.zero_grad()
    loss = -model.objective(x)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Setting `k = 1` recovers the standard VAE gradient. Training uses Adam; for evaluation, `log_likelihood_estimate` with a large `k` (e.g. 5000) gives the stochastic lower-bound estimate used for held-out log-likelihood.
