# Variational Inference and the ELBO

Given observed data `x`, latent variables `z`, and a tractable joint density `p(x,z)`, approximate the posterior by choosing a tractable family `Q` and solving

```text
q* = argmin_{q in Q} KL(q(z) || p(z | x)).
```

This converts posterior inference into optimization over a distribution we control.

## ELBO/KL Decomposition

Expand the KL:

```text
KL(q || p(z | x))
  = E_q[log q(z)] - E_q[log p(z | x)]
  = E_q[log q(z)] - E_q[log p(x,z)] + log p(x).
```

The unknown `log p(x)` is constant in `q`, so minimizing the KL is equivalent to maximizing

```text
ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]
        = E_q[log p(x,z)] + H(q).
```

Rearranging gives the evidence decomposition

```text
log p(x) = ELBO(q) + KL(q || p(z | x)).
```

Because `KL >= 0`, the ELBO is a lower bound on the log evidence. The gap is exactly the posterior-approximation error in `KL(q || p(z | x))`.

## Mean-Field CAVI

Use a fully factorized family

```text
q(z) = product_{j=1}^m q_j(z_j).
```

Holding all factors except `q_j` fixed, the coordinate optimum is

```text
q_j*(z_j) proportional to exp(E_-j[log p(x,z)])
                 proportional to exp(E_-j[log p(z_j | z_-j, x)]).
```

Cycling these updates monotonically raises the ELBO to a local optimum. This is coordinate-ascent variational inference:

```text
initialize q_1, ..., q_m
repeat
    for j = 1, ..., m:
        q_j(z_j) <- normalized exp(E_-j[log p(x,z)])
    compute ELBO(q)
until ELBO improvement is small
```

## Exponential-Family Closed Form

If the complete conditional is

```text
p(z_j | z_-j, x) = h(z_j) exp(eta_j(z_-j,x)^T z_j - a(eta_j)),
```

then the optimal factor is in the same family:

```text
q_j*(z_j) = h(z_j) exp(nu_j^T z_j - a(nu_j)),
nu_j = E_-j[eta_j(z_-j,x)].
```

Thus the update becomes expected-natural-parameter bookkeeping.

## EM Relation

For models with parameters `theta`,

```text
L(q, theta) = E_q[log p(x,z | theta)] + H(q)
```

lower-bounds `log p(x | theta)` with gap `KL(q || p(z | x, theta))`. If `q` ranges over all distributions, the best `q` is the exact posterior and coordinate ascent in `(q, theta)` is EM. If the exact posterior is unavailable, restricting `q` gives a tractable approximate E step while still optimizing the same lower-bound form.

## Gaussian-Mixture Instance

For the model

```text
mu_k ~ Normal(0, sigma2)
c_i  ~ Categorical(1/K)
x_i | c_i, mu ~ Normal(mu_{c_i}, 1)
```

with

```text
q(mu, c) = product_k Normal(mu_k; m_k, s2_k) product_i Categorical(c_i; phi_i),
```

the CAVI updates are

```text
phi_ik proportional to exp(m_k x_i - (m_k^2 + s2_k) / 2)
s2_k = 1 / (1/sigma2 + sum_i phi_ik)
m_k  = s2_k sum_i phi_ik x_i.
```

The runnable implementation is in `methods/variational-inference-elbo/code/cavi_gaussian_mixture.py`. It computes the ELBO, checks monotone ascent, and keeps the best of several random initializations because the objective is non-convex.

## Generic ELBO Training Step

The same objective can be optimized directly with black-box gradient estimators. The snippet below defines `elbo(x, encoder, decoder, prior)` and a single training step.

```python
import torch
import torch.nn as nn


def elbo(x, encoder, decoder, prior):
    """ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))."""
    q = encoder(x)
    z = q.rsample()                      # reparameterized sample
    log_likelihood = decoder(z).log_prob(x).sum()
    kl_divergence = torch.distributions.kl_divergence(q, prior).sum()
    return log_likelihood - kl_divergence


def training_step(x, encoder, decoder, prior, optimizer):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    loss = -elbo(x, encoder, decoder, prior).mean()
    loss.backward()
    optimizer.step()
    return float(loss)


# Example skeleton (not run)
encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 2 * 10))
decoder = nn.Sequential(nn.Linear(10, 256), nn.ReLU(),
                        nn.Linear(256, 784), nn.Sigmoid())
prior = torch.distributions.Normal(0, 1)
optimizer = torch.optim.Adam(list(encoder.parameters()) +
                             list(decoder.parameters()), lr=1e-3)
```
