# PEARL context encoder (MLP encoder), distilled

PEARL — Probabilistic Embeddings for Actor-Critic RL — is an off-policy meta-RL algorithm that
disentangles **task inference** from **control**. Its task-inference component is a probabilistic
context encoder: a stateless per-transition MLP `f_φ` that maps each transition `(s, a, r, s')`
to a Gaussian factor over a latent task variable `z`, with the per-transition factors aggregated
into the task posterior `q_φ(z | c)` by a **product of Gaussians**. The control component is a
soft actor-critic conditioned on a sampled `z`. The MLP encoder described here is the original
PEARL encoder.

## Problem it solves

Adapt quickly to an unseen task drawn from a task distribution `p(T)` (tasks differ in reward or
dynamics), using only collected transitions as evidence, while keeping meta-training
sample-efficient and supporting uncertainty-aware exploration (crucial for sparse rewards).

## Key idea

1. **Disentangle inference from control** so each can be trained on the data distribution it
   needs. Only the encoder must obey the meta-learning matching principle (its train/test inputs
   should agree), so it is fed *recently collected* (near-on-policy) context, while the actor and
   critic are trained on the whole off-policy replay buffer — making meta-training cheap.

2. **Probabilistic latent task `z`** inferred by amortized variational inference, optimizing a
   downstream objective with an information-bottleneck KL penalty:
   ```
   max_φ  E_T  E_{z ~ q_φ(z | c^T)} [ R(T, z) - β · D_KL( q_φ(z | c^T) || p(z) ) ],   p(z) = N(0, I).
   ```
   `β·KL(q‖p)` is a variational upper bound on `I(z; c)`; it compresses `z` to task-relevant
   information and mitigates overfitting to training tasks. A probabilistic `z` enables
   **posterior sampling** for temporally extended exploration.

3. **Permutation-invariant, uncertainty-fusing aggregation.** The context `c = {(s_n,a_n,r_n,s'_n)}`
   is a variable-sized, unordered set (each transition is self-contained Markov evidence about the
   task). The encoder emits one Gaussian factor `Ψ_φ(z|c_n) = N(μ_n, σ_n²)` per transition, and
   the posterior is their product:
   ```
   q_φ(z | c_{1:N}) ∝ ∏_{n=1}^N Ψ_φ(z | c_n) = N(μ, σ²),
   ```
   which is permutation-invariant (product commutes), handles any `N`, and is closed-form.

## Product-of-Gaussians closed form

Per latent dimension (diagonal factors), matching the exponent of `∏_n exp(−(z−μ_n)²/(2σ_n²))`
to a single Gaussian:

```
1/σ² = Σ_n 1/σ_n²            (precisions add — belief sharpens as transitions accumulate)
σ²   = 1 / Σ_n (1/σ_n²)
μ    = σ² · Σ_n (μ_n / σ_n²) (precision-weighted mean — confident transitions dominate)
```

More transitions strictly increase the combined precision, so `σ²` shrinks: Bayesian evidence
accumulation. With zero transitions the agent resets the belief to the unit-Gaussian prior; with
observed context, oyster's product is over the encoder's transition factors. Variances are
floored (`clamp(σ_n², min=1e-7)`) for numerical safety.

## Encoder architecture

- Input: one transition's features concatenated, `[s, a, r (, s')]` of dimension `input_size`.
- 3 hidden layers of 200 ReLU units; oyster's **fan-in init** on hidden weights
  (`uniform(−1/√size(0), +1/√size(0))` for 2-D tensors), bias `0.1`.
- Output linear layer to `output_size = 2 · latent_dim`, **small init** `uniform(−3e-3, +3e-3)`
  so initial factor outputs are small rather than prematurely committed.
- Output split into `[μ | pre-softplus σ²]`; `σ² = softplus(·)` (smooth, positive, ≈0.69 at 0).
- **Stateless**: applied independently per transition; `reset()` is a no-op (no recurrent state).

## Training

Built on SAC. The encoder is trained from the **critic's Bellman gradient** (found to outperform
maximizing actor returns or reconstructing states/rewards — `z` need only make the value
predictable) plus the KL term:

```
L_critic = E_{(s,a,r,s',d)~B, z~q_φ(z|c)} [ ( Q_θ(s,a,z) − ( r_scaled + (1-d) γ V̄(s', z̄) ) )² ],
L_KL     = β · D_KL( q_φ(z|c) || N(0,I) ).
```

`z̄` denotes a stop-gradient on the bootstrap target; `z` is detached into the actor and value so
the encoder receives gradients only through the critic. The reparameterization trick
(`z = μ + √σ² · ξ`, `ξ ~ N(0,I)`) backpropagates through the sample. The context sampler draws
from recently collected data; the RL batch is drawn from the full buffer.

At meta-test: with no data the posterior is the prior; sample `z`, act for an episode
(temporally extended exploration), multiply the new transitions' factors into the posterior
(it tightens), resample. Behavior anneals from exploratory to exploitative as the belief narrows.

## Working code

The per-transition MLP encoder and the product-of-Gaussians aggregation the agent uses:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = int(np.prod(size[1:]))
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MlpEncoder(nn.Module):
    """Original PEARL MLP context encoder: 3 x 200 ReLU, per-transition,
    output_size = 2 * latent_dim -> [ mu | pre-softplus sigma^2 ]."""

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size           # = 2 * latent_dim
        self.hidden_activation = hidden_activation

        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:               # [200, 200, 200]
            fc = nn.Linear(in_dim, h_dim)
            fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim
        self.last_fc = nn.Linear(in_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        output = preactivation
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass                                     # stateless: nothing to reset


def product_of_gaussians(mus, sigmas_squared):
    """Aggregate per-transition Gaussian factors -> posterior N(mu, sigma^2).
    Precisions add; mean is precision-weighted; permutation-invariant."""
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def infer_posterior(encoder, context, latent_dim):
    """context: (num_tasks, num_transitions, input_size). Returns a
    reparameterized sample z plus batched posterior parameters."""
    params = encoder(context)
    params = params.view(context.size(0), -1, encoder.output_size)
    mu = params[..., :latent_dim]
    sigma_squared = F.softplus(params[..., latent_dim:])       # positive variance
    z_params = [
        product_of_gaussians(m, s)
        for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
    ]
    z_means = torch.stack([p[0] for p in z_params])
    z_vars = torch.stack([p[1] for p in z_params])
    posteriors = [
        torch.distributions.Normal(m, torch.sqrt(s))
        for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))
    ]
    z = torch.stack([d.rsample() for d in posteriors])
    return z, z_means, z_vars
```

## Relation to prior methods

- **RL² / recurrent meta-RL** aggregates context with an order-dependent RNN and is on-policy;
  the MLP encoder + product-of-Gaussians is permutation-invariant by construction, handles
  arbitrary context size, and trains off-policy.
- **Prototypical networks / Deep Sets** give the permutation-invariant "encode-each-then-combine"
  template; PEARL replaces the deterministic mean with a *product of Gaussian factors* so the
  aggregation fuses uncertainty (precisions add).
- **SAC** is the off-policy backbone; PEARL conditions actor and critic on `z` and trains the
  encoder from the critic gradient.
- **MAESN** also uses probabilistic task variables but adapts them by gradient descent and
  explores from the prior; PEARL updates the posterior online from collected evidence and
  explores by posterior sampling from it.
