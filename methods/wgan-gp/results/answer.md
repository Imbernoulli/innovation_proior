# WGAN-GP: Wasserstein GAN with Gradient Penalty

## Problem

Training GANs is unstable. Jensen-Shannon-based GANs lose generator gradients when data and model supports lie on disjoint manifolds, and KL-style objectives are infinite, undefined, or saturated rather than smoothly sensitive to how far those manifolds are apart. Recasting the game as estimating the Wasserstein-1 (Earth-Mover) distance via its Kantorovich-Rubinstein dual fixes that geometry, but it requires the critic to be 1-Lipschitz. Enforcing that with weight clipping underuses the critic's capacity (it collapses to oversimple functions) and makes gradients explode or vanish with depth, demanding fragile per-model tuning of the clip threshold. WGAN-GP replaces the weight clamp with a direct soft constraint on the critic's input-gradient norm.

## Key idea

The Wasserstein-1 / Earth-Mover distance is

  W(Pr, Pg) = inf_{γ in Π(Pr,Pg)} E_{(x,y)~γ}[||x - y||],

where Π(Pr,Pg) is the set of couplings with marginals Pr and Pg. Kantorovich-Rubinstein duality turns it into

  W(Pr, Pg) = sup_{||f||_L <= 1} E_{x~Pr}[f(x)] - E_{x̃~Pg}[f(x̃)].

For differentiable critics on a convex input domain, ||∇D|| ≤ 1 everywhere is the local form of the 1-Lipschitz condition. Rather than clip weights, penalize the critic's input-gradient norm toward 1. Two facts make this precise and non-distorting:

- **The optimal critic has unit-norm gradients on coupling segments.** For the Wasserstein dual optimum f* and an optimal coupling π, call the generated endpoint x̃ and the real endpoint x. Tightness gives f*(x) - f*(x̃) = ||x - x̃||. Along x_t = (1 - t)x̃ + t x, f* rises exactly linearly. The unit direction toward the real endpoint is v = (x - x_t)/||x - x_t|| = (x - x̃)/||x - x̃||, the directional derivative in that direction is 1, and ||∇f*|| ≤ 1 by Lipschitzness. Pythagoras then forces the orthogonal component to vanish, so ∇f*(x_t) = v at differentiability points. A *two-sided* penalty pushing the norm to exactly 1 targets a genuine local property of the solution.
- **Enforce it where it matters.** Since the optimum's gradient is pinned along generated-to-real coupling segments, sample tractable surrogate points on straight real/fake segments: x̂ = ε x + (1 - ε) x̃ with ε ~ U[0,1], x ~ Pr, x̃ ~ Pg.

## Final objective

Critic loss to minimize (combining the Wasserstein critic loss and the gradient penalty):

  L = E_{x̃~Pg}[D(x̃)] − E_{x~Pr}[D(x)] + λ · E_{x̂~P_x̂}[ (||∇_x̂ D(x̂)||₂ − 1)² ],

with x̂ = ε x + (1−ε) x̃, ε ~ U[0,1]. Generator minimizes −E_z[D(G(z))].

## Algorithm

Defaults: λ = 10, n_critic = 5, Adam(α = 1e-4, β₁ = 0, β₂ = 0.9).

```
while not converged:
    for t in 1..n_critic:                       # critic toward optimality
        sample real x ~ Pr, noise z ~ p(z), eps ~ U[0,1]
        x_tilde = stop_gradient(G(z))
        x_hat   = eps*x + (1-eps)*x_tilde
        L = D(x_tilde) - D(x) + lambda*(||grad_{x_hat} D(x_hat)||_2 - 1)^2
        w <- Adam(grad_w mean(L))
    sample noise z ~ p(z)
    theta <- Adam(grad_theta mean(-D(G(z))))     # generator step
```

## Design choices and why

- **Penalize gradient norm, not clip weights.** Weights are the wrong handle on the Lipschitz constraint; the gradient norm is the constraint itself. Removes the capacity-underuse and exploding/vanishing-gradient pathologies of clipping.
- **Target norm 1, two-sided.** The optimal critic has unit-norm gradients on the relevant coupling segments, so pushing toward 1 encodes the desired local shape; a one-sided penalty would leave too-flat slopes uncorrected.
- **Sample on real/fake interpolation lines.** Enforcing ||∇D|| = 1 everywhere is intractable; interpolation targets the region between the two distributions that controls the generator's gradient.
- **λ = 10.** Fixed default that gives slope violations enough weight without replacing the Wasserstein score difference as the critic's main signal.
- **No batch norm in the critic; use layer norm.** The penalty is per-example, but batch norm makes each output depend on the whole batch, so ∇_x̂ D(x̂) is no longer a per-example quantity. Layer norm normalizes per-example and is a clean drop-in.
- **Adam with β₁ = 0.** The adversarial objective is non-stationary; first-moment momentum overshoots a moving target.
- **n_critic = 5.** The generator minimizes a true Wasserstein distance only if the critic is near-optimal; clipping prevented training the critic hard, the penalty does not.
- **Smooth activations when second derivatives matter.** The penalty's parameter-gradient involves second derivatives of the activations; ReLU/leaky-ReLU usually work because sampled points almost never land exactly on kinks, and smooth analogues are safer when an activation's derivative causes trouble.

## Working code (PyTorch)

The load-bearing piece is taking the critic's gradient w.r.t. its interpolated input with `create_graph=True` so the penalty is itself differentiable w.r.t. the critic's parameters.

```python
import torch
import torch.nn as nn
import torch.autograd as autograd


def lipschitz_enforcement(critic, real, fake, lambda_gp=10.0):
    batch_size = real.size(0)
    # one eps per example, broadcast over feature dims, uniform on [0, 1]
    eps = torch.rand(batch_size, *([1] * (real.dim() - 1)), device=real.device)
    x_hat = eps * real + (1 - eps) * fake          # point on real<->fake segment
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)
    grads = autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,      # second-order: penalty differentiable in params
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)               # ||grad_{x_hat} D(x_hat)||_2
    return lambda_gp * ((grad_norm - 1) ** 2).mean()   # two-sided penalty


def train_step(generator, critic, opt_g, opt_c, real, noise_dim,
               n_critic=5, lambda_gp=10.0):
    device = real.device
    for _ in range(n_critic):                      # train critic to near-optimum
        z = torch.randn(real.size(0), noise_dim, device=device)
        fake = generator(z).detach()
        loss_c = critic(fake).mean() - critic(real).mean()   # Wasserstein loss
        loss_c = loss_c + lipschitz_enforcement(critic, real, fake, lambda_gp)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        # no weight clipping: the penalty replaces it

    z = torch.randn(real.size(0), noise_dim, device=device)
    loss_g = -critic(generator(z)).mean()          # generator step
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return loss_c.item(), loss_g.item()


# Critic: scalar output, NO sigmoid, NO batch norm (use nn.LayerNorm / per-example
# normalization). Momentum-free Adam for the non-stationary adversarial objective:
#   opt_c = torch.optim.Adam(critic.parameters(),    lr=1e-4, betas=(0.0, 0.9))
#   opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
```
