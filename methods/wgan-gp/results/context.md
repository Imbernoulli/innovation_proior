# Context

## Research question

Generative adversarial networks can produce strikingly realistic samples, but training them is unreliable: the game between generator and discriminator frequently fails to converge, collapses onto a few modes, or stalls because the generator stops receiving usable gradients. The most promising stabilization route reframes the game so the discriminator estimates the Wasserstein-1 (Earth-Mover) distance between the data distribution and the generator distribution — but that route requires the discriminator to be a 1-Lipschitz function, and the only known way to enforce that constraint at the time is crude. The question is: **how can the 1-Lipschitz constraint on the critic be enforced in a way that does not cripple the critic's capacity or destabilize its gradients, so that a wide range of architectures train reliably with little or no per-model tuning?**

## Background

**The GAN game and why it is unstable.** A generator G maps noise z ~ p(z) to samples; a discriminator D scores inputs as real or fake. The original objective is
  min_G max_D  E_{x~Pr}[log D(x)] + E_{x̃~Pg}[log(1 - D(x̃))],
where Pr is the data distribution and Pg the model distribution implicitly defined by x̃ = G(z). When D is trained to optimality, the inner problem makes the generator minimize the Jensen-Shannon divergence between Pr and Pg. Goodfellow et al. (2014) introduced this and noted the saturation problem; in practice the generator is trained to maximize E[log D(x̃)] (the non-saturating "-log D" trick) to avoid vanishing gradients early on.

**The manifold mismatch diagnostic.** Arjovsky & Bottou (2017, "Towards principled methods for training generative adversarial networks") analyzed *why* the saturation happens. Pr and Pg are typically supported on low-dimensional manifolds in a high-dimensional ambient space, and these supports generically do not overlap (or intersect only on a measure-zero set). Where the supports are disjoint, a sufficiently powerful discriminator can separate them perfectly; JS divergence is then locally constant (it maxes out at log 2), so its gradient with respect to the generator's parameters is zero almost everywhere. KL-style divergences are no rescue: with mutually singular supports, one direction is infinite or undefined and the classifier-based training signal still does not vary smoothly with the distance between the manifolds. The better the discriminator, the less the generator learns — an observed pathology, not a hypothetical one.

**Earth-Mover distance as the fix.** The Wasserstein-1 / Earth-Mover distance,
  W(Pr, Pg) = inf_{γ ∈ Π(Pr, Pg)}  E_{(x,y)~γ}[ ||x - y|| ],
where Π(Pr, Pg) is the set of joint distributions (couplings) with marginals Pr and Pg, measures the minimum cost (mass × distance) of transporting one distribution onto the other. Unlike JS, W varies continuously and is differentiable almost everywhere as the generator moves, *even when the supports are disjoint*, because it registers how far apart the masses are rather than merely whether they overlap. Under mild assumptions W(Pr, Pg) is continuous everywhere and differentiable almost everywhere in the generator's parameters.

**Kantorovich-Rubinstein duality.** The primal infimum over couplings is intractable. Its dual (Villani, *Optimal Transport: Old and New*, 2008) is
  W(Pr, Pg) = sup_{||f||_L ≤ 1}  E_{x~Pr}[f(x)] - E_{x~Pg}[f(x)],
a supremum over all 1-Lipschitz functions f. A function f is 1-Lipschitz iff |f(a) - f(b)| ≤ ||a - b|| for all a, b; for differentiable functions on a convex input domain, ||∇f|| ≤ 1 everywhere is the local form of the same constraint. The dual turns an intractable transport problem into a maximization over a function class that a neural network can parameterize.

**A property of the dual optimum.** For the dual objective max_f E_{y~Pr}[f(y)] - E_{x~Pg}[f(x)], on a compact metric space an optimal 1-Lipschitz f* exists, and for the optimal coupling π it must hold that f*(y) - f*(x) = ||y - x|| for π-almost-every coupled generated-real pair (x, y). The Lipschitz bound is *tight* on the pairs the optimal transport plan actually moves, constraining not just the magnitude of f* but the *direction* of its gradient along the segments connecting coupled points.

**Lipschitz enforcement by weight clipping, and its diagnosed failures.** To keep a network 1-Lipschitz (up to a scale), the earliest approach clamps every weight into a small box [-c, c] after each gradient step. The resulting function lies in *some* k-Lipschitz class, with k depending on c and the architecture. Two failure modes are observed with this scheme on existing systems:

- *Capacity underuse.* Because the dual optimum must use the full allowed slope on the transport segments, a network trying to attain its maximal Lipschitz constant under a weight box is pushed to drive weights to the box extremes; the weight histogram becomes bimodal at ±c. Critics trained to optimality on simple two-dimensional distributions (eight Gaussians arranged in a ring, twenty-five Gaussians on a grid, a swiss roll), with the generator held fixed at the real data plus unit-variance Gaussian noise, learn pathologically simple value surfaces that ignore higher moments of the data.
- *Exploding and vanishing gradients.* The interaction between the hard weight box and the loss makes the gradient of the critic loss with respect to successive layers either grow or shrink exponentially with depth. On the swiss roll with a deep ReLU MLP critic and no batch normalization, clipping thresholds c ∈ {10⁻¹, 10⁻², 10⁻³} each produce gradients that explode or vanish across layers; c sits on a knife-edge. Batch normalization in the critic mitigates this somewhat, but very deep clipped critics still fail to converge.

**Layer normalization.** Ba, Kiros & Hinton (2016) introduced layer normalization, which normalizes the activations of each example over its own features (independent of other examples in the batch), as opposed to batch normalization which normalizes each feature over the batch and therefore couples examples together.

## Baselines

- **Original GAN (Goodfellow et al. 2014).** The minimax log-loss game above; discriminator outputs a probability through a sigmoid. Core idea: adversarial training drives Pg toward Pr via (implicitly) the Jensen-Shannon divergence. Gap: when supports are disjoint the JS gradient vanishes, and related KL-style objectives are infinite, undefined, or insensitive to the actual distance between singular supports, so training is unstable and prone to mode collapse; the value function does not correlate with sample quality.

- **DCGAN (Radford, Metz & Chintala 2015).** A set of architectural guidelines (strided/fractional-strided convolutions, batch normalization in both networks, ReLU/LeakyReLU) that made the original GAN objective trainable for images. Core idea: stable convolutional architecture. Gap: stability is bought through careful, fragile architecture choices specific to the log-loss objective; deviating from the recipe (removing normalization, changing nonlinearities, going very deep) breaks training.

- **WGAN with weight clipping (Arjovsky, Chintala & Bottou 2017).** Replaces the log-loss with the Kantorovich-Rubinstein dual: the discriminator becomes a *critic* (no sigmoid, not a classifier) trained to maximize E_{x~Pr}[D(x)] - E_{x̃~Pg}[D(x̃)], and the generator minimizes the same value, which approximates W(Pr, Pg). The 1-Lipschitz constraint is enforced by clipping critic weights into [-c, c]. Core idea: a value function with better-behaved gradients and a loss that correlates with sample quality. Gap: weight clipping is a poor proxy for the Lipschitz constraint — it causes the capacity underuse and exploding/vanishing-gradient pathologies described above, and requires batch normalization plus careful tuning of c, yet very deep critics still fail.

- **Least-Squares GAN (Mao et al. 2016).** Replaces the log-loss with a least-squares loss on the discriminator outputs, aiming to provide non-vanishing gradients to generated samples far from the decision boundary. Core idea: a different divergence that mitigates saturation. Gap: still does not give the stability across architectures that a Lipschitz-constrained Wasserstein critic targets.

## Evaluation settings

The natural yardsticks:
- *Toy 2D distributions* for diagnosing critic behavior: eight Gaussians in a ring, twenty-five Gaussians on a grid, and a swiss roll, where the value surface and gradient flow of a critic can be inspected directly (optionally with the generator held fixed at real data plus Gaussian noise).
- *Image datasets*: CIFAR-10 (32×32 natural images) and LSUN bedrooms; downsampled 32×32 ImageNet for architecture-robustness sweeps.
- *Metrics*: the Inception score for image sample quality; direct inspection of the critic's value surface and of per-layer gradient norms versus depth for the diagnostics; the critic's loss value tracked over training as a candidate convergence/overfitting signal.
- *Protocol*: alternate several critic updates per generator update; standard convolutional generator/critic backbones (DCGAN-style and deep residual networks); Adam or RMSProp optimizers.

## Code framework

The existing scaffold is a standard Wasserstein critic loop: a generator maps noise to samples; a critic returns one scalar score per sample with no sigmoid; the critic optimizer minimizes fake score minus real score plus one open Lipschitz-enforcement term; the generator optimizer minimizes negative critic score on generated samples. The open slot is the scalar term that enforces the 1-Lipschitz requirement inside the critic objective.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # standard generator backbone (e.g. DCGAN-style); maps noise -> sample
        ...
    def forward(self, z):
        ...

class Critic(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # standard scalar critic backbone, no sigmoid
        ...
    def forward(self, x):
        ...

def critic_loss(critic, real, fake):
    # Wasserstein critic loss (to be minimized): pushes scores up on real,
    # down on fake. The Lipschitz term is supplied by the open slot below.
    return critic(fake).mean() - critic(real).mean()

def lipschitz_enforcement(critic, real, fake):
    # The open slot: return a scalar Lipschitz-enforcement term.
    # TODO
    pass

def train_step(generator, critic, opt_g, opt_c, real_batch, noise_dim, n_critic=5):
    for _ in range(n_critic):
        z = torch.randn(real_batch.size(0), noise_dim, device=real_batch.device)
        fake = generator(z).detach()
        loss_c = critic_loss(critic, real_batch, fake)
        loss_c = loss_c + lipschitz_enforcement(critic, real_batch, fake)
        opt_c.zero_grad(); loss_c.backward(); opt_c.step()
    z = torch.randn(real_batch.size(0), noise_dim, device=real_batch.device)
    loss_g = -critic(generator(z)).mean()
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
```
