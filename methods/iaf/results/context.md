# Research question

We want stochastic variational inference — and the variational autoencoders built on it — to work
with *much more flexible* approximate posteriors than they currently use, while staying cheap and,
crucially, scaling to **high-dimensional latent spaces**. In a VAE we fit an inference model q(z|x)
to approximate the true posterior p(z|x) by maximizing a variational lower bound on log p(x). That
bound is tight exactly when q(z|x) matches p(z|x); but the posteriors that are computationally
affordable today are diagonal Gaussians, q(z|x) = N(μ(x), σ²(x)), and true posteriors are essentially
never factorized. So the bound is loose by exactly the amount that the posterior fails to factorize.

The constraint that has forced everyone onto diagonal Gaussians is threefold: the inference model
must (1) be cheap to evaluate and differentiate its density, and (2) be cheap to *sample* — both are
done for every datapoint at every optimization step — and (3) for high-dimensional z on a GPU, those
operations must be *parallelizable across dimensions*. The question is whether there is a family of
posteriors that is far more expressive than a diagonal Gaussian yet still satisfies all three, and —
where the existing flexible families break down — keeps working as the latent space grows to thousands
of spatially-organized dimensions.

# Background

**The variational lower bound.** For a generative model p(x, z) with latents z, the marginal
log-likelihood is intractable, so we introduce an inference model q(z|x) and optimize

  log p(x) ≥ E_{q(z|x)}[ log p(x, z) − log q(z|x) ] = L(x; θ),

and equivalently L(x; θ) = log p(x) − KL(q(z|x) ‖ p(z|x)). Maximizing L over the parameters of both p
and q simultaneously raises log p(x) and tightens the gap KL(q ‖ p(z|x)); since that KL is a function
of *both* models, increasing the flexibility of either generally helps. For continuous z the bound is
optimized efficiently by reparameterizing q (Kingma & Welling 2013; Rezende et al. 2014) so that
gradients pass through the sampling.

**Normalizing flows for the posterior.** A normalizing flow (Rezende & Mohamed 2015) builds a flexible
distribution by starting from a simple one, z_0 ~ q(z_0|x), and applying a chain of invertible maps
z_t = f_t(z_{t-1}, x) for t = 1, …, T. As long as each transformation's Jacobian determinant is
computable, the density of the last iterate is

  log q(z_T|x) = log q(z_0|x) − Σ_{t=1}^{T} log |det(∂z_t/∂z_{t-1})|.

Rezende & Mohamed used a planar flow f_t(z) = z + u·h(wᵀz + b) (with vectors u, w, scalar b,
nonlinearity h), which routes all information through a single-unit bottleneck wᵀz + b — so capturing
high-dimensional dependencies requires a very long chain. In practice these planar/radial flows are
effective only for low-dimensional latent spaces (at most a few hundred dimensions); how to scale a
flow to the latent space of a generative model of large images was open.

**Gaussian autoregressive functions.** A separate, very successful line uses autoregressive neural
networks for density estimation — MADE (Germain et al. 2015), PixelCNN/PixelRNN (van den Oord et al.
2016), WaveNet, RNN estimators. For a variable y = {y_i} with a chosen order, such a network outputs a
mean and standard deviation [μ(y), σ(y)] with the autoregressive structure ∂[μ_i, σ_i]/∂y_j = [0, 0]
for j ≥ i — each element's parameters depend only on the earlier elements. Sampling from such a model
is the sequential recursion

  y_0 = μ_0 + σ_0·ε_0,   y_i = μ_i(y_{1:i-1}) + σ_i(y_{1:i-1})·ε_i,   ε ~ N(0, I),

whose cost is proportional to the dimensionality D and is inherently sequential, so these models are
not directly attractive as posteriors to sample from. But the same structure makes the *inverse* map
ε_i = (y_i − μ_i)/σ_i interesting, which is the lever the method will pull.

**Building blocks available.** Masked autoregressive networks (MADE) and masked/causal convolutions
(PixelCNN, WaveNet) give cheap, parallel autoregressive functions. ResNets (He et al. 2015, 2016)
make deep image models trainable. Weight normalization with data-dependent initialization (Salimans &
Kingma 2016) stabilizes such training. The LSTM "forget-gate bias" trick (Jozefowicz et al. 2015) —
biasing a gate so it starts near 1 — is a known way to make a gated update begin as near-identity. Adam
is the default optimizer; the importance-weighted bound (Burda et al. 2015) gives a tighter likelihood
estimator for evaluation.

**Deep stochastic-layer optimization.** Deep VAEs with many latent groups can settle into a
zero-information state q(z|x) ≈ p(z), especially early in training when the likelihood signal is weak.
Then the KL terms are small, the stochastic layers are unused, and encoder gradients have poor
signal-to-noise for escaping. KL annealing in sentence VAEs and ladder-style VAEs is one known way
to avoid over-penalizing latent information too early; the general requirement is an objective or
schedule that prevents immediate collapse to an unused-latent solution.

# Baselines

**Diagonal-Gaussian VAE.** q(z|x) = N(μ(x), σ²(x)) with μ, σ nonlinear functions of x, reparameterized
as z = μ(x) + σ(x) ⊙ ε. *Math/algorithm:* cheap density, cheap sampling, fully parallel. *Gap:* the
posterior is forced factorized, so the bound is loose whenever the true posterior has correlations —
which it almost always does.

**Planar / radial normalizing flows (Rezende & Mohamed 2015).** A chain of invertible maps
z + u·h(wᵀz + b) (planar) refining the posterior, with a tractable per-step Jacobian. *Math/algorithm:*
log q updated by the per-step log-determinant. *Gap:* each step squeezes all information through a
single-unit bottleneck, so it does not scale to high-dimensional latent spaces — effective only up to a
few hundred dimensions.

**Coupling-layer flows — NICE / Real NVP (Dinh et al. 2014, 2016).** A flow that splits z into two
halves, copies one half, and transforms the other half as a neural-network function of the copied half.
*Math/algorithm:* the large copied block gives a triangular Jacobian and a computationally cheap
*inverse* in one pass. *Gap:* updating only half the variables per layer as a function of the other half
was found generally less powerful per transformation than other flows in low dimensions (Rezende &
Mohamed 2015), and these were developed for density estimation rather than as a posterior.

**Hamiltonian variational inference (Salimans et al. 2014).** Generates a transformation by simulating
Hamiltonian dynamics on the latents with auxiliary momentum, guided by the exact posterior and leaving
it invariant for small steps. *Math/algorithm:* could approach the exact posterior given enough steps.
*Gap:* very demanding computationally, and requires an auxiliary variational bound for the momentum
variables that can impede progress if not tight.

# Evaluation settings

The yardstick is generative modeling of binarized MNIST and of CIFAR-10 natural images under a VAE,
measured by the **variational lower bound** and by an **importance-sampled estimate of the marginal
log-likelihood** (e.g. 128 samples), with CIFAR-10 also reported in **bits per dimension** (lower is
better). The MNIST protocol follows the dynamically/statically binarized setup used in prior work
(Burda et al. 2015): a convolutional VAE with ResNet blocks, a layer of Gaussian stochastic units,
ELU nonlinearities, weight normalization with data-dependent initialization, optimized with Adam. The
expressiveness of the posterior is probed by varying the depth and width of the refining chain and by
comparing against a diagonal-Gaussian posterior. For CIFAR-10, a deep latent-variable architecture with
many stochastic layers is the setting, and synthesis speed (seconds per image) is compared against a
fully sequential autoregressive image model on the same hardware. Comparison points from the literature
include prior latent-variable models and autoregressive image models.

# Code framework

The primitives exist already: masked linear layers, autoregressive networks that map a vector to
per-coordinate parameters obeying the autoregressive property, encoder networks, a standard Gaussian
reparameterization, and Adam. A VAE posterior harness can therefore be written with a generic
placeholder transformation after the initial sample and a generic density accumulator.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        # TODO: store and apply the fixed autoregressive mask.
        pass
    def forward(self, x):
        pass

class AutoregressiveNN(nn.Module):
    """One pass: output i depends only on z_{1:i-1} and on context h."""
    def __init__(self, dim, hidden, context_dim):
        super().__init__()
        # TODO: build masked layers and an unconstrained context projection.
        pass
    def forward(self, z, h):
        pass

class Encoder(nn.Module):
    """Maps x to the parameters of an initial simple posterior and an extra context vector."""
    def __init__(self, x_dim, z_dim, context_dim, hidden):
        super().__init__()
        pass
    def forward(self, x):
        # TODO: return (mu0, log_sigma0, h)
        pass

class RefiningStep(nn.Module):
    """One invertible refinement of the latent sample, conditioned on context, with a tractable
    log-determinant."""
    def __init__(self, dim, hidden, context_dim, forget_bias=1.5):
        super().__init__()
        self.net = AutoregressiveNN(dim, hidden, context_dim)
        self.forget_bias = forget_bias
    def forward(self, z, h):
        # TODO: transform z invertibly; return (z_new, log_density_delta).
        pass

class FlexiblePosterior(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, n_steps, hidden, reverse_between_steps=True):
        super().__init__()
        self.encoder = Encoder(x_dim, z_dim, context_dim, hidden)
        self.steps = nn.ModuleList()  # TODO: n_steps RefiningStep layers
        self.reverse_between_steps = reverse_between_steps
    def sample_and_log_prob(self, x):
        # TODO: draw the initial sample, refine through the steps, accumulate log q(z|x).
        pass

def elbo(decoder, prior, x, z, log_q):
    # TODO: E[ log p(x|z) + log p(z) - log q(z|x) ].
    pass

def make_optimizer(params):
    return torch.optim.Adam(params)
```
