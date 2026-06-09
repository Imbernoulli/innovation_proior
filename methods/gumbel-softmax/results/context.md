# Context

## Research question

We want to train stochastic neural networks whose internal computation involves a **discrete
random choice** — a categorical latent variable — end-to-end with gradient descent. Concretely,
somewhere inside the network a variable `z` is drawn from a categorical distribution
`z ~ Categorical(π)` whose class probabilities `π = π(θ)` are produced by upstream parameters `θ`,
and the rest of the network turns `z` into a scalar cost `f(z)`. The quantity we actually optimize
is the expected cost

```
L(θ) = E_{z ~ p_θ(z)} [ f(z) ],
```

and training by SGD requires an estimate of `∇_θ L(θ)`. The difficulty is that the path from `θ`
to the loss runs **through a sampling operation**. Drawing a categorical sample is, at heart, an
`argmax` over perturbed scores followed by a one-hot encoding; both `argmax` and one-hot have zero
gradient almost everywhere, so the chain rule cannot carry a gradient from `f(z)` back to `θ`.
A solution has to deliver a gradient estimator for `∇_θ E[f(z)]` over a categorical node that is
(i) usable with a single sample per step, (ii) low enough variance to converge at a practical
rate, and (iii) a drop-in inside ordinary backpropagation. Discrete latents matter because they
are the natural representation for semantic classes, attention/region selection, memory
addressing, and discrete actions — and they are often more interpretable and more compute-efficient
than continuous codes — yet they are rarely used precisely because of this gradient obstacle.

## Background

The field has a clean and successful template for the *continuous* version of this problem. When a
latent variable is reparameterizable — expressible as a deterministic function of the parameters
and an independent noise source — the gradient of an expectation can be pushed inside the
expectation as an ordinary path derivative. This is what makes variational autoencoders trainable
with backprop (Kingma & Welling, 2013; Rezende et al., 2014): a Gaussian latent
`z ~ N(μ, σ²)` is rewritten `z = μ + σ·ε`, `ε ~ N(0,1)`, so `∂z/∂μ` and `∂z/∂σ` are trivial and
the noise carries none of the parameter dependence. The estimator is unbiased and, empirically,
low variance.

The trouble is that this template breaks for discrete variables. There is no smooth deterministic
map from `(θ, ε)` to a *one-hot* output: any function whose output is constrained to the corners of
the simplex is piecewise constant, so its derivative with respect to `θ` is zero wherever it is
defined and undefined on the boundaries. So the reparameterization route, as it stands, simply does
not apply to a categorical draw.

Two facts about discrete sampling are load-bearing here. First, there is an exact way to sample a
categorical via a noisy `argmax`: the **Gumbel-Max** construction (Gumbel, 1954; Maddison, Tarlow &
Minka, 2014). If one adds i.i.d. `Gumbel(0,1)` noise to the log-probabilities and takes the
`argmax`, the index returned is distributed exactly as `Categorical(π)`. The `Gumbel(0,1)`
distribution has CDF `F(g) = exp(-e^{-g})` and is sampled by inverse transform as
`g = -log(-log u)`, `u ~ U(0,1)`; it is **max-stable** (the maximum of i.i.d. Gumbels is again
Gumbel), which is exactly the property that makes the noisy-argmax identity hold.

A second relevant fact is empirical, about the *cost* of the existing workaround for discrete
latents. In a semi-supervised generative model with a categorical class variable `y` and a Gaussian
style variable `z` (Kingma et al., 2014), the only way to train on unlabeled data without a
discrete-gradient estimator is to **marginalize** `y` over all `k` classes — running the inference
and generative networks once per class. If `D, I, G` are the per-call costs of the
class network `q(y|x)`, the inference network `q(z|x,y)`, and the generative network `p(x|y,z)`,
the unlabeled step costs `O(D + k(I + G))`, scaling linearly in the number of classes. This is the
concrete bottleneck a single-sample discrete estimator would remove (bringing the step to
`O(D + I + G)`).

## Baselines

**Score-function / REINFORCE / likelihood-ratio estimator** (Williams, 1992; Glynn; L'Ecuyer).
Using `∇_θ p_θ(z) = p_θ(z) ∇_θ log p_θ(z)`,

```
∇_θ E_z[f(z)] = E_z[ f(z) ∇_θ log p_θ(z) ].
```

It is unbiased, requires only that `p_θ(z)` be continuous in `θ`, and notably does **not**
backpropagate through `f` or through the sample — `f` can be a black box. Its defect is variance:
the estimator multiplies the raw learning signal `f(z)` by the score, and its variance grows
roughly linearly with the dimensionality of the sample, which makes it slow to converge and
especially painful for high-dimensional categorical structure.

**Control-variate-corrected score-function estimators.** The standard remedy subtracts a baseline
or control variate `b(z)` from the learning signal and adds back the corresponding score term to
stay unbiased:

```
∇_θ E_z[f(z)] = E_z[ (f(z) - b(z)) ∇_θ log p_θ(z) ] + E_z[ b(z) ∇_θ log p_θ(z) ].
```

When `b` has a tractable expectation and no direct dependence on `θ`, the correction term can be
computed analytically as `∇_θ E_z[b(z)]`; when `b` is constant with respect to `z`, it vanishes.
Concrete instances: **NVIL** (Mnih & Gregor) centers `f` with a moving average and an
input-dependent neural baseline, plus variance normalization. **DARN** uses a first-order Taylor
expansion `b = f(z̄) + f'(z̄)(z - z̄)` and drops the correction term (biased for non-quadratic `f`).
**MuProp** (Gu et al., 2016) uses the same Taylor baseline but keeps the correction
`μ_b = f'(z̄) ∇_θ E[z]`, computed via a mean-field forward pass. **VIMCO** (Mnih & Rezende, 2016)
builds, for multi-sample objectives, a per-sample baseline from the other samples. These reduce
variance but add machinery (extra networks, normalization, Taylor approximations) and still inherit
the score-function form.

**Straight-Through estimator for Bernoulli** (Bengio, Léonard & Courville, 2013). A biased
path-derivative for non-reparameterizable variables: replace the true sample gradient with that of
a differentiable proxy `m(θ)`, i.e. `∇_θ z ≈ ∇_θ m(θ)`. For a Bernoulli with mean `θ`, the proxy is
the mean itself, giving `∇_θ z ≈ 1` — the forward pass thresholds, the backward pass passes the
gradient straight through. Cheap and simple, but biased, and because the gradient is taken with
respect to a sample-*independent* mean there is a mismatch between the forward and backward passes
that inflates variance. It is also formulated for binary, not general categorical, variables.

**Slope-annealed Straight-Through for binary gates** (Chung et al., 2016). A hard binary sample is
used in the forward pass, while the backward pass follows a smooth sigmoid or hard-sigmoid proxy
whose slope is increased during training. The slope schedule lets the proxy approach a step
function, but the construction remains tied to Bernoulli variables and inherits the biased
forward/backward mismatch of Straight-Through methods.

## Evaluation settings

The natural yardsticks are the standard benchmarks for discrete-latent gradient estimators on
binarized MNIST. (1) **Structured output prediction** with stochastic binary networks: predict the
lower half of a `28×28` MNIST digit from the upper half, optimizing an importance-sampled
log-likelihood (`m=1` sample for training, `m=1000` for evaluation), with latents arranged as
either Bernoulli units or grouped categorical units (e.g. 20 variables of 10 classes each). (2)
**Variational autoencoders** on binarized MNIST with discrete latents (200 Bernoulli, or `20×10`
categorical), evaluated by a multi-sample variational bound (`m=1000`). (3) **Semi-supervised
classification** on MNIST with a small labeled set (e.g. 100 labeled, 50,000 unlabeled), measuring
the variational lower bound, unlabeled-data classification accuracy, and training throughput
(steps/sec) as the number of classes grows. Optimizer: SGD with momentum 0.9; learning rate chosen
on a validation split. Discrete-estimator baselines for comparison: SF, DARN, MuProp,
Straight-Through, and slope-annealed Straight-Through.

## Code framework

The primitives that already exist: a deep-learning framework with autograd, `softmax`, an SGD
optimizer, a data pipeline for binarized MNIST, and the VAE training loop (encoder → sample latent
→ decoder → ELBO). What does **not** exist yet is a differentiable way to sample the discrete latent
node; that is the single empty slot.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_categorical_latent(logits, **kwargs):
    """Draw a latent from a categorical node parameterized by `logits`, in a way that
    lets a gradient flow back to `logits`."""
    # TODO: a differentiable (approximate) categorical sample.
    pass


class CategoricalLatentVAE(nn.Module):
    def __init__(self, x_dim, n_vars, n_classes, hidden=256):
        super().__init__()
        self.n_vars, self.n_classes = n_vars, n_classes
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_vars * n_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_vars * n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, **kwargs):
        logits = self.encoder(x).view(-1, self.n_vars, self.n_classes)
        z = sample_categorical_latent(logits, **kwargs)   # TODO: the differentiable sample
        x_logits = self.decoder(z.view(x.size(0), -1))
        return x_logits, logits


def elbo(x, x_logits, logits):
    # reconstruction term
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="none").sum(-1)
    # KL between categorical posterior and uniform/learned prior
    q = F.softmax(logits, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    kl = (q * (log_q + math.log(logits.size(-1)))).sum(-1).sum(-1)
    return (recon + kl).mean()


def train_step(model, optimizer, x, **kwargs):
    optimizer.zero_grad()
    x_logits, logits = model(x, **kwargs)
    loss = elbo(x, x_logits, logits)
    loss.backward()
    optimizer.step()
    return loss.item()
```
