## Research question

We have a directed probabilistic model with a continuous latent variable per datapoint: a value `z` is drawn from a prior `p(z)`, then the observation `x` is drawn from a conditional likelihood `p(x|z)`. We want `p(x|z)` to be an arbitrarily flexible function of `z` — concretely a neural network with one or more nonlinear hidden layers, mapping `z` to the parameters of the distribution over `x`.

For an i.i.d. dataset `X = {x^(1), …, x^(N)}` of `N` samples, three quantities are of interest:

1. Approximate maximum-likelihood (or MAP) estimation of the global generative parameters `θ`, so the process can be analyzed and so new data resembling `X` can be generated.
2. Approximate posterior inference of `z` given an observed `x` for a chosen `θ` — i.e. a `z ≈ q(z|x)` for coding, representation, and visualization.
3. Approximate marginal inference over `x` (denoising, inpainting, super-resolution), which needs a handle on `p(x)`.

Two properties characterize the setting:

- **Intractability.** The marginal likelihood `p(x) = ∫ p(z) p(x|z) dz` has no closed form once `p(x|z)` is a nonlinear network. The true posterior `p(z|x) = p(x|z)p(z)/p(x)` shares the same denominator, and so do the expectations a mean-field scheme would require. This is the generic case for moderately complicated likelihoods.
- **Scale.** `N` is large, so full-batch optimization is expensive; parameters are updated on small minibatches, ideally single datapoints.

The question is how to estimate the training objective and its gradients for such a model so that the generative model and the inference machinery train jointly under ordinary minibatch gradient ascent.

## Background

The standard object for learning a latent-variable model is the marginal log-likelihood `log p(x)`. For *any* distribution `q(z)` it decomposes exactly as

```
log p(x) = KL( q(z) || p(z|x) ) + L,    where  L = E_q[ log p(x,z) - log q(z) ].
```

Because the KL divergence is non-negative, `L` is a lower bound on `log p(x)` — the evidence lower bound. The bound is tight precisely when `q(z) = p(z|x)`. Maximizing `L` jointly in `q` and the model does two jobs at once: pushing `L` up by improving `q` shrinks the gap `KL(q‖p(z|x))` (inference), and pushing `L` up by improving the model raises a guaranteed underestimate of `log p(x)` (learning).

One piece of supporting theory matters: how to differentiate an expectation whose measure itself depends on the parameter being differentiated.

The **score-function (log-derivative) identity**: for a density `q_φ(z)` and an integrand with no explicit `φ`-dependence, because `∇_φ q_φ = q_φ ∇_φ log q_φ`,

```
∇_φ E_{q_φ}[f(z)] = E_{q_φ}[ f(z) ∇_φ log q_φ(z) ].
```

This gives an unbiased Monte-Carlo gradient of an expectation whose *measure* depends on `φ`, requiring only the ability to evaluate `f`, not differentiate it. If the integrand is itself `f_φ`, the full derivative has one more term: `∇_φ E_{q_φ}[f_φ] = E_{q_φ}[f_φ ∇_φ log q_φ + ∇_φ f_φ]`. For the ELBO integrand `f_φ(z)=log p_θ(x,z)-log q_φ(z|x)`, the extra term is `-E_q[∇_φ log q_φ]=0`, so the naive ELBO score estimator is unbiased.

A second structural idea is **amortized inference**. Classical variational inference assigns *each* datapoint `x^(i)` its own free variational parameters (e.g. a mean and variance per point) and optimizes those local parameters before every global update. An alternative — visible in the recognition network of the Helmholtz machine and in predictive sparse decomposition — is to *predict* the local variational parameters with a single shared network `q_φ(z|x)` whose weights `φ` are tied across all datapoints. Inference then costs one forward pass and applies to new `x`, with a parameter count fixed in `N`.

Around 2013 the practical substrate is mature: stochastic gradient descent with backpropagation through multilayer perceptrons is routine, GPUs make minibatch training cheap, and adaptive step-size methods (Adagrad; Duchi et al. 2010) are available.

## Baselines

**EM (Dempster et al. 1977; linear-Gaussian instance via Roweis & Ghahramani 1999).**
EM maximizes `log p(x)` by alternating: the E-step sets `q(z) = p(z|x; θ_old)`, which makes `KL = 0` and the bound tight; the M-step maximizes `E_q[log p(x,z|θ)]` over `θ` (the complete-data log-likelihood, with the log *inside* the expectation). It uses a tractable posterior `p(z|x)` for the E-step. Roweis & Ghahramani showed PCA is the ML solution of a special linear-Gaussian model (`p(z)=N(0,I)`, `p(x|z)=N(Wz, εI)` as `ε→0`), establishing the lineage from linear autoencoders to generative latent-variable models for linear/Gaussian models with a closed-form posterior.

**Mean-field variational inference and Stochastic VI (Hoffman, Blei, Wang & Paisley 2013).**
Replace the posterior with a factorized `q(z) = ∏_j q_j(z_j)` from a tractable family and maximize the ELBO. Coordinate-ascent (CAVI) updates each factor by `log q_j*(z_j) = E_{-j}[log p(x,z)] + const`, which needs the expectation `E_{-j}[log p(x,z)]` in closed form (conditionally-conjugate exponential-family models). Stochastic VI scales the *global* update with stochastic natural-gradient steps over minibatches.

**Score-function / black-box variational gradients (Blei, Jordan & Paisley 2012; Ranganath, Gerrish & Blei 2014).**
Estimate the ELBO gradient directly with the score-function identity, dropping the conjugacy requirement. For `f_φ(z)=log p_θ(x,z)-log q_φ(z|x)`, the full derivative is `E_q[f_φ(z)∇_φ log q_φ(z|x)+∇_φ f_φ(z)]`; the explicit term is `-E_q[∇_φ log q_φ]=0`, so the Monte Carlo estimator is `(1/L) Σ_l f_φ(z^(l)) ∇_φ log q_φ(z^(l)|x)`, `z^(l) ~ q_φ`. This is unbiased, "black-box" (only evaluates the joint density and the approximate-posterior density), and works even for discrete `z`. It is typically paired with control variates, Rao-Blackwellization, and baselines.

**Wake-sleep / the Helmholtz machine (Hinton, Dayan, Frey & Neal 1995).**
A prior online method for this same general class of continuous-latent directed models. It introduces a separate **recognition network** `q(z|x)` (an "encoder") that approximates the posterior, trained alongside a **generative network** `p(x|z)`. The *wake* phase samples `z ~ q(z|x)` and updates the generative weights to raise `log p(x,z)`; the *sleep* phase "dreams" `(x,z) ~ p` and updates the recognition weights to predict `z` from `x` (the sleep phase minimizes a reversed KL on fantasy data). It also handles discrete latents.

**Stochastic change-of-variables device (Salimans & Knowles 2013).**
Within a fixed-form stochastic variational inference scheme, they used a change of variables to learn the natural parameters of exponential-family approximating distributions, tied to exponential-family natural parameters.

**Sampling-based EM baseline (Monte Carlo EM with Hybrid/Hamiltonian Monte Carlo; Duane et al. 1987).**
When the E-step posterior is intractable, sample from it with gradient-based MCMC using `∇_z log p(z|x) = ∇_z log p(z) + ∇_z log p(x|z)` (the normalizer drops out), then take M-step updates. It runs a sampling chain per datapoint.

**Autoencoder lineage.**
Denoising / contractive / sparse autoencoders (Vincent et al. 2010; Bengio et al. 2013) train an encoder-decoder by reconstruction plus a regularizer; under an infomax reading (Linsker 1989), reconstruction lower-bounds the mutual information `I(X;Z)`. Predictive sparse decomposition (Kavukcuoglu et al. 2008) is a predictive encoder for sparse coding. The regularizers carry hyperparameters set outside any marginal-likelihood objective.

## Evaluation settings

The natural yardstick is generative modeling of images on two pre-existing datasets:

- **MNIST** — 28×28 grayscale handwritten digits, treated as (binarized) data well-suited to a Bernoulli observation model; the larger of the two, suited to wider hidden layers such as ~500 units.
- **Frey Faces** — a small set of ~20×20 grayscale video frames of a face, continuous-valued in (0,1), suited to a Gaussian observation model with means squashed to (0,1); being small, it uses fewer hidden units (~200) to avoid overfitting.

The metrics are the **variational lower bound** (average ELBO per datapoint, used to compare optimization quality and convergence speed) and the **estimated marginal log-likelihood** `log p(x)` (held-out generative quality). A standard protocol initializes parameters by random sampling from `N(0, 0.01)`, uses minibatches of size `M = 100`, uses one latent sample per datapoint when the minibatch is large enough, optimizes by stochastic gradient ascent with Adagrad step sizes chosen from `{0.01, 0.02, 0.1}` on early training performance, and adds a small weight-decay term corresponding to an `N(0, I)` prior on the parameters (so optimization is approximate MAP).

The **marginal-likelihood estimation protocol** is a self-contained recipe usable to score any trained model, reliable in low-dimensional latent spaces (≲5 dimensions, e.g. 3 latents with ~100 hidden units): for a datapoint `x`, (1) draw `L` samples `{z^(l)}` from the posterior with gradient-based MCMC (HMC, using `∇_z log p(z|x) = ∇_z log p(z) + ∇_z log p(x|z)`); (2) fit a simple density estimator `q(z)` to those samples; (3) draw a fresh set of posterior samples and form the harmonic-mean-style estimator

```
p(x) ≈ ( (1/L) Σ_l  q(z^(l)) / [ p(z) p(x|z^(l)) ] )^{-1},   z^(l) ~ p(z|x),
```

which follows from `1/p(x) = ∫ p(z|x) · q(z)/p(x,z) dz`. A practical low-dimensional scoring run estimates marginal likelihood on the first 1000 train and test points, sampling ~50 posterior values per point with HMC (≈4 leapfrog steps). The Monte Carlo EM reference uses ~10 HMC leapfrog steps with step size auto-tuned to ~90% acceptance, followed by 5 weight updates per acquired sample, under the same Adagrad schedule.

## Code framework

The implementation substrate is a standard deep-learning stack, illustrated with PyTorch: a minibatch data pipeline, multilayer-perceptron layers with reverse-mode autodiff, and an adaptive stochastic optimizer. The model is a directed latent-variable model with a simple prior `p(z)=N(0,I)` and a neural observation model mapping latent samples to the parameters of `p(x|z)`. A small optional module may map an observation to latent-side quantities so per-datapoint inference can be amortized instead of solved from scratch.

```python
import math
import torch
from torch import nn, optim
from torch.nn import functional as F

train_loader = ...  # yields random minibatches X^M of M datapoints from the dataset

def log_prior(z):
    return -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)

class ObservationModel(nn.Module):
    def __init__(self, z_dim, x_dim, hidden_dim):
        super().__init__()
        # TODO: define an MLP that maps z to parameters of p(x|z)
        pass

    def forward(self, z):
        # TODO: return observation-distribution parameters
        pass

class InferenceModule(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        # TODO: optional x -> latent-side quantities for fast per-datapoint inference
        pass

    def forward(self, x):
        # TODO: return quantities used by the training-time latent sampler
        pass

def draw_latent(batch_size, z_dim, x=None, inference_out=None):
    # TODO: produce the latent sample(s) used by the objective
    pass

def objective(x, observation_model, inference_module=None):
    # TODO: compute a tractable minibatch training objective for the latent-variable model
    pass

observation_model = ObservationModel(z_dim=20, x_dim=784, hidden_dim=400)
inference_module = InferenceModule(x_dim=784, z_dim=20, hidden_dim=400)
parameters = list(observation_model.parameters()) + list(inference_module.parameters())
optimizer = optim.Adagrad(parameters)

for x, _ in train_loader:
    optimizer.zero_grad()
    loss = -objective(x, observation_model, inference_module)
    loss.backward()
    optimizer.step()
```
