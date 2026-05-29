# Research question

We have a directed probabilistic model with a continuous latent variable per datapoint: a value `z` is drawn from a prior `p(z)`, then the observation `x` is drawn from a conditional likelihood `p(x|z)`. We want `p(x|z)` to be an arbitrarily flexible function of `z` — concretely a neural network with one or more nonlinear hidden layers, mapping `z` to the parameters of the distribution over `x`. This is precisely the regime in which the field, around 2013, has no good tool.

For an i.i.d. dataset `X = {x^(1), …, x^(N)}` of `N` samples, three quantities are wanted at once:

1. Efficient approximate maximum-likelihood (or MAP) estimation of the global generative parameters `θ`, so the process can be analyzed and so new data resembling `X` can be generated.
2. Efficient approximate posterior inference of `z` given an observed `x` for a chosen `θ` — i.e. a fast `z ≈ q(z|x)` for coding, representation, and visualization.
3. Efficient approximate marginal inference over `x` (denoising, inpainting, super-resolution), which needs a usable handle on `p(x)`.

Two properties make this hard and form the crux of the problem:

- **Intractability.** The marginal likelihood `p(x) = ∫ p(z) p(x|z) dz` has no closed form once `p(x|z)` is a nonlinear network, so it can neither be evaluated nor differentiated. The true posterior `p(z|x) = p(x|z)p(z)/p(x)` is therefore also intractable — it shares the same uncomputable denominator — and so are the expectations any standard mean-field scheme would require. These intractabilities are the *generic* case for moderately complicated likelihoods, not a corner case.
- **Scale.** `N` is large enough that full-batch optimization is too expensive; we want to update parameters on small minibatches, ideally single datapoints. Any method that runs an expensive per-datapoint inner loop — an inner optimization to convergence, or a sampling chain — before each parameter update is disqualified at this scale, and also blocks inference on a fresh test point.

The goal is a single estimator of the training objective that is (a) differentiable, (b) low-variance, (c) computable on a minibatch, and (d) able to pass gradients through a neural-network approximate posterior — so that the generative model and the inference machinery train jointly under ordinary gradient ascent.

# Background

The standard object for learning a latent-variable model is the marginal log-likelihood `log p(x)`. For *any* distribution `q(z)` it decomposes exactly as

```
log p(x) = KL( q(z) || p(z|x) ) + L,    where  L = E_q[ log p(x,z) - log q(z) ].
```

Because the KL divergence is non-negative, `L` is a lower bound on `log p(x)` — the evidence lower bound. The bound is tight precisely when `q(z) = p(z|x)`. Every classical method below is a different strategy for handling this decomposition, and each rests on an assumption that the nonlinear-likelihood / large-data regime violates. Maximizing `L` jointly in `q` and the model does two jobs at once: pushing `L` up by improving `q` shrinks the gap `KL(q‖p(z|x))` (so `q` becomes a better posterior approximation — inference), and pushing `L` up by improving the model raises a *guaranteed* underestimate of `log p(x)` (learning). There is no way for the true likelihood to be worse than the number being optimized.

Two pieces of supporting theory matter, and the contrast between them is the conceptual hinge of the whole area — they are the two ways to differentiate an expectation whose measure depends on a parameter.

First, the **score-function (log-derivative) identity**: for a density `q_φ(z)` and an integrand with no explicit `φ`-dependence, because `∇_φ q_φ = q_φ ∇_φ log q_φ`,

```
∇_φ E_{q_φ}[f(z)] = E_{q_φ}[ f(z) ∇_φ log q_φ(z) ].
```

This gives an unbiased Monte-Carlo gradient of an expectation whose *measure* depends on `φ`, requiring only the ability to evaluate `f`, not differentiate it. If the integrand is itself `f_φ`, the full derivative has one more term: `∇_φ E_{q_φ}[f_φ] = E_{q_φ}[f_φ ∇_φ log q_φ + ∇_φ f_φ]`. For the ELBO integrand `f_φ(z)=log p_θ(x,z)-log q_φ(z|x)`, the extra term is `-E_q[∇_φ log q_φ]=0`, so the naive ELBO score estimator is still unbiased; its problem is variance, not bias.

Second, the **change-of-variables / pushforward fact**: if `z = g_φ(ε)` is a differentiable transform of a *fixed* auxiliary noise `ε ~ p(ε)`, then `q_φ(z) ∏_i dz_i = p(ε) ∏_i dε_i` (probability conservation), so any expectation `E_{q_φ(z)}[f(z)] = E_{p(ε)}[f(g_φ(ε))]` — the *same* expectation written against a `φ`-independent measure. With the measure no longer carrying `φ`, the gradient can move inside the expectation and flow through `g_φ`.

A third structural idea is **amortized inference**. Classical variational inference assigns *each* datapoint `x^(i)` its own free variational parameters (e.g. a mean and variance per point) and optimizes those local parameters to convergence before every global update. With `N` in the millions this is ruinous: the parameter count grows with `N`, the inner optimization (or sampling chain) per point dominates cost, and a never-before-seen test `x` requires its own fresh inner optimization. The alternative — visible in the recognition network of the Helmholtz machine and in predictive sparse decomposition — is to *predict* the local variational parameters with a single shared network `q_φ(z|x)` whose weights `φ` are tied across all datapoints. Inference then costs one forward pass and generalizes to new `x` for free, with a parameter count fixed in `N`.

The prevailing wisdom around 2013 is that deep directed generative models with continuous latents and neural-network likelihoods are attractive but effectively untrainable: exact EM is impossible (no closed-form posterior), mean-field needs a conjugacy a network destroys, generic black-box variational gradients are too noisy, sampling-based EM is too slow per datapoint, and the one online competitor optimizes two objectives that do not jointly bound the likelihood. Meanwhile the practical substrate is mature: stochastic gradient descent with backpropagation through multilayer perceptrons is routine, GPUs make minibatch training cheap, and adaptive step-size methods (Adagrad; Duchi et al. 2010) are available. The missing piece is squarely a *gradient estimator*, not new hardware or a new network class.

# Baselines

**EM (Dempster et al. 1977; linear-Gaussian instance via Roweis & Ghahramani 1999).**
EM maximizes `log p(x)` by alternating: the E-step sets `q(z) = p(z|x; θ_old)`, which makes `KL = 0` and the bound tight; the M-step maximizes `E_q[log p(x,z|θ)]` over `θ` (the complete-data log-likelihood, with the log *inside* the expectation). Its core requirement is a tractable posterior `p(z|x)` for the E-step. Roweis & Ghahramani showed PCA is the ML solution of a special linear-Gaussian model (`p(z)=N(0,I)`, `p(x|z)=N(Wz, εI)` as `ε→0`), establishing the lineage from linear autoencoders to generative latent-variable models — but only for linear/Gaussian models with a closed-form posterior. **Gap:** for a nonlinear-network likelihood the E-step posterior has no closed form, so EM cannot even start.

**Mean-field variational inference and Stochastic VI (Hoffman, Blei, Wang & Paisley 2013).**
Replace the intractable posterior with a factorized `q(z) = ∏_j q_j(z_j)` from a tractable family and maximize the ELBO. Coordinate-ascent (CAVI) updates each factor by `log q_j*(z_j) = E_{-j}[log p(x,z)] + const`. Stochastic VI scales the *global* update with stochastic natural-gradient steps over minibatches. **Gaps:** (1) the CAVI update needs the expectation `E_{-j}[log p(x,z)]` in closed form, which holds only for conditionally-conjugate exponential-family models and breaks immediately under a nonlinear-network likelihood; (2) the factorial assumption cannot represent posterior correlations; (3) classically, per-datapoint local variational parameters must be optimized to convergence before each global update — expensive at scale (SVI mitigates the scale issue but still assumes conjugacy / analytic local updates).

**Score-function / black-box variational gradients (Blei, Jordan & Paisley 2012; Ranganath, Gerrish & Blei 2014).**
Drop the conjugacy requirement: estimate the ELBO gradient directly with the score-function identity. For `f_φ(z)=log p_θ(x,z)-log q_φ(z|x)`, the full derivative is `E_q[f_φ(z)∇_φ log q_φ(z|x)+∇_φ f_φ(z)]`; the explicit term is `-E_q[∇_φ log q_φ]=0`, so the usual Monte Carlo estimator is `(1/L) Σ_l f_φ(z^(l)) ∇_φ log q_φ(z^(l)|x)`, `z^(l) ~ q_φ`. This is unbiased, "black-box" (only evaluates the joint density and the approximate-posterior density), and works even for discrete `z`. **Gap:** the variance is very high — the ELBO integrand multiplies the zero-mean but noisy score, and the variance does not shrink just because the model term is smooth (when a parameter-free reward is nearly constant, the true gradient is near zero yet the estimator still emits reward-times-score noise) — so it needs control variates, Rao-Blackwellization, and baselines just to be usable, and is impractical as a default for continuous-latent neural models.

**Wake-sleep / the Helmholtz machine (Hinton, Dayan, Frey & Neal 1995).**
The only prior online method for this same general class of continuous-latent directed models. It already introduces a separate **recognition network** `q(z|x)` (an "encoder") that approximates the posterior, trained alongside a **generative network** `p(x|z)`. The *wake* phase samples `z ~ q(z|x)` and updates the generative weights to raise `log p(x,z)`; the *sleep* phase "dreams" `(x,z) ~ p` and updates the recognition weights to predict `z` from `x`. **Gap:** it optimizes two distinct objectives that do not jointly correspond to optimizing one bound on `log p(x)` — the sleep phase minimizes a reversed KL on fantasy data rather than the KL of the recognition model from the true posterior on real data — so there is no single coherent objective and no guarantee of improving the marginal likelihood. (Worth noting: wake-sleep also handles discrete latents.)

**Closest prior use of a reparameterization-style trick (Salimans & Knowles 2013).**
Within a fixed-form stochastic variational inference scheme, they used a reparameterization-like change of variables to learn the natural parameters of exponential-family approximating distributions. **Gap:** it is tied to exponential-family natural parameters, not a general neural-network recognition model, so it does not deliver an amortized, network-based posterior trainable jointly with a nonlinear generative model.

**Sampling-based EM baseline (Monte Carlo EM with Hybrid/Hamiltonian Monte Carlo; Duane et al. 1987).**
When the E-step posterior is intractable, sample from it with gradient-based MCMC using `∇_z log p(z|x) = ∇_z log p(z) + ∇_z log p(x|z)` (computable because the normalizer drops out), then take M-step updates. **Gap:** it runs a sampling chain per datapoint, so it is not an online algorithm and does not scale to a large dataset — usable only as a small-scale reference.

**Autoencoder lineage (the "auto-encoding" half of the eventual picture).**
Denoising / contractive / sparse autoencoders (Vincent et al. 2010; Bengio et al. 2013) train an encoder-decoder by reconstruction plus an *ad hoc* regularizer; under an infomax reading (Linsker 1989), reconstruction lower-bounds the mutual information `I(X;Z)`. Predictive sparse decomposition (Kavukcuoglu et al. 2008) is a predictive encoder for sparse coding. **Gap:** reconstruction alone does not yield useful representations, and the regularizers carry nuisance hyperparameters with no probabilistic meaning; there is no marginal-likelihood objective behind them.

# Evaluation settings

The natural yardstick is generative modeling of images on two pre-existing datasets:

- **MNIST** — 28×28 grayscale handwritten digits, treated as (binarized) data well-suited to a Bernoulli observation model; the larger of the two, suited to wider hidden layers such as ~500 units.
- **Frey Faces** — a small set of ~20×20 grayscale video frames of a face, continuous-valued in (0,1), suited to a Gaussian observation model with means squashed to (0,1); being small, it uses fewer hidden units (~200) to avoid overfitting.

The metrics are the **variational lower bound** (average ELBO per datapoint, used to compare optimization quality and convergence speed) and the **estimated marginal log-likelihood** `log p(x)` (held-out generative quality). A standard protocol initializes parameters by random sampling from `N(0, 0.01)`, uses minibatches of size `M = 100`, uses one latent sample per datapoint when the minibatch is large enough, optimizes by stochastic gradient ascent with Adagrad step sizes chosen from `{0.01, 0.02, 0.1}` on early training performance, and adds a small weight-decay term corresponding to an `N(0, I)` prior on the parameters (so optimization is approximate MAP).

The **marginal-likelihood estimation protocol** is a self-contained recipe usable to score any trained model, reliable only in low-dimensional latent spaces (≲5 dimensions, e.g. 3 latents with ~100 hidden units): for a datapoint `x`, (1) draw `L` samples `{z^(l)}` from the posterior with gradient-based MCMC (HMC, using `∇_z log p(z|x) = ∇_z log p(z) + ∇_z log p(x|z)`); (2) fit a simple density estimator `q(z)` to those samples; (3) draw a fresh set of posterior samples and form the harmonic-mean-style estimator

```
p(x) ≈ ( (1/L) Σ_l  q(z^(l)) / [ p(z) p(x|z^(l)) ] )^{-1},   z^(l) ~ p(z|x),
```

which follows from `1/p(x) = ∫ p(z|x) · q(z)/p(x,z) dz`. A practical low-dimensional scoring run estimates marginal likelihood on the first 1000 train and test points, sampling ~50 posterior values per point with HMC (≈4 leapfrog steps). The Monte Carlo EM reference uses ~10 HMC leapfrog steps with step size auto-tuned to ~90% acceptance, followed by 5 weight updates per acquired sample, under the same Adagrad schedule.

# Code framework

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
    # TODO: choose the latent sampling/proposal step and its gradient estimator
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
