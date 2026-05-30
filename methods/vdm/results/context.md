# Context

## Research question

Likelihood-based generative modeling of natural images is measured in bits per dimension (BPD) on
held-out data: the negative log-likelihood the model assigns to test images, normalized per pixel-channel.
On the standard density-estimation benchmarks (CIFAR-10, downsampled ImageNet), autoregressive models have
held the top of the leaderboard for years. A separate family — models that corrupt data with Gaussian noise
and learn to reverse the corruption — has begun producing perceptually striking samples, but it has not
matched autoregressive likelihoods. The open question is sharp: can a noise-corruption generative model also
be a *great likelihood* model, competitive with or better than autoregressive models on BPD?

A solution would have to deliver three things. First, an objective that is genuinely a bound on the data
log-likelihood (not a quality-tuned surrogate), so that minimizing it minimizes BPD. Second, a tractable,
numerically stable way to evaluate and optimize that bound — existing implementations of this model family
require 64-bit floating point to avoid catastrophic rounding, which is awkward on modern accelerators. Third,
a way to handle the part of the likelihood that is most punishing: the fine-scale, exact-pixel-value detail
that perceptual-quality methods can afford to ignore but a likelihood number cannot.

## Background

**Gaussian corruption processes.** The field's foundational idea is to define a forward process that
gradually adds Gaussian noise to data `x`, producing a sequence of increasingly noisy latents `z_t`, and to
learn a model that reverses it. A convenient closed form for the marginal of the corruption is
`q(z_t|x) = N(α_t x, σ_t² I)`, where `α_t` shrinks the signal and `σ_t` grows the noise as `t` runs from
`0` (clean) to `1` (pure noise). Two regimes are common in the literature: the *variance-preserving* choice
`α_t² = 1 − σ_t²` (signal scaled down as noise is added) and the *variance-exploding* choice `α_t² = 1`
(signal kept, noise blown up). The quantity that summarizes "how corrupted" a latent is, is the
signal-to-noise ratio `SNR(t) = α_t²/σ_t²`, monotonically decreasing in `t`.

**The latent-variable / VAE view.** A variational autoencoder (Kingma & Welling 2013; Rezende et al. 2014)
maximizes an evidence lower bound `log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x)‖p(z))`, using the
reparameterization trick to get low-variance gradients through the Gaussian latent. A corruption process with
a learned reverse process is exactly a VAE in which the inference network is *fixed* (it is the known Gaussian
`q`) and the latent is *deep* — a long Markov chain `z_0, z_1, …, z_1` rather than a single layer. This makes
the ELBO decompose, layer by layer, into a sum of KL terms — one per transition — plus a reconstruction term
and a prior term. The depth of the chain is a free hyperparameter, and one can ask what happens as it goes to
infinity.

**Denoising score matching and the score view.** Vincent (2011) established that fitting a model to the score
`∇ log q(z_t|x)` of the single-example corruption is equivalent, up to an `x`-independent constant, to fitting
the score `∇ log q(z_t)` of the marginal. This "denoising score matching" identity means a denoiser trained on
noisy/clean pairs recovers the true marginal score — a consistency guarantee. Because for `q(z_t|x)=N(α_t x,σ_t²)`
we have `∇_{z_t} log q(z_t|x) = −(z_t − α_t x)/σ_t² = −ε/σ_t`, predicting the noise `ε`, predicting the clean
`x`, and predicting the score are three linearly-related reparameterizations of the same object.

**Continuous-time / SDE framing.** The corruption chain, taken to infinitely many steps, is a stochastic
differential equation (Song et al. 2020); its time-reversal is another SDE driven by the marginal score, and
there is an associated deterministic probability-flow ODE whose change-of-variables gives an exact likelihood.
This continuous view unifies the variance-preserving and variance-exploding regimes as instances of one SDE.

**Motivating empirical facts about existing systems.** (1) The discrete-time ELBO of a corruption model, when
the per-step variances are shared across dimensions, equals a sum of denoising-score-matching losses with a
*specific per-noise-level weighting*; the perceptual-quality recipes drop that weighting (set all weights to
one on the noise-prediction MSE), which improves samples but is no longer the likelihood bound. (2) Naive
implementations of the discrete-time loss compute intermediate quantities very close to 1, where floating
point is least precise; practitioners resort to 64-bit arithmetic to get correct results. (3) The noise
schedule (the shape of `SNR(t)`) is hand-designed — β-linear, cosine — and treated as a fixed hyperparameter;
different schedules give very different bounds and very different estimator variance. (4) At the lowest noise
levels the marginal `q(z_t)` is sharply peaked because 8-bit pixel data is discrete; a denoiser that processes
data only through smooth convolutions struggles to capture this fine-scale structure, and likelihood (unlike
FID) is acutely sensitive to it.

## Baselines

**Deep diffusion / unfolded VAE (Sohl-Dickstein et al. 2015).** Defines the forward Gaussian corruption and a
learned reverse Markov chain, trained by maximizing the ELBO, which decomposes into a prior KL, a
reconstruction term, and a sum of transition KLs. Establishes the template but yields weak samples and
uncompetitive likelihood; the objective is cumbersome.

**Denoising diffusion with noise prediction (Ho et al. 2020).** Parameterizes the reverse-process mean through
a *noise-prediction network* `ε̂_θ(z_t,t)`, with `x̂ = (z_t − σ_t ε̂)/α_t`. Shows the discrete ELBO equals a
weighted sum of denoising-score-matching losses, then trains the *simplified* objective
`E‖ε − ε̂_θ(z_t,t)‖²` — the same MSE but with the likelihood weighting removed. Uses a fixed β-linear
schedule and requires 64-bit arithmetic for the exact bound. Excellent FID; the simplified loss is not the
likelihood bound, and likelihood lags autoregressive models.

**Noise-conditional score networks (Song & Ermon 2019; 2020).** Learn the marginal score at multiple noise
scales via denoising score matching, with a variance-exploding schedule (`α_t = 1`, `σ_t` a geometric series)
and Langevin-style sampling. The implied per-level weighting happens to be constant — consistent with the
likelihood bound — but the schedule endpoints are fixed by hand and not optimized.

**Score-SDE (Song et al. 2020).** Casts corruption as a forward SDE and generation as the reverse SDE /
probability-flow ODE, unifying variance-preserving and variance-exploding regimes and giving exact likelihood
through the ODE. The likelihood comes from continuous-flow machinery rather than a short, transparent
variational expression; the diffusion specification is fixed.

**Improved discrete diffusion (Nichol & Dhariwal 2021).** A cosine schedule and learned reverse variances
improve likelihood over the original recipe, but the schedule is still a fixed hand-designed family and the
objective remains a perceptual-quality-leaning weighted loss.

**VAEs / flows as the likelihood incumbents on these benchmarks.** Hierarchical VAEs (IAF-VAE, Very Deep VAE,
NVAE), normalizing flows (Glow, Flow++), and especially deep autoregressive transformers (PixelCNN/++,
Image Transformer, Sparse/Routing Transformer) are the standing BPD baselines a new method must beat.

## Evaluation settings

Datasets: CIFAR-10 (32×32, with and without standard augmentations — flips, 90° rotations, channel swaps) and
downsampled ImageNet at 32×32 and 64×64. Metric: bits per dimension (negative log-likelihood per pixel-channel
on the test set), for which a variational upper bound on the negative log-likelihood is the reported quantity.
A secondary perceptual metric (FID) exists but is not the target. A further yardstick for a hierarchical latent
model is lossless compression: the negative ELBO is the expected codelength under a bits-back coding scheme
(Hinton 1993; Townsend et al. 2018; Kingma et al. 2019), so net codelength on the test set measures how close
to the theoretical optimum a practical coder gets. Hardware/optimization protocol: deep convolutional U-Net
denoisers trained with Adam, exponential-moving-average parameters for evaluation, on accelerator hardware.

## Code framework

The pieces below already exist before the method: a data pipeline that yields integer pixel batches, an
optimizer, a U-Net-style convolutional backbone that maps a noisy image (plus a scalar time/level embedding)
to an output of the same shape, and a training loop that averages a per-example loss. The contribution will
fill the empty slots: the exact corruption marginal, the three-part bound, the transition loss, the schedule,
and the input featurization.

```python
import jax, jax.numpy as jnp
from flax import linen as nn

# --- existing: convolutional denoiser backbone (U-Net), sinusoidal level embedding ---
class Denoiser(nn.Module):
    """Maps (noisy image z, scalar level embedding) -> output of same shape as z."""
    @nn.compact
    def __call__(self, z, level, conditioning, deterministic=True):
        # standard ResNet/U-Net stack with level conditioning; predicts a same-shape field
        ...

# --- existing: discrete-data likelihood head (encode pixels to [-1,1], decode to per-pixel logits) ---
class EncDec(nn.Module):
    def encode(self, x): ...     # integers -> [-1, 1]
    def decode(self, z, level): ...  # -> per-pixel categorical logits
    def logprob(self, x, z, level): ...

# --- TODO (the method): the corruption process + its learned bound ---

class NoiseSchedule(nn.Module):
    """Maps t in [0,1] to a scalar level controlling how much signal vs noise.
    TODO: decide the functional form and whether/how it is learned."""
    @nn.compact
    def __call__(self, t):
        raise NotImplementedError

class GenerativeModel(nn.Module):
    config: object
    def setup(self):
        self.encdec = EncDec(self.config)
        self.denoiser = Denoiser(self.config)
        self.schedule = NoiseSchedule(self.config)  # TODO

    def __call__(self, images, conditioning, deterministic=True):
        # TODO: encode x; define the corruption marginal q(z_t|x);
        # TODO: reconstruction term  E_q[-log p(x|z_0)]
        # TODO: prior term           KL(q(z_1|x) || p(z_1))
        # TODO: the per-transition / continuous-time corruption loss
        # return the three terms whose sum is the negative bound
        raise NotImplementedError

    def sample(self, ...):
        # TODO: ancestral reverse process using the denoiser
        raise NotImplementedError


def featurize_input(z):
    # TODO: how to present z to the denoiser so it can model fine pixel-level detail
    return z


def train_step(state, batch, rng):
    def loss_fn(params):
        out = state.apply_fn(params, **batch, rngs={"sample": rng})
        # TODO: combine the three terms into the per-example negative bound (in bits/dim)
        ...
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)
```
