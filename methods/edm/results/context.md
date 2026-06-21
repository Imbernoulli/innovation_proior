# Context

## Research question

We want a generative model for images that turns pure Gaussian noise into samples from a complicated data distribution, and we care about three axes: sample quality (low FID), training cost, and generation speed, i.e. how many times we have to evaluate a large neural network to produce one image.

A family of models built around iteratively denoising a noisy image has emerged as a leading approach, often matching or beating GANs. Each member is derived from its own theoretical scaffold — a discrete Markov chain trained by a variational bound, a ladder of score estimates trained by score matching, or a continuous stochastic differential equation — and each derivation arrives bundled with a long list of choices: the schedule of noise levels, whether and how the image is rescaled as noise is added, how the continuous process is discretized into sampling steps, which numerical solver advances those steps, how the network's inputs and outputs are normalized, how the per-noise-level training losses are weighted, and which noise levels the training spends its effort on.

The question is how to treat these degrees of freedom on their own terms — to study, for a given pre-trained or freshly trained model, how to set the network parameterization, the sampling discretization and solver, and the training loss, and what the best setting of each turns out to be.

## Background

**The denoising-and-marginals picture.** Take a data distribution p_data(x) with per-coordinate standard deviation σ_data. Convolving it with isotropic Gaussian noise of standard deviation σ gives a one-parameter family of mollified distributions p(x; σ) = p_data * N(0, σ²I). For σ much larger than σ_data, p(x; σ) is indistinguishable from pure Gaussian noise; for σ → 0 it is the data. The generative idea is to start from a high-σ sample (essentially noise) and walk it down to σ = 0.

**Score matching and the optimal denoiser (Hyvärinen 2005; Vincent 2011).** The object that drives this walk is the score ∇_x log p(x; σ), a vector field pointing toward higher data density at a given noise level; it does not require the intractable normalizing constant of p(x; σ). Vincent's denoising score matching connects the score to a denoising regression. If D(x; σ) is the function minimizing E_{y∼p_data} E_{n∼N(0,σ²I)} ‖D(y+n; σ) − y‖², separately at each σ, then the minimizer is the conditional mean D(x; σ) = E[y | x] (regression toward the clean signal), and the score is recovered as ∇_x log p(x; σ) = (D(x; σ) − x)/σ². For a finite training set {y_i}, the mollified density is a Gaussian mixture p(x; σ) = (1/Y)Σ_i N(x; y_i, σ²I), and the optimal denoiser is the softmax-weighted average of the data points, D(x; σ) = Σ_i N(x; y_i, σ²I) y_i / Σ_i N(x; y_i, σ²I). So a network trained to denoise *is* a score model: the two are one object viewed two ways.

**Continuous-time formulation (Song et al. 2021).** A forward stochastic differential equation dx = f(t) x dt + g(t) dω, with ω the Wiener process, turns data into noise; its perturbation kernel is N(s(t) x_0, s(t)² σ(t)² I) with s(t) = exp ∫₀ᵗ f and σ(t) = √(∫₀ᵗ g²/s²). The marginals p_t(x) of this process can be reproduced either by the time-reversed SDE (Anderson 1982) or by a deterministic **probability-flow ODE** dx = [f(t) x − ½ g(t)² ∇_x log p_t(x)] dt — both have the same p_t at every t. The deterministic ODE has only the initial noise sample as its randomness, is invertible, and lets one reason about the geometry of its solution trajectories. A general SDE that keeps the same marginals can be written as the probability-flow ODE plus a Langevin term (a deterministic score-driven decay of noise plus a fresh-noise injection whose net noise contribution cancels).

**Variance-preserving vs variance-exploding.** Two standard instantiations. Variance-preserving (the DDPM lineage, Ho et al. 2020): the forward Markov chain q(x_t|x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I) scales the signal down as it injects noise, keeping total variance roughly constant; closed form q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I) with ᾱ_t = ∏(1−β_s). Variance-exploding (the score-matching lineage, Song & Ermon 2019): the data is left unscaled and noise of growing σ is simply added. In the continuous picture these are particular choices of f, g (hence of s, σ).

**Parameterizing the network.** Standard supervised-training practice keeps network input and output magnitudes near unit variance and avoids large per-sample swings in gradient magnitude. A network asked to output the clean signal directly from x = y + n sees an input whose magnitude varies with σ. The prevailing parameterization normalizes the input by a σ-dependent factor and trains the network to predict the unit-variance noise vector, reconstructing the signal as x − σ·(network output).

**Per-noise-level loss.** The denoising loss can be inspected as a function of σ after training: it is largest at intermediate noise levels and small at the extremes. At very low σ the noise component is vanishingly small; at very high σ the best possible answer collapses toward the dataset average regardless of the input.

**Sampling steps and curvature.** Any numerical ODE solver only approximates the true trajectory; each step incurs a local truncation error that accumulates. Euler's method is first order (local error O(h²)); higher-order Runge–Kutta methods converge faster per step but cost extra network evaluations. The local error of a step can be measured directly along these trajectories, and its magnitude tracks the curvature of the trajectory.

**Stochasticity.** Sampling can be run purely deterministically (the probability-flow ODE) or with fresh noise re-injected each step (the SDE). The deterministic ODE and the noise-injecting SDE share the same marginals in the continuum.

## Baselines

- **DDPM / variance-preserving (Sohl-Dickstein et al. 2015; Ho et al. 2020).** Forward chain adds noise while shrinking the signal; reverse chain is trained by a variational bound that reduces to the simple objective ‖ε_θ(x_t, t) − ε‖² (predict the noise). Sampling is ancestral over the full chain (commonly T = 1000 steps), with the noise schedule and discretization fused into the chain.

- **NCSN / SMLD / variance-exploding (Song & Ermon 2019).** Estimate the score at a geometric ladder of noise levels via denoising score matching, then sample by annealed Langevin dynamics, running several Langevin steps at each level.

- **Score-SDE, VP and VE (Song et al. 2021).** Unifies the two above as one SDE with the probability-flow ODE as its deterministic counterpart, and trains a single continuous-σ score network. The first-class objects are the drift/diffusion coefficients f, g, from which the marginals are derived. The noise schedule σ ∝ √t corresponds to constant-speed heat diffusion. Sampling typically uses Euler on the reverse process or a predictor–corrector scheme, with the σ-dependent network normalization and noise-prediction parameterization.

- **DDIM (Song et al. 2020).** A deterministic, non-Markovian sampler that can be read as Euler integration of an ODE, implicitly using the unscaled, identity-rescaled schedule. Reaches good quality in far fewer steps than ancestral DDPM, with the step schedule inherited from the discrete chain.

- **Improved DDPM / ADM (Nichol & Dhariwal 2021; Dhariwal & Nichol 2021).** Refinements of the VP chain — a cosine ᾱ schedule, learned variances, a strong U-Net backbone (ADM) — that pushed image quality to the front of the field. Trained for a discrete set of noise levels, with the noise-prediction parameterization and the schedule tied to the discrete-chain derivation.

- **Higher-order / adaptive solvers for diffusion (Jolicoeur-Martineau et al. 2021).** Explored Heun-type and adaptive SDE solvers for diffusion sampling, showing that more accurate integrators can help.

## Evaluation settings

Datasets: CIFAR-10 at 32×32 (class-conditional and unconditional), ImageNet at 64×64 (class-conditional), and FFHQ/AFHQv2 at 64×64. Pre-trained checkpoints from the VP and VE score-SDE models and from the ADM/iDDPM ImageNet model are available and serve as drop-in score networks. Quality is measured by Fréchet inception distance (FID) between 50,000 generated images and the real images, computed with the standard Inception-v3 feature extractor; to suppress run-to-run variation FID is computed a few times and the best reported. The cost axis is reported as neural function evaluations (NFE) — the number of network calls per generated image — since the network call dominates sampling time. Trajectory accuracy can be probed directly by measuring local truncation error: take a noisy sample at one level, advance it with a single solver step, and compare against a high-resolution reference integration. Training is measured in millions of images seen, with an exponential moving average of weights used for evaluation.

## Code framework

The primitives that already exist: a U-Net backbone that maps an image plus a scalar noise-level embedding (and optional class label or augmentation label) to an image-shaped output; an optimizer and an exponential-moving-average wrapper; a data pipeline yielding clean images with optional non-leaking augmentations; and generic numerical-integration stepping. The model-specific decisions — how to wrap the raw network into a denoiser, what the loss weights and noise distribution are, how to discretize and integrate the reverse process — are left as empty slots.

```python
import torch

# --- given: a raw U-Net F_theta(input_image, noise_cond, class_labels, **kwargs) -> image-shaped output ---

class NoiseConditionedDenoiser(torch.nn.Module):
    """Wraps the raw network into a denoiser D(x; sigma)."""
    def __init__(self, model, sigma_min=0, sigma_max=float('inf'),
                 sigma_data=0.5, label_dim=0, use_fp16=False):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.label_dim = label_dim
        self.use_fp16 = use_fp16

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        # TODO: turn the raw network output into the denoiser D(x; sigma).
        pass

    def round_sigma(self, sigma):
        # TODO: choose how continuous noise levels are matched to supported ones.
        pass


class DenoisingLoss:
    """Weighted denoising-regression objective over noise levels."""
    def __init__(self, **kwargs):
        pass

    def sample_sigma(self, batch):
        # TODO: which noise levels to train on (the training sigma distribution)
        pass

    def weight(self, sigma):
        # TODO: per-noise-level loss weight lambda(sigma)
        pass

    def __call__(self, denoiser, images, labels=None, augment_pipe=None):
        # TODO: optionally augment images, add Gaussian noise, call denoiser,
        #       and return the weighted denoising residual.
        pass


def noise_schedule(denoiser, num_steps, **kwargs):
    # TODO: the decreasing sequence of supported noise levels {sigma_i}, ending at 0.
    pass


@torch.no_grad()
def sample(denoiser, latents, class_labels=None, randn_like=torch.randn_like,
           num_steps=18, **kwargs):
    # integrate the reverse process from high noise down to zero
    sigmas = noise_schedule(denoiser, num_steps, **kwargs)
    x = latents * sigmas[0]
    for i in range(num_steps):
        # TODO: the integrator step that advances x from sigmas[i] to sigmas[i+1]
        pass
    return x
```
