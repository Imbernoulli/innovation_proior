# Context

## Research question

We want a generative model for images that turns pure Gaussian noise into samples from a complicated data distribution, and we want it to be good on three axes at once: sample quality (low FID), training cost, and, increasingly the bottleneck, generation speed, i.e. how many times we have to evaluate a large neural network to produce one image.

A family of models built around iteratively denoising a noisy image has emerged as the leading approach, often matching or beating GANs. But the family is a mess to reason about. Each member is derived from its own theoretical scaffold, such as a discrete Markov chain trained by a variational bound, a ladder of score estimates trained by score matching, or a continuous stochastic differential equation, and each derivation tangles together a long list of choices that are, on closer inspection, logically independent of one another: the schedule of noise levels, whether and how the image is rescaled as noise is added, how the continuous process is discretized into sampling steps, which numerical solver advances those steps, how the network's inputs and outputs are normalized, how the per-noise-level training losses are weighted, and which noise levels the training even spends its effort on. Because these choices arrive welded to a particular theory, a model looks like a package where touching one piece appears to break everything.

The precise problem: separate these degrees of freedom from the theory that introduced them, so that each can be set on its own merits, and then determine what the best setting of each actually is — for the network parameterization, for the sampling discretization and solver, for the training loss. A solution would have to show that, e.g., the sampler is genuinely independent of how the network was trained (a sampler improvement should drop into any pre-trained model and help), and it would have to derive the parameterization choices from stated principles rather than inheriting them.

## Background

**The denoising-and-marginals picture.** Take a data distribution p_data(x) with per-coordinate standard deviation σ_data. Convolving it with isotropic Gaussian noise of standard deviation σ gives a one-parameter family of mollified distributions p(x; σ) = p_data * N(0, σ²I). For σ much larger than σ_data, p(x; σ) is indistinguishable from pure Gaussian noise; for σ → 0 it is the data. The generative idea is to start from a high-σ sample (essentially noise) and walk it down to σ = 0.

**Score matching and the optimal denoiser (Hyvärinen 2005; Vincent 2011).** The object that drives this walk is the score ∇_x log p(x; σ), a vector field pointing toward higher data density at a given noise level; crucially it does not require the intractable normalizing constant of p(x; σ). Vincent's denoising score matching connects the score to a denoising regression. If D(x; σ) is the function minimizing E_{y∼p_data} E_{n∼N(0,σ²I)} ‖D(y+n; σ) − y‖², separately at each σ, then the minimizer is the conditional mean D(x; σ) = E[y | x] (regression toward the clean signal), and the score is recovered as ∇_x log p(x; σ) = (D(x; σ) − x)/σ². For a finite training set {y_i}, the mollified density is a Gaussian mixture p(x; σ) = (1/Y)Σ_i N(x; y_i, σ²I), and the optimal denoiser is the softmax-weighted average of the data points, D(x; σ) = Σ_i N(x; y_i, σ²I) y_i / Σ_i N(x; y_i, σ²I). So a network trained to denoise *is* a score model: the two are one object viewed two ways.

**Continuous-time formulation (Song et al. 2021).** A forward stochastic differential equation dx = f(t) x dt + g(t) dω, with ω the Wiener process, turns data into noise; its perturbation kernel is N(s(t) x_0, s(t)² σ(t)² I) with s(t) = exp ∫₀ᵗ f and σ(t) = √(∫₀ᵗ g²/s²). The marginals p_t(x) of this process can be reproduced either by the time-reversed SDE (Anderson 1982) or by a deterministic **probability-flow ODE** dx = [f(t) x − ½ g(t)² ∇_x log p_t(x)] dt — both have the same p_t at every t. The deterministic ODE is attractive for analysis: its only randomness is the initial noise sample, it is invertible, and one can reason about the geometry of its solution trajectories. A general SDE that keeps the same marginals can be written as the probability-flow ODE plus a Langevin term (a deterministic score-driven decay of noise plus a fresh-noise injection whose net noise contribution cancels).

**Variance-preserving vs variance-exploding.** Two standard instantiations. Variance-preserving (the DDPM lineage, Ho et al. 2020): the forward Markov chain q(x_t|x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I) scales the signal down as it injects noise, keeping total variance roughly constant; closed form q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I) with ᾱ_t = ∏(1−β_s). Variance-exploding (the score-matching lineage, Song & Ermon 2019): the data is left unscaled and noise of growing σ is simply added. In the continuous picture these are particular choices of f, g (hence of s, σ).

**A diagnostic about parameterizing the network.** Standard supervised-training wisdom says to keep network input and output magnitudes near unit variance and to avoid large per-sample swings in gradient magnitude. A network asked to output the clean signal directly from x = y + n sees an input whose magnitude varies enormously with σ, which violates this. The prevailing fix normalizes the input by a σ-dependent factor and trains the network to predict the unit-variance noise vector, reconstructing the signal as x − σ·(network output). But this has an observable failure mode: at large σ the network must finely tune its output to cancel the actual noise present, and any error it makes is then multiplied by σ — error amplification grows without bound as σ increases. At those noise levels it would plainly be easier to predict the expected clean output directly. This is a pre-method observation about the existing parameterization, independent of any particular fix.

**A diagnostic about per-noise-level loss.** Inspecting the denoising loss as a function of σ after training reveals it is largest at intermediate noise levels and small at the extremes: at very low σ the noise component is vanishingly small, so it is both easy and unimportant to resolve; at very high σ the best possible answer collapses toward the dataset average regardless of the input. So training effort spent at the extreme σ is mostly wasted.

**A diagnostic about sampling steps and curvature.** Any numerical ODE solver only approximates the true trajectory; each step incurs a local truncation error that accumulates. Euler's method is first order (local error O(h²)); higher-order Runge–Kutta methods converge faster per step but cost extra network evaluations. Empirically, the local error of an Euler step on these trajectories is large at low σ and small at high σ, and is nearly independent of the particular sample — suggesting the step schedule should shrink steps as σ decreases and need not be chosen per-sample. The magnitude of truncation error tracks the curvature of the trajectory.

**A diagnostic about stochasticity.** Purely deterministic sampling is convenient (invertible, few steps) but tends to give slightly worse quality than sampling that re-injects fresh noise each step. Since the deterministic ODE and the noise-injecting SDE share the same marginals in the continuum, this quality gap is a discretization-level effect rather than a difference in the underlying distribution. Excessive noise replacement, however, is observed to gradually wash out image detail and drift colors toward oversaturation.

## Baselines

- **DDPM / variance-preserving (Sohl-Dickstein et al. 2015; Ho et al. 2020).** Forward chain adds noise while shrinking the signal; reverse chain is trained by a variational bound that reduces to the simple objective ‖ε_θ(x_t, t) − ε‖² (predict the noise). Sampling is ancestral over the full chain (commonly T = 1000 steps). Gap: hundreds to thousands of network evaluations per image; the noise schedule and the discretization are fused into the chain and not freely adjustable.

- **NCSN / SMLD / variance-exploding (Song & Ermon 2019).** Estimate the score at a geometric ladder of noise levels via denoising score matching, then sample by annealed Langevin dynamics, running several Langevin steps at each level. Gap: many steps, sensitive level/step tuning, and like DDPM the schedule is entangled with the rest.

- **Score-SDE, VP and VE (Song et al. 2021).** Unifies the two above as one SDE with the probability-flow ODE as its deterministic counterpart, and trains a single continuous-σ score network. The "first-class" objects are the drift/diffusion coefficients f, g, from which the marginals are only indirectly derived. The noise schedule is chosen for theoretical tidiness (σ ∝ √t corresponds to constant-speed heat diffusion). Sampling typically uses Euler on the reverse process or a predictor–corrector scheme. Gaps: the practically important objects (the marginals) are buried behind f, g; the schedule rests on convenience rather than on sampling cost; the σ-dependent network normalization and noise-prediction parameterization carry the error-amplification failure mode above; sampling still uses first-order steps.

- **DDIM (Song et al. 2020).** A deterministic, non-Markovian sampler that can be read as Euler integration of an ODE, implicitly using the unscaled, identity-rescaled schedule. Reaches good quality in far fewer steps than ancestral DDPM. Gap: still a first-order solver, and the step schedule is inherited from the discrete chain rather than chosen to minimize trajectory curvature.

- **Improved DDPM / ADM (Nichol & Dhariwal 2021; Dhariwal & Nichol 2021).** Refinements of the VP chain — a cosine ᾱ schedule, learned variances, a strong U-Net backbone (ADM) — that pushed image quality to the front of the field. Trained for a discrete set of noise levels. Gap: many sampling steps; the noise-prediction parameterization and the schedule are still tied to the discrete-chain derivation.

- **Higher-order / adaptive solvers for diffusion (Jolicoeur-Martineau et al. 2021).** Explored Heun-type and adaptive SDE solvers for diffusion sampling, establishing that more accurate integrators can help. Gap: adaptive general-purpose solvers carry overhead, and a tailored fixed-schedule procedure had not been pinned down.

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
