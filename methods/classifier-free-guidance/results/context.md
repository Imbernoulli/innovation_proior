## Research question

We have a conditional generative model of images — concretely a diffusion model that, given a class label `c`, can sample `x ~ p(x | c)`. The samples are diverse and cover the conditional distribution well, but individually they are often mediocre: a fraction look blurry, malformed, or off-class. Other families of generative models offer a knob at *sampling time* that trades diversity for per-sample fidelity, and we would like the same for diffusion.

The pain point is concrete. In GANs, the **truncation trick** resamples the latent from a truncated normal — shrinking the range of the input noise — which sharply raises Inception Score (perceptual quality) at the cost of recall/diversity, tracing out a smooth fidelity↔diversity curve as the truncation threshold is varied. Flow-based models get the same effect from **low-temperature sampling**: scaling the std of the base distribution by `T < 1`. Diffusion models have no equivalent. The obvious analogues do not work: multiplying the model's predicted score (or noise) by a constant, or shrinking the variance of the Gaussian noise injected during the reverse process, both produce blurry, washed-out, low-quality samples rather than sharp, high-fidelity ones.

So the question is: **is there a sampling-time knob for a conditional diffusion model that, by sweeping a single scalar, trades sample diversity for sample fidelity** — reproducing the IS/FID tradeoff curve that truncation gives GANs — and can it be obtained cheaply, without adding machinery that complicates the training pipeline?

## Background

**Diffusion models in continuous time.** Take data `x ~ p(x)` and a variance-preserving forward (noising) process indexed by a log-signal-to-noise ratio `λ` (running from `λ_max`, nearly clean, down to `λ_min`, nearly pure noise). The forward marginals are Gaussian:
`q(z_λ | x) = N(α_λ x, σ_λ² I)`, with `α_λ² = sigmoid(λ)` and `σ_λ² = 1 − α_λ²`, so `λ = log(α_λ²/σ_λ²)` is exactly the log-SNR. A sample at level `λ` is `z_λ = α_λ x + σ_λ ε`, `ε ~ N(0, I)`. Conditioned on `x`, the process can be run in reverse through tractable Gaussian transitions `q(z_{λ'} | z_λ, x)` with closed-form mean `μ̃_{λ'|λ}(z_λ, x)` and variance.

**ε-prediction and the training objective.** The reverse-process generative model needs, at each step, an estimate of the clean `x`. We parameterize it through **noise prediction**: a network `ε_θ(z_λ)` (also fed `λ`) predicts the noise that was added, and `x_θ(z_λ) = (z_λ − σ_λ ε_θ(z_λ)) / α_λ` is plugged into the posterior to form the reverse mean. Training is plain regression on the noise:
`E_{ε, λ} ‖ ε_θ(z_λ) − ε ‖²`, with `ε ~ N(0,I)`, `z_λ = α_λ x + σ_λ ε`, and `λ ~ p(λ)`.

**The score interpretation (the load-bearing fact).** This objective is denoising score matching (Vincent 2011; Hyvärinen & Dayan 2005) applied at every noise level (Song & Ermon 2019). Its consequence is that the trained noise predictor is, up to a known scale, the gradient of the log-density of the noisy data:
`ε_θ(z_λ) ≈ −σ_λ ∇_{z_λ} log p(z_λ)`,
i.e. the **score** `∇_{z_λ} log p(z_λ) ≈ −ε_θ(z_λ) / σ_λ`. Sampling then resembles Langevin dynamics walking up `∇ log p` across a sequence of noise scales (Song et al. 2020). One caveat: `ε_θ` is an unconstrained neural network, so it is not necessarily the gradient of any scalar potential — the learned vector field need not be conservative.

**The conditional case.** For class-conditional generation, the data `x` is drawn jointly with conditioning `c`. The only change is that the network also receives `c`: `ε_θ(z_λ, c) ≈ −σ_λ ∇_{z_λ} log p(z_λ | c)`. Everything above carries over with `p(·)` replaced by `p(· | c)`.

**Bayes' rule relates the conditional, unconditional, and the posterior over labels.** For any joint `p(x, c)`, `p(c | x) = p(x | c) p(c) / p(x)`. Taking the log and the gradient in `x`, the `p(c)` term is constant in `x` and drops:
`∇_x log p(c | x) = ∇_x log p(x | c) − ∇_x log p(x)`.
This identity — the gradient of a *classifier's* log-posterior equals the difference between the conditional and unconditional *generative* scores — is the standard bridge between discriminative guidance and generative score models.

**The diagnostic that motivates the problem.** Naive truncation analogues in diffusion (scaling scores, shrinking reverse-process noise) have been measured to degrade quality rather than sharpen it. This is the empirical wall: diffusion lacks a working fidelity knob, even though GANs and flows have one.

## Baselines

**GAN truncation / flow low-temperature sampling.** Not diffusion methods, but the behavior to be matched. Truncating the GAN latent or lowering the flow temperature concentrates sampling on high-density regions; sweeping the parameter yields a monotone IS-up / diversity-down tradeoff and a characteristic IS/FID frontier. *Gap for us:* these rely on the model's explicit, well-behaved latent geometry; diffusion's iterative denoising has no single latent whose variance can simply be shrunk.

**Classifier guidance (Dhariwal & Nichol 2021).** The first method to give diffusion a truncation-like knob. Train, in addition to the diffusion model, a separate **classifier** `p_φ(c | z_λ)` on noisy images, and at sampling time push the score toward higher class-probability:
`ε̃_θ(z_λ, c) = ε_θ(z_λ, c) − w σ_λ ∇_{z_λ} log p_φ(c | z_λ) ≈ −σ_λ ∇_{z_λ}[ log p(z_λ | c) + w log p_φ(c | z_λ) ]`.
This samples from the tilted distribution `p̃(z_λ | c) ∝ p(z_λ | c) · p_φ(c | z_λ)^w`. Increasing `w > 0` up-weights inputs the classifier is confident about, raising IS and lowering diversity — exactly the desired tradeoff, and verified to reproduce a truncation-like IS/FID curve. *Core algorithm:* during each reverse step, evaluate the diffusion score and the classifier gradient, add them with weight `w`, and proceed with the usual sampler. *Gaps it leaves open:*
1. It needs an **extra classifier**, and that classifier must be trained on **noisy** inputs `z_λ` at every noise level — a standard pretrained classifier cannot be dropped in. This complicates the training pipeline.
2. Sampling steps along `∇_{z_λ} log p_φ(c | z_λ)` look like a **gradient-based adversarial perturbation** of the classifier, raising the worry that the gains on classifier-based metrics (IS, FID, which are themselves built on Inception-network features) are partly an artifact of fooling those networks rather than genuinely better images.
3. (Noted by the same authors:) applying classifier guidance with weight `w+1` to an *unconditional* model is algebraically equivalent to weight `w` on a conditional model, since `p(z|c) p(c|z)^w ∝ p(z) p(c|z)^{w+1}`; in practice guiding the already-conditional model works best.

## Evaluation settings

The natural yardstick is class-conditional ImageNet (Russakovsky et al. 2015) at downsampled resolutions — `64×64` and `128×128` — the standard setting for studying fidelity/diversity tradeoffs since the BigGAN paper (Brock et al. 2019). Sample quality is measured by **Fréchet Inception Distance** (FID, lower is better; Heusel et al. 2017) for distributional match, and **Inception Score** (IS, higher is better; Salimans et al. 2016) for per-sample perceptual quality and confidence. The protocol computes both from 50,000 generated samples. Comparison points are the truncation curves of BigGAN-deep and prior diffusion results (e.g. cascaded diffusion, classifier-guided diffusion). Knobs that exist at evaluation time include the number of reverse-sampling steps `T` and the sampler used (ancestral, or a deterministic variant such as DDIM).

## Code framework

The primitives that already exist: a continuous-time variance-preserving diffusion process with log-SNR schedule, an ε-prediction network, the denoising MSE loss, and an ancestral reverse-process sampler. The conditional network takes a class label. The open places are how the label is handled during training and how the per-step noise estimate is formed at sampling time.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- existing diffusion primitives (variance-preserving, log-SNR parameterization) ---

def log_snr_schedule(t):
    # maps t in [0,1] to log-SNR lambda; cosine-style schedule
    ...

def alpha_sigma(lam):
    # alpha_lambda^2 = sigmoid(lambda), sigma_lambda^2 = 1 - alpha^2
    alpha_sq = torch.sigmoid(lam)
    return alpha_sq.sqrt(), (1 - alpha_sq).sqrt()

def expand_to_data(v, x):
    # reshape a scalar or batch vector so it broadcasts over image/audio dimensions
    if v.ndim == 0:
        v = v.expand(x.shape[0])
    return v.view(-1, *([1] * (x.ndim - 1)))

def sample_log_snr(batch, device):
    # sample λ from the existing log-SNR schedule distribution
    ...

class NoisePredictor(nn.Module):
    """An ε-prediction network ε_θ(z_λ, λ, c). Conditioning c is fed in alongside z and λ."""
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, z, lam, c):
        # returns predicted noise of shape like z
        ...

# --- training: corrupt, predict noise, regress (denoising score matching) ---

def diffusion_loss(model, x, c):
    lam = sample_log_snr(x.shape[0], x.device)    # λ ~ p(λ)
    eps = torch.randn_like(x)
    alpha, sigma = alpha_sigma(expand_to_data(lam, x))
    z = alpha * x + sigma * eps                   # z_λ = α_λ x + σ_λ ε
    # TODO: choose how labels are presented to the model during training.
    eps_pred = model(z, lam, c)
    return F.mse_loss(eps_pred, eps)

# --- sampling: form the per-step noise estimate, then take an ancestral reverse step ---

def predict_x_from_eps(z, lam, eps_hat):
    alpha, sigma = alpha_sigma(expand_to_data(lam, z))
    return (z - sigma * eps_hat) / alpha          # x_θ = (z_λ − σ_λ ε̂)/α_λ

def reverse_step(z, lam, lam_next, x_pred, v):
    # ancestral transition q(z_{λ'} | z_λ, x_pred) with log-space variance interpolation
    ...

def guided_eps(model, z, lam, c, w):
    # TODO: construct the noise estimate used by the sampler.
    pass

@torch.no_grad()
def sample(model, c, schedule, w, shape, v):
    z = torch.randn(shape, device=c.device)       # z ~ N(0, I)
    x_pred = None
    for lam, lam_next in schedule:
        eps_hat = guided_eps(model, z, lam, c, w)
        x_pred = predict_x_from_eps(z, lam, eps_hat)
        z = reverse_step(z, lam, lam_next, x_pred, v)
    return x_pred
```
