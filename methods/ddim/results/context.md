# Context

## Research question

A class of iterative generative models — trained by denoising data at many Gaussian noise levels and sampling by progressively denoising white noise back into an image — has reached sample quality comparable to GANs with a likelihood-based, non-adversarial, stable objective. Sampling runs the reverse of a forward noising chain that may have on the order of a thousand steps, and producing a single image requires running every one of those steps sequentially, each step a full forward pass through a large network. On a single GPU this is roughly twenty hours to draw fifty thousand 32×32 images, versus under a minute for a GAN that needs one pass; at 256×256 it approaches a thousand hours. The chain is long because each reverse step is modeled as Gaussian, which is accurate only when the per-step noise is small.

The question is how to draw samples from these trained models using far fewer sequential network passes — a few tens rather than a thousand — given a network that has already been trained with the existing denoising objective.

## Background

A latent-variable generative model of this family writes p_θ(x_0) = ∫ p_θ(x_{0:T}) dx_{1:T} with latents x_1,…,x_T living in the same space as the data x_0. A *fixed* (non-trainable) inference process q corrupts data into noise; a *generative* process p_θ runs the other way, denoising noise into data. Parameters are fit by maximizing a variational lower bound, E_q[log p_θ(x_0)] ≥ E_q[log p_θ(x_{0:T}) − log q(x_{1:T}|x_0)].

The concrete inference process in use is a Gaussian Markov chain with a schedule α_{1:T} decreasing in (0,1]. (Throughout, α_t denotes the *cumulative* signal-retention coefficient; the chain's per-step variances are derived from it as α_t/α_{t-1}.) The defining property is the closed-form marginal q(x_t|x_0) = N(√α_t x_0, (1−α_t)I), i.e. x_t = √α_t x_0 + √(1−α_t) ε with ε ∼ N(0,I). As α_T → 0 this marginal converges to a standard Gaussian regardless of x_0, so the generative prior is set to p_θ(x_T) = N(0,I).

When the reverse conditionals are modeled as Gaussians with trainable means and fixed variances, the variational bound simplifies, term by term, to a weighted denoising regression. With a network ε_θ^{(t)} predicting the noise added to x_t,

  L_γ(ε_θ) = Σ_{t=1}^T γ_t · E_{x_0, ε} [ ‖ ε_θ^{(t)}(√α_t x_0 + √(1−α_t) ε) − ε ‖² ],

where the per-step weights γ_t are positive and depend on the schedule. The variance-reduced derivation that produces this form conditions the forward posterior on x_0: q(x_{t-1}|x_t,x_0) is Gaussian, each bound term becomes a KL between two Gaussians, and with matched variances that KL is a squared distance between means, which the noise-prediction parameterization turns into the ε-MSE above. Training at γ = 1 (an unweighted MSE) is the setting that gives the best samples; it is also the objective used by the score-matching line.

The loss L_γ is a sum of per-t terms, each an expectation under a single draw √α_t x_0 + √(1−α_t) ε; the network may or may not share parameters across t, and the schedule enters each term through the weight γ_t.

The length T is large (≈1000) for a variational reason: a large T keeps each true reverse conditional close to Gaussian, so modeling it with a Gaussian is accurate. All T steps run sequentially to draw one sample.

The score-matching relatives of this model learn the score of the data at multiple noise scales and sample by annealed Langevin dynamics — many small noisy steps. The two families are close: both train a denoising autoencoder across noise levels, and both sample by a Langevin-like procedure. Langevin dynamics is a discretization of a gradient flow.

Standard background facts available: a Gaussian product/marginalization identity (for a linear-Gaussian chain x_{t-1} → x_t with a Gaussian prior on x_t given x_0, the marginal of x_{t-1} is Gaussian with mean and covariance obtained by propagating the affine map and adding variances); the equivalence, for Gaussian corruption, between predicting the added noise and predicting the score of the noised data (denoising score matching), so a trained noise predictor is, up to a known positive scale, a learned score; the notion of an *implicit* generative model, whose samples are a deterministic pushforward of latent noise through a fixed procedure (no explicit per-step density, as in a GAN); and the probability-flow/neural-ODE viewpoint, in which a continuous-time generative process is an ODE whose Euler discretization is a sampler, fewer steps meaning a coarser discretization.

## Baselines

**Ancestral-sampling Gaussian diffusion (Sohl-Dickstein et al. 2015; Ho et al. 2020).** Fixed Gaussian forward Markov chain with the marginals above; learned Gaussian reverse chain trained by the variational bound, parameterized as noise prediction, optimized with the unweighted ε-MSE objective. The reverse step is x_{t-1} = (1/√(α_t/α_{t-1}))·(x_t − ((1−α_t/α_{t-1})/√(1−α_t)) ε_θ(x_t,t)) + σ_t z with fresh noise z each step. The generative chain is the reverse of the full-length forward chain — T sequential network passes per sample — and is stochastic, drawing fresh z at every step.

**Noise-conditional score networks with annealed Langevin sampling (Song & Ermon 2019).** One noise-conditional network learns the score at a geometric ladder of Gaussian noise scales via denoising score matching; sampling walks the ladder from high to low noise with Langevin updates x ← x + (δ/2)·s_θ(x,σ) + √δ z. Same denoising-over-noise-levels objective as Gaussian diffusion. Sampling is many small Langevin steps with hand-tuned step sizes and schedule.

**GANs (Goodfellow et al. 2014; Brock et al. 2018; Karras et al. 2018).** A generator pushes latent noise through one network pass to an image; an implicit model with no explicit likelihood, trained by adversarial min-max optimization. Sampling is a single forward pass. The map is deterministic in the latent, supporting semantic interpolation.

## Evaluation settings

Image datasets at several resolutions: CIFAR10 32×32 (unconditional), CelebA 64×64, LSUN Bedroom 256×256, LSUN Church 256×256. Sample quality is measured by Fréchet Inception Distance (Heusel et al. 2017) over fifty thousand generated samples. The noise-predictor architecture is a U-Net (Ronneberger et al. 2015) built on Wide-ResNet blocks (Zagoruyko & Komorodski 2016), trained with the unweighted ε-MSE objective at T = 1000; the same trained network is held fixed across all comparisons, with only the sampling procedure varied. Wall-clock sampling time on a single GPU is also reported: on the order of twenty hours to draw fifty thousand 32×32 images by the full sequential chain, against under a minute for a single-pass model, with the gap widening sharply at higher resolution.

## Code framework

The machinery below already exists: a network that predicts the noise in a corrupted image conditioned on the timestep, the schedule of cumulative coefficients, the closed-form one-shot corruption x_t = √α_t x_0 + √(1−α_t) ε, the unweighted ε-MSE training loop, and the existing full-length sequential sampler. The open slot is a generative procedure that turns an initial noise tensor into samples from the trained noise predictor.

```python
import torch

# Pretrained noise predictor: model(x_t, t) -> predicted noise eps.
# Schedule buffers: alphas[t] = cumulative signal coefficient, alpha_0 := 1.

def alpha_at(alphas, t):
    # cumulative-coefficient lookup with alpha_0 = 1 for the t = -1 sentinel
    pass  # TODO

def q_sample(x0, t, alphas, eps):
    # one-shot corruption: x_t = sqrt(alpha_t) x0 + sqrt(1 - alpha_t) eps
    a = alpha_at(alphas, t)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

def training_loss(model, x0, alphas):
    # unweighted noise-prediction MSE over a random timestep
    t = torch.randint(0, len(alphas), (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, alphas, eps)
    return ((model(xt, t.float()) - eps) ** 2).mean()

@torch.no_grad()
def sample(model, alphas, x):
    # TODO: generative procedure that turns the initial noise tensor x into a sample
    #       from the trained noise predictor.
    pass

# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
# ema, training loop over batches calling training_loss / opt.step / ema.update
```
