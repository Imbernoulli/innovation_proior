## Research question

Generative models built on iterative denoising produce excellent samples by repeatedly evaluating a neural network — tens to thousands of forward passes — because generation is implemented as the numerical solution of a differential equation that crawls from pure noise back to data along a long trajectory. Single-step generators (adversarial networks, variational autoencoders, normalizing flows) draw a sample in one forward pass: adversarial models reach high fidelity through adversarial training, while the likelihood-based one-step models are trained by maximum likelihood.

The setting: study how to generate samples from a score-based / diffusion model in few — ideally one — network evaluations, while reusing the strong backbones developed for iterative denoisers and supporting the editing tasks (inpainting, colorization, super-resolution, interpolation, stroke-guided synthesis) that iterative samplers already perform by reusing the same model on partially-observed inputs.

## Background

**Score-based / diffusion generative modeling.** Let p_data(x) be the data distribution. A forward stochastic differential equation (SDE) progressively corrupts data into noise,

    dx_t = μ(x_t, t) dt + σ(t) dw_t,  t ∈ [0, T],

with drift μ, diffusion coefficient σ, and Brownian motion w_t. Writing p_t for the marginal of x_t, one has p_0 = p_data and, by design, p_T close to a tractable Gaussian. Song et al. (2021) established that this SDE admits a deterministic counterpart — the **Probability Flow (PF) ODE** — whose trajectories have the *same* time marginals p_t:

    dx_t = [ μ(x_t, t) − ½ σ(t)² ∇ log p_t(x_t) ] dt.

Here ∇ log p_t is the **score function**. A neural network s_φ(x, t) is trained to approximate ∇ log p_t via score matching (Hyvärinen 2005; Vincent 2011; Song & Ermon 2019); substituting it into the PF ODE yields an *empirical* PF ODE that can be integrated numerically from a Gaussian sample at t = T down to t ≈ 0 to obtain a data sample.

**The EDM formulation (Karras et al. 2022).** A particularly clean choice sets μ(x, t) = 0 and σ(t) = √(2t). Then the perturbation kernel is simply p_t = p_data ⊗ N(0, t² I): adding noise at "time" t means adding Gaussian noise of standard deviation exactly t, so t *is* the noise level. The terminal distribution is π = N(0, T² I). With these choices the empirical PF ODE collapses to

    dx_t / dt = − t · s_φ(x_t, t).

Karras et al. fix T = 80 and stop the solver at a small ε = 0.002 to avoid the numerical blow-up of the score as t → 0, accepting x_ε as the sample. They integrate with **Heun's method** (a second-order predictor–corrector), which reaches competitive quality in fewer steps than first-order Euler. They discretize [ε, T] with the schedule

    t_i = ( ε^{1/ρ} + (i−1)/(N−1) · ( T^{1/ρ} − ε^{1/ρ} ) )^ρ,   ρ = 7,

which places more steps at small noise levels. They also precondition the network: instead of predicting the clean signal directly, the denoiser is written

    D_θ(x, σ) = c_skip(σ) x + c_out(σ) F_θ( c_in(σ) x, c_noise(σ) ),

with c_skip(σ) = σ_data² / (σ² + σ_data²), c_out(σ) = σ·σ_data / √(σ² + σ_data²), c_in(σ) = 1/√(σ² + σ_data²), σ_data = 0.5, and c_noise = ¼ ln σ. The skip term lets the network output a correction whose magnitude is well-scaled at every noise level.

**Cost of iterative sampling.** Each solver step is one (or, for Heun, two) evaluations of s_φ. The best ODE solvers use more than ten steps for competitive samples; the SDE samplers use many more.

**Facts about the setting.** (i) Faster numerical solvers (DDIM, DPM-solver, GENIE) reduce the step count — they use at least about ten evaluations. (ii) Distillation methods first generate a dataset of teacher samples and then regress a one-step student (Luhman & Luhman 2021; Zheng et al. 2022). (iii) Because μ = 0 and the kernel is Gaussian, the PF ODE is deterministic: for each noise sample at t = T there is a *single* trajectory ending at a *single* data point.

## Baselines

**Score-based diffusion sampling (Song et al. 2021; Karras et al. 2022).** Train s_φ ≈ ∇ log p_t, then integrate the PF ODE (Euler/Heun) or the reverse SDE from noise to data, one network evaluation per (Euler) step.

**Knowledge distillation by offline regression (Luhman & Luhman 2021; Zheng et al. 2022).** Run the full teacher sampler to produce many (noise → sample) pairs, then train a one-step student to regress the mapping. One network evaluation at inference.

**Progressive distillation (Salimans & Ho 2022).** Train a student to reproduce, in **one** DDIM step, what the teacher does in **two** DDIM steps; then make the student the new teacher and repeat, halving the step count each round (… → 4 → 2 → 1). Each step's targets are computed on the fly from the current teacher, with no pre-collected sample dataset.

**Single-step likelihood/adversarial models (GAN, VAE, normalizing flow).** One forward pass to sample. GANs use adversarial training; VAEs are trained by a variational bound; normalizing flows use an invertible architecture trained by exact likelihood.

## Evaluation settings

- **Datasets:** CIFAR-10 (32×32), ImageNet 64×64, LSUN Bedroom and Cat at 256×256.
- **Metrics:** Fréchet Inception Distance (FID, primary) and Inception Score; sample quality reported as a function of the number of network evaluations (one-step, two-step, few-step). Negative log-likelihood is not the target.
- **Protocol:** pixel values rescaled to [−1, 1]; the budget axis is *number of function evaluations* (NFE) per sample, the compute/quality tradeoff. For editing tasks: inpainting, colorization, super-resolution, denoising, interpolation, and stroke-guided synthesis on the LSUN images, evaluated without task-specific training.
- **Backbones:** the NCSN++ architecture for 32×32, and the ADM/U-Net architectures for 64×64 and 256×256 — the same networks already used for score-based denoisers.

## Code framework

An EDM-style harness already has denoiser preconditioning, the rho = 7 noise grid, Euler/Heun steps for the empirical PF ODE, U-Net backbones, EMA utilities, and a minibatch training loop. The fast generator, the objective that trains it, and the sampler that runs it are left open.

```python
import torch as th
import torch.nn.functional as F


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def mean_flat(x):
    return x.flatten(1).mean(1)


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "uniform":
        return th.ones_like(snrs)
    if weight_schedule == "karras":
        return snrs + 1.0 / sigma_data**2
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1
    if weight_schedule == "truncated-snr":
        return th.clamp(snrs, min=1.0)
    raise NotImplementedError(weight_schedule)


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_edm_scalings(self, sigmas):
        c_skip = self.sigma_data**2 / (sigmas**2 + self.sigma_data**2)
        c_out = sigmas * self.sigma_data / (sigmas**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigmas**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_fast_sampler_scalings(self, sigmas):
        # TODO: choose any additional preconditioning scalings required by the fast sampler
        pass

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(v, x_t.ndim) for v in self.get_edm_scalings(sigmas)
            ]
            rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
            model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
            return model_output, c_out * model_output + c_skip * x_t

        # TODO: fill the fast sampler parameterization
        pass

    def fast_sampler_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        # TODO: define the training objective for the fast generator
        pass


def sample_fast_once(denoiser, x, sigmas, generator=None):
    # TODO: evaluate the fast generator once
    pass


def iterative_refinement_sampler(denoiser, x, sigmas, generator, ts, sigma_min, sigma_max, rho, steps):
    # TODO: trade extra compute for quality across multiple evaluations
    pass


def create_ema_and_scales_fn(
    target_ema_mode, start_ema, scale_mode, start_scales, end_scales, total_steps, distill_steps_per_iter
):
    # TODO: return the target EMA rate and number of noise scales for a train step
    pass


def ema_update(target_params, params, mu):
    for tp, p in zip(target_params, params):
        tp.detach().mul_(mu).add_(p, alpha=1 - mu)
```
