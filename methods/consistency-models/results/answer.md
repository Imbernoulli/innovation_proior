# Consistency Models

## Problem

Score-based diffusion models generate by numerically integrating the probability-flow (PF) ODE from noise to data, costing many sequential network evaluations. Consistency models replace that trajectory-crawling with a single network call that maps any point on a PF-ODE trajectory directly to the trajectory's origin, while still supporting a few-step quality/compute tradeoff and zero-shot editing, and without adversarial training.

## Key idea

Use the EDM Gaussian perturbation: the SDE drift is μ = 0 and its diffusion coefficient is √(2t), so the marginal at noise level t is p_t = p_data ⊗ N(0, t² I) and the empirical PF ODE is

    dx_t/dt = − t · s(x_t, t),   t ∈ [ε, T],  T = 80, ε = 0.002.

Each trajectory deterministically connects an endpoint x_T (noise) to an origin x_ε (≈ data). Define the **consistency function** f(x_t, t) = x_ε. It satisfies:

- **Self-consistency:** f(x_t, t) = f(x_{t'}, t') for any two points on the same trajectory.
- **Boundary condition:** f(x_ε, ε) = x_ε (identity at ε). This both pins the target and excludes the trivial constant solution.

**Parameterization (boundary for free).** Use a skip connection
f_θ(x, t) = c_skip(t) x + c_out(t) F_θ(x, t) with c_skip(ε) = 1, c_out(ε) = 0. Concretely (modified EDM):

    c_skip(t) = σ_data² / ((t − ε)² + σ_data²),   c_out(t) = σ_data (t − ε) / √(σ_data² + t²),   σ_data = 0.5,

with input scaling c_in(t) = 1/√(σ_data² + t²) and a log-noise conditioning. Differentiable everywhere, EDM-architecture compatible.

## Training

Discretize [ε, T] with the EDM ρ = 7 grid t_i = (ε^{1/ρ} + (i−1)/(N−1)(T^{1/ρ} − ε^{1/ρ}))^ρ. Sample x ~ p_data, level index n, noise z ~ N(0,I); place x_{t_{n+1}} = x + t_{n+1} z. Produce an adjacent point on the same trajectory by one ODE step, then match the model's outputs:

    L = E[ λ(t_n) · d( f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}(x̂_{t_n}, t_n) ) ],

with θ⁻ an EMA stop-gradient target, θ⁻ ← stopgrad(μ θ⁻ + (1−μ) θ); metric d ∈ {ℓ2, ℓ1, LPIPS}; λ ≡ 1.

- **Consistency Distillation (CD):** the adjacent point uses a pretrained score s_φ via a numerical solver (Heun),
  x̂_{t_n} = x_{t_{n+1}} + (t_n − t_{n+1}) Φ(x_{t_{n+1}}, t_{n+1}; φ). If the loss reaches 0 and f_θ is Lipschitz, the learned model converges to the true consistency function at the solver's order: sup‖f_θ − f‖ = O(Δt^p).

- **Consistency Training (CT):** no pretrained model. Using the exact score identity ∇ log p_t(x_t) = −E[(x_t − x)/t² | x_t], the single-sample estimate −(x_t − x)/t² replaces s_φ. The Euler step then collapses the adjacent pair to the same x noised by the same z to two levels, giving

      L_CT = E[ λ(t_n) · d( f_θ(x + t_{n+1} z, t_{n+1}), f_{θ⁻}(x + t_n z, t_n) ) ].

  A Taylor expansion shows L_CD = L_CT + o(Δt) (with an exact teacher and Euler solver), and L_CT ≥ O(Δt) dominates the remainder, so as Δt → 0 the two objectives coincide. CT anneals the level count N(k) upward (low-variance/high-bias early → low-bias late) and ties μ(k) = exp(s_0 ln μ_0 / N(k)).

Continuous-time limits exist but split by target handling. For CD with θ⁻ = θ and squared ℓ2, the true limiting loss is E[(λ/((τ⁻¹)')²)‖∂f_θ/∂t − t (∂f_θ/∂x) s_φ‖²], the squared violation of self-consistency along the ODE field. With stop-gradient targets, CD and CT instead yield pseudo-objectives whose gradients match the scaled discrete gradients; CT replaces the PF-ODE velocity −t s_φ with +(x_t − x)/t. These limits remove the grid but require Jacobian-vector products.

## Sampling

- **One step:** x̂_T ~ N(0, T² I); return f_θ(x̂_T, T).
- **Multistep (compute ↔ quality):** x ← f_θ(x̂_T, T); for decreasing τ_1 > τ_2 > …: z ~ N(0,I), x̂_{τ_n} = x + √(τ_n² − ε²) z, x ← f_θ(x̂_{τ_n}, τ_n). The τ's are chosen by greedy ternary search on sample quality. The same add-noise/denoise loop, with masking, yields zero-shot inpainting, colorization, super-resolution, interpolation, and stroke-guided synthesis.

## Code

```python
import numpy as np
import torch as th
import torch.nn.functional as F
from piq import LPIPS


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def mean_flat(x):
    return x.flatten(1).mean(1)


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1
    if weight_schedule == "karras":
        return snrs + 1.0 / sigma_data**2
    if weight_schedule == "truncated-snr":
        return th.clamp(snrs, min=1.0)
    if weight_schedule == "uniform":
        return th.ones_like(snrs)
    raise NotImplementedError(weight_schedule)


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.lpips_loss = LPIPS(replace_pooling=True, reduction="none") if loss_norm == "lpips" else None
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        scalings = self.get_scalings_for_boundary_condition if self.distillation else self.get_scalings
        c_skip, c_out, c_in = [append_dims(v, x_t.ndim) for v in scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def consistency_losses(
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
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            raise NotImplementedError("Must have a target model")

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_model, x, t, **model_kwargs)[1]

        if teacher_model is not None:
            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(samples, next_t)
            next_d = (samples - denoiser) / append_dims(next_t, dims)
            return x + (d + next_d) * append_dims((next_t - t) / 2, dims)

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            return x + d * append_dims(next_t - t, dims)

        indices = th.randint(0, num_scales - 1, (x_start.shape[0],), device=x_start.device)
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho
        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t)

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2).detach()

        weights = get_weightings(self.weight_schedule, self.get_snr(t), self.sigma_data)
        if self.loss_norm == "l1":
            loss = mean_flat((distiller - distiller_target).abs()) * weights
        elif self.loss_norm == "l2":
            loss = mean_flat((distiller - distiller_target) ** 2) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(distiller_target, size=32, mode="bilinear")
            loss = mean_flat((distiller - distiller_target) ** 2) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(distiller_target, size=224, mode="bilinear")
            loss = self.lpips_loss((distiller + 1) / 2.0, (distiller_target + 1) / 2.0) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        return {"loss": loss}


@th.no_grad()
def sample_onestep(distiller, x, sigmas, generator=None, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * (next_t**2 - t_min**2) ** 0.5
    return x


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema, scales = start_ema, start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    step / total_steps * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1) + 1
        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    step / total_steps * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            target_ema = np.exp(start_scales * np.log(start_ema) / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = np.maximum(start_scales // (2**distill_stage), 2)
            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1), 0
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = np.maximum(2 // (2**sub_stage), 1)
            scales = np.where(scales == 2, sub_scales, scales)
            target_ema = 1.0
        else:
            raise NotImplementedError
        return float(target_ema), int(scales)
    return ema_and_scales_fn


def ema_update(target_params, params, mu):
    for tp, p in zip(target_params, params):
        tp.detach().mul_(mu).add_(p, alpha=1 - mu)
```

For CD, `teacher_model` and `teacher_diffusion` are supplied and the target point uses the Heun solver. For CT, `teacher_model=None` makes the denoiser output equal to `x_start`, so the Euler derivative is `(x_t - x_start) / t`, the PF-ODE velocity implied by the one-sample score estimate. The target network is an EMA copy of the online weights with stop-gradient.
