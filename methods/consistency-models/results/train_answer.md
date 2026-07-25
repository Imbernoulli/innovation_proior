Score-based diffusion models generate excellent samples, but sampling is slow because every generated image requires numerically integrating the probability-flow ODE from pure noise back to data. Even with fast solvers the process needs many sequential network evaluations, since we trace the whole trajectory even though we only want its endpoint. Existing one-step generators avoid this cost, but adversarial models are unstable and hard to tune, while likelihood-based one-step models usually fall short on sample quality. What is missing is a single-step generator that matches iterative diffusion in quality and editability, without adversarial training or an expensive offline distillation dataset.

The key observation is that, under the EDM Gaussian perturbation, the PF ODE has deterministic trajectories. With zero drift and diffusion coefficient sqrt(2t), the marginal at noise level t is just the data distribution convolved with Gaussian noise of standard deviation t, and the ODE becomes dx_t/dt = -t s(x_t, t). Every trajectory therefore connects one noisy endpoint x_T to one clean origin x_epsilon. If we could learn a map that sends any point on such a trajectory straight to its origin, sampling would reduce to a single network call. Such a map must be constant along trajectories, a property called self-consistency, and it must equal the identity at t = epsilon, because the trajectory through x_epsilon ends at x_epsilon itself. That boundary condition is essential: without it the model could collapse to a constant function.

The method is Consistency Models. A consistency model is a neural network f_theta(x, t) trained so that its output is the same for every point lying on the same PF-ODE trajectory. We parameterize it with a skip connection that enforces the boundary condition by construction: f_theta(x, t) = c_skip(t) x + c_out(t) F_theta(c_in(t) x, c_noise(t)), with c_skip(epsilon) = 1 and c_out(epsilon) = 0. This makes the network act as the identity at the smallest noise level while remaining differentiable everywhere and compatible with standard diffusion backbones. The concrete scalings are c_skip(t) = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2) and c_out(t) = sigma_data (t - epsilon) / sqrt(t^2 + sigma_data^2), using sigma_data = 0.5 and a log-noise time embedding.

Training proceeds by discretizing the interval [epsilon, T] with the EDM rho = 7 noise schedule. For a data point x, a level index n, and Gaussian noise z, we form the noised point x_{t_{n+1}} = x + t_{n+1} z. We then build a neighboring point on the same trajectory and pull the online network's prediction at the noisier point toward an exponential-moving-average target at the cleaner point. In Consistency Distillation the neighbor is produced by one numerical ODE step using a pretrained score model. In Consistency Training no teacher is needed: the exact score identity for a Gaussian-blurred density lets us replace the learned score with the single-sample estimate -(x_t - x) / t^2, so the Euler step collapses to pairing the same clean sample noised to two adjacent levels. The number of discretization levels is annealed upward during training, starting coarse for low variance and refining later for low bias.

Once trained, sampling is one forward pass: draw x_T from N(0, T^2 I) and return f_theta(x_T, T). Quality can be improved by spending extra compute through a re-noise-and-denoise loop: starting from the one-step estimate, add noise to an intermediate level and denoise again for a few steps. Because the model is defined at every noise level, the same loop also supports zero-shot inpainting, colorization, super-resolution, interpolation, and stroke-guided synthesis by masking in known information and letting the model fill the rest.

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
