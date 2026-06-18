# EDM

## Core method

EDM rewrites diffusion sampling around the noise-marginal family

  p(x; σ) = p_data * N(0, σ²I)

and treats the score model as a denoiser:

  ∇_x log p(x; σ) = (D(x; σ) - x) / σ².

For a time-dependent noise level σ(t) and optional signal scaling s(t), the probability-flow ODE becomes

  dx = [ (ṡ/s) x - s² σ̇ σ ∇_x log p(x/s; σ) ] dt.

The recommended deterministic sampler sets σ(t)=t and s(t)=1, giving

  dx/dt = (x - D(x; t)) / t.

It uses the polynomial noise grid

  σ_i = ( σ_max^(1/ρ) + (i/(N-1))(σ_min^(1/ρ) - σ_max^(1/ρ)) )^ρ,  i < N,
  σ_N = 0,

with ρ=7, σ_min=0.002, σ_max=80, and integrates with Heun's second-order method, falling back to Euler for the final step to σ=0.

## Preconditioning and loss

The raw network F_θ is wrapped as a denoiser

  D_θ(x; σ) = c_skip(σ) x + c_out(σ) F_θ(c_in(σ)x; c_noise(σ)).

The EDM coefficients are

  c_in = 1 / sqrt(σ² + σ_data²),
  c_skip = σ_data² / (σ² + σ_data²),
  c_out = σ σ_data / sqrt(σ² + σ_data²),
  c_noise = (1/4) ln σ.

The training loss samples ln σ ~ N(P_mean, P_std²), with P_mean=-1.2, P_std=1.2, σ_data=0.5, and weights the denoising residual by

  λ(σ) = 1 / c_out² = (σ² + σ_data²) / (σ σ_data)².

## Reference-shaped implementation

```python
import numpy as np
import torch


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        model,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        label_dim=0,
        use_fp16=False,
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.label_dim = label_dim
        self.use_fp16 = use_fp16

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([1, self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == "cuda" else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        return weight * ((D_yn - y) ** 2)


@torch.no_grad()
def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
```

Stochastic sampling is the same loop with `S_churn > 0`: each step first increases the current noise level to `t_hat = round_sigma(t_cur + gamma * t_cur)` by adding fresh Gaussian noise with variance `t_hat² - t_cur²`, then takes the deterministic Euler/Heun step down to `t_next`. Deterministic sampling is recovered by `S_churn = 0`.
