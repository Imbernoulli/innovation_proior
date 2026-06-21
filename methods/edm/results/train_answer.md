The diffusion-model landscape looks like a collection of incompatible machines—DDPM as a discrete Markov chain, score matching as annealed Langevin dynamics, Score-SDE as a pair of stochastic differential equations—each bundling its own noise schedule, image scaling, discretization, solver, network normalization, and loss weighting. The trouble is that these choices are presented as welded to the theory that introduced them, so improving one part appears to require re-deriving everything else. In practice that means sampling still costs dozens to hundreds of network evaluations per image, training is inefficiently spread across noise levels, and the network parameterization amplifies errors at the very levels where the task is hardest. The real opportunity is to strip the model down to its essentials, recognize that most of these knobs are logically independent, and then set each one on its own merits.

All of these models share the same underlying object: a data distribution convolved with isotropic Gaussian noise, giving a one-parameter family of marginals p(x; σ) = p_data * N(0, σ²I). The score is just the denoiser in disguise, because Vincent's denoising identity says the optimal denoiser D(x; σ) satisfies ∇_x log p(x; σ) = (D(x; σ) − x)/σ². So instead of deriving everything from an SDE drift f and diffusion g, we can write the probability-flow ODE directly in terms of the noise level σ(t) and an optional signal rescaling s(t): dx = [(ṡ/s)x − s²σ̇σ ∇_x log p(x/s; σ)]dt. This makes every prior model a reparameterization of a single canonical process.

The method is EDM, short for Elucidating the Design Space of Diffusion-Based Generative Models. The first choice it makes is to set σ(t)=t and s(t)=1, which is the schedule that straightens trajectories: the ODE becomes dx/dt = (x − D(x; t))/t, so the tangent at every point points straight at the current denoiser estimate. Next it replaces the usual first-order Euler steps with Heun's second-order method, which costs one extra network evaluation per step but improves the local error from O(h²) to O(h³). Because the truncation error is largest at low noise levels, the schedule warps a uniform grid toward low σ via σ_i = (σ_max^(1/ρ) + (i/(N−1))(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ with ρ=7, concentrating steps where they are perceptually load-bearing. The final step to σ=0 falls back to Euler because the derivative would otherwise diverge.

The network itself cannot directly output the denoiser, because the noisy input x has variance σ_data² + σ² that varies enormously across noise levels. EDM therefore wraps a raw network F_θ as D_θ(x; σ) = c_skip(σ)x + c_out(σ)F_θ(c_in(σ)x; c_noise(σ)) and derives the coefficients from first principles. Requiring unit input variance gives c_in(σ) = 1/√(σ²+σ_data²). Requiring unit target variance and minimal error amplification gives c_skip(σ) = σ_data²/(σ²+σ_data²) and c_out(σ) = σσ_data/√(σ²+σ_data²). With these choices the skip automatically interpolates between predicting the noise at low σ and predicting the signal at high σ, and c_out stays bounded instead of blowing up linearly in σ. The training loss weights the residual by λ(σ) = 1/c_out² = (σ²+σ_data²)/(σσ_data)², which equalizes the per-level loss at initialization, and it samples σ from a log-normal distribution centered on the intermediate noise levels where there is actually something to learn. Optional non-leaking augmentations are applied before noise is added, with the augmentation labels passed into the network.

Stochasticity is treated as a tunable add-on rather than a mandatory SDE. The deterministic ODE and the noise-injecting SDE share the same marginals in continuous time, so any quality benefit from adding noise is a discretization effect. EDM implements it as Langevin churn: at each step, add fresh Gaussian noise to raise the level from t_cur to t_hat = t_cur + γt_cur, then take one Heun step down to t_next. The churn is fenced inside a noise window [S_min, S_max] and capped so it never injects more noise than is already present; setting S_churn=0 recovers deterministic sampling. This scrubbing of accumulated error helps quality, but too much churn washes out detail, so it is kept mild.

Here is a concise implementation.

```python
import numpy as np
import torch


class EDMPrecond(torch.nn.Module):
    def __init__(self, model, sigma_min=0, sigma_max=float("inf"), sigma_data=0.5, label_dim=0, use_fp16=False):
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
def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like, num_steps=18,
                sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float("inf"), S_noise=1):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
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
