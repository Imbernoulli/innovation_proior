# EDM

## Problem

Diffusion models from different theoretical origins (VP/DDPM, VE/SMLD, score-SDE, iDDPM, DDIM) tangle together choices that are actually independent: the noise schedule, image scaling, time discretization, ODE/SDE solver, network input/output normalization, loss weighting, and training noise distribution. EDM puts them all in one continuous-time framework, exposes each as a free knob, and derives the best setting of each from first principles — aiming for far fewer network evaluations per image without sacrificing quality, with sampler improvements that drop into any pre-trained model.

## Key ideas

**One ODE.** Write the probability-flow ODE directly in terms of the marginals p(x; σ) = p_data * N(0, σ²I) rather than the SDE's drift/diffusion f, g:

  dx = [ (ṡ/s) x − s² σ̇ σ ∇_x log p(x/s; σ) ] dt.

σ(t) reparameterizes time, s(t) reparameterizes the image. The score is the denoiser: ∇_x log p(x; σ) = (D(x; σ) − x)/σ², where D(x;σ)=E[y|x] is the optimal L2 denoiser (Vincent 2011).

**Best schedule/scaling: σ(t) = t, s(t) = 1.** Then dx/dt = (x − D(x; t))/t; the trajectory tangent always points at the denoiser output, which changes slowly with σ, so trajectories are nearly straight and cheap to integrate.

**Sampling.** Discretize with a warped schedule

  σ_i = ( σ_max^{1/ρ} + (i/(N−1))(σ_min^{1/ρ} − σ_max^{1/ρ}) )^ρ,  σ_N = 0,

with ρ = 7 (puts step accuracy at low σ, which matters perceptually; σ_min=0.002, σ_max=80). Integrate with **Heun's 2nd-order method** (one extra denoiser eval per step → O(h³) local error vs Euler's O(h²)), reverting to Euler on the final step to σ=0. Optional **stochastic churn**: per step add fresh noise to climb from σ_i to σ̂_i = σ_i + γ_i σ_i, then take one Heun step down; γ_i = S_churn/N (clamped ≤ √2−1) only within σ ∈ [S_min, S_max], with new-noise std scaled by S_noise (slightly > 1).

**Preconditioning.** Wrap the raw network F_θ as

  D_θ(x; σ) = c_skip(σ) x + c_out(σ) F_θ( c_in(σ) x ; c_noise(σ) ),

with the coefficients derived from: unit-variance input, unit-variance training target, and minimal error amplification:

  c_in = 1/√(σ² + σ_data²),
  c_skip = σ_data²/(σ² + σ_data²),
  c_out = σ·σ_data/√(σ² + σ_data²),
  c_noise = ¼ ln σ  (empirical).

**Loss.** E_{σ,y,n}[ λ(σ) ‖D_θ(y+n;σ) − y‖² ] with λ(σ) = 1/c_out(σ)² = (σ² + σ_data²)/(σ·σ_data)² (equalizes the effective per-σ weight; the initial per-σ loss is exactly 1). Draw noise levels log-normally: ln σ ∼ N(P_mean, P_std²), with P_mean = −1.2, P_std = 1.2, σ_data = 0.5 — concentrating training on the intermediate σ where there is signal to learn.

## Code

```python
import numpy as np
import torch

# Network preconditioning: derived c_skip, c_out, c_in from variance constraints.
class EDMPrecond(torch.nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model              # raw F_theta(scaled_x, noise_cond, class_labels)
        self.sigma_data = sigma_data

    def forward(self, x, sigma, class_labels=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out  = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in   = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels)
        return c_skip * x + c_out * F_x.to(torch.float32)


# Training loss: log-normal sigma, weight = 1 / c_out^2.
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean, self.P_std, self.sigma_data = P_mean, P_std, sigma_data

    def __call__(self, net, images, class_labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, class_labels)
        return weight * ((D_yn - y) ** 2)


# Sampling: rho=7 schedule + Heun (Euler fallback at sigma=0) + optional churn.
@torch.no_grad()
def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like,
                num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    sigma_min = max(sigma_min, getattr(net, 'sigma_min', sigma_min))
    sigma_max = min(sigma_max, getattr(net, 'sigma_max', sigma_max))

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily (churn).
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step (sigma = t, so dx/dt = (x - D(x;t)) / t).
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Heun 2nd-order correction (skip on the last step to sigma=0).
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
```

Deterministic sampling is `S_churn = 0`. Earlier methods are recovered as choices in this same framework: VE uses σ(t)=√t, VP uses σ(t)=√(e^{½β_d t²+β_min t}−1) with s=1/√(σ²+1), DDIM uses σ(t)=t, s=1 — all run by the same sampler with their respective {c_skip, c_out, c_in, c_noise}.
