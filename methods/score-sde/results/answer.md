# Score SDE

## Problem

Two strong "perturb-then-reverse" generative model families — score matching with Langevin dynamics
(SMLD/NCSN) and denoising diffusion probabilistic models (DDPM) — each used a hand-chosen *finite*
ladder of noise scales, with separate training losses and separate samplers, and no shared theory.
Goal: a single continuous-time framework that (i) bridges data to a tractable prior without choosing a
discrete ladder, (ii) unifies the samplers, (iii) yields exact likelihoods, and (iv)
enables conditional generation from one unconditional model.

## Key idea

Replace the discrete noise ladder with a continuous-time forward diffusion (an Itô SDE) that has no
trainable parameters,

  dx = f(x, t) dt + g(t) dw,  t ∈ [0, T],  x(0) ~ p_data,  x(T) ~ prior.

By Anderson's (1982) time-reversal result, the reverse process is again a diffusion,

  dx = [f(x, t) − g(t)² ∇_x log p_t(x)] dt + g(t) dw̄,

with dt taken backward from T to 0. Its only unknown is the time-dependent score ∇_x log p_t(x).
Estimate it with a network s_θ(x, t) via continuous denoising score matching, then integrate the
reverse SDE to generate samples.

## Final method

**Training objective (denoising score matching).** With affine drift the transition kernel
p_{0t}(x(t)|x(0)) = N(μ_t x(0), σ_t² I) is Gaussian, so x(t) = μ_t x(0) + σ_t z and the conditional
score is −z/σ_t. Train

  θ* = argmin_θ E_t { λ(t) · E_{x(0)} E_{z} ||s_θ(x(t), t) + z/σ_t||² },  t ~ U(0, T), z ~ N(0, I),

with λ(t) ∝ σ_t² (equalizes the loss across t), giving the clean form ||σ_t s_θ(x(t),t) + z||².

**Three SDEs (the first two are the continuous limits of NCSN and DDPM).**
- VE (Variance Exploding, ≡ NCSN limit): dx = sqrt(d[σ²(t)]/dt) dw, no drift; σ(t) geometric,
  variance explodes, prior N(0, σ_max² I). The exact kernel variance is [σ²(t)−σ²(0)] I; the
  implementation convention returns σ(t) itself as the std because σ_min is a small positive floor.
- VP (Variance Preserving, ≡ DDPM limit): dx = −½β(t) x dt + sqrt(β(t)) dw; bounded variance
  (≡ I if started at I), prior N(0, I). Kernel N(x(0) e^{−½∫β}, (1 − e^{−∫β}) I).
- sub-VP: dx = −½β(t) x dt + sqrt(β(t)(1 − e^{−2∫β})) dw; same mean as VP, transition
  variance [1 − e^{−∫β}]² I ≤ the VP transition variance at every t, and the kernel standard
  deviation is therefore 1 − e^{−∫β}, not sqrt(1 − e^{−∫β}). The marginal tends to I as ∫β grows.

**Predictor–Corrector (PC) sampling.** Alternate a numerical reverse-SDE step (predictor) with a few
score-based Langevin MCMC steps (corrector) that re-project the sample onto the correct marginal p_t.
NCSN sampling = corrector-only; DDPM ancestral sampling = predictor-only (and is a particular
discretization of the reverse VP SDE). The Langevin step size is set from a target signal-to-noise
ratio r: ε = 2α(r||z||/||s||)². A reverse-diffusion predictor (mirror the forward discretization) is
automatic for any SDE; a single final denoising (Tweedie) step removes residual noise.

**Probability flow ODE.** Rewriting the Fokker–Planck equation as a continuity equation gives a
deterministic process with the same marginals:

  dx = [f(x, t) − ½ g(t)² ∇_x log p_t(x)] dt.

It is a neural ODE: exact log-likelihood via the instantaneous change of variables,
log p_0(x(0)) = log p_T(x(T)) + ∫_0^T ∇·f̃_θ(x(t),t) dt, with the divergence estimated by the
Skilling–Hutchinson trace estimator ∇·f̃_θ = E_v[vᵀ ∇f̃_θ v]. It also gives uniquely identifiable
latents (the forward process has no parameters), latent manipulation, and fast adaptive sampling via
black-box ODE solvers (RK45).

**Controllable generation.** The conditional reverse SDE uses ∇log p_t(x|y) = ∇log p_t(x) + ∇log
p_t(y|x): the unconditional score plus a cheap conditional term (a time-dependent classifier for
class-conditional generation, or an unconditional-model approximation for inpainting/colorization).
No retraining of the generator per task.

## Code

```python
import abc
import numpy as np
import torch
from scipy import integrate

class SDE(abc.ABC):
    def __init__(self, N): self.N = N
    @property
    @abc.abstractmethod
    def T(self): ...
    @abc.abstractmethod
    def sde(self, x, t): ...
    @abc.abstractmethod
    def marginal_prob(self, x, t): ...
    @abc.abstractmethod
    def prior_sampling(self, shape): ...
    @abc.abstractmethod
    def prior_logp(self, z): ...

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        return drift * dt, diffusion * torch.sqrt(torch.tensor(dt, device=t.device))

    def reverse(self, score_fn, probability_flow=False):
        N, T = self.N, self.T
        sde_fn, discretize_fn = self.sde, self.discretize
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
            @property
            def T(self):
                return T
            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score \
                        * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion
            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                score = score_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score \
                        * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    @property
    def T(self): return 1
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t)
    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean))
        return mean, std
    def prior_sampling(self, shape): return torch.randn(*shape)
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        return torch.sqrt(alpha)[:, None, None, None] * x - x, torch.sqrt(beta)

class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))
    @property
    def T(self): return 1
    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * torch.sqrt(torch.tensor(
            2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return torch.zeros_like(x), diffusion
    def marginal_prob(self, x, t):
        return x, self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    def prior_sampling(self, shape): return torch.randn(*shape) * self.sigma_max
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(
            z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent = torch.where(
            timestep == 0, torch.zeros_like(t), self.discrete_sigmas.to(t.device)[timestep - 1])
        return torch.zeros_like(x), torch.sqrt(sigma ** 2 - adjacent ** 2)

class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
    @property
    def T(self): return 1
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t * discount)
    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean)[:, None, None, None] * x
        # The kernel variance is (1 - exp(-int beta))^2, so this is the std.
        std = 1 - torch.exp(2. * log_mean)
        return mean, std
    def prior_sampling(self, shape): return torch.randn(*shape)
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.

def get_loss_fn(sde, eps=1e-5, likelihood_weighting=False):
    def loss_fn(model, batch):
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        x_t = mean + std[:, None, None, None] * z
        score = model(x_t, t)
        if likelihood_weighting:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1) * g2
        else:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1)
        return torch.mean(losses)
    return loss_fn

def reverse_diffusion_predictor(rsde, x, t):
    f, G = rsde.discretize(x, t)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * torch.randn_like(x)
    return x, x_mean

def langevin_corrector(score_fn, sde, x, t, snr, n_steps):
    if hasattr(sde, "alphas"):
        timestep = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[timestep]
    else:
        alpha = torch.ones_like(t)
    for _ in range(n_steps):
        grad, noise = score_fn(x, t), torch.randn_like(x)
        gn = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        nn = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step = (snr * nn / gn) ** 2 * 2 * alpha
        x_mean = x + step[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step * 2)[:, None, None, None] * noise
    return x, x_mean

def pc_sample(model, sde, shape, snr=0.16, n_steps=1, eps=1e-3, device='cpu'):
    score_fn = lambda x, t: model(x, t)
    rsde = sde.reverse(score_fn)
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(1, eps, sde.N, device=device)
    for i in range(sde.N):
        vec_t = torch.ones(shape[0], device=device) * timesteps[i]
        x, x_mean = langevin_corrector(score_fn, sde, x, vec_t, snr, n_steps)
        x, x_mean = reverse_diffusion_predictor(rsde, x, vec_t)
    return x_mean                                                            # final denoising

def ode_sample(model, sde, shape, eps=1e-3, device='cpu'):
    rsde = sde.reverse(lambda x, t: model(x, t), probability_flow=True)
    def ode_func(t, xf):
        x = torch.tensor(xf.reshape(shape), dtype=torch.float32, device=device)
        vec_t = torch.ones(shape[0], device=device) * t
        return rsde.sde(x, vec_t)[0].detach().cpu().numpy().reshape(-1)
    x0 = sde.prior_sampling(shape).cpu().numpy().reshape(-1)
    sol = integrate.solve_ivp(ode_func, (1, eps), x0, rtol=1e-5, atol=1e-5, method='RK45')
    return torch.tensor(sol.y[:, -1].reshape(shape))

def divergence_fn(fn):
    def div(x, t, noise):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.sum(fn(x, t) * noise)
            grad = torch.autograd.grad(y, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad * noise, dim=tuple(range(1, len(x.shape))))
    return div
```
