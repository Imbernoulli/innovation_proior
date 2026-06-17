# Diffusion Policy, distilled

Diffusion Policy represents a visuomotor robot policy as a **conditional denoising diffusion model on
the action**. Instead of regressing one action per observation, it learns the *gradient field* (score)
of the conditional action distribution `p(A | O)` by denoising, and generates an action sequence by
running a stochastic reverse (Langevin-style) chain from Gaussian noise, conditioned on the
observation. This gives energy-based-policy expressivity (arbitrary, sharp, multimodal action
distributions) with stable supervised-regression training, because the intractable EBM normalizer `Z`
is never formed.

## Problem it solves

Behavior cloning as L2 regression fits a unimodal Gaussian and averages the multiple valid actions in
multimodal demonstrations (producing invalid in-between actions at decision points). Discretized,
mixture (LSTM-GMM), and clustering (BET) policies must pre-specify the number of modes and model each
step independently, so rollouts are temporally inconsistent (jitter between valid plans). Implicit /
energy-based policies (IBC) are natively multimodal but train unstably because their InfoNCE loss
estimates the intractable normalizer `Z(o,θ)` by negative sampling. Trajectory-diffusion planners
(Diffuser) model the joint `p(A,O)` and must run the (vision) model at every denoising step, which is
too slow for real-time control. Diffusion Policy wants: arbitrary `p(A|O)`, temporally consistent
action sequences, stable training (reliable checkpoint selection), and real-time inference from images.

## Key idea

Model the **score** of the action distribution, not the density. For an energy-based policy
`p_θ(a|o) = e^{-E_θ(o,a)}/Z(o,θ)`, the action-gradient of the log-density drops the normalizer
identically:

```
∇_a log p_θ(a|o) = -∇_a E_θ(o,a) - ∇_a log Z(o,θ)  =  -∇_a E_θ(o,a)      (since ∇_a log Z = 0).
```

Denoising diffusion learns this gradient field by regression. With the forward process
`A^k = √ᾱ_k A^0 + √(1-ᾱ_k) ε`, the noise `ε` is the negative score up to the noise scale
(`∇_{A^k} log q(A^k|A^0) = -ε/√(1-ᾱ_k)`), so training a network `ε_θ` to predict `ε` learns the score
with a plain MSE — no negatives, no `Z`. Sampling follows the learned field with injected noise
(Langevin dynamics); the random initialization and per-step noise produce the multimodal samples.

Three adaptations turn an image diffusion model into a policy:
1. **The denoised variable is the action (sequence)** `A_t = (a_t, …, a_{t+T_p-1})` — diffusion scales
   to high dimensions, so a whole chunk is denoised jointly, giving temporal consistency for free.
2. **The observation is a condition, not part of the denoised variable**: `ε_θ(O_t, A_t^k, k)` models
   `p(A_t|O_t)`, so the (vision) encoder runs once per step (not per denoising iteration) and trains
   end-to-end.
3. **Receding-horizon execution**: predict `T_p`, execute the first `T_a` (`1<T_a<T_p`), then replan —
   trading temporal consistency against reactivity.

## Training objective and sampling

Training (conditional simplified objective):

```
L = E_{k, A^0_t, ε} ‖ ε - ε_θ(O_t, √ᾱ_k A^0_t + √(1-ᾱ_k) ε, k) ‖²,   k ~ Uniform{1..K}, ε ~ N(0,I).
```

Sampling (ancestral / DDPM reverse step), starting `A^K_t ~ N(0,I)`:

```
A^{k-1}_t = (1/√α_k)( A^k_t - (1-α_k)/√(1-ᾱ_k) · ε_θ(O_t, A^k_t, k) ) + σ_k z,   z~N(0,I) (k>1), z=0 (k=1),
```

with `σ_k² ∈ {β_k, β̃_k = (1-ᾱ_{k-1})/(1-ᾱ_k) β_k}`. Compactly,
`A^{k-1}_t = α(A^k_t - γ ε_θ(O_t,A^k_t,k) + N(0,σ²I))` — a noisy gradient-descent step on the action
energy, with `(α,γ,σ)(k)` the noise schedule (learning-rate schedule). The score interpretation
follows because the optimal `ε_θ` equals `-√(1-ᾱ_k)·∇_{A^k} log p(A^k|O)`.

## Key design choices and why

- **Predict ε (not μ̃ or A^0)**: gives the score / Langevin interpretation, keeps the target as the
  actual perturbation added at level `k`, and yields the clean unweighted loss `L_simple`.
- **Cosine (square-cosine) noise schedule** (iDDPM): distributes noise levels better for control
  signals; CleanDiffuser uses the variance-preserving `(α_k, σ_k)` form with
  `α_k = cos(π/2·(t_k+s)/(1+s)) / cos(π/2·s/(1+s))` (so `α_k = √ᾱ_k`),
  `σ_k = √(1-α_k²)`, `s=0.008`.
- **Condition, don't joint-model** (vs Diffuser): real-time inference + end-to-end vision.
- **Action sequence** (vs single step): temporal consistency, robustness to idle/pause actions.
- **Normalize actions to [-1,1]** (not zero-mean unit-var): the sampler clips the prediction to the box
  each step; unit-var would make part of the action range unreachable.
- **GroupNorm in the vision encoder** (not BatchNorm): EMA-of-weights (standard in diffusion) conflicts
  with BatchNorm running statistics.
- **CNN backbone default, transformer for high-frequency/velocity actions**: temporal convolutions have a
  low-frequency inductive bias and over-smooth fast action changes.
- **DDIM at inference**: decouples training steps `K` from inference steps; fewer steps → real-time.
- **EMA of weights** (`ema_rate=0.995`) for evaluation: standard diffusion stabilizer.

## Limiting-case check (linear control)

For a linear plant `s_{t+1}=A s_t+B a_t+w_t` and demonstrations from `a_t=-K s_t`, with `T_p=1` the
MSE-optimal denoiser is `ε_θ(s,a,k) = (a + √ᾱ_k K s)/√(1-ᾱ_k)`; in the `α≈1` additive-noise
normalization this becomes `ε_θ(s,a,k)=(1/σ_k)[a+Ks]`. Deterministic sampling converges to `a=-Ks`. For `T_p>1`,
the clean target is `a^0_{t+t'}=-K(A-BK)^{t'}s_t`, so the denoiser component is
`ε_θ^{(t')}(s_t,a^k_{t+t'},k)=(a^k_{t+t'}+α_k K(A-BK)^{t'}s_t)/σ_k`; action-sequence cloning
therefore implicitly learns a task-relevant dynamics model.

## Working code (low-dim, single-action diffusion BC core)

Faithful to CleanDiffuser's `DiscreteDiffusionSDE` + `DQLMlp` with the critic / Q removed: the
noise-prediction MSE objective and the ancestral DDPM sampler, conditioned on the observation, with
actions in `[-1,1]`. This is the form used as the `diffusion_policy` baseline on D4RL MuJoCo.

```python
import numpy as np
import torch
import torch.nn as nn


def cosine_alpha_sigma(t, s=0.008):
    # Variance-preserving (alpha, sigma) form of the iDDPM square-cosine schedule:
    # alpha_k = sqrt(abar_k), sigma_k = sqrt(1 - abar_k); x_k = alpha_k*x0 + sigma_k*eps.
    alpha = (np.pi / 2.0 * (t.clip(0., 0.9946) + s) / (1 + s)).cos() / np.cos(np.pi / 2.0 * s / (1 + s))
    sigma = (1.0 - alpha ** 2).sqrt()
    return alpha, sigma


def positional_embed(k, dim):                       # sinusoidal embedding of the step index k
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=k.device) / (half - 1))
    args = k[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredMLP(nn.Module):                       # eps_theta(O, A^k, k)
    def __init__(self, obs_dim, act_dim, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))
        self.mid = nn.Sequential(
            nn.Linear(obs_dim + act_dim + emb_dim, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish())
        self.head = nn.Linear(256, act_dim)

    def forward(self, x, k, obs):                    # x:(b,act_dim) k:(b,) obs:(b,obs_dim)
        t = self.time_mlp(positional_embed(k, self.emb_dim))
        return self.head(self.mid(torch.cat([x, t, obs], dim=-1)))


class DiffusionPolicy:
    """Diffusion behavior cloning: conditional noise-prediction diffusion on the action.
    No critic, no Q, single-action sampling at inference."""
    def __init__(self, obs_dim, act_dim, diffusion_steps=1000, lr=3e-4, ema_rate=0.995, device="cpu"):
        self.K, self.device = diffusion_steps, device
        self.act_low, self.act_high = -1.0, 1.0          # actions normalized to [-1, 1]
        self.net = NoisePredMLP(obs_dim, act_dim).to(device)
        self.net_ema = NoisePredMLP(obs_dim, act_dim).to(device)
        self.net_ema.load_state_dict(self.net.state_dict())
        for p in self.net_ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.ema_rate = ema_rate
        t_grid = torch.linspace(1e-3, 1.0, self.K, device=device)
        self.alpha, self.sigma = cosine_alpha_sigma(t_grid)

    def add_noise(self, x0):                              # forward: x_k = alpha_k x0 + sigma_k eps
        k = torch.randint(self.K, (x0.shape[0],), device=self.device)
        eps = torch.randn_like(x0)
        a, s = self.alpha[k][:, None], self.sigma[k][:, None]
        return a * x0 + s * eps, k, eps

    def loss(self, act, obs):                             # L_simple = || eps_theta - eps ||^2
        xt, k, eps = self.add_noise(act)
        return ((self.net(xt, k, obs) - eps) ** 2).mean()

    @torch.no_grad()
    def ema_update(self):
        for p, pe in zip(self.net.parameters(), self.net_ema.parameters()):
            pe.mul_(self.ema_rate).add_(p, alpha=1 - self.ema_rate)

    @torch.no_grad()
    def _clip_eps(self, eps, xt, a, s):                  # keep implied x0=(xt-s*eps)/a inside [-1,1]
        lo = (xt - a * self.act_high) / s
        hi = (xt - a * self.act_low) / s
        return eps.clip(lo, hi)

    @torch.no_grad()
    def sample(self, obs, n_samples, steps=None, use_ema=True, temperature=1.0):
        net = self.net_ema if use_ema else self.net
        act_dim = net.head.out_features
        steps = self.K if steps is None else steps
        sched = torch.linspace(0, self.K - 1, steps + 1, device=self.device).long()
        a, s = self.alpha[sched], self.sigma[sched]
        stds = torch.zeros(steps + 1, device=self.device)
        stds[1:] = s[:-1] / s[1:] * (1 - (a[1:] / a[:-1]) ** 2).sqrt()   # ancestral per-step std
        xt = torch.randn(n_samples, act_dim, device=self.device) * temperature      # A^K ~ N(0, I)
        for i in reversed(range(1, steps + 1)):
            k = torch.full((n_samples,), int(sched[i]), dtype=torch.long, device=self.device)
            eps = self._clip_eps(net(xt, k, obs), xt, a[i], s[i])
            # x_{k-1} = (a[i-1]/a[i])(xt - s[i]*eps) + sqrt(s[i-1]^2 - stds[i]^2)*eps  (+ noise if i>1)
            xt = (a[i - 1] / a[i]) * (xt - s[i] * eps) + (s[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps
            if i > 1:
                xt = xt + stds[i] * torch.randn_like(xt)
        return xt.clip(self.act_low, self.act_high)


# Training / inference loops (the harness the policy plugs into)
def train(policy, dataloader, gradient_steps, device, ema_update_interval=1):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(policy.opt, T_max=gradient_steps)
    for step, batch in zip(range(gradient_steps), dataloader):
        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)                       # demonstrated action in [-1, 1]
        loss = policy.loss(act, obs)                         # noise-prediction MSE (BC only)
        policy.opt.zero_grad(); loss.backward(); policy.opt.step(); sched.step()
        if step % ema_update_interval == 0 and step >= 1000:
            policy.ema_update()


@torch.no_grad()
def act_for(policy, obs_np, normalizer, num_envs, device):  # one action per env, no reranking
    obs = torch.tensor(normalizer.normalize(obs_np), device=device, dtype=torch.float32)
    return policy.sample(obs, n_samples=num_envs, use_ema=True).cpu().numpy()
```
