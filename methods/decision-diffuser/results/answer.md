# Decision Diffuser, distilled

Decision Diffuser recasts offline reinforcement learning as conditional generative modeling
of trajectories. It trains a return-conditional diffusion model over **state sequences**
(actions excluded), recovers actions with a separate inverse-dynamics model, and at sampling
time uses **classifier-free guidance** with **low-temperature sampling** to extract
high-return trajectories — performing the trajectory stitching of dynamic programming without
ever learning a value function or imposing an explicit in-distribution constraint.

## Problem it solves

Recover a return-maximizing policy from a fixed dataset of mostly sub-optimal, reward-labeled
trajectories, with no environment access. The good trajectory may not be in the data but is
assemblable from fragments (stitching). TD-based offline RL gets stitching from the
`max_{a'} Q` backup but sits on the deadly triad (function approximation + bootstrapping +
off-policy), over-estimates `Q` on out-of-distribution actions offline, and needs an explicit
divergence penalty `D(d^{pi} || d^{mu})` that requires per-task tuning.

## Key idea

Model `max_theta E_{tau~D}[log p_theta(x_0(tau) | y(tau))]`. Maximum-likelihood density
estimation has no Bellman fixed point (no deadly triad) and stays on the data support by
construction (no in-distribution penalty). The DDPM mechanics are standard and exact:

```
x_k = sqrt(bar_alpha_k) x_0 + sqrt(1 - bar_alpha_k) eps
x0_hat = sqrt(1 / bar_alpha_k) x_k - sqrt(1 / bar_alpha_k - 1) eps_theta(x_k, k)
mu_k = [beta_k sqrt(bar_alpha_{k-1}) / (1 - bar_alpha_k)] x0_hat
     + [sqrt(alpha_k)(1 - bar_alpha_{k-1}) / (1 - bar_alpha_k)] x_k
var_k = beta_k(1 - bar_alpha_{k-1}) / (1 - bar_alpha_k)
```

Three modeling choices make it work:

1. **Diffuse over states only.** `x_k(tau) = (s_t, ..., s_{t+H-1})_k`, a 2D array (state-dim ×
   horizon) treated as a 1D temporal image. States are continuous and smooth; actions (often
   joint torques) are high-frequency, less smooth, and harder for a denoiser — so they are not
   diffused. Actions are recovered by an inverse-dynamics model `a_t = f_phi(s_t, s_{t+1})`
   trained by regression on the same offline transitions.

2. **Classifier-free guidance, not value guidance.** Conditioning on the normalized return
   `y = R(tau) in [0,1]` via a value/classifier gradient would re-introduce an offline
   Q-function (deadly triad, OOD over-estimation) and divorce the training objective from the
   sampling objective. Instead train one network as both conditional and unconditional via
   condition-dropout (`y -> ∅` with prob `p`), and sample with the perturbed noise
   ```
   eps_hat = eps_theta(x_k, ∅, k) + omega·( eps_theta(x_k, y, k) - eps_theta(x_k, ∅, k) ).
   ```
   Conditioning on `R = 1` with `omega > 1` extrapolates toward the best behavior in the data.

3. **Low-temperature sampling.** In distribution notation, draw the reverse step
   `x_{k-1} ~ N(mu_{k-1}, alpha·Sigma_{k-1})` when `alpha` names the covariance scale. The
   temporal-U-Net implementation below uses a standard-deviation multiplier `noise_std = 0.5`,
   so its literal covariance multiplier is `noise_std^2`. Low temperature concentrates on
   high-likelihood (hence, after return-conditioning, high-return) sequences without removing
   all entropy needed for diverse stitching.

Together, return-conditioning + low-temperature sampling + diffusion's compositional sampling
recombine sub-optimal fragments into high-return trajectories — **implicit dynamic
programming** with no value function, no Bellman backup, no max over actions.

## Conditioning beyond returns, and composition

`y` can also be a one-hot constraint `1(tau ∈ C_i)` or skill `1(tau ∈ B_i)`. For attributes
`{y^i}` conditionally independent given the trajectory, Bayes gives an additive score, so the
composed (guided) noise is
```
eps_hat = eps_theta(x_k, ∅, k) + omega·sum_{i=1}^n ( eps_theta(x_k, y^i, k) - eps_theta(x_k, ∅, k) ),
```
which combines constraints or sequences skills never seen jointly in training (n=1 recovers
single-condition guidance). A NOT composition flips the sign of an attribute's score
difference. Feeding a summed one-hot condition into a single forward pass instead fails —
that input was never seen in training — so composition must live in score space.

## Training (joint loss)

```
L(theta, phi) = E_{k, tau~D, beta~Bern(p)} || eps - eps_theta( x_k(tau), (1-beta)y(tau) + beta·∅, k ) ||^2
              + E_{(s,a,s')~D} || a - f_phi(s, s') ||^2
```
`p = 0.25` condition-dropout (higher than the ~0.1 typical for image classifier-free guidance,
since the return is a weak scalar condition and guidance leans hard on the unconditional
branch). `K = 100` diffusion steps, cosine `beta` schedule. The diffusion piece predicts the
noise; the first state of the plan is clamped to the observed history, so its loss weight is
zeroed; remaining timesteps are weighted by a normalized per-step discount.

## Planning (receding horizon)

Maintain a history queue of length `C`. Each control step: observe `s`; start from scaled
Gaussian noise `x_K = noise_std·z`; for `k = K..1`, clamp the leading state(s) of the plan to
the history, form the classifier-free guided `eps_hat`, take the low-temperature reverse step;
from the clean plan `x_0` take `(s_t, s_{t+1})` and execute `a_t = f_phi(s_t, s_{t+1})`.

## Architecture and hyperparameters

- `eps_theta`: temporal U-Net (6 residual blocks; each = two 1D temporal convolutions,
  kernel 5, group norm, Mish), state sequence as a 1D temporal image (state-dim × horizon).
- Timestep and condition embeddings: each 128-dim, separate 2-layer MLPs (256 hidden, Mish),
  combined and injected into the first conv of each block; when `y = ∅` the condition
  embedding is zeroed.
- Inverse dynamics `f_phi`: 2-layer MLP, 512 hidden, ReLU.
- Adam, lr `2e-4`, batch `32`; returns normalized to `[0,1]`, condition on `R = 1` at test;
  guidance scale `omega in {1.2, 1.4, 1.6, 1.8}` per task; low-temperature setting `0.5`;
  context length `C = 20`; horizon `H = 100` (locomotion).

## Working code

Grounded in the canonical temporal-U-Net inverse-dynamics diffusion implementation (states-only
diffusion + inverse dynamics + classifier-free guidance + low-temperature DDPM sampler).

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import Bernoulli


def cosine_beta_schedule(K, s=0.008):
    steps = K + 1
    x = torch.linspace(0, K, steps)
    ac = torch.cos(((x / K) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)


def extract(a, t, shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))


def apply_conditioning(x, cond, action_dim=0):
    for tstep, val in cond.items():
        x[:, tstep, action_dim:] = val.clone()  # clamp leading state(s) to observed history
    return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, k):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=k.device) * -(math.log(10000) / (half - 1)))
        e = k[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, embed_dim, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv1d(c_in, c_out, kernel, padding=pad),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(c_out, c_out, kernel, padding=pad),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, c_out),
                                      nn.Unflatten(-1, (c_out, 1)))
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.time_mlp(emb)
        return self.conv2(out) + self.res(x)


class TemporalUnet(nn.Module):
    """Canonical temporal U-Net denoiser; cond is carried in the signature because
    the diffusion wrapper clamps observed state columns with the same dictionary."""
    def __init__(self, horizon, transition_dim, dim=128, dim_mults=(1, 2, 4, 8),
                 returns_condition=True, condition_dropout=0.25):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4),
                                      nn.Mish(), nn.Linear(dim * 4, dim))
        self.returns_condition = returns_condition
        if returns_condition:
            self.returns_mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(),
                                             nn.Linear(dim, dim * 4), nn.Mish(),
                                             nn.Linear(dim * 4, dim))
            self.mask_dist = Bernoulli(probs=1 - condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, (ci, co) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(ci, co, embed_dim),
                ResidualTemporalBlock(co, co, embed_dim),
                nn.Conv1d(co, co, 3, 2, 1) if not last else nn.Identity()]))
        mid = dims[-1]
        self.mid1 = ResidualTemporalBlock(mid, mid, embed_dim)
        self.mid2 = ResidualTemporalBlock(mid, mid, embed_dim)
        for i, (ci, co) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(co * 2, ci, embed_dim),
                ResidualTemporalBlock(ci, ci, embed_dim),
                nn.ConvTranspose1d(ci, ci, 4, 2, 1) if not last else nn.Identity()]))
        self.final = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2),
                                   nn.GroupNorm(8, dim), nn.Mish(),
                                   nn.Conv1d(dim, transition_dim, 1))

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        x = einops.rearrange(x, 'b h c -> b c h')
        t = self.time_mlp(time)
        if self.returns_condition:
            assert returns is not None
            z = self.returns_mlp(returns)
            if use_dropout:
                z = self.mask_dist.sample((z.size(0), 1)).to(z.device) * z  # drop y -> ∅
            if force_dropout:
                z = 0 * z                                                   # the ∅ branch
            t = torch.cat([t, z], dim=-1)
        h = []
        for a, b, down in self.downs:
            x = b(a(x, t), t); h.append(x); x = down(x)
        x = self.mid2(self.mid1(x, t), t)
        for a, b, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = b(a(x, t), t); x = up(x)
        x = self.final(x)
        return einops.rearrange(x, 'b c h -> b h c')


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=100,
                 hidden_dim=512, returns_condition=True, condition_guidance_w=1.2,
                 noise_std=0.5, loss_discount=1.0, predict_epsilon=True):
        super().__init__()
        self.model, self.horizon = model, horizon
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.n_timesteps = int(n_timesteps)
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.noise_std = noise_std
        self.predict_epsilon = predict_epsilon
        self.inv_model = nn.Sequential(
            nn.Linear(2 * observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))           # a_t = f_phi(s_t, s_{t+1})
        betas = cosine_beta_schedule(n_timesteps)
        ac = torch.cumprod(1 - betas, 0)
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_ac', ac.sqrt())
        self.register_buffer('sqrt_1m_ac', (1 - ac).sqrt())
        self.register_buffer('sqrt_recip_ac', (1 / ac).sqrt())
        self.register_buffer('sqrt_recipm1_ac', (1 / ac - 1).sqrt())
        self.register_buffer('post_mean_c1', betas * ac_prev.sqrt() / (1 - ac))
        self.register_buffer('post_mean_c2', (1 - ac_prev) * (1 - betas).sqrt() / (1 - ac))
        self.register_buffer('post_log_var',
                             torch.log((betas * (1 - ac_prev) / (1 - ac)).clamp(min=1e-20)))
        discounts = loss_discount ** torch.arange(horizon, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        lw = discounts[:, None] * torch.ones(horizon, observation_dim)
        lw[0] = 0.                                      # first state clamped -> no signal
        self.register_buffer('loss_weight', lw)

    def q_sample(self, x_start, k, noise):
        return extract(self.sqrt_ac, k, x_start.shape) * x_start \
            + extract(self.sqrt_1m_ac, k, x_start.shape) * noise

    def predict_start_from_noise(self, x_k, k, noise):
        if self.predict_epsilon:
            return extract(self.sqrt_recip_ac, k, x_k.shape) * x_k \
                - extract(self.sqrt_recipm1_ac, k, x_k.shape) * noise
        return noise

    def q_posterior(self, x_start, x_k, k):
        mean = extract(self.post_mean_c1, k, x_k.shape) * x_start \
             + extract(self.post_mean_c2, k, x_k.shape) * x_k
        return mean, extract(self.post_log_var, k, x_k.shape)

    def p_losses(self, x_start, cond, k, returns=None):
        eps = torch.randn_like(x_start)
        x_k = apply_conditioning(self.q_sample(x_start, k, eps), cond, action_dim=0)
        pred = self.model(x_k, cond, k, returns)         # condition-dropout inside
        target = eps if self.predict_epsilon else x_start
        if not self.predict_epsilon:
            pred = apply_conditioning(pred, cond, action_dim=0)
        return (self.loss_weight * (pred - target) ** 2).mean()

    def loss(self, trajectory, cond, returns=None):
        obs = trajectory[:, :, self.action_dim:]         # diffuse over states only
        act = trajectory[:, :, :self.action_dim]
        b = obs.shape[0]
        k = torch.randint(0, self.n_timesteps, (b,), device=obs.device).long()
        diffuse_loss = self.p_losses(obs, cond, k, returns)
        s, s_next, a = obs[:, :-1], obs[:, 1:], act[:, :-1]
        pred_a = self.inv_model(torch.cat([s, s_next], -1).reshape(-1, 2 * self.observation_dim))
        inv_loss = F.mse_loss(pred_a, a.reshape(-1, self.action_dim))
        return 0.5 * (diffuse_loss + inv_loss)

    def p_mean_variance(self, x, cond, k, returns=None):
        if self.returns_condition:
            eps_cond = self.model(x, cond, k, returns, use_dropout=False)
            eps_uncond = self.model(x, cond, k, returns, force_dropout=True)
            eps = eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)
        else:
            eps = self.model(x, cond, k)
        x0 = self.predict_start_from_noise(x, k, eps)
        return self.q_posterior(x0, x, k)

    @torch.no_grad()
    def p_sample(self, x, cond, k, returns=None):
        mean, log_var = self.p_mean_variance(x, cond, k, returns)
        noise = self.noise_std * torch.randn_like(x)                 # low-temperature std
        nonzero = (1 - (k == 0).float()).reshape(-1, *((1,) * (x.dim() - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, returns=None, horizon=None):
        b = len(cond[0])
        horizon = horizon or self.horizon
        x = self.noise_std * torch.randn((b, horizon, self.observation_dim),
                                         device=self.betas.device)
        x = apply_conditioning(x, cond, action_dim=0)
        for i in reversed(range(self.n_timesteps)):
            k = torch.full((b,), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, cond, k, returns)
            x = apply_conditioning(x, cond, action_dim=0)
        return x

    @torch.no_grad()
    def plan_action(self, obs, target_return):
        cond = {0: obs}
        x = self.p_sample_loop(cond, returns=target_return)
        return self.inv_model(torch.cat([x[:, 0], x[:, 1]], -1))     # a_t = f_phi(s_t, s_{t+1})
```

Composition (constraints / skills): replace the single `(eps_cond - eps_uncond)` difference in
`p_mean_variance` with `sum_i (eps_theta(x, y^i, k) - eps_theta(x, ∅, k))`, scaled by `omega`.
