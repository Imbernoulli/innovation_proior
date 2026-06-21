The task is to recover a return-maximizing policy from a fixed offline dataset of mostly sub-optimal trajectories, without further environment interaction. The core difficulty is stitching: the best trajectory may never appear in the data, yet it can be assembled from fragments of several logged runs. TD-based offline RL addresses this through the max-action Bellman backup, but that places the learner on the deadly triad of function approximation, bootstrapping, and off-policy data. Offline, the policy quickly queries value estimates at actions outside the data support, the network extrapolates overly optimistic values, and there is no environment to correct the error. The usual fix is an explicit in-distribution constraint, which turns training into a brittle constrained optimization that must be re-tuned per task. Existing trajectory models such as Diffuser avoid value learning but still rely on a separately trained return classifier or value-style guidance, reimporting the very instability they sought to escape, while return-conditioned sequence models like Decision Transformer do not compose attributes or stitch fragments well because they are autoregressive likelihood models rather than score-based samplers.

I propose Decision Diffuser. It recasts offline RL as conditional generative modeling of trajectories, using a diffusion model to capture the data distribution over state sequences and classifier-free guidance to steer samples toward high return. Because the objective is maximum-likelihood density estimation, there is no Bellman backup, no value function, and no deadly triad. Because sampling stays on the support of the learned data distribution by construction, there is no need for an engineered in-distribution penalty. The stitching that dynamic programming provides through value propagation is replaced by the compositional nature of diffusion sampling, which recombines fragments across trajectories without ever computing a max over actions.

The first design choice is to diffuse only over states, not over actions. State sequences in locomotion tasks are continuous and smooth, whereas actions such as joint torques are high-frequency and jerky, making them harder for a denoiser to fit. So the diffusion model operates on x_k(tau) = (s_t, ..., s_{t+H-1})_k, a two-dimensional array with state dimension as channels and horizon as time. Actions are recovered afterward by a small inverse-dynamics model a_t = f_phi(s_t, s_{t+1}) trained by ordinary supervised regression on the same offline transitions. This separates the hard-to-denoise action signal from the smooth planning signal.

The second choice is how to condition on return. Classifier guidance would require training a return predictor on noisy trajectories, which is again a value function and reintroduces offline over-estimation; it also divorces the training objective from the sampling objective. Instead, Decision Diffuser uses classifier-free guidance. A single network is trained both conditionally and unconditionally by randomly dropping the return label with probability 0.25 during training. The null token corresponds to an all-zero condition embedding. At sampling time the guided noise is eps_hat = eps_theta(x_k, emptyset, k) + omega * (eps_theta(x_k, y, k) - eps_theta(x_k, emptyset, k)), where y is the normalized return. Setting y to the top of the normalized range and omega > 1 pushes the sampler toward the best behavior in the data while remaining on the data manifold.

A third ingredient is low-temperature sampling. Each reverse step is drawn as x_{k-1} = mu_{k-1} + eta * sigma_{k-1} * z with eta = 0.5, so samples concentrate on the high-likelihood mode of the conditional distribution. This suppresses the mediocre behavior that dominates the dataset while retaining enough entropy for the model to recombine trajectory fragments differently for different starting states. The result is a receding-horizon planner: at each environment step the current state is clamped to the first column of the plan, the reverse chain is run, the first two predicted states are read off, and the inverse-dynamics model outputs the action to execute.

The same conditioning machinery extends beyond returns to constraints or skills represented as one-hot indicators. Because scores compose additively under conditional independence, multiple attributes can be combined at sampling time even when no training trajectory satisfied them jointly, simply by summing their conditional-minus-unconditional score differences. This is the same compositional primitive that enables trajectory stitching, now applied across attributes.

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
        x[:, tstep, action_dim:] = val.clone()
    return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, k):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=k.device) * -(math.log(10000) / (half - 1)))
        e = k[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, embed_dim, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad),
            nn.GroupNorm(8, c_out), nn.Mish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel, padding=pad),
            nn.GroupNorm(8, c_out), nn.Mish())
        self.time_mlp = nn.Sequential(
            nn.Mish(), nn.Linear(embed_dim, c_out), nn.Unflatten(-1, (c_out, 1)))
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.time_mlp(emb)
        return self.conv2(out) + self.res(x)


class TemporalUnet(nn.Module):
    def __init__(self, horizon, transition_dim, dim=128, dim_mults=(1, 2, 4, 8),
                 returns_condition=True, condition_dropout=0.25):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(),
            nn.Linear(dim * 4, dim))
        self.returns_condition = returns_condition
        if returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim * 4), nn.Mish(),
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
        self.final = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2), nn.GroupNorm(8, dim), nn.Mish(),
            nn.Conv1d(dim, transition_dim, 1))

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        x = einops.rearrange(x, 'b h c -> b c h')
        t = self.time_mlp(time)
        if self.returns_condition:
            z = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample((z.size(0), 1)).to(z.device)
                z = mask * z
            if force_dropout:
                z = 0 * z
            t = torch.cat([t, z], dim=-1)
        h = []
        for a, b, down in self.downs:
            x = b(a(x, t), t)
            h.append(x)
            x = down(x)
        x = self.mid2(self.mid1(x, t), t)
        for a, b, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = b(a(x, t), t)
            x = up(x)
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
            nn.Linear(hidden_dim, action_dim))
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
        lw[0] = 0.
        self.register_buffer('loss_weight', lw)

    def q_sample(self, x_start, k, noise):
        return (extract(self.sqrt_ac, k, x_start.shape) * x_start
                + extract(self.sqrt_1m_ac, k, x_start.shape) * noise)

    def predict_start_from_noise(self, x_k, k, noise):
        if self.predict_epsilon:
            return (extract(self.sqrt_recip_ac, k, x_k.shape) * x_k
                    - extract(self.sqrt_recipm1_ac, k, x_k.shape) * noise)
        return noise

    def q_posterior(self, x_start, x_k, k):
        mean = (extract(self.post_mean_c1, k, x_k.shape) * x_start
                + extract(self.post_mean_c2, k, x_k.shape) * x_k)
        return mean, extract(self.post_log_var, k, x_k.shape)

    def p_losses(self, x_start, cond, k, returns=None):
        eps = torch.randn_like(x_start)
        x_k = apply_conditioning(self.q_sample(x_start, k, eps), cond, action_dim=0)
        pred = self.model(x_k, cond, k, returns)
        target = eps if self.predict_epsilon else x_start
        if not self.predict_epsilon:
            pred = apply_conditioning(pred, cond, action_dim=0)
        return (self.loss_weight * (pred - target) ** 2).mean()

    def loss(self, trajectory, cond, returns=None):
        obs = trajectory[:, :, self.action_dim:]
        act = trajectory[:, :, :self.action_dim]
        b = obs.shape[0]
        k = torch.randint(0, self.n_timesteps, (b,), device=obs.device).long()
        diffuse_loss = self.p_losses(obs, cond, k, returns)
        s, s_next, a = obs[:, :-1], obs[:, 1:], act[:, :-1]
        pred_a = self.inv_model(
            torch.cat([s, s_next], -1).reshape(-1, 2 * self.observation_dim))
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
        noise = self.noise_std * torch.randn_like(x)
        nonzero = (1 - (k == 0).float()).reshape(-1, *((1,) * (x.dim() - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, returns=None, horizon=None):
        b = len(cond[0])
        horizon = horizon or self.horizon
        x = self.noise_std * torch.randn(
            (b, horizon, self.observation_dim), device=self.betas.device)
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
        return self.inv_model(torch.cat([x[:, 0], x[:, 1]], -1))
```
