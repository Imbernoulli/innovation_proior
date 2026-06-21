The problem is to learn a robot policy from expert demonstrations of observation-action pairs. The naive approach is supervised regression from the current observation to a single action, but that minimizes squared error and therefore converges to the conditional mean action. When the demonstrations are multimodal, e.g. the robot may legitimately go left or right around an obstacle, the mean of those modes is an action no expert ever performed and can be exactly the wrong thing to do. The policy then stalls at decision points or lunges into invalid in-between actions. Prior fixes each pay a different cost. Discretizing the action space into bins gives a multimodal categorical, but the number of bins explodes with action dimension. Mixture-density or LSTM-GMM policies are multimodal by construction, yet the number of components must be fixed in advance, training is prone to mode collapse, and each timestep is predicted independently so the executed trajectory jitters between valid plans. Clustering-plus-offset methods such as BET have the same per-step independence problem. Implicit energy-based policies are natively multimodal and sharp, but their maximum-likelihood training requires the intractable normalizer Z, which is estimated by negative sampling; noisy estimates make training oscillate and force expensive per-checkpoint hardware evaluation to pick a usable model. Trajectory-diffusion planners show that diffusion can model whole trajectories, but they model the joint distribution over states and actions, so the full encoder-decoder must run at every denoising step and cannot meet real-time visuomotor control budgets.

The method I propose is Diffusion Policy. It represents a visuomotor policy as a conditional denoising diffusion model over the action. Instead of predicting one action or a parametric distribution, it learns the gradient field, i.e. the score, of the conditional action distribution p(A | O). The key insight is that for an energy-based policy p(a | o) proportional to exp(-E(o,a)), the action-gradient of the log-density eliminates the intractable normalizer: the gradient of log Z with respect to a is zero. So the score is simply minus the gradient of the energy, and if we can learn that score directly we obtain energy-based expressivity without ever touching Z. Denoising diffusion provides exactly this: with a forward process that adds Gaussian noise to a clean action, A^k = sqrt(alpha_bar_k) A^0 + sqrt(1 - alpha_bar_k) epsilon, the noise epsilon is, up to a scaling factor, the negative score of the noised conditional distribution. Training a network epsilon_theta to predict epsilon with plain mean-squared error therefore learns the score field by stable regression, with no negative sampling and no normalizer. Sampling then runs a stochastic reverse chain starting from Gaussian noise, nudging the sample along the learned score field with injected noise; the random initialization and the per-step stochasticity let each rollout settle into one coherent mode rather than averaging modes together.

Three design choices turn an image diffusion model into a real-time policy. First, the denoised variable is the action sequence A_t = (a_t, ..., a_{t+T_p-1}), not a single action and not the observation. Because diffusion scales well to high-dimensional outputs, the whole action chunk is denoised jointly from one sample of the joint distribution, which automatically removes the jitter caused by independently sampling each timestep. Second, the observation is a condition, not part of the noised variable: the denoiser is epsilon_theta(O_t, A_t^k, k), modeling p(A_t | O_t). The vision encoder therefore runs once per control step rather than once per denoising iteration, enabling closed-loop real-time inference, and the encoder trains end-to-end through the action loss. Third, execution follows a receding-horizon scheme: predict T_p steps, execute the first T_a steps with 1 < T_a < T_p, then replan from the new observation. This trades temporal consistency against reactivity in the standard model-predictive-control way. Training uses the simplified noise-prediction objective L = E_{k, A^0, epsilon} || epsilon - epsilon_theta(O, sqrt(alpha_bar_k) A^0 + sqrt(1 - alpha_bar_k) epsilon, k) ||^2, with k sampled uniformly over the diffusion steps and epsilon standard Gaussian. Actions are normalized to the box [-1, 1] because the sampler clips the running prediction to that box; a zero-mean unit-variance normalization would make parts of the action range unreachable.

The code below is a self-contained low-dimensional, single-action implementation of the diffusion-behavior-cloning core. It removes the visuomotor sequence pieces for clarity while keeping the same training objective and ancestral sampler that the full method uses.

```python
import numpy as np
import torch
import torch.nn as nn


def cosine_alpha_sigma(t, s=0.008):
    # Variance-preserving square-cosine schedule used in iDDPM.
    # alpha_k = sqrt(alpha_bar_k), sigma_k = sqrt(1 - alpha_bar_k).
    alpha = (np.pi / 2.0 * (t.clip(0., 0.9946) + s) / (1 + s)).cos() / np.cos(np.pi / 2.0 * s / (1 + s))
    sigma = (1.0 - alpha ** 2).sqrt()
    return alpha, sigma


def positional_embed(k, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=k.device) / (half - 1))
    args = k[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim)
        )
        self.mid = nn.Sequential(
            nn.Linear(obs_dim + act_dim + emb_dim, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
        )
        self.head = nn.Linear(256, act_dim)

    def forward(self, x, k, obs):
        t = self.time_mlp(positional_embed(k, self.emb_dim))
        return self.head(self.mid(torch.cat([x, t, obs], dim=-1)))


class DiffusionPolicy:
    """Conditional noise-prediction diffusion for behavior cloning."""
    def __init__(self, obs_dim, act_dim, diffusion_steps=1000, lr=3e-4,
                 ema_rate=0.995, device="cpu"):
        self.K = diffusion_steps
        self.device = device
        self.act_low, self.act_high = -1.0, 1.0
        self.net = NoisePredMLP(obs_dim, act_dim).to(device)
        self.net_ema = NoisePredMLP(obs_dim, act_dim).to(device)
        self.net_ema.load_state_dict(self.net.state_dict())
        for p in self.net_ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.ema_rate = ema_rate
        t_grid = torch.linspace(1e-3, 1.0, self.K, device=device)
        self.alpha, self.sigma = cosine_alpha_sigma(t_grid)

    def add_noise(self, x0):
        k = torch.randint(self.K, (x0.shape[0],), device=self.device)
        eps = torch.randn_like(x0)
        a, s = self.alpha[k][:, None], self.sigma[k][:, None]
        return a * x0 + s * eps, k, eps

    def loss(self, act, obs):
        xt, k, eps = self.add_noise(act)
        return ((self.net(xt, k, obs) - eps) ** 2).mean()

    @torch.no_grad()
    def ema_update(self):
        for p, pe in zip(self.net.parameters(), self.net_ema.parameters()):
            pe.mul_(self.ema_rate).add_(p, alpha=1 - self.ema_rate)

    @torch.no_grad()
    def _clip_eps(self, eps, xt, a, s):
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
        stds[1:] = s[:-1] / s[1:] * (1 - (a[1:] / a[:-1]) ** 2).sqrt()
        xt = torch.randn(n_samples, act_dim, device=self.device) * temperature
        for i in reversed(range(1, steps + 1)):
            k = torch.full((n_samples,), int(sched[i]), dtype=torch.long, device=self.device)
            eps = self._clip_eps(net(xt, k, obs), xt, a[i], s[i])
            xt = (a[i - 1] / a[i]) * (xt - s[i] * eps) + (s[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps
            if i > 1:
                xt = xt + stds[i] * torch.randn_like(xt)
        return xt.clip(self.act_low, self.act_high)


from torch.optim.lr_scheduler import CosineAnnealingLR


def train(policy, dataloader, gradient_steps, device, ema_update_interval=1):
    sched = CosineAnnealingLR(policy.opt, T_max=gradient_steps)
    for step, batch in zip(range(gradient_steps), dataloader):
        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)
        loss = policy.loss(act, obs)
        policy.opt.zero_grad()
        loss.backward()
        policy.opt.step()
        sched.step()
        if step % ema_update_interval == 0 and step >= 1000:
            policy.ema_update()


@torch.no_grad()
def act_for(policy, obs_np, normalizer, num_envs, device):
    obs = torch.tensor(normalizer.normalize(obs_np), device=device, dtype=torch.float32)
    return policy.sample(obs, n_samples=num_envs, use_ema=True).cpu().numpy()
```
