# SPOT (Supported Policy OpTimization)

SPOT is an offline RL method that enforces the behavior-support constraint *directly* and
*pluggably*: it adds to a standard TD3 actor loss a single penalty equal to the negative estimated
behavior log-density of the action the policy takes, where the density is modeled by a conditional
VAE. Because the term is just a penalty on a plain deterministic policy, inference is one forward
pass, the algorithm is a small modification of TD3, and setting the penalty weight to zero recovers
the TD3 actor objective up to a detached positive Q-scale factor — which makes offline-to-online
fine-tuning a matter of decaying that weight.

## Problem it solves

Off-policy actor-critic on a fixed dataset suffers *extrapolation error*: the Bellman target
evaluates the critic at the actor's proposed action, which can be out-of-distribution, where the
critic extrapolates to arbitrarily large values; the maximizing actor is drawn to those actions
and the error bootstraps and diverges. The cure is the *support constraint* — keep the policy's
actions where the behavior policy `pi_beta` has density above a threshold. Prior enforcement was
either *parameterization* (BCQ/PLAS/EMaQ: density-correct but intrusive and slow at inference) or
*divergence regularization* (BEAR/TD3+BC: pluggable but enforces distributional/imitation
closeness, which mismatches the support condition and either over-constrains or leaks OOD actions).
SPOT is simultaneously pluggable and a direct match to the density-based support condition.

## Key idea

The support constraint `pi_beta(a|s) > eps` is a statement about the *value of the behavior density
at the action taken*, not a distance between distributions. So constrain it directly:

```
max_phi  E_{s~D}[ Q_theta(s, pi_phi(s)) ]   s.t.   min_s log pi_beta(pi_phi(s)|s) > log eps.
```

The per-state constraint (infinitely many states) is relaxed to the average over states, then
Lagrangianized into the unconstrained actor objective

```
J_pi(phi) = E_{s~D}[ -Q_theta(s, pi_phi(s)) - lambda * log pi_beta(pi_phi(s)|s) ],
```

a pluggable penalty whose coefficient `lambda` is the constraint-strength lever controlling the
same tradeoff as `eps`. Larger `lambda` -> tighter support; `lambda = 0` disables the support
penalty and leaves the TD3 actor direction, up to Q-normalization.

The constraint is *safe*: with the supported backup operator
`T_eps Q(s,a) = E[r + gamma max_{a': pi_beta(a'|s')>eps} Q(s',a')]` and
`alpha(eps) = ||T Q* - T_eps Q*||_inf`, the supported optimal value function obeys
`||Q* - Q*_eps||_inf <= alpha(eps)/(1-gamma)` (triangle inequality + gamma-contraction of `T_eps`).
`eps` (hence `lambda`) trades extrapolation risk against suboptimality.

## Density estimation (the regularizer)

`pi_beta` is unknown; estimate it with a conditional VAE (flexible enough for multimodal behavior,
unlike a single Gaussian behavior model). The ELBO lower-bounds the log-density:

```
log p_psi(a|s) >= E_{q_varphi(z|a,s)}[log p_psi(a|z,s)] - KL(q_varphi(z|a,s) || p(z|s)) = -L_ELBO,
```

with gap exactly `KL(q_varphi(z|a,s) || p_psi(z|a,s)) >= 0` (so `-L_ELBO` is a conservative lower
bound, the right direction for a "density above threshold" constraint). An `L`-sample
importance-weighted estimator tightens the bound (`-> log p` as `L -> inf`), but the practical
default uses `L = 1` because the KL remains analytic and the actor receives a lower-variance
constraint gradient. The regularizer is therefore the VAE ELBO loss evaluated at the policy action:
`neg_log_beta = recon + beta * KL`, proportional to the negative ELBO estimator up to constants,
with Gaussian-decoder reconstruction `mean((u - a)^2)` and analytic Gaussian KL
`-0.5 * mean(1 + log std^2 - mean^2 - std^2)`, `beta = 0.5`.

## Design choices and why

- **Density penalty, not a divergence:** matches the *support* definition exactly; a divergence is
  distributional and over-/under-constrains.
- **Average-constraint relaxation:** per-state constraint is intractable (infinite states); average
  is the standard CPO relaxation (TRPO/AWR/BEAR).
- **VAE estimator (not Gaussian):** captures multimodal behavior without forcing separated action
  modes into one smeared density.
- **`L = 1` ELBO:** the constraint needs a monotone signal, not a calibrated density; `L = 1` is
  simpler and lower-variance because the KL stays analytic.
- **TD3 base (not SAC):** deterministic actor + clipped twin critics + target smoothing suppress
  OOD overestimation; SAC's stochastic sampling and entropy bonus push toward / past the support.
- **Q-normalization (from TD3+BC):** divide the value term by `(1/N) sum_i |Q(s_i, pi(s_i))|`
  (detached) so a single `lambda` transfers across reward scales without changing the actor
  gradient direction except for a positive scale.
- **`lambda` cooling online + frozen VAE:** since `lambda = 0` removes the support penalty, fine-tune
  by decaying `lambda` toward a floor `lambda_end > 0` as online data arrives (relax conservatism
  without an early Q-collapse); freeze the VAE because behavior models are hard to update online.
  A floor (not zero) keeps the critic stable on hard sparse-reward tasks.

## Final algorithm

```
Phase 1 (VAE pretrain): for T1 steps, sample (s,a)~D, minimize  recon + beta*KL.   # freeze after
Phase 2 (policy training, TD3):
  for each step:
    a' = clip(pi_target(s') + clip(N(0, policy_noise), -c, c), -A, A)               # target smoothing
    y  = r + gamma * not_done * min(Q1_t(s',a'), Q2_t(s',a'))                       # clipped double-Q
    minimize MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
    every policy_freq steps:
      pi = pi_phi(s);  q = Q1(s, pi)
      neg_log_beta = L_ELBO(s, pi)                                                  # density penalty
      norm_q = 1 / mean(|q|).detach()                                              # Q normalization
      minimize  -norm_q * mean(q) + lambda * mean(neg_log_beta)
      soft-update targets with tau
Online fine-tuning: same loop on a growing buffer, with
  lambda_t = lambda * max(lambda_end, 1 - online_it / max_online_steps),  VAE frozen.
```

Defaults: VAE hidden 750, 3-layer enc/dec, latent dim `2*action_dim`, lr 1e-3, `10^5` iters,
`beta=0.5`. TD3: nets 2x256, `gamma=0.99` (0.995 on hard long-horizon mazes), `tau=0.005`,
`policy_noise=0.2`, `noise_clip=0.5`, `policy_freq=2`, Adam (critic 3e-4; actor 3e-4 offline, 1e-4
on AntMaze). `lambda` is the one tuned hyperparameter (a small grid).

## Working code

VAE density estimator:

```python
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class VAE(nn.Module):
    """Conditional VAE for the behavior-density penalty."""

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim

    def encode(self, state, action):
        h = F.relu(self.e1(torch.cat([state, action], -1)))
        h = F.relu(self.e2(h))
        log_std = self.log_std(h).clamp(-4, 15)
        return self.mean(h), torch.exp(log_std)

    def decode(self, state, z):
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        return self.decode(state, z), mean, std

    def elbo_loss(self, state, action, beta, num_samples=1):
        mean, std = self.encode(state, action)
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_s + std_s * torch.randn_like(std_s)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        u = self.decode(state_r, z)
        recon = ((u - action_r) ** 2).mean(dim=(1, 2))
        kl = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon + beta * kl

    def iwae_loss(self, state, action, beta, num_samples=10):
        return -self.importance_sampling_estimator(state, action, beta, num_samples)

    def importance_sampling_estimator(self, state, action, beta, num_samples=500):
        mean, std = self.encode(state, action)
        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_enc + std_enc * torch.randn_like(std_enc)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        mean_dec = self.decode(state_r, z)
        std_dec = torch.ones_like(mean_dec) * math.sqrt(beta / 4)
        log_qzx = td.Normal(mean_enc, std_enc).log_prob(z)
        log_pz = td.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)
        log_pxz = td.Normal(mean_dec, std_dec).log_prob(action_r)
        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        return w.logsumexp(dim=-1) - math.log(num_samples)
```

TD3 actor/critic and the SPOT algorithm:

```python
def weights_init_(m, init_w=3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dropout=None,
                 hidden_dim=256, init_w=None):
        super().__init__()
        if dropout:
            self.l1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Dropout(dropout))
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        if init_w:
            weights_init_(self.l3, init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return self.max_action * torch.tanh(a) if self.max_action is not None else a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=None):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        if init_w:
            weights_init_(self.l3, init_w)
            weights_init_(self.l6, init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class SPOT_TD3:
    def __init__(self, vae, state_dim, action_dim, max_action, device="cuda",
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2, beta=0.5, lambd=1.0, lr=3e-4, actor_lr=None,
                 without_Q_norm=False, num_samples=1, iwae=False,
                 actor_hidden_dim=256, critic_hidden_dim=256, actor_dropout=0.1,
                 actor_init_w=None, critic_init_w=None,
                 lambd_cool=False, lambd_end=0.2, max_online_steps=1_000_000):
        self.device = device
        self.total_it = 0
        self.vae = vae.eval()
        self.actor = Actor(state_dim, action_dim, max_action, dropout=actor_dropout,
                           hidden_dim=actor_hidden_dim, init_w=actor_init_w).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden_dim,
                             init_w=critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.max_action = max_action
        self.discount, self.tau = discount, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.beta, self.lambd, self.num_samples = beta, lambd, num_samples
        self.iwae = iwae
        self.without_Q_norm = without_Q_norm
        self.lambd_cool, self.lambd_end = lambd_cool, lambd_end
        self.max_online_steps = max_online_steps
        self.online_it = 0

    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            s = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(s).cpu().data.numpy().flatten()
            self.actor.train()
            return action

    def _train_step(self, replay_buffer, batch_size, lambd):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.Q1(state, pi)
            neg_log_beta = (self.vae.iwae_loss(state, pi, self.beta, self.num_samples)
                            if self.iwae else
                            self.vae.elbo_loss(state, pi, self.beta, self.num_samples))
            if self.without_Q_norm:
                actor_loss = -q.mean() + lambd * neg_log_beta.mean()
            else:
                norm_q = 1.0 / q.abs().mean().detach()
                actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train(self, replay_buffer, batch_size=256):
        self._train_step(replay_buffer, batch_size, self.lambd)

    def train_online(self, replay_buffer, batch_size=256, max_online_steps=None):
        self.online_it += 1
        max_online_steps = max_online_steps or self.max_online_steps
        lambd = (self.lambd * max(self.lambd_end, 1.0 - self.online_it / max_online_steps)
                 if self.lambd_cool else self.lambd)
        self._train_step(replay_buffer, batch_size, lambd)
```

VAE training (phase 1, run to convergence before policy training, then frozen):

```python
def pretrain_vae(vae, replay_buffer, iterations=100_000, batch_size=256, beta=0.5, lr=1e-3):
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for _ in range(iterations):
        state, action, *_ = replay_buffer.sample(batch_size)
        loss = vae.elbo_loss(state, action, beta).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae
```
