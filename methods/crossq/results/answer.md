# CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency and Simplicity

## Problem it solves

Off-policy actor-critic methods for continuous control (DDPG/TD3/SAC) all bootstrap their Q-targets
through a *target network* — a slowly tracked copy of the critic — which is treated as necessary to
stop the bootstrap from chasing itself, but is deliberately stale and adds the τ knob. The
sample-efficient high-UTD methods (REDQ, DroQ) keep the target network and buy accuracy with large
ensembles and many gradient steps per environment step. CrossQ removes target networks entirely,
keeps the update-to-data ratio at 1 (one critic update per step, no ensemble), and matches or beats
high-UTD accuracy with only a few lines added to SAC.

## Key idea

The target network is really compensating for a **distribution mismatch**: a single critic is
evaluated on the current state-action batch (actions from replay) and the next state-action batch
(actions from the current policy), which are different input distributions, and a deep net relating
them through the Bellman equation can drift to different scales on the two clouds. The fix is to
normalize that mismatch away inside the critic:

1. **BatchRenorm critic.** Put Batch Renormalization (Ioffe 2017) — BatchNorm with clipped
   correction terms `(r, d)` that tie batch statistics to running statistics, robust to
   non-stationary RL data — into the critic.
2. **Joint forward pass.** Concatenate `(s,a)` and `(s',a')` into one batch and forward them
   *together* through the critic, then split. This forces both to be normalized under one shared
   distribution, so the prediction and the bootstrap value live on the same normalized scale. Naive
   BatchNorm with *separate* passes normalizes the two batches under different moments and
   destabilizes training — this is why BatchNorm had a bad reputation in RL.
3. **No target networks.** Joint-batch normalization supplies both the stationarity and the
   cross-batch consistency the target network was approximating, so it is deleted; the next-state
   value comes from the live critic (detached), normalized jointly with the prediction.

Everything else is SAC: stochastic tanh-Gaussian actor, twin critics with a `min` next-state target,
automatic entropy temperature tuning. Removing the target network lets the critic be widened (paper:
2048-unit hidden layers) and learn from an un-stale bootstrap at UTD = 1.

## Algorithm

Per environment step: act with the stochastic actor, store the transition, then run one update:
1. Sample minibatch `(s,a,r,s',d)`; sample `a' ∼ π(s')`, `log π(a'|s')` with no grad.
2. With the critics in train mode, forward `cat[(s,a),(s',a')]` through each twin critic and split
   into current/next halves.
3. Detach the next halves, take their elementwise `min`, subtract `α log π(a'|s')`, form
   `target = r + γ(1−d)(min Q(s',a') − α log π(a'|s'))`.
4. Critic loss `MSE(Q1_cur, target) + MSE(Q2_cur, target)`; step.
5. Every `policy_delay` steps: with critics in eval mode, sample `ã ∼ π(s)`, step the actor on
   `(α log π(ã|s) − min Q(s,ã)).mean()`, and update α on `−log α (log π + H_target)`,
   `H_target = −dim(A)`.

No target-network synchronization anywhere. Canonical hyperparameters: `γ=0.99`, Adam `1e-3`, batch
size 256, wide critics (2048), narrow actor (256), BatchRenorm momentum `0.01`, eps `1e-3`, clip
`rmax=3, dmax=5` widened over warmup, policy delay 3, UTD = 1.

## Working code

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class BatchRenorm1d(nn.Module):
    # Batch Renormalization (Ioffe 2017): batch-norm with clipped corrections
    # (r, d) that tie batch stats to running stats; robust to non-stationary RL data.
    def __init__(self, num_features, momentum=0.01, eps=1e-3, rmax=3.0, dmax=5.0):
        super().__init__()
        self.momentum, self.eps, self.rmax, self.dmax = momentum, eps, rmax, dmax
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            bmean = x.mean(0)
            bvar = x.var(0, unbiased=False)
            bstd = (bvar + self.eps).sqrt()
            rstd = (self.running_var + self.eps).sqrt()
            r = (bstd / rstd).detach().clamp(1.0 / self.rmax, self.rmax)
            d = ((bmean - self.running_mean) / rstd).detach().clamp(-self.dmax, self.dmax)
            xhat = (x - bmean) / bstd * r + d
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * bmean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * bvar.detach())
        else:
            xhat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return self.weight * xhat + self.bias


class Critic(nn.Module):
    # wide BatchRenorm Q-network; no target copy exists for this net
    def __init__(self, obs_dim, act_dim, hidden=2048):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + act_dim, hidden); self.bn1 = BatchRenorm1d(hidden)
        self.l2 = nn.Linear(hidden, hidden); self.bn2 = BatchRenorm1d(hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x).view(-1)


class Actor(nn.Module):
    # SAC stochastic tanh-Gaussian actor, unchanged
    def __init__(self, obs_dim, act_dim, max_action, hidden=256):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.max_action = max_action

    def forward(self, s):
        x = F.relu(self.l1(s)); x = F.relu(self.l2(x))
        log_std = torch.tanh(self.log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mu(x), log_std

    def sample(self, s):
        mu, log_std = self(s)
        normal = torch.distributions.Normal(mu, log_std.exp())
        u = normal.rsample()
        y = torch.tanh(u)
        a = y * self.max_action
        logp = normal.log_prob(u) - torch.log(self.max_action * (1 - y.pow(2)) + 1e-6)
        return a, logp.sum(1, keepdim=True)


class CrossQ:
    def __init__(self, obs_dim, act_dim, max_action, device,
                 gamma=0.99, lr=1e-3, policy_delay=3):
        self.device, self.gamma, self.policy_delay = device, gamma, policy_delay
        self.max_action, self.total_it = max_action, 0
        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.qf1 = Critic(obs_dim, act_dim).to(device)
        self.qf2 = Critic(obs_dim, act_dim).to(device)  # twin critics, NO targets
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def _joint(self, qf, s, a, s2, a2):
        # current and next forwarded TOGETHER so BatchRenorm shares one distribution
        q = qf(torch.cat([s, s2], 0), torch.cat([a, a2], 0))
        return q[: s.shape[0]], q[s.shape[0]:]

    def update(self, batch):
        self.total_it += 1
        s, s2, a, r, d = batch
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
        self.qf1.train(); self.qf2.train()
        q1_cur, q1_nxt = self._joint(self.qf1, s, a, s2, a2)
        q2_cur, q2_nxt = self._joint(self.qf2, s, a, s2, a2)
        with torch.no_grad():
            min_nxt = torch.min(q1_nxt.detach(), q2_nxt.detach()) - self.alpha * logp2.view(-1)
            target = r + (1 - d) * self.gamma * min_nxt
        q_loss = F.mse_loss(q1_cur, target) + F.mse_loss(q2_cur, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        if self.total_it % self.policy_delay == 0:
            self.qf1.eval(); self.qf2.eval()
            pi, logp = self.actor.sample(s)
            min_pi = torch.min(self.qf1(s, pi), self.qf2(s, pi))
            a_loss = (self.alpha * logp.view(-1) - min_pi).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            alpha_loss = (-self.log_alpha.exp() * (logp.detach().view(-1) + self.target_entropy)).mean()
            self.al_opt.zero_grad(); alpha_loss.backward(); self.al_opt.step()
            self.alpha = self.log_alpha.exp().item()
        return {"q_loss": q_loss.item()}
```
