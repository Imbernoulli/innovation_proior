# SAC-RND — Anti-Exploration by Random Network Distillation

## Problem

Offline RL on a fixed dataset `D`: a bootstrapped critic over-values out-of-distribution (OOD) actions
the policy proposes, and the policy chases the inflation, with no new data to correct it. Express the
cure as *uncertainty*: penalize value and policy in proportion to how unfamiliar a `(s, a)` is — the
anti-exploration mirror of an online novelty bonus. Ensembles (SAC-N/EDAC) supply that uncertainty but
pay `N` critics for it; the goal is ensemble-quality pessimism from a single small network.

## Key idea

Use **Random Network Distillation** as the OOD-action detector: a frozen random target `g(s,a)`, a
predictor `ĝ(s,a)` trained on dataset `(s,a)`, novelty `b(s,a) = ‖ĝ(s,a) − g(s,a)‖²`. Naive
`[s,a]`-concatenated RND is *escapable* — the actor can drive `b` down on OOD actions without returning
to the data ("not discriminative enough"). The fix is the **conditioning**: condition the RND prior on
the action via **FiLM** (the state is the feature stream; the action emits per-unit `(γ, β)` that
scale/shift a hidden layer — or vice-versa), so the predictor's error is a *sharp* function of the
action, minimizable only by proposing an in-distribution action. Subtract `β·b` from **both** the actor
objective and the critic bootstrap target.

## Algorithm

Base SAC (continuous): Tanh-Gaussian actor; twin LayerNorm critics with `min`; automatic entropy
temperature `α` tuned to target entropy `−dim(A)`; Polyak targets.

- RND: `b(s,a) = ‖ĝ(s,a) − g(s,a)‖² / rms.std`, target frozen, predictor trained by MSE on dataset
  `(s,a)`; FiLM/bilinear conditioning of the prior on the action (`switch_features` in the reference).
- Critic target: `y = r + γ(1−d)·[min_i Q_i^tgt(s',a') − α·logπ(a'|s') − β·b(s',a')]`, `a'∼π(·|s')`.
- Actor: minimize `(α·logπ(a|s) + β·b(s,a) − min_i Q_i(s,a)).mean()`, `a∼π(·|s)`.
- Temperature: minimize `−(logα · (logπ + target_entropy).detach()).mean()`.

Canonical hyperparameters: hidden 256, RND embedding 32, `num_critics = 2`, `critic_layernorm = True`,
`γ = 0.99`, Polyak `τ = 5e-3`, Adam `3e-4` (medium configs use `1e-3`, batch 1024); actor/critic
`log_σ` clipped to `(−5, 2)`. Penalty coefficient `β` is swept per dataset — e.g.
halfcheetah-medium-v2 `β = 0.3`, walker2d-medium-v2 `β = 8.0`.

Reported D4RL Gym (normalized): halfcheetah-medium-v2 `66.6 ± 1.6`, walker2d-medium-v2 `91.6 ± 2.8`,
Gym average `85.2` — ahead of ensemble-free TD3+BC (67.5), IQL (68.9), CQL (73.6) and matching the
ensemble methods SAC-N (84.4) / EDAC (85.2).

## Code

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FiLMMLP(nn.Module):
    """MLP over `feature`, FiLM-modulated by `context`."""
    def __init__(self, feature_dim, context_dim, out_dim, hidden=256):
        super().__init__()
        self.film = nn.Linear(context_dim, 2 * hidden)
        self.l1 = nn.Linear(feature_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, feature, context):
        gamma, beta = torch.chunk(self.film(context), 2, dim=-1)
        h = F.relu(gamma * self.l1(feature) + beta)
        h = F.relu(self.l2(h))
        return self.out(h)


class RND(nn.Module):
    """Anti-exploration novelty over (s, a): action conditions the prior."""
    def __init__(self, obs_dim, act_dim, embedding_dim=32):
        super().__init__()
        self.predictor = FiLMMLP(act_dim, obs_dim, embedding_dim)
        self.target = FiLMMLP(act_dim, obs_dim, embedding_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.register_buffer("rms_var", torch.ones(()))

    def _embed(self, s, a):
        pred = self.predictor(a, s)
        with torch.no_grad():
            targ = self.target(a, s)
        return pred, targ

    def bonus(self, s, a):
        pred, targ = self._embed(s, a)
        return ((pred - targ) ** 2).sum(-1) / (torch.sqrt(self.rms_var) + 1e-8)

    def distill_loss(self, s, a):
        pred, targ = self._embed(s, a)
        raw = ((pred - targ) ** 2).sum(-1)
        with torch.no_grad():
            self.rms_var.mul_(0.99).add_(0.01 * raw.var(unbiased=False))
        return raw.mean()


class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_sigma = nn.Linear(hidden, act_dim)

    def sample(self, s):
        h = self.trunk(s)
        dist = Normal(self.mu(h), self.log_sigma(h).clamp(-5.0, 2.0).exp())
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1) - torch.log(1 - torch.tanh(raw) ** 2 + 1e-6).sum(-1)
        return torch.tanh(raw), logp

    @torch.no_grad()
    def act(self, s):
        return torch.tanh(self.mu(self.trunk(s)))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


def update(batch, actor, critics, critic_targs, rnd, log_alpha, target_entropy,
           opts, beta, gamma=0.99, tau=5e-3):
    s, a, r, s2, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
    alpha = log_alpha.exp()

    rnd_loss = rnd.distill_loss(s, a)
    opts["rnd"].zero_grad(); rnd_loss.backward(); opts["rnd"].step()

    with torch.no_grad():
        a2, logp2 = actor.sample(s2)
        q2 = torch.min(critic_targs[0](s2, a2), critic_targs[1](s2, a2))
        q2 = q2 - alpha * logp2 - beta * rnd.bonus(s2, a2)
        target = r + (1 - done) * gamma * q2
    c_loss = sum(F.mse_loss(c(s, a), target) for c in critics)
    opts["critic"].zero_grad(); c_loss.backward(); opts["critic"].step()

    pi, logp = actor.sample(s)
    q_pi = torch.min(critics[0](s, pi), critics[1](s, pi))
    a_loss = (alpha.detach() * logp + beta * rnd.bonus(s, pi) - q_pi).mean()
    opts["actor"].zero_grad(); a_loss.backward(); opts["actor"].step()

    alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
    opts["alpha"].zero_grad(); alpha_loss.backward(); opts["alpha"].step()

    for c, ct in zip(critics, critic_targs):
        for p, tp in zip(c.parameters(), ct.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
# beta: per-dataset (halfcheetah-medium 0.3, walker2d-medium 8.0); target_entropy = -act_dim
```
