# AWAC — Advantage-Weighted Actor-Critic

## Problem

Pre-train a policy from a fixed offline dataset (of arbitrary quality — demonstrations, suboptimal, or
random) and then keep improving it with a small amount of online interaction. This requires three
things at once that prior methods cannot combine: efficient off-policy reuse of the data, stability
when training on a static dataset (no exploding bootstrap error from out-of-distribution actions), and
unobstructed online improvement. Naive off-policy actor-critic (SAC) is efficient but unstable offline;
explicit-constraint offline RL (BCQ/BEAR/BRAC) is stable offline but stalls online because the
behavior model it constrains to cannot be re-fit accurately on streaming data; advantage-weighted
Monte-Carlo methods (AWR) are stable but data-inefficient.

## Key idea

Use an off-policy actor-critic, but enforce the stay-near-the-data constraint **implicitly**, with no
explicit behavior model. Solve the KL-constrained policy-improvement problem exactly and project its
solution onto the parametric policy by **forward** KL, which lets the behavior-policy factor cancel:

  `max_π E_{a~π}[A^{π_k}(s,a)]  s.t.  KL(π‖π_β) ≤ ε`  ⟹  `π*(a|s) ∝ π_β(a|s)·exp(A^{π_k}(s,a)/λ)`,

and projecting gives the actor update

  `θ_{k+1} = argmax_θ E_{(s,a)~β}[ log π_θ(a|s) · exp(A^{π_k}(s,a)/λ) ]`,

an advantage-weighted maximum-likelihood (supervised) update on **buffer actions only** — so it stays
in the data distribution without any behavior model and never queries `Q` at a policy-proposed
out-of-distribution action during improvement. The advantage uses an **off-policy bootstrapped `Q^π`**
of the *current* policy (twin-Q, min target, Polyak) — not a Monte-Carlo `V^{π_β}` — which is what makes
it sample-efficient and able to improve past a single step. The same update runs unchanged offline and
online.

## Algorithm

Critic (policy evaluation), twin-Q TD:
  `L(φ) = E_D[(Q_φ(s,a) − y)²]`,  `y = r + γ·min_i Q_{φ̄ᵢ}(s', a')`,  `a' ~ π(·|s')`.

Actor (policy improvement), advantage-weighted MLE:
  `A(s,a) = Q(s,a) − V(s)`,  `V(s) = E_{a~π}[Q(s,a)]` (estimated at a policy sample),
  `L(θ) = −E_{(s,a)~β}[ exp(A(s,a)/λ) · log π_θ(a|s) ]`, weights normalized over the batch.

Derivation: the Lagrangian `E_π[A] + λ(ε − KL(π‖π_β)) + α(1 − ∫π)` gives, on `∂/∂π = 0`,
`π*(a|s) = (1/Z(s))·π_β(a|s)·exp(A/λ)`. Forward-KL projection
`argmin_θ E_{π*}[−log π_θ]` is importance-sampled from the buffer, `E_{π*}[·] = E_{π_β}[(1/Z)exp(A/λ)·]`,
canceling `π_β`. The per-state `Z(s) = E_{a~π_β}[exp(A/λ)]` is dropped (estimating it empirically hurt,
and it only reweights states, not actions) in favor of batch normalization. `λ` (the KL Lagrange
multiplier) is a fixed temperature: small `λ` → greedier; large `λ` → closer to behavioral cloning.

Hyperparameters: twin-SAC base with AWAC examples setting entropy `α=0`; Adam lr 3e-4 (actor
weight-decay 1e-4, critic 0); discount 0.99; Polyak `τ = 5e-3`; ReLU MLPs; batch 1024; replay 1e6;
offline RL pretraining before online collection; one train batch per online env step. I use
`λ = 0.3` for manipulation and `λ = 1.0` for MuJoCo; rlkit names the same denominator `beta` and its
example scripts sweep/use nearby values (`0.5` with clipping for hand, `2` for MuJoCo).

## Code

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__(); self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
    def dist(self, s):
        h = self.trunk(s)
        return Normal(torch.tanh(self.mu(h)), self.log_std(h).clamp(-6, 0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
    def sample_and_log_prob(self, s):
        dist = self.dist(s)
        a = dist.rsample()
        return a, dist.log_prob(a).sum(-1)
    def sample(self, s):
        return self.sample_and_log_prob(s)[0]


def critic_td_loss(batch, critic, target_critic, policy, discount, alpha=0.0):
    s, a, r, s2, done = (batch["obs"], batch["act"], batch["rew"],
                         batch["obs2"], batch["done"])
    with torch.no_grad():
        a2, logp2 = policy.sample_and_log_prob(s2)           # next action from current policy
        tq1, tq2 = target_critic(s2, a2)
        target_v = torch.min(tq1, tq2) - alpha * logp2       # AWAC configs set alpha=0
        y = r + discount * (1.0 - done) * target_v
    q1, q2 = critic(s, a)
    return F.mse_loss(q1, y) + F.mse_loss(q2, y)


def actor_awac_loss(batch, critic, policy, lam):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q = torch.min(*critic(s, a))                        # Q(s, a_data)
        v = torch.min(*critic(s, policy.sample(s)))         # V(s) = Q(s, a~pi)
        adv = q - v                                         # advantage
        weight = torch.softmax(adv / lam, dim=0) * adv.numel()
    logp = policy.log_prob(s, a)                            # MLE on buffer actions -> implicit constraint
    return -(weight * logp).mean()


def polyak(critic, target_critic, tau):
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def update(batch, critic, target_critic, policy, opts, hp):
    q_loss = critic_td_loss(batch, critic, target_critic, policy,
                            hp["discount"], hp.get("alpha", 0.0))
    opts["q"].zero_grad(); q_loss.backward(); opts["q"].step()

    pi_loss = actor_awac_loss(batch, critic, policy, hp["lam"])
    opts["pi"].zero_grad(); pi_loss.backward(); opts["pi"].step()

    polyak(critic, target_critic, hp["tau"])


def train_awac(env, buffer, critic, policy, hp,
               pretrain_steps=25000, online_steps=int(1e6)):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q":  torch.optim.Adam(critic.parameters(), lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-4),
    }
    for _ in range(pretrain_steps):                         # offline: buffer holds the dataset
        update(buffer.sample(hp["batch_size"]), critic, target_critic, policy, opts, hp)
    o = env.reset()
    for _ in range(online_steps):                           # online: identical update
        a = policy.sample(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
        o2, r, done, _ = env.step(a)
        buffer.add(o, a, r, o2, done)
        o = env.reset() if done else o2
        update(buffer.sample(hp["batch_size"]), critic, target_critic, policy, opts, hp)

# hp = dict(discount=0.99, tau=5e-3, alpha=0.0, batch_size=1024, lam=0.3)  # manipulation
# hp = dict(discount=0.99, tau=5e-3, alpha=0.0, batch_size=1024, lam=1.0)  # MuJoCo locomotion
```
