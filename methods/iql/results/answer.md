# IQL — Implicit Q-Learning for Offline RL

## Problem

Learn a policy from a fixed dataset `D` (collected by behavior policy `π_β`) with no environment
interaction, improving over `π_β` while never trusting the value of actions outside the data.
Standard Q-learning's target `max_{a'} Q(s', a')` queries out-of-distribution (OOD) actions, where
the value function extrapolates upward, and the policy then chases that error. Prior methods either
constrain the policy toward `π_β` or regularize `Q` down on OOD actions — both trading improvement
against robustness, and both still querying a learned `Q` at unseen actions. The goal: never query an
unseen action during value training, yet still perform multi-step dynamic programming (so the method
can "stitch" suboptimal trajectories).

## Key idea

Read `Q(s, a)` over the behavior action distribution as a per-state random variable. SARSA's MSE
recovers its *mean* (the value of `π_β`); what improvement needs is the *maximum over in-support
actions*. Estimate that upper tail by **expectile regression** — an asymmetric squared loss whose
τ-expectile approaches the supremum of the support as `τ → 1` — using only dataset actions, so no OOD
action is ever queried.

To avoid being optimistic about lucky stochastic transitions, split the estimate across two networks:
a value net `V` takes the upper expectile over **actions** (transition held fixed), and `Q` is backed
up onto `r + γ V(s')` with an honest **MSE** that averages over the **dynamics**. This is provably
multi-step dynamic programming — the resulting `V_τ` is monotone in `τ`, bounded by the in-support
optimum, and converges to it as `τ → 1`, spanning SARSA (`τ = 0.5`) to in-support Q-learning
(`τ → 1`).

Value training uses no policy and no OOD action. The policy is then extracted by **advantage-weighted
regression**, `exp(β(Q − V))·log π(a|s)` over dataset actions only — improvement with an implicit
constraint to `π_β`, again querying nothing unseen, and decoupled from value training (so it can run
concurrently or afterward, and gives a strong initialization for online finetuning).

## Algorithm

Expectile loss: `L_2^τ(u) = |τ − 1(u<0)|·u²` (τ = 0.5 is MSE; τ > 0.5 weights positive residuals more).

  Value:   `L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − V_ψ(s) ) ]`, with `Q_θ̂ = min(Q1, Q2)` (target critic)
  Q:       `L_Q(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ V_ψ(s') − Q_θ(s,a) )² ]`
  Policy:  `L_π(φ) = E_{(s,a)~D}[ −exp(β(Q_θ̂(s,a) − V_ψ(s)))·log π_φ(a|s) ]`, weight clipped to ≤ 100
  Target:  `θ̂ ← (1−α)θ̂ + αθ` (Polyak)

Per step: update `V`, then policy, then `Q`, then Polyak the target critic.

Theory: with `V_τ(s) = E^τ_{a~π_β}[Q_τ(s,a)]`, `Q_τ(s,a) = r + γ E_{s'}[V_τ(s')]`, one has (i)
`τ₁ < τ₂ ⟹ V_{τ₁} ≤ V_{τ₂}` (policy-improvement monotonicity), (ii) `V_τ(s) ≤ max_{a:π_β>0} Q*(s,a)`
(an expectile is ≤ the max), and (iii) `lim_{τ→1} V_τ(s) = max_{a:π_β(a|s)>0} Q*(s,a)`.

Canonical hyperparameters: 2-layer 256-unit MLPs, ReLU; Adam lr 3e-4 for all three nets; cosine decay
on the actor lr; `γ = 0.99`, Polyak `α = 0.005`; batch 256; 1M gradient steps. `τ = 0.7, β = 3.0` for
MuJoCo locomotion; `τ = 0.9, β = 10.0` for Ant Maze; `τ = 0.7, β = 0.5` and actor dropout 0.1 for
Kitchen/Adroit. The Gaussian policy has a state-independent std and a tanh-bounded mean, without a
tanh-squashed log-prob distribution. Rewards are preprocessed per D4RL convention (locomotion scaled
by the spread of trajectory returns; ant-maze shifted by −1).

## Code

```python
import copy
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__(); self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__(); self.net = mlp([obs_dim, *hidden, 1])
    def forward(self, s):
        return self.net(s).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.mean = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))      # state-independent std
    def dist(self, s):
        mean = torch.tanh(self.mean(s))                        # official code bounds the Gaussian mean
        return Normal(mean, self.log_std.clamp(-5.0, 2.0).exp())
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)
    def act(self, s):
        return torch.tanh(self.mean(s))


def expectile_loss(diff, tau):                                 # asymmetric L2
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff ** 2).mean()


def update_v(value, target_critic, batch, tau):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.minimum(q1, q2)                              # clipped double-Q
    return expectile_loss(q - value(s), tau)                   # V <- tau-expectile of Q over data actions


def update_actor(policy, target_critic, value, batch, beta):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q1, q2 = target_critic(s, a)
        q = torch.minimum(q1, q2)
        adv = q - value(s)
        weight = torch.clamp(torch.exp(beta * adv), max=100.0)
    return -(weight * policy.log_prob(s, a)).mean()            # advantage-weighted regression


def update_q(critic, value, batch, discount):
    s, a, r, s2, mask = (batch["obs"], batch["act"], batch["rew"],
                         batch["obs2"], batch["mask"])          # mask = 1 - done
    with torch.no_grad():
        target_q = r + discount * mask * value(s2)             # honest MSE; V already picked the action
    q1, q2 = critic(s, a)
    return ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()


def polyak(critic, target_critic, alpha):
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        tp.data.mul_(1 - alpha).add_(alpha * p.data)


def update(batch, critic, target_critic, value, policy, opts, hp):
    v_loss = update_v(value, target_critic, batch, hp["tau"])
    opts["v"].zero_grad(); v_loss.backward(); opts["v"].step()

    a_loss = update_actor(policy, target_critic, value, batch, hp["beta"])
    opts["pi"].zero_grad(); a_loss.backward(); opts["pi"].step()

    q_loss = update_q(critic, value, batch, hp["discount"])
    opts["q"].zero_grad(); q_loss.backward(); opts["q"].step()

    polyak(critic, target_critic, hp["alpha"])


def train_offline(dataset, critic, value, policy, hp, steps=int(1e6), batch_size=256):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q":  torch.optim.Adam(critic.parameters(), lr=3e-4),
        "v":  torch.optim.Adam(value.parameters(),  lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4),
    }
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opts["pi"], steps)
    for _ in range(steps):
        update(dataset.sample(batch_size), critic, target_critic, value, policy, opts, hp)
        sched.step()

# hp = dict(tau=0.7, beta=3.0, discount=0.99, alpha=0.005)   # locomotion
# hp = dict(tau=0.9, beta=10.0, discount=0.99, alpha=0.005)  # antmaze
```
