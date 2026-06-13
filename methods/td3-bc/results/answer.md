# TD3+BC, distilled

TD3+BC is a minimalist offline reinforcement-learning algorithm: take TD3 unchanged and make
exactly two additions — a behavior-cloning regularizer on the actor objective (with a single
scale-free coefficient), and zero-mean/unit-variance normalization of the dataset's state
features. The actor regularizer reuses the same minibatch TD3 already samples and fits no extra
behavior model.

## Problem it solves

Learn a continuous-control policy from a fixed dataset `D = {(s, a, r, s')}` with no environment
interaction. Off-policy actor-critic methods collapse offline: the actor climbs the critic
toward out-of-distribution actions, where the critic's value is an unconstrained (typically
over-) estimate; with no fresh data to refute it, the error backs up through the Bellman
recursion and the actor-critic loop diverges. The policy must be kept inside the data's action
support so the critic is only queried where it has support — and, because nothing can be
validated offline, with the fewest possible added components and hyperparameters.

## Key idea

Keep the policy near the data with the cheapest possible constraint. Because the actor `π_φ(s)`
is *deterministic* and the dataset gives one action `a` per state `s`, "stay near the data" is
not a divergence between distributions (which would force a fitted behavior model) but a point
distance `(π(s) - a)^2` — plain behavior cloning. Add it as a regularizer onto TD3's
value-maximizing actor objective:

```
π = argmax_π  E_{(s,a)~D} [ λ · Q(s, π(s)) − (π(s) − a)^2 ].
```

The `Q` term drives improvement; the BC term tethers the actor inside the support, starving the
extrapolation that caused the offline collapse — with no generative model and one extra line.

**Scale-free coefficient.** Assuming actions in `[-1, 1]`, the mean-squared BC term is bounded
by `4`, but `Q` scales with the reward magnitude, which differs by orders of magnitude across
tasks. A fixed `λ` therefore cannot transfer. Normalize the value term by the batch mean
absolute actor-update `Q`:

```
λ = α / ( (1/N) Σ_i |Q_{θ1}(s_i, π_φ(s_i))| ),     α = 2.5.
```

Then `λ·Q ≈ α`, pinned regardless of reward scale, so a single dimensionless `α` sets the
RL/BC balance across all tasks. `λ` is a *scalar scale*: it is detached (not differentiated
through), so only `Q(s, π(s))` and `(π(s) − a)^2` carry gradient. As a side effect this also
normalizes the actor's effective learning rate, since `∇_a Q` scales with the reward scale too.

**State normalization.** The dataset is fixed, so exact per-feature statistics are free:

```
s_i ← (s_i − μ_i) / (σ_i + ε),     ε = 1e-3,
```

with `μ_i, σ_i` the per-feature mean/std over `D`, applied to both `s` and `s'` (and to live
observations at evaluation via the stored `μ, σ`). A near-free stabilizer, not the core change.

## What stays exactly TD3

Twin critics `Q_{θ1}, Q_{θ2}` fit by MSE to the clipped-double-Q target
`y = r + γ(1−d)·min_i Q_{θ'_i}(s', ã)`; target policy smoothing
`ã = clip(π_{φ'}(s') + clip(N(0,σ), −c, c), −a_max, a_max)`, `σ = 0.2·a_max`, `c = 0.5·a_max`;
delayed actor and soft target updates (`τ = 5e-3`) every `policy_freq = 2` critic steps;
`γ = 0.99`, Adam `3e-4`, batch 256, `256×256` ReLU MLPs. The *only* algorithmic change is the
actor objective; the *only* implementation change is state normalization.

## Defaults and why

- `α = 2.5`: the dimensionless RL/BC ratio after normalization. Small `α` → pure imitation
  (ceilinged by the data); large `α` → BC tether goes slack, drift back to the offline blow-up;
  `2.5` is value-led but firmly tethered, and is task-invariant because `λ` already absorbed the
  reward scale.
- `ε = 1e-3` in state normalization guards near-constant features.
- Everything else inherited from TD3 unchanged.

## Final algorithm

```
Precompute μ, σ over D; normalize s, s' ← (s − μ)/(σ + ε), ε = 1e-3.
Set target smoothing σ_n = 0.2·a_max and c = 0.5·a_max.
for each gradient step:
    sample (s, a, r, s', not_done) ~ D
    # critic (TD3, unchanged):
    ã    = clip(π_{φ'}(s') + clip(N(0,σ_n), −c, c), −a_max, a_max)
    y    = r + γ · not_done · min(Q_{θ'_1}(s', ã), Q_{θ'_2}(s', ã))
    minimize  (Q_{θ1}(s,a) − y)^2 + (Q_{θ2}(s,a) − y)^2
    # actor (every policy_freq=2 steps): the one algorithmic change
    if step % policy_freq == 0:
        λ = α / mean_i |Q_{θ1}(s_i, π_φ(s_i))|   # detached scalar
        maximize  λ · Q_{θ1}(s, π_φ(s)) − (π_φ(s) − a)^2
        soft-update target networks: θ' ← τθ + (1−τ)θ'
```

## Relation to prior methods

- **BCQ / BEAR / BRAC**: keep the policy near the data via a *fitted* behavior model `π̂_β`
  (VAE or MLE density) and a divergence to it; TD3+BC replaces the whole apparatus with a point
  L2 to the dataset action, available for free from a deterministic actor — no model, no
  divergence choice.
- **CQL**: regularizes the critic (pessimism on OOD actions) via a logsumexp over sampled
  actions; TD3+BC regularizes the actor instead, with no sampling and no extra forward passes.
- **Fisher-BRC**: behavior model + offset critic + gradient penalty + reward bonus; TD3+BC keeps
  the base TD3 critic and adds only the actor regularizer plus state normalization.
- **BC**: TD3+BC is BC plus a value term, so it can exceed the data instead of merely copying it;
  setting `α → 0` recovers (normalized) behavior cloning.

## Working code

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_states(states, next_states, eps=1e-3):
    """Per-feature zero-mean/unit-variance over the fixed dataset; eps guards near-constant features.
    Return (mean, std) to normalize live observations the same way at evaluation time."""
    mean = states.mean(0, keepdims=True)
    std = states.std(0, keepdims=True) + eps
    return (states - mean) / std, (next_states - mean) / std, mean, std


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action     # smoothing noise scaled by action range
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha                                 # the single hyperparameter (RL/BC ratio)
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ---- critic: untouched TD3 (clipped double-Q + target policy smoothing) ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)     # min over twins: no extra overestimation
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- delayed actor update: the one algorithmic change ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()   # scalar normalizer, detached -> no gradient
            # minimize -(lambda*Q) + (pi - a)^2  <=>  maximize  lambda*Q(s, pi(s)) - (pi(s) - a)^2
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft target updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```
