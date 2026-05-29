# Soft Actor-Critic (SAC)

## Problem it solves
A model-free, off-policy deep RL algorithm for continuous control that is both sample-efficient
(reuses a replay buffer) and stable across random seeds with little per-task tuning — combining the
data efficiency of off-policy actor-critics (like DDPG) with stability and exploration that DDPG's
brittle deterministic actor lacks, and avoiding the intractable energy-based sampling of soft
Q-learning.

## Key idea
Optimize the **maximum-entropy objective** — reward plus policy entropy —

  J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[ r(s_t,a_t) + α·H(π(·|s_t)) ]

with a true actor-critic derived from **soft policy iteration**:

- The implementation below uses reward scaling c = 1/α, i.e. scaled reward r̃ = c r, so the remaining
  formulas set the entropy coefficient to 1. With explicit α, replace each `−log π` term by
  `−α log π`.
- **Soft policy evaluation** computes the current policy's soft value by iterating the soft Bellman
  backup T^π Q(s,a) = r̃(s,a) + γ E_{s'}[V(s')], where V(s) = E_{a~π}[Q(s,a) − log π(a|s)]. This is
  a γ-contraction because the entropy term is fixed for a fixed policy and cancels when two
  Q-functions are subtracted.
- **Soft policy improvement** updates the policy by the **information projection** of a tractable
  policy class Π onto the exponentiated Q,
  π_new = argmin_{π'∈Π} D_KL( π'(·|s) ‖ exp(Q^{π_old}(s,·))/Z(s) ).
  Because π_old∈Π is feasible, the minimizer is no worse, and bootstrapping the soft Bellman
  equation shows Q^{π_new} ≥ Q^{π_old} everywhere. Alternating the two steps converges to the
  optimal policy *within Π*, for any policy parameterization.

The KL projection is what removes the intractable partition function Z (it is independent of the
policy parameters and drops from the gradient) and makes this a genuine actor-critic rather than an
energy-based-sampling Q-learning method.

## Practical algorithm
Approximate the tabular steps with function approximators (Q-networks, a value network, a policy
network) and take SGD steps instead of running each step to convergence. Three objectives:

- **Value** V_ψ regresses onto the soft value of the current policy:
  J_V(ψ) = E_{s~D}[ ½(V_ψ(s) − E_{a~π_φ}[min_i Q_{θ_i}(s,a) − log π_φ(a|s)])² ].
- **Q** (two of them) regresses onto the soft Bellman target through a slow target value net:
  J_Q(θ) = E_{(s,a)~D}[ ½(Q_θ(s,a) − (c r + γ(1-d)V_{ψ̄}(s')))² ];  ψ̄ ← τψ + (1−τ)ψ̄.
- **Policy** minimizes the KL via the **reparameterization trick** a = f_φ(ε;s), ε~N(0,I):
  J_π(φ) = E_{s~D, ε}[ log π_φ(f_φ(ε;s)|s) − min_i Q_{θ_i}(s, f_φ(ε;s)) ].
  Its pathwise gradient contains
  ∇_φ log π_φ(a|s) + (∇_a log π_φ(a|s) − ∇_a min_i Q_{θ_i}(s,a))∇_φ f_φ(ε;s): the loss has a
  `−∇_a Q` term, so gradient descent moves the sampled action uphill in Q while the log-prob term
  preserves entropy.

Components: a **squashed Gaussian** policy (Gaussian sample u, action a = tanh(u)) with the
change-of-variables log-prob correction log π(a|s) = log μ(u|s) − Σ_i log(1 − tanh²(u_i));
the stable computation uses log(1 − tanh²(u)) = 2(log 2 − u − softplus(−2u));
**two Q-functions**, using their **minimum** wherever Q feeds the value/policy updates, to curb
overestimation; a slowly-tracked **target value network**; and **reward scaling** as the inverse
entropy temperature — the single hyperparameter that needs per-task tuning (too small → near-uniform
policy that ignores reward; too large → near-deterministic policy stuck in poor optima).

Defaults: Adam, lr 3e-4, γ=0.99, replay buffer 1e6, batch 256, two hidden layers of 256 ReLU units,
τ=0.005, one environment step per gradient step. Evaluation uses the mean action.

## Algorithm (pseudocode)
```
Initialize θ1, θ2 (Q), ψ (V), ψ̄ ← ψ (target V), φ (policy), replay buffer D
for each environment step:
    a ~ π_φ(·|s);  s', r ← env.step(a);  D ← D ∪ {(s,a,r,s')}
    for each gradient step:
        sample minibatch from D
        θ_i ← θ_i − λ_Q ∇_{θ_i} J_Q(θ_i)   for i ∈ {1,2}
        ψ  ← ψ  − λ_V ∇_ψ J_V(ψ)
        φ  ← φ  − λ_π ∇_φ J_π(φ)
        ψ̄ ← τ ψ + (1−τ) ψ̄
```

## Working code (PyTorch)
```python
import math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20, 2

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden), nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))
    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu(h)
        std = torch.exp(torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX))
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()          # reparameterized sample
        logp = None
        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            correction = 2 * (math.log(2.0) - u - F.softplus(-2 * u))
            logp = logp - correction.sum(-1)                 # tanh correction
        return torch.tanh(u) * self.act_limit, logp

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden) + [1])
    def forward(self, o, a): return self.q(torch.cat([o, a], -1)).squeeze(-1)

class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden) + [1])
    def forward(self, o): return self.v(o).squeeze(-1)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.o  = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a  = np.zeros((size, act_dim), np.float32)
        self.r  = np.zeros(size, np.float32)
        self.d  = np.zeros(size, np.float32)
        self.ptr, self.size, self.max = 0, 0, size
    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.max; self.size = min(self.size + 1, self.max)
    def sample(self, bs):
        idx = np.random.randint(0, self.size, bs)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return dict(obs=t(self.o[idx]), act=t(self.a[idx]), rew=t(self.r[idx]),
                    obs2=t(self.o2[idx]), done=t(self.d[idx]))

class SAC:
    def __init__(self, obs_dim, act_dim, act_limit, hidden=(256, 256),
                 gamma=0.99, tau=0.005, lr=3e-4, reward_scale=5.0):
        self.actor   = SquashedGaussianActor(obs_dim, act_dim, hidden, act_limit)
        self.q1      = QFunction(obs_dim, act_dim, hidden)
        self.q2      = QFunction(obs_dim, act_dim, hidden)
        self.vf      = ValueFunction(obs_dim, hidden)
        self.vf_targ = deepcopy(self.vf)
        for p in self.vf_targ.parameters(): p.requires_grad = False
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr)
        self.opt_q  = torch.optim.Adam(list(self.q1.parameters()) +
                                       list(self.q2.parameters()), lr)
        self.opt_v  = torch.optim.Adam(self.vf.parameters(), lr)
        self.gamma, self.tau, self.reward_scale = gamma, tau, reward_scale

    def _set_q_requires_grad(self, requires_grad):
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad = requires_grad

    def update(self, data):
        o, a, r, o2, d = (data['obs'], data['act'], data['rew'],
                          data['obs2'], data['done'])
        # --- Q update: soft Bellman residual toward reward_scale * r + gamma * V_targ(s') ---
        with torch.no_grad():
            q_target = self.reward_scale * r + self.gamma * (1 - d) * self.vf_targ(o2)
        loss_q = 0.5 * ((self.q1(o, a) - q_target) ** 2).mean() + \
                 0.5 * ((self.q2(o, a) - q_target) ** 2).mean()
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        # fresh action from current policy (on-policy term inside the off-policy loop)
        # --- V update: regress onto E_{a~pi}[min Q - log pi] ---
        with torch.no_grad():
            a_pi, logp = self.actor(o)
            q_pi = torch.min(self.q1(o, a_pi), self.q2(o, a_pi))
            v_target = q_pi - logp
        loss_v = 0.5 * ((self.vf(o) - v_target) ** 2).mean()
        self.opt_v.zero_grad(); loss_v.backward(); self.opt_v.step()

        # --- policy update: minimize KL == E[log pi - Q] via reparameterization ---
        # Freeze Q weights, but keep the Q(a) computation differentiable with respect to a.
        self._set_q_requires_grad(False)
        try:
            a_pi, logp = self.actor(o)
            q_pi = torch.min(self.q1(o, a_pi), self.q2(o, a_pi))
            loss_pi = (logp - q_pi).mean()
            self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
        finally:
            self._set_q_requires_grad(True)

        # --- slow-tracking target value network ---
        with torch.no_grad():
            for p, pt in zip(self.vf.parameters(), self.vf_targ.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic, with_logprob=False)
        return a.cpu().numpy()

def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs2, reward, terminated, truncated, info = out
        return obs2, reward, terminated or truncated, info
    obs2, reward, done, info = out
    return obs2, reward, done, info

def train(env, agent, steps=int(1e6), batch_size=256, start_steps=10000):
    buf = ReplayBuffer(env.observation_space.shape[0],
                       env.action_space.shape[0], int(1e6))
    o = reset_env(env)
    for t in range(steps):
        a = env.action_space.sample() if t < start_steps else agent.act(o)
        o2, r, done, _ = step_env(env, a)
        buf.store(o, a, r, o2, float(done))
        o = reset_env(env) if done else o2
        if t >= start_steps and buf.size >= batch_size:
            agent.update(buf.sample(batch_size))
```

The value target is detached, the policy objective is recomputed after the Q and V updates, and Q
parameters are frozen during the policy step so gradients flow through Q to the action but do not
update the critics.
