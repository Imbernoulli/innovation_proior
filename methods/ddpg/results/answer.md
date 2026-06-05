# DDPG: Deep Deterministic Policy Gradient

## Problem it solves

DDPG is model-free reinforcement learning for continuous action spaces. It avoids the intractable `argmax_a Q(s,a)` required by discrete-action Q-learning by learning a deterministic actor `μ(s)` that outputs the action directly, while a critic estimates `Q(s,a)`.

## Key idea

For an unnormalized discounted state-visitation measure `ρ^μ`, the deterministic policy gradient is

`∇_θ J = ∫ ρ^μ(s) [∇_θ μ_θ(s)] [∇_a Q^μ(s,a)]_{a=μ_θ(s)} ds`.

With a normalized state distribution `d^μ=(1-γ)ρ^μ`, the same update has an extra constant `1/(1-γ)`, absorbed into the learning rate. Off-policy replay samples this chain-rule integrand under the replay/behavior state distribution; this is a behavior-weighted approximation, not an exact recovery of the original start-state gradient, but it needs no action-space importance ratio. The critic target is the deterministic Bellman backup with terminal masking:

`y = r + γ(1-d) Q_targ(s', μ_targ(s'))`.

The actor loss is `-Q(s, μ(s)).mean()`, so minimizing it ascends the critic. Target actor and critic networks are updated by Polyak averaging, `θ_targ ← ρ θ_targ + (1-ρ)θ`; with `ρ=0.995`, the live-network coefficient is `τ=0.005`. The code uses independent Gaussian action noise after an initial random-action warm-up; if an OU process is substituted, the zero-mean unit-step update is `x ← x + θ_ou(-x) + σξ`.

## Algorithm

Initialize actor `μ`, critic `Q`, target copies `μ_targ`, `Q_targ`, and replay buffer `R`.
For each environment step:
1. Use random actions while `t <= start_steps`; afterward act with `clip(μ(s) + σξ)`.
2. Store `(s,a,r,s',d)` in `R`, treating time-limit truncation as nonterminal for bootstrapping.
3. After enough data is collected, sample replay minibatches.
4. Update the critic by minimizing `mean((Q(s,a) - (r + γ(1-d)Q_targ(s', μ_targ(s'))))²)`.
5. Freeze critic parameters for efficiency and update the actor by minimizing `-mean(Q(s, μ(s)))`.
6. Polyak-update all target parameters with `ρ=0.995`, equivalently `τ=0.005` on the live parameters.

Typical code defaults: `γ=0.99`, `polyak=0.995`, actor and critic learning rates `1e-3`, replay size `10^6`, batch size `100`, `start_steps=10000`, `update_after=1000`, `update_every=50`, `act_noise=0.1`, and hidden layers `(256,256)`.

## Working code

```python
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)
        self.act_limit = act_limit
    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def train(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
          steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
          polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,
          start_steps=10000, update_after=1000, update_every=50,
          act_noise=0.1, max_ep_len=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = ac.q(o, a)
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ
        return ((q - backup) ** 2).mean()

    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    def update(data):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = False
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        for p in ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    total_steps = steps_per_epoch * epochs
    o, ep_len = env.reset(), 0
    for t in range(total_steps):
        a = get_action(o, act_noise) if t > start_steps else env.action_space.sample()
        o2, r, d, _ = env.step(a)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            o, ep_len = env.reset(), 0
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                update(replay_buffer.sample_batch(batch_size))
```
