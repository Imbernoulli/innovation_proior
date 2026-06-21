The problem is learning a controller for a continuous, high-dimensional action space from a model-free reinforcement signal. In discrete-action domains, deep Q-learning gives a practical recipe: estimate Q(s,a) with a neural network, store transitions in a replay buffer, and use a target network so the bootstrap target does not move on the same step as the regressor. The control rule is greedy, a = argmax_a Q(s,a), which is cheap when the action set is small. With real-valued torques, steering angles, or joint velocities, that greedy maximization becomes a non-convex optimization over a real vector. It would have to be solved at action-selection time and inside every bootstrap target, which is incompatible with a real-time controller. Discretizing the action space is no escape: a modest seven-degree-of-freedom arm with only three choices per joint already yields three to the seventh actions, and finer resolution explodes exponentially while discarding the metric structure between nearby torque values.

Existing ideas each leave a gap. The deterministic policy gradient theorem shows how to improve a deterministic actor mu(s) using the chain rule through a critic, but the original work did not combine that insight with the stabilizers that make large neural Q-learning practical. Neural fitted Q-iteration with continuous actions relies on full-batch retraining, which is expensive at scale. Stochastic policy-gradient actor-critic is general, but in high-dimensional continuous action spaces the score term adds variance, and off-policy variants bring in action likelihood ratios. What is missing is a method that keeps the low-variance deterministic update, reuses off-policy data without action-space importance weights, and tames the bootstrap instability that afflicts deep actor-critic methods.

The clean fix is to stop asking the value function to do two jobs at once. Keep a critic Q(s,a) for evaluation, and introduce a separate deterministic actor mu(s) whose only job is to emit the action. Because the actor is differentiable in its parameters and the critic is differentiable in the action, the chain rule moves the actor in the direction that increases value: nabla_theta Q(s, mu_theta(s)) equals nabla_theta mu_theta(s) times nabla_a Q(s,a) evaluated at the actor's output. This amortizes the per-step argmax. Instead of solving an inner optimization problem at every timestep, we train a network whose outputs gradually converge to actions the critic rates highly. The deterministic policy gradient is an expectation over states only, with no action integral, so replay states can be sampled under a noisy behavior policy without action importance ratios. Bootstrap stability comes from maintaining target copies of both actor and critic, updated slowly via Polyak averaging, and from masking the Bellman backup at terminal transitions.

The method is Deep Deterministic Policy Gradient, or DDPG. It maintains a live actor mu(s), a live critic Q(s,a), and target copies of both. The critic regresses to the deterministic Bellman target y = r + gamma (1 - d) Q_targ(s', mu_targ(s')), computed without gradients so the regression target does not chase itself. The actor is trained to maximize the critic's estimate by minimizing the negative value, loss_pi = -mean(Q(s, mu(s))). Exploration is handled off-policy by adding independent Gaussian noise to the deterministic action during data collection, after an initial random-action warmup fills the replay buffer with diverse states. The behavior policy only affects which transitions enter replay; the learned target policy remains deterministic. This gives a single, simple loop: act, store, sample a minibatch, update the critic, freeze the critic parameters for efficiency and update the actor, then Polyak-average all target parameters toward their live counterparts.

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
