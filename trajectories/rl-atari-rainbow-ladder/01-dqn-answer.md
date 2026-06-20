# DQN — the floor: replay + target network

**Problem.** Neural Q-learning on the live Atari stream is unstable: consecutive frames are highly
correlated, the bootstrap target $r+\gamma\max_{a'}Q(s',a';\theta)$ is computed from the same $\theta$
being updated (so it runs away every step), and the data distribution shifts with the policy. All three
couplings sit on one set of weights.

**Key idea.** Decouple the three moving parts.
- **Experience replay**: store transitions in a large ($\sim10^6$) buffer and train on *uniform random
  minibatches*, which decorrelates the samples, averages the training distribution over many past
  policies, and reuses each transition many times. Replayed data is off-policy, which is exactly what
  Q-learning's $\max$-target is built to handle.
- **Target network**: compute the bootstrap from a frozen copy $\theta^-$, hard-synced
  $\theta^-\!\leftarrow\!\theta$ every $C$ steps, so between syncs the regression target is stationary —
  ordinary supervised regression rather than a chase.

**Cross-suite hyperparameters.** Reward clipped to $\{-1,0,+1\}$ and a clipped/Huber TD error so one
$\gamma=0.99$, one learning rate, one loss serve all 57 games. $\epsilon$-greedy annealed $1\!\to\!0.1$.
Nature-DQN encoder (three conv layers $\to$ 512-unit FC $\to$ linear $|\mathcal A|$-head, one forward pass
scores every action).

**What this is and isn't.** The smallest recipe that is *stable* across the whole suite. It leaves every
later axis untouched: the single $\max$ in the target (biased upward), unstructured $\epsilon$-greedy
exploration, uniform replay, a single-stream head, and a scalar value object. Those are the rungs to come.

```python
# DQN agent: Nature-DQN encoder + experience replay + periodically-synced target network.
# Code home: vwxyzjn/cleanrl (cleanrl/dqn_atari.py); excerpted from methods/dqn/results/answer.md.
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """State -> one Q-value per action (all actions scored in a single forward pass)."""
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),    # 84x84x4 -> 20x20x32
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # -> 9x9x64
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),   # -> 7x7x64
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e, end_e, duration, t):
    return max((end_e - start_e) / duration * t + start_e, end_e)


def dqn_target(target_net, rewards, next_obs, dones, gamma):
    """y = r + gamma * max_a' Q(s', a'; theta-) * (1 - done).  Bootstrap from the frozen copy."""
    with torch.no_grad():
        max_next = target_net(next_obs).max(dim=1).values
        return rewards.flatten() + gamma * max_next * (1.0 - dones.flatten())


def clipped_td_loss(td_error, kappa=1.0):
    """Huber / clipped-error loss: quadratic for |delta|<=kappa, linear beyond (cross-suite stability)."""
    abs_e = td_error.abs()
    quad = torch.minimum(abs_e, torch.full_like(abs_e, kappa))
    lin = abs_e - quad
    return (0.5 * quad.pow(2) + kappa * lin).mean()


def train_step(online_net, target_net, optimizer, batch, gamma):
    obs, actions, rewards, next_obs, dones = batch
    y = dqn_target(target_net, rewards, next_obs, dones, gamma)
    q_sa = online_net(obs).gather(1, actions.long().view(-1, 1)).squeeze(1)
    loss = clipped_td_loss(y - q_sa)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# ---- loop sketch (off-policy, replay buffer, periodic hard target sync) ----
# online = QNetwork(n_actions); target = QNetwork(n_actions); target.load_state_dict(online.state_dict())
# optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)         # reward clipped to {-1,0,+1} in env
# for step in range(total_frames // 4):
#     eps = linear_schedule(1.0, 0.1, 0.1 * (total_frames // 4), step)
#     a = env.action_space.sample() if random.random() < eps else online(obs).argmax(1)
#     next_obs, r, done, _ = env.step(a); replay.add(obs, a, r, next_obs, done); obs = next_obs
#     if step > learning_starts and step % train_frequency == 0:
#         train_step(online, target, optimizer, replay.sample(batch_size), gamma=0.99)
#     if step % target_network_frequency == 0:
#         target.load_state_dict(online.state_dict())          # hard sync theta- <- theta
```
