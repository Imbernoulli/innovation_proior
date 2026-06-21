## Research question

Value-based reinforcement learning on the full Atari suite from raw pixels: one agent, one set of
hyperparameters, evaluated on all 57 Atari 2600 games in the Arcade Learning Environment (ALE) under
the standard **no-ops** regime after $200$M training frames. The reported score is the **median
human-normalized score (HNS)** over the 57 games,
$\text{HNS}=100\times\frac{\text{agent}-\text{random}}{\text{human}-\text{random}}$; the median
captures the typical game rather than letting a few games dominate.

The designed object is the deep Q-learning agent itself — how the action-value (or the return) is
represented, how the bootstrap target is built, how transitions are sampled from replay, and how the
agent explores. Everything around that is held fixed: the ALE games, the $84\times84$ grayscale
4-frame-stack preprocessing, the $200$M-frame budget, the no-ops evaluation protocol, and the
single-agent/single-hyperparameter constraint.

## Prior art / Background / Baselines

The current baselines:

- **Neural fitted Q / online Q-learning from pixels.** A single network bootstraps from its own
  output on the correlated stream of live frames, with the target computed from the same weights
  being updated.

- **Tabular Q-learning and its overestimation.** The max over noisy action-value estimates
  systematically selects the higher values.

- **Exploration by $\epsilon$-greedy dithering.** Random actions are injected independently of state
  with a hand-set schedule.

- **The scalar value object.** The learned object is a single number per state-action, the mean of
  the return.

## Fixed substrate / Code framework

The shared substrate that every rung keeps unless it explicitly changes it: a convolutional encoder
over the $84\times84\times4$ input feeding a value head, a large experience-replay buffer, a
periodically synced target network, the Bellman/TD update, and discount $\gamma=0.99$. The starting
point is the bare deep Q-learning loop — replay + a frozen target copy + a scalar $Q$-head trained
by the clipped TD error.

```python
# Bare DQN scaffold. Code home: vwxyzjn/cleanrl (cleanrl/dqn_atari.py).
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def dqn_target(target_net, rewards, next_obs, dones, gamma):
    with torch.no_grad():
        max_next = target_net(next_obs).max(dim=1).values
        return rewards + gamma * max_next * (1.0 - dones)


def train_step(online_net, target_net, optimizer, batch, gamma):
    obs, actions, rewards, next_obs, dones = batch
    y = dqn_target(target_net, rewards, next_obs, dones, gamma)
    q_sa = online_net(obs).gather(1, actions.long().view(-1, 1)).squeeze(1)
    loss = F.huber_loss(y, q_sa)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

The training loop stores transitions in replay, samples mini-batches, calls `train_step`, and
periodically hard-copies `online_net` into `target_net`.

## Editable interface

Everything inside the deep Q-learning agent is on the table, one axis at a time:

- the **representation** of the value (what object the head predicts per action);
- the **bootstrap target** (how the next-state value is selected and evaluated, and over how many steps);
- the **head architecture** (how the encoder features are mapped to per-action outputs);
- the **replay sampling** distribution (how transitions are drawn from the buffer for each update);
- the **exploration** mechanism (how the agent deviates from the greedy action to gather data).

Each rung changes exactly one of these axes and re-measures the median HNS over all 57 games.

## Evaluation settings

- **Benchmark.** All 57 ALE games. One architecture and one set of hyperparameters for every game;
  no per-game tuning.
- **Preprocessing.** RGB to grayscale, downsample to $84\times84$, max-pool over the two most recent
  emulator frames, frame-skip $4$, stack the last $4$ processed frames as the state, and clip rewards
  during training. Terminal-on-life-loss may be used during training; evaluation uses true episode
  termination.
- **Budget and protocol.** $200$M training frames ($50$M agent steps). Evaluation in the no-ops
  regime: each episode begins with up to $30$ random no-op actions, then the greedy policy is rolled
  out to true termination.
- **Metric.** Median human-normalized score over the 57 games, higher is better. Each rung's feedback
  is this single percentage.
