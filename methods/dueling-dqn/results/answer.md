# Dueling Network Architecture

## Problem

A value-based deep RL agent learns Q(s,a;θ) with a generic single-stream network: a conv
encoder followed by one fully-connected stream emitting |A| action values. In many states
the action barely changes the outcome, so the action values are nearly equal and only their
common level — the state value — matters; yet the network re-estimates each action's value
independently, and the shared state value only receives a gradient through whichever single
action was sampled in a TD update. The state value, which a bootstrapping algorithm needs
accurate at every state, is the most diluted quantity in the network.

## Key idea

Reshape the network, not the algorithm. Split the head into two streams off a shared conv
trunk: a **value stream** producing a scalar V(s;θ,β) and an **advantage stream** producing
a vector A(s,a;θ,α), where A(s,a) = Q(s,a) − V(s) is the advantage. Recombine them into a
single Q output so the module keeps the exact input/output interface of an ordinary
Q-network and can be trained, unchanged, by any value-based algorithm (DQN, Double DQN,
SARSA, prioritized replay) via plain backpropagation, with no extra supervision.

The naive recombination Q = V + A is **unidentifiable**: adding a constant c to V and
subtracting it from A leaves Q unchanged, so gradient descent has no reason to make V the
true value. Stop that value/advantage offset trade by subtracting a per-state reference
from the advantage before forming Q:

- Max version (clean semantics): Q(s,a) = V(s) + ( A(s,a) − max_{a'} A(s,a') ).
  Then at the greedy action a* = argmax_a A, Q(s,a*) = V(s): V is exactly max_a Q, and
  the centered advantage is zero at the best action and non-positive elsewhere. But
  max_{a'} A jumps whenever the best action flips, destabilizing training.
- Mean version (used in practice): Q(s,a) = V(s) + ( A(s,a) − (1/|A|) Σ_{a'} A(s,a') ).
  Loses the exact max-value semantics: V becomes the mean of the Q-values over actions,
  and the centered advantage becomes Q(s,a) − mean_{a'} Q(s,a'). The mean is a smooth
  reference, so the advantages only have to track the mean rather than compensate every
  change to the optimal action's advantage — more stable.

Subtracting a per-state constant never changes the rank order of the actions, so the
greedy/ε-greedy policy is identical to the naive sum; the aggregation is purely a
training-time offset-control device. Because V now sits underneath all |A| outputs, it
receives a gradient on every transition regardless of which action was taken — the shared
state value is learned from all the data, and redundant actions can be flattened to ≈0
advantage rather than fit one-by-one.

## Architecture and training details

- Shared conv trunk (same as DQN): 32 filters 8×8 stride 4 → 64 filters 4×4 stride 2 →
  64 filters 3×3 stride 1, ReLU throughout, flatten.
- Two streams, each a 512-unit FC layer + ReLU; value stream → 1 output, advantage stream
  → |A| outputs. Combine via the mean-subtraction aggregator (inside the forward pass).
- Both streams backprop into the shared conv trunk, which can increase the incoming shared
  gradient; rescale the trunk-bound feature gradient by 1/√2 for stability.
- Clip the global gradient norm to ≤ 10 (RNN-style); slightly lower learning rate than
  plain Double DQN.
- Trained with the Double-DQN target; combines with prioritized replay unchanged.

## Code

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        n_actions = env.single_action_space.n
        self.trunk_grad_scale = 1.0 / math.sqrt(2.0)
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x / 255.0)
        if torch.is_grad_enabled() and feat.requires_grad:
            feat.register_hook(lambda grad: grad * self.trunk_grad_scale)
        value = self.value_stream(feat)               # (B, 1)
        advantage = self.advantage_stream(feat)       # (B, |A|)
        # Q = V + (A - mean_a' A): blocks the V+c / A-c trade and preserves action ranks.
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def train_step(q_network, target_network, optimizer, data, gamma, max_grad_norm=10.0):
    with torch.no_grad():
        # Double DQN: online net selects, target net evaluates.
        next_actions = torch.argmax(q_network(data.next_observations), dim=1, keepdim=True)
        next_q = target_network(data.next_observations).gather(1, next_actions).squeeze(1)
        y = data.rewards.flatten() + gamma * next_q * (1 - data.dones.flatten())
    q = q_network(data.observations).gather(1, data.actions).squeeze(1)
    loss = F.mse_loss(y, q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
    optimizer.step()
    return loss
```

The network keeps the same state-to-Q interface; replay, target-network sync, ε-greedy
behavior, and the Double-DQN target/loss stay the same, while the head, trunk-gradient
rescale, and norm clip are the added implementation pieces.
