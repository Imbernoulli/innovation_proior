# Dueling DDQN — value/advantage streams

**Problem.** A single-stream head estimates each action value independently, and the state value $V(s)$ is
never represented as its own quantity — it is smeared across all $|\mathcal A|$ outputs and corrected only
through whichever action a TD update sampled, i.e. $1/|\mathcal A|$ as often as it should be. Yet a
bootstrapping algorithm needs $V$ accurate at *every* state, and in most Atari states the action barely
matters, so $V$ is the most diluted quantity in the network.

**Key idea.** Reshape the head, not the algorithm. Split the shared features into a **value stream**
$V(s)$ and an **advantage stream** $A(s,a)=Q(s,a)-V(s)$, recombined into a single $Q$ output keeping the
exact state$\to Q$ interface. Now $V$ sits underneath all outputs and gets a gradient on *every* transition.

**Identifiability fix.** Naive $Q=V+A$ is unidentifiable ($V\!+\!c$, $A\!-\!c$ leaves $Q$ fixed). Subtract a
per-state reference to pin the offset. Max version $Q=V+(A-\max_{a'}A)$ has clean semantics ($V=\max_a Q$)
but jumps when the argmax flips; **mean version** $Q=V+(A-\frac1{|\mathcal A|}\sum_{a'}A)$ is the practical
choice — a smooth reference, more stable. Subtracting a per-state constant never changes the action ranks,
so the greedy/$\epsilon$-greedy policy is identical to the naive sum: a pure training-time offset-control
device.

**Details.** Two 512-unit FC streams off the shared conv trunk; rescale the trunk-bound gradient by
$1/\sqrt2$ (both streams now feed it), clip global grad-norm $\le10$, slightly lower LR. Kept on the
Double-DQN target (composes unchanged).

**Bar.** Every game has many action-irrelevant states, so learning $V$ from all transitions should give a
broad lift (clearing 140%), larger where the action set is larger. What stays open: $Q$, $V$, $A$ are still
*scalars* — point estimates of a return whose distribution the agent never sees. That is the next rung.

```python
# Dueling head: value + advantage streams, mean-subtraction aggregator, 1/sqrt(2) trunk-grad rescale.
# Code home: vwxyzjn/cleanrl; excerpted from methods/dueling-dqn/results/answer.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.trunk_grad_scale = 1.0 / math.sqrt(2.0)
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                              nn.Linear(512, n_actions))

    def forward(self, x):
        feat = self.feature(x / 255.0)
        if torch.is_grad_enabled() and feat.requires_grad:
            feat.register_hook(lambda grad: grad * self.trunk_grad_scale)   # both streams feed the trunk
        value = self.value_stream(feat)                                     # (B, 1)
        advantage = self.advantage_stream(feat)                            # (B, |A|)
        # Q = V + (A - mean_a' A): blocks the V+c / A-c trade, preserves action ranks.
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def train_step(q_network, target_network, optimizer, data, gamma, max_grad_norm=10.0):
    with torch.no_grad():
        next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)   # Double-DQN
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
