# Context

## Research question

In off-policy actor-critic RL for continuous control, how do we stabilize the bootstrapped
Q-target so that a single critic learns accurately from one update per environment step
(update-to-data, UTD = 1)? The standard mechanism for supplying a stationary regression target is a
*target network* — a slowly tracked copy of the critic. The high-UTD line of work (REDQ, DroQ)
raises sample efficiency by doing many critic updates per environment step against large or
regularized ensembles. The setting here is to ask what role the target network plays in stabilizing
value learning and what else could supply that stability.

## Background

**Deterministic and stochastic off-policy actor-critics.** DDPG (Lillicrap et al. 2016) learns a
deterministic actor that ascends a learned Q-critic via the deterministic policy gradient, with a
replay buffer and Polyak-averaged target networks. TD3 (Fujimoto et al. 2018) adds twin critics
with a clipped-double-Q (min) target, delayed actor updates, and target-policy smoothing to
counter the overestimation that DDPG's actor-driven max induces. SAC (Haarnoja et al. 2018)
replaces the deterministic actor with a stochastic tanh-Gaussian one under a maximum-entropy
objective, keeps twin critics with a min target, and auto-tunes the entropy temperature. All three
share the same bootstrap skeleton: target network(s) supply the next-state value used in the
Bellman target.

**The target network as a stationarity device.** Target networks come from DQN (Mnih et al. 2015):
the regression target `r + γ max_a Q(s',a)` is computed from a frozen copy `Q⁻` updated slowly, so
the critic fits a stationary objective over many gradient steps instead of one that moves every
step. The frozen copy is tracked toward the live critic by Polyak averaging with rate `τ`.

**Batch Normalization (Ioffe & Szegedy 2015) and Batch Renormalization (Ioffe 2017).** BatchNorm
normalizes a layer's pre-activations by the *batch's* mean/variance during training and by running
statistics at inference, which accelerates and stabilizes supervised training. Batch
Renormalization introduces clipped correction terms `r` and `d` that align the batch normalization
with the running statistics, making it more robust to small or non-i.i.d. batches.

**High-UTD efficiency methods.** REDQ (Chen et al. 2021) reaches high sample efficiency by raising
the update-to-data ratio (many critic updates per environment step) over a large randomized
ensemble. DroQ (Hiraoka et al. 2022) does the same with dropout + LayerNorm critics. Both keep
target networks.

## Code framework

```python
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
        )
        self.q1 = nn.Linear(hidden, 1)
        self.q2 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        h = self.encoder(x)
        return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)
```
