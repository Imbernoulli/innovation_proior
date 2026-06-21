## Research Question

A reinforcement-learning agent in an MDP `M = (X, A, R, P, gamma)` has to solve two
problems at once. It must improve its value function or policy from the data it has already
collected, and it must choose actions that will generate useful future data. The usual deep-RL
training loop makes this second problem awkward: the function approximator is powerful, the
state space is large, and the agent usually sees only a thin stream of experience from its own
current behaviour.

The open question is how to design an exploration mechanism that fits inside existing deep value
and actor-critic agents such as DQN, dueling-DQN, and A3C.

## Baseline Machinery

DQN learns an action-value network `Q(x, a; theta)` from replay-buffer transitions
`(x, a, r, y)` using a target network `theta^-`:

```text
L(theta) = E_D[(r + gamma max_b Q(y, b; theta^-) - Q(x, a; theta))^2].
```

The network acts through an `epsilon`-greedy policy, which chooses the greedy action most of
the time and a uniformly random action with probability `epsilon`. Apart from the exploration
rule, the architecture and training loop are fixed: a convolutional torso maps stacked Atari
frames to a feature vector, fully connected layers produce action values, and the target network
is copied periodically.

Dueling DQN keeps the same value-learning setting but decomposes the final head into a value
stream and an advantage stream:

```text
Q(x, a) = V(f(x)) + A(f(x), a) - (1 / |A|) sum_b A(f(x), b).
```

Its target is the double-DQN target: the online network selects the next action and the target
network evaluates it. The exploration mechanism remains `epsilon`-greedy.

A3C is on-policy actor-critic. Its network outputs both a policy `pi(. | x; theta_pi)` and a value
estimate `V(x; theta_V)`. It accumulates `k`-step returns

```text
Qhat_i = sum_{j=i}^{k-1} gamma^{j-i} r_{t+j}
         + gamma^{k-i} V(x_{t+k}; theta_V)
```

and uses `Qhat_i - V(x_{t+i}; theta_V)` as the advantage in the policy-gradient update. The
standard exploration aid is an entropy term weighted by a hand-tuned coefficient.

## Prior Routes

Optimism-based algorithms and confidence bounds are statistically grounded in small or
linear settings. Intrinsic reward methods add novelty, prediction-gain, or information-gain
bonuses to the environment reward.

Randomized value functions offer another route: sample a plausible value function and act
greedily with respect to it. Bootstrapped DQN extends this idea to neural networks by
maintaining several value heads and choosing one head for behaviour.

Parameter-space perturbation is a nearby idea. Perturbing the parameters of a policy can
make behaviour state-dependent and temporally coherent.

## Implementation Slot

The working codebase already has the Atari preprocessing, replay buffer, convolutional torso,
optimizers, target-network updates, and baseline losses. The narrow slot is the linear layer used
in value or policy heads and the action-selection rule that sits on top of the resulting network.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExploringLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self):
        raise NotImplementedError

    def resample(self):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = None

    def forward(self, x):
        raise NotImplementedError

def act(net, x):
    raise NotImplementedError

def loss_fn(online_net, target_net, batch, gamma):
    raise NotImplementedError
```
