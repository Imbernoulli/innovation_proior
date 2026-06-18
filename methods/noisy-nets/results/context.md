## Research Question

A reinforcement-learning agent in an MDP `M = (X, A, R, P, gamma)` has to solve two
problems at once. It must improve its value function or policy from the data it has already
collected, and it must choose actions that will generate useful future data. The usual deep-RL
training loop makes this second problem awkward: the function approximator is powerful, the
state space is large, and the agent usually sees only a thin stream of experience from its own
current behaviour.

The exploration mechanism therefore has to do more than add arbitrary randomness. It should
produce actions that are useful over multi-step trajectories, adapt as the agent learns, and fit
inside existing deep value and actor-critic agents without replacing their optimization machinery.
The open question is how to make exploratory variation structured and learnable while leaving the
rest of the agent close to the standard DQN, dueling-DQN, or A3C setup.

## Exploration Failure Mode

The dominant mechanisms are local dithering rules. In a value-based agent, `epsilon`-greedy
chooses the greedy action most of the time and a uniformly random action with probability
`epsilon`. In a policy-gradient agent, an entropy bonus discourages the softmax policy from
collapsing too quickly. Both mechanisms inject stochasticity at the action distribution.

This is simple, but it is not a good match for deep exploration. Random action perturbations are
drawn afresh at each step and do not remember why the previous exploratory action was taken.
For `epsilon`-greedy, the exploratory action is also state-independent. If the useful information is
only reached after a coordinated sequence of actions, independent per-step dithering spends most
of its effort on local jitter rather than a coherent exploratory trajectory. The scale is also external:
`epsilon` schedules and entropy weights are hand-set numbers, usually shared across tasks even
though uncertainty and useful exploration differ across games, states, and stages of training.

## Baseline Machinery

DQN learns an action-value network `Q(x, a; theta)` from replay-buffer transitions
`(x, a, r, y)` using a target network `theta^-`:

```text
L(theta) = E_D[(r + gamma max_b Q(y, b; theta^-) - Q(x, a; theta))^2].
```

The network acts through an `epsilon`-greedy policy. Apart from the exploration rule, the
architecture and training loop are fixed: a convolutional torso maps stacked Atari frames to a
feature vector, fully connected layers produce action values, and the target network is copied
periodically.

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

Optimism-based algorithms and confidence bounds can be statistically efficient in small or
linear settings, but they do not drop cleanly into large nonlinear deep networks. Intrinsic reward
methods add novelty, prediction-gain, or information-gain bonuses, but they introduce another
reward scale that must be balanced against the environment reward and can alter the objective
if misweighted.

Randomized value functions offer a more relevant pre-method route: sample a plausible value
function and act greedily with respect to it. This can produce deep exploration because the agent
commits to a sampled value estimate instead of dithering action by action. Bootstrapped DQN
extends this idea to neural networks by maintaining several value heads and choosing one head
for behaviour, but the duplicated heads and bootstrap masks are extra machinery.

Parameter-space perturbation is another nearby idea. Perturbing the parameters of a policy can
make behaviour state-dependent and temporally coherent, but if the perturbation scale is fixed or
externally adapted, the amount of exploration is still not learned directly by the RL loss.

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
