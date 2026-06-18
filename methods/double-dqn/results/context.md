# Context

## Research Question

The control problem is to learn a policy from interaction by estimating action
values. For a state $s$ and action $a$,
$$
Q_\pi(s,a)=\mathbb{E}\left[\sum_{k\ge 0}\gamma^k R_{t+k+1}\mid S_t=s,A_t=a,\pi\right],
$$
and $Q_*(s,a)=\max_\pi Q_\pi(s,a)$. If $Q_*$ were known, acting greedily would be
enough: choose an action in $\arg\max_a Q_*(s,a)$. In large visual domains the
values cannot be stored in a table, so a parameterized approximator
$Q(s,a;\theta)$ must carry them.

The hard part is that control bootstraps through a maximization over estimates.
The estimates are inaccurate during learning, and with function approximation
they can remain unevenly inaccurate across states and actions. A learning rule
therefore has to build targets from its own imperfect predictions without
turning ordinary approximation error into a systematic distortion of the greedy
ordering.

A useful diagnostic makes the issue concrete. During training one can run the
current greedy policy and compare two quantities: the value predicted by the
network for states it visits, and the discounted return actually obtained from
those states. Large gaps between these two quantities mean the value estimates
are not just noisy; they are systematically miscalibrated in the direction that
matters for bootstrapping.

## Background

Temporal-difference learning updates a current estimate toward a one-step target
instead of waiting for a full return. Q-learning is the off-policy control
version. With parameters $\theta_t$, after observing
$(S_t,A_t,R_{t+1},S_{t+1})$ it uses
$$
Y_t^{\mathrm{Q}}=R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta_t)
$$
and updates $Q(S_t,A_t;\theta_t)$ toward this target.

The maximization is statistically delicate. If action-value estimates can be
written as true values plus zero-mean errors, the maximum of the estimates is
not generally an unbiased estimate of the maximum true value. The pointwise max
is convex, so Jensen's inequality gives
$$
\mathbb{E}\left[\max_a \widehat Q(s,a)\right]\ge
\max_a \mathbb{E}\left[\widehat Q(s,a)\right].
$$
The max preferentially selects actions whose estimation error happened to be
positive.

An upward shift that were identical for every action in every state would not
change the greedy policy. The concern is non-uniform overestimation: the amount
of inflation depends on local approximation error, action count, data coverage,
and bootstrap history. Non-uniform errors change relative action and state
values, and those wrong relative values propagate backward through later
updates.

## Baselines

Tabular Q-learning uses one set of estimates both to choose the next greedy
action and to evaluate that action inside the target. Thrun and Schwartz showed
that, with uniformly distributed approximation errors in $[-\epsilon,\epsilon]$,
this maximization can overestimate targets by as much as
$\gamma\epsilon(m-1)/(m+1)$ for $m$ actions, and they gave examples where the
effect leads to suboptimal policies.

Deep Q-Networks combine Q-learning with a convolutional network, experience
replay, and a target network. Replay stores transitions and samples them
uniformly for minibatch updates. The target network has parameters $\theta^-$,
copied periodically from the online network and held fixed between copies. The
DQN target is
$$
Y_t^{\mathrm{DQN}}=R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta_t^-).
$$
Replay and the target network stabilize optimization, but the target-network
values still select and evaluate the greedy next action with the same estimates.

Tabular Double Q-learning provides a smaller-scale clue. It maintains two value
functions. When one is updated, it chooses the greedy action with one estimate
set and evaluates that chosen action with the other. This decoupling addresses
the maximization bias in the tabular setting, but it leaves open how to get the
same benefit inside a deep DQN system without adding another complete training
pipeline.

## Evaluation Setting

The large-scale testbed is the Arcade Learning Environment. A single algorithm,
with one fixed set of hyperparameters per condition, learns Atari 2600 games
from screen pixels and clipped rewards. The DQN-style setup preprocesses frames
to $84\times84$ grayscale, stacks the last four frames, repeats each chosen
action for four frames, clips rewards to $[-1,1]$, stores about $10^6$
transitions, samples minibatches of 32, and uses $\gamma=0.99$.

The Nature DQN architecture used by the target work has three convolutional
layers followed by a 512-unit fully connected layer and one linear output per
action: 32 filters of size $8$ stride $4$, 64 filters of size $4$ stride $2$, and
64 filters of size $3$ stride $1$. The optimizer is RMSProp with momentum
parameter $0.95$ and learning rate $0.00025$. The target network is copied every
$10{,}000$ agent steps in the untuned condition.

Two diagnostics matter. The first is value accuracy: predicted greedy-policy
values during evaluation phases versus actual discounted returns from running
the learned policy. The second is policy quality: normalized Atari score,
$(\mathrm{agent}-\mathrm{random})/(\mathrm{human}-\mathrm{random})$, under the
standard no-op start protocol and under harder starts sampled from human
trajectories.

## Code Framework

The available training scaffold is a standard DQN loop: choose actions
$\epsilon$-greedily with the online network, store transitions in replay, sample
minibatches, compute a one-step target with the target network machinery, update
the online network, and periodically copy online parameters into the target
network. The unresolved slot is the next-state bootstrap value.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


def compute_bootstrap_value(next_obs, online_net, target_net):
    """Return the estimated value of each sampled next state."""
    raise NotImplementedError


def train_step(batch, gamma, online_net, target_net, optimizer):
    obs, actions, rewards, next_obs, dones = batch
    discounts = gamma * (1.0 - dones.float())
    with torch.no_grad():
        bootstrap = compute_bootstrap_value(next_obs, online_net, target_net)
        td_target = rewards + discounts * bootstrap
    q_sa = online_net(obs).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```
