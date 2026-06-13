# Context

## Research question

The task is to learn good policies for a sequential decision problem from
interaction alone, by estimating, for every state $s$ and action $a$, the
optimal action value
$$
Q_*(s,a) = \max_\pi \mathbb{E}\!\left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s, A_t = a, \pi \right],
$$
the expected discounted return from taking $a$ in $s$ and acting optimally
thereafter, with $\gamma \in [0,1)$. Once $Q_*$ is known, the optimal policy is
trivial — act greedily, $\pi_*(s) = \arg\max_a Q_*(s,a)$ — so the whole problem
reduces to estimating one function. For any interesting domain the state space
is far too large to store a value per state, so $Q_*$ must be carried by a
parametric function approximator $Q(s,a;\theta)$, and learned online from a
single stream of experience.

The obstacle is that learning the value of acting optimally requires bootstrapping
through a $\max$ over *estimated* values. The estimates are wrong — they are
always wrong early in training, and with a function approximator they stay wrong
in a structured, state-dependent way even late in training. A usable method has
to produce value estimates that are accurate enough, and consistent enough across
states, that acting greedily on them yields a good policy; and it has to do this
while the very target it regresses toward is built out of its own imperfect
estimates. The danger is not merely that estimates are noisy but that the
learning rule might systematically distort them in one direction, and that the
distortion might be uneven across states so that it corrupts the *relative*
ordering of actions — which is all the greedy policy actually depends on.

A concrete, motivating observation sharpens this. When a value-based agent is
trained at scale and one tracks, during training, the value it *predicts* for its
own greedy policy against the *actual* discounted return that same policy obtains
when rolled out, the predicted value frequently runs far above the realized
return, and on some tasks it diverges upward while the agent's score
simultaneously falls. So the question is precise: where does this upward
discrepancy come from, is it actually harming the policy, and can it be removed
without disturbing the rest of a working system.

## Background

**Value functions and the optimal-value recursion.** Under a fixed policy $\pi$,
$Q_\pi(s,a) = \mathbb{E}[\sum_{k\ge 0}\gamma^k R_{t+k+1}\mid S_t=s,A_t=a,\pi]$.
The optimal values obey the Bellman optimality recursion
$Q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q_*(S_{t+1},a') \mid S_t=s,A_t=a]$.
The appearance of $\max_{a'}$ inside the expectation is what distinguishes
*control* (learning to act optimally) from *prediction* (evaluating a fixed
policy): the bootstrap target contains a maximization over the next state's
action values.

**Temporal-difference learning and Q-learning.** Temporal-difference learning
(Sutton 1988) updates a value estimate toward a bootstrapped target built from
the immediate reward plus the discounted estimate at the next state, rather than
waiting for a full Monte-Carlo return. Q-learning (Watkins 1989) is the control
version: it learns $Q_*$ off-policy by regressing $Q(S_t,A_t)$ toward
$$
Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a),
$$
the sampled one-step Bellman optimality target. With a parametric
$Q(s,a;\theta)$ the update is a semi-gradient step
$$
\theta_{t+1} = \theta_t + \alpha\,\big(Y_t - Q(S_t,A_t;\theta_t)\big)\,\nabla_{\theta_t} Q(S_t,A_t;\theta_t).
$$
The $\max_a$ is evaluated on the *current estimates*, and the same estimates both
choose which action attains the max and supply that action's value.

**Estimation error and the maximum.** It was understood that this maximization
is statistically delicate. Thrun and Schwartz (1993) studied Q-learning with
function approximation and showed that if the action-value estimates carry random
error uniformly distributed in $[-\epsilon, \epsilon]$, then the target is
overestimated by as much as $\gamma\,\epsilon\,\frac{m-1}{m+1}$, where $m$ is the
number of actions — and they gave a concrete toy problem in which this upward
bias drives the learned policy to be suboptimal even asymptotically. They
attributed the effect to the generalization error of an insufficiently flexible
approximator. Separately, van Hasselt (2010) argued that environmental noise
alone inflates Q-learning's estimates even with an exact tabular representation,
because the $\max$ over noisy unbiased estimates exceeds the $\max$ over their
true means. The common thread is Jensen's inequality: the maximum is a convex
function of the estimated values, so $\mathbb{E}[\max_a \hat Q(s,a)] \ge \max_a
\mathbb{E}[\hat Q(s,a)]$ — averaging *after* the max gives something at least as
large as the max of the averages. Whenever the estimates are noisy but unbiased,
the bootstrap target is biased upward.

**Why an upward bias might or might not matter.** A uniform upward shift of all
action values in a state leaves the greedy ordering intact and would not, by
itself, change the policy; moreover deliberate optimism about *uncertain*
state–actions is a standard exploration heuristic (Kaelbling et al. 1996). The
concern is different: the overestimation discussed here arises *after* an update,
on values the agent now treats as settled, and it is generally *non-uniform*
across states and actions. Non-uniform overestimation distorts the relative
value ordering, and because targets bootstrap off of other states' values, a
local overestimate propagates to the states that lead into it. A learning rule
that overestimates unevenly can therefore degrade the policy even when each
individual error is small.

## Baselines

**Tabular Q-learning (Watkins 1989).** The reference control algorithm. Core
idea and gap as above: a single set of estimates is used to both select and
evaluate the greedy next action inside the target $\max_a Q(S_{t+1},a)$. With any
estimation error this biases the target upward, and the bias can be uneven.

**Deep Q-Network (Mnih et al. 2015).** Q-learning with a deep convolutional
network $Q(s,a;\theta)$ mapping a stacked-frame state to a vector of action
values, made stable enough to train end-to-end from pixels. Two ingredients are
essential. (i) *Experience replay* (Lin 1992): observed transitions
$(S_t,A_t,R_{t+1},S_{t+1})$ are stored in a large buffer and minibatches are
sampled uniformly for the updates, decorrelating consecutive samples and reusing
data. (ii) *Target network*: a second copy of the parameters $\theta^-$ that is
held fixed and refreshed by copying $\theta$ every $\tau$ steps; the bootstrap
target is computed with it,
$$
Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-),
$$
so the regression target does not chase the parameters at every gradient step.
Both stabilizers attack optimization instability. Neither addresses the
statistics of the $\max$: the target still selects and evaluates the greedy
action with the *same* set of values $\theta^-$, so the upward bias of the
maximum is untouched. This is the open gap — a flexible, low-asymptotic-error
approximator on deterministic environments is close to a best case for
Q-learning, yet the maximization bias mechanism still applies.

**Two-estimator maximization (van Hasselt 2010), tabular.** A way to estimate
the value of the best action *without* the upward bias of the single-estimator
$\max$. Maintain two independent estimate sets $\theta$ and $\theta'$. To form a
target, use one set to *select* the greedy action and the *other* to *evaluate*
it:
$$
Y_t = R_{t+1} + \gamma\,Q\big(S_{t+1}, \arg\max_a Q(S_{t+1},a;\theta);\ \theta'\big).
$$
Because the selecting estimates and the evaluating estimates are independent, the
action that happens to look best under $\theta$ is not the one whose $\theta'$
value is inflated, so the evaluation is not systematically too high. The two sets
are updated symmetrically by randomly assigning each experience to one of them.
The gap: it was developed and analyzed only for small/tabular problems and
doubles the number of value functions; whether anything like this carries over to,
or helps under, large-scale function approximation was untested.

## Evaluation settings

The natural large-scale yardstick is the Arcade Learning Environment (Bellemare
et al. 2013): a suite of Atari 2600 games in which a single algorithm, with one
fixed set of hyperparameters, must learn each game separately from raw screen
pixels and the reward signal alone. Inputs are high-dimensional and the visuals
and mechanics differ sharply across games, so success must come from the learning
algorithm rather than per-game tuning. The standard protocol (Mnih et al. 2015):
gray-scale and rescale frames to $84\times84$, stack the last four as the state,
repeat each chosen action over four frames, and clip rewards to $[-1,1]$. The
network is a 3-layer convolutional net (32 filters $8\times8$ stride 4; 64
$4\times4$ stride 2; 64 $3\times3$ stride 1) into a 512-unit fully connected
layer and a linear output of one value per action (~1.5M parameters), trained
with RMSProp. Two diagnostics matter here beyond raw game score: (a) the value
the agent predicts for its greedy policy, averaged over visited states during
periodic evaluation phases, compared against (b) the *actual* discounted return
that policy obtains when run — if the estimates were unbiased these two would
coincide. Evaluation uses an $\epsilon$-greedy policy with small $\epsilon$ over
many episodes, optionally started from human-trajectory states to test
generalization rather than memorized action sequences. Normalized score per game
is $(\text{agent}-\text{random})/(\text{human}-\text{random})$.

## Code framework

The primitives that already exist: a deep-learning library with autodiff and an
optimizer, a CNN that maps a stacked-frame state to a vector of per-action
values, an environment wrapper that does the frame preprocessing/stacking and
reward clipping, and a uniformly-sampled replay buffer. The control loop —
$\epsilon$-greedy acting, storing transitions, periodically sampling a minibatch,
forming a bootstrap target, regressing onto it, and periodically refreshing a
held-fixed copy of the parameters — is also known. What is *not* fixed is exactly
how the bootstrap target's next-state value is computed; that single slot is left
open.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class QNetwork(nn.Module):
    """Stacked-frame state -> vector of action values. Known architecture."""
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
        return self.net(x / 255.0)


def compute_next_state_value(next_obs, reward, done, gamma,
                             online_net, target_net):
    """Form the bootstrap value of the next state for the regression target.
    This is the one open slot. The online net (theta) and the held-fixed
    target net (theta^-) are both available here.
    """
    pass  # TODO: bootstrap target's next-state value


def train_step(batch, gamma, online_net, target_net, optimizer):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        bootstrap = compute_next_state_value(
            next_obs, rewards, dones, gamma, online_net, target_net)
        td_target = rewards + gamma * bootstrap * (1.0 - dones)
    q_sa = online_net(obs).gather(1, actions).squeeze()
    loss = F.mse_loss(td_target, q_sa)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss


# Known control loop: epsilon-greedy act -> store -> sample minibatch ->
# train_step -> every tau steps copy online_net params into target_net.
```
