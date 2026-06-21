# Context: off-policy actor-critic for continuous control (circa 2017-2018)

## Research question

We want a single model-free deep RL algorithm for continuous control. Two broad families are
in use. On-policy policy-gradient methods (TRPO, PPO, A3C) collect fresh trajectories for
essentially every gradient update and then discard them, consuming millions of environment
steps even for moderate tasks. Off-policy value-based methods reuse a replay buffer and so
draw many updates from each transition; extending them to continuous actions requires pairing
a continuous-action actor with a learned critic, where the actor and critic are optimized
against each other.

The setting, precisely: design an off-policy actor-critic for continuous control that learns
from a replay buffer of past transitions and scales to high-dimensional continuous action
spaces (up to the 21-dimensional Humanoid). The open slots are how to parameterize the policy,
what objective the actor optimizes, how many value estimates to keep and how to form their
targets, and how exploration is produced.

## Background

By 2017 the dominant recipe for continuous control splits into two camps.

**Actor-critic from policy iteration.** Classical policy iteration alternates *policy
evaluation* — compute the value function `Q^pi` of the current policy — and *policy
improvement* — make the policy greedy with respect to `Q^pi` (Barto et al. 1983; Sutton &
Barto 1998). In deep RL neither step is run to convergence; instead a critic network and an
actor network are optimized jointly by stochastic gradient descent, the actor playing the role
of an amortized "argmax over actions" that a continuous action space makes otherwise
intractable.

**The maximum-entropy framework.** A separate line of work augments the standard
reward-maximizing objective with the *entropy* of the policy, so the agent is asked to succeed
at the task while acting as randomly as possible (Ziebart et al. 2008; Ziebart 2010; Todorov
2008; Toussaint 2009; Rawlik et al. 2012; Fox et al. 2015; Haarnoja et al. 2017). The
augmented objective is

```
J(pi) = sum_t E_{(s_t,a_t) ~ rho_pi} [ r(s_t, a_t) + alpha * H(pi(.|s_t)) ],
```

where `H` is the policy's entropy and the temperature `alpha` trades off reward against
randomness; as `alpha -> 0` the standard objective is recovered. Ziebart (2010) argued such
policies are robust to model and estimation error (they hedge across actions of similar
value), and Haarnoja et al. (2017) observed they explore broadly and can represent multiple
modes of near-optimal behavior.

**Value overestimation in Q-learning.** A documented property of bootstrapped value learning:
the target maximizes over a noisy value estimate, so noise becomes a consistent upward bias
that the Bellman backup then propagates (Thrun & Schwartz 1993; van Hasselt 2010; Fujimoto et
al. 2018). DDPG, the standard off-policy continuous-control method, is reported to be sensitive
to hyperparameters and to make little progress on high-dimensional tasks such as Ant and
Humanoid (Duan et al. 2016; Henderson et al. 2017; Gu et al. 2016).

**The reparameterization trick** (Kingma & Welling 2013; Rezende et al. 2014). For a
stochastic node `a` whose distribution depends on parameters, drawing `a = f_phi(eps; s)` with
`eps` from a fixed noise distribution lets gradients flow *through* the sample as a
deterministic function of `phi`, giving a low-variance pathwise estimator in place of the
high-variance likelihood-ratio (score-function) estimator. This is the standard tool when the
quantity inside the expectation is itself differentiable in the action.

## Baselines

**DDPG — deep deterministic policy gradient (Lillicrap et al. 2015; Silver et al. 2014).** An
off-policy actor-critic with a *deterministic* actor `mu_phi(s)` and a critic `Q_theta(s,a)`.
The critic is trained on the standard Bellman residual using a target network; the actor is
trained by the deterministic policy gradient, pushing `mu_phi` uphill on the critic:
`grad_phi J = E[ grad_a Q_theta(s,a)|_{a=mu_phi(s)} * grad_phi mu_phi(s) ]`. Replay and target
networks supply the off-policy reuse. Exploration is supplied by an external noise process
added to the actor's output during data collection.

**TD3 — twin delayed DDPG (Fujimoto et al. 2018).** A set of refinements to DDPG addressing
overestimation, developed concurrently. It keeps *two* critics trained independently and forms
the bootstrap target from the *minimum* of the two — a value estimate suffering from
overestimation in one network is unlikely to be the smaller, so the min favors underestimation,
which does not get amplified through the Bellman backup. It also *delays* policy and target
updates relative to critic updates, and adds smoothing noise to the target action. The actor
remains deterministic with externally injected exploration noise.

**Soft Q-learning (Haarnoja et al. 2017).** Brings the maximum-entropy objective to
off-policy deep RL as a *Q-learning* method: the optimal max-entropy policy is the
energy-based distribution `pi(a|s) ∝ exp(Q(s,a)/alpha)`, and the critic is trained to estimate
the *optimal* soft Q-function directly. To act in continuous spaces it trains a separate
sampling network (via Stein variational gradient descent) to draw approximate samples from
that energy-based policy.

**On-policy policy gradients — TRPO / PPO / A3C (Schulman et al. 2015, 2017; Mnih et al.
2016).** Well-understood policy-gradient methods used on the hardest continuous-control tasks
of the day. Each batch of data is used for one (or a few) updates and then discarded; PPO in
particular uses large batches on high-dimensional tasks.

## Evaluation settings

The natural yardsticks already in use for continuous control:

- **MuJoCo / OpenAI Gym continuous-control suite** (Brockman et al. 2016): Hopper, Walker2d,
  HalfCheetah, Ant, and Humanoid, with state observations and continuous torque/position
  actions bounded to a box (commonly normalized to `[-1, 1]`). Action dimensions range from 3
  (Hopper) up to 17 (Humanoid-v1) and 21 (the rllab Humanoid), the high-dimensional cases
  being the discriminating ones.
- **Metric**: total average return of evaluation rollouts as a function of environment steps
  (the learning curve), capturing both sample efficiency (how fast) and asymptotic
  performance (how high). Stability is read off the spread across several random seeds.
- **Protocol**: multiple independent seeds per algorithm (e.g. five), one evaluation rollout
  at a fixed interval of environment steps, identical environment interfaces; for off-policy
  methods a replay buffer of transitions `(s, a, r, s')` and a discount `gamma` (e.g. 0.99).
  At evaluation, exploration is turned off (deterministic methods drop their injected noise).

## Code framework

A method in this space plugs into the standard off-policy actor-critic training harness that
already exists for DDPG-style methods: a replay buffer of transitions, a policy network that
maps states to bounded continuous actions, value networks that score state-action pairs,
target networks updated by Polyak averaging, separate optimizers, and an outer loop that
interleaves environment steps with gradient updates sampled from the buffer. The policy
parameterization, the number and form of value estimates, and the losses that drive the
updates are the open slots.

```python
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Maps observations to bounded continuous actions.
    Architecture, output parameterization, and sampling/evaluation behavior
    are to be designed."""

    def __init__(self, n_obs, n_act, hidden_dim=256):
        super().__init__()
        # TODO: define the policy network.
        pass

    def get_action(self, obs):
        # TODO: produce an action and any quantities needed by the updates.
        pass


class QNetwork(nn.Module):
    """Scores a state-action pair. Architecture and how many value estimators
    are needed are to be designed."""

    def __init__(self, n_obs, n_act, hidden_dim=256):
        super().__init__()
        # TODO: define the value estimator.
        pass

    def forward(self, obs, action):
        # TODO: return the estimated value of (obs, action).
        pass


def build_algorithm(n_obs, n_act, device):
    """Construct policy, value estimator(s), target network(s), optimizers,
    and any auxiliary learnable components the update rules require."""
    # TODO: assemble the components.
    pass


def update_critic(batch, components, gamma):
    """One critic gradient step from a replay minibatch.
    The bootstrap target and the loss are to be designed."""
    # TODO: form the target and minimize the critic loss.
    pass


def update_actor(batch, components):
    """One actor gradient step. The policy objective is to be designed."""
    # TODO: define and minimize the actor objective.
    pass


@torch.no_grad()
def soft_update(src, tgt, tau):
    """Polyak-average the target network toward the online network."""
    for p, p_t in zip(src.parameters(), tgt.parameters()):
        p_t.mul_(1.0 - tau).add_(p, alpha=tau)


# existing off-policy training loop the algorithm plugs into
def train(envs, replay_buffer, components, gamma, tau, total_steps):
    obs = envs.reset()
    for step in range(total_steps):
        action, *_ = components["actor"].get_action(obs)  # interact with the env
        next_obs, reward, done, _ = envs.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if step > learning_starts:
            batch = replay_buffer.sample(batch_size)   # reuse past experience
            update_critic(batch, components, gamma)
            update_actor(batch, components)
            # TODO: update any target networks required by the design.
```

The outer loop supplies replay minibatches; the actor/critic definitions and the two update
rules are where the algorithm will live.
