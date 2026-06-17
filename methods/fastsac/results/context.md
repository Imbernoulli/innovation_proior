# Context: off-policy actor-critic for continuous control (circa 2017-2018)

## Research question

We want a single model-free deep RL algorithm for continuous control that is at once
*sample-efficient* and *stable*. The two properties are in tension with the methods then
available. On-policy policy-gradient methods (TRPO, PPO, A3C) are stable and easy to get
working, but they must collect fresh trajectories for essentially every gradient update and
then discard them, so their sample complexity is enormous — millions of environment steps for
even moderate tasks, and worse as the action dimension and task difficulty grow. Off-policy
value-based methods reuse a replay buffer and so are far more sample-efficient in principle,
but bolting a continuous-action actor onto Q-learning has proven brittle: the learned actor
and the learned critic feed back into each other, and small errors in the critic get chased
and amplified by the actor. The concrete pain is that the off-policy continuous-control
methods of the day need careful, per-task tuning of learning rates, exploration noise, and
target-update schedules, and even then collapse or diverge on the hardest benchmarks (the
21-dimensional Humanoid is the canonical failure case for deterministic off-policy methods).

The goal, precisely, is an off-policy actor-critic that (1) reuses past data from a replay
buffer for efficiency; (2) is robust across tasks and random seeds without re-tuning per task;
(3) explores well enough to escape the poor local optima that plague deterministic actors in
high dimensions; and (4) scales to high-dimensional continuous action spaces. Each existing
method gets a subset; none gets all four at once.

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
value), and Haarnoja et al. (2017) observed they explore better and can represent multiple
modes of near-optimal behavior. This was largely background theory; turning it into a stable,
sample-efficient deep continuous-control algorithm was open.

**Empirical pain points of the prior art that motivate the work.** Several documented
observations about *existing* systems set up the problem. DDPG, the standard off-policy
continuous-control method, is widely reported to be extremely sensitive to hyperparameters and
to fail outright on high-dimensional tasks such as Ant and Humanoid (Duan et al. 2016;
Henderson et al. 2017; Gu et al. 2016). The diagnosed cause, established concurrently, is
*value overestimation*: in Q-learning the bootstrapped target maximizes over a noisy value
estimate, so noise turns into a consistent upward bias that the Bellman backup then propagates
(Thrun & Schwartz 1993; van Hasselt 2010; Fujimoto et al. 2018). A deterministic actor that
maximizes such an overestimated critic chases phantom value. Separately, deterministic actors
have no exploration of their own, so they require an externally injected, hand-scheduled noise
process, one more brittle knob.

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
`grad_phi J = E[ grad_a Q_theta(s,a)|_{a=mu_phi(s)} * grad_phi mu_phi(s) ]`. Replay + target
networks give sample efficiency. **Gap:** the deterministic actor maximizes the learned
critic, so any overestimation in `Q_theta` is directly exploited; training is unstable and
acutely hyperparameter-sensitive, and exploration must be supplied by an external noise
process that itself has to be scheduled. On high-dimensional tasks it frequently makes no
progress at all.

**TD3 — twin delayed DDPG (Fujimoto et al. 2018).** A set of fixes to DDPG's overestimation,
developed concurrently. It keeps *two* critics trained independently and forms the bootstrap
target from the *minimum* of the two — a value estimate suffering from overestimation in one
network is unlikely to be the smaller, so the min favors underestimation, which (unlike
overestimation) does not get amplified through the Bellman backup. It also *delays* policy and
target updates relative to critic updates, and adds smoothing noise to the target action.
**Gap:** still a deterministic actor with externally injected exploration noise; the
stochasticity used for target smoothing is a regularizer bolted onto the target, not a
property the policy itself optimizes, so exploration and the brittleness it causes remain
separate concerns from the objective.

**Soft Q-learning (Haarnoja et al. 2017).** Brings the maximum-entropy objective to
off-policy deep RL, but as a *Q-learning* method: the optimal max-entropy policy is the
energy-based distribution `pi(a|s) ∝ exp(Q(s,a)/alpha)`, and the critic is trained to estimate
the *optimal* soft Q-function directly. To act in continuous spaces it trains a separate
sampling network (via Stein variational gradient descent) to draw approximate samples from
that energy-based policy. **Gap:** the actor is motivated as an approximate sampler, not as
the actor of an actor-critic; the method's correctness hinges on how well the sampler matches
the true posterior, the approximate-inference machinery is complex and can be unstable, and in
practice these methods do not exceed strong off-policy baselines like DDPG/TD3 when learning
from scratch.

**On-policy policy gradients — TRPO / PPO / A3C (Schulman et al. 2015, 2017; Mnih et al.
2016).** Stable, well-understood, and the methods that actually work on the hardest tasks at
the time — but on-policy, so each batch of data is used for one (or a few) updates and then
thrown away. **Gap:** sample complexity is prohibitive; PPO in particular needs very large
batches to be stable on high-dimensional tasks, which is exactly the regime we care about.

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
