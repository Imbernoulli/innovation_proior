## Research question

Model-free reinforcement learning can train agents for sophisticated behaviors with very few
assumptions on the task. A supervised-learning practitioner writes a loss, calls a regression
routine, and gets a fit; an RL practitioner instead works with target networks, double critics,
clipped surrogates, importance ratios, trust-region solvers, and entropy schedules. The question
here is whether a reinforcement-learning algorithm for continuous (and discrete) control can be
built so that its training steps look as much as possible like ordinary supervised fitting —
likelihood-style and least-squares losses — while reusing off-policy data: data collected by
earlier versions of the policy held in a replay buffer, or a static dataset gathered by some
other policy.

## Background

By this time, deep RL for continuous control has two dominant camps.

**On-policy policy search.** Policy-gradient methods (Williams 1992; Sutton et al. 1999)
parameterize a stochastic policy `pi_theta(a|s)` and ascend the expected discounted return
`J(pi) = E_{tau~p_pi}[ sum_t gamma^t r_t ]` by differentiating it. They are conceptually
direct and work on a wide range of tasks. They are *on-policy* (or nearly so): the gradient is
an expectation under the current policy's own trajectory distribution, so data collected by an
older policy is formally invalid, and each batch is used for a single update.

**Off-policy value-based methods.** Q-learning and its actor-critic descendants reuse a replay
buffer by fitting an action-value function with Bellman backups, using stabilizers such as
target networks, double-Q estimates, clipped double-Q, and exploration noise. The Bellman backup
evaluates the critic at the *next* action chosen by the current policy, `Q(s', a')` with
`a' ~ pi`, so when `a'` falls outside the action distribution present in the data, the critic's
value there is an extrapolation. On static datasets, a common comparison point is a plain
behavioral-cloning policy, which is fit only to actions present in the data.

**Importance sampling** makes a return estimate from one policy valid under another by
reweighting with the policy ratio. It is unbiased, and the ratio's variance grows as the two
policies separate.

Two background frames are relevant here.

The first is the **performance-difference / expected-improvement identity** (Kakade & Langford
2002). For any two policies, the difference in their returns is the advantage of one
accumulated under the other's state visitation:

```
J(pi) - J(mu) = E_{tau ~ p_pi}[ sum_t gamma^t A^mu(s_t, a_t) ]
             = E_{s ~ d_pi(s)} E_{a ~ pi(a|s)} [ A^mu(s, a) ],
```

where `A^mu(s, a) = R^mu_{s,a} - V^mu(s)` is the advantage of taking action `a` and thereafter
following the *sampling* policy `mu`, `R^mu_{s,a}` is the return so obtained, `V^mu` is `mu`'s
value function, and `d_pi(s) = sum_t gamma^t p(s_t = s | pi)` is the unnormalized discounted
state distribution. The right-hand side averages over `d_pi`, the states of the *new* policy,
which depends on the `pi` being optimized. A standard construction (used in the trust-region
line below) forms a surrogate that replaces `d_pi` by `d_mu`, the sampling policy's state
distribution:

```
hat_eta(pi) = E_{s ~ d_mu(s)} E_{a ~ pi(a|s)} [ A^mu(s, a) ].
```

This surrogate agrees with the true improvement to first order at `pi = mu` (Kakade & Langford
2002) and is a good approximation as long as `pi` stays close to `mu` in KL divergence
(Schulman et al. 2015) — which is why methods in this frame pair the surrogate with a
KL trust-region constraint between the new and sampling policies. The exact KL direction varies
across formulations; the shared requirement is that the update remain local enough for the
state-distribution swap to be credible.

The second frame is the **maximum-entropy / KL-regularized view of policy updates**, in which
a policy improvement step is posed as constrained optimization: improve the objective while
staying within a bounded relative entropy of a reference distribution. A recurring consequence
of this frame, seen across several methods below, is that the *constrained* optimum has a
closed Gibbs/Boltzmann form — the reference distribution multiplied by an exponential of the
objective term, divided by a normalizer.

## Baselines

These are the prior methods a new algorithm in this space would be measured against.

**Reward-weighted regression (RWR), Peters & Schaal (2007).** Casts policy search as
expectation-maximization, so that the policy improvement step becomes *supervised regression*.
Treating reward as an (improper) likelihood, the E-step weights each observed action by an
exponential of its reward and the M-step does a weighted maximum-likelihood fit of the policy
to those actions. Per iteration it solves

```
pi_{k+1} = argmax_pi  E_{s ~ d_{pi_k}(s)} E_{a ~ pi_k(a|s)} [ log pi(a|s) * exp( (1/beta) * R_{s,a} ) ],
```

where `R_{s,a}` is the return and `beta > 0` a temperature, typically adapted across
iterations. Each update is a weighted regression. The weight is an exponential of the *raw
return*, and the sampling policy is the current policy `pi_k`.

**Relative entropy policy search (REPS), Peters, Mülling & Altün (2010).** Maximizes expected
reward subject to a bound on the relative entropy ("information loss") between the new
state-action distribution and the observed data distribution `q(s,a)` — the first KL
trust-region in RL:

```
max_pi  E_{s,a}[ R^a_s ]   s.t.   sum_{s,a} mu^pi(s) pi(a|s) log( mu^pi(s)pi(a|s) / q(s,a) ) <= epsilon.
```

Solving the constrained problem through its dual yields a closed Gibbs-form policy,

```
pi(a|s) = q(s,a) * exp( (1/eta) * delta(s,a) ) / sum_b q(s,b) * exp( (1/eta) * delta(s,b) ),
```

where `delta(s,a) = R^a_s + sum_{s'} P^a_{ss'} V(s') - V(s)` is the Bellman error and `eta` is
the Lagrange multiplier from the relative-entropy constraint. As an algorithm it minimizes a
convex *dual function* `g(theta, eta)` over both the multiplier `eta` and the value-function
parameters `theta` (e.g. with BFGS); the value function is tied to a feature-matching constraint
and a linear form `V = phi^T theta`; the weight uses a one-step Bellman error; and the
policy-iteration procedure models the sampling distribution as the *latest* policy.

**Maximum a posteriori policy optimization (MPO), Abdolmaleki et al. (2018).** A deep variant
of the REPS/EM line. It first fits an action-value `Q` of the current policy by bootstrapping,
using Retrace(λ) for off-policy correction, then performs a KL-constrained policy improvement
with respect to `Q`; the non-parametric E-step again produces a Boltzmann form `q(a|s)
proportional to pi_old(a|s) exp(Q/temperature)`. It combines a bootstrapped Q-critic with an
off-policy return correction, a dual optimization for the temperature, and the policy
projection.

**Trust-region / proximal on-policy methods (Schulman et al. 2015; Schulman et al. 2017).**
Maximize the advantage surrogate `hat_eta(pi)` above under a KL trust region between the new
and old policy, either as a constrained natural-gradient step or, more simply, by clipping the
importance ratio so the surrogate is not improved beyond a fixed neighborhood. They are stable
and widely used; the surrogate and the ratio are defined against the policy that *collected the
current batch*.

**Off-policy actor-critics (DDPG, TD3, SAC).** Reuse a replay buffer through Bellman backups
and a critic-to-actor gradient; sample-efficient and strong on standard benchmarks. They use
target networks and double critics, and the action is produced through the critic-to-actor
gradient path.

## Evaluation settings

The natural yardsticks already in use for continuous control:

- **OpenAI Gym MuJoCo continuous-control tasks** — HalfCheetah, Hopper, Walker2d, Ant,
  Humanoid (the `-v2` suite), with continuous action spaces of varying dimension and dynamics.
  Metric: mean episodic return over evaluation episodes within a fixed interaction budget,
  averaged across several random seeds; a learning curve of return vs. environment steps.
- **A discrete-action control task** — LunarLander — included to check that the same algorithm
  works unchanged on discrete actions.
- **High-dimensional motion-imitation tasks** with complex simulated characters (a 34-DoF
  humanoid, an 82-DoF dog) imitating motion-capture clips, following an established imitation
  framework; a stress test for scaling to many degrees of freedom. Metric: normalized
  return per episode.
- **Fully off-policy / static-dataset tasks** — learning the best policy from a fixed dataset
  of transitions (states, actions, rewards) collected by a separate demonstration policy, with
  no further environment interaction. Standard comparison points are the demonstration policy's
  own return and a behavioral-cloning policy.
- Protocol: matched interaction budgets and matched network architectures across algorithms;
  comparisons read off learning curves and final-policy returns averaged over seeds.

## Code framework

The editable substrate is a standard CleanRL-style actor-critic harness for continuous control.
The pieces that already exist are generic: a Gaussian policy whose mean comes from a fixed
two-hidden-layer MLP and whose log-standard-deviation is a learned vector; a separate
value/critic MLP of the same shape; a rollout loop that collects a batch of transitions; an
estimator that turns rewards and value estimates into per-step advantages and returns; and a
minibatch optimization loop that normalizes the advantages per minibatch before applying a
loss. The open slot is the loss itself: given observed actions, old log-probabilities,
advantages, returns, and old values, how this one slot updates the policy and the value
network.

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """Actor-critic scaffold with a fixed-capacity Gaussian policy and a separate
    value network. Architecture is fixed; only the action/value readout and the
    per-minibatch loss are open to design."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        h = 64
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        # Gaussian policy: sample an action (if none given) and report its
        # log-probability, entropy, and the critic's value of the state.
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Turn one minibatch of (observations, actions, old log-probs, advantages,
    returns, old values) into a scalar training loss.

    The value-update piece and the policy-update piece are exactly what is to be
    designed: given per-step advantages and returns, what objective should the
    policy be trained against?
    """
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)

    # TODO: the policy-update objective and the value-update objective.
    pass
```

The rollout loop, the advantage estimator, and the minibatch iteration are fixed; the policy
and value objectives inside `compute_losses` are the open slot.
