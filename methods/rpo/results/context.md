## Research question

On-policy actor-critic methods for continuous control use a parametric Gaussian policy
`pi_theta(a|s) = N(mu_theta(s), sigma_theta)`. As training proceeds, the surrogate objective
rewards committing to high-advantage actions, and the learned standard deviation `sigma` shrinks
while the mean `mu` sharpens. Entropy regularization (an explicit bonus on `H[pi]`) and alternative
action-distribution families are available tools for managing this. The question is how to shape
the policy-update rule for on-policy Gaussian actor-critics in continuous control so that
action-space exploration is maintained across a range of environments with varied dynamics.

## Background

The base learner this question sits on is PPO, itself the resolution of a line of policy-optimization
methods, and the question concerns how that line trains a *parametric* distribution.

**Vanilla policy gradients (Williams 1992; Sutton et al. 1999).** Parameterize `pi_theta(a|s)` and
ascend `J(pi) = E_{tau~p_pi}[ sum_t gamma^t r_t ]` with the score-function estimator
`g = E_t[ grad log pi_theta(a_t|s_t) A_t ]`. General and simple, but each batch feeds a single
gradient step (data-inefficient) and a large step can collapse the policy irrecoverably. For a Gaussian
policy the estimator pushes the mean toward high-advantage actions and, through the log-prob, also
shrinks the variance around them.

**Trust-region / proximal methods (TRPO, Schulman et al. 2015; PPO, Schulman et al. 2017).** Maximize
the advantage surrogate `E_t[ r_t(theta) A_t ]`, `r_t = pi_theta/pi_old`, inside a trust region —
TRPO with a hard KL constraint solved by natural gradient, PPO with the cheaper clipped surrogate
`L^CLIP = E_t[ min(r_t A_t, clip(r_t,1-eps,1+eps) A_t) ]`, `eps=0.2`, so one batch is safely reused
for several epochs. Reliable and first-order, and the standard base for continuous control with GAE
advantages (`lambda=0.95`) and a clipped value loss.

**Entropy regularization (A3C, Mnih et al. 2016).** Add a bonus `c_2 H[pi_theta(.|s)]` to the
objective to slow premature determinism. The coefficient `c_2` is a single global knob that interacts
with the advantage scale; tuning it is task-dependent.

**Alternative action distributions / data augmentation.** Replacing the diagonal Gaussian with a
Beta, a normalizing flow, or a mixture, or augmenting observations, can improve exploration or
robustness on some tasks. Each option introduces its own parameters and assumptions.

## Evaluation settings

The yardsticks already in use for on-policy continuous control:

- **Continuous-control benchmark suites** — OpenAI Gym / Gymnasium MuJoCo (HalfCheetah, Hopper,
  Walker2d, Swimmer, and the rest of the `-v2`/`-v4` suite), DeepMind Control, PyBullet, and IsaacGym
  — spanning a wide range of action dimensions and dynamics. Metric: mean episodic return over
  evaluation episodes within a fixed interaction budget, averaged across several random seeds; a
  learning curve of return vs. environment steps.
- **Protocol:** matched interaction budgets and matched network architecture across algorithms, so the
  comparison isolates the update rule. Comparisons read off final-policy returns and learning curves
  averaged over seeds. The relevant baseline is PPO under the identical harness; additional baselines
  include entropy regularization, alternative distributions, and data augmentation.

## Code framework

The editable substrate is a standard CleanRL-style on-policy actor-critic harness for continuous
control. The fixed pieces are generic: a Gaussian policy whose mean comes from a fixed two-hidden-layer
(2x64, tanh) MLP and whose log-standard-deviation is a learned state-independent vector; a separate
value MLP of the same shape; a rollout loop that collects a fixed-length batch with the current policy;
a GAE estimator that turns rewards and value estimates into per-step advantages and returns; and a
K-epoch minibatch optimization loop that normalizes advantages per minibatch before applying the loss.
The network architecture and parameter count are fixed — the contribution must be algorithmic. What is
open is the **action/value readout** `get_action_and_value` (how the action distribution is formed at
sample time and at update time) and the **per-minibatch loss** `compute_losses`.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """On-policy actor-critic with a fixed-capacity Gaussian policy and a separate
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
        # Gaussian policy: at rollout (action is None) sample from N(mu, sigma);
        # at update (action given) re-evaluate the log-prob of the stored action.
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Turn one minibatch into a scalar training loss. The action distribution
    used at update time and the policy/value objectives are the design surface."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)

    # TODO: the policy-update objective and the value-update objective.
    pass
```

The rollout loop, the GAE estimator, and the minibatch iteration are fixed; the action distribution
at update time and the policy/value objectives are the open slot.
