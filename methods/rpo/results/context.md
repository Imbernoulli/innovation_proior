## Research question

On-policy actor-critic methods for continuous control have a recurring, quiet failure: the
stochasticity of the policy collapses too fast. A Gaussian policy `pi_theta(a|s) = N(mu_theta(s),
sigma_theta)` starts diffuse and, as the surrogate objective rewards committing to whichever action
currently looks best, the learned standard deviation `sigma` shrinks and the mean `mu` sharpens. Once
that happens early — before the agent has actually seen enough of the action space to know the optimum
— exploration dies and the policy locks into a sub-optimal basin it can no longer escape, because the
gradient it now produces only reinforces what it already does. Entropy regularization (an explicit
bonus on `H[pi]`) is the usual patch, but the right coefficient is task-dependent and a fixed one
either barely moves the policy or floods it with noise; swapping the Gaussian for a heavier-tailed or
otherwise reshaped distribution helps on some environments and hurts on others and needs its own
tuning. The precise goal is an on-policy update that *keeps the policy exploratory* — maintains a
useful level of action-space entropy across training — without a hand-tuned entropy schedule, without
changing the action distribution family, and without changing the network or the surrogate objective
that already make the base method stable. It must hold across continuous-control environments with
very different dynamics rather than being tuned to one.

## Background

The base learner this question sits on is PPO, itself the resolution of a line of policy-optimization
methods, and the failure being attacked is a property of how that line trains a *parametric*
distribution.

**Vanilla policy gradients (Williams 1992; Sutton et al. 1999).** Parameterize `pi_theta(a|s)` and
ascend `J(pi) = E_{tau~p_pi}[ sum_t gamma^t r_t ]` with the score-function estimator
`g = E_t[ grad log pi_theta(a_t|s_t) A_t ]`. General and simple, but each batch feeds a single
gradient step (data-inefficient) and a large step can collapse the policy irrecoverably. For a Gaussian
policy the estimator pushes the mean toward high-advantage actions and, through the log-prob, also
shrinks the variance around them — so the gradient itself drives the policy deterministic over time.
**Gap:** no mechanism resists the entropy collapse that the estimator induces.

**Trust-region / proximal methods (TRPO, Schulman et al. 2015; PPO, Schulman et al. 2017).** Maximize
the advantage surrogate `E_t[ r_t(theta) A_t ]`, `r_t = pi_theta/pi_old`, inside a trust region —
TRPO with a hard KL constraint solved by natural gradient, PPO with the cheaper clipped surrogate
`L^CLIP = E_t[ min(r_t A_t, clip(r_t,1-eps,1+eps) A_t) ]`, `eps=0.2`, so one batch is safely reused
for several epochs. Reliable, first-order, the standard base for continuous control with GAE
advantages (`lambda=0.95`) and a clipped value loss. **Gap:** the clipped surrogate controls only *how
far* the policy moves per update, not *in which way the distribution sharpens*. Across the multi-epoch
updates the optimizer is free to drive `sigma` down and `mu` to a sharp peak; PPO's own entropy decays,
often before the agent has explored enough, and the trust region does nothing to stop it.

**Entropy regularization (A3C, Mnih et al. 2016).** Add a bonus `c_2 H[pi_theta(.|s)]` to the
objective to slow premature determinism. **Gap:** `c_2` is a single global knob fighting the
advantage scale; too small and the policy still collapses, too large and the policy is permanently
noisy and never commits. The coefficient that works on one environment is wrong on the next, so it
needs per-task tuning — exactly the fragility the research question wants to remove.

**Alternative action distributions / data augmentation.** Replacing the diagonal Gaussian with a
Beta, a normalizing flow, or a mixture, or augmenting observations, can improve exploration or
robustness on some tasks. **Gap:** each adds its own parameters and assumptions, changes the network,
and helps unevenly across environments — again task-specific.

The load-bearing observation underneath all of these: a Gaussian policy's exploration is entirely
controlled by `(mu, sigma)`, and the on-policy update sharpens both. If exploration could be kept alive
by *perturbing the distribution that the policy-likelihood is evaluated under during the update* —
rather than by an extra loss term or a different distribution family — it would cost no new
parameters, leave the surrogate and the network untouched, and be the same one knob on every task.

## Evaluation settings

The yardsticks already in use for on-policy continuous control:

- **Continuous-control benchmark suites** — OpenAI Gym / Gymnasium MuJoCo (HalfCheetah, Hopper,
  Walker2d, Swimmer, and the rest of the `-v2`/`-v4` suite), DeepMind Control, PyBullet, and IsaacGym
  — spanning a wide range of action dimensions and dynamics. Metric: mean episodic return over
  evaluation episodes within a fixed interaction budget, averaged across several random seeds; a
  learning curve of return vs. environment steps.
- **Protocol:** matched interaction budgets and matched network architecture across algorithms, so the
  comparison isolates the update rule (and, for this method, a single perturbation coefficient).
  Comparisons read off final-policy returns and learning curves averaged over seeds. The relevant
  baseline to beat is PPO under the identical harness; the additional baselines are entropy
  regularization, alternative distributions, and data augmentation, each with its own tuning.

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
