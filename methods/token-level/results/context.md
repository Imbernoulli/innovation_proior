# Context: reusable policy-gradient updates from stale rollout data

## Research question

We have a stochastic policy `pi_theta(a | s)` represented by a neural network, and we want to
improve its expected return from sampled interaction data. Each rollout is expensive: in a
control problem it costs environment steps, and in a language-model setting it costs full
generations plus scoring. Once a batch has been collected, a practical optimizer should reuse it
for several minibatch updates instead of throwing it away after one gradient step.

That reuse creates a central tension. The batch was sampled from the data-collecting policy
`pi_old`, but the parameters being updated define a different policy `pi_theta`. A usable
policy-loss rule has to extract more work from the same batch while keeping the update consistent
with the data-collecting policy so the batch remains a trustworthy local guide. The question is:
what per-step policy-loss rule, operating on stored old log-probabilities, recomputed current
log-probabilities, and supplied advantage estimates, enables multiple first-order minibatch
updates on a single rollout batch, in settings ranging from continuous control to language-model
reinforcement learning where the policy acts at token granularity?

## Background

The vanilla policy-gradient estimator is

```text
E_t[ grad_theta log pi_theta(a_t | s_t) * A_hat_t ],
```

where `A_hat_t` is an advantage estimate. The advantage estimator is not the design target here;
it is an input supplied by the rollout pipeline. The design target is the per-step policy loss
that turns stored old log-probabilities, recomputed current log-probabilities, and advantages into
a scalar update.

If the batch is treated as current-policy data after the first update, the estimator is no longer
honest. Importance sampling gives the local correction: with
`r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)`, the old-policy batch estimates the
new-policy expected advantage by `E_t[r_t(theta) A_hat_t]`. At `theta = theta_old`, `r_t = 1`,
and differentiating this surrogate recovers the usual policy-gradient direction.

Conservative policy iteration explains why this surrogate is local. For a conservative mixture
step `(1 - alpha) pi + alpha pi_prime`, the first derivative of return is proportional to the
candidate policy's average advantage, while the improvement guarantee subtracts a term quadratic
in the step size. In other words, the linear surrogate is reliable near the current policy and
becomes over-optimistic as the update moves away.

The natural distance signal is the KL divergence from `pi_old` to `pi_theta` on states in the
batch. Theory supports leashing the local surrogate with a KL term or constraint, and the
practical question is how that leash can be made simple enough for repeated first-order minibatch
updates.

## Baselines

**Vanilla policy gradient / actor-critic.** Differentiate
`E_t[log pi_theta(a_t | s_t) A_hat_t]` and take a gradient step, usually with a learned value
baseline. Core idea: follow the variance-reduced policy-gradient estimate directly.

**Conservative policy iteration.** Optimize the candidate policy's expected advantage under the
old state distribution, using conservative mixture steps to retain a monotonic-improvement style
guarantee. Core idea: the average-advantage surrogate is the right first-order object.

**KL-constrained trust-region update.** Maximize the importance-weighted surrogate subject to a
KL bound between old and new policies. Core idea: optimize the local return model only inside the
region where it remains reliable. The usual implementation uses a linearized objective, a
quadratic KL approximation, Fisher-vector products, conjugate gradient, and line search.

**KL-penalized surrogate.** Replace the hard constraint with a penalty
`E_t[r_t A_hat_t - beta KL(pi_old, pi_theta)]`, optionally adapting `beta` after each update.
Core idea: keep the same trust-region pressure in a first-order objective.

## Evaluation settings

The policy-loss rule would be compared in settings that already stress policy-gradient updates:

- Continuous-control tasks with Gaussian neural policies, learned value baselines, multiple
  random seeds, and fixed rollout budgets.
- Discrete-action visual-control tasks with shared convolutional policy/value networks and
  minibatch updates over rollout segments.
- Higher-dimensional control tasks where curvature-based trust-region machinery is expensive.
- Language-model reinforcement-learning loops where a response mask selects valid completion
  tokens, old/current log-probabilities are available per token, and the policy-loss rule is the
  only component being varied.

The important protocol constraint is that rollout collection, advantage estimation, optimizer,
masking, and model architecture are fixed; only the map from
`(old_log_prob, log_prob, advantages)` to the per-step loss is open.

## Code framework

The training loop already stores the log-probability assigned by the data-collecting policy to
each sampled action, recomputes the current policy's log-probability for those same actions, and
provides a mask for valid response steps. Aggregation over valid entries is handled by a generic
masked reducer.

```python
import torch


def masked_mean(values, mask):
    """Mean of `values` over valid entries selected by `mask`. Already provided."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean", **kwargs):
    """Aggregate a per-step loss matrix into a scalar over valid steps."""
    if loss_agg_mode == "token-mean":
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    raise NotImplementedError(loss_agg_mode)


def compute_policy_loss(
    old_log_prob: torch.Tensor,   # (batch, length) log pi_old(a) at collection time
    log_prob: torch.Tensor,       # (batch, length) log pi_theta(a) now
    advantages: torch.Tensor,     # (batch, length) advantage estimate (given)
    response_mask: torch.Tensor,  # (batch, length) 1 for valid steps, 0 for padding
    loss_agg_mode: str = "token-mean",
    policy_loss_config=None,
):
    """Map rollout-batch log-probs and advantages to a scalar policy loss."""
    # TODO: define the per-step policy-loss rule.
    pass
```
