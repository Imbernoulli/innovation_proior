# Context: shaping the reward signal in policy-gradient fine-tuning

## Research question

A policy `π_θ` (a language model) is improved by reinforcement learning against a scalar
reward. For each sampled response the environment returns a single number — for math-reasoning
fine-tuning this is typically a correctness signal at the end of the sequence (1.0 for a right
final answer, 0.0 otherwise, sometimes with a small format bonus). That per-response scalar is
broadcast onto a `(batch_size, response_length)` per-token tensor (the scalar sitting at the
last valid token), and a policy-gradient / advantage estimator then turns it into a parameter
update.

The estimator is what actually moves the weights, and it is a Monte-Carlo estimator: it
averages `reward × ∇log π` over a finite batch of sampled responses. Two properties of the
*reward numbers themselves* control whether that estimate is usable. First, **variance** — the
single-sample term `R·∇log π` is noisy, and with a handful of rollouts per prompt the estimate
of the true gradient can swing wildly, so each update is partly chasing sampling noise rather
than signal. Second, **scale and location** — the raw reward's absolute magnitude and its
offset from zero feed straight into the size and the sign-pattern of the gradient, and through
that into the effective learning rate, which is otherwise a hand-tuned constant.

The precise problem: given the batch of per-response reward scalars produced by the reward
source, transform them — *before* they reach the advantage estimator — so that the resulting
policy-gradient update has lower variance and a stable, reward-magnitude-independent scale,
without biasing the direction of the update and without erasing the information that
distinguishes good responses from bad ones. The transform sees only the rewards (and which
tokens are valid); it must return a tensor of the same shape, and it must be cheap and
non-differentiable (a pure data transform on the reward tensor).

## Background

**Policy gradients and their variance.** The score-function (REINFORCE) estimator writes the
gradient of expected reward as an expectation that can be sampled:
`∇_θ E[R] = E[ R · ∇_θ log π_θ(a) ]`. Estimating this from a finite batch is unbiased but
high-variance, and the variance is the dominant practical obstacle to making policy gradients
learn quickly and stably — it has been the central concern of the policy-gradient literature
since the method's origin (Williams 1992; Sutton 1984). The variance comes from two places:
the spread of the reward `R` across sampled responses, and the spread of `∇log π`. Anything
that shrinks the effective spread of the term being averaged, without changing what it
estimates on average, directly buys faster, steadier learning.

**An offset that does not change the gradient.** A foundational fact about this estimator
(Williams 1992; Sutton 1984's *reinforcement comparison*) is that the reward can be measured
relative to a reference level without biasing the gradient. For any quantity `b` that does not
depend on the sampled action,
`E[ b · ∇_θ log π_θ(a) ] = b · ∑_a π_θ(a) ∇_θ log π_θ(a) = b · ∑_a ∇_θ π_θ(a) = b · ∇_θ ∑_a π_θ(a) = b · ∇_θ 1 = 0`,
using `π ∇log π = ∇π` and that the probabilities sum to one. So `E[(R−b)∇log π] = E[R∇log π]`
exactly — the gradient is invariant to subtracting any reference that is fixed with respect to
the sampled action, while the *variance* of the per-sample term `(R−b)∇log π` does depend on
`b`. A baseline estimated from the same finite batch is a plug-in approximation to this identity
rather than the identity itself; the exact statement is the action-independent reference result.
This is the lever the field has long used to reduce variance, and it is purely a property of the
estimator, knowable before any particular transform is chosen.

**The all-same-sign pathology with bounded rewards.** When the reward is bounded and one-signed
— the binary correctness signal of math RL is the extreme case, never negative — every sampled
response carries a non-negative weight, so the gradient pushes *up* the log-probability of
*every* sampled response, differing only in how hard. Learning then has to proceed entirely
through differences in magnitude between samples, which is slow and unstable: the policy cannot
explicitly *suppress* a bad response, only fail to reinforce it as strongly. This is a known
failure mode of un-referenced policy gradients on bounded rewards and it is especially acute
for sparse 0/1 outcome rewards.

**Reward/return scale couples to the learning rate.** In the deep-RL implementation literature
(Engstrom et al. 2020, "Implementation Matters in Deep Policy Gradients"; Andrychowicz et al.
2020, "What Matters in On-Policy Reinforcement Learning?") it is documented and ablated that
the *magnitude* of the reward or return signal fed to the estimator behaves like an extra
multiplier on the step size: if the signal lives on scale `σ`, the gradient — and the effective
learning rate — scales with `σ`. Because that scale is arbitrary (a reward model's output range,
or whether a bonus is 0.1 or 1.0, is incidental), it has to be neutralized for a fixed learning
rate to behave consistently across tasks and across training. Engstrom et al. identify a
return-scaling step (divide the signal by the standard deviation of a running discounted sum of
rewards) as a load-bearing code-level optimization of strong PPO implementations; Andrychowicz
et al. ablate a per-minibatch normalization of the signal (subtract its mean, divide by its
standard deviation) as a standard configuration choice. These observations explain how signal
location and scale propagate into the update.

**The RLHF setting.** In RL fine-tuning from a learned reward, the reward model's raw scalar
output has an arbitrary additive offset and scale; reward models are typically re-centered after
training (Stiennon et al. 2020, Ouyang et al. 2022 normalize the reward model so reference/demo
outputs have mean zero) precisely because the offset is meaningless and only relative reward
matters. The reward signal then drives PPO/GRPO-style updates. In the GRPO family the downstream
advantage estimator itself subtracts a per-prompt group baseline and may divide by a per-prompt
group standard deviation; so any upstream shaping of the reward sits on top of a downstream
normalizer and must be designed not to fight it.

## Baselines

These are the prior approaches to handling the reward signal that a new transform would be
measured against.

**Raw / outcome-only (no transform).** Feed the per-response scalar straight through; the
gradient sees the reward exactly as produced. Core idea: trust the reward source and let the
downstream estimator do all the work. **Gap:** inherits every problem above — the high variance
of the un-referenced estimator, the all-same-sign pathology on bounded/sparse rewards, and an
effective learning rate that drifts with whatever scale the reward happens to have. It is the
natural null baseline, not a solution.

**Constant reference subtraction (Williams 1992; Sutton 1984 reinforcement comparison).**
Subtract a fixed reference level `b` from the reward before forming the gradient. Williams
(1992) proves this leaves the gradient unbiased and shows that a well-chosen `b` reduces
variance; the variance-minimizing constant reference is
`b* = E[R ||∇log π||²] / E[||∇log π||²]`, a squared-score-norm-weighted average of the reward. **Gap:**
the optimal `b*` requires per-sample score-norm weights that are awkward and noisy to estimate
online, and a single hand-set constant does nothing about the *scale* of the signal — it
re-centers but does not rescale, so the learning-rate coupling to reward magnitude remains.

**Per-prompt group baseline (the GRPO-style statistic at the reward stage).** Compute, for the
set of responses that share a prompt, the within-group mean (and optionally within-group spread)
and reference each response to *its own prompt's* group. Core idea: the cleanest reference for a
response is the typical reward of other responses to the same prompt, since prompts differ
wildly in difficulty. **Gap:** the per-group statistic is computed from only a handful of
rollouts per prompt, so it is itself a noisy estimate; and it is exactly the statistic the
downstream GRPO estimator already applies, so applying it again upstream risks collapsing the
signal twice with no new information — a different, narrower reference than a single
batch-spanning one.

## Evaluation settings

The natural yardsticks at this point are:

- **Math-reasoning accuracy** of the fine-tuned policy, reported as `mean@1` (greedy / single-
  sample accuracy) on standard math test sets (grade-school word problems, competition-style
  problem sets), with the headline number a mean across several such sets. The reward during
  training is final-answer correctness, so the held-out metric is the same correctness on
  unseen problems.
- A **fixed RL fine-tuning pipeline** so the reward transform is the only thing varied: a small
  instruction-tuned base policy, a GRPO-style advantage estimator, a fixed number of rollouts
  per prompt (a group of responses sampled per prompt), a fixed batch size, fixed optimizer and
  KL-loss settings, and a fixed training-problem set. Everything except the upstream reward
  transform is held constant.
- For the variance-reduction ancestors, the historical yardsticks are the on-policy continuous-
  control benchmarks of the policy-gradient literature, where reward/return normalization is
  ablated as a configuration choice; the metric there is average return over training.

## Code framework

The transform plugs into a fixed RL fine-tuning harness. The reward source has already produced
a per-response scalar and placed it at the last valid token of each response, giving a
`(batch_size, response_length)` tensor `token_level_scores`; a `response_mask` of the same shape
marks valid response tokens; an optional `index` array tags which prompt each response came from
(responses sharing an index are rollouts of the same prompt). Downstream, a *read-only*
advantage estimator (GRPO and relatives) consumes whatever this transform returns and produces
the policy-gradient update. Already-existing primitives in the harness: masked reductions over
the valid tokens (a masked mean, a masked variance), basic tensor ops, and the convention that
an "outcome reward" lives at the last valid token and is otherwise zero.

Nothing about *how* to reshape the rewards is settled — that rule is exactly what is to be
designed — so the substrate is only the generic plumbing: recover per-response scalars from the
token tensor, compute whatever statistics the rule needs, produce a same-shape tensor, and re-
place the result so the "reward at the last valid token" convention is preserved. The single
empty slot is the transform itself.

```python
import torch
import numpy as np
from typing import Optional


@torch.no_grad()
def normalize_rewards(
    token_level_scores: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,        # (bs, response_length); 1 on valid response tokens
    index: np.ndarray = None,           # (bs,) prompt id; responses sharing an id share a prompt
    epsilon: float = 1e-6,
    config: Optional[object] = None,
    **kwargs,
) -> torch.Tensor:                      # (bs, response_length); same shape as input
    bsz, seq_len = token_level_scores.shape

    # recover one scalar per response from the per-token tensor
    scores = token_level_scores.sum(dim=-1)              # (bs,)

    # TODO: the reward transform we will design.
    #       From the batch of per-response scalars (and any statistics we choose to
    #       compute from them), produce a transformed per-response scalar.
    #       new_scores = <transform>(scores, ...)
    new_scores = scores  # placeholder

    # re-place the transformed scalar at each response's last valid token,
    # preserving the "outcome reward at the last valid token" convention
    out = torch.zeros_like(token_level_scores)
    last_idx = (response_mask.long().sum(dim=-1) - 1).clamp(min=0)   # (bs,)
    out[torch.arange(bsz, device=out.device), last_idx] = new_scores
    return out * response_mask
```

The harness hands over the per-response scalars and expects a same-shape tensor back with the
last-token convention preserved; the transform is the one slot to fill.
