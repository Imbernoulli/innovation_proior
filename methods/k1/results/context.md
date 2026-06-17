## Research question

An autoregressive language-model policy is being fine-tuned with policy-gradient RL, and a
reference penalty is used to keep the current policy close to a frozen reference model. At each
response position the divergence of interest is the forward KL from the current next-token
distribution to the reference next-token distribution,

```text
KL[p_theta || p_ref] = sum_x p_theta(x) log(p_theta(x) / p_ref(x)).
```

The exact vocabulary sum is expensive at training scale, and the actor path commonly carries
only aligned per-token log-probabilities for the sampled response token: one under the current
policy and one under the reference. The problem is to turn those two same-shaped tensors into a
per-token penalty that can be masked, aggregated, and added to the actor loss while preserving
the intended value and gradient behavior as clearly as possible.

## Background

The actor samples response tokens from the current policy, computes log-probabilities for those
sampled tokens under both the current policy and the reference, and then optimizes a
policy-gradient surrogate. A reference penalty can enter as reward shaping before advantage
estimation or as a separate differentiable actor-loss term. Those two placements ask different
questions: a reward-side estimate mainly needs the right sampled value, while a differentiable
loss-side estimate also has to be judged by the gradient obtained when backpropagating through
the sampled log-probability tensor.

- **Monte-Carlo estimation.** If a target quantity is an expectation under the sampling
  distribution, a sample average is unbiased for that expectation, and the sampling error is
  governed by the variance of the per-sample integrand. In an RL update the number of response
  tokens is finite, so high per-token variance becomes noisy loss and noisy diagnostics.
- **Signed per-sample estimates.** A divergence is nonnegative after averaging over the support,
  but a single sampled log-ratio can be negative. That is not itself a bias error; it is a
  warning that the estimate relies on cancellation and can have large relative noise when the
  true divergence is small.
- **A zero-mean control variate is available.** With `r = p_ref(x) / p_theta(x)` and
  `x ~ p_theta`, `E[r - 1] = sum_x p_ref(x) - 1 = 0`. Adding a multiple of `r - 1` to a
  sampled estimator preserves its mean while potentially reducing variance.
- **Control variates.** For an unbiased estimator `m` and a known-mean statistic `t`, the family
  `m + c(t - E[t])` remains unbiased. The variance-minimizing coefficient is
  `-Cov(m,t) / Var(t)`, but a fixed coefficient can still be useful if it gives a stable,
  nonnegative per-sample form.
- **Score-function identity.** For a normalized parametric distribution,
  `E_{x ~ p_theta}[grad_theta log p_theta(x)] = 0`. This identity separates the gradient of a
  sampled tensor value from the gradient of the expectation that tensor estimates.
- **Local f-divergence equivalence.** Near `p_theta = p_ref`, twice-differentiable
  f-divergences with the same `f''(1)` share the same second-order Fisher quadratic. This is why
  a low-variance quadratic log-ratio can track KL locally even when its value is not exactly KL
  away from the diagonal.

## Baselines

The surrounding design space consists of sampled divergence penalties built from the same
aligned log-probabilities.

- **Squared log-ratio.** The per-token value `0.5 * (log p_theta - log p_ref)^2` is always
  nonnegative and has low variance near the diagonal. Its expectation is an f-divergence that
  matches KL to second order, but it is not an unbiased value estimator of KL away from
  `p_theta = p_ref`.
- **Control-variate log-ratio form.** Using `r = p_ref / p_theta`, the value
  `(r - 1) - log r` adds the zero-mean control variate with coefficient one. Concavity of `log`
  makes it nonnegative, and its expectation is KL, but it requires `exp(ref_logprob - logprob)`
  and therefore needs numerical guarding on outlier tokens.
- **Absolute log-ratio.** The absolute value is simple and nonnegative, but its expectation is
  not KL and the kink at zero is awkward in the near-reference regime.
- **Straight-through variants.** A forward value can be combined with the backward pass of the
  squared log-ratio by returning `backward - backward.detach() + forward.detach()`. This adds
  machinery, but it directly addresses the mismatch between value-unbiased sampled estimates
  and the gradient of the expected KL.

The key comparison is therefore not only value bias and variance. In a differentiable actor
loss, the expected autodiff gradient of the sampled penalty must be compared with the true
gradient of the KL expectation.

## Evaluation settings

- **Distribution-level checks:** use simple distributions with analytic KL to compare the
  sample mean, bias, and standard deviation of candidate per-sample estimators.
- **Code-level checks:** feed aligned `logprob` and `ref_logprob` tensors through the actor KL
  hook and verify shape preservation, masking compatibility, numerical behavior, and whether
  gradients flow or are intentionally replaced by a straight-through construction.
- **Training-level checks:** hold the policy model, rollout procedure, reward function,
  optimizer, aggregation mode, response mask, and KL coefficient fixed while swapping only the
  per-token penalty mode. Compare task metrics, measured KL diagnostics, loss noise, and
  gradient behavior without using those outcomes to define the estimator.

## Code framework

The actor code already supplies same-shaped tensors for sampled current-policy and reference
log-probabilities. The open slot is the dispatcher branch that maps those tensors and a mode
string to a same-shaped per-token penalty.

```python
import torch


def kl_penalty_forward(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty: str,
) -> torch.FloatTensor:
    """Return a sampled per-token reference penalty for the selected mode."""
    # TODO: fill the selected same-shaped per-token penalty from aligned sampled log-probs.
    raise NotImplementedError
```

The caller is responsible for response masking, aggregation, coefficient scaling, and adding
the scalar result to the actor objective. The returned tensor must not be detached unless the
selected mode explicitly implements a forward/backward split.
