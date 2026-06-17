## Research question

When a language-model policy is fine-tuned with reinforcement learning against a reward, the
policy needs a pressure that keeps it near a trusted frozen reference policy. Without that
pressure, the policy can drift toward text that exploits the reward signal rather than preserving
the behavior that made the reference useful. The principled regularizer is the per-token
forward KL from the current policy to the reference,
`KL(π_θ ‖ π_ref) = Σ_v π_θ(v) log(π_θ(v)/π_ref(v))`, but that exact sum runs over the full
vocabulary at every generated position.

The training loop usually does not keep the full distributions for this purpose. For each sampled
response token `x`, it has the current-policy log-probability `log π_θ(x)` and the frozen-reference
log-probability `log π_ref(x)`. The problem is to turn those two numbers into a per-token penalty
tensor that can be masked, aggregated across response tokens, multiplied by a KL coefficient, and
added to the actor loss. Because this quantity is in the differentiated loss path, its gradient
with respect to the current policy matters as much as its forward value.

## Background

**Single-sample KL estimation.** With `x ∼ π_θ`, define the signed log-ratio
`l = log π_θ(x) − log π_ref(x)`, the current-over-reference ratio
`r = exp(l) = π_θ(x)/π_ref(x)`, and the reference-over-current ratio
`s = exp(−l) = π_ref(x)/π_θ(x)`. The target divergence is
`KL(π_θ ‖ π_ref) = E_{x∼π_θ}[l]`. A Monte-Carlo estimator chooses some per-sample function of
`l` and averages it over sampled tokens.

**A control-variate fact.** Under on-policy samples, `s` has mean one:
`E_{x∼π_θ}[s] = Σ_x π_θ(x) π_ref(x)/π_θ(x) = 1`. Therefore `s − 1` has zero mean and can be
added to a value estimator as a control variate without changing its expectation. The value
estimator and the differentiated loss do not have the same requirements, though: a zero-mean
control variate for the forward estimate can still change the gradient that reaches the actor.

**Local second-order geometry.** Around `π_θ = π_ref`, many smooth divergence-like objectives share
the same Fisher quadratic. For the squared log-ratio surrogate, the local scalar function is
`f(s) = ½(log s)^2`; its derivatives are
`f'(s) = log(s)/s` and `f''(s) = (1 − log s)/s²`, so `f''(1) = 1`, matching the local curvature of
the KL integrand `−log s`. This explains why a biased per-sample surrogate can track KL closely
when the current policy and reference policy remain close.

**KL-regularized RL for language models.** A pretrained or supervised-fine-tuned model supplies
`π_ref`, and `π_θ` is optimized to raise reward while staying close to it. PPO-style RLHF commonly
uses a KL penalty either as a reward adjustment or as an actor-side loss term. This distinction is
important: an estimator that is excellent as a non-differentiated diagnostic or reward shaping term
can have the wrong actor gradient when its sampled value is placed directly inside the loss.

**Per-token log-ratio tails.** In long reasoning rollouts, most sampled tokens may sit near the
reference while a few rare tokens or branch-changing tokens have much larger positive or negative
log-ratios. The natural yardstick for a per-token penalty is therefore not only whether its average
estimates KL, but also how much influence one extreme sampled token has on the scalar loss and on
the gradient.

## Baselines

The existing estimator choices all consume the same two tensors, `logprob` and `ref_logprob`, and
return one penalty value per sampled token.

**Signed estimator (`k1`).** The direct Monte-Carlo estimate is `k1 = l`. Its mean is exactly
`E_{π_θ}[l] = KL(π_θ ‖ π_ref)`, so its value estimate is unbiased. Its weakness is that the
per-token value is signed even though KL is non-negative: sampled tokens with lower current-policy
probability than reference probability contribute negative values, and positive and negative
excursions cancel only after averaging. In the actor-loss path, differentiating the sample value
gives `∇_θ k1 = ∇_θ log π_θ(x)`, whose expectation under `π_θ` is the score identity
`E_{π_θ}[∇_θ log π_θ(x)] = 0`. The value is unbiased, but the differentiated loss contributes
zero-mean score noise rather than the target KL gradient.

**Squared estimator (`k2`).** The low-variance smooth surrogate is `k2 = ½ l²`. Its value estimate is
biased, but it is always non-negative and has the local KL curvature described above because
`f''(1)=1` for `f(s)=½(log s)^2`. Its sampled-loss gradient is
`∇_θ k2 = l ∇_θ log π_θ(x)`, and the expectation of that sampled gradient is
`∇_θ KL(π_θ ‖ π_ref)`. Its weakness is tail sensitivity: the value grows quadratically with the
log-ratio magnitude, and the gradient weight grows linearly as `l`, so a small number of extreme
sampled tokens can dominate both the penalty and its gradient.

**Control-variate estimator (`k3`).** Using the zero-mean quantity `s − 1` with unit coefficient
gives `k3 = (s − 1) − log s = (s − 1) + l`. Since `log s ≤ s − 1` for `s > 0`, this estimator is
non-negative pointwise, and because `E_{π_θ}[s − 1] = 0`, it is still unbiased for
`KL(π_θ ‖ π_ref)`. Its weakness appears in the actor-loss path: differentiating the sampled value
gives `∇_θ k3 = (1 − s) ∇_θ log π_θ(x)`, and the expectation of that sampled gradient is the
reverse-KL gradient `∇_θ KL(π_ref ‖ π_θ)`, not `∇_θ KL(π_θ ‖ π_ref)`. It also requires an exponential
of `−l`, so implementations clamp that exponent input for numerical stability.

## Evaluation settings

The natural evaluation setting is RL fine-tuning for autoregressive language models with a frozen
reference policy, sampled rollouts, response masks, and a small KL coefficient in the actor loss.
The comparison keeps the model, prompts, reward manager, advantage estimator, optimizer, rollout
sampling, masking, aggregation rule, and KL coefficient fixed while changing only the per-token
function that maps `logprob` and `ref_logprob` to a penalty tensor.

Metrics are downstream reasoning accuracy and training stability under the same compute budget.
The divergence value itself is a diagnostic; the practical question is whether the actor loss gets
a usable regularization signal without letting a few sampled tokens dominate updates.

## Code framework

The surrounding trainer already computes current-policy and reference log-probabilities at sampled
response tokens, applies the response mask, aggregates across valid response tokens, scales by the
KL coefficient, and adds the result to the actor objective. The only open slot is the per-token
function of the two log-probability tensors.

```python
import torch


def compute_kl_penalty(
    logprob: torch.Tensor,      # current policy log-probs at sampled response tokens
    ref_logprob: torch.Tensor,  # frozen reference log-probs at the same tokens
) -> torch.Tensor:
    """Return one differentiable penalty value per sampled token."""
    # TODO: fill in this per-token mapping.
    pass


def actor_kl_loss(logprob, ref_logprob, response_mask, kl_coef, aggregate):
    kl = compute_kl_penalty(logprob, ref_logprob)
    kl = kl * response_mask
    kl_loss = aggregate(kl, response_mask)
    return kl_coef * kl_loss
```
