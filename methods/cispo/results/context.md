# Context: clipped importance sampling for LLM RL (circa 2025)

## Research question

Large language models are trained with reinforcement learning to lift reasoning ability —
competition mathematics, coding — by rewarding correct final answers and letting the model
discover longer chains of thought. The standard recipe is a PPO-style clipped policy-gradient
update: roll out responses from a frozen "old" policy `π_old`, score them, partition the batch
into mini-batches, and take several gradient steps on the current policy `π_θ`, reweighting
each token by an importance ratio between current and old policies to correct for the fact that
the data is off-policy. The single design axis under study is **how the per-token log-prob ratio
is formed, clipped, and turned into a loss** — the advantage estimator, reward, rollout, and KL
configuration are held fixed.

Within reasoning RL, the tokens whose ratio moves the most after an update are frequently the
low-probability, high-entropy "fork" tokens — the ones that begin a reflection or a correction
("However", "Wait", "Recheck"). Because they were rare under `π_old`, even a modest absolute
change in their probability is a large *ratio*. The question is how to form a clipped,
off-policy-corrected per-token objective for this setting.

## Background

**The off-policy setting and why clipping exists.** In PPO-style RL the policy is updated on
samples from an earlier snapshot `π_old`. To reuse those samples for the current parameters `θ`,
each sample's contribution is reweighted by the importance ratio
`r_{i,t}(θ) = π_θ(y_{i,t}|x,y_{i,<t}) / π_old(y_{i,t}|x,y_{i,<t})`, and the update is constrained
to a proximal region of `π_old` so samples that drift too far off-policy do not dominate the
gradient. The PPO clipped surrogate enforces this by the pessimistic combination
`min( r A, clip(r, 1−ε, 1+ε) A )`, whose gradient is zero once the ratio leaves `[1−ε, 1+ε]` in
the direction that would over-improve.

**The PPO/GRPO token-level objective.** The token-level objective is
`E_t[ min( r_{i,t} Â_{i,t}, clip(r_{i,t}, 1−ε, 1+ε) Â_{i,t} ) ]`. For a positive-advantage token
whose probability the policy raises past `1+ε`, the clipped branch is flat: `∂/∂θ` of that term
is zero. The clip band `ε` is typically symmetric, with optional asymmetric edges `ε_low` and
`ε_high`.

**REINFORCE with an IS weight.** The off-policy policy gradient can also be written without a
clipped surrogate at all: the score-function estimator `E[ w · A · ∇_θ log π_θ ]`, where `w` is
an importance weight correcting the `π_old`→`π_θ` mismatch. Here the IS weight `w` multiplies the
gradient as a *scalar coefficient*, and the gradient flows through `log π_θ` directly. In this
form the clip would act on `w`, the coefficient, rather than on the gradient path.

**Stop-gradient as an available primitive.** PyTorch's `detach` (`sg[·]`) lets a quantity
contribute its forward *value* while contributing *no gradient*. It is the standard tool when a
factor in a loss should set a magnitude without itself being a path the optimizer can move along —
the surrounding live factors carry the gradient.

## The editable interface (this study's scaffold)

The object under design is a single policy-loss function in the verl PPO trainer that maps the
per-token tensors to a scalar loss plus diagnostics:

```python
@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, response_length): log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length): log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length): advantage estimate
    response_mask: torch.Tensor,     # (bs, response_length): 1 on real response tokens
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ...
```

`config.clip_ratio` (with optional asymmetric `clip_ratio_low` / `clip_ratio_high`) supplies the
clip band; `agg_loss(loss_mat, loss_mask, loss_agg_mode, **config.global_batch_info)` performs the
mask-aware aggregation; `verl_F.masked_mean` is available for diagnostics. The function must
return `(pg_loss, metrics)` with at least `actor/pg_clipfrac` and `actor/ppo_kl` as Python floats.
The design question is what to put between the inputs and `agg_loss`.
