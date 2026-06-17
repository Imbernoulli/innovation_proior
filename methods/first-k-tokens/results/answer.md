# DAPO, distilled

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) is a critic-free, group-relative
policy-gradient algorithm for large-scale long-chain-of-thought LLM reasoning RL with rule-based
(verifiable) rewards. It is GRPO with four targeted fixes — each closing a specific failure mode
observed when running the naive baseline at scale — plus removal of the KL penalty.

## Problem it solves

Online RL from a base model on verifiable math, where the obvious recipe (GRPO + symmetric PPO
clip + sample-level loss + punitive truncation + KL leash) plateaus and destabilizes: the policy's
entropy collapses (exploration dies), all-correct/all-wrong groups give zero gradient and the
effective batch shrinks over training, sample-level loss reduction under-weights long responses, and
truncated overlong samples inject reward noise.

## Key idea

Keep the clipped group-relative surrogate, but:

1. **Clip-Higher (decoupled clip).** Split the symmetric clip into `ε_low` and `ε_high` and raise
   only the upper one. The symmetric clip caps a positive-advantage token's new probability at
   `π_old·(1+ε)`; for a low-probability exploration token (`π_old=0.01`) this is a tight absolute
   cap (`0.01 * 1.2 = 0.012`), while for a high-probability exploitation token (`π_old=0.9`) it
   exceeds the probability simplex (`0.9 * 1.2 = 1.08`). Loosening only the upper side
   (`ε_low=0.2`, `ε_high=0.28`) gives exploration tokens more room to rise; keeping `ε_low` tight
   prevents probabilities being driven to zero (which would collapse the sampling space).
2. **Dynamic Sampling.** Under the binary correctness reward, group-relative advantage
   `Â = (R−mean)/(std+ε_num)` is zero whenever a group is all-correct or all-wrong; the all-correct
   fraction grows over training, starving the batch.
   Over-sample and keep only mixed groups (`0 < #correct < G`), refilling to a fixed batch size, so
   every prompt carries a nonzero, well-defined advantage and the effective batch stays constant.
3. **Token-Level Policy-Gradient Loss.** GRPO's sample-level reduction `(1/G)Σ_i (1/|o_i|)Σ_t`
   weights a token by `1/|o_i|`, under-learning long high-quality chains and under-penalizing long
   garbage. Pool all tokens and normalize by the total token count, `(1/Σ_i|o_i|)Σ_iΣ_t`, so every
   token counts equally and longer responses have proportionally more influence.
4. **Overlong Reward Shaping.** Truncated overlong samples poison the verifiable reward. Mask their
   loss when overlong filtering is enabled, and add a smooth length-aware penalty `R_length` that is
   0 through `L_max − L_cache`, ramps linearly to `−1` across the cache, and is `−1` beyond the hard
   generation cap — steering length without punishing sound long reasoning before the buffer starts.

The KL penalty to a frozen reference is removed: in long-CoT RL the policy is meant to diverge far
from the base model, so the alignment-era leash hurts.

## Final objective

```
J_DAPO(θ) = E_{(q,a)~D, {o_i}_{i=1}^G ~ π_old(·|q)} [
              (1 / Σ_{i=1}^G |o_i|) Σ_{i=1}^G Σ_{t=1}^{|o_i|}
                min( r_{i,t}(θ) Â_{i,t},  clip(r_{i,t}(θ), 1−ε_low, 1+ε_high) Â_{i,t} ) ]
        s.t.  0 < |{ o_i : is_equivalent(a, o_i) }| < G,
```

with

```
r_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t}),
Â_{i,t}    = (R_i − mean({R_j}_{j=1}^G)) / (std({R_j}_{j=1}^G) + ε_num),
```

rule-based correctness reward `R(ŷ,y) = +1 if is_equivalent(ŷ,y) else −1`, plus the soft overlong
penalty added to `R_i`. `ε_num` is the small implementation guard against division by zero; when all
group rewards are equal, the numerator is still zero:

```
R_length(y) = 0                                      if |y| ≤ L_max − L_cache
            = ((L_max − L_cache) − |y|) / L_cache    if L_max − L_cache < |y| ≤ L_max
            = −1                                     if |y| > L_max.
```

At `|y| = L_max − L_cache` the middle branch is `0`; at `|y| = L_max` it is `−1`, so the
piecewise penalty is continuous at both joins. Defaults: `ε_low = 0.2`, `ε_high = 0.28`,
`max_response_length = 20480`, `overlong_buffer_len = 4096` (no penalty through `16384`), `G = 16`.
No KL term.

## Algorithm

```
for step = 1 .. M:
    π_old ← π_θ
    # Dynamic Sampling: refill the batch with mixed groups only
    buffer = []
    while |buffer| < N:
        sample a prompt q (with answer a); sample G outputs {o_i} ~ π_old(·|q)
        score each o_i with the rule-based + overlong-shaped reward
        if 0 < #correct < G:  add the group to buffer
    for each group: Â_{i,t} = (R_i − mean) / (std + ε_num)  # group-relative advantage
    for inner iteration = 1 .. μ:
        maximize J_DAPO(θ)  (token-level, decoupled clip)  # mask truncated samples' loss
```

## Implementation

The clipped surrogate is computed in loss (minimization) form, where maximizing
`min(r·Â, clip(r)·Â)` becomes minimizing `max(−r·Â, −clip(r)·Â)`.

```python
import torch
from typing import Any, Optional


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, T)  log π_old
    log_prob: torch.Tensor,          # (bs, T)  log π_θ
    advantages: torch.Tensor,        # (bs, T)  group-relative Â
    response_mask: torch.Tensor,     # (bs, T)  1 on response tokens
    loss_agg_mode: str = "token-mean",      # token-level pooling: (1 / Σ|o_i|)
    config: Optional["ActorConfig"] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)        # dual lower clip for Â<0 tail
    assert clip_ratio_c > 1.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)     # = -min(r·Â, clip·Â)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(                                          # token-mean = (1 / Σ|o_i|) ΣΣ
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }


def shape_reward(
    raw_correct,
    length,
    truncated,
    response_mask,
    max_response_length=20480,
    overlong_buffer_len=4096,
    penalty_factor=1.0,
):
    length_f = length.to(torch.float32)
    R = torch.where(raw_correct, torch.ones_like(length_f), -torch.ones_like(length_f))

    expected_len = max_response_length - overlong_buffer_len
    exceed_len = length_f - expected_len
    R_length = torch.minimum(
        -exceed_len / overlong_buffer_len * penalty_factor,
        torch.zeros_like(length_f),
    ).clamp(min=-penalty_factor)                         # 0 through expected_len, -1 at hard cap

    sequence_mask = (~truncated).to(response_mask.dtype).unsqueeze(-1)
    filtered_response_mask = response_mask * sequence_mask
    return R + R_length, filtered_response_mask


def build_training_batch(prompts, policy_old, reward_fn, sampler, G, target_prompts):
    kept = []
    while len(kept) < target_prompts:
        for q, a in sampler(prompts):
            outputs = policy_old.sample_group(q, G)
            correct = torch.tensor([is_equivalent(a, o) for o in outputs])
            n_correct = int(correct.sum().item())
            if 0 < n_correct < G:
                rewards, response_mask = reward_fn(q, a, outputs)
                kept.append((q, a, outputs, rewards, response_mask))
                if len(kept) >= target_prompts:
                    break
    return assemble_group_relative_advantages(kept)
```

Token-level aggregation (`"token-mean"`) is `masked_sum(loss, mask) / total_valid_tokens` — exactly
`(1/Σ|o_i|) Σ_i Σ_t`; the rejected sample-level reduction is the per-sequence-then-average mode.
The decoupled clip is realized by passing `clip_ratio_low = ε_low` and `clip_ratio_high = ε_high`.
