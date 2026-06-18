**Problem.** Rung 1 (truncated IS) lost worst (`score_mean −0.751`) by discarding the late-token
gradient that long-response benchmarks need. The fix must keep *every* token in the gradient while
making the per-token ratios less noisy. The per-token weight `w_{i,t}` is a one-sample IS estimate
per next-token distribution — all variance, no averaging — and inside the clipped loss it acts as a
noisy per-token gradient multiplier that stacks over the response.

**Key idea (sequence-level IS / GSPO).** Match the unit of the correction to the unit of the reward:
the verifier scores a whole response, so do importance sampling per sequence. Replace the per-token
ratio with the length-normalized geometric mean
`s_i = (π_θ(y_i|x)/π_old(y_i|x))^{1/|y_i|} = exp((1/|y_i|) Σ_t log(π_θ/π_old))`, broadcast to every
token, clipped and aggregated at the sequence level. The `1/|y_i|` normalization unifies the
numerical scale across lengths (one clip band fits all) and reduces variance (averaging beats
summing per-token log-ratios). The gradient differs from the per-token method *only* in the weight
on each token's score function: one equal `s_i` per response instead of `|y_i|` unequal noisy
`w_{i,t}` — every token kept, the weight disparity removed.

**Why over rung 1.** Truncation kept the noisy weights but deleted most tokens; sequence-level IS
keeps every token but deletes the weight disparity — the opposite, principled remedy. It should
recover exactly the long-response benchmarks (MATH-500, AMC) that truncation hurt.

**Straight-through realization.** `log_seq_ratio = log_prob − sg[log_prob] + sg[neg_kl_seq]`: forward
value is the broadcast sequence ratio `s_i`; backward path runs through the live per-token `log_prob`,
giving each token's score the equal weight `s_i`. Clamp `log_seq_ratio ≤ 10` before `exp`.

**Hyperparameters.** Clip band read from config (`clip_ratio_low`/`clip_ratio_high`, much narrower
than a token-level ε because `s_i` hugs 1); **`seq-mean-token-mean` aggregation** (mirrors the
objective's `(1/G)(1/|y_i|)` two-level average; a flat token-mean would let long responses dominate
and undo the length normalization); no dual-clip floor; `pg_clipfrac_lower` returned as `0.0`.

```python
# EDITABLE region of custom_policy_loss.py — rung 2: sequence-level IS (GSPO)
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sequence-level IS (GSPO): one scalar ratio per sequence."""
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    neg_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # straight-through: keep per-token log_prob gradient, ratio value is per-sequence
    log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    seq_ratio = torch.exp(log_seq_ratio)

    pg_losses1 = -advantages * seq_ratio
    pg_losses2 = -advantages * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # GSPO aggregates at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
```
