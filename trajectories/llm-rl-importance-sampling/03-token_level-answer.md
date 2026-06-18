**Problem.** Sequence-level IS (rung 2, `score_mean −0.6973`) beat truncation by keeping every token,
but it reduces variance by *coarsening* the gradient — collapsing every token's weight to one `s_i`
per response. Coarsening only pays when the per-token signal it discards was mostly noise. In this
small, dense (non-MoE) 0.5B regime the per-token ratios are well-behaved, so collapsing them throws
away genuine per-token credit assignment.

**Key idea (token-level clipped PPO / GRPO).** Keep the original granularity: one importance ratio
per token, clipped per token. The surrogate is the pessimistic
`L^CLIP = E_t[ min( r_t Â, clip(r_t, 1−ε, 1+ε) Â ) ]`, whose `min` is a lower bound on the unclipped
surrogate and, by the sign of `Â`, ceilings over-improvement of good tokens while keeping the full
penalty for making them worse (and symmetrically for bad tokens). Plus the **dual-clip floor**: for
`Â < 0`, floor the objective at `c·Â` (`c = 3`) so a large-ratio negative-advantage token cannot
explode the loss (`~ r_t|Â|`). Every token kept, full per-token resolution retained.

**Why strongest here.** It keeps every token (no truncation bias, unlike rung 1) *and* keeps
per-token resolution (no sequence coarsening, unlike rung 2), paying only the modest per-token
variance that was never the binding constraint for a small dense model. Expected to edge above rung 2
on `score_mean`, with GSM8K — short, well-behaved responses where per-token credit assignment pays
cleanest — leading the gain.

**Hyperparameters.** `clip_ratio` (asymmetric `clip_ratio_low`/`clip_ratio_high`) read from config;
`clip_ratio_c = 3.0` (`> 1`, dual-clip floor, engages only on genuine blow-ups); log-ratio clamp
`[−20, 20]` before `exp`; **`token-mean` aggregation** (every valid token weighted equally — the
counterpoint to rung 2's `seq-mean-token-mean`); emits `pg_clipfrac`, `ppo_kl`, and
`pg_clipfrac_lower`.

```python
# EDITABLE region of custom_policy_loss.py — rung 3: token-level clipped PPO (GRPO)
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
    """Token-level vanilla PPO: per-token ratio, per-token clip."""
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
```
