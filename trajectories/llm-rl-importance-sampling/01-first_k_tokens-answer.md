**Problem.** LLM online RL reuses an off-policy rollout batch for several clipped policy-gradient
steps; each token carries an importance ratio `r = exp(log_prob − old_log_prob)`. Late tokens in a
response are conditioned on a long generated prefix, so their ratios drift furthest and are noisiest.
The first rung asks whether freezing those late-token ratios as a *gradient* signal buys stability.

**Key idea (truncated IS).** Keep the ordinary per-token PPO clipped surrogate for the first
`K = 64` response positions, and for every position `t ≥ K` replace the ratio with its detached
value, so no gradient flows through the late-token log-probs via the IS weight. The forward loss
still counts all tokens (loss scale is honest); only the first K tokens contribute to the backward
pass. A position mask `prefix_mask = (arange(T) < K)` selects the live region:
`ratio_eff = ratio·prefix_mask + ratio.detach()·(1−prefix_mask)`.

**Why it is the floor.** Detaching the ratio at `t ≥ K` *removes* those tokens from the policy
gradient — the estimator is the IS gradient computed as if only the first 64 tokens existed. That is
a real, length-growing bias: a 300-token solution learns from only its first ~64 tokens. The bet is
that the discarded late-token signal is mostly drift-noise; in this small, dense (non-MoE) 0.5B
regime with moderate-length MATH responses, the per-token ratios are already well-behaved, so the
trade — discard most of the gradient to suppress a small noise source — is expected to lose. I place
it first as the weakest rung.

**What to watch.** Primary `score_mean` (mean of GSM8K / MATH-500 / AMC accuracy), and the *shape*
of the degradation: a 64-token prefix covers most of a short GSM8K answer but only a fraction of a
long MATH/AMC solution, so truncation should spare GSM8K and bite the long-response benchmarks. That
pattern would refute the late-token-noise hypothesis and tell the next rung to stop discarding tokens.

**Hyperparameters.** `K = 64` (live-gradient prefix length); `clip_ratio` (and asymmetric
`clip_ratio_low`/`clip_ratio_high`) read from config; log-ratio clamp `[−20, 20]` before `exp`;
`token-mean` aggregation; no dual-clip floor (minimal variant, to attribute the result to truncation
alone); `pg_clipfrac_lower` returned as `0.0`.

```python
# EDITABLE region of custom_policy_loss.py — rung 1: first-K truncated IS (K=64)
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
    """First-K truncated IS: per-token ratio for t<K, detached for t>=K."""
    assert config is not None
    K = 64  # prefix length with live IS gradient
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Build prefix mask: 1 for the first K response positions, 0 afterwards.
    T = ratio.shape[-1]
    positions = torch.arange(T, device=ratio.device).unsqueeze(0)  # (1, T)
    prefix_mask = (positions < K).to(ratio.dtype)                  # (1, T)
    # Detach ratio beyond prefix: ratio_eff = ratio*prefix + detach(ratio)*(1-prefix)
    ratio_eff = ratio * prefix_mask + ratio.detach() * (1.0 - prefix_mask)

    pg_losses1 = -advantages * ratio_eff
    pg_losses2 = -advantages * torch.clamp(ratio_eff, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
```
