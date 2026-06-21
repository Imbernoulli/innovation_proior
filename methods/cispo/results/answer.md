# CISPO (Clipped IS-weight Policy Optimization), distilled

CISPO is a policy-optimization objective for LLM online RL that bounds the variance contributed
by large importance ratios **without ever zeroing a token's gradient**. Instead of clipping the
PPO/GRPO *surrogate* (which deletes the gradient of any token whose ratio leaves the band), CISPO
clips the importance-sampling *weight*, places that clipped weight inside a stop-gradient, and
multiplies it onto a REINFORCE-style `Â · log π_θ` term. Every token — including the rare,
high-entropy fork tokens that drive reasoning — stays in the update, just with a bounded
coefficient.

## Problem it solves

The PPO/GRPO clipped surrogate `min( r Â, clip(r, 1−ε, 1+ε) Â )` has zero slope once the ratio
leaves `[1−ε, 1+ε]`, so it *deletes* that token's gradient. The tokens pushed past the band are
disproportionately the ones that were rare under `π_old` — the high-entropy "However" / "Wait" /
"Recheck" fork tokens that begin a reflection or self-correction. The standard clip therefore
systematically discards the learning signal on the tokens most important for long-horizon
reasoning, on every update. Widening `ε` does not fix it (it also loosens the leash on genuinely
stale tokens): "rare" and "stale" are different axes, and any gate that responds to a large ratio
by removing the gradient cannot tell them apart.

## Key idea

The danger a large ratio poses is *variance*, which is a property of the magnitude of the
coefficient, not of whether the gradient is gated. So bound the coefficient while keeping the
gradient on every token. Write the off-policy gradient as IS-weighted REINFORCE,
`E[ w_{i,t} Â_{i,t} ∇_θ log π_θ ]`, where the weight is a scalar coefficient and the gradient
flows through `log π_θ`. Bound `w` with a clip, and put the clipped weight inside a stop-gradient
`sg[·]` so it bounds the forward coefficient but carries no gradient (no flat region, no gating):

```
L = − sg[ clip(w_{i,t}, 1−ε_low, 1+ε_high) ] · Â_{i,t} · log π_θ(y_{i,t}|x,y_{i,<t}),
    w_{i,t} = π_θ(y_{i,t}|·) / π_old(y_{i,t}|·).
```

The gradient is `∇_θ L = − sg[clip(w)] · Â · ∇_θ log π_θ` — bounded-coefficient REINFORCE, every
token present.

## Why it works

- **No dropped tokens.** A fork token with `w = 5` contributes `(1+ε_high) Â ∇ log π_θ`, a
  bounded but nonzero gradient, where the PPO clip gave it zero.
- **Variance still bounded.** The stop-gradient'd `clip(w)` caps each token's coefficient at
  `1+ε_high`, so no single stale large-ratio token can dominate the gradient.
- **Stop-gradient kills the gate.** Because the clipped weight is detached, the optimizer cannot
  move `θ` to game the clip, and the clip's flat region never zeros a gradient — the gradient runs
  entirely through the live `log π_θ`.

## Defaults and why

- The clip is on the IS *weight*, not the surrogate, so the band differs from PPO's. The **upper
  edge** `1+ε_high` is the variance knob and is load-bearing; it can be set generously because a
  token exceeding it is *kept* (coefficient pinned), not dropped. The **lower edge** `1−ε_low` is
  nearly inert (a small `w` just shrinks a coefficient harmlessly). Both are read from config
  (`clip_ratio_low` / `clip_ratio_high`, falling back to `clip_ratio`); in practice the high edge
  is the knob that carries the variance/signal trade-off.
- Log-ratio clamp `[−20, 20]` before `exp` — overflow guard; the bounds are far outside any band
  that affects a real update.
- **No dual-clip floor.** Token-level PPO needed a `c·A` floor (`c > 1`) to bound the
  large-ratio / negative-advantage surrogate; here the coefficient is already bounded by
  `clip(w)`, so that runaway cannot occur. `pg_clipfrac_lower` is returned as `0.0`.
- Aggregation is **token-mean**: the ratio, clip, and gradient are all per-token, so every valid
  token is weighted equally (not the sequence-level `seq-mean-token-mean`).

## Working code

verl policy-loss function filling the `compute_custom_policy_loss` slot, faithful to verl's
`compute_policy_loss_cispo`. `clipped_ratio_sg` is the detached clipped IS weight; the gradient
flows through the live `log_prob` in `pg_losses`.

```python
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,      # (bs, response_length): log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length): log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length): advantage estimate
    response_mask: torch.Tensor,     # (bs, response_length): 1 on real response tokens
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CISPO: clip the IS weight under a stop-gradient; never zero a token's gradient."""
    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # importance ratio w = π_θ / π_old, with an overflow guard
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # k1 KL diagnostic

    # KEY: clip the IS weight, then stop-gradient it so it bounds the coefficient
    # without zeroing any token's gradient. Gradient flows only through log_prob.
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # objective (maximize): J = sg(clip(w)) * A * log π_θ ; loss = -J
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # fraction of tokens whose IS weight was actually clipped (coefficient pinned)
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )

    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)  # no dual-clip floor in CISPO
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
```

## Relation to prior methods

- **PPO / GRPO** (token-level): clip the *surrogate*, which zeroes the gradient of any token
  whose ratio leaves the band. CISPO clips the *weight* under a stop-gradient, keeping every
  token's gradient.
- **GSPO** (sequence-level): collapses the per-token ratio to one length-normalized
  per-sequence ratio to reduce variance. CISPO keeps the per-token granularity but bounds the
  per-token coefficient instead of replacing it.
- **REINFORCE with an IS weight**: CISPO *is* this estimator with the weight clipped and
  detached; the clip bounds variance, the stop-gradient keeps every token in the update.
