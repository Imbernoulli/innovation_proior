**Problem.** Token-level clipped PPO won the ladder (`score_mean −0.6758`), but its clip *deletes*
a token's gradient once the ratio leaves the band (the surrogate's `min` switches to a flat,
θ-independent branch). The tokens pushed past the band are disproportionately the low-probability,
high-entropy *fork* tokens — "However", "Wait", "Recheck" — that begin a reflection or
self-correction. So the strongest baseline systematically discards, on every update, the
highest-leverage gradients for long reasoning.

**Key idea (CISPO — clip the IS weight, not the surrogate).** The danger a large ratio poses is
variance, a property of the *coefficient magnitude*, not of whether the gradient is gated. Write the
off-policy gradient as IS-weighted REINFORCE, `E[ w_{i,t} Â ∇ log π_θ ]`; bound the coefficient with
a clip; and put the clipped weight inside a **stop-gradient** so it bounds the forward coefficient
while the gradient flows only through the live `log π_θ`:

```
L = − sg[ clip(w_{i,t}, 1−ε_low, 1+ε_high) ] · Â_{i,t} · log π_θ(y_{i,t}|·),
  w_{i,t} = exp(log_prob − old_log_prob),   ∇_θ L = − sg[clip(w)] · Â · ∇ log π_θ.
```

Every token stays in the update (a fork token with `w=5` contributes `(1+ε_high) Â ∇ log π_θ`, not
zero), and the coefficient is bounded so no stale large-ratio token can dominate. The stop-gradient
is essential: differentiating `clip(w)` directly would add a spurious `Â log π_θ ∇clip(w)` term —
zero out-of-band (re-creating the gate) or double-counting in-band.

**Why past the strongest baseline.** CISPO is token-level with one change — the clip moves from the
*surrogate* to the IS *weight* under a stop-gradient — so it keeps token-level's per-token
granularity and resolution while no longer zeroing the rare reasoning-token gradients. Bar to clear:
token-level's −0.6758, with the gain expected on the multi-step benchmarks (MATH-500, AMC) where
fork tokens are most decisive.

**Hyperparameters.** Clip band read from config (`clip_ratio_low`/`clip_ratio_high`; the **upper**
edge is the load-bearing variance knob and can be generous since exceeding it pins, not drops, a
token; the lower edge is nearly inert); log-ratio clamp `[−20, 20]` before `exp`; `token-mean`
aggregation; **no dual-clip floor** (the coefficient is already bounded by `clip(w)`);
`pg_clipfrac` = fraction of tokens whose weight was clipped; `pg_clipfrac_lower` returned as `0.0`.
Verified line-by-line against verl's canonical `compute_policy_loss_cispo`.

```python
# EDITABLE region of custom_policy_loss.py — finale: CISPO (clipped IS weight, stop-grad)
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
    """CISPO: clip the IS weight under a stop-gradient; never zero a token's gradient."""
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # importance ratio w = pi_theta / pi_old, with an overflow guard
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # KEY: clip the IS weight, then stop-gradient it so it bounds the coefficient
    # without zeroing any token's gradient. Gradient flows only through log_prob.
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # objective (maximize): J = sg(clip(w)) * A * log pi_theta ; loss = -J
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
