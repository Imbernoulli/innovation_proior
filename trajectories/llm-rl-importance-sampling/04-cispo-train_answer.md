Token-level clipped PPO is the strongest rung so far: $\text{score\_mean} = -0.6758$, above the sequence rung's $-0.6973$, and the gain came exactly where I bet it would — GSM8K led, climbing to 0.4554 from the sequence rung's 0.4443, the short-response benchmark where per-token credit assignment pays cleanest, while MATH-500 and AMC essentially held (0.3078, 0.0964). The ladder told a coherent story: discarding tokens was worst (truncation, $-0.751$); coarsening every token's weight into one per-sequence number recovered the long-response benchmarks but over-applied for a small dense model (sequence, $-0.6973$); keeping every token *and* its per-token resolution, with the per-token clip as the leash, was best. And yet I do not think it keeps all the *gradient*, and that is the crack to open.

Look at what the per-token clip does mechanically, because the feature that made it win also makes it leak signal. The clipped surrogate $\min(r_t\hat A,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A)$ has zero slope once a token's ratio leaves the band in the over-improving direction: the $\min$ switches to the clipped branch, which has no dependence on $\theta$, so that token's gradient is identically zero. The clip does not down-weight an out-of-band token; it *deletes* it from the update. For most tokens that is desirable — the leash that keeps a stale ratio from being milked — but which tokens leave the band? A token's ratio $r_t = \exp(\text{log\_prob} - \text{old\_log\_prob})$ is large in *relative* terms when its old probability was small and the new policy now likes it more, so the clip fires preferentially on *low-probability* tokens. And in a reasoning trace the low-probability, high-entropy tokens are not noise — they are the fork tokens, the "However", the "Wait", the "Let me reconsider" that begins a self-correction or switches the line of attack, the tokens that decide whether a long chain recovers from a wrong turn. The per-token clip, on every update, zeroes the gradient on precisely the tokens that move the reasoning. Token-level won the ladder, but it won while systematically discarding its highest-leverage gradients.

This reframes the question. The previous three rungs all argued about *granularity* — token vs sequence vs prefix — but they all inherited the same clipped-*surrogate* shape, and that shape is what zeroes the fork-token gradients. So the question is no longer "what granularity" but "is clipping the surrogate even the right way to control variance?" Go back to why a large ratio is dangerous: an importance weight is unbiased only in aggregate, and a heavy-tailed weight makes the gradient estimate wander. The per-token clip controls that variance by capping a large-ratio token's influence at *zero* — capping by deletion. But capping influence and deleting the gradient are not the same operation. What I actually want is for a large-ratio token to contribute a *bounded* amount, not nothing: keep every token's gradient *direction* $\nabla\log\pi_\theta$ while bounding the *magnitude* of its coefficient.

I propose **CISPO** — clip the importance-sampling weight, not the surrogate, under a stop-gradient. Write the gradient from the off-policy policy gradient itself rather than from the clipped surrogate. The score-function form is $\mathbb{E}[w_{i,t}\hat A_{i,t}\nabla\log\pi_\theta(y_{i,t}\mid\cdot)]$, plain REINFORCE with an importance weight: the weight multiplies the gradient as a scalar coefficient and the gradient flows through $\log\pi_\theta$ directly. In this form there is no clipped surrogate, so nothing whose slope goes to zero — every token contributes $w_{i,t}\hat A\nabla\log\pi_\theta$ no matter how large $w_{i,t}$ is — and the variance problem is isolated cleanly in the *magnitude of the coefficient*, which I can bound directly without deleting anything. Replace $w_{i,t}$ with $\mathrm{clip}(w_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$: now the coefficient on every token's score gradient is at most $1+\varepsilon_{\text{high}}$ in magnitude, bounding the variance any single stale large-ratio token can contribute — the property the surrogate clip was buying — but a fork token with $w = 5$ now contributes $(1+\varepsilon_{\text{high}})\hat A\nabla\log\pi_\theta$, a bounded but *nonzero* gradient in the right direction, instead of the zero token-level gave it.

There is one trap that, gotten wrong, re-creates token-level's failure. If I literally put $\mathrm{clip}(w_{i,t})$ in the loss and let autograd differentiate it, the product rule gives two terms: the one I want, $\mathrm{clip}(w)\hat A\nabla\log\pi_\theta$, plus a spurious $\hat A\log\pi_\theta\nabla\mathrm{clip}(w)$. In the clipped flat region $\nabla\mathrm{clip}(w) = 0$, so out-of-band tokens lose the coefficient-weighted score term I was trying to keep — right back to gating; in-band $\nabla\mathrm{clip}(w) = w\nabla\log\pi_\theta$, a spurious extra contribution. Either way wrong. The whole point is that the *value* of the coefficient should be the clipped weight while *no gradient* flows through it — precisely a stop-gradient. Write the coefficient as $\mathrm{sg}[\mathrm{clip}(w_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})]$. In the forward pass it is the clipped weight, bounding each token's coefficient; in the backward pass it is a constant — the optimizer cannot move $\theta$ to game the clip, there is no flat region to kill a gradient, and the gradient flows only through the live $\log\pi_\theta$. The loss to minimize is

$$L = -\,\mathrm{sg}\!\big[\mathrm{clip}(w_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\big]\cdot\hat A_{i,t}\cdot\log\pi_\theta(y_{i,t}\mid\cdot),\qquad \nabla_\theta L = -\,\mathrm{sg}[\mathrm{clip}(w)]\cdot\hat A\cdot\nabla\log\pi_\theta,$$

bounded-coefficient REINFORCE with every token present. The clip caps the variance; the stop-gradient keeps the gate from ever deleting a gradient.

The clip band is not the same as token-level's even though it reuses the symbol. In the per-token surrogate $\varepsilon$ defines a trust region and a symmetric ~0.2 is natural; here the clip sits on the IS *weight* and its only job is variance control of the coefficient, and the asymmetry matters. The dangerous direction is $w$ becoming *large* — a rare token the new policy now loves — which inflates the coefficient and the variance; $w$ becoming small just shrinks a coefficient toward zero, harmless. So the upper edge $1+\varepsilon_{\text{high}}$ is load-bearing and can be set generously, since a token exceeding it is *kept*, coefficient pinned, not dropped, while the lower edge is nearly inert. I read both from config (`clip_ratio_low`/`clip_ratio_high`, falling back to `clip_ratio`), never hardcoding, and keep the $[-20, 20]$ log-ratio clamp before `exp` as the overflow guard. I do *not* carry the dual-clip floor from token-level: that floor bounded a large-ratio / negative-advantage *surrogate* blow-up, but here the coefficient is already bounded by $\mathrm{clip}(w)$, so that runaway cannot occur, and `pg_clipfrac_lower` returns 0.0. The diagnostics shift meaning slightly: `pg_clipfrac` is now the fraction of tokens whose *weight* was actually clipped, $\mathrm{masked\_mean}(\text{ratio} \ne \text{clipped\_ratio})$, not where a surrogate branch bit; `ppo_kl` is the same k1 estimate. Aggregation is `token-mean`, matching token-level's granularity exactly — the ratio, clip, and gradient are all per-token; the only thing I changed is *what gets clipped*, the clip moved from the surrogate to the IS weight under a stop-gradient. This rung is deliberately the smallest possible departure from the winning method.

The bar this finale must clear is token-level's $-0.6758$ (GSM8K 0.4554, MATH-500 0.3078, AMC 0.0964). CISPO is token-level with one change — the fork tokens its clip deletes now contribute bounded, nonzero gradients — so the clean prediction is that CISPO at least matches it overall and improves specifically where retaining rare reasoning-token gradients matters most: the multi-step benchmarks, MATH-500 and AMC, where reflection and self-correction are most decisive. The sharpest falsifiable test is a gain concentrated there rather than on GSM8K, which would confirm that token-level was leaking exactly the rare-token gradients long reasoning needs. The honest risk I hold rather than hide: this is a small dense 0.5B model at only 100 steps, and CISPO's advantage was established where the deleted tokens were both numerous and high-leverage; if the fork tokens are too few to matter at this scale, CISPO will merely tie. The mechanistic confirmation I would look for is a `pg_clipfrac` comparable to token-level's clip rate but with *no* zeroed gradients on those tokens — the bounded-coefficient gradient doing what the derivation says.

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
