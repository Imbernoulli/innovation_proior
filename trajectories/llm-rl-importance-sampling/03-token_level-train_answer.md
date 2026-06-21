The sequence-level rung did what I bet it would: $\text{score\_mean}$ climbed from $-0.751$ to $-0.6973$, and the recovery landed exactly where predicted — MATH-500 from 0.2875 back to 0.3108, AMC from 0.0828 to 0.0964, the two long-response benchmarks truncation had crushed, while GSM8K barely moved (0.4391 to 0.4443). Keeping every token and replacing the noisy per-token weight with one coherent per-sequence weight was the right move *relative to discarding tokens*. But the gradient comparison that justified GSPO cuts both ways, and now I have to take its cost seriously: the only difference between the sequence and per-token gradients is the weight on each token's score function — one equal $s_i$ for every token versus the individual $w_{i,t}$ — and "unequal" is not the same as "noise." When the per-token ratios are *well-behaved* (small dense model, no expert reshuffling, moderate length), the per-token weight is genuine *resolution*: it tells the gradient which specific tokens within a response moved the policy and by how much. GSPO buys variance reduction by *coarsening* the gradient, and coarsening only pays when the fine-grained signal it discards was mostly noise. In the large-MoE regime that signal was noise, so coarsening was free; here, where the ratios are trustworthy, it discards real per-token credit assignment — a cost paid whether or not the variance it saves was ever the binding constraint.

So this rung's hypothesis is the inverse of the last one. Rung 1 bet that late-token noise dominated (false — the late tokens carried signal); rung 2 bet that per-token weight disparity dominated and should be averaged away (true against truncation, but possibly over-applied here). I now bet that in *this* regime the per-token ratios are good enough that the right thing is to keep them at full resolution and control their downside not by averaging them into one number but by *clipping each one independently*.

I propose **token-level clipped PPO** (the GRPO surrogate): one importance ratio per token, clipped per token — the original granularity, and the method this whole granularity question was a perturbation of. "Go back to per-token" is not "do nothing," because there is real machinery that makes the per-token clip safe and I want all of it. The off-policy surrogate $L^{\text{CPI}} = \mathbb{E}_t[r_{i,t}\hat A_{i,t}]$ with $r_{i,t} = \exp(\text{log\_prob} - \text{old\_log\_prob})$ is only a local model of the return and overestimates improvement as the policy drifts, so it needs a leash. The per-token leash is the clip: form both $r_t\hat A$ and $\mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A$ and take the pessimistic minimum,

$$L^{\text{CLIP}} = \mathbb{E}_t\big[\min\big(r_t\hat A,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A\big)\big].$$

The $\min$ is a lower bound on the unclipped surrogate, so improvement is never over-estimated, and by the sign of $\hat A$ it acts asymmetrically. For $\hat A > 0$ the smaller candidate is $\min(r_t, 1+\varepsilon)\hat A$: when $r_t < 1+\varepsilon$ this is the full $r_t\hat A$, so driving a good token's probability *down* ($r_t < 1$) feels the full loss — correct — and when $r_t > 1+\varepsilon$ it caps at $(1+\varepsilon)\hat A$, so over-improving a good token from one stale sample stops paying and its gradient there is zero. For $\hat A < 0$ the minimum is the *larger* multiplier $\max(r_t, 1-\varepsilon)\hat A$: when $r_t > 1-\varepsilon$ this is the full $r_t\hat A$, so pushing a bad token's probability *up* feels the full penalty, and when $r_t < 1-\varepsilon$ it floors at $(1-\varepsilon)\hat A$, so over-suppressing a bad token stops paying extra. The single $\min$ thus gives a ceiling on the upside with the full downside penalty for good tokens, and a floor on the downside with the full upside penalty for bad ones — the pessimism *is* the asymmetry. The clip kills the incentive (and the gradient) to move a ratio outside the band, but it does so *per token*, preserving the resolution the sequence method discarded.

There is one corner the two-sided clip leaves open, and unlike the previous rungs I will not skip the guard, because the strongest baseline should be the complete method. Take $\hat A < 0$, where the clipped objective is $\max(r_t, 1-\varepsilon)\hat A$. The clip protects the low side of $r_t$, but the high side is wide open: if $r_t$ becomes very large — a token the old policy gave tiny probability that the new policy now likes — then $\max(r_t, 1-\varepsilon)\hat A = r_t\hat A$, a large negative surrogate, i.e. a positive loss $\sim r_t|\hat A|$ growing without bound, so a handful of stale low-probability negative-advantage tokens could dominate the whole minibatch gradient. I add the **dual-clip floor**: for $\hat A < 0$, never let the objective drop below $c\hat A$ for a constant $c > 1$. Since $\hat A < 0$ and $c > 1$, $c\hat A$ is a finite negative floor capping the loss at $c|\hat A|$ no matter how large $r_t$ grows. I take $c = 3$, large enough that the floor engages only on genuine blow-ups, never on ordinary updates. For $\hat A > 0$ there is no symmetric pathology — the upper clip already ceilings the objective at $(1+\varepsilon)\hat A$ — so the floor applies only to negative-advantage tokens.

Translating to the loss the optimizer minimizes, carrying the sign convention: $\text{negative\_approx\_kl} = \text{log\_prob} - \text{old\_log\_prob}$, clamped to $[-20, 20]$ before `exp`; $\text{ratio} = \exp(\cdot)$; the k1 diagnostic $\text{ppo\_kl} = \mathrm{masked\_mean}(-\text{negative\_approx\_kl})$. The unclipped loss is $\text{pg\_losses1} = -\hat A\cdot\text{ratio}$, the clipped loss $\text{pg\_losses2} = -\hat A\cdot\mathrm{clamp}(\text{ratio}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$; the pessimistic min over objectives becomes the *max* over losses, $\text{clip\_pg\_losses1} = \max(\text{pg\_losses1}, \text{pg\_losses2})$, with $\text{pg\_clipfrac} = \mathrm{masked\_mean}(\text{pg\_losses2} > \text{pg\_losses1})$. The dual floor $c\hat A$ over the objective becomes a *ceiling* on the loss, $\text{pg\_losses3} = -\hat A\cdot c$ (which is $c|\hat A|$ when $\hat A < 0$), and $\text{clip\_pg\_losses2} = \min(\text{pg\_losses3}, \text{clip\_pg\_losses1})$ caps it; $\text{pg\_clipfrac\_lower}$ tracks where it engaged on negative-advantage tokens. The floor applies only for negative advantages: $\text{pg\_losses} = \mathrm{where}(\hat A < 0,\ \text{clip\_pg\_losses2},\ \text{clip\_pg\_losses1})$.

The aggregation is where this rung makes its granularity statement, and it is the deliberate counterpoint to GSPO. I use `token-mean`: sum the per-token clipped losses over the response mask and divide by the token count, every valid token weighted equally, each contributing its own ratio and its own clip decision. There is no per-sequence ratio here to mirror — the ratio, the clip, and the credit assignment are all per-token, so the flat token-mean is the aggregation that matches the object. Rung 2 collapsed the per-token weights to one $s_i$ and aggregated per sequence to protect that collapse; this rung keeps the per-token weights distinct and aggregates per token to *use* them.

The falsifiable expectation against the two measured rungs: this is the complete per-token method — every token kept (no truncation bias, unlike rung 1's $-0.751$) and full per-token resolution retained (no sequence coarsening, unlike rung 2's $-0.6973$). I claim that here the per-token ratios are trustworthy enough that resolution beats coarsening, so token-level should edge above the sequence rung. The sharpest test is GSM8K, where per-token credit assignment on short, well-behaved responses pays cleanest: I expect its accuracy to rise above the sequence rung's 0.4443, toward ~0.455. If token-level fell below sequence-level instead, the per-token variance *was* the binding constraint even at 0.5B and the coarsening was justified — I do not expect that, but it is the outcome that would refute this rung.

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
