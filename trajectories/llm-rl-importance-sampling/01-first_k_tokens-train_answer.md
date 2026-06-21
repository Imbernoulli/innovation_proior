The policy-loss slot in this training loop reuses a frozen rollout batch for several clipped gradient steps, so each token carries an importance ratio $r_{i,t} = \exp(\text{log\_prob} - \text{old\_log\_prob})$ that corrects for the policy having drifted since the batch was sampled. The per-token clipped surrogate that ships as the default descends from PPO and GRPO, and the standard complaint against it is that the per-token ratio is a single noisy draw with no averaging, whose accumulated noise can destabilize training. Before I spend a full 100-step H200 run on anything clever, I want a clean floor that isolates the cheapest possible version of that complaint: the hypothesis that the gradient noise in clipped LLM RL is concentrated in the *late* tokens of a response, and that I can buy stability simply by refusing to let the importance-sampling ratio carry a live gradient past some prefix length.

The reason to suspect the late tokens is that the per-token log-ratio $\text{log\_prob} - \text{old\_log\_prob}$ tends to grow in magnitude with position. Early tokens are conditioned on the prompt alone, where the old and current policies are still close; late tokens are conditioned on a long generated prefix that itself reflects choices the old policy made and the new policy now scores differently, so the drift compounds token by token. Inside a clipped objective the noisiest ratios are the ones most likely to flip the clip decision back and forth and to inject the largest gradient swings, which makes the late-token ratios the natural suspect.

I propose **First-K truncated IS**: keep the ordinary per-token PPO clipped surrogate for the first $K = 64$ response positions, and for every position $t \ge K$ replace the ratio with its detached value, so no gradient flows through the late-token log-probs via the IS weight. What "freeze the ratio" actually does to the gradient is the load-bearing point, and it is worth being precise about because it is easy to overclaim. In the unclipped term $-\hat A_{i,t}\, r_{i,t}$, detaching $r$ at a position turns it into $-\hat A_{i,t}\, \mathrm{sg}[r]$, a constant times nothing differentiable, whose gradient with respect to $\theta$ is exactly zero. The token is not down-weighted; it is *removed* from the policy gradient — exactly as a clip would remove it, except the removal is keyed on position ($t \ge K$) rather than on ratio magnitude. So the truncation is a hard position gate: the first $K$ tokens get the full per-token treatment — live ratio, per-token clip, pessimistic min — and everything past position $K$ is dead weight in the backward pass, present only so the forward loss still counts all tokens and keeps the loss scale honest.

I build the gate as a position mask rather than by slicing, because slicing would change tensor shapes and complicate the autograd graph. With $\text{positions} = \mathrm{arange}(T)$ and $\text{prefix\_mask} = (\text{positions} < K)$ broadcast over the batch, the effective ratio is

$$r^{\text{eff}}_{i,t} = r_{i,t}\cdot \text{prefix\_mask}_t + \mathrm{sg}[r_{i,t}]\cdot(1 - \text{prefix\_mask}_t),$$

live ratio inside the prefix and detached ratio outside. The forward value of $r^{\text{eff}}$ equals $r$ everywhere; only the backward graph differs, and autograd sees exactly one differentiable path — the first $K$ columns. From there it is the ordinary PPO machinery applied to $r^{\text{eff}}$: the unclipped loss $\text{pg\_losses1} = -\hat A\, r^{\text{eff}}$, the clipped loss $\text{pg\_losses2} = -\hat A\, \mathrm{clamp}(r^{\text{eff}}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$, and the per-token loss $\max(\text{pg\_losses1}, \text{pg\_losses2})$ — the max-over-losses that realizes the pessimistic min-over-surrogates. I read $\varepsilon$ from `config.clip_ratio`, honoring asymmetric `clip_ratio_low`/`clip_ratio_high` with fallback, never hardcoding it, and I clamp $\text{log\_prob} - \text{old\_log\_prob}$ to $[-20, 20]$ before `exp` so one pathological token cannot overflow the batch.

The choice of $K = 64$ is the natural first guess: long enough to cover the opening of a reasoning trace — the restatement of the problem, the first setup step, where the early high-signal tokens live — and short enough that the bulk of a multi-hundred-token MATH solution falls in the frozen region. I deliberately leave out the dual-clip floor that full token-level PPO carries: it guards against a real negative-advantage / large-ratio runaway, but for this crude first rung I want the *minimal* truncation variant, the smallest change from a plain clipped surrogate, so that any result is attributable to the truncation alone rather than to a second mechanism. The aggregation is `token-mean`, because the ratio and clip are still per-token; every valid token (including the frozen ones, which carry their forward loss value) is weighted equally.

I should be honest about the bias this introduces, because it is what decides whether this is a sensible floor or a self-inflicted wound. The unbiased off-policy policy gradient is $\mathbb{E}[\,r_{i,t}\,\hat A_{i,t}\,\nabla \log \pi_\theta\,]$ summed over all tokens; by detaching the ratio for $t \ge K$ I zero the gradient on those tokens, so my estimator is the gradient computed *as if only the first $K$ tokens existed*, scaled by a loss that still counts all of them. That bias grows with response length — a 300-token solution learns from only its first ~64 tokens, discarding roughly four-fifths of the per-token signal. The bet is that the discarded signal is mostly drift-noise and the early ratios are the trustworthy ones. I expect that bet to lose here, and to lose worst of any granularity choice, because the regime is the opposite of the one truncation was designed for: a small, dense, non-MoE 0.5B model on moderate-length MATH, where the per-token ratios are already well-behaved, so I would be paying the largest bias cost to defeat the smallest noise problem. That is precisely why this rung sits at the bottom of the ladder — its result tells me whether the late-token-noise story is even true in this regime.

The diagnostic I most want is the *shape* of the degradation across benchmarks. A 64-token prefix covers most of a short GSM8K answer, so freezing the rest should cost little there; it covers only a fraction of a long, multi-step MATH or AMC solution, so freezing the rest should hurt those most. If I see GSM8K roughly intact while MATH-500 and AMC sag, that confirms the truncation is biting the long-response benchmarks by discarding their late-token gradient — the cleanest possible refutation of the late-token-noise hypothesis, and the signal that the next rung should stop discarding tokens and instead change how the per-token ratios are *combined*, keeping every token in the gradient while making the combination less noisy. I also emit $\text{pg\_clipfrac} = \mathrm{masked\_mean}(\text{pg\_losses2} > \text{pg\_losses1})$ and the cheap k1 KL estimate $\text{ppo\_kl} = \mathrm{masked\_mean}(-(\text{log\_prob} - \text{old\_log\_prob}))$ over the response mask, so the next rung can diagnose *why* this one landed where it did.

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
