# GSPO (Group Sequence Policy Optimization), distilled

GSPO is a policy-optimization objective for LLM online RL that performs importance sampling,
clipping, and optimization at the **sequence level** rather than the token level. It replaces
GRPO's per-token importance ratio with a single, length-normalized importance ratio per
response, clips whole responses, and aggregates the loss per sequence to reduce gradient
variance and avoid dependence on token-level routing stabilization.

## Problem it solves

PPO/GRPO-style LLM RL becomes unstable at scale (large models, MoE, long responses):
high-variance gradient noise accumulates with response length, is amplified by clipping, and
escalates into irreversible model collapse that retuning, checkpoint reverts, longer
generation, or new queries cannot recover.

## Key idea

Importance sampling, `E_{π_tar}[f] = E_{π_beh}[(π_tar/π_beh) f]`, realizes its
distribution-correction role through averaging samples from `π_beh`. GRPO's per-token weight
`w_{i,t} = π_θ(y_{i,t}|·)/π_old(y_{i,t}|·)` is built from a *single* sample per next-token
distribution. As a one-sample term it is formally an IS estimator, but inside the clipped GRPO
objective it has no local averaging mechanism and behaves as a high-variance per-token
gradient weight that compounds over `|y_i|` tokens.

The fix: the **unit of optimization should match the unit of reward**. The reward is granted
to the whole response, so do importance sampling per sequence. The sequence ratio
`π_θ(y_i|x)/π_old(y_i|x)` is a coherent measure of how off-policy the response is. To make it
usable, **length-normalize** it into the per-token geometric mean, which reduces variance and
unifies the numerical range across response lengths so one clip band works:

```
s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) )^{1/|y_i|}
       = exp( (1/|y_i|) Σ_t log( π_θ(y_{i,t}|x,y_{i,<t}) / π_old(y_{i,t}|x,y_{i,<t}) ) ).
```

## Objective

With group-relative advantage `Â_i = (r(x,y_i) − mean_j r(x,y_j)) / std_j r(x,y_j)`:

```
J_GSPO(θ) = E_{x~D, {y_i}~π_old}[ (1/G) Σ_i min( s_i(θ) Â_i, clip(s_i(θ), 1−ε, 1+ε) Â_i ) ].
```

Clipping is applied to entire responses. Because `s_i` is the exponential of an *average*
per-token log-ratio it sits very close to 1, so the clip band `ε` is far narrower than GRPO's:
left/right `3e-4 / 4e-4` for the sequence ratio versus `0.2 / 0.27` for GRPO's token ratio.
Asymmetric (decoupled) low/high edges are allowed.

## Gradient analysis

Dropping clipping, `∇_θ s_i = s_i ∇_θ log s_i` and
`∇_θ log s_i = (1/|y_i|) Σ_t ∇_θ log π_θ(y_{i,t}|·)`, so

```
∇_θ J_GSPO = E[ (1/G) Σ_i s_i(θ) Â_i · (1/|y_i|) Σ_t ∇_θ log π_θ(y_{i,t}|·) ],
∇_θ J_GRPO = E[ (1/G) Σ_i Â_i · (1/|y_i|) Σ_t w_{i,t}(θ) ∇_θ log π_θ(y_{i,t}|·) ].
```

The *only* difference is the weight on each token's score gradient: GRPO weights tokens
unequally by their individual noisy `w_{i,t}` (active-gradient range `(0,1+ε]` for `Â_i>0`,
`[1−ε,∞)` for `Â_i<0`); GSPO weights every token of a response equally by the single `s_i`.
Equal weighting removes the per-token weight disparity identified in GRPO.

## Why it stabilizes MoE training

For MoE models, ~10% of activated experts can change after one gradient update (48-layer
Qwen3-30B-A3B), making per-token `w_{i,t}` swing drastically. The routing replay workaround
caches/replays old routing so numerator and denominator share a sub-network. `s_i` uses the
length-normalized aggregate sequence likelihood rather than exposing any individual token
ratio as a gradient weight, so the loss no longer needs routing replay to make token-level
ratios stable.

## Infrastructure consequence

Token-level ratios are sensitive to precision discrepancies between rollout/inference engines
and training engines, so PPO/GRPO-style stacks often recompute old-policy likelihoods in the
training engine. The sequence ratio averages token log-ratio discrepancies before exponentiation,
making the loss more tolerant of using rollout-side likelihoods directly in partial-rollout,
multi-turn, or disaggregated training setups.

## GSPO-token (the implemented form)

For per-token advantage flexibility (e.g. multi-turn), define a token-level surrogate ratio
with stop-gradient `sg[·]` (PyTorch `detach`):

```
s_{i,t}(θ) = sg[ s_i(θ) ] · π_θ(y_{i,t}|·) / sg[ π_θ(y_{i,t}|·) ].
```

Numerically `s_{i,t} = s_i` (second factor = 1), so the forward value is the sequence ratio
broadcast to every token; the gradient is carried by the un-detached `log π_θ`, giving
`∇J = E[(1/G) Σ_i s_i · (1/|y_i|) Σ_t Â_{i,t} ∇log π_θ(y_{i,t}|·)]`, identical to GSPO when
`Â_{i,t} = Â_i`. In log space:
`log s_{i,t} = sg[log s_i] + log π_θ − sg[log π_θ]`.

## Working code

verl policy-loss function. The forward value of `seq_importance_ratio` is the
length-normalized sequence ratio broadcast to every token; the backward path runs through the
per-token `log_prob`. The loss is aggregated **seq-mean-token-mean** to mirror the objective's
`(1/G)(1/|y_i|)` two-level average (a flat token-mean would let long responses dominate and
undo the length normalization). The public verl function accepts `loss_agg_mode` in its
signature but passes `"seq-mean-token-mean"` to `agg_loss` inside the GSPO loss.

```python
from typing import Any, Optional

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,      # (bs, response_length): log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length): log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length): group-relative advantage
    response_mask: torch.Tensor,     # (bs, response_length): 1 on real response tokens
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for GSPO."""
    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # per-token log-ratio  log π_θ − log π_old  (the negative approximate KL)
    negative_approx_kl = log_prob - old_log_prob

    # length-normalized sequence log-ratio: (1/|y_i|) Σ_t (log π_θ − log π_old), masked mean
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # straight-through: forward value is the broadcast sequence ratio s_i;
    # backward path is the per-token log π_θ, so each token's score gradient is weighted by s_i.
    #   log s_{i,t} = sg[log s_i] + log π_θ − sg[log π_θ]
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)       # max of negatives = min of surrogates

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # (1/G) Σ_i (1/|y_i|) Σ_t [...]  → per-seq token-mean, then mean over sequences
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    )

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics
```

## Relation to prior methods

- **PPO** (Schulman et al., 2017): same clipped-surrogate shape, but PPO uses a per-token ratio
  and a learned critic; GSPO uses the sequence ratio and group-relative advantages.
- **GRPO** (Shao et al., 2024): identical advantage estimator; GRPO clips per token with the
  per-token ratio `w_{i,t}`, GSPO clips per sequence with the length-normalized `s_i`. The
  gradient differs only in the per-token weighting (unequal `w_{i,t}` vs. equal `s_i`).
- **Sequence likelihood** as an optimization target follows Zheng et al. (2023, "Click"); the
  `1/|y_i|` length normalization is GSPO's own choice for variance and range control.
