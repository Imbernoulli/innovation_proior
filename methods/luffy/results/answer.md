# LUFFY (Learning to reason Under oFF-policY guidance), distilled

LUFFY augments on-policy RLVR (GRPO) with off-policy reasoning traces from a stronger model and balances
imitation against exploration *automatically*, in a single training stage. Two pieces: **Mixed-Policy
GRPO**, which adds the off-policy trace as a member of the rollout group and standardizes the advantage
over the union of on-policy and off-policy rollouts; and **policy shaping via regularized importance
sampling**, which reweights the off-policy gradient toward low-probability ("surprising") teacher tokens
so the model learns the new reasoning instead of collapsing onto the parts it already agrees with.

## Problem it solves

On-policy RL is bounded by the base model: on a prompt every rollout fails, the group advantage
`(R − mean)/std` degenerates to zero, and there is no gradient — so RLVR sharpens Pass@1 but does not
expand the capability boundary (Pass@k). LUFFY uses an off-the-shelf teacher trace to bootstrap exactly
those dead-gradient prompts, while keeping RL's exploration (which plain SFT on the teacher would kill).

## Mixed-Policy GRPO

Add the off-policy trace to the group; standardize the advantage over the union `G_on ∪ G_off`:

```
Â_i = ( R(τ_i) − mean(G_on ∪ G_off) ) / std(G_on ∪ G_off),
G_on = {R(τ_i) : τ_i ~ π_{θ_old}},   G_off = {R(τ_j) : τ_j ~ π_φ}.
```

When the model's own rollouts all fail (0) and the teacher is correct (1), the union has spread and the
teacher gets a large positive advantage — bootstrapping the prompt. When the model already solves the
prompt, its own rollouts take precedence and the signal stays self-driven. The mixing self-adjusts.

The mixed objective sums two clipped surrogates over the union (normalized by total tokens `Z`):

```
J_Mixed(θ) = (1/Z) [ Σ_{j∈off} Σ_t CLIP(r̂_{j,t}(θ,φ), Â_j, ε)  +  Σ_{i∈on} Σ_t CLIP(r_{i,t}(θ), Â_i, ε) ],
r_{i,t}   = π_θ(τ_{i,t}|·) / π_{θ_old}(τ_{i,t}|·)        # on-policy ratio
r̂_{j,t}  = π_θ(τ_{j,t}|·) / π_φ(τ_{j,t}|·)              # off-policy ratio (teacher denominator)
```

**Setting `π_φ = 1`.** To avoid the teacher's per-token densities and tokenizer matching, LUFFY adopts
`π_φ = 1`, so `r̂_{j,t} = π_θ(τ_{j,t}|·)` (the model's own probability of the teacher token). The clip is
**omitted on the off-policy branch** (a `[1−ε,1+ε]` clip is ill-posed for a raw probability). This is a
biased choice traded for tokenizer-independence and off-the-shelf data; the convergence guarantee still
holds for any well-defined behavior policy (`π_φ = 1` is the constant special case).

## Policy shaping via regularized importance sampling

The bare off-policy term `Σ_t π_θ(τ_{j,t}) Â_j` "hacks" the objective: its gradient scales with `π_θ`, so
it pushes teacher tokens the model already agrees with (large `π_θ`) and ignores the low-probability
surprising tokens that carry the new reasoning — entropy collapses. Replace the off-policy ratio with a
shaping transform `f(r̂)`:

```
J_SHAPING(θ) = (1/Z) [ Σ_{j∈off} Σ_t f(r̂_{j,t}(θ,φ)) Â_j  +  Σ_{i∈on} Σ_t CLIP(r_{i,t}(θ), Â_i, ε) ],
f(x) = x / (x + γ),    γ = 0.1   (swept over {0.01, 0.1, 0.2}).
```

`f` is increasing and concave; `f'(x) = γ/(x+γ)²` is **largest at `x → 0`**, so the gradient on a
low-probability teacher token is amplified (relative to the identity, by ≈`1/γ`) and damped on
already-likely tokens. From the off-policy gradient
`∇J_SHAPING-OFF = E_{τ~π_φ}[ f'(π_θ) (π_θ/π_φ) ∇log π_θ · Â_j ]`, the per-logit identity-case scale is
bounded by `f'(π_θ)·π_θ(1−π_θ)`; vanilla mixed-policy is the linear `f(π)=π` (`f'=1`), whose scale
`π(1−π)` vanishes at small `π`. Shaping with `f' = γ/(π+γ)²` restores gradient to the surprising tokens
and (informally) reduces the importance-weight variance, stabilizing training.

## Why these choices

- **Union-group advantage, not a separate SFT loss:** lets the teacher's contribution be governed by the
  model's current competence on the prompt (high advantage when stuck, low when competent) — adaptive,
  not a fixed coefficient.
- **`π_φ = 1` + no off-policy clip:** removes the dependence on the teacher's densities/tokenizer and the
  ill-posed clip; biased but practical and theory-compatible.
- **Shaping `f(x)=x/(x+γ)`:** the minimal fix for the entropy-collapse "hacking" — it emphasizes exactly
  the low-probability crucial actions standard policy gradients ignore.
- **RL practice:** `β = 0` (no KL), entropy coef 0.01, Dr.GRPO normalization (remove length and std-error
  normalization), 1 off-policy + 7 on-policy rollouts per prompt, temperature 1.0, lr 1e-6 AdamW.

## Results (Qwen2.5-Math-7B, in-distribution Pass@1; AIME/AMC avg@32)

| Method | AIME24/25 | AMC | MATH-500 | Avg (6 bench) |
|---|---|---|---|---|
| SFT | 22.2/22.3 | 52.8 | 82.6 | 44.1 |
| SFT+RL | 25.8/23.1 | 62.7 | 87.2 | 48.2 |
| On-Policy RL (GRPO/Dr.GRPO) | 25.1/15.3 | 62.0 | 84.4 | 45.5 |
| **LUFFY** | **29.4/23.1** | **65.6** | **87.6** | **50.1** |

LUFFY beats on-policy GRPO on AIME24, AMC, and MATH-500 (avg +4.6), beats all off-policy alternatives
(SFT, RL-w/-SFT-loss, SFT+RL), and gains +6.2 on out-of-distribution (ARC-c, GPQA-diamond, MMLU-Pro). It
also extends to Qwen2.5-Math-1.5B, Qwen2.5-Instruct-7B, and LLaMA-3.1-8B where on-policy RLVR is weaker.

## Working code

Paper-style core (a GRPO trainer): one function forms the union-group advantage (Eq. 4), one forms the
mixed loss — clipped on-policy surrogate + unclipped shaped off-policy term with `π_φ = 1` (Eq. 5/6).

```python
from collections import defaultdict

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_mixed_group_advantage(rewards, index, epsilon=1e-6):
    """Eq. 4: standardize advantage over the UNION of on-policy and off-policy rollouts."""
    id2rewards = defaultdict(list)
    for i in range(rewards.shape[0]):
        id2rewards[int(index[i])].append(rewards[i])
    id2mean, id2std = {}, {}
    for uid, vals in id2rewards.items():
        g = torch.stack(vals)
        id2mean[uid] = g.mean()
        id2std[uid] = g.std() if g.numel() > 1 else torch.tensor(1.0, device=rewards.device)
    advantages = torch.zeros_like(rewards)
    for i in range(rewards.shape[0]):
        uid = int(index[i])
        advantages[i] = (rewards[i] - id2mean[uid]) / (id2std[uid] + epsilon)
    return advantages


def shaping_fn(x, gamma=0.1):
    """f(x) = x/(x+gamma); f'(x)=gamma/(x+gamma)^2 is largest at x->0 (emphasize low-prob tokens)."""
    return x / (x + gamma)


def mixed_policy_loss(log_prob, old_log_prob, advantages, response_mask, is_off_policy,
                      clip_ratio=0.2, gamma=0.1):
    """Eq. 5/6: clipped on-policy surrogate + unclipped shaped off-policy term (pi_phi = 1)."""
    adv_t = advantages.unsqueeze(-1)
    on_mask = ~is_off_policy
    ratio_on = torch.exp(log_prob - old_log_prob)
    surr1 = ratio_on * adv_t
    surr2 = torch.clamp(ratio_on, 1 - clip_ratio, 1 + clip_ratio) * adv_t
    on_term = torch.min(surr1, surr2)
    pi_theta = torch.exp(log_prob)                         # pi_phi = 1 -> off-policy ratio = pi_theta
    off_term = shaping_fn(pi_theta, gamma) * adv_t         # unclipped, shaped
    per_token = torch.where(on_mask.unsqueeze(-1), on_term, off_term)
    return -masked_mean(per_token, response_mask)
```

## Practical defaults

`γ = 0.1`, clip `ε = 0.2` (on-policy only), `β = 0` (no KL), entropy coef 0.01, Dr.GRPO normalization,
8 rollouts/prompt (1 off-policy + 7 on-policy), temperature 1.0, lr 1e-6 AdamW, single pass over a 45k
OpenR1-Math-220k subset with DeepSeek-R1 off-policy traces (filtered ≤8192 tokens, verified correct).
