# DAPO (and its length-aware reward correction), distilled

DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization — is a critic-free,
group-relative RL recipe for training LLMs to do long chain-of-thought reasoning against a
verifiable (+1/−1) reward. It takes the GRPO objective and makes four targeted repairs to the
failure modes a naive long-CoT run exhibits: **Clip-Higher** (decouple the clip range, raise
the upper bound) against entropy collapse; **Dynamic Sampling** (oversample, filter out
all-correct / all-wrong groups) against vanishing gradients; **Token-Level Policy Gradient
Loss** (aggregate per token, not per sample) against the response-length bias; and **Overlong
Reward Shaping** (soft length-aware penalty) against truncation noise. The reference-KL is
removed and the reward is a rule, not a learned model.

The **length-aware** correction is the through-line: the sample-level `1/|o_i|` loss
aggregation gives a token weight that shrinks with its response's length, which (sign by sign)
pressures correct answers to be brief and tolerates length in wrong answers, dragging average
length upward. DAPO fixes this in the loss (token-level aggregation) and at the truncation
boundary (soft overlong punishment). A reward-stage proxy for the same fix divides each
response's scalar reward by `√(response_length)` before the group-relative advantage is
formed — a halfway exponent between "do nothing" (linear verbosity pressure) and "divide by
length" (the full sequence-level averaging that over-corrects into brevity bias).

## Problem it solves

Critic-free, verifiable-reward RL for very long responses with a group of rollouts per
prompt, where the naive group-relative recipe (i) collapses the policy's entropy, (ii)
produces a growing fraction of zero-gradient (all-correct / all-wrong) groups, (iii) drifts
to ever-longer, lower-quality responses via a length bias in the loss, and (iv) injects noise
by penalizing sound-but-truncated responses.

## Objective

```
J_DAPO(θ) = E_{(q,a)~D, {o_i}_{i=1}^G ~ π_old(·|q)} [
              (1 / Σ_i |o_i|) Σ_i Σ_t  min( r_{i,t}(θ) Â_{i,t},
                                            clip(r_{i,t}(θ), 1-ε_low, 1+ε_high) Â_{i,t} ) ]
   s.t.  0 < |{ o_i : is_equivalent(a, o_i) }| < G,

   r_{i,t}(θ) = π_θ(o_{i,t}|q,o_{i,<t}) / π_old(o_{i,t}|q,o_{i,<t}),
   Â_{i,t}    = ( R_i − mean({R_j}_{j=1}^G) ) / std({R_j}_{j=1}^G).
```

Rule reward `R(ŷ,y) = +1 if is_equivalent(ŷ,y) else −1`; no `β·D_KL` term.

## The four techniques and why

1. **Clip-Higher.** Symmetric clip `ε=0.2` with `Â>0` lets a high-prob token rise to
   `π_old·1.2` (≈ uncapped at 0.9 → 1.08) but caps a low-prob token at `0.01 → 0.012`, so
   exploitation tokens grow freely while exploration tokens are throttled → entropy collapse.
   Decouple into `ε_low, ε_high` and raise only `ε_high` (e.g. 0.2 / 0.28): give low-prob
   tokens room to grow. `ε_low` is kept low because raising it would let probabilities be
   driven to 0, collapsing the sampling space from the other side.

2. **Dynamic Sampling.** If a group is all-correct (all `R_i=+1`) or all-wrong, then
   `R_i − mean(R) = 0` for every member → zero advantage → zero gradient; such groups grow
   over training, shrinking the effective batch and raising gradient variance. Oversample and
   keep only prompts with `0 < (#correct) < G` (the constraint above); resample until the
   batch is full. Cheap because generation is dominated by long-tail samples.

3. **Token-Level Policy Gradient Loss.** GRPO's sample-level aggregation
   `(1/G) Σ_i (1/|o_i|) Σ_t` gives token weight `(1/G)(1/|o_i|)`, shrinking with `|o_i|`.
   For `Â>0` (correct) the per-token push ∝ `1/|o_i|` → shorter correct responses pushed
   harder (brevity bias); for `Â<0` (incorrect) the per-token penalty ∝ `1/|o_i|` → longer
   wrong responses penalized less (verbosity tolerance). Net: average length drifts up,
   dragged by bloated incorrect responses. Fix: aggregate at the token level,
   `(1 / Σ_i |o_i|) Σ_i Σ_t`, so every token has weight `1/(total batch tokens)` regardless
   of host-sequence length — longer correct responses count more, every junk token in a long
   wrong response carries its own full penalty.

4. **Overlong Reward Shaping.** Truncating overlong samples and giving them the punitive
   reward penalizes sound-but-long reasoning purely for length → reward noise. First,
   **Overlong Filtering** masks the loss of truncated samples (stabilizes training). Then a
   **Soft Overlong Punishment** ramps a penalty linearly across a buffer below the hard cap:

```
                  ⎧ 0,                                    |y| ≤ L_max − L_cache
R_length(y) =     ⎨ ((L_max − L_cache) − |y|) / L_cache,  L_max − L_cache < |y| ≤ L_max
                  ⎩ −1,                                    |y| > L_max
```

   Continuous (0 at the buffer start, −1 at the cap), added to the rule reward. In the
   `verl` reward manager the measured valid length is already capped by generation, so the
   implementation's one-sided `min(raw_penalty, 0)` reaches `−penalty_factor` at the hard cap
   and cannot go below it. (Defaults: expected length `L_max − L_cache = 16,384`, buffer
   `L_cache = 4,096`, hard cap 20,480.)

## Relation to prior methods

- **PPO** (Schulman et al. 2017): same clipped surrogate, but with a learned value `V` for the
  advantage; DAPO drops the critic.
- **GRPO** (Shao et al. 2024): same group-relative critic-free advantage; DAPO removes the
  sample-level `1/|o_i|` aggregation, the reference-KL, and the all-correct/all-wrong dead
  groups, and adds asymmetric clipping and overlong shaping.
- **Length-bias analysis** (Liu et al. 2025, Dr. GRPO): the `1/|o_i|` factor is a
  response-level length bias and `/std(R)` a question-level difficulty bias; the unbiased
  advantage is `R_i − mean(R)` (= RLOO up to `G/(G−1)`, foldable into the learning rate). The
  token-level loss and the `√T` reward proxy below both lean against the same length bias.

## Length-aware reward-stage correction (the `√T` proxy)

A one-knob, upstream proxy for the length fix, applied to the per-response scalar reward
before the group-relative advantage is formed. With one scalar advantage broadcast across
`T` tokens, the raw per-response contribution scales like `r·T`. Write the multiplier as
`T^{-α}`: `α=0` leaves the linear length pressure, while `α=1` gives the full `1/T`
sequence-level averaging that creates the brevity/under-penalty sign bias. The proxy chooses
`α=1/2`, so the contribution scales like `r·T/√T = r·√T`: a partial damping, not an exact
unbiased-policy-gradient correction.

```python
import torch
import numpy as np
from typing import Optional


def normalize_rewards(
    token_level_scores: torch.Tensor,   # (bs, response_length): outcome scalar at last valid token
    response_mask: torch.Tensor,        # (bs, response_length): 1 on valid response tokens
    index: np.ndarray = None,           # (bs,): group / prompt id (unused here)
    epsilon: float = 1e-6,
    config: Optional[object] = None,
    **kwargs,
) -> torch.Tensor:                      # (bs, response_length): length-corrected reward tensor
    """length_aware: divide each response's scalar reward by sqrt(response_length)."""
    with torch.no_grad():
        bsz, seq_len = token_level_scores.shape
        scores = token_level_scores.sum(dim=-1)               # (bs,): recover per-response scalar

        lengths = response_mask.sum(dim=-1).to(scores.dtype)  # (bs,): T = # valid tokens
        denom = torch.sqrt(lengths.clamp(min=1.0)) + epsilon  # sqrt(T), floored against 0-division
        scores = scores / denom                               # pre-divide scalar by sqrt(T)

        out = torch.zeros_like(token_level_scores)
        last_idx = response_mask.long().sum(dim=-1) - 1       # index of last valid token
        last_idx = last_idx.clamp(min=0)
        out[torch.arange(bsz, device=out.device), last_idx] = scores   # re-place rescaled scalar
        return out * response_mask                            # keep last-token outcome semantics
```

## Canonical DAPO implementation (the in-loss length fix)

In a verl-style trainer the token-level loss is the `loss_agg_mode = "token-mean"` reduction,
and the soft overlong punishment is added to the reward.

```python
import torch


def masked_mean(values, mask, dim=None):
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)


def policy_loss(log_prob, old_log_prob, advantages, response_mask,
                clip_low=0.2, clip_high=0.28, loss_agg_mode="token-mean"):
    """DAPO clipped surrogate with decoupled clip (Clip-Higher) and token-level aggregation."""
    ratio = torch.exp(log_prob - old_log_prob)                     # r_{i,t}(theta)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
    per_token_loss = -torch.min(surr1, surr2)                      # maximize surrogate -> minimize -min

    if loss_agg_mode == "token-mean":                             # Token-Level Policy Gradient Loss
        # every token weighted 1 / (total valid tokens in batch), independent of |o_i|
        return masked_mean(per_token_loss, response_mask, dim=None)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq = torch.sum(per_token_loss * response_mask, dim=-1)
        return torch.mean(seq)
    elif loss_agg_mode == "seq-mean-token-mean":                  # GRPO-style sample-level (length-biased)
        seq = torch.sum(per_token_loss * response_mask, dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)
        return torch.mean(seq)


def soft_overlong_punishment(reward, valid_response_length,
                             max_response_length=20480, overlong_buffer_len=4096,
                             overlong_penalty_factor=1.0):
    """verl-style linear penalty across the soft buffer below the hard generation cap."""
    expected_len = max_response_length - overlong_buffer_len      # L_max - L_cache = 16384
    exceed_len = valid_response_length - expected_len             # how far into the buffer
    raw_penalty = -exceed_len / overlong_buffer_len * overlong_penalty_factor
    overlong_reward = torch.minimum(raw_penalty, torch.zeros_like(raw_penalty))
    # valid_response_length is computed after generation truncation, so the lowest reachable
    # value is -overlong_penalty_factor at max_response_length.
    return reward + overlong_reward
```

## Defaults

`G = 16` rollouts per prompt; `ε_low = 0.2`, `ε_high = 0.28`; rule reward `+1/−1`, no KL;
expected length 16,384, soft buffer 4,096, hard cap 20,480; advantage
`(R_i − mean(R))/std(R)` group-relative, broadcast over response tokens.
