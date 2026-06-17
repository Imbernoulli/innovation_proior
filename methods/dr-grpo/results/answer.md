# Dr. GRPO (GRPO Done Right), distilled

Dr. GRPO is a critic-free advantage estimator for LLM reasoning RL that takes GRPO and
removes its two unrequested normalization terms — the per-question standard-deviation scaling
of the advantage, and the per-response `1/|o_i|` length normalization in the loss. What remains
is the Monte-Carlo policy-gradient direction induced by a leave-one-out baseline, written in its
group-mean-centered scaled form: the advantage of every token of response `i` is its return
minus the group mean, `R_i − mean(R)`, broadcast over its tokens, with the loss aggregated by a
*constant* divisor instead of by response length.

## Problem it solves

In RL with verifiable rewards, GRPO induces an optimization bias that inflates response length
— especially for *incorrect* responses — so the model can spend more tokens without gaining
accuracy ("overthinking"), and it distorts how much each question contributes to the update.
Dr. GRPO removes both biases so the per-token gradient matches the intended PPO /
policy-gradient objective while keeping the critic-free setup.

## The two biases removed

GRPO sets the advantage of every token of response `i` to
`Â_{i,t} = (R_i − mean(R)) / std(R)` and weights the loss by `(1/G) Σ_i (1/|o_i|) Σ_t (…)`.
Both extra factors are reweightings of the centered advantage `Ã_{i,t} = R_i − mean(R)`:

- **Response-level length bias** (from `1/|o_i|`). Per-token gradient ∝ `Â/|o_i|`. For a
  *correct* response (`Â>0`) shorter ones get a larger push (favor brevity); for an
  *incorrect* response (`Â<0`) longer ones are penalized *less* per token (favor verbosity).
  The wrong-answer branch has no correctness ceiling, so incorrect responses grow without
  bound. The PPO surrogate is a bare token *sum* — there is no `1/|o|` in it; the divisor is an
  artifact of `masked_mean(…, dim=-1)` (a pretraining habit of dividing by a *constant* context
  length, miscarried to RL where response length is not constant).
- **Question-level difficulty bias** (from `std(R)`). Advantage normalization is standard, but
  scoped *per group* it divides each question by its own `std`, changing questions' relative
  weights. Near-unanimous groups (almost all-correct or all-wrong, small `std`) get up-weighted
  vs. mixed-outcome questions. Exactly unanimous groups have zero centered signal, but the
  `1/std` scale is singular and has to be patched with epsilon. Scoped per *batch*,
  normalization would only rescale the global step.

## Key idea / derivation

Maximize `J(π_θ) = E_q E_{o~π_θ}[R(q,o)]` (KL dropped, `β=0`, since a rule-based verifier has
no reward-model distribution to stay near). The Monte-Carlo policy gradient is

```
∇_θ J = E[ Σ_t ∇_θ log π_θ(o_t|q,o_<t) ( Σ_{t'≥t} r(q,o_≤t') − B(q,o_<t) ) ],
```

using causality (a token only affects rewards at/after `t`) and any action-independent
**baseline** `B`, which leaves the gradient unbiased because
`E_{o_t}[∇_θ log π_θ(o_t|·) B] = B ∇_θ Σ_{o_t} π_θ(o_t|·) = B ∇_θ 1 = 0`.

With **outcome reward** (nonzero only at the last token), the reward-to-go from any `t` equals
the whole return: `Σ_{t'≥t} r = R(q,o)`. So the advantage is one scalar per response, broadcast
over its tokens. The cheap group statistic is the centered return
`Ã_{i,t} = R_i − mean({R_1,…,R_G})`. Neither `std` nor `1/|o|` appears; the unbiasedness check is
the RLOO identity below.

**Unbiasedness (equivalence to RLOO).** Scaling by `G/(G−1)`:

```
(G/(G−1)) (R_i − (1/G) Σ_j R_j) = (G/(G−1)) R_i − (1/(G−1)) Σ_j R_j
                                = R_i − (1/(G−1)) Σ_{j≠i} R_j
                                = R_i − mean_{j≠i}(R_j),
```

which is the REINFORCE leave-one-out advantage. Thus `R_i − mean(R)` is `((G−1)/G)` times the
unbiased leave-one-out advantage, and that global factor folds into the learning rate.

## Final estimator

For each question group: sum per-token rewards to per-response returns `R_i`; subtract the
group mean; broadcast over valid tokens. Loss: PPO clipped surrogate `min[ρ Ã, clip(ρ,1−ε,1+ε)
Ã]`, with the token-sum divided by a **constant** (generation budget) rather than `|o_i|`.

```python
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,         # (bs, response_length); 1 = valid response token
    index: np.ndarray = None,            # (bs,) group id; same id == same question
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-only group-relative advantages for outcome rewards."""
    scores = token_level_rewards.sum(dim=-1)                  # (bs,) per-sequence return R_i

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)              # single sample: no group baseline
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)      # B = mean(R)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]         # Ã_i = R_i - mean(R); NO / std
        scores = scores.unsqueeze(-1) * response_mask         # broadcast over tokens, mask

    return scores, scores                                     # advantage == return (outcome reward)
```

Equivalently, in verl the built-in `compute_grpo_outcome_advantage(..., norm_adv_by_std_in_grpo=False)`
is Dr. GRPO: the `True` branch divides by `(std + epsilon)`, the `False` branch subtracts the
mean only.

## Loss aggregation (the length-bias fix)

```python
def masked_loss_aggregate(per_token_loss, response_mask, max_tokens):
    # GRPO / typical PPO impl: divide each response by its own length -> length bias
    #   return (per_token_loss * response_mask).sum(-1) / response_mask.sum(-1)
    # Dr. GRPO: divide by a constant (generation budget) -> no per-response length factor
    return (per_token_loss * response_mask).sum(-1) / max_tokens
```

## Settings (as used for this regime)

Base policy RL-tuned with a rule-based verifier reward (1 if correct final answer else 0),
`β=0` (no KL), PPO clip `ε=0.2`, `G` responses per prompt (8–16), AdamW `(0.9, 0.95)`, constant
LR ~`1e-6`, grad-norm clip `1.0`, one inner epoch, max response ~3000 tokens, temperature `1.0`.
Evaluation by `mean@1` accuracy on held-out math benchmarks (GSM8K, MATH-500, AMC), with
response-length-by-correctness tracked as the diagnostic for the length pathology.
