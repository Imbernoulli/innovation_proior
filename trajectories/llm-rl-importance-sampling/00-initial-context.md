## Research question

For LLM online RL, the importance-sampling granularity in the clipped policy-gradient loss is an open design axis. Everything else is fixed: the GRPO advantage estimator, the reward manager, the rollout setup, the KL configuration, the model, and the optimizer. The only variable is how the old-policy and current-policy log-probs are converted into ratios, clipped, and aggregated. The goal is higher math-reasoning accuracy and lower gradient variance. Concretely: given per-token `old_log_prob` and `log_prob` (and per-token GRPO advantages), what is the right granularity at which to form the ratio `r = exp(log_prob − old_log_prob)`, clip it, and reduce it to a scalar loss?

## Prior art / Background / Baselines

The clipped surrogate descends from a short lineage. Each ancestor leaves a concrete gap.

- **Conservative policy iteration / the off-policy surrogate.** Reusing a frozen rollout batch for several gradient steps is importance sampling: with `r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)`, the surrogate `E_t[r_t Â_t]` is an off-policy estimate of the new policy's expected advantage. Its observed limitation: as the policy drifts, the surrogate overestimates improvement because it is only a local model of the return.
- **Trust-region constraint (Schulman et al., TRPO, 2015).** Leashes drift with a hard KL constraint solved by Fisher-vector products and conjugate gradient. Its observed limitation: the second-order optimization is heavy and does not fit naturally into standard multi-epoch SGD loops.
- **Per-token clipped surrogate (Schulman et al., PPO, 2017; Shao et al., DeepSeekMath/GRPO, 2024).** Replaces the constraint with a per-token clip of `r_t` to `[1−ε, 1+ε]` and takes the pessimistic min of clipped and unclipped surrogate, giving a first-order, dropout-friendly update. Its observed limitation: the per-token ratio is a single noisy draw per next-token distribution, and accumulated per-token noise has been observed to destabilize training at scale.

## Fixed substrate / Code framework

The training loop is verl with the GRPO advantage estimator: for each prompt, sample a group of responses, score them with the reward manager, and set the per-token advantage to the group-normalized reward (shared across every token of a response). Policy is Qwen2.5-0.5B, full-parameter training, one H200 GPU, 100 PPO steps, 16 rollout samples per prompt, batch size 128. The training set is simpleRL-Zoo MATH level 3–5 (Qwen split). The advantage estimator, reward manager, model, rollout setup, optimizer, KL configuration, and evaluation are frozen.

The loop hands the policy-loss function four aligned `(bs, response_length)` tensors — `old_log_prob`, `log_prob`, `advantages`, `response_mask` — plus an `ActorConfig` and an optional `rollout_is_weights`. It expects back a scalar `pg_loss` and a metrics dict with at least `actor/pg_clipfrac` and `actor/ppo_kl` as Python floats. Helpers are `verl_F.masked_mean`, `verl_F.masked_whiten`, and `agg_loss` (which reduces a per-token loss matrix to a scalar under `loss_agg_mode`, forwarding `config.global_batch_info` kwargs).

## Editable interface

Exactly one function is editable — `compute_custom_policy_loss` in `verl/verl/trainer/ppo/custom_policy_loss.py`, registered under the `"custom"` policy-loss name. The contract:

- `assert config is not None` and read `config.clip_ratio` (do not hardcode ε); honor the optional asymmetric `clip_ratio_low` / `clip_ratio_high`, each falling back to `clip_ratio` if `None`.
- Clamp `log_prob − old_log_prob` to a safe range (e.g. `[−20, 20]`) before `exp` for numerical stability.
- Aggregate with `agg_loss(loss_mat=..., loss_mask=response_mask, loss_agg_mode=..., **config.global_batch_info)`. A token-level method uses `loss_agg_mode="token-mean"`; a sequence-aggregating method uses `"seq-mean-token-mean"`.
- Apply `rollout_is_weights` multiplicatively on the per-token losses if it is not `None`.
- Return `(pg_loss, metrics)` with `actor/pg_clipfrac` and `actor/ppo_kl` (and, if computed, `actor/pg_clipfrac_lower`) as Python floats.

The default fill is the unimplemented template:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Custom policy loss / importance-sampling strategy for verl PPO training."""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss

# =====================================================================
# EDITABLE: Implement your custom importance-sampling policy loss below.
# =====================================================================


@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, response_length) log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length) log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length) GRPO group-relative advantage
    response_mask: torch.Tensor,     # (bs, response_length) 1 on real response tokens
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective. The design axis is IS granularity:
    how r = exp(log_prob − old_log_prob) is formed, clipped, and aggregated."""
    raise NotImplementedError(
        "Implement your custom importance-sampling policy loss here. "
        "See core_algos.py (compute_policy_loss_vanilla / gspo) for reference."
    )
```

## Evaluation settings

Math-reasoning accuracy (`mean@1`) on GSM8K, MATH-500, and AMC 23; the primary score is the mean across the three. One seed (42). Each run is the full 100-step pipeline on one H200 (~18–21k seconds wall clock at this configuration). The leaderboard records, per run, a normalized `score_mean` (higher is better) alongside per-benchmark accuracies.
