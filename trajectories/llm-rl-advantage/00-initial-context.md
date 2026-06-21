## Research question

Online RL fine-tuning of a small LLM on math with a *verifiable* outcome reward: sample a group of responses per problem, grade each by whether its final answer is correct, and push the policy toward higher reward. The only design choice left open is the **advantage estimator** — the function that turns per-token rewards, response masks, and group identifiers into per-token advantages (and returns) for the PPO/GRPO actor loss. The policy, reward manager, rollout config, optimizer, clipped actor loss, and KL-loss setting are fixed. Success is sample efficiency and stable learning, measured by downstream math accuracy.

## Prior art / Background / Baselines

These are the standard policy-gradient estimators already in use.

- **REINFORCE / vanilla policy gradients.** Ascend the expected return using the raw trajectory reward as a log-probability multiplier.

- **PPO with GAE.** Subtract a learned state-value estimate and build per-token advantages with generalized advantage estimation, then update with the clipped surrogate objective. It is reliable in many domains; the critic is a full-size network and the sparse end-of-response reward is concentrated at EOS.

- **RLOO.** Drop the critic and use the mean reward of the other responses in the same group as a leave-one-out baseline for each sample. Because the baseline does not depend on the sampled action it is unbiased.

## Fixed substrate / Code framework

The training loop is frozen and must not be touched: **Qwen2.5-0.5B**, full-parameter training on **simpleRL-Zoo MATH level 3–5** (~8K problems), **100 steps**, **16 rollout samples per prompt**, batch size **128**, one H200 per experiment. A rule-based verifier emits the outcome reward at the last valid token. The clipped PPO actor loss, optimizer, and KL-loss setting are fixed. The loop calls the advantage estimator through verl's registry and applies its own actor update to the `(advantages, returns)` returned. The estimator may use `verl_F.masked_whiten`, `verl_F.masked_mean`, `defaultdict`, `torch`, and `numpy`.

## Editable interface

Only the body of `compute_custom_advantage()` in `verl/verl/trainer/ppo/custom_advantage.py` is editable; the function is registered under the name `"custom"`. The contract is:

- Inputs: `token_level_rewards (bs, response_length)` (outcome reward at the last valid token; `.sum(dim=-1)` recovers the per-sequence score), `response_mask (bs, response_length)`, `index (bs,)` (group/prompt id; 16 responses share an id), `epsilon`, an `AlgoConfig`, and optionally `old_log_probs` / `ref_log_probs`.
- Outputs: `(advantages, returns)`, both `(bs, response_length)` and masked by `response_mask`; computation runs under `torch.no_grad()`.
- For outcome-level estimators, the per-sequence advantage is broadcast to all valid tokens.

The actor loss, its clipping, and token aggregation are fixed and live outside the edit surface, so an estimator can only shape the advantages.

The default fill is the scaffold below. Replace only the function body.

```python
# EDITABLE region of custom_advantage.py (lines 17-72) — default fill (not implemented)
# =====================================================================
# EDITABLE: Implement your custom advantage estimator below.
# =====================================================================


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,         # (bs, response_length)
    index: np.ndarray = None,            # (bs,) group/prompt id
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    ref_log_probs: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token advantages and returns for the PPO/GRPO actor loss."""
    raise NotImplementedError(
        "Implement your custom advantage estimator here. "
        "See core_algos.py for reference implementations (GRPO, RLOO, REINFORCE++, etc.)."
    )
```

## Evaluation settings

Math-reasoning accuracy (`mean@1`, higher is better) after the fixed 100-step run on three held-out benchmarks: **GSM8K** (1,319 grade-school problems), **MATH-500** (500 competition problems), and **AMC 23** (AMC 2022–2023 subset). The leaderboard also tracks a hidden AIME-2024 split and a `gsm8k_reward_mean`. The primary score is the leaderboard `score_mean` column, a normalized aggregate of the benchmark accuracies. One seed (42).
