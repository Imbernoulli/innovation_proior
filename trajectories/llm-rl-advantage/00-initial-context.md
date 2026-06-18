## Research question

Online RL fine-tuning of a small LLM on math, with a *verifiable* reward: sample a group of
responses per problem, grade each by whether its final answer is correct, and push the policy toward
what earned reward. The one thing being designed is the **advantage estimator** — the function that
turns per-token rewards, response masks, and group identifiers into the per-token advantages (and
returns) the PPO/GRPO actor loss then consumes. Everything else about the agent — the policy, the
reward manager, the rollout config, the optimizer, the clipped actor loss, the KL-loss setting — is
fixed. The goal is sample efficiency and policy-learning stability for math reasoning, measured by
downstream accuracy.

## Prior art before the first rung (policy-optimization lineage)

The estimator the first rung uses is the resolution of a line of critic-free policy-gradient methods.
These precede the ladder; the fixed substrate below is what they react to.

- **Vanilla policy gradients (REINFORCE, Williams 1992).** Ascend
  `g = E[ R(o,q) ∇_θ log π_θ(o|q) ]` directly. General and simple, but the raw return is a
  high-variance, often all-positive multiplier — every response's log-prob gets pushed up and the
  useful signal drowns. Gap: usable only with a variance-reducing baseline.
- **Actor-critic / PPO with GAE (Schulman et al. 2015, 2017).** Subtract a learned state value
  `V_ψ(s)` and build per-token advantages with GAE,
  `Â_t = Σ_l (γλ)^l δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`; update with the clipped surrogate
  `min(ρ_t Â_t, clip(ρ_t,1−ε,1+ε) Â_t)`. Reliable, but `V_ψ` is a second network the size of the
  policy, and in the LLM setting the reward arrives only at the last token, so an accurate *per-token*
  value — exactly what GAE needs at every step — is precisely the hardest thing to fit. Gap: the
  critic is both a heavy cost and the unreliable part.
- **RLOO (Ahmadian et al., ICLR 2024).** Drop the critic; for each of `k` sampled responses use the
  mean reward of the *other* `k−1` as a leave-one-out baseline,
  `A_i = r_i − (1/(k−1)) Σ_{j≠i} r_j`. Exactly action-independent, hence unbiased — but the baseline
  is estimated from a handful of samples, and the advantages of different prompts are never put on a
  common scale. Gap: noisy small-group baseline, no cross-prompt normalization.

## The fixed substrate

A verl PPO/GRPO training loop is frozen and must not be touched: **Qwen2.5-0.5B** trained full-
parameter on **simpleRL-Zoo MATH level 3–5** (~8K problems); **100 steps**, **16 rollout samples per
prompt**, **batch size 128**, one H200 per experiment; a rule-based verifier emitting the outcome
reward at the last valid token; the clipped PPO actor loss, the optimizer, and the KL-loss setting all
fixed. The loop calls the advantage estimator through verl's registry and then runs its own actor
update on the `(advantages, returns)` the estimator returns. The estimator may use the masked helpers
the loop provides — `verl_F.masked_whiten(values, mask)`, `verl_F.masked_mean(values, mask)` — plus
`defaultdict`, `torch`, `numpy`.

## The editable interface

Exactly one region is editable — the body of `compute_custom_advantage()` in
`verl/verl/trainer/ppo/custom_advantage.py`, lines 17–72 of the template, registered under the name
`"custom"`. Every method on the ladder is a fill of this same contract. The function receives
`token_level_rewards (bs, response_length)` (outcome reward at the last valid token; `.sum(dim=-1)`
recovers the per-sequence score), `response_mask (bs, response_length)` (1 on valid response tokens),
`index (bs,)` (the group/prompt id — 16 responses share an id), `epsilon`, an `AlgoConfig`, and
optionally per-token `old_log_probs` / `ref_log_probs`. It must return `(advantages, returns)`, both
`(bs, response_length)` and masked by `response_mask`; the computation runs under `torch.no_grad()`.
For outcome-level estimators the per-sequence advantage is broadcast to all tokens. **Only this
function is editable — the actor loss, its clipping, and its token-aggregation are fixed and live
outside the edit surface**, so an estimator cannot change how the loss reduces tokens; it can only
shape the advantages.

The starting point is the scaffold default: a `NotImplementedError`. Each method replaces exactly this
function body and nothing else.

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

Math-reasoning accuracy (`mean@1`, higher is better) after the fixed 100-step run, on three held-out
benchmarks spanning the difficulty range — **GSM8K** (1,319 grade-school problems), **MATH-500** (a
500-problem competition subset), and **AMC 23** (an AMC 2022–2023 competition subset). The leaderboard
also tracks a hidden AIME-2024 split and a `gsm8k_reward_mean`. The primary score is a single
normalized number aggregating the benchmark accuracies (the leaderboard `score_mean` column). One seed
(42).
