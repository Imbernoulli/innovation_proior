**Problem (from step 2).** The HPT switch won AMC23 (0.325) but left MATH-500 flat (0.250) and AIME24
pinned at the pure-SFT value (0.062): it still routes every all-wrong prompt to SFT and pays the residual
imitation tax on the broad split. To isolate that tax, pay it zero times — never imitate.

**Key idea (pure GRPO).** Keep every prompt's on-policy GRPO group and never switch — the demonstration
`τ★` is never touched. GRPO is the right on-policy learner: it replaces PPO's policy-sized value network
(impossible to fit per token under a last-token-only verifier reward) with a *free* baseline — the mean
verifier reward of the prompt's own rollout group — z-scored per question and broadcast to every token:
`Â_{i,t} = (R_i − mean(R)) / (std(R) + ε)`. Optimized with PPO's clipped surrogate
`min(ρ_{i,t} Â, clip(ρ_{i,t}, 1−ε, 1+ε) Â)`, with the KL-to-reference moved into the loss (k3 estimator).

**Why it works.** The group mean is a Monte-Carlo state value at the only granularity the reward exists,
matching the comparative nature of verifier scores; the std-normalization puts easy and hard prompts on a
comparable footing. No critic, no imitation. Every contrast prompt learns from its own graded, signed
rollouts instead of copying a teacher — no narrowing, full practice of its own reasoning. The cost: an
all-wrong group has `std = 0`, so its centered advantage is zero and it contributes no gradient — pure RL
cannot bootstrap a capability the model never samples.

**Scaffold edit.** Replace the controller with the unconditional `(0, 0, 0)`: never remove on-policy
rollouts, never add on-policy rollouts, never add off-policy samples.

**What to watch.** MATH-500 should lift substantially above 0.250 (imitation tax removed); AIME24 should
stay near 0.062 (no bootstrap); AMC23 risks falling back from 0.325 (the small split's lost bootstrap). A
clean dissociation: broad-split gain bought at the price of small-split imitation.

```python
# EDITABLE region of mix_trainer.py — step 3: pure GRPO (never switch)
    def select_on_off_ada_balance(self, on_solve_num: int):
        del on_solve_num
        return 0, 0, 0
```
