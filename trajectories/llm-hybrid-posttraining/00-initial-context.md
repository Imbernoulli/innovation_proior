## Research question

A fixed hybrid post-training stack — `Unify-Post-Training`'s HPT scaffold — already pairs each math prompt with two signals: the model's own verifier-scored on-policy rollouts (for RL) and a teacher demonstration `τ★` (for SFT). The advantage estimator (`grpo`), the unify strategy (`switch`), and the off-policy loss type (`sft`) are frozen. The single thing being designed is the **per-question router** — the trainer-side rule that decides, for each prompt, whether to keep its on-policy RL samples, drop them, regenerate more, or replace them with off-policy SFT (or off-policy RL) samples. Everything else about the stack is fixed.

## Prior art / Background / Baselines

- **Behavior cloning / SFT.** Fit the policy to a demonstration set by token-level NLL. Cheap, stable, and the only thing that reliably raises competence on prompts the model cannot solve at all. Gap: it learns only to copy demonstrations, injects no signal about which of the model's own behaviors are good, and tends to overfit and degrade out-of-distribution.
- **GRPO.** Critic-free on-policy RL: per prompt, sample a group of rollouts, score them with the verifier, and standardize within-group rewards into advantages. Excellent at sharpening reasoning the model can already partly do, and cheap because it needs no value network. Gap: purely on-policy, so it is bounded by what the base model can sample; on a hard prompt where every rollout gets the same score, the within-group advantage collapses to zero and GRPO contributes no gradient.
- **Fixed SFT+RL blends and SFT→RL pipelines.** Combine SFT and RL via hand-tuned coefficients, schedules, or fixed multi-stage pipelines. Gap: the combination is committed in advance and ignores how the model's competence varies across prompts and over training; the knobs must be retuned per model and dataset.

## Fixed substrate / Code framework

The HPT training stack is frozen and read-only. `algorithm.adv_estimator=grpo` (group-relative, critic-free advantages over a prompt's rollouts), `trainer.unify_strategy=switch` (hard per-question routing, not a soft blend), and `actor.offline_loss_type=sft` (the actor consumes off-policy SFT samples as token-NLL with a `sft_loss_coef`, and on-policy samples as the GRPO clipped surrogate). The rollout side (`mix_vllm_rollout.py`, `mix_hf_rollout.py`) defines how on-policy and off-policy sequences are built; `rl_dataset_with_target.py` supplies each prompt's demonstration target `τ★`. The advantage code group-normalizes over on-policy samples only. The loop draws `rollout.n_verify=8` on-policy rollouts per prompt, computes the per-question solve count `on_solve_num ∈ {0,…,8}` (how many of the eight passed the verifier), and hands it to the router.

## Editable interface

Exactly one logical region is editable across the trainer: the `select_on_off_ada_balance(on_solve_num)` controller (lines 394–414), with the per-question routing state (599–620), off-policy sample construction (759–799), and on-policy retention/deletion (942–963) wired to its output. The controller returns one triple per prompt-id:

```text
(on_remove_num, on_add_num, off_add_num)
```

- `on_remove_num`: how many of this question's on-policy rollouts are removed (8 = drop the whole group)
- `on_add_num`: how many additional on-policy rollouts to generate
- `off_add_num > 0`: add that many off-policy **SFT** samples (consumed by the fixed `sft` actor)
- `off_add_num < 0`: add off-policy **RL** samples (sets `whether_off=True`; consumed as off-policy RL)
- `off_add_num = 0`: add no off-policy samples (keep the on-policy GRPO group)

The starting scaffold is the default fill: **never switch** — return `(0, 0, 0)` for every question, so each prompt keeps its on-policy GRPO rollouts untouched.

```python
# EDITABLE region of mix_trainer.py — default fill (never switch; pure on-policy)
    def select_on_off_ada_balance(self, on_solve_num: int):
        del on_solve_num
        return 0, 0, 0
```

## Evaluation settings

Train **once** on a shared mixed OpenR1 subset (one deterministic random subset, 25600 requested rows, no source-bucket quota or length sorting) and validate on three held-out math splits: **AIME24** (30 examples), **AMC23** (40), and **MATH-500** (500, hidden). Backbone `Qwen/Qwen2.5-Math-1.5B`; backend vLLM, `tensor_model_parallel_size=2`, 4×H200; fixed `offline_loss_type=sft`, fixed `adv_estimator=grpo`, `reward_impl_version=6`. Training: `train_batch_size=128`, `max_response_length=8192`, `ppo_mini_batch_size=64`, `rollout.n=8`, `rollout.n_verify=8`, `rollout.n_val=16`, `total_training_steps=200`, `sft_loss_coef=1.0`, `max_position_embeddings=16384`, `rope_theta=40000`. The metric is the final `val/test_score/<benchmark>` for each split; higher is better. Seed 42.
