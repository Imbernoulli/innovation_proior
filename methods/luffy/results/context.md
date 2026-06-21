# Context: RL fine-tuning for reasoning with off-policy guidance from a stronger model

## Research question

We have a pretrained language-model policy `π_θ` and want to post-train it to reason — to solve
math/competition problems where a rule-based verifier scores any candidate solution as correct (1)
or incorrect (0). The dominant recipe is "zero-RL": run on-policy RL with verifiable rewards (RLVR)
directly on the base model. We *also* have, for many prompts, off-the-shelf reasoning traces from
a much stronger model (e.g. DeepSeek-R1). The question is how to use those off-policy traces
together with on-policy rollouts inside a single training stage.

## Background

The load-bearing concepts:

- **The policy-gradient frame and GRPO (Shao et al. 2024).** Post-training is policy optimization. For
  a prompt `q`, sample `N` rollouts `τ_i ~ π_{θ_old}`, score each with the verifier `R(τ_i) ∈ {0,1}`,
  and form a critic-free group-relative advantage by standardizing within the group,
  `A_i = (R(τ_i) − mean({R(τ_k)})) / std({R(τ_k)})`. Optimize the PPO clipped surrogate
  `J_GRPO = (1/Σ|τ_i|) Σ_i Σ_t CLIP(r_{i,t}(θ), A_i, ε) − β·KL(π_θ ‖ π_ref)`, where
  `r_{i,t} = π_θ(τ_{i,t}|q,τ_{i,<t}) / π_{θ_old}(τ_{i,t}|q,τ_{i,<t})` is the importance ratio and
  `CLIP(r, A, ε) = min(r·A, clip(r, 1−ε, 1+ε)·A)`. The group mean is the baseline (no value network).
- **The "one gradient, many estimators" stance (GAE; Schulman et al. 2015).** There is a family of
  estimators of the same underlying policy gradient; one picks an instance by choosing knobs (the
  reference/sampling distribution, the advantage). This is the conceptual ground for treating
  on-policy and off-policy data as instances of one estimator.
- **Importance sampling / change of measure.** A gradient defined as an expectation under one
  distribution can be estimated from samples of another by reweighting with the ratio of the two
  densities; the ratio's denominator is the reference policy.
- **The on-policy RL setting (SimpleRL-Zoo, Zeng et al. 2025; Yue et al. 2025).** Zero-RL presupposes
  the model can sample reward. RLVR lifts Pass@1 and is the current standard for math reasoning
  post-training. Pass@k to large `k` is also used as an explicit probe of exploration capacity.

## Baselines

**SFT on demonstrations (Wei et al. 2021; behavior cloning).** Minimize token-level NLL on the teacher
traces. Raises competence on prompts the model cannot solve.

**On-policy RL / GRPO / Dr.GRPO (Shao et al. 2024; Liu et al. 2025).** Critic-free on-policy RLVR; the
standard zero-RL engine. Sharpens reasoning on prompts the model can partly solve.

**SFT→RL pipelines and RL-with-SFT-loss (Fu et al. 2025).** Distill the teacher first (or add an SFT
loss during RL), then run RL. Combines both signals in a sequential or joint way.

## Evaluation settings

- **Backbones:** Qwen2.5-Math-7B (default), Qwen2.5-Math-1.5B, Qwen2.5-Instruct-7B, LLaMA-3.1-8B.
- **In-distribution math benchmarks:** AIME 2024, AIME 2025, AMC, MATH-500 (Hendrycks et al. 2021),
  Minerva (Lewkowycz et al. 2022), OlympiadBench (He et al. 2024).
- **Out-of-distribution suites:** ARC-c (Clark et al. 2018), GPQA-diamond (Rein et al. 2024), MMLU-Pro.
- **Metrics / protocol:** rule-based answer verification (binary), Pass@1; for the small benchmarks
  (AIME, AMC) report avg@32; temperature 0.6, top-p 0.95, max length 8192 at test. Pass@k to large `k`
  as an explicit probe of exploration capacity / capability boundary.
- **Training:** GRPO-style on-policy optimization, rollout group of 8 per prompt at temperature 1.0,
  constant lr 1e-6 with AdamW, single pass over a mixed dataset pairing each prompt with both an
  off-policy teacher trace and on-policy rollouts. Dataset: a 45k-prompt subset of OpenR1-Math-220k
  with off-policy traces from DeepSeek-R1, filtered (≤8192 tokens, verified correct).

## Code framework

A GRPO-style trainer already exists: per prompt it draws on-policy rollouts, verifies them, computes
group-relative advantages, and optimizes the clipped surrogate. It also has access, per prompt, to an
off-policy teacher trace. What is not settled is how to assemble the on-policy rollouts and the
off-policy trace into one advantage group and one actor objective.

```python
import torch


def grpo_group_advantage(rewards):
    """Group-relative advantage over a set of scored trajectories (already provided).
    rewards: per-trajectory verifier scores in {0,1}. Returns standardized advantages."""
    mean = rewards.mean()
    std = rewards.std()
    if std == 0:
        std = torch.tensor(1.0)
    return (rewards - mean) / (std + 1e-6)


def clipped_surrogate(log_prob, old_log_prob, advantages, mask, clip=0.2):
    """On-policy clipped surrogate (GRPO/PPO), already provided."""
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -masked_mean(torch.min(unclipped, clipped), mask)


class MixedPolicyTrainer:
    """Per prompt: verifier-scored on-policy rollouts AND one off-policy teacher trace."""

    def train_step(self, batch):
        total = 0.0
        for prompt in batch:
            on_rollouts = self.sample_rollouts(prompt)       # τ_i ~ π_{θ_old}
            on_rewards = self.verify(on_rollouts)            # R(τ_i) ∈ {0,1}
            off_trace = prompt.off_policy_trace              # τ from a stronger model (R-1)
            off_reward = self.verify([off_trace])            # typically correct -> 1

            # TODO: the mixed-policy objective we will design.
            loss_q = self.mixed_objective(on_rollouts, on_rewards, off_trace, off_reward)
            total = total + loss_q
        total.backward()
        self.optimizer.step(); self.optimizer.zero_grad()    # AdamW, lr 1e-6

    def mixed_objective(self, on_rollouts, on_rewards, off_trace, off_reward):
        # TODO: combine on-policy and off-policy into one advantage group + one objective.
        pass
```

The empty mixed-objective slot is the only missing piece; the group advantage and the clipped surrogate
already exist.
