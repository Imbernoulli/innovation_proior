## Research question

Large language models can reason — solve competition mathematics, write code, work
through STEM problems — and the prevailing way to make them reason well relies on
*human-annotated reasoning demonstrations*: chain-of-thought exemplars and supervised
fine-tuning on curated multi-step solutions. The question: can a model's reasoning
ability be *incentivized* directly, by reinforcement learning against a reward that only
checks whether the final answer is correct, with no supervised reasoning traces at all?
And how should a full training pipeline be staged so the final model is both a strong
reasoner and a well-aligned general assistant?

## Background

**The post-training paradigm: SFT then RL.** A pre-trained base LLM is usually refined
by supervised fine-tuning (SFT) on curated input–output pairs (minimizing cross-entropy
to human-written targets), then by reinforcement learning from human/AI feedback to
align with preferences. SFT gives precise task grounding and is sample-efficient; RL then
optimizes broader objectives (helpfulness, brevity) that fixed targets cannot capture, as
in the InstructGPT recipe.

**Chain-of-thought and emergent reasoning.** Prompting a model to produce intermediate
steps ("Let's think step by step") substantially improves performance on complex tasks,
and reasoning ability emerges as models scale. Letting a model generate a long chain of
thought before answering is a behavior that improves complex-task performance.

**Reinforcement learning for LLMs and PPO.** The standard RL algorithm for LLM
post-training is Proximal Policy Optimization. For each prompt the policy samples an
output; a learned *value model* (critic) of comparable size to the policy estimates a
per-token baseline, and Generalized Advantage Estimation (GAE) combines rewards and the
value estimates into per-token advantages; the policy is updated with a clipped
importance-ratio surrogate, and a per-token KL-to-reference penalty is added as a dense
reward to keep the policy near a reference.

**Verifiable rewards.** For tasks with deterministic answers — math with a final numeric
result, code judged by a compiler against test cases — correctness can be checked by a
*rule*, not a learned model. Rule-based rewards are cheap and reliable. Neural reward
models are trained to score responses or reasoning steps and provide a generalizable
signal for tasks without a rule.

## Baselines

**SFT on human reasoning traces.** Fine-tune the base model on curated chain-of-thought
solutions with cross-entropy. Core idea: teach reasoning by imitation.

**PPO-based RLHF (Schulman et al. 2017; Ouyang et al. 2022).** Policy + value model,
GAE advantages, clipped surrogate, per-token KL-to-reference penalty as dense reward.
Core idea: optimize a (possibly learned) reward while staying near a reference policy.

**Group Relative Policy Optimization (Shao et al. 2024, GRPO).** For each question,
sample a *group* of `G` outputs from the old policy; compute each output's reward;
standardize the rewards within the group to get a scalar advantage per output
(mean-subtract, divide by std); optimize a clipped importance-ratio surrogate using
these group-relative advantages, with the KL-to-reference added directly in the loss
(not as a per-token reward). Core idea: replace the critic with a group baseline — the
advantage of an output is how much better its reward is than its group-mates'. This
removes the value model entirely.

**Neural reward models (outcome- or process-based).** Train a model to score responses
or reasoning steps. Core idea: a learned, generalizable reward for tasks without a rule;
used for subjective tasks (helpfulness, safety).

## Evaluation settings

The base model is a large pre-trained mixture-of-experts LLM (DeepSeek-V3-Base). RL
prompts for the reasoning stages are verifiable problems: competition mathematics (final
answer a number / expression / equation, matched against a reference; proofs excluded as
hard to verify), coding-competition problems (judged by a compiler against test suites),
and logic problems. Benchmarks used as yardsticks are pre-existing reasoning,
coding-competition, STEM, and general-assistant evaluations, including AIME 2024,
Codeforces-style tasks, graduate-level STEM sets, IF-Eval, ArenaHard, and AlpacaEval 2.0.
RL configuration: GRPO with group size 16 per question, rollout temperature 1, learning
rate 3e-6, KL coefficient 0.001, large clip ratio, batch of 32 questions per step (512
samples), reference policy refreshed every 400 steps; the maximum generation length can
grow from 32,768 to 65,536 tokens during training.

## Code framework

The harness has a frozen reference policy, a trainable policy initialized from a
pre-trained base model, a sampler (rollout) that generates multiple completions per
prompt, a reward function, and an RL update loop. The advantage estimator, the reward
design, the KL handling, and the overall training pipeline are the empty slots.

```python
import torch

class Policy:
    """Autoregressive LLM policy; can sample completions and score token log-probs."""
    def sample(self, prompt, n, temperature, max_len):  # -> list of completions
        pass
    def token_logprobs(self, prompt, completion):       # -> token log-probs
        pass

def reward(prompt, completion):
    # TODO: design the training signal (verifiable? rule-based? model-based?).
    pass

def compute_advantages(rewards_per_group):
    # TODO: turn a group of rewards for one prompt into per-output advantages.
    pass

def rl_objective(policy, ref_policy, prompt, old_logprobs, completions, advantages, beta, eps):
    # TODO: the surrogate objective and how KL-to-reference is incorporated.
    pass

def rl_step(policy, old_policy, ref_policy, prompts, G, temperature, max_len, beta, eps):
    for q in prompts:
        outs = old_policy.sample(q, n=G, temperature=temperature, max_len=max_len)
        rs   = [reward(q, o) for o in outs]
        adv  = compute_advantages(rs)
        old_logprobs = [old_policy.token_logprobs(q, o) for o in outs]
        # ... optimize rl_objective(policy, ref_policy, q, old_logprobs, outs, adv, beta, eps) ...

def pipeline(base_model):
    # TODO: train and align the model.
    pass
```
