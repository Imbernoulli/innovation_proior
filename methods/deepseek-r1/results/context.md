## Research question

Large language models can reason — solve competition mathematics, write code, work
through STEM problems — but the prevailing way to make them reason well leans heavily on
*human-annotated reasoning demonstrations*: chain-of-thought exemplars, supervised
fine-tuning on curated multi-step solutions. That dependence is the pain. It does not
scale (high-quality reasoning traces are expensive to write), it injects the
annotators' cognitive biases, and — most fundamentally — it *caps* the model at human
ability: by training the model to imitate human thought processes, it can never explore
reasoning strategies that no human wrote down. The question: can a model's reasoning
ability be *incentivized* directly, by reinforcement learning against a reward that only
checks whether the final answer is correct, with no supervised reasoning traces at all —
letting the model discover its own reasoning behaviors (reflection, verification,
backtracking) through trial and error? And if pure RL produces strong but rough reasoning
(poor readability, language mixing), how should a full training pipeline be staged so the
final model is both a strong reasoner and a well-aligned general assistant?

## Background

**The post-training paradigm: SFT then RL.** A pre-trained base LLM is usually refined
by supervised fine-tuning (SFT) on curated input–output pairs (minimizing cross-entropy
to human-written targets), then by reinforcement learning from human/AI feedback to
align with preferences. SFT gives precise task grounding and is sample-efficient; RL then
optimizes broader objectives (helpfulness, brevity) that fixed targets cannot capture, as
in the InstructGPT recipe. The diagnostic limitation that motivates this work: human SFT
targets often *omit* the very components good reasoning needs — explicit reflection,
verification, dead-end recovery — and being fixed, they cap exploration. So SFT may
actively impede a model's discovery of effective reasoning.

**Chain-of-thought and emergent reasoning.** Prompting a model to produce intermediate
steps ("Let's think step by step") substantially improves performance on complex tasks,
and reasoning ability emerges as models scale. Letting a model generate a long chain of
thought before answering is the behavior we want to *learn to produce*, not hand-engineer.

**Reinforcement learning for LLMs and PPO.** The standard RL algorithm for LLM
post-training is Proximal Policy Optimization. For each prompt the policy samples an
output; a learned *value model* (critic) of comparable size to the policy estimates a
per-token baseline, and Generalized Advantage Estimation (GAE) combines rewards and the
value estimates into per-token advantages; the policy is updated with a clipped
importance-ratio surrogate, and a per-token KL-to-reference penalty is added as a dense
reward to keep the policy near a reference. The pre-method costs that matter here:
(i) the critic doubles memory and compute; (ii) the critic must predict expected
future reward from a *partial* response, which is intrinsically hard when only a final
outcome reward exists — and harder still for long chains of thought, where early text may
later be revised or contradicted, so a partial-response value is barely meaningful; and
(iii) folding the KL penalty into the per-token reward penalizes the cumulative KL,
which implicitly discourages longer responses — exactly the wrong pressure if we want the
model to think longer.

**Verifiable rewards.** For tasks with deterministic answers — math with a final numeric
result, code judged by a compiler against test cases — correctness can be checked by a
*rule*, not a learned model. Rule-based rewards are cheap and reliable. Neural reward
models, in contrast, are susceptible to *reward hacking*: over a long RL run the policy
finds shortcuts that fool the reward model rather than genuinely improving. This is the
pre-method fact behind preferring rules where the task allows it.

## Baselines

**SFT on human reasoning traces.** Fine-tune the base model on curated chain-of-thought
solutions with cross-entropy. Core idea: teach reasoning by imitation. Gap: expensive to
annotate, injects human bias, and caps performance at the demonstrations — no exploration
of non-human reasoning.

**PPO-based RLHF (Schulman et al. 2017; Ouyang et al. 2022).** Policy + value model,
GAE advantages, clipped surrogate, per-token KL-to-reference penalty as dense reward.
Core idea: optimize a (possibly learned) reward while staying near a reference policy.
Gaps: the value model's memory/compute overhead; the difficulty of value estimation from
partial long-CoT responses; and the length-discouraging effect of penalizing cumulative
per-token KL.

**Group Relative Policy Optimization (Shao et al. 2024, GRPO).** The RL algorithm this
work adopts. For each question, sample a *group* of `G` outputs from the old policy;
compute each output's reward; standardize the rewards within the group to get a scalar
advantage per output (mean-subtract, divide by std); optimize a clipped importance-ratio
surrogate using these group-relative advantages, with the KL-to-reference added directly
in the loss (not as a per-token reward). Core idea: replace the critic with a group
baseline — the advantage of an output is how much better its reward is than its
group-mates'. This removes the value model entirely, sidestepping all three PPO costs
above. (It is the immediate ancestor and the RL engine of the method.)

**Neural reward models (outcome- or process-based).** Train a model to score responses
or reasoning steps. Core idea: a learned, generalizable reward for tasks without a rule.
Gap: vulnerable to reward hacking under large-scale RL, especially process reward models;
costly to retrain. Hence rules are preferred for verifiable reasoning, and neural reward
models are reserved for genuinely subjective tasks (helpfulness, safety).

## Evaluation settings

The base model is a large pre-trained mixture-of-experts LLM (DeepSeek-V3-Base). RL
prompts for the reasoning stages are verifiable problems: competition mathematics (final
answer a number / expression / equation, matched against a reference; proofs excluded as
hard to verify), coding-competition problems (judged by a compiler against test suites),
and logic problems. Benchmarks used as yardsticks (all pre-existing): AIME 2024 (reported
as pass@1 and, with self-consistency / majority voting, cons@16, against the average
human competitor score), coding-competition and graduate-level STEM problem sets, plus
general-assistant evaluations (instruction-following IF-Eval, ArenaHard, AlpacaEval 2.0)
for the alignment stages. RL configuration: GRPO with group size 16 per question, rollout
temperature 1, learning rate 3e-6, KL coefficient 0.001, large clip ratio, batch of 32
questions per step (512 samples), reference policy refreshed every 400 steps; max
generation length grown from 32,768 to 65,536 tokens during training. No outcome numbers
are part of these settings.

## Code framework

The harness has a frozen reference policy, a trainable policy initialized from a
pre-trained base model, a sampler (rollout) that generates multiple completions per
prompt, a reward function, and an RL update loop. The advantage estimator, the reward
design, the KL handling, and the overall multi-stage pipeline are the empty slots.

```python
import torch, torch.nn.functional as F

class Policy:
    """Autoregressive LLM policy; can sample completions and score token log-probs."""
    def sample(self, prompt, n, temperature, max_len):  # -> list of completions
        pass
    def logprob(self, prompt, completion):              # -> sum/seq of token log-probs
        pass

def reward(prompt, completion):
    # TODO: design the training signal (verifiable? rule-based? model-based?).
    pass

def compute_advantages(rewards_per_group):
    # TODO: turn a group of rewards for one prompt into per-output advantages
    #       (decide whether a learned value model is needed at all).
    pass

def rl_objective(policy, ref_policy, old_logprobs, completions, advantages, beta, eps):
    # TODO: the surrogate objective and how KL-to-reference is incorporated.
    pass

def rl_step(policy, ref_policy, prompts, G, temperature, max_len, beta, eps):
    for q in prompts:
        outs = policy.sample(q, n=G, temperature=temperature, max_len=max_len)
        rs   = [reward(q, o) for o in outs]
        adv  = compute_advantages(rs)
        # ... compute old log-probs, then optimize rl_objective ...

def pipeline(base_model):
    # TODO: stage the training (pure-RL exploration vs. a multi-stage aligned model).
    pass
```
