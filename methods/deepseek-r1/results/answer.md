# DeepSeek-R1

## Problem

Teaching LLMs to reason via supervised fine-tuning on human chain-of-thought traces is
expensive, injects human bias, and caps the model at the demonstrations — it cannot
explore non-human reasoning, and human traces omit the reflection/verification good
reasoning needs. Goal: incentivize reasoning directly with reinforcement learning against
a reward that only checks final-answer correctness, with no supervised reasoning traces;
then stage a pipeline so the final model is both a strong reasoner and a well-aligned
assistant.

## Key idea

**Pure-RL reasoning (R1-Zero).** Run RL on the base model with a *rule-based* reward
(accuracy + format, equal weight) and a minimal "think then answer" template — no SFT,
no process supervision, and no learned reward model for reasoning tasks. Long,
self-checking reasoning is incentivized only through final-answer correctness.

**GRPO** replaces PPO's value model with a *group baseline*. For each question sample `G`
outputs from the old policy, reward each, and standardize within the group:

```
A_i = (r_i − mean({r_1..r_G})) / std({r_1..r_G})     # one scalar per output
```

Optimize the clipped importance-ratio surrogate over each sampled output, with
KL-to-reference added *in the loss* (not as a per-token reward, which would penalize
cumulative KL and therefore discourage length):

```
J(θ) = E_q,{o_i} (1/G) Σ_i [ min( ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i ) − β·D_KL(π_θ‖π_ref) ]
ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)
D̂_KL = π_ref(o_i|q)/π_θ(o_i|q) − log(π_ref(o_i|q)/π_θ(o_i|q)) − 1   # = t − log t − 1, ≥0, unbiased
```

This removes the critic (no memory overhead; no ill-posed partial-response value
estimation for long CoT) and decouples KL from the reward. A deliberately *large* clip
ratio keeps more token gradients alive; the first R1 RL stage sets `ε = 10`. The
reference policy is refreshed to the current policy every 400 steps. R1-Zero grows max
generation length from 32,768 to 65,536 tokens after the 8.2k step.

**Reward design.** Rule-based for verifiable tasks: accuracy (math answer matched to
reference / code judged by compiler) + format (reasoning inside `<think>...</think>`,
answer inside `<answer>...</answer>`), equal weight. Rules are preferred because neural
reward models are susceptible to reward hacking over long RL. R1 adds a
language-consistency reward, `Num(Words_target) / Num(Words)`, to curb language mixing.

**Multi-stage R1 pipeline** (keeps emergent reasoning, fixes readability/alignment):
1. **Cold start:** build thousands of long CoT examples by sampling R1-Zero at
   temperature 1, keeping correct/readable outputs, converting seed traces with human
   annotators, using DeepSeek-V3 to rewrite/refine more data, and human-verifying the
   result; SFT the base model for conversational, first-person readability.
2. **Reasoning RL:** GRPO with rule rewards + language-consistency reward.
3. **Rejection-sampling SFT:** keep correct samples from the RL model, mix with
   non-reasoning data, SFT for broad capability.
4. **Second RL (alignment):** mixed rewards — rule-based on reasoning prompts,
   model-based reward on general prompts plus format and language rewards; general
   instruction data and preference rewards are used only in the final 400 of 1,700 steps
   to limit reward hacking.

## Code

```python
import torch

def group_advantages(rewards):                       # rewards: list of G scalars
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

def grpo_loss(policy, ref_policy, q, outputs, old_logp, advantages, beta, eps):
    total = 0.0
    for o, lp_old, A in zip(outputs, old_logp, advantages):
        lp_new = policy.token_logprobs(q, o).sum()    # log π_θ(o|q)
        lp_old = lp_old.sum()                         # log π_old(o|q)
        lp_ref = ref_policy.token_logprobs(q, o).sum() # log π_ref(o|q)
        ratio  = torch.exp(lp_new - lp_old)           # ρ_i for the whole output

        surrogate = torch.min(ratio * A,
                              torch.clamp(ratio, 1 - eps, 1 + eps) * A)
        t  = torch.exp(lp_ref - lp_new)               # t = π_ref/π_θ
        kl = t - (lp_ref - lp_new) - 1.0              # t - log t - 1  (≥0, unbiased)

        total += surrogate - beta * kl                # KL in the loss, not the reward
    return -(total / len(outputs))                     # negate to minimize

def rule_reward(q, o):
    return accuracy_reward(q, o) + format_reward(o)    # equal weight; +language_consistency(o) in stage 2

def rl_step(policy, old_policy, ref_policy, questions, opt,
            G=16, temp=1.0, max_len=32768, beta=1e-3, eps=10):
    for q in questions:
        outs = old_policy.sample(q, n=G, temperature=temp, max_len=max_len)
        adv  = group_advantages([rule_reward(q, o) for o in outs])
        old_logp = [old_policy.token_logprobs(q, o) for o in outs]
        loss = grpo_loss(policy, ref_policy, q, outs, old_logp, adv, beta, eps)
        loss.backward(); opt.step(); opt.zero_grad()

def build_cold_start_data(zero_model, prompts, n_per_prompt=16):
    samples = []
    for q in prompts:
        samples += [(q, o) for o in zero_model.sample(q, n=n_per_prompt,
                                                       temperature=1.0, max_len=32768)]
    kept = [o for q, o in samples
            if answer_correct(q, o) and readable(o) and not language_mixed(o)]
    refined = deepseek_v3_refine(human_style_convert(kept))
    return human_verify(refined)

def general_reward(q, o):
    rm = helpful_reward(o) if helpfulness_prompt(q) else safety_reward(o)
    return rm + format_reward(o)

def pipeline(base, zero_model):
    cold_start_traces = build_cold_start_data(zero_model, reasoning_prompts)
    policy = sft(base, cold_start_traces)              # 1. conversational cold start
    policy = grpo_train(policy, reasoning_prompts,     # 2. reasoning RL + lang consistency
                        reward=lambda q, o: rule_reward(q, o) + language_consistency(o))
    data   = rejection_sample(policy, prompts) + non_reasoning_data
    policy = sft(policy, data)                          # 3. rejection-sampling SFT
    policy = grpo_train(policy, mixed_prompts,          # 4. mixed-reward alignment RL
                        reward=lambda q, o: (rule_reward(q, o) if verifiable(q)
                                             else general_reward(q, o)) + language_consistency(o),
                        temperature=0.7, steps=1700, general_reward_steps=400)
    return policy
```

RL config: GRPO, group size 16, rollout temperature 1 for R1-Zero and the first R1 RL
stage, lr 3e-6, KL coef 0.001, batch 32 questions (512 samples) per step, reference
refreshed every 400 steps, 8,192 rollouts split into 16 minibatches with one inner epoch.
R1-Zero trains 10,400 steps and grows max generation length from 32,768 to 65,536 tokens
after 8.2k steps; the first R1 RL stage uses `ε = 10`; the second RL stage uses
temperature 0.7 and 1,700 steps.
