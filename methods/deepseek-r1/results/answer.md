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
(accuracy + format) and a minimal "think then answer" template — no SFT, no process
supervision. Long, self-checking reasoning (reflection, verification, backtracking, the
"aha" re-examination) *emerges* because it raises answer correctness.

**GRPO** replaces PPO's value model with a *group baseline*. For each question sample `G`
outputs from the old policy, reward each, and standardize within the group:

```
A_i = (r_i − mean({r_1..r_G})) / std({r_1..r_G})     # one scalar per output, all its tokens
```

Optimize the clipped importance-ratio surrogate, with KL-to-reference added *in the loss*
(not as a per-token reward, which would penalize length):

```
J(θ) = E_q,{o_i} (1/G) Σ_i [ min( ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i ) − β·D_KL(π_θ‖π_ref) ]
ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)
D̂_KL = π_ref(o_i|q)/π_θ(o_i|q) − log(π_ref(o_i|q)/π_θ(o_i|q)) − 1   # = t − log t − 1, ≥0, unbiased
```

This removes the critic (no memory overhead; no impossible partial-response value
estimation for long CoT) and decouples KL from length. A deliberately *large* clip ratio
keeps more token gradients alive. The reference policy is refreshed to the current policy
periodically; the max generation length is grown during training (32k → 64k).

**Reward design.** Rule-based for verifiable tasks: accuracy (math answer matched to
reference / code judged by compiler) + format (reasoning inside `<think>...</think>`,
answer inside `<answer>...</answer>`), equal weight. Rules are preferred because neural
reward models get hacked over long RL. A language-consistency reward (fraction of CoT in
the target language) is added to curb language mixing.

**Multi-stage R1 pipeline** (keeps emergent reasoning, fixes readability/alignment):
1. **Cold start:** small SFT on a few thousand clean conversational CoTs — for *style*,
   kept small so it does not cap exploration.
2. **Reasoning RL:** GRPO with rule rewards + language-consistency reward.
3. **Rejection-sampling SFT:** keep correct samples from the RL model, mix with
   non-reasoning data, SFT for broad capability.
4. **Second RL (alignment):** mixed rewards — rule-based on reasoning prompts, model-based
   (helpfulness + safety reward models) on general prompts, used briefly at the end to
   limit reward hacking.

## Code

```python
import torch

def group_advantages(rewards):                       # rewards: list of G scalars
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std() + 1e-6)          # A_i = (r_i - mean)/std

def grpo_loss(policy, ref_policy, q, outputs, old_logp, advantages, beta, eps):
    total, n_tok = 0.0, 0
    for o, lp_old, A in zip(outputs, old_logp, advantages):
        lp_new = policy.token_logprobs(q, o)          # (T,)
        lp_ref = ref_policy.token_logprobs(q, o)      # (T,)
        ratio  = torch.exp(lp_new - lp_old)           # ρ per token

        surrogate = torch.min(ratio * A,
                              torch.clamp(ratio, 1 - eps, 1 + eps) * A)
        t  = torch.exp(lp_ref - lp_new)               # t = π_ref/π_θ
        kl = t - (lp_ref - lp_new) - 1.0              # t - log t - 1  (≥0, unbiased)

        total += (surrogate - beta * kl).sum()        # KL in the loss, not the reward
        n_tok += len(lp_new)
    return -(total / n_tok)                            # negate to minimize

def rule_reward(q, o):
    return accuracy_reward(q, o) + format_reward(o)    # equal weight; +language_consistency(o) in stage 2

def rl_step(policy, old_policy, ref_policy, questions, opt,
            G=16, temp=1.0, max_len=32768, beta=1e-3, eps=...):  # large eps
    for q in questions:
        outs = old_policy.sample(q, n=G, temperature=temp, max_len=max_len)
        adv  = group_advantages([rule_reward(q, o) for o in outs])
        old_logp = [old_policy.token_logprobs(q, o) for o in outs]
        loss = grpo_loss(policy, ref_policy, q, outs, old_logp, adv, beta, eps)
        loss.backward(); opt.step(); opt.zero_grad()

def pipeline(base):
    policy = sft(base, cold_start_traces)              # 1. small style SFT
    policy = grpo_train(policy, reasoning_prompts,     # 2. reasoning RL + lang consistency
                        reward=lambda q, o: rule_reward(q, o) + language_consistency(o))
    data   = rejection_sample(policy, prompts) + non_reasoning_data
    policy = sft(policy, data)                          # 3. rejection-sampling SFT
    policy = grpo_train(policy, mixed_prompts,          # 4. mixed-reward alignment RL
                        reward=lambda q, o: rule_reward(q, o) if verifiable(q)
                                            else rm_helpful(o) + rm_safety(o))
    return policy
```

RL config: GRPO, group size 16, temperature 1, lr 3e-6, KL coef 0.001, batch 32
questions (512 samples) per step, reference refreshed every 400 steps, generation length
grown from 32,768 to 65,536 tokens.
