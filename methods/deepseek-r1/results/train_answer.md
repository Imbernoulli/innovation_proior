The central problem is how to make a large language model reason well without depending on human-written chain-of-thought traces. The usual recipe—supervised fine-tuning on curated step-by-step solutions—works in the small but hits a hard ceiling. Hand-authored traces are expensive, carry the annotators’ blind spots, and almost never contain the messy real reasoning that solves hard problems: the false starts, the rechecking, the "wait, that is wrong" moments. Worse, because SFT rewards matching the demonstration, it actively punishes any reasoning strategy that a human did not write down. The model can never exceed the demonstrations. Reinforcement learning seems like the natural alternative, but the standard choice for LLM post-training, PPO, brings its own problems for long reasoning: a value model that doubles memory and compute, the near-impossible task of predicting final reward from a half-finished chain of thought that the model may later contradict, and a per-token KL penalty that discourages the longer responses we actually want.

The right signal is not imitation but verification. For a wide class of reasoning tasks—competition math, coding problems, logic puzzles—the final answer can be checked by a cheap, reliable rule rather than a learned reward model. That removes the risk of reward hacking and avoids supervising the process at all. The question becomes how to set up the RL algorithm so that a base model, given only a final-answer reward and a minimal format constraint, discovers reflection, verification, and backtracking on its own, and then how to stage the rest of training so the resulting model is also readable, coherent, and useful as a general assistant.

I propose DeepSeek-R1. It has two pieces: R1-Zero, a pure-RL demonstration that strong reasoning can emerge from base-model exploration with no supervised traces, and the full DeepSeek-R1 pipeline, which keeps that emergent reasoning while fixing the predictable rough edges. The algorithmic engine is Group Relative Policy Optimization (GRPO). Instead of PPO’s value model, GRPO samples a group of G completions for each question, rewards each completion by the rule, and uses the group itself as a baseline. The advantage for output i is just its reward standardized against the group: A_i = (r_i − mean(r)) / (std(r) + eps). This is one scalar per completion, so it sidesteps the critic entirely. The policy update keeps PPO’s clipped importance-ratio surrogate, applied to the whole output probability ratio ρ_i = π_θ(o_i|q) / π_old(o_i|q), with a deliberately large clip ratio ε = 10 in the first RL stage so that gradients are not truncated away across long chains of thought. The KL-to-reference penalty is added directly in the loss, not folded into the reward, using the non-negative unbiased estimator D_KL_hat = t − log t − 1 where t = π_ref(o|q) / π_θ(o|q). Putting KL in the loss decouples it from response length, so the model is free to think longer when that helps correctness.

The reward itself is rule-based: an accuracy reward for a correct final answer and a format reward for wrapping reasoning in `<think>...</think>` and the final answer in `<answer>...</answer>`, combined with equal weight. No process supervision, no neural reward model for reasoning. For the full DeepSeek-R1 model, a language-consistency reward—the fraction of words in the target language over the whole chain of thought—is added to curb language mixing. The training pipeline then proceeds in four stages. First, a cold start: sample the zero-RL model at high temperature, keep correct and readable traces, have annotators convert them into a natural first-person conversational style, use a strong model to rewrite and expand the set, and verify the result; SFT the base model on this small, style-oriented corpus. Second, reasoning RL with GRPO using rule rewards plus language consistency. Third, rejection sampling plus SFT: keep correct completions from the RL model, filter chaotic traces, and mix about 600k reasoning samples with 200k non-reasoning samples for broad capability. Fourth, a final mixed-reward RL stage: rule rewards on verifiable reasoning prompts, model-based helpfulness or safety rewards on general prompts, plus format and language consistency, with model-based rewards restricted to the last 400 of 1,700 steps to limit reward hacking.

```python
import torch

def group_advantages(rewards):
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

def grpo_loss(policy, ref_policy, q, outputs, old_logp, advantages, beta, eps):
    total = 0.0
    for o, lp_old, A in zip(outputs, old_logp, advantages):
        lp_new = policy.token_logprobs(q, o).sum()
        lp_old = lp_old.sum()
        lp_ref = ref_policy.token_logprobs(q, o).sum()
        ratio = torch.exp(lp_new - lp_old)

        unclipped = ratio * A
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * A
        surrogate = torch.min(unclipped, clipped)

        t = torch.exp(lp_ref - lp_new)
        kl = t - (lp_ref - lp_new) - 1.0

        total += surrogate - beta * kl
    return -(total / len(outputs))

def rule_reward(q, o):
    return accuracy_reward(q, o) + format_reward(o)

def language_consistency(o):
    return num_target_lang_words(o) / (num_words(o) + 1e-6)

def rl_step(policy, old_policy, ref_policy, questions, opt,
            G=16, temp=1.0, max_len=32768, beta=1e-3, eps=10):
    for q in questions:
        outs = old_policy.sample(q, n=G, temperature=temp, max_len=max_len)
        rewards = [rule_reward(q, o) + language_consistency(o) for o in outs]
        adv = group_advantages(rewards)
        old_logp = [old_policy.token_logprobs(q, o) for o in outs]
        loss = grpo_loss(policy, ref_policy, q, outs, old_logp, adv, beta, eps)
        loss.backward()
        opt.step()
        opt.zero_grad()

def build_cold_start_data(zero_model, prompts, n_per_prompt=16):
    samples = []
    for q in prompts:
        samples += [(q, o) for o in zero_model.sample(
            q, n=n_per_prompt, temperature=1.0, max_len=32768)]
    kept = [o for q, o in samples
            if answer_correct(q, o) and readable(o) and not language_mixed(o)]
    refined = deepseek_v3_refine(human_style_convert(kept))
    return human_verify(refined)

def general_reward(q, o):
    rm = helpful_reward(o) if helpfulness_prompt(q) else safety_reward(o)
    return rm + format_reward(o)

def pipeline(base, zero_model):
    cold_start_traces = build_cold_start_data(zero_model, reasoning_prompts)
    policy = sft(base, cold_start_traces)
    policy = grpo_train(policy, reasoning_prompts,
                        reward=lambda q, o: rule_reward(q, o) + language_consistency(o))
    data = rejection_sample(policy, prompts) + non_reasoning_data
    policy = sft(policy, data)
    policy = grpo_train(policy, mixed_prompts,
                        reward=lambda q, o: (
                            rule_reward(q, o) if verifiable(q) else general_reward(q, o)
                        ) + language_consistency(o),
                        temperature=0.7, steps=1700, general_reward_steps=400)
    return policy
```
