Let me start from what's actually limiting reasoning models, because the standard recipe has a ceiling baked in. The way we make an LLM reason is to fine-tune it on human-written chain-of-thought: curated step-by-step solutions, supervised, cross-entropy to the human target. It works, but think about what it *can't* do. The targets are human reasoning traces, and they're fixed. So the model is being trained to imitate how a person wrote down their thinking — which means (a) it's expensive, every good trace is hand-authored; (b) it inherits whatever blind spots and biases the annotators had; and (c) most importantly, it can never get *better* at reasoning than the demonstrations, because the loss only rewards matching them. If there's a smarter, non-human way to attack a problem — a weird backtrack, an unusual verification — SFT actively punishes it, because it doesn't match the human target. Worse, the human traces usually *omit* the very things good reasoning needs: people write the clean final derivation, not the false starts, the "wait, that's wrong, let me recheck," the explicit verification. So SFT on human traces may be teaching the model to *not* do the messy, effortful reasoning that actually solves hard problems.

So the question flips: instead of telling the model *how* to think, can I just reward it for *getting the right answer* and let it figure out how to think on its own? For a big class of problems I don't need a human to grade the reasoning — I need only to check the final answer. A competition math problem has a deterministic answer; I can match the model's boxed final answer against the reference. A coding problem has test cases; a compiler tells me pass or fail. So the reward can be a *rule*, not a judgment: 1 if the final answer is correct, 0 otherwise. No supervision on the reasoning process at all. That's the bet — give the model the right incentive and freedom, and see whether reasoning *emerges*.

Now, the reward is a rule and that matters for a second reason. The alternative is a learned neural reward model that scores responses. But over a long RL run, a neural reward model gets *hacked*: the policy finds inputs that fool the scorer rather than genuinely improving, especially a process reward model that tries to score intermediate steps. A rule can't be hacked — correct is correct. So for verifiable reasoning, rule-based reward. Specifically two pieces: an accuracy reward (is the final answer right, checked by rule/compiler) and a format reward (did the model put its reasoning between `<think>...</think>` and its answer between `<answer>...</answer>`), combined with equal weight. The format reward isn't about quality — it's so the thinking is delineated and I can reliably extract the final answer to grade it. And I deliberately keep the template minimal: just "think first inside `<think>`, then answer inside `<answer>`," no content guidance, so I'm not smuggling in human reasoning priors and I can watch the model's natural progression.

Which RL algorithm? The default is PPO. But look at what PPO needs and why it hurts here. PPO trains a *value model* — a critic the same size as the policy — to estimate, at each token, the expected future reward, and uses GAE to turn that into per-token advantages. Three problems, all sharp for long reasoning. First, the critic doubles the memory and compute. Second, and worse, the critic has to predict the eventual reward from a *partial* response, which is intrinsically hard when the only reward is the final outcome — and for long chains of thought it's nearly hopeless, because the model might write something, then later reflect and *contradict* it, so the value of a half-finished reasoning chain barely means anything. Third, PPO folds the KL-to-reference penalty into the per-token reward as a dense term; since RL maximizes cumulative reward, that penalizes *cumulative* KL, which implicitly penalizes *length* — and I want the model to think *longer*, not shorter. So PPO is fighting me on the exact axis I care about.

Can I get advantages without a critic? Here's the move. For a given question, I don't have an absolute baseline, but I can *sample a group* of outputs from the current policy and use the group itself as the baseline. Sample `G` outputs `{o_1,…,o_G}` for question `q` from the old policy `π_{θ_old}`, get each one's reward `r_i` by rule. Then "how good is output `i`" relative to what this policy typically does on `q` is just how far `r_i` is above the group's average. Standardize within the group:
`A_i = (r_i − mean({r_1,…,r_G})) / std({r_1,…,r_G})`.
That's the advantage — group-relative, one scalar per output, the *same* advantage assigned to every token of that output. No value model. The group of sampled rewards *is* the baseline, and standardizing by the std normalizes the scale across questions of differing difficulty. This kills all three PPO problems: no critic (no memory/compute overhead, no impossible partial-response value estimation), and I'm free to put KL wherever I want.

Now the policy update. Keep PPO's clipped importance-ratio surrogate — it's the right tool for taking multiple gradient steps on data sampled from a slightly stale policy without the update running away. Let `ρ_i = π_θ(o_i|q) / π_{θ_old}(o_i|q)` be the probability ratio. The clipped term is `min( ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i )`: when `A_i > 0` (good output) the ratio is allowed to rise only to `1+ε` before the gradient is clipped off, so the policy can't lurch too far toward it in one update; when `A_i < 0` (bad output) the ratio is floored at `1−ε`. The clip is what bounds the trust region per step.

Where does the KL go? Not into the reward — I established that penalizing cumulative per-token KL discourages length. Instead add the KL-to-reference *directly in the loss* as a separate term, `−β · D_KL(π_θ ‖ π_ref)`. This keeps the policy from drifting too far from a sane reference without coupling the penalty to response length.

I need an *estimator* of `D_KL(π_θ ‖ π_ref)` I can compute from samples — I only have sampled outputs, not the full distributions. The naive Monte Carlo estimate of `KL = E_{x∼π_θ}[log(π_θ/π_ref)]` is `log(π_θ(o)/π_ref(o))`, but that has high variance and can go *negative* on a single sample, which is ugly for something that should be ≥ 0. There's a better unbiased estimator that's always non-negative. Let `t = π_ref(o)/π_θ(o)`. Then use
`D̂_KL = t − log t − 1 = π_ref(o)/π_θ(o) − log(π_ref(o)/π_θ(o)) − 1`.
Check it: this is `f(t) = t − log t − 1`, which is ≥ 0 for all `t > 0` (it's 0 at `t=1` and convex), so the estimate is never negative. And its expectation under `π_θ`: `E[t] = E_{π_θ}[π_ref/π_θ] = 1`, and `E[−log t] = E_{π_θ}[log(π_θ/π_ref)] = KL`, so `E[D̂_KL] = 1 + KL − 1 = KL`. Unbiased *and* non-negative — exactly what I want for a stable penalty. So the per-question GRPO objective is
`J(θ) = (1/G) Σ_i [ min(ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i) − β · D̂_KL(o_i) ]`,
maximized, averaged over the group and over questions.

One more practical lever on the clip. The standard clip range is symmetric and small. But notice that with such a clip, many tokens whose ratio drifts past the band get their gradient zeroed — and in long-CoT training that truncates the learning signal for a lot of tokens, degrading the model. So I want a *large* clip ratio `ε` to keep more tokens contributing gradient. Too large and training destabilizes; there's a sweet spot, and it's biased high relative to the usual PPO value. So tune `ε` up.

Let me write the GRPO step, because the per-token broadcast of the advantage and the KL estimator are easy to get wrong.

```python
import torch, torch.nn.functional as F

def group_advantages(rewards):                      # rewards: (G,)
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std() + 1e-6)        # A_i = (r_i - mean) / std

def grpo_loss(policy, ref_policy, q, outputs, old_logp, advantages, beta, eps):
    """outputs: G completions; old_logp[i]: per-token logp under π_old at sample time.
    advantages[i]: scalar A_i, broadcast to every token of output i."""
    total, n_tok = 0.0, 0
    for o, lp_old, A in zip(outputs, old_logp, advantages):
        lp_new = policy.token_logprobs(q, o)        # (T,) current policy
        lp_ref = ref_policy.token_logprobs(q, o)    # (T,) frozen reference
        ratio  = torch.exp(lp_new - lp_old)         # ρ per token

        unclipped = ratio * A
        clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * A
        surrogate = torch.min(unclipped, clipped)   # PPO-style clip, same A for all tokens

        # K3 KL estimator: t - log t - 1 with t = π_ref/π_θ, always >= 0, unbiased
        t  = torch.exp(lp_ref - lp_new)
        kl = t - (lp_ref - lp_new) - 1.0            # = t - log t - 1

        total += (surrogate - beta * kl).sum()      # maximize; KL added in the LOSS, not the reward
        n_tok += o_len(o)
    return -(total / n_tok)                          # negate to minimize
```

```python
def rl_step(policy, old_policy, ref_policy, questions, G, temp, max_len, beta, eps):
    for q in questions:
        outs = old_policy.sample(q, n=G, temperature=temp, max_len=max_len)
        rs   = [accuracy_reward(q, o) + format_reward(o) for o in outs]   # rule-based, equal weight
        adv  = group_advantages(rs)
        old_logp = [old_policy.token_logprobs(q, o) for o in outs]
        loss = grpo_loss(policy, ref_policy, q, outs, old_logp, adv, beta, eps)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

So I run this directly on the pretrained base model, with the minimal think/answer template and rule-based reward, no SFT first. Call the result the zero model. And the striking thing is what happens during training: with nothing but "get the answer right" as the signal, the model's accuracy on hard math climbs dramatically, and — unprompted — its *responses get longer*. It starts spending more tokens thinking. Inside those tokens, behaviors I never coded for appear: it reflects, it re-derives, it explores alternative approaches, it catches its own mistakes. There's a point where it spontaneously starts writing "wait" and re-examining a step. I gave it no reasoning template beyond the tags; the reflection and verification *emerged* from optimizing answer-correctness, because longer, self-checking reasoning genuinely raises the probability of a correct final answer, and the group-relative reward rewards exactly that. This is the core finding — reasoning can be incentivized, not taught.

But the zero model has rough edges, and they're predictable consequences of having zero supervision on the *process*. Its outputs are poorly readable, and it mixes languages mid-reasoning (the base model is multilingual, and nothing in a pure correctness reward discourages switching languages). So it reasons well but communicates badly, and it's not a general assistant. I need a pipeline that keeps the emergent reasoning but fixes readability and broad alignment, without falling back into the SFT trap of capping reasoning.

Stage it. (1) *Cold start*: collect a small set — thousands — of clean, conversational, human-aligned reasoning traces and SFT the base on them, just enough to fix readability and give a sane conversational thinking format. This is a tiny amount of SFT to set *style*, not to teach reasoning; I keep it small precisely so I don't cap exploration. (2) *Reasoning RL*: run the same GRPO-with-rule-rewards as the zero model on this cold-started policy to push reasoning back up, and add a *language-consistency reward* — the fraction of the chain-of-thought that's in the target language — to stop language mixing. That reward slightly lowers raw accuracy (it's a constraint), but it makes the output usable, so it's worth it; add it directly to the reward. (3) *Rejection sampling + SFT*: once this RL model is strong, sample many completions from it, keep the good ones (rejection sampling on correctness), and combine them with *non-reasoning* data (writing, factual QA, etc.) to SFT again — now the model is both a strong reasoner and broadly capable. (4) *Second RL stage for alignment*: a final RL pass that mixes reward signals — rule-based rewards on reasoning prompts (as before) and *model-based* rewards on general prompts (a helpfulness reward model and a safety reward model, used only where there's no rule to lean on), over a diverse prompt distribution — to make it helpful and harmless while preserving reasoning. Use the model-based preference reward only in the *last* part of this stage and briefly, since a neural reward model invites reward hacking over long runs.

A couple of details that keep the long-CoT RL stable. The reference policy can drift far from the initial one over thousands of steps; to balance exploration against stability, *periodically refresh the reference* to the current policy (every few hundred steps) rather than holding the original frozen forever. And grow the maximum generation length during training (e.g. 32k → 64k tokens) as responses get longer, so the model isn't truncated mid-thought once it learns to think at length.

```python
def pipeline(base):
    # 1. cold start: tiny SFT for readability/format, NOT to teach reasoning
    policy = sft(base, cold_start_traces)            # thousands of clean conversational CoTs
    # 2. reasoning RL with rule rewards + language-consistency reward
    policy = grpo_train(policy, reasoning_prompts,
                        reward=lambda q, o: accuracy(q, o) + fmt(o) + lang_consistency(o))
    # 3. rejection-sample good outputs, mix with non-reasoning data, SFT
    data   = rejection_sample(policy, prompts) + non_reasoning_data
    policy = sft(policy, data)
    # 4. second RL: rule rewards on reasoning + model-based rewards on general prompts
    policy = grpo_train(policy, mixed_prompts,
                        reward=lambda q, o: rule_reward(q, o) if verifiable(q)
                                            else rm_helpful(o) + rm_safety(o))
    return policy
```

So the chain: SFT on human reasoning traces is expensive, biased, and caps the model at human ability while omitting the messy reflection real reasoning needs; so reward only final-answer correctness by rule (un-hackable, unlike a neural scorer) and let reasoning emerge; optimize with GRPO, which replaces PPO's costly, ill-posed critic with a *group-relative* standardized advantage `A_i = (r_i − mean)/std`, keeps the clipped surrogate (with a deliberately large clip to spare token gradients), and adds KL-to-reference in the loss via the non-negative unbiased estimator `t − log t − 1` rather than as a length-penalizing per-token reward; run this on the base model and watch long self-checking reasoning emerge from correctness alone; then, because pure-RL reasoning is unreadable and language-mixed, wrap it in a multi-stage pipeline — small cold-start SFT for style, reasoning RL with a language-consistency reward, rejection-sampling SFT with non-reasoning data, and a final mixed-reward RL stage with model-based rewards only where no rule exists — refreshing the reference and growing the length cap to keep long-CoT training stable.
