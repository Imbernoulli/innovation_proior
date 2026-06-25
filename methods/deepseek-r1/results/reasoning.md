Let me start from what's actually limiting reasoning models, because the standard recipe has a ceiling baked in. The way we make an LLM reason is to fine-tune it on human-written chain-of-thought: curated step-by-step solutions, supervised, cross-entropy to the human target. It works, but think about what it *can't* do. The targets are human reasoning traces, and they're fixed. So the model is being trained to imitate how a person wrote down their thinking — which means (a) it's expensive, every good trace is hand-authored; (b) it inherits whatever blind spots and biases the annotators had; and (c) most importantly, it can never get *better* at reasoning than the demonstrations, because the loss only rewards matching them. If there's a smarter, non-human way to attack a problem — a weird backtrack, an unusual verification — SFT actively punishes it, because it doesn't match the human target. Worse, the human traces usually *omit* the very things good reasoning needs: people write the clean final derivation, not the false starts, the "wait, that's wrong, let me recheck," the explicit verification. So SFT on human traces may be teaching the model to *not* do the messy, effortful reasoning that actually solves hard problems.

So the question flips: instead of telling the model *how* to think, can I just reward it for *getting the right answer* and let it figure out how to think on its own? For a big class of problems I don't need a human to grade the reasoning — I need only to check the final answer. A competition math problem has a deterministic answer; I can match the model's boxed final answer against the reference. A coding problem has test cases; a compiler tells me pass or fail. So the main training signal can be a *rule*, not a judgment: correct final answer or not, plus a small structural rule that makes the answer extractable. No supervision on the reasoning process at all. That's the bet — give the model the right incentive and freedom, and see whether reasoning *emerges*.

Now, the reward is a rule and that matters for a second reason. The alternative is a learned neural reward model that scores responses. But over a long RL run, a neural reward model can be exploited: the policy may find shortcuts that fool the scorer rather than genuinely improving, especially if a process reward model tries to score intermediate steps. When the task has a reliable verifier, the safer signal is the verifier. Specifically two pieces: an accuracy reward (is the final answer right, checked by rule/compiler) and a format reward (did the model put its reasoning between `<think>...</think>` and its answer between `<answer>...</answer>`), combined with equal weight. The format reward isn't about quality — it's so the thinking is delineated and I can reliably extract the final answer to grade it. And I deliberately keep the template minimal: just "think first inside `<think>`, then answer inside `<answer>`," no content guidance, so I'm not smuggling in human reasoning priors and I can watch the model's natural progression.

Which RL algorithm? The default is PPO. But look at what PPO needs and why it hurts here. PPO trains a *value model* — a critic the same size as the policy — to estimate, at each token, the expected future reward, and uses GAE to turn that into per-token advantages. Three problems, all sharp for long reasoning. First, the critic doubles the memory and compute. Second, and worse, the critic has to predict the eventual reward from a *partial* response, which is intrinsically hard when the only reward is the final outcome — and for long chains of thought it's nearly hopeless, because the model might write something, then later reflect and *contradict* it, so the value of a half-finished reasoning chain barely means anything. Third, PPO folds the KL-to-reference penalty into the per-token reward as a dense term; since RL maximizes cumulative reward, that penalizes *cumulative* KL, which implicitly penalizes *length* — and I want the model to think *longer*, not shorter. So PPO is fighting me on the exact axis I care about.

Can I get advantages without a critic? The critic only ever served as a baseline to subtract from the reward, so the gradient pushes outputs that beat the baseline up and outputs that fall short down. For a given question I don't have an absolute baseline, but I can *sample a group* of outputs and use the group itself as the baseline. Sample `G` outputs `{o_1,…,o_G}` for question `q` from the old policy `π_{θ_old}`, get each one's reward `r_i` by rule. Then "how good is output `i`" relative to what this policy typically does on `q` is how far `r_i` sits above the group's average. Standardize within the group:
`A_i = (r_i − mean({r_1,…,r_G})) / std({r_1,…,r_G})`.

Let me make sure this does the sane thing on a concrete group before I trust it. Take `G = 4` outputs whose rule rewards came back `[1, 0, 1, 0]` — two solved the problem, two didn't. The mean is `0.5`, the population std is `0.5`, so the advantages are `[(1−0.5)/0.5, (0−0.5)/0.5, …] = [+1, −1, +1, −1]`. The two correct outputs get `+1`, the two wrong ones get `−1`, and the advantages sum to exactly `0` — a clean zero-mean signal, correct outputs reinforced, incorrect ones suppressed, with no critic anywhere. Now the edge case that worries me: what if every sampled output is correct, rewards `[1, 1, 1, 1]`? Then the std is `0`, and `(1−1)/0` is undefined. That's why I divide by `std + 1e-6`: the advantages come out `[0, 0, 0, 0]`, so a question the policy already solves every way contributes essentially no gradient. That's actually the behavior I want — no learning signal from a question that's already saturated, instead of a divide-by-zero blowing up the step — but it does mean the group has to contain *some* spread in correctness to teach anything, which will matter for how I pick training prompts.

So the group of sampled rewards *is* the baseline, and standardizing by the std normalizes the scale across questions of differing difficulty. If I compute the output probability as the sum of token log-probabilities and backpropagate through it, that scalar advantage reaches every token through the sequence log-probability, without asking a critic to assign a separate value to each partial chain. The memory overhead is gone, the partial-response value-estimation problem is gone, and I'm free to put KL where it belongs.

The policy update can keep PPO's clipped importance-ratio surrogate — it's the right tool for taking gradient steps on data sampled from a slightly stale policy without the update running away. Let `ρ_i = π_θ(o_i|q) / π_{θ_old}(o_i|q)` be the probability ratio for the whole sampled output. The clipped term is `min( ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i )`. When `A_i > 0`, pushing the ratio above `1+ε` stops helping the objective, so the policy cannot lurch too far toward a lucky completion in one update. When `A_i < 0`, pushing the ratio below `1−ε` stops helping, so the policy cannot drive a bad completion's probability down too aggressively in one update. The clip is what bounds the step.

Where does the KL go? Not into the reward — I established that penalizing cumulative per-token KL discourages length. Instead add the KL-to-reference *directly in the loss* as a separate term, `−β · D_KL(π_θ ‖ π_ref)`. This keeps the policy from drifting too far from a sane reference without coupling the penalty to response length.

I need an *estimator* of `D_KL(π_θ ‖ π_ref)` I can compute from samples — I only have sampled outputs, not the full distributions. The naive Monte Carlo estimate of `KL = E_{x∼π_θ}[log(π_θ/π_ref)]` is `log(π_θ(o)/π_ref(o))`. The trouble is that on a single sample this can come out *negative*: any output where `π_ref` happens to assign higher probability than `π_θ` gives a negative log-ratio, even though the true KL it estimates is ≥ 0. So as a per-sample penalty term it can have the wrong sign, which is both noisy and conceptually wrong for something I want to push toward zero.

There's a one-sample estimator that avoids this. Let `t = π_ref(o)/π_θ(o)` and try
`D̂_KL = t − log t − 1 = π_ref(o)/π_θ(o) − log(π_ref(o)/π_θ(o)) − 1`.
Two things I need to check: is it actually unbiased for the KL, and is it really non-negative per sample? The expectation is algebra: `E_{π_θ}[t] = E_{π_θ}[π_ref/π_θ] = Σ_o π_θ(o)·π_ref(o)/π_θ(o) = Σ_o π_ref(o) = 1`, and `E_{π_θ}[−log t] = E_{π_θ}[log(π_θ/π_ref)] = KL`, so `E[D̂_KL] = 1 + KL − 1 = KL`. Unbiased. Non-negativity is the shape of `f(t) = t − log t − 1`: `f'(t) = 1 − 1/t` is zero only at `t = 1`, where `f(1) = 0`, and `f''(t) = 1/t² > 0`, so `t = 1` is the global minimum and `f ≥ 0` everywhere on `t > 0`.

Let me put real numbers on both claims so I'm not just trusting the algebra. Take a single token with `π_θ = (0.5, 0.3, 0.2)` over three outcomes and `π_ref = (0.4, 0.4, 0.2)`. The true `KL(π_θ‖π_ref) = 0.5·log(0.5/0.4) + 0.3·log(0.3/0.4) + 0.2·log(0.2/0.2) = 0.02527`. Averaging the naive estimator `log(π_θ/π_ref)` against `π_θ` gives `0.02527` — unbiased, good — but its *per-outcome* values are `[+0.223, −0.288, 0]`: the middle outcome is negative, exactly the bad-sign case. Averaging `t − log t − 1` against `π_θ` also gives `0.02527` (same expectation, so same unbiasedness), and its per-outcome values are `[+0.0231, +0.0457, 0]` — all non-negative, as the convexity argument said. So this estimator matches the KL in expectation while never contributing a negative penalty on any one sample, which is what makes it stable to subtract every step. So the per-question GRPO objective is
`J(θ) = (1/G) Σ_i [ min(ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i) − β · D̂_KL(o_i) ]`,
maximized, averaged over the group and over questions.

One more practical lever on the clip. The clip is what decides, for each output, whether its advantage still contributes gradient or gets flattened to a constant. Let me see how much the clip width actually matters by pricing the surrogate at a concrete ratio. Suppose an output has `A > 0` and the new policy has moved its probability so that `ρ = 2` (it's now twice as likely as under `π_old`). With the standard PPO clip `ε = 0.2`, the surrogate is `min(2·A, clip(2, 0.8, 1.2)·A) = min(2A, 1.2A) = 1.2A` — clipped, so the gradient through `ρ` is killed past `1.2`. With `ε = 10`, `clip(2, −9, 11) = 2`, so the surrogate is `min(2A, 2A) = 2A` — unclipped, the gradient still flows. With the large clip the ratio has to reach `11` before the term saturates at `11A`. In long-CoT training many tokens get large per-step probability swings, and at `ε = 0.2` a lot of those outputs would be in the flattened region every step, wasting their learning signal; widening the clip keeps them live. But it can't go to infinity — with no clip at all a single output with a huge ratio dominates the update and training destabilizes. So the clip has to be deliberately large but still finite; in the first cold-started RL stage, set `ε = 10`.

The update is easy to get wrong if I confuse token log-probabilities with the sequence probability in the objective, so I compute token log-probs and immediately sum them into `log π(o|q)`.

```python
import torch

def group_advantages(rewards):                      # rewards: (G,)
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

def grpo_loss(policy, ref_policy, q, outputs, old_logp, advantages, beta, eps):
    """outputs: G completions; old_logp[i]: token log-probs under π_old."""
    total = 0.0
    for o, lp_old, A in zip(outputs, old_logp, advantages):
        lp_new = policy.token_logprobs(q, o).sum()  # log π_θ(o|q)
        lp_old = lp_old.sum()                       # log π_old(o|q)
        lp_ref = ref_policy.token_logprobs(q, o).sum()
        ratio  = torch.exp(lp_new - lp_old)         # ρ_i for the whole output

        unclipped = ratio * A
        clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * A
        surrogate = torch.min(unclipped, clipped)

        # t - log t - 1 with t = π_ref(o|q) / π_θ(o|q)
        t  = torch.exp(lp_ref - lp_new)
        kl = t - (lp_ref - lp_new) - 1.0            # = t - log t - 1

        total += surrogate - beta * kl              # maximize; KL added in the loss
    return -(total / len(outputs))                   # negate to minimize
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

So I run this directly on the pretrained base model, with the minimal think/answer template and rule-based reward, no SFT first. Call the result the zero model. The training setup needs to give it room to extend its reasoning once longer traces become useful: group size 16, rollout temperature 1, learning rate `3e-6`, KL coefficient `0.001`, 32 unique questions per step for 512 samples, reference refresh every 400 steps, and a maximum generation length that can grow from 32,768 to 65,536 tokens after the 8.2k step. Each rollout can generate 8,192 outputs, split into 16 minibatches for a single inner epoch, so the infrastructure spends its time on fresh reasoning attempts rather than repeated inner-loop fitting.

The zero model is a clean experiment, but its rough edges are predictable consequences of zero supervision on the *process*. A correctness-and-format reward does not care whether the trace is pleasant to read. A multilingual base trained heavily on Chinese and English has no reason, under pure correctness reward, to keep the chain of thought in the user's language. And a reasoning-only RL distribution does not make a general assistant for writing, factual QA, role-play, or safety-sensitive requests. I need a pipeline that keeps the RL exploration pressure while adding just enough supervised structure and preference alignment to make the model usable.

Stage it. (1) *Cold start*: I should not replace exploration with a giant human CoT corpus. Instead, sample multiple trajectories from the zero model at temperature 1, keep the ones with correct final answers and readable format, filter language mixing and repetition, use human annotators to convert seed traces into a natural first-person conversational style, use those as examples for a stronger model to rewrite and refine more traces, and then verify the generated data again with humans. This gives thousands of long CoT examples for style and readability, not a ceiling on reasoning. (2) *Reasoning RL*: run the same GRPO-with-rule-rewards on this cold-started policy, and add a *language-consistency reward* — `Num(Words_target) / Num(Words)` over the chain of thought — so the model has an explicit reason not to switch languages. Add that directly to the final reward. (3) *Rejection sampling + SFT*: once this RL model is the generator, sample many completions, keep the correct ones, filter chaotic traces with mixed languages, long paragraphs, or code blocks, and combine about 600k reasoning samples with about 200k non-reasoning samples from writing, factual QA, self-cognition, translation, software engineering, and related data. SFT on that mixture gives broad capability without giving up the RL-discovered reasoning data. (4) *Second RL stage for alignment*: run GRPO again on a mixed prompt distribution. For reasoning data, use the rule reward. For general data, use the appropriate model-based reward — helpfulness for helpfulness prompts or safety for safety prompts — plus format. Add language consistency across the batch. Keep most parameters from the first RL stage, lower rollout temperature to 0.7 because higher temperature makes this stage incoherent, and restrict general instruction data plus preference-model rewards to the final 400 of 1,700 steps so reward hacking has less room to take over.

A couple of details that keep the long-CoT RL stable. The reference policy can drift far from the initial one over thousands of steps; to balance exploration against stability, *periodically refresh the reference* to the current policy (every few hundred steps) rather than holding the original frozen forever. And grow the maximum generation length during training (e.g. 32k → 64k tokens) as responses get longer, so the model isn't truncated mid-thought once it learns to think at length.

```python
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
    # 1. cold start: zero-model samples -> filters -> human/LLM rewrite -> verification
    cold_start_traces = build_cold_start_data(zero_model, reasoning_prompts)
    policy = sft(base, cold_start_traces)
    # 2. reasoning RL with rule rewards + language-consistency reward
    policy = grpo_train(policy, reasoning_prompts,
                        reward=lambda q, o: accuracy(q, o) + fmt(o) + lang_consistency(o))
    # 3. rejection-sample good outputs, mix with non-reasoning data, SFT
    data   = rejection_sample(policy, prompts) + non_reasoning_data
    policy = sft(policy, data)
    # 4. second RL: rule rewards on reasoning + model-based rewards on general prompts
    policy = grpo_train(policy, mixed_prompts,
                        reward=lambda q, o: (rule_reward(q, o) if verifiable(q)
                                             else general_reward(q, o)) + lang_consistency(o),
                        temperature=0.7, steps=1700, general_reward_steps=400)
    return policy
```

So the chain: SFT on human reasoning traces is expensive, biased, and caps the model at human ability while omitting the messy reflection real reasoning needs; a reliable verifier can reward final-answer correctness and a minimal format rule without supervising the reasoning path; GRPO replaces PPO's costly, ill-posed critic with a *group-relative* standardized advantage `A_i = (r_i − mean)/std`, keeps the clipped surrogate with a deliberately large clip to spare gradients, and adds KL-to-reference in the loss via the non-negative unbiased estimator `t − log t − 1` rather than as a length-penalizing per-token reward; the pure-RL run starts from the base model with only the tags and those rules; then the full model keeps that exploration pressure but repairs the predictable product gaps with cold-start SFT for readable first-person style, reasoning RL with a language-consistency reward, rejection-sampling SFT with non-reasoning data, and a final mixed-reward RL stage where model-based rewards are confined to general prompts and used only near the end.
