**Problem (from step 4).** SFT leaves GSM8K at 0.0455 — the weakest report-card row. SFT is imitation: it trains the model to reproduce one reference solution chain, but a math problem has many valid paths and a binary outcome (right final number or not). Imitation optimizes reproducing a derivation, not solving the problem, and never sees the model's own near-misses. The missing signal: of the solutions *this model* produces, reward the ones that land the answer.

**Key idea.** Reinforcement learning on GSM8K with a free, exact, programmatic reward (extract the number after `####`, compare to ground truth → 1/0; tool calls executed in the real Python sandbox). For each problem, sample a *group* of completions, score each, and push up the probability of the winning rollouts. Take GRPO as the reference and strip everything that doesn't apply at this scale/setting, landing at **REINFORCE with a group-mean baseline**:
- **keep** the group as a per-prompt baseline: advantage = reward − group-mean (zero gradient when all samples agree — nothing to teach);
- **drop the z-score** (no divide-by-std — the mean already does the variance reduction, the std is finicky here);
- **drop PPO ratio+clip** — the loop is strictly on-policy (sample → one step → re-sample), so the ratio is 1 and the clip never binds;
- **drop the KL / reference model** — the reward is an exact answer-checker, not a learnable critic that can be hacked, so there is no trust region to enforce (and it saves a whole forward pass + a second model copy);
- **keep DAPO-style token-level normalization** (divide by total valid tokens, not per-sequence — unbiased w.r.t. completion length);
- **mask** the prompt and the tool-*output* tokens (produced by the interpreter, not the model) — only reinforce the model's own decisions, exactly as SFT masked them.

**Why it works.** The model learns from its own *executed, verified* successes instead of from one imitation chain — the signal imitation structurally cannot give. The exact verifier removes the reward-hacking pressure that justifies KL/clip elsewhere, so the whole loop collapses to the simplest unbiased estimator.

**Change / code.** The advantage (group-mean baseline) and the on-policy policy-gradient objective:

```python
# Calculate the rewards for each sample (run tool calls in the sandbox, check #### answer)
rewards = []
for sample_tokens in generated_token_sequences:
    generated_text = tokenizer.decode(sample_tokens[prefix_length:])
    rewards.append(train_task.reward(conversation, generated_text))  # 1.0 if final number matches else 0.0
rewards = torch.tensor(rewards, dtype=torch.float, device=device)
# Advantage = reward minus the group mean (NO divide-by-std)
mu = rewards.mean()
advantages = rewards - mu
```

```python
# Calculate log probabilities. NLL = -logp, so negate
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)   # (B, T)
# Policy-gradient objective. ignore_index=-1 zeros out prompt / tool-output / padding tokens.
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
# Token-level normalization (DAPO-style): divide by the number of valid tokens
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
# No PPO ratio+clip (we are on policy); no KL (exact verifier, no reference model)
loss = -pg_obj
loss.backward()
```

```python
# The whole simplification, stated verbatim at the top of chat_rl.py:
# Reinforcement learning on GSM8K via "GRPO".
# I put GRPO in quotes because we actually end up with something a lot
# simpler and more similar to just REINFORCE:
# 1) Delete trust region, so there is no KL regularization to a reference model
# 2) We are on policy, so there's no need for PPO ratio+clip.
# 3) We use DAPO style normalization that is token-level, not sequence-level.
# 4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.
```
