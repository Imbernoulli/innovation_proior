SFT got the assistant to ChatCORE 0.0884, and the row I keep staring at is GSM8K at 0.0455. It nearly doubled from midtraining's 0.0250 when I poured in more epochs, but it is still the weakest thing on the report card by far, and more SFT epochs will not move it much. The reason is structural: SFT is imitation. On GSM8K I show the model one reference solution per problem — a specific chain of tool calls and text that reaches the answer — and train it to reproduce that chain token by token. But a grade-school word problem has *many* valid solution paths (different orderings of the arithmetic, different intermediate quantities, different phrasings of the same plan), so by training on one reference chain I teach the model to match *that derivation* when what I actually care about is a binary outcome — does it get the final number right. Imitation optimizes the wrong thing, and worse, the model's own failures are invisible to it: when it samples a solution that is *almost* right but flubs one tool call, SFT has nothing to say, because SFT only ever sees the gold path, never the model's near-misses. The signal I need is exactly the one imitation cannot give: *of the solutions this model actually produces, reward the ones that land the answer and discourage the ones that do not.*

I propose **reinforcement learning on GSM8K** with a free, exact, programmatic reward, implemented as a deliberately simplified GRPO that collapses to **REINFORCE with a group-mean baseline**. The reward is the whole appeal of doing RL on math: GSM8K answers are checkable, so I extract the number after the `####` marker from the model's completion, compare it to the ground truth, and assign reward 1 if they match and 0 otherwise — no learned reward model, no human labels, no preference data, the task itself is the verifier. The loop primes the model at `<|assistant_start|>` and lets it *generate* its own solution, sampling several completions per problem with temperature so I see a spread of attempts; each completion's tool calls run through the real Python sandbox, so the model is graded on *executed* arithmetic, not on what it claims the calculator said; each completion is scored 0/1; and then I push up the probability of the tokens in the winners and down those of the losers. That is the policy gradient — the objective is $\mathbb{E}[r\cdot\log\pi(\text{tokens})]$, whose gradient nudges the model toward its own successful rollouts.

The design work is deciding *how* to do that policy gradient. GRPO, the modern reasoning-model recipe, is the obvious reference: sample a *group* of completions per prompt, use the group to estimate a baseline, compute each completion's advantage relative to that baseline, and do a PPO-style clipped update with a KL penalty toward a frozen reference policy. I take it apart and ask, at *this* scale and setting, which parts I actually need, because each carries a cost and an assumption and several do not apply. The baseline I keep wholesale. A raw policy gradient $r\cdot\nabla\log\pi$ has enormous variance with 0/1 rewards whose absolute scale drifts as the model improves, and GRPO's fix is perfect here: the *group itself* is the baseline. Having sampled, say, 16 completions for a problem, their mean reward is a free per-prompt baseline, and the advantage of a completion is its reward minus that group mean —

$$A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j.$$

A completion better than the group average gets a positive advantage (push up), worse gets negative (push down), and if all 16 fail or all 16 succeed the advantages are zero and the problem contributes no gradient — exactly right, since a problem the model already always gets or never gets has nothing to teach this step.

What I strip away is everything else. GRPO further normalizes the advantage to a z-score, $(r-\mu)/\sigma$, rescaling each problem to unit variance, but that amplifies the gradient on near-boundary problems (some succeed, some fail → small $\sigma$) and handles trivially-easy or trivially-hard problems oddly; at this scale that normalization is finicky, and the mean-subtraction already does the essential variance reduction, so I drop the divide-by-std and use $A_i = r_i - \mu$. The PPO ratio and clip I drop too, because they exist for *off-policy* correction — when you take several gradient steps on data sampled from an *old* policy, the importance ratio $\pi_{\text{new}}/\pi_{\text{old}}$ reweights the stale samples and the clip stops the update from running too far. But my loop samples a fresh group, computes the gradient, takes *one* optimizer step, and re-samples, so I am strictly **on-policy**: the data came from the very policy I am updating, the ratio is identically 1, and the clip never binds — the entire ratio+clip machinery is dead weight. The KL penalty to a reference model I drop as well. GRPO inherits it from RLHF to keep the policy from drifting into degenerate text that games a *learned* reward model, but my reward is an exact answer-checker on executed code, not a learnable critic that can be hacked: the only way to earn reward is to actually produce the right number, so the trust region's justification is gone, and the KL term would cost a whole extra forward pass through a frozen reference plus a second copy of the model in memory — meaningful on this budget.

What remains after stripping the z-score, the ratio+clip, and the KL is essentially REINFORCE with a group-mean baseline — far less machinery than GRPO, with the same justification at this scale and with an exact verifier. One detail I do keep, from the DAPO-style recipes, is **token-level** loss normalization rather than sequence-level. If I averaged the per-sequence loss, long completions would have their per-token gradient diluted and short ones amplified, biasing the model's length behavior; instead I sum the advantage-weighted token log-probs across the batch and divide by the *total number of valid tokens*, so every token is treated evenly regardless of which completion it came from. The masking carries straight over from how conversations are rendered: when I score a rollout, the only tokens that receive gradient are the ones the *model* produced and is responsible for — its solution text and its tool *calls*. The prompt tokens and, crucially, the *tool-output* tokens (produced by the actual Python interpreter during the rollout, not the model) are masked out via the ignore-index, exactly as in SFT. The same discipline that kept SFT from training the model to fabricate calculator results keeps RL from rewarding it for the interpreter's tokens; I only reinforce the model's own decisions — when to call the tool, what expression to write, how to use the result — never the deterministic tool reply.

A few practical choices the loop forces: I sample a group of 16 completions per problem with temperature and top-k so there is real diversity to baseline over; I run them in sub-batches to fit memory; I generate with a KV-cache engine for speed, since each step is dominated by *sampling*, not the gradient. The learning rate is tiny — RL on a small model is delicate — so I start at a small fraction of the base LR and ramp it down linearly, and I watch pass@k on a held-out GSM8K split during training to confirm the policy is actually improving. I run RL *only on GSM8K*, the task where the reward is cleanly programmatic, the SFT model is weakest, and the binary-outcome / many-paths structure is exactly what imitation cannot crack, so the RL report-card column carries only GSM8K. The bar is to beat SFT's 0.0455 clearly by letting the model learn from its own executed, verified successes rather than from one imitation chain — the falsifiable expectation being that pass@1 climbs over the run as probability concentrates on solution paths that land the answer, and final GSM8K accuracy clears 0.0455 by a real margin. The honest risk is that with a model this small the success rate is so low that for many problems *all* group samples fail and contribute zero gradient, so the learning signal is sparse and may plateau quickly — the RL gains are real but bounded by how often the model can stumble into a correct rollout to reinforce in the first place. A clear GSM8K gain with pass@1 visibly rising during training is what confirms the diagnosis: the SFT ceiling was imitation, not capability.

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
