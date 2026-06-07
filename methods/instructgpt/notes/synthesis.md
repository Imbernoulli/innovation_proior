# Synthesis — InstructGPT / RLHF for instruction-following

## The pain point (research question)
- Pretrained LLMs (GPT-3) maximize next-token likelihood on web text. That objective is a *proxy*: "predict the next token a random webpage would have" ≠ "follow the user's instruction helpfully, truthfully, harmlessly."
- Symptoms of this misalignment: making up facts, ignoring explicit constraints, toxic/biased output, not actually doing what the prompt asks. Scaling the model does NOT fix this — bigger ≠ more aligned (a 175B model still misbehaves). This is the motivating diagnostic finding: capability scaling and alignment are decoupled.
- Prompting (few-shot) helps a bit but is brittle and the user has to do the work. We want the *behavior* baked into the weights.
- Goal: take a pretrained LM and a distribution of real user prompts, and produce a model whose default behavior is to follow intent. Constraint: must not destroy the pretrained capabilities (alignment tax).

## Why the obvious approaches fail (the wall sequence)
1. **Just supervised fine-tune on good demonstrations (SFT).** Collect human-written ideal responses, maximize their log-likelihood. This is behavior cloning. It works and is the necessary first step — it teaches the *format* of instruction-following. BUT: (a) demonstrations are expensive; (b) cloning is capped by demonstrator quality and by the fact that for open-ended generation there are many good answers and one demonstration; (c) the model only learns to imitate, it never learns *what makes one answer better than another*. There's no signal that response A is better than response B — only "imitate A." Validation loss overfits after 1 epoch, yet more epochs still help downstream because the real objective (human preference) is not validation loss.

2. **Optimize human judgment directly with RL.** Treat the LM as a policy, sample a response, ask a human "good or bad?", use that as reward, policy-gradient. The wall: RL is extremely sample-inefficient — millions of episodes — and you cannot put a human in that loop. Humans are slow and expensive. Querying a human per rollout is infeasible.

3. **Resolution: learn a model of the reward.** Collect human judgments *offline* into a dataset, fit a reward model (RM) r_θ(x,y) → scalar that predicts human preference, then optimize the policy against the *cheap, differentiable-to-query* RM with RL. The RM is queried billions of times for free; humans are queried only to build the RM's training set. This is the Christiano-2017 / Ziegler-2019 / Stiennon-2020 line.

## Why preferences, not scalar ratings (RM data design)
- Asking a human "rate this 1-7" is noisy and uncalibrated across labelers and across time. Asking "which of these two is better?" is far more reliable — relative comparison is a much easier and more consistent human judgment than absolute scoring.
- So the RM is trained on *comparisons*, not on absolute labels. Need a way to turn "y_w preferred over y_l" into a loss on a scalar function r_θ.

## The Bradley-Terry / pairwise loss (derive inline)
- Bradley-Terry (1952) model of paired comparisons: each item i has a latent strength; P(i beats j) = strength_i / (strength_i + strength_j). Parameterize strength_i = exp(r_i) (positive, and makes the math linear in r). Then
  P(i ≻ j) = exp(r_i)/(exp(r_i)+exp(r_j)) = 1/(1+exp(-(r_i−r_j))) = σ(r_i − r_j).
- So under BT, the probability a human prefers y_w to y_l is σ(r_θ(x,y_w) − r_θ(x,y_l)). The reward DIFFERENCE is the log-odds of preference. Absolute level of r is unidentified (only differences matter) → reward is invariant to a constant shift → must normalize later (set mean reward of demonstrations to 0).
- Maximize likelihood of observed comparisons = minimize negative log-likelihood:
  loss(θ) = − E_{(x,y_w,y_l)~D} [ log σ(r_θ(x,y_w) − r_θ(x,y_l)) ].
- This is exactly logistic regression / binary cross-entropy on the reward difference. Gradient pushes r_w up and r_l down with strength (1−σ(Δ)) — large updates when the model is wrong, vanishing when it already ranks them correctly.
- RM architecture: take the SFT model, drop the unembedding (vocab projection), add a scalar head. Initialize from a capable LM so it already "understands" text; only need to learn the preference readout. Paper uses a 6B RM (175B was unstable as the value function and far more compute; 6B was stable across LRs and gave equally strong PPO).

## The K-way ranking → batching trick (efficiency + overfitting)
- To collect comparisons fast, show a labeler K responses (K=4..9) and have them rank → C(K,2) pairwise comparisons per prompt for the price of one labeling session.
- Naive: shuffle all C(K,2) pairs into the dataset, treat each as an independent example. Problem: each completion appears in K−1 pairs, so it's seen K−1 times per epoch, and the C(K,2) comparisons from one prompt are highly correlated → the RM overfits after a single epoch.
- Fix: put all C(K,2) comparisons from one prompt into a single batch element. One forward pass per completion (K forwards) instead of C(K,2)·2 forwards → big compute saving, AND because each completion's gradient contribution is aggregated coherently within the batch element it stops overfitting → better val accuracy/loss. Loss has the 1/C(K,2) normalization to average within the prompt.
- Ties dropped. Single epoch (multiple epochs overfit fast).

## The RL stage — why PPO, why the KL penalty (derive inline)
- Now optimize policy π_φ to maximize E_{y~π_φ(·|x)}[ r_θ(x,y) ]. This is a one-step (bandit) RL problem: state = prompt x, action = whole response y, reward at end of episode = r_θ(x,y). No transitions, no discounting needed within the bandit framing (paper uses no discount, GAE λ over the token sequence treated as the trajectory for credit assignment / value baseline).
- Why not just plain REINFORCE / vanilla policy gradient? Two killers:
  (a) **High variance + destructive large updates.** Policy gradient ∇ = E[r · ∇log π] is high variance; a large step can collapse the policy (mode collapse / gibberish). Need a trust-region-ish constraint so each update doesn't move π too far. PPO's clipped surrogate gives that cheaply.
  (b) **Reward over-optimization / reward hacking.** r_θ is only an approximation of human preference, accurate near the distribution it was trained on (samples from the SFT policy). If the policy is free to move anywhere to maximize r_θ, it will find adversarial regions where r_θ is high but actual human preference is low — the RM gets exploited. The policy drifts off-distribution and the RM's score becomes meaningless.
- **The KL penalty solves (b) and helps (a).** Add a per-token penalty β·KL(π_φ || π_SFT) that keeps the RL policy close to the SFT policy (which is where the RM is reliable). Combined per-token reward used by PPO:
  reward_token = (RM score, added at the final token) − β·(log π_φ(y|x) − log π_SFT(y|x)) per token.
  Note log π_φ − log π_SFT is a single-sample estimate of the KL integrand; summed over tokens it estimates KL. β controls how far the policy may roam. Too small β → reward hacking; too large β → policy can't improve. Paper: β=0.02 optimal around 0.01–0.02; β=2.0 (100×) cripples it.
- The KL term also acts as an entropy bonus / regularizer (keeps the policy from collapsing to a single high-reward response), addressing part of (a).
- Full PPO objective being maximized:
  objective(φ) = E_{(x,y)~π_φ}[ r_θ(x,y) − β log(π_φ(y|x)/π_SFT(y|x)) ] + γ E_{x~D_pretrain}[ log π_φ(x) ].
- Value function initialized from the RM (it already maps text→scalar, good init for predicting return). PPO clipped surrogate with ratio = π_φ/π_old, clip 0.2, GAE λ, 1 inner epoch per batch, EMA of weights.

## The alignment tax and the pretraining-mix fix (γ term, ptx)
- Diagnostic finding: after PPO-RLHF, performance *regresses* on public NLP benchmarks (SQuAD, DROP, HellaSwag, WMT translation) vs the original GPT-3. The alignment procedure costs raw capability — an "alignment tax." Bad: incentivizes deploying unaligned-but-capable models.
- First idea: just increase β (stay closer to SFT)? Tested: even β=2.0 does NOT fix the regressions, and it tanks the reward. KL-to-SFT is the wrong anchor — SFT itself has drifted from pretraining, and KL doesn't preserve the specific capabilities.
- Fix that works: mix the *pretraining* gradient back in during PPO. Add γ·E_{x~D_pretrain}[log π_φ(x)] — keep maximizing pretraining log-likelihood on real pretraining data alongside the RL objective. This directly anchors the model to the pretraining distribution where those capabilities live. γ=27.8; need γ≳20 to recover DROP/SQuAD on 1.3B; one γ works across sizes; Likert insensitive to γ. Implementation: compute PPO grad and pretraining grad in consecutive steps, accumulate both. Use 8× pretraining examples vs RL episodes (4× → pretrain loss creeps up; 32× → better Likert but much slower; 8× is the compromise). This variant = PPO-ptx = InstructGPT.

## Load-bearing ancestors (for context.md baselines/background)
- **GPT-3 (Brown 2020):** the pretrained base; in-context/few-shot learning; the misalignment-with-intent observation. Architecture used for all models.
- **Deep RL from human preferences (Christiano 2017):** the original "learn a reward model from human pairwise comparisons, optimize policy with RL" template, on robotics/Atari. Established preferences > demonstrations for hard-to-specify rewards, and the RM-from-comparisons loss.
- **Fine-tuning LMs from human preferences (Ziegler 2019):** first port of that template to language models (stylistic continuation, summarization), introduced the per-token KL-to-pretrained penalty for LMs.
- **Learning to summarize from human feedback (Stiennon 2020):** the direct methodological parent — SFT → RM (BT loss) → PPO with KL penalty, on summarization; studied reward over-optimization vs KL strength explicitly. InstructGPT = this recipe generalized from one task (summarization) to the broad open-ended API instruction distribution + the ptx fix.
- **PPO (Schulman 2017):** the RL optimizer; clipped surrogate as a cheap trust region.
- **GAE (Schulman 2016):** advantage estimation / variance reduction for the policy gradient.
- **Bradley-Terry (1952):** the pairwise-comparison probability model underlying the RM loss.
- **Cross-task instruction tuning (FLAN, Wei 2021; T0, Sanh 2021):** the *other* school — SFT on many public NLP tasks with instructions. The baseline/contrast: improves held-out NLP tasks but the datasets don't match real user prompt diversity (classification/QA heavy), so it underperforms on the API distribution. Establishes that *what data you align on* matters more than just "instruction tuning."

## Design-decision → why table
- Three stages (SFT→RM→PPO) not one: SFT alone caps at imitation; RM turns cheap offline comparisons into a dense optimizable signal; PPO lets the policy *exceed* demonstrations by optimizing the preference signal. Each stage removes the previous stage's ceiling.
- Comparisons not ratings: relative judgment more reliable than absolute.
- BT σ(Δr) loss: principled ML estimator turning ordinal prefs into a scalar reward; = logistic regression; reward identified up to a shift.
- RM init from a capable LM, scalar head replacing unembedding: reuse language understanding, learn only the readout.
- 6B RM (not 175B): 175B unstable as value function + huge compute; 6B stable, equally strong.
- All C(K,2) per prompt as one batch element: avoids correlated-pair overfitting + K vs C(K,2) forward-pass saving.
- Normalize reward mean→0 on demonstrations: BT loss shift-invariant, fix the gauge before RL.
- PPO not vanilla PG: clipped surrogate = cheap trust region, controls variance/destructive steps.
- Per-token KL-to-SFT penalty: keep policy where RM is valid (anti reward-hacking) + entropy regularization. β≈0.02.
- Value fn init from RM: text→scalar already learned.
- ptx pretraining-mix term γ: KL-to-SFT can't fix the alignment tax (tested); anchoring to pretraining distribution can. γ=27.8, 8× data ratio.
- Bandit framing, no discount, GAE over tokens: episode = single prompt→response; per-token KL gives dense shaping; value baseline reduces variance over the token sequence.

## Canonical implementation (code grounding)
- HuggingFace TRL v0.7.10: `RewardTrainer.compute_loss` → `loss = -logsigmoid(rewards_chosen - rewards_rejected).mean()` (BT loss). `PPOTrainer`:
  - `compute_rewards(scores, logprobs, ref_logprobs, masks)`: `kl = logprob - ref_logprob`; `non_score_reward = -kl_ctl.value * kl`; `reward = non_score_reward.clone(); reward[last_token] += score`. (per-token KL penalty + terminal RM score — exactly Eq. structure.)
  - `compute_advantages(values, rewards, mask)`: GAE loop `delta = r_t + γ·V_{t+1} − V_t; lastgaelam = delta + γ·λ·lastgaelam; returns = adv + values; adv whitened`.
  - `loss(...)`: `ratio = exp(logprobs − old_logprobs); pg = max(−A·ratio, −A·clip(ratio,1−ε,1+ε)); vf = 0.5·max((v−R)², (clip(v)−R)²); loss = pg + vf_coef·vf`.
  - value head: `modeling_value_head.py` AutoModelForCausalLMWithValueHead — LM + scalar value head.
- ptx term is not in stock TRL PPOTrainer's main loss but is added by accumulating a pretraining-LM-loss gradient; will represent it in the loop.
