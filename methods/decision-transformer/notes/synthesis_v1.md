# Synthesis — Decision Transformer (paper-to-reasoning)

## Pain point / research question
Offline RL: learn a good policy from a *fixed* dataset of trajectories from arbitrary (mixed-quality) policies, no environment interaction. Hard because:
- Standard RL = TD learning (Bellman bootstrapping). Off-policy + function approximation + bootstrapping = **deadly triad** → divergence/instability (Sutton & Barto).
- Offline specifically: querying a learned Q at out-of-distribution actions → **value overestimation / extrapolation error** (Fujimoto 2019 off-policy; Levine 2020 offline survey). Fixes layer on conservatism: action constraints (BCQ/BEAR), value pessimism (CQL), pessimistic dynamics models (MOReL/MOPO). All are patches on top of bootstrapping.
- Discounting γ<1 in TD induces short-sighted behavior; sparse/delayed reward → Bellman backups propagate signal one step at a time, very slowly, prone to distractor signals (Hung 2019).

## The core reframing (the discovery)
Drop TD entirely. Treat a trajectory as a *sequence of tokens* and train a causal autoregressive model (GPT) to predict the next token — pure supervised learning (cross-entropy for discrete actions, MSE for continuous). No bootstrapping, no Bellman backup, no learned value function, no policy-gradient.

But plain next-action prediction = behavior cloning, which just copies the (mediocre) data. The missing ingredient: **condition on the return you want**. Feed *return-to-go* tokens.

### Return-to-go (RTG)
- Trajectory return at t: R_t = Σ_{t'=t..T} r_{t'} (sum of *future* rewards). Don't feed past rewards r_t — model must condition on *future desired* return.
- Define returns-to-go R̂_t = Σ_{t'=t..T} r_{t'}. Recursion: R̂_t = r_t + R̂_{t+1}; equivalently R̂_{t+1} = R̂_t − r_t.
- Trajectory representation: τ = (R̂_1, s_1, a_1, R̂_2, s_2, a_2, …, R̂_T, s_T, a_T). 3 tokens per timestep.

### GPT objective
- Causal mask: token i attends only to ≤ i. Interleave so the order is (R̂_1, s_1, a_1, R̂_2, …). The hidden state at the *state* token s_t predicts a_t. So the model learns a_t ~ p(a_t | R̂_1,s_1,a_1,…,R̂_t,s_t) — action conditioned on history AND the desired return-to-go at t.
- Loss only on action predictions (state/return prediction didn't help on the locomotion/atari benches, though permissible). Averaged over timesteps.
- Context window K timesteps = 3K tokens. K>1 matters (ablation: K=1 collapses on Pong/Breakout/Seaquest) — context lets the model identify which policy in the data distribution generated the behavior.

### Test-time conditional generation (the recursion that produces behavior)
1. Pick target return-to-go R̂_1 (set high — e.g. expert-level, or 1×/5× max-in-dataset for Atari, expert score for Gym).
2. Observe start state s_1. Feed (R̂_1, s_1) → sample a_1, execute.
3. Receive r_1. **Decrement**: R̂_2 = R̂_1 − r_1.
4. Observe s_2. Feed the whole context (R̂_1,s_1,a_1,R̂_2,s_2) → a_2. Repeat, cropping to last K timesteps.
This keeps R̂_t = "return I still want to collect from here," so the conditioning stays consistent with the training distribution at every step.

## Why does conditioning on return let it act near-optimally from mixed data?
- Training pairs every behavior with the return it actually achieved (hindsight relabeling — every trajectory is a "success" at achieving *its own* return). So the model learns the full map return→behavior, not just the average behavior.
- At test time, prompting a high return selects the "expert skill" slice of that map.
- **Stitching**: because actions are conditioned on local state + RTG and trained across all trajectory sub-segments, the model can combine sub-optimal segments into a path better than any single training trajectory (graph shortest-path example: 15.8% of generated paths are novel stitches). This is the policy-improvement-without-DP claim.
- **Credit assignment via attention**: self-attention forms direct state↔return associations (dot-product), one hop, instead of Bellman backups creeping reward back one step per iteration. → works under sparse/delayed reward (Key-to-Door; delayed D4RL). CQL (TD) collapses there.
- Avoids conservatism: TD methods optimize *against* a learned value fn, which exploits its approximation errors → need pessimism/regularization. DT never optimizes against a learned function (just supervised regression to data actions), so no overestimation to defend against.

## Architecture / design decisions → why
- **Per-modality linear embeddings** (embed_R, embed_s, embed_a: Linear to hidden_size) + LayerNorm. Visual states → DQN conv encoder (Mnih 2015) instead of linear. Why separate heads: R̂, s, a live in different spaces/scales; one shared projection would conflate them.
- **Timestep embedding added to all 3 tokens of a timestep**, learned `nn.Embedding(max_ep_len, h)`. Why not standard per-token positional: one timestep = 3 tokens; we want the model to know which env-timestep a token belongs to, not just its position in the 3K sequence. Added (not concatenated) to keep dim.
- **GPT causal mask**: enables autoregressive generation and ensures a_t depends only on past + current RTG/state, not future.
- **Predict action only**: from hidden at the *state* token (x[:,1] in gym code = state-token stream). pred_a = Linear (+Tanh for bounded continuous actions). Discrete → logits + cross-entropy.
- **Context K**: Gym K=20 (5 for Reacher, shorter goal-conditioned episodes); Atari K=30 (50 Pong). Larger model than typical RL (helps model the *distribution* of returns/policies, vs single policy).
- **No discounting** in RTG (gamma=1 for the sum) — sequence modeling doesn't need γ for convergence; γ<1 would bias toward myopia.
- **Return scaling**: divide RTG by a `scale` (e.g. 1000 in gym) so token magnitudes are reasonable for the linear embed.
- **Optimizer**: AdamW; minGPT-style decay/no-decay param split; LR warmup + cosine decay (Atari) / linear warmup (Gym); grad clip 0.25 (gym) / 1.0 (atari); weight decay 1e-4/0.1.
- Atari uses Tanh after embeddings instead of LayerNorm (minor).

## Trajectory representation choice — RTG vs raw reward
Feeding past rewards r_t would let the model condition only on what already happened; we need to steer toward a *future* outcome, so the token at time t must be the still-to-collect return R̂_t. That's why RTG, decremented online, is the right token.

## Baselines for context.md
- TD/Q-learning (Watkins; DQN Mnih 2013/2015): Bellman backup Q(s,a) ← r + γ max_a' Q(s',a'). Deadly triad.
- Offline TD with conservatism: BCQ/BEAR (Fujimoto 2019, Kumar 2019 — constrain action space to data support), CQL (Kumar 2020 — value pessimism, lower-bounds Q on OOD actions), BRAC (Wu 2019), AWR (Peng 2019). REM/QR-DQN (Agarwal 2020 / Dabney 2018) for Atari offline.
- Behavior cloning: supervised a~π(s) on data; no reward use; can't exceed data; %BC = BC on top-X% by return.
- UDRL / reward-conditioned policies (Schmidhuber/Srivastava 2019; Kumar 2019 RCP) — supervised, return-conditioned, K=1. DT = sequence-modeling generalization with long context.
- Transformer-in-RL (Parisotto 2020 stabilizing; Zambaldi 2018 relational) — architecture inside actor-critic, not a replacement.
- Trajectory Transformer (Janner 2021, concurrent) — also seq model but with state/return prediction + discretization + beam search (model-based flavor).

## Evaluation settings (no outcomes)
- Atari (Bellemare 2013 ALE): 1% DQN-replay (Agarwal 2020), 500k of 50M transitions; gamer-normalized (Hafner 2020), 100=pro, 0=random; games Breakout/Qbert/Pong/Seaquest; 3 seeds.
- OpenAI Gym / D4RL (Fu 2020): HalfCheetah/Hopper/Walker/Reacher; datasets Medium / Medium-Replay / Medium-Expert; normalized 100=expert; 3 seeds.
- Key-to-Door (Mesnard 2020): long-horizon credit assignment, binary delayed reward; trained on random walks.
- Delayed (sparse) D4RL: all reward at final step.
- Graph shortest path: 20 nodes, 1000 random walks T=10, reward -1 per step / 0 at goal.

## Code grounding
- gym/decision_transformer/models/decision_transformer.py (HF GPT2, continuous, the canonical clean version)
- gym/decision_transformer/evaluation/evaluate_episodes.py (evaluate_episode_rtg — the decrement loop)
- gym/decision_transformer/training/seq_trainer.py (MSE on masked action preds)
- atari/mingpt/model_atari.py (discrete, cross-entropy, conv encoder, minGPT)
- create_dataset.py / experiment.py get_batch (RTG via discount_cumsum gamma=1)
