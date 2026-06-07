# MuZero — synthesis notes (Phase 1.5)

## Pain point / research question
Tree-based planning (MCTS) wins in chess/Go/shogi BUT needs a perfect simulator (rules) — can't apply to domains where dynamics are unknown (robotics, Atari from pixels, real world). Model-free RL (DQN/R2D2/IMPALA/Ape-X) is SOTA in Atari but far from SOTA where precise lookahead matters (Go/chess). Goal: get BOTH — planning power without a given model.

Classic model-based RL tries to learn the model by reconstructing observations (pixels: PlaNet, SimPLe) or true env state. Two problems: (1) reconstructing pixels wastes model capacity on irrelevant detail and is intractable at scale; results lag model-free even on data-efficiency. (2) Tripartite separation of representation learning / model learning / planning — model is not optimized FOR planning, so model errors compound through the search.

## Load-bearing ancestors
- **AlphaZero (Silver et al. 2017/2018)**: MCTS + self-play policy iteration with a joint policy/value net f(s)->(p,v); uses a PERFECT simulator for (1) state transitions in tree, (2) legal-action masking, (3) terminal detection. pUCT selection; policy target = MCTS visit counts; value target = game outcome z∈{-1,0,+1}; MSE value loss, CE policy loss. Two-player, undiscounted, ±1 terminal only. LIMITATION: needs the rules.
- **DQN / model-free value learning (Mnih 2015)**: learn Q directly, no model; n-step returns, target network, replay. SOTA Atari. LIMITATION: no lookahead; high-bias/high-variance bootstrap targets; weak in precision-planning domains.
- **MCTS (Coulom 2006; Kocsis-Szepesvári UCT 2006; Rosin pUCT)**: simulate rollouts, UCB to balance explore/exploit, converges to optimal (single agent) / minimax (zero-sum). Needs a generative model to traverse.
- **n-step bootstrapping (Sutton)**: z = Σ γ^i u_{t+1+i} + γ^n V(s_{t+n}); bias/variance knob between MC and TD(0).
- **Model-based RL that predicts observations** (PlaNet Hafner; SimPLe Kaiser; world models Ha; embed-to-control): latent state-space models trained to reconstruct/predict pixels. Waste capacity; intractable; lag model-free.
- **Value-equivalence line** — the real precursors:
  - **Predictron (Silver 2017)**: abstract MDP as a hidden layer of a net; trained so cumulative reward of trajectory through abstract MDP matches real value via TD. No actions.
  - **Value Iteration Networks (Tamar 2016)**, **TreeQN (Farquhar 2018)**: abstract MDP whose planning approximates optimal value.
  - **Value Prediction Networks (Oh 2017)** — closest precursor: MDP model grounded in real actions, unrolled, trained so cumulative reward (conditioned on actions from a simple lookahead) matches real env. But: NO policy prediction, search uses only value.

## The key idea (value equivalence, taken all the way)
Don't make the latent model mean anything. Train three functions so their PREDICTIONS (policy, value, reward) match targets — nothing forces the latent state to reconstruct the observation or to equal the true env state. The model only needs to be value-equivalent: planning in it must produce the same policy improvement / value as planning in the real env would.

Three functions:
- representation h_θ: o_1..o_t -> s^0
- dynamics g_θ: (s^{k-1}, a^k) -> (r^k, s^k)  [recurrent, MDP-shaped, deterministic]
- prediction f_θ: s^k -> (p^k, v^k)

Combined: μ_θ(o_1..o_t, a^1..a^k) -> (p^k, v^k, r^k).

## MCTS in latent space (Methods/Search)
Edges store {N, Q, P, R, S}. Selection by pUCT:
a^k = argmax_a [ Q(s,a) + P(s,a)·(√(Σ_b N(s,b))/(1+N(s,a)))·(c1 + log((Σ_b N(s,b)+c2+1)/c2)) ], c1=1.25, c2=19652.
Within tree, transitions/rewards looked up from S,R tables; at leaf, expand via g (reward,state) + f (policy,value). One call to g and f per simulation.
Backup generalized to intermediate rewards + discount γ + unbounded values:
G^k = Σ_{τ=0}^{l-1-k} γ^τ r_{k+1+τ} + γ^{l-k} v^l ; Q update = running mean (N·Q + G)/(N+1), N+=1.
Value normalization: Q̄ = (Q - min)/(max - min) using running min/max in the tree (because values are unbounded, unlike AlphaZero's [0,1]/[-1,1]).
Differences from AlphaZero: learned transitions instead of simulator; legal-action masking only at root (net learns not to predict never-seen actions); no special terminal handling (terminal = absorbing state, net predicts constant). Single-agent + discounted intermediate rewards.

## Targets
- policy target: π_{t+k} = MCTS visit-count distribution at step t+k (search = policy improvement operator).
- value target: bootstrapped n-step from the SEARCH value ν: z_t = u_{t+1}+γu_{t+2}+...+γ^{n-1}u_{t+n}+γ^n ν_{t+n}. Board games: bootstrap to end = final outcome u_T (since γ=1, no intermediate reward, td_steps = max_moves). Atari: n=10.
- reward target: observed reward u_{t+k}.

## Loss (Eq 1)
l_t(θ) = Σ_{k=0}^{K} [ l^r(u_{t+k}, r^k_t) + l^v(z_{t+k}, v^k_t) + l^p(π_{t+k}, p^k_t) ] + c‖θ‖².
- l^p = π^T log p (cross-entropy) always.
- Board games: l^v = (z-q)^2 MSE; l^r = 0 (no intermediate reward).
- Atari/general MDP: l^v = φ(z)^T log q, l^r = φ(u)^T log r — categorical cross-entropy over a transformed support.
- K=5 unroll steps.

## Design decisions -> why
| Decision | Why / alternative rejected |
|---|---|
| Latent model with NO reconstruction loss | Reconstructing pixels wastes capacity on planning-irrelevant detail and is intractable; only need value-equivalence. Frees state to encode whatever helps predict v/p/r. |
| Predict reward, value, policy (the 3 planning-relevant quantities) | These are exactly what MCTS consumes at a node. VPN had no policy -> couldn't do AlphaZero-style search-based policy improvement. |
| Dynamics deterministic | Simplicity; stochastic left to future work. |
| MCTS in latent space (not naive max over k-step values) | Reuses AlphaZero's strong search-based policy-improvement operator; scales with compute; caches computation in tree. |
| Policy target = visit counts | Search is a policy-improvement operator (AlphaZero); visit distribution is the improved policy. |
| Value target = n-step bootstrap from search value | Long discounted episodes with intermediate rewards (Atari) — can't use final outcome like AlphaZero; bootstrap from ν (better than raw return) trades bias/variance. |
| Bootstrap from SEARCH value ν not net value v | ν is the improved estimate after search; stronger learning signal than raw Q-learning target (ablation: beats model-free Q-learning head). |
| Categorical (softmax over 601 support, [-300,300]) value/reward + invertible h(x)=sign(x)(√(|x|+1)-1)+εx scaling | Atari rewards/values span many orders of magnitude; cross-entropy over a transformed categorical support is more stable than MSE at variable scale (Pohlen et al. 2018). |
| pUCT with c2=19652 log term | AlphaZero's rule; controls prior vs value as visits grow. |
| Min-max normalize Q in tree | Values unbounded in general MDPs (unlike AlphaZero ±1); can't fix pUCT constants without per-game prior knowledge. Use running min/max. |
| Legal mask only at root | Inside the tree the env can't be queried; net learns not to predict unseen actions. |
| Terminal = absorbing state | No simulator to detect terminal; train net to predict constant value past end. |
| Scale dynamics-input gradient by 1/2 | Keeps total gradient into the dynamics function constant across the recurrence (BPTT through K steps). |
| Scale each head loss by 1/K | Total gradient magnitude independent of unroll length K. |
| Scale hidden state to [0,1] (min-max) | Bounds activations, matches action-input range, stabilizes BPTT. |
| 16 residual blocks (vs AZ 20), 256 planes | Same arch family as AlphaZero; slightly cheaper because called repeatedly in search. |
| Atari input: last 32 frames + 32 actions, 96x96, downsample to 6x6 | Actions encoded because Atari actions may have no visible effect; downsample for tractable latent MCTS. |
| Prioritized replay p_i=|ν_i - z_i| | Sample where search value disagrees with return; α=β=1. |
| Reanalyze (variant): re-run MCTS with latest params for fresh policy targets; target net v^- for value | Sample efficiency — reuse old trajectories with up-to-date targets. |

## Code grounding
- werner-duvaud/muzero-general (PyTorch): models.py (h/g/f as MuZeroResidualNetwork / MuZeroFullyConnectedNetwork; support_to_scalar/scalar_to_support categorical transform; hidden-state min-max scaling), self_play.py (MCTS, Node, pUCT ucb_score, backpropagate with discount+reward, MinMaxStats, visit-count action selection), trainer.py (unroll K steps, CE loss for value/reward/policy, grad scale 0.5 on hidden state via register_hook, 1/K via gradient_scale_batch), replay_buffer.py (make_target, compute_target_value = n-step bootstrap from root_values).
- Official arXiv pseudocode.py: confirms scale_gradient(hidden_state,0.5), per-step loss weight 1/len(actions), pUCT ucb_score, make_target n-step.

Note: in muzero-general, ucb_score value term uses child.reward + γ·normalize(value); pUCT pb_c = log((N+pb_c_base+1)/pb_c_base)+pb_c_init then ·√N/(child.N+1). Matches paper Eq for pUCT and backup.
