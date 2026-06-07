# AlphaZero synthesis (for Phase 2)

## Pain point / research question
- Prior game AI = two paradigms, both leaning on heavy human/domain priors:
  - Chess/shogi: handcrafted linear evaluation φ(s)^T w + alpha-beta minimax with many heuristics (quiescence, null-move/futility pruning, killer/history move ordering, transposition tables, opening books, endgame tablebases). Deep Blue, Stockfish, Elmo.
  - Go: AlphaGo = supervised init from human games + policy/value nets + MCTS with rollouts (Monte-Carlo rollouts to estimate leaf value).
- Both need decades of human knowledge. Question: can a SINGLE network learn a game from zero (tabula rasa), playing only itself, with NO human data and NO rollouts, using search to manufacture a better training target than the network alone can produce?

## Core method (derive inline)
- Single dual-head net (p, v) = f_θ(s): p = move-prob vector (p_a = Pr(a|s)), v ≈ E[z|s] scalar value. Replaces both handcrafted eval (v) and move-ordering heuristics (p).
- MCTS, no rollouts. Each edge (s,a) stores N(s,a), W(s,a), Q(s,a), P(s,a).
  - Selection (PUCT): a* = argmax_a [ Q(s,a) + U(s,a) ], U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)).
    - Q exploits; U explores: weighted by prior P (network's belief) and by 1/(1+N) so less-visited high-prior moves get the bonus; √(Σ_b N) keeps exploring as parent visits grow.
  - Expansion: at a leaf, evaluate (p,v) = f_θ(s_L). v REPLACES rollouts. Init children N=W=Q=0, P=p_a.
  - Backup: walk path back, for each edge: N += 1; W += v; Q = W/N. Two-player: flip sign of v each ply (value is from mover's perspective).
- Improved policy from search: π(a|s_root) = N(s_root,a)^{1/τ} / Σ_b N(s_root,b)^{1/τ}. τ=1 early (proportional, exploration), τ→0 late / at eval (greedy argmax). This is the policy-improvement operator: π beats the raw prior p.
- Self-play → train loop:
  - Play a full game with both sides moving by MCTS, a_t ~ π_t. Score terminal: z ∈ {−1,0,+1} (from each state's player's perspective).
  - Store (s_t, π_t, z) for every position.
  - Train f_θ toward search policy and outcome: l = (z − v)^2 − π^T log p + c‖θ‖^2. MSE on value, cross-entropy on policy, L2 weight decay.
  - Updated θ used to generate the next self-play games. AlphaZero: single continually-updated net, no best-player gating (vs AGZ which gated on 55% win-rate eval).
- Root exploration noise (Dirichlet): P(s_root,a) = (1−ε)p_a + ε·η_a, η ~ Dir(α), ε=0.25. α scaled inversely to typical #legal moves: α = 0.3 (chess), 0.15 (shogi), 0.03 (Go). Guarantees all root moves get tried.

## Why each design choice
- Net does both p and v from one body: shared representation, and the two tasks regularize each other (multi-task); cheaper than two nets. (AGZ ablation: combined > separate.)
- v instead of rollouts: rollouts are noisy random playouts; a learned value is lower-variance and stronger once trained. MCTS averages over the net's approximation errors across the subtree, so spurious value errors cancel — unlike alpha-beta minimax which propagates the single worst error to the root. This is why MCTS+nonlinear-net works where alpha-beta+nonlinear-net failed (too slow) and MCTS+rollouts was weak.
- PUCT (not plain UCT): plain UCT explores uniformly; PUCT weights exploration by the network prior P so search concentrates on moves the net likes — "human-like" selective search (Shannon type-B). c_puct trades off.
- π ∝ N^{1/τ}: visit counts ARE the search's verdict (good moves get searched more). Raising to 1/τ controls temperature; τ=1 gives a distribution to sample for exploration in self-play, τ→0 gives the strongest move at eval. The KEY insight: this distribution is provably an improvement over p (the search is a policy-improvement operator), so it is a better training target than the net's own p.
- Loss form: cross-entropy −π^T log p drives p → π (chase the improved policy); MSE (z−v)^2 drives v → z (predict the true outcome). Together: the net is trained to imitate its own search and to predict the games that search produced. This is approximate policy iteration: MCTS = improvement, network fit = evaluation/projection.
- AlphaZero changes from AGZ: (1) optimize expected outcome (handles draws), not P(win); (2) no symmetry augmentation (chess/shogi not symmetric); (3) single continually-updated net, no best-player gating; (4) shared hyperparameters across games, only Dirichlet α scaled by move count.
- Tabula rasa: random init θ, no human games. Only domain knowledge = rules (for MCTS to step states / detect terminals / score), board encoded as planes, action encoded as planes.

## Code grounding (alpha-zero-general by Surag Nair — clean canonical impl)
- MCTS.py: Qsa, Nsa, Ns, Ps dicts; search() recursion; PUCT u = Qsa + cpuct·Ps·√Ns/(1+Nsa); leaf → nnet.predict; mask valids; backup with -v sign flip; getActionProb returns counts^{1/temp}.
- Coach.py: executeEpisode (self-play, a~pi, store (board,pi,None), assign z at end with sign per player); learn() loop (self-play → train → [AGZ-style arena gating, which AlphaZero drops]).
- Game.py / NeuralNet.py: abstract base — getNextState/getValidMoves/getGameEnded (the rules = simulator), predict → (pi, v).

## In-frame discipline
- Never name "AlphaZero"/"AlphaGo Zero" in reasoning.md/context.md as the source paper. May name AlphaZero as the method in answer.md.
- AlphaGo, AlphaGo Zero, Stockfish/alpha-beta, MCTS-with-rollouts, TD-Gammon, NeuroChess/KnightCap/Giraffe (TD-leaf) = prior-art ancestors → Background/Baselines, cite freely.
- Reasoning starts from the question, derives PUCT / π∝N^{1/τ} / loss inline, no hindsight.
