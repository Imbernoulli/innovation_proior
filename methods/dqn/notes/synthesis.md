# DQN (2013 NIPS workshop) — synthesis notes

## What this version IS (critical: NOT the 2015 Nature version)
- **No target network.** The TD target uses the *same* online weights θ: y = r + γ max_a' Q(s',a';θ). (Nature 2015 adds a separate frozen θ⁻.)
- Architecture: input 84×84×4 (4 grayscale frames stacked). Conv1: 16 filters 8×8 stride 4 + ReLU. Conv2: 32 filters 4×4 stride 2 + ReLU. FC hidden: 256 ReLU units. Output: linear, one unit per action (4–18). (Nature uses 32/64/64 conv + 512 FC.)
- Optimizer: RMSProp, minibatch 32.
- Reward clipping to {-1,0,+1} (training only). Frame-skip k=4 (k=3 for Space Invaders, repeat last action on skipped frames). Replay memory = last N=1M frames, uniform sampling. ε-greedy, ε annealed 1→0.1 over first 1M frames, fixed 0.1 after. Trained 10M frames total. γ discount (0.99 standard).
- Preprocessing φ: RGB→gray, downsample 210×160 → 110×84, crop 84×84 playing area, stack last 4 frames.
- One output unit per action (single forward pass gives all Q-values) — vs. NFQ's (state,action)→scalar which needs one pass per action.

## The pain point / research question
Learn control policies **directly from high-dimensional raw sensory input (pixels)** via RL, with a single algorithm/architecture across many tasks, no hand-engineered features. Deep learning had cracked vision/speech from raw inputs, but applying it to RL hit three obstacles:
1. **Reward is sparse/noisy/delayed**, not dense labeled supervision; credit assignment over thousands of steps.
2. **Samples are highly correlated** (consecutive frames), violating the i.i.d. assumption SGD relies on.
3. **Non-stationary data distribution**: the policy changes as it learns, so the data distribution shifts under training — and the TD *targets* themselves move because they depend on the weights being trained.

## Background / load-bearing ancestors

### Q-learning (Watkins & Dayan 1992)
- Optimal action-value Q*(s,a) = E[r + γ max_a' Q*(s',a') | s,a] (Bellman optimality).
- Value iteration: Q_{i+1}(s,a) = E[r + γ max_a' Q_i(s',a')] converges to Q* (tabular).
- Q-learning = stochastic, sample-based, off-policy, model-free version: update toward r + γ max_a' Q(s',a') from a single transition. Converges tabularly under Robbins-Monro step sizes.
- **Limitation:** tabular — one entry per state. No generalization. Useless for 84×84×4 ≈ 256^28224 state space.

### Function approximation + the deadly triad
- Replace table with Q(s,a;θ). Minimize L(θ) = E[(y - Q(s,a;θ))²], y = r + γ max_a' Q(s',a';θ_old) treated as fixed target. Gradient: E[(r + γ max Q(s',·;θ_old) - Q(s,a;θ))∇_θ Q(s,a;θ)]. This is a **semi-gradient**: we differentiate Q(s,a;θ) but NOT the bootstrapped target (target's θ-dependence dropped). It is not the gradient of any fixed objective.
- **Deadly triad** (Sutton & Barto): function approximation + bootstrapping + off-policy → can diverge. Baird (1995) residual-gradient counterexample: linear TD off-policy diverges to ∞. Tsitsiklis & Van Roy (1997): TD with FA can diverge off-policy; converges on-policy (linear). This is why the field retreated to **linear** FA with provable guarantees.
- Generalization is double-edged: an update at s also moves Q at s' (shared θ), which moves the target, which can feed back and blow up.

### TD-Gammon (Tesauro 1995)
- MLP value function V(s), trained on-policy by TD(λ) from self-play. Super-human backgammon.
- **But:** follow-ups to chess/Go/checkers failed → belief it was special to backgammon (dice stochasticity smooths value fn & aids exploration; Pollack & Blair 1996). On-policy, online, every-sample update.

### Gradient-TD / GTD (Sutton, Maei et al. 2008-2010)
- Proper-gradient methods on the mean-squared projected Bellman error (MSPBE) → provably convergent off-policy with FA. Maei et al. 2009: nonlinear convergence but only for *policy evaluation* (fixed policy). Maei 2010 (Greedy-GQ): control but only *linear* FA, restricted Q-learning variant.
- **Gap:** none extended to **nonlinear control** (deep nets choosing actions). So as of this work, off-policy + nonlinear + control had no convergent recipe.

### NFQ — Neural Fitted Q (Riedmiller 2005); Lange & Riedmiller 2010 (DFQ)
- Closest prior. Same loss sequence, but **batch**: refit Q-network (RPROP) on the *entire* dataset each iteration. Cost per iteration ∝ dataset size → doesn't scale.
- DFQ: deep autoencoder learns a low-dim representation *first*, then NFQ on top → two-stage, not end-to-end.
- NFQ architecture: (state, action) → scalar, one forward pass per action.
- **Gap:** batch (not scalable to large data / online), and not end-to-end from pixels.

### Experience replay (Lin 1993)
- Store transitions, replay them to a (small) neural net. Q-learning + replay existed — but from low-dimensional engineered state, not raw pixels.

### ALE (Bellemare et al. 2013) + baselines on it
- Arcade Learning Environment: standard Atari testbed, 210×160 RGB @ 60Hz, reward = score delta.
- **Sarsa** (Bellemare 2013): linear policy on hand-engineered feature sets (best feature set reported).
- **Contingency** (Bellemare 2012): Sarsa-style + learned representation of screen regions under agent control. Both use background subtraction, treat each of 128 colors as a channel (heavy visual prior).
- **HyperNEAT** (Hausknecht 2013): evolve a net per game; exploits deterministic sequences.

## The derivation chain (insight → method)
1. Want deep net from pixels via RL. Q-learning gives a model-free, off-policy, online target. Put a CNN as Q(s,a;θ) with one output per action (cheap: all actions in one forward pass; NFQ's per-action pass is wasteful).
2. **Obstacle: correlated, non-stationary online data breaks SGD and can drive the semi-gradient TD update to oscillate/diverge** (the on-policy feedback loop: maximizing action → data skews left → distribution shifts). Linear-FA theory says off-policy + bootstrapping is the danger zone.
3. **Resolution — experience replay (Lin 1993, repurposed):** store last N transitions, sample uniform minibatches. (a) decorrelates samples → variance ↓, restores near-i.i.d.; (b) averages behavior distribution over many past policies → smooths/avoids feedback oscillation; (c) reuses each transition many times → data efficiency. **Replay forces off-policy learning** (the sample's behavior policy ≠ current θ), which is exactly why Q-learning (off-policy) is the right base and on-policy TD-Gammon-style won't do.
4. **Choose Q-learning over batch NFQ** for scalability: low constant cost per update (one minibatch SGD step), online.
5. Practical knobs: reward clipping (one LR across games, bounds error magnitude — at cost of magnitude discrimination); frame-skip (≈k× more play for free); RMSProp (per-weight adaptive LR, faster); 4-frame stack (partial observability — single frame can't show velocity/direction; fixed-length φ instead of full history → no RNN/backprop-through-1000s-of-steps).
6. Track average max-Q on a held-out fixed state set as a stable progress metric (avg reward is noisy).

## Design-decision → why table
- **CNN as Q, one output per action**: single forward pass → all Q-values; vs (s,a)-input needs one pass/action.
- **Experience replay**: decorrelation + distribution smoothing + data efficiency. Alternatives: pure online (correlated, unstable), batch refit/NFQ (doesn't scale).
- **Uniform sampling from replay**: simple; ack'd limitation vs prioritized sweeping (Moore & Atkeson 1993). No differentiation of important transitions; overwrites oldest.
- **Off-policy Q-learning** (not Sarsa/on-policy): replay samples come from old policies → must learn off-policy. max in target = learn greedy policy while behaving ε-greedy.
- **Semi-gradient (target's θ frozen per step)**: full residual gradient (Baird) is the alternative — but residual gradient is slow / biased with stochastic dynamics (double-sampling). Semi-gradient is the standard Q-learning update; the loss is written with θ_{i-1} fixed.
- **ε-greedy, ε 1→0.1**: exploration early, exploit late. Standard.
- **Reward clip {-1,0,1}**: same LR across games, bounds gradient scale. Cost: can't tell big from small rewards.
- **Frame skip 4 (3 SI)**: cheap extra experience; emulator step ≪ net forward. SI=3 so blinking lasers stay visible.
- **RMSProp**: per-parameter adaptive LR; faster than plain SGD on some games.
- **4-frame stack**: partial observability (velocity/direction need >1 frame); fixed-length φ avoids recurrent backprop over thousands of steps.
- **2 conv + 1 FC, 16/32 filters, 256 hidden**: enough capacity for the visual control task at 2013 GPU budget; trained from scratch end-to-end (no autoencoder pretraining like DFQ).
- **No target network**: this 2013 version uses the online θ in the target directly. (The stabilizing role of holding θ_{i-1} fixed is only *within* a loss-iteration definition; replay is the primary stabilizer here.)

## Canonical code grounding
- CleanRL `dqn_atari.py` (downloaded to code/) is the cleanest canonical single-file impl — but it is the **Nature 2015 variant** (target network, 32/64/64 conv, 512 FC, Adam). I will keep its overall structure (QNetwork module, replay buffer, linear ε schedule, env wrappers, training loop) but render the **2013 architecture and the no-target-network single-θ update** faithfully (16/32 conv, 256 FC, RMSProp, td_target from the same q_network under no_grad). Atari wrappers (NoopReset, MaxAndSkip=4, grayscale, resize 84, FrameStack 4, ClipReward) match the paper's preprocessing.
