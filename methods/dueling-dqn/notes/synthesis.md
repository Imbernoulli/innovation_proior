# Dueling DQN — synthesis (design-decision → why)

## Pain point
DQN/DDQN use a single-stream Q-network: conv encoder → FC → |A| outputs Q(s,a). The Q
head must learn a separate number for every action in every state. But the return is
dominated by the state value V(s); the *relative* differences between actions
(advantage) are often small and, in many states, the action barely matters (Enduro:
choice only matters when a car is in front). The single-stream net re-estimates each
action's value from scratch even in those states, and V(s) only gets a gradient through
the one action that was sampled in the TD update — so the shared notion "how good is
this state" is learned slowly and redundantly.

## Ancestors (load-bearing)
- **Q(s,a), V(s)** (Sutton & Barto): Q^π(s,a)=E[R_t|s,a,π]; V^π(s)=E_{a~π}[Q^π(s,a)].
  Optimal: V*(s)=max_a Q*(s,a); Bellman optimality Q*=E[r+γ max_a' Q*(s',a')].
- **Advantage A(s,a)=Q(s,a)−V(s)** (Baird 1993, advantage updating): relative importance
  of each action. E_{a~π}[A^π]=0; for greedy a*, A(s,a*)=0 and Q(s,a*)=V(s). Baird
  decomposed the Bellman-residual update into a V update and an A update (two update
  rules); advantage updating converged faster than Q-learning in continuous-time domains
  (Harmon 1995). Advantage learning (Harmon 1996) kept a single advantage function.
  Limitation: separate update rules / separate representations, not a single model.
- **DQN** (Mnih 2015): Q(s,a;θ) conv net, target net θ⁻, experience replay, MSE TD loss
  y = r + γ max_a' Q(s',a';θ⁻).
- **DDQN** (van Hasselt 2015): decouple selection/evaluation in target,
  y = r + γ Q(s', argmax_a' Q(s',a';θ); θ⁻), reduces overestimation.
- **Prioritized replay** (Schaul 2015): sample high-|TD-error| transitions more often;
  the dueling net is shown complementary to it.
- **Advantage in policy gradients** (Sutton 2000; Schulman GAE 2015): A used to reduce
  variance — same object, different use.

## Derivations to live out in reasoning.md
1. Decompose Q = V + A. V captures state quality (shared across actions); A captures the
   per-action relative importance. Two streams off a shared conv encoder: one head → scalar
   V(s;θ,β), one head → vector A(s,a;θ,α). Combine to one Q (keeps the Q interface, so any
   Q-learning algorithm trains it unchanged via backprop only).
2. Naive aggregator Q = V + A is **unidentifiable**: add constant c to V and subtract c
   from A → same Q. Given Q alone you cannot recover V and A. Empirically this directly
   sums causes poor performance (the two streams drift; V is not forced to be the value).
3. Fix — force a constraint pinning the gauge. Max version:
   Q = V + (A − max_a' A). Then for a* = argmax A, Q(s,a*) = V → V recovers the value,
   A recovers the advantage exactly (A(s,a*)=0). Semantically clean.
4. Mean version (used in practice):
   Q = V + (A − (1/|A|) Σ_a' A). Loses exact semantics (V and A now off by a constant —
   V = mean_a Q, not max_a Q), but more stable: with mean-subtraction the advantages only
   have to track the *mean* advantage, instead of having to compensate any change to the
   *optimal* action's advantage (the max can jump between actions). Softmax variant ≈ mean.
5. Mean/max subtraction does **not** change the relative rank of A (and hence Q), so the
   greedy/ε-greedy policy is unchanged from the naive sum; acting only needs the A stream.
6. Why faster: V(s) gets a gradient on *every* update regardless of which action was taken
   (it sits under all |A| outputs through the aggregation), so the shared state value is
   learned from every transition. Corridor experiment confirms: gap grows with |A|.

## Implementation knobs → why
- Conv: 32@8×8/s4, 64@4×4/s2, 64@3×3/s1 (same as DQN, so comparable).
- Two FC streams each 512 units; V→1 output, A→|A| outputs. ReLU between all layers.
- Aggregate via mean-subtraction (eq combo2) — implemented inside the network (forward
  pass), not a separate algorithmic step; trained by plain backprop.
- Rescale gradient entering last conv layer by 1/√2: both streams backprop into the shared
  conv trunk, doubling the gradient magnitude there; 1/√2 compensates, mildly more stable.
- Clip combined gradient norm ≤ 10 (uncommon in RL, borrowed from RNN training; "Single
  Clip" with clipping already beats "Single", so clipping is applied to all variants).
- Slightly lower learning rate than DDQN for the dueling net (helps; not used for plain
  DDQN as it can hurt).
- Trained with DDQN target (and shown to combine with prioritized replay).

## Canonical implementation
CleanRL DQN (dqn_atari.py) as the single-stream base; the dueling head replaces QNetwork.
Dueling variant = same conv trunk, split into value (512→1) and advantage (512→|A|) streams,
combined by Q = V + (A − A.mean(dim=1, keepdim=True)). This is the standard dueling-DQN
implementation across CleanRL / Stable-Baselines3 / rl-zoo.
