# Synthesis — Prioritized Experience Replay (1511.05952)

## Pain point / research question
Online value-based RL (DQN) stores transitions in a sliding-window replay memory of size N=1e6 and samples a minibatch of 32 *uniformly at random*, doing one update per 4 new transitions (each transition replayed ~8x). Uniform sampling replays transitions at the same frequency they were experienced, regardless of how much can be learned from them. In environments where informative transitions are rare and hidden among redundant ones (rare reward, "needle in a haystack"), uniform replay wastes compute on transitions the network already predicts well. Goal: replay important transitions more often -> learn more efficiently (fewer env interactions for same performance).

## Load-bearing ancestors
- **Experience replay (Lin 1992)**: store transitions, reuse them for multiple updates; breaks temporal correlation (i.i.d.-ish minibatches) and rescues rare experience from single-update forgetting. DQN (Mnih 2013/2015) made it work at scale with a deep net Q. Gap: uniform sampling.
- **TD error as learning-progress proxy (gen. prioritized sweeping, van Seijen & Sutton 2013)**: δ = R + γ max_a Q(S',a) - Q(S,A) measures how "surprising" a transition is — how far the current estimate is from its bootstrap target. Incremental algorithms (Q-learning, SARSA) already update *in proportion to δ*, so |δ| is a natural importance measure and is computed for free.
- **Prioritized sweeping (Moore & Atkeson 1993; Andre et al. 1998)**: in model-based planning, order Bellman backups by the magnitude of value change they'd produce; propagate updates backward from where values changed most. Idea: prioritize *which* update to do. Gap: model-based / tabular; needs a model and a priority queue over states. PER ports the idea to model-free, function-approximation, sample-based RL.
- **Q-learning (Watkins & Dayan 1992)**: off-policy TD control; update θ += η δ ∇_θ Q.
- **Double DQN (van Hasselt 2016) / Double Q (van Hasselt 2010)**: decouples action selection from evaluation in the target to reduce max-operator overestimation bias: target = R + γ Q_target(S', argmax_a Q(S',a)). This is the base agent PER plugs into.
- **Weighted importance sampling (Mahmood, van Hasselt & Sutton 2014)**: WIS is lower-variance than ordinary IS; normalizing the weights (here by 1/max_i w_i) is the WIS-flavored choice.
- **Hinton 2007**: non-uniform sampling by error with an IS correction gave ~3x speedup on MNIST — precedent that error-based sampling + IS correction works in supervised learning.

## The derivation chain
1. Motivating "Blind Cliffwalk": n states, only the all-"right" action sequence (prob 2^-n) reaches reward 1; replay memory holds all 2^(n+1)-2 transitions. An oracle that greedily picks the transition that most reduces global loss (in hindsight) gives an *exponential* speedup over uniform. So order matters enormously; we need a practical proxy for the oracle.
2. Proxy: |δ|. Greedy TD-error prioritization: store last |δ| with each transition, always replay the max, update its weights in proportion to δ, recompute its δ, new transitions get max priority (so everything is seen once). Tabular: binary heap, O(1) max, O(log N) update. Works on Cliffwalk.
3. Greedy fails three ways: (a) δ only updated for replayed transitions, so a transition with low initial δ may never be revisited (and with a sliding window, effectively never); (b) sensitive to noise spikes (stochastic reward, bootstrap approximation noise), which greedy chases; (c) collapses onto a tiny high-error subset — errors shrink slowly under FA — so the same few transitions are replayed repeatedly -> overfitting, lost diversity.
4. **Stochastic prioritization** to interpolate greedy <-> uniform: P(i) = p_i^α / Σ_k p_k^α. α=0 -> uniform; α=1 -> (almost) greedy. Monotonic in priority, non-zero prob for every transition (fixes (a) and (c), softens (b)). Two priority variants:
   - proportional: p_i = |δ_i| + ε (ε keeps p>0 so zero-error transitions still revisited)
   - rank-based: p_i = 1/rank(i) where rank is position when sorted by |δ|. P becomes a power law; robust to outliers/scale (good for (b)), guaranteed heavy tail -> diversity.
5. **Bias**: SGD estimate of an expectation requires samples drawn from the same distribution as the expectation. The Q-learning solution is the fixed point of E_uniform[δ∇Q]=0 (uniform over the replay memory). Sampling from P(i)≠1/N changes the expected update, hence changes the fixed point the estimates converge to — even with policy & state distribution fixed.
6. **IS correction**: w_i = (1/N · 1/P(i))^β. With β=1, E_P[w_i δ_i ∇Q] = Σ_i P(i)·(1/(N P(i)))·δ_i∇Q = (1/N)Σ_i δ_i∇Q = E_uniform[δ∇Q] — exact debiasing. Fold w into the update: use w_i δ_i instead of δ_i (weighted IS, not ordinary IS). Normalize by 1/max_j w_j so weights only scale *down* (stability; bounds step size). β interpolates: 0 = no correction, 1 = full.
7. **Anneal β: β0 -> 1** linearly. Early training is highly non-stationary anyway (changing policy, state dist, bootstrap targets), so a small bias is tolerable and the aggressive prioritization (large effective steps) speeds early learning; unbiasedness matters most near convergence, so push β->1 by the end. Also β and α interact: raising both prioritizes harder while correcting harder.
8. IS bonus with nonlinear FA: large steps are disruptive (first-order/Taylor approx only locally valid). Prioritization makes high-error transitions seen many times while IS shrinks the per-update gradient magnitude (smaller effective step), letting the optimizer track curvature via constant re-approximation.
9. Efficiency: cannot be O(N) per sample. Proportional -> sum-tree (parent = sum of children, leaves = p_i^α): O(log N) update and sampling. Sample minibatch of k: split [0, p_total] into k equal ranges, draw one uniform value per range, descend the tree to the leaf (find_prefixsum_idx) -> stratified, balances the minibatch across error magnitudes. Rank-based -> array-based binary heap used as an approximate sorted array (resorted every ~1e6 steps); precompute k equal-probability segments of the CDF (power law), sample one transition per segment.
10. Plug into Double DQN: only change is sampling (P instead of uniform) + w_iδ_i in the update + priority bookkeeping. New transitions inserted at max priority. One extra hyperparam tweak: reduce step-size η by 4x (prioritization picks high-|δ| -> larger typical gradients). Chosen: rank α=0.7,β0=0.5; proportional α=0.6,β0=0.4.

## Design-decision -> why table
- |δ| as priority: proxy for expected learning progress; already computed by TD update; "surprise". Alt: weight-change norm (lets adaptive optimizer handle unlearnable transitions), |δ|-derivative (handles noisy/unlearnable but higher variance, didn't beat |δ|), episodic return (neuro-inspired). All in appendix; |δ| simplest and effective.
- ε in proportional: avoid p=0 -> transition never revisited once δ=0.
- rank vs proportional: rank robust to outliers & error scale, heavy tail guarantees diversity; proportional exploits error magnitude structure (better in sparse-reward). In practice similar — DQN clips rewards & δ to [-1,1], removing the outliers rank would have protected against.
- α: interpolates uniform<->greedy; tunable aggressiveness.
- IS weights w_i=(1/(N P(i)))^β: debias the P-sampled estimator back to uniform expectation.
- normalize by 1/max w: only scale down -> stability, bounded steps; WIS-style; interacts with β annealing (as β->1 normalizer grows, effective step shrinks, like annealing η).
- anneal β0->1: bias tolerable early (non-stationary), unbiasedness matters near convergence.
- new transition = max priority: guarantee every transition seen at least once.
- random/optimistic Q init for greedy: else zero-init unrewarded transitions look like zero error, sink to bottom, never revisited.
- sum-tree (proportional) / heap-as-sorted-array (rank): O(log N) sampling+update at N=1e6.
- stratified k-segment sampling: one sample per equal-prob range -> balanced minibatch, lower variance.
- η/4: prioritization raises typical gradient magnitude.

## Canonical code (1.4)
OpenAI Baselines `segment_tree.py` (SumSegmentTree with find_prefixsum_idx, MinSegmentTree for max_weight) + `replay_buffer.py` (PrioritizedReplayBuffer: add at max_priority^α, _sample_proportional stratified, sample returns weights normalized by max_weight, update_priorities sets p^α). Proportional variant. Final code mirrors this.
</content>
