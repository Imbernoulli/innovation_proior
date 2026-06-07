# Synthesis — Double DQN

## Pain point
DQN's learned action-value estimates look implausibly high — the predicted value
of the greedy policy keeps climbing far above the actual discounted return that
the same policy achieves when rolled out. On some games (Asterix, Wizard of Wor)
the estimates explode and the score collapses at the same time. Question: where
does this upward bias come from, is it harmful, and can it be removed cheaply?

## Load-bearing ancestors
- **Q-learning (Watkins 1989; Sutton 1988 TD).** Learn Q*(s,a) by bootstrapping
  toward target y = r + γ max_a Q(s',a). The max over estimated values is the
  whole game.
- **DQN (Mnih et al. 2015).** Q(s,a;θ) a CNN; two stabilizers: (i) experience
  replay (Lin 1992) — store transitions, sample minibatches uniformly, breaks
  correlation; (ii) target network θ⁻ — a periodic copy of θ every τ steps,
  used in the target so the regression target doesn't move every gradient step.
  Target: y = r + γ max_a Q(s',a;θ⁻).
- **Tabular Double Q-learning (van Hasselt 2010).** Two value tables θ, θ′.
  Each update, pick one at random; use one to SELECT the argmax action, the
  OTHER to EVALUATE it: y = r + γ Q'(s', argmax_a Q(s',a;θ)). Decoupling
  selection from evaluation removes the max-induced positive bias. Motivated by
  the observation that even environment noise inflates Q-learning's tabular
  estimates.
- **Thrun & Schwartz (1993).** First overestimation analysis. If Q errors are
  uniform in [−ε, ε], each target is overestimated by up to
  γ·ε·(m−1)/(m+1), m = #actions. Gave a toy example where this leads to
  asymptotically suboptimal policies. Attributed it to function-approximation
  generalization error.
- **Jensen's inequality.** max is convex; E[max] ≥ max E. The mechanism.

## The bias mechanism (DERIVE inline)
Target uses max_a Q(s',a;θ⁻). The same values both SELECT which action is
greedy (argmax) and EVALUATE its value. Suppose Q(s',a) = Q*(s',a) + e_a with
zero-mean noise. Then E[max_a (Q* + e_a)] ≥ max_a (Q* + E[e_a]) = max_a Q* by
Jensen (max convex). So the target is biased UPWARD, regardless of noise source
(env noise, approximation, nonstationarity). The selecting-and-evaluating with
the same noisy values is what picks out exactly the actions whose noise is most
positive.

## Lower bound (Theorem 1, with proof)
Setup: state s where all true optimal values equal, Q*(s,a)=V*(s). Estimates
Q_t unbiased on the whole: Σ_a (Q_t(s,a)−V*(s)) = 0, not all zero, with
(1/m) Σ_a (Q_t(s,a)−V*(s))² = C > 0, m ≥ 2. Then
  max_a Q_t(s,a) ≥ V*(s) + √(C/(m−1)),  tight.
And the Double estimate's lower bound on absolute error is 0.

Proof (contradiction): let ε_a = Q_t(s,a)−V*(s), Σε_a=0, Σε_a²=mC. Suppose
max_a ε_a < √(C/(m−1)). Let {ε⁺_i} be the n positive errors, {ε⁻_j} the m−n
strictly negative. If n=m then Σε_a=0 ⟹ all ε_a=0, contradicting Σε_a²=mC>0.
So n ≤ m−1. Then Σε⁺_i ≤ n·max ε⁺_i < n√(C/(m−1)), and by Σε_a=0,
Σ|ε⁻_j| = Σε⁺_i < n√(C/(m−1)), hence max_j|ε⁻_j| < n√(C/(m−1)). By Hölder,
Σ(ε⁻_j)² ≤ (Σ|ε⁻_j|)·max|ε⁻_j| < n√(C/(m−1))·n√(C/(m−1)) = n²C/(m−1).
Then Σε_a² = Σ(ε⁺_i)² + Σ(ε⁻_j)² < n·C/(m−1) + n²C/(m−1) = C·n(n+1)/(m−1).
With n ≤ m−1, n(n+1)/(m−1) ≤ (m−1)m/(m−1) = m, so Σε_a² < mC — contradicts
Σε_a² = mC. Hence max_a ε_a ≥ √(C/(m−1)).
Tight: ε_a = √(C/(m−1)) for a=1..m−1 and ε_m = −√((m−1)C). Check Σε_a=0 and
Σε_a²=(m−1)·C/(m−1)+(m−1)C = C+(m−1)C = mC. ✓
Double error 0: set Q_t(s,a_1)=V*+√(C(m−1)/m), Q_t(s,a_i)=V*−√(C/(m(m−1)))
for i>1 (satisfies constraints); the SECOND estimate Q'_t(s,a_1)=V* gives error
exactly 0 on the selected action.

## Uniform-error expectation (Theorem 2, with proof)
Q*(s,a)=V*(s) all a; errors ε_a = Q_t(s,a)−Q*(s,a) iid Uniform[−1,1]. Then
E[max_a ε_a] = (m−1)/(m+1). Proof: CDF of ε_a is (1+x)/2 on (−1,1); by
independence P(max ≤ x)=((1+x)/2)^m; density f(x)=(m/2)((1+x)/2)^{m−1};
∫_{−1}^1 x f(x) dx = [((x+1)/2)^m (mx−1)/(m+1)]_{−1}^1 = (m−1)/(m+1).
Note: lower bound DECREASES in m (artifact of needing very specific values);
the typical/expected overestimation INCREASES in m.

## Thrun bound (ancestor)
Uniform[−ε,ε] error ⟹ overestimate ≤ γ·ε·(m−1)/(m+1). (Scale Theorem 2 by ε,
multiply by γ for the discounted bootstrap.)

## The fix — Double DQN
Decouple selection from evaluation but reuse the target network as the second
estimator (no new network):
  y_DoubleDQN = r + γ Q(s', argmax_a Q(s',a;θ); θ⁻).
Online net θ SELECTS the greedy action; target net θ⁻ EVALUATES it. vs Q-learning
untangled: y_Q = r + γ Q(s', argmax_a Q(s',a;θ⁻); θ⁻) (same θ⁻ for both).
vs tabular Double-Q: replaces θ′ with θ⁻. Why it reduces but doesn't fully
remove bias: θ and θ⁻ are correlated (θ⁻ is a stale copy of θ), so the
selection and evaluation errors aren't independent; right after a target copy
θ⁻=θ and it reverts to plain Q-learning. (Tuned variant raises τ 10k→30k.)

## Code grounding (CleanRL dqn_atari)
Only change from DQN: in the no_grad target block, replace
`target_max,_ = target_network(next_obs).max(1)` with
`a* = q_network(next_obs).argmax(1); target_q = target_network(next_obs).gather(1,a*)`.
Everything else identical: CNN (32@8s4, 64@4s2, 64@3s1, FC512, FC|A|), replay
buffer 1M, batch 32, γ=0.99, target copy every τ, ε-greedy, RMSProp/Adam, reward
clip [−1,1], 4-frame stack 84x84.

## Design decisions → why
- Target network as 2nd estimator (not new net): minimal change, no extra
  params/compute, fair comparison to DQN; θ⁻ already a separate-ish estimator.
- Reuse DQN hyperparameters unchanged: isolate the effect of the target swap.
- Why bias is harmful even if "just optimism": non-uniform across states/actions
  + bootstrapping propagates wrong relative value info → bad policy. Not the same
  as optimism-in-the-face-of-uncertainty (that's pre-update on uncertain states;
  this is post-update overoptimism on apparently-certain states).
