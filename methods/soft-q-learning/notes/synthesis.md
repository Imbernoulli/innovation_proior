# Soft Q-Learning — synthesis (1702.08165, Haarnoja, Tang, Abbeel, Levine, ICML 2017)

## Pain point
Deep RL methods optimize the standard expected-return objective whose optimum (under full observability) is a deterministic policy. Stochasticity is bolted on heuristically (action-space noise, high-entropy init). We sometimes WANT genuinely stochastic, multimodal policies: better exploration in multimodal reward landscapes (don't prematurely commit to one mode), compositionality / good initialization for finetuning, robustness. Need an objective that PROMOTES stochasticity, and a policy representation expressive enough to be multimodal — prior maxent methods used tabular reps or simple parametric families (Gaussian, multinomial), which can't be arbitrarily multimodal even if a NN outputs the params.

## Objective
Maximum-entropy RL: augment reward with entropy at each visited state.
π*_MaxEnt = argmax_π Σ_t E_{(s,a)~ρ_π}[ r(s,a) + α H(π(·|s)) ].
α trades entropy vs reward; 1/α can be folded into reward. Differs from greedy Boltzmann exploration (PGQ, O'Donoghue 2016) which maxes entropy only at the current step — maxent maxes entropy of the WHOLE trajectory distribution, planning to reach future high-entropy states.
Discounted version (Appendix A): π* = argmax Σ_t E_{(s,a)~ρ_π}[ Σ_{l=t}^∞ γ^{l-t} E[ r + α H | s_t,a_t ] ]. Discount on rewards+entropy, not state distribution (Thomas 2014 bias).

## Energy-based policy / soft value functions (Theorem 1)
General energy-based policy: π(a|s) ∝ exp(-E(s,a)). Set E(s,a) = -(1/α) Q_soft(s,a).
Define (set α=1 in derivations, divide rewards by α to recover general):
- Soft Q (optimal): Q*_soft(s_t,a_t) = r_t + E_{ρ_π}[ Σ_{l=1}^∞ γ^l ( r_{t+l} + α H(π*_MaxEnt(·|s_{t+l})) ) ].
- Soft V: V*_soft(s) = α log ∫_A exp( (1/α) Q*_soft(s,a') ) da'.   (log-sum-exp = "soft max")
- Optimal policy: π*_MaxEnt(a|s) = exp( (1/α)( Q*_soft(s,a) - V*_soft(s) ) ).
So (1/α)Q = negative energy, (1/α)V = log-partition. As α→0, V→hard max → standard greedy.

## Soft Bellman equation (Theorem 2)
Q*_soft(s,a) = r + γ E_{s'~p}[ V*_soft(s') ].
Proof (Appendix A.2): for π(a|s)=exp(Q^π - V^π), show Q^π(s,a)=r+γE_{s'}[ H(π(·|s')) + E_{a'~π}[Q^π(s',a')] ] = r+γE_{s'}[V^π(s')], using the identity
H(π(·|s)) + E_{a~π}[Q^π_soft(s,a)] = -KL(π || π̃) + log∫exp(Q^π_soft) where π̃ ∝ exp(Q^π_soft). Max over π is at π=π̃, giving V^π = log∫exp Q^π.

## Policy improvement theorem (Appendix A.1)
Given π, define π̃(·|s) ∝ exp(Q^π_soft(s,·)). Then Q^{π̃}_soft ≥ Q^π_soft everywhere. Proof: one-step lookahead, H(π)+E_π[Q] ≤ H(π̃)+E_{π̃}[Q] (since π̃ is the maximizer of entropy+value, equivalently minimizes the KL), then telescope/iterate forever. Policy iteration π_{i+1}∝exp(Q^{π_i}) converges to π_∞ ∝ exp(Q^{π_∞}); optimal policy must have energy-based form → proves Theorem 1.

## Soft Q-iteration (Theorem 3) + contraction
Fixed point:
- Q(s,a) ← r + γ E_{s'~p}[ V(s') ]    (∀ s,a)
- V(s) ← α log ∫_A exp( (1/α) Q(s,a') ) da'   (∀ s)
Operator T Q(s,a) = r + γ E_{s'}[ log∫exp Q(s',a') da' ] is a γ-contraction in sup-norm (Appendix A.2): if ε=||Q1-Q2||_∞ then log∫exp Q1 ≤ ε + log∫exp Q2 and ≥ -ε+..., so ||TQ1-TQ2||_∞ ≤ γε. Unique fixed point. (Fox et al. 2015 G-learning presented same.)

## Soft Q-learning (practical, Section 3.2)
Two intractabilities: integral in V; sampling from π∝exp(Q).
1) V as importance-sampled expectation:
   V^θ_soft(s) = α log E_{a'~q_{a'}}[ exp((1/α)Q^θ_soft(s,a')) / q_{a'}(a') ],  q_{a'} arbitrary positive density.
2) Bellman error via the identity g1=g2 ∀x ⟺ E_{x~q}[(g1-g2)²]=0:
   J_Q(θ) = E_{s~q_s, a~q_a}[ ½ ( Q̂^{θ̄}_soft(s,a) - Q^θ_soft(s,a) )² ],
   with target Q̂^{θ̄}_soft(s,a) = r + γ E_{s'~p}[ V^{θ̄}_soft(s') ], θ̄ = delayed target params.
   q_s,q_a positive; use on-policy rollouts of π∝exp((1/α)Q). For q_{a'}: uniform (simple, scales poorly in high-dim) or current policy (unbiased).

## Approximate sampling — amortized SVGD (Section 3.3)
Want sampler a = f^φ(ξ;s), ξ~N(0,I), induced π^φ(a|s), minimizing
   J_π(φ;s) = KL( π^φ(·|s) || exp( (1/α)(Q^θ_soft(s,·) - V^θ_soft) ) ).
SVGD (Liu & Wang 2016): the steepest-descent direction in unit ball of RKHS of kernel κ to reduce KL to target p∝exp(...) is
   Δf^φ(·;s) = E_{a~π^φ}[ κ(a, f^φ(·;s)) ∇_{a'} Q^θ_soft(s,a')|_{a'=a} + α ∇_{a'} κ(a', f^φ(·;s))|_{a'=a} ].
(first term = attraction toward high-Q regions weighted by kernel; second = repulsive term spreading particles — α scales it because target is (1/α)Q so ∇log p has 1/α; written with α multiplying the κ-gradient when energy is Q not Q/α.)
Amortized SVGD (Wang & Liu 2016, "learning to draw samples"): treat Δf^φ as ∂J_π/∂a and chain-rule into network:
   ∂J_π/∂φ ∝ E_ξ[ Δf^φ(ξ;s) ∂f^φ(ξ;s)/∂φ ].
Empirical (Appendix C.1): two sets of particles a_i=f(ξ_i) (M, "updated"/i) and ã_j=f(ξ̃_j) (K, "fixed"/j):
   ∇̂_φ J_π(φ;s) = (1/KM) Σ_j Σ_i ( κ(a_i,ã_j) ∇_{a'}Q(s,a')|_{a_i} + ∇_{a'}κ(a',ã_j)|_{a_i} ) ∇_φ f^φ(ξ̃_j;s).

## Connections (Section 4 / Appendix B)
- DDPG (Lillicrap 2015): hard Bellman critic + backprop Q-grad into actor (à la NFQCA). Soft Q-learning actor update = DDPG actor update + the κ-repulsion term. Without it, actor estimates MAP action (single mode). So DDPG ≈ approximate (soft) Q-learning where actor is approximate maximizer → explains DDPG off-policy performance.
- Entropy-regularized policy gradient = soft Q-learning (Appendix B): parametrize π^φ(a|s)=exp(E^φ(s,a)-Ē^φ(s)), Ē=log∫exp E. Entropy-reg PG with baseline b=Ē+1 gives ∇_φ J = E[(∇E - ∇Ē)(Q̂ - E)]. Choosing energy E=Q_soft and a particular Bellman-error target Q̂=Â+V (advantage independent of grad) recovers exactly the soft Q-learning Bellman-error gradient. (Concurrent: Schulman 2017 equivalence.)

## Implementation / hyperparameters (Appendix C)
- Squashing: f outputs unbounded; squash with tanh so actions in [-1,1]; Q applies σ. Code adds squash correction log(1-a²+ε) to the log-density when computing SVGD target log p (change-of-variables for tanh).
- Density of sampled actions q_{a'}(a') = p_ξ(ξ') / |det ∂a'/∂ξ'|; Jacobian singular early → start with uniform sampling then switch to f^φ.
- Code (haarnoja/softqlearning, garage/rllab): V via reduce_logsumexp over value_n_particles uniform samples in [-1,1], minus log(N) (IS const for uniform), plus action_dim*log(2) (uniform density 1/2^d → +d log2). ys = stop_gradient(reward_scale*r + (1-term)*γ*next_value). Bellman residual = 0.5 mean (ys-Q)². reward_scale ↔ temperature (scaling reward = scaling 1/α).
- SVGD: kernel_n_particles=16 split by kernel_update_ratio=0.5 into fixed/updated; adaptive isotropic Gaussian kernel κ(a,a')=exp(-||a-a'||²/h), h=median_sq/log(Kx) (paper: h=d/(2 log(M+1)) with d=median pairwise dist → median is of squared dists in code), h≥h_min. kernel_grad = -2 diff/h * κ.
- ADAM: Q lr 1e-3, policy/sampling-net lr 1e-4. Replay 1e6, start after 10k. Minibatch 64. Two hidden layers, 200 units, ReLU. Target freeze every 1000 steps (5000 swimmer), hard copy τ=1. K=M=32 action samples (100 multigoal), K_V=50 value samples. OU noise θ=0.15 σ=0.3. α=10 multigoal, 0.1 swimmer/maze; finetune anneal α→0.001 over 20 epochs.

## Code scaffold mapping
- StochasticNNPolicy.actions_for → f^φ: concat (obs, gaussian latents) → feedforward → tanh. THIS is the sampler/actor.
- SQL._create_td_update → Q-function Bellman-error loss (Eq for V via logsumexp uniform, target ys, residual).
- SQL._create_svgd_update → amortized SVGD policy update (split particles, grad_log_p = ∇Q + squash_corr, kernel dict, action_gradients = κ·grad_log_p + κ_grad averaged, backprop via grad_ys surrogate loss).
- adaptive_isotropic_gaussian_kernel → κ and ∇κ.

## Design decisions → why
- EBM policy exp((1/α)Q): only family expressive enough for arbitrary multimodal π; any density = exp(log density).
- Soft V = α logsumexp: it's exactly the normalizer of the EBM AND the value that makes soft Bellman hold; log-sum-exp is the smooth max, recovers hard max as α→0.
- Importance sampling for V: integral intractable; IS turns it into samplable expectation, any positive proposal works.
- Squared-Bellman-error stochastic objective: the g1=g2⟺E[(g1-g2)²]=0 identity converts the ∀(s,a) fixed point into one minimizable by SGD on sampled (s,a).
- Target network θ̄: same reason as DQN — moving target destabilizes; delayed copy.
- SVGD over MCMC: MCMC too slow for online action selection; amortized sampler = O(1) forward pass, and gives actor-critic structure.
- Repulsive ∇κ term: without it particles collapse to MAP (single mode) — it's what makes the sampler cover all modes; this is the ONE term separating soft Q-learning's actor from DDPG's.
- tanh squash + correction: bounded action spaces; SVGD on R^d would saturate at boundary; squash + change-of-variables keeps density correct.
- reward_scale as temperature: scaling r by c = scaling 1/α by c (since 1/α folds into reward), convenient knob.
