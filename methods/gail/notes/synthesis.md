# GAIL synthesis notes

## Pain point
Imitation learning from expert demos only: learner gets sampled trajectories, no
expert queries during training, no reward signal. Two existing routes:
- **Behavioral cloning (BC, Pomerleau 1991):** supervised max-likelihood of expert
  action given state, over the demo state-action pairs. Single-timestep fit. Fails:
  covariate shift / compounding error (Ross & Bagnell 2010, Ross et al. 2011 DAgger).
  Small per-step errors push the agent off the expert's state distribution; off-distribution
  it makes bigger errors → quadratic-in-horizon regret O(T^2 ε). Needs lots of data.
- **Inverse RL (IRL, Russell 1998; Ng & Russell 2000; maxent: Ziebart 2008,2010):**
  recover a cost function that makes the expert look optimal, then run RL on it. Cost
  ranks whole trajectories → no compounding error. But: (a) expensive — RL in an inner
  loop of every cost update; (b) it yields a *cost*, not actions; you still must plan/RL.
  Why learn a cost at all if you only want actions?

Goal: an algorithm that directly outputs a **policy** from data, as if it were
RL∘IRL, but without the nested RL loop, scaling to high-dim continuous control.

## Maxent causal entropy IRL (the starting formalism)
γ-discounted infinite horizon. E_π[c] = E[Σ γ^t c(s_t,a_t)].
Causal entropy H(π) = E_π[-log π(a|s)].
IRL: maximize_c ( min_π -H(π) + E_π[c] ) - E_{π_E}[c].
RL(c) = argmin_π -H(π) + E_π[c].
Use the LARGEST cost class C = R^{S×A} (all functions), plus a closed proper convex
regularizer ψ: R^{S×A}→R̄ to avoid overfitting:
IRL_ψ(π_E) = argmax_c -ψ(c) + (min_π -H(π)+E_π[c]) - E_{π_E}[c].

## Occupancy measure (the linearizing change of variable)
ρ_π(s,a) = π(a|s) Σ_t γ^t P(s_t=s|π) = Σ_t γ^t P(s_t=s, a_t=a | π).
Then E_π[c] = Σ_{s,a} ρ_π(s,a) c(s,a) — LINEAR in ρ.
Valid set D = {ρ_π : π∈Π} = affine (Bellman flow) constraints:
  ρ≥0 and Σ_a ρ(s,a) = p_0(s) + γ Σ_{s',a} P(s|s',a) ρ(s',a) ∀s.
Bijection Π↔D (Syed 2008 Thm 2): given ρ∈D, π_ρ(a|s)=ρ(s,a)/Σ_{a'}ρ(s,a'), and π_ρ
is the unique policy with occupancy ρ. So we can optimize over ρ instead of π.
Lemma (occ-meas causal entropy): H̄(ρ) = -Σ ρ(s,a) log(ρ(s,a)/Σ_{a'}ρ(s,a')). Then
H̄ strictly concave (log-sum inequality), H(π)=H̄(ρ_π), H̄(ρ)=H(π_ρ).
Exchange lemma: L(π,c)=-H(π)+E_π[c] ⇔ L̄(ρ,c)=-H̄(ρ)+Σρc.

## Central result (Proposition: RL∘IRL_ψ = ψ*-regularized occupancy matching)
RL∘IRL_ψ(π_E) = argmin_π -H(π) + ψ*(ρ_π - ρ_{π_E}),
ψ* = convex conjugate, ψ*(x)=sup_y x^T y - ψ(y).
Proof = saddle point. Define L̄(ρ,c) = -H̄(ρ) - ψ(c) + Σ_{s,a}(ρ(s,a)-ρ_E(s,a))c(s,a).
- IRL_ψ finds c̃ = argmax_c min_ρ L̄ (one saddle coordinate).
- π_A = argmin_π -H + ψ*(ρ_π-ρ_E) = argmin_ρ max_c L̄ (expand ψ* as a max over c).
- D compact convex, R^{S×A} convex; L̄(·,c) convex (−H̄, +linear), L̄(ρ,·) concave
  (−ψ +linear) → minimax duality: min_ρ max_c = max_c min_ρ. So (ρ_A, c̃) is a saddle
  point ⇒ ρ_A ∈ argmin_ρ L̄(ρ, c̃). RL on c̃ gives ρ̃ ∈ argmin_ρ L̄(ρ,c̃). Since L̄(·,c)
  strictly convex (H̄ strictly concave), the argmin is unique ⇒ ρ_A = ρ̃ ⇒ π_A = π̃.
Interpretation: IRL = dual of occupancy matching; recovered cost = dual optimum;
RL∘IRL = recovering the primal optimum from the dual.

## Constant-ψ corollary (exact matching)
ψ const ⇒ argmin_ρ -H̄(ρ) s.t. ρ=ρ_E ∀(s,a). Costs are the dual variables (Lagrange
multipliers) on the equality constraints. Strong duality + strict convexity ⇒ recovered
ρ̃ = ρ_E exactly. But intractable: |S×A| constraints; with finite samples most ρ_E(s,a)=0,
forcing learner to never visit unseen pairs.

## Relax to a smooth distance: min_π d_ψ(ρ_π,ρ_E) - H(π),  d_ψ := ψ*(ρ_π-ρ_E).
- ψ = δ_C (indicator of a cost class C): ψ*(ρ_π-ρ_E)=max_{c∈C} E_π[c]-E_{π_E}[c] →
  **apprenticeship learning** (Abbeel-Ng 2004 with C_linear gives feature-expectation
  matching ‖E_π f - E_{π_E} f‖_2; Syed 2007/2008 with C_convex gives max-over-features).
  Ho et al. 2016 scaled this with TRPO + neural nets (policy gradient of max_c objective
  = RL grad with c* = argmax). Con: C is a low-dim linear subspace; if true cost ∉ C, no
  guarantee π=π_E. δ_C is fixed, can't adapt to data.

## GAIL regularizer ψ_GA and the JS/GAN reduction
ψ_GA(c) = E_{π_E}[g(c(s,a))] if c<0 else +∞, with g(x)=-x-log(1-e^x) for x<0 else +∞.
Penalizes c that assigns near-zero cost to expert; allows ANY negative cost (not a
finite subspace); is an average over expert data (adapts to data).
Construction (Prop riskconv): from a strictly decreasing convex surrogate φ, define
g_φ(x)=-x+φ(-φ^{-1}(-x)) on T=range(-φ), ψ_φ(c)=Σ ρ_E(s,a) g_φ(c(s,a)). Then
ψ_φ*(ρ_π-ρ_E) = -R_φ(ρ_π,ρ_E), the min surrogate risk
R_φ = Σ_{s,a} min_γ ρ_π φ(γ) + ρ_E φ(-γ). (Nguyen 2009: R_φ ↔ f-divergences.)
Logistic loss φ(x)=log(1+e^{-x}) reduces g_φ to g above and gives
ψ_GA*(ρ_π-ρ_E) = max_{D∈(0,1)^{S×A}} E_π[log D(s,a)] + E_{π_E}[log(1-D(s,a))].
This is the optimal binary-classifier (policy vs expert) negative log loss = (up to const)
Jensen-Shannon divergence D_JS(ρ_π,ρ_E) = KL(ρ_π‖(ρ_π+ρ_E)/2)+KL(ρ_E‖(ρ_π+ρ_E)/2).
Final objective: min_π ψ_GA*(ρ_π-ρ_E) - λH(π) = D_JS(ρ_π,ρ_E) - λH(π).
A true metric (squared) → can match exactly, unlike linear AL; tractable like AL.

## Algorithm (saddle point of E_π[log D] + E_{π_E}[log(1-D)] - λH(π))
Function approx: policy π_θ, discriminator D_w:S×A→(0,1).
Alternate:
1. Adam step on w to INCREASE E_{τ_i}[∇_w log D_w] + E_{τ_E}[∇_w log(1-D_w)]
   (binary cross-entropy: policy label 0, expert label 1). In canonical code D outputs
   "scores" (logits); loss = sigmoid_cross_entropy, regularized by -ent_reg*logit-Bernoulli-entropy.
2. TRPO step on θ with cost c(s,a)=log D_w(s,a) (surrogate reward = -log D, drives toward
   expert-classified regions), minus λ∇_θ H(π_θ). Q(s̄,ā)=E_{τ_i}[log D(s,a)|s_0,a_0].
   GAE (Schulman 2015b, γ=.995 λ=.97) + a fitted value baseline reduce variance.
Causal entropy gradient (Lemma entgrad): ∇_θ E_{π_θ}[-log π_θ] =
  E_{π_θ}[∇_θ log π_θ(a|s) Q_log(s,a)], Q_log(s̄,ā)=E_{π_θ}[-log π_θ|s_0,a_0]
  — i.e. just RL policy grad with per-step cost c_log=-log π_θ(a|s); the Σ_s ρ(s)Σ_a ∇π
  term vanishes since Σ_a ∇π = ∇ Σ_a π = ∇1 = 0.

Reward sign detail (canonical code): two conventions. favor_zero_expert_reward → reward
= log σ(score) = log D (≤0, →0 for expert-like); else reward = -log(1-σ(score)) (≥0,
→+∞ for expert-like). Matches truncation conventions of envs.

## Design-decision → why
- Largest cost class C=R^{S×A}: max expressiveness; ψ does the regularizing, not C.
- ψ convex on all of R^{S×A}: needed for ψ* / minimax duality; not restrictive (acts on
  full space not a parameter space).
- Occupancy measure: linearizes E_π[c] and convexifies the policy optimization → duality.
- Causal entropy H: disambiguates IRL (forces expert uniquely optimal); makes L̄ strictly
  convex so primal recoverable uniquely; as a λ-weighted policy regularizer in practice.
- ψ_GA over δ_C: data-adaptive, allows any negative cost (not a finite linear subspace),
  yields a true (JS) metric → exact matching; δ_C cannot.
- D∈(0,1), surrogate reward -log D: this is exactly the GAN discriminator; matching ρ_π
  to ρ_E ⇔ fooling D. Avoids BC compounding error (matches full visitation dist, not
  per-step) AND avoids IRL inner RL loop (single TRPO step per iter, cost from D).
- TRPO not vanilla PG: gradient of the imitation objective is noisy; KL-constrained
  natural-gradient step prevents the policy from diverging.
- Adam for D: standard GAN-style discriminator training.

## Canonical impl: openai/imitation (Jonathan Ho), Theano.
- policyopt/imitation.py: TransitionClassifier (the discriminator/reward), ImitationOptimizer
  (the GAIL loop), LinearReward (apprenticeship baselines), BehavioralCloningOptimizer.
- policyopt/rl.py: TRPO step, GAE advantage, value function.
Will reimplement faithfully (PyTorch-flavored but structurally identical) for answer/reasoning.
