# TRPO — Synthesis notes (pre–Phase 2)

## The pain point (what existed, where each fell short)

Circa 2014: policy optimization splits into (1) policy iteration, (2) policy
gradient, (3) derivative-free (CEM/CMA). Embarrassing fact: gradient-free random
search (CEM, CMA) was hard to beat on Tetris and on hand-engineered locomotion,
even though gradient methods enjoy far better oracle/sample-complexity bounds
(Nemirovski). Supervised deep learning had shown SGD scales to millions of params;
RL had not inherited that. Goal: a gradient-based policy method that (a) scales to
large nonlinear policies (neural nets, tens of thousands of params), and (b) makes
*monotonic*, *robust* progress with little hyperparameter tuning.

Why vanilla policy gradient (REINFORCE / GPOMDP, Sutton, Bartlett-Baxter,
Peters-Schaal 2008a) is not enough:
- It is `∇η = E[∇log π(a|s) A(s,a)]`, a steepest-ascent step in *Euclidean*
  parameter space. But Euclidean distance in θ is meaningless for distributions:
  the same `‖Δθ‖` can be a tiny or a catastrophic change in `π`. So step size is
  unprincipled — too big collapses the policy (one bad update destroys it), too
  small wastes samples. No guidance on step size at all.
- High variance, sensitive to learning rate / reward scaling.

Why natural policy gradient (Kakade 2002 "A Natural Policy Gradient";
Bagnell-Schneider covariant; Peters-Schaal 2008b natural actor-critic) is better
but still incomplete:
- Replace Euclidean metric with the Fisher information metric: step direction is
  `F^{-1} g` where `F` is the Fisher matrix = Hessian of KL. This is invariant to
  reparameterization, measures distance in distribution space. Big conceptual win.
- BUT it uses a *fixed* step size / fixed Lagrange penalty (1/λ tuned as a
  hyperparameter). No constraint enforced per update. So on hard problems the fixed
  step is either too aggressive (policy collapse) or too timid. Empirically (paper's
  own diagnostic ablation, the "natural gradient" baseline) it solved the easy
  locomotion tasks but could NOT learn hopping/walking gaits. And forming/inverting
  F for large nets is infeasible if done naively.

Why conservative policy iteration (Kakade & Langford 2002 "Approximately Optimal
Approximate RL", CPI) is the key theoretical ancestor but impractical:
- It gives the only known *explicit lower bound* on policy improvement. Defines
  local approximation L (uses old visitation freq) and bounds
  `η(π_new) ≥ L(π_new) − (2εγ/(1−γ)^2)α^2` — but only for *mixture* policies
  `π_new = (1−α)π_old + α π'`. Mixture policies are unwieldy; nobody uses them with
  neural nets. So the guarantee never reached practice.

Other KL-constrained-update line: REPS (Peters, Mülling, Altün 2010) constrains the
*state-action marginal* p(s,a) and needs a costly nonlinear inner optimization;
Bagnell-Schneider covariant policy search; Levine-Abbeel 2014 (KL to stay near a
learned dynamics model). All circle the same idea — keep the policy from moving too
far — without a scalable monotonic-improvement guarantee for general policies.

## The first-principles object

`η(π̃) = η(π) + E_{τ∼π̃}[ Σ_t γ^t A_π(s_t,a_t) ]`  (Kakade-Langford identity).
Rewrite as sum over states with discounted visitation ρ:
`η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)`.
Central difficulty: `ρ_{π̃}` depends on the *new* policy in a complicated way →
can't optimize directly. Local approx swaps ρ_{π̃}→ρ_π:
`L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)`.
L matches η to first order at π (same value, same gradient). But no step-size
guidance — that is exactly what the bound must supply.

## Chain of approximations theory→practice (the load-bearing arc)

1. Extend CPI's mixture bound to ALL stochastic policies: replace α (mixture coef)
   by a distance `α = D_TV^max(π,π̃)`, change constant 2ε→4ε:
   `η(π̃) ≥ L_π(π̃) − (4εγ/(1−γ)^2) α^2`, ε = max_{s,a}|A_π(s,a)|.
   Proof via coupling: α-coupled pair agrees w.p. ≥1−α; L accounts for advantage
   the first time they disagree; error is from ≥2 disagreements → O(α^2).
2. Pinsker: `D_TV(p,q)^2 ≤ D_KL(p,q)` → bound with KL:
   `η(π̃) ≥ L_π(π̃) − C D_KL^max(π,π̃)`, `C = 4εγ/(1−γ)^2`.
3. This is a minorant tight at π → Algorithm 1 (MM/minorize-maximize): maximizing
   the RHS each step guarantees `η` monotonically non-decreasing. Telescoping proof:
   η(π_{i+1}) ≥ M_i(π_{i+1}) ≥ M_i(π_i) = η(π_i).
4. Penalty C from theory is way too big (huge `1/(1−γ)^2`, worst-case ε) → steps
   microscopic. Trade the penalty for a *trust-region constraint*: maximize L s.t.
   D_KL ≤ δ. δ is interpretable (how far the policy may move).
5. `D_KL^max` (constraint at every state) is intractable (∞ constraints) → use
   *average* KL `D̄_KL^ρ = E_{s∼ρ}[KL]` (heuristic, empirically as good as max KL).
6. Sample-based: expand L; replace `Σ_s ρ` by `(1/(1−γ)) E_{s∼ρ}`; replace A by Q
   (differ by a constant, baseline-invariant); replace `Σ_a` by importance sampling
   with sampling dist q. Single-path: q=π_old, Q from trajectory returns. Vine:
   rollout set + K actions per state + common random numbers (CRN) for low-variance
   Q-differences. Final objective:
   `max_θ E_{s∼ρ_old, a∼q}[ (π_θ(a|s)/q(a|s)) Q_old(s,a) ]  s.t.  E_{s∼ρ_old}[KL] ≤ δ`.

## Solving the constrained problem (Appendix C — the practical engine)

- Linearize objective, quadraticize constraint around θ_old:
  `max g·(θ−θ_old)  s.t.  ½(θ−θ_old)^T A (θ−θ_old) ≤ δ`, A = Fisher = Hessian of
  avg KL. Lagrangian solution: search direction `x = A^{-1} g` (the natural
  gradient direction falls out). Step length from saturating the constraint:
  `½ β^2 x^T A x = δ ⟹ β = sqrt(2δ / (x^T A x))`. Update `θ = θ_old + β x`.
- Forming/inverting A is infeasible for big nets. Conjugate gradient solves `A x = g`
  using only matrix-vector products `v ↦ A v`. ~10 CG iters suffice.
- Fisher-vector product without forming A: A = J^T M J where J = ∂μ/∂θ (Jacobian of
  distribution params), M = Hessian of KL w.r.t. mean-params (the second term with
  ∂²μ/∂θ² vanishes because kl'_a = 0 at the same point). In code, the slick general
  way: `Av = ∇_θ( (∇_θ D̄_KL)·v )` — grad of (grad·v). Subsample data (~10%) for the
  FVP since it's just a metric → FVP cost ≈ gradient cost.
- Why analytic FIM (Hessian of KL) not empirical FIM (outer product of grads):
  analytic integrates over the action, doesn't depend on sampled a, no need to store
  dense Hessian / all per-sample grads; similar improvement rate (diagnostic).
- Line search: quadratic/linear approx is only local. Backtrack: try
  `θ_old + α^j β x`, j=0,1,2,…, shrink until the *true* (nonlinear) surrogate L
  improves AND true avg KL ≤ δ. Without it, occasional huge steps cause catastrophic
  collapse. This is what makes it robust where fixed-step natural gradient fails.

## Unification (the aha) — special cases of one update
- Natural policy gradient = linearize L + quadraticize KL constraint, fixed
  penalty/step (don't enforce constraint each step). TRPO = NPG + enforce δ each
  step + line search.
- Vanilla PG = use an ℓ2 trust region `½‖θ−θ_old‖^2 ≤ δ` instead of KL.
- Policy iteration = drop constraint, fully maximize L.
- Also: proximal gradient / mirror descent with entropy regularizer (KL = Bregman
  divergence of entropy).

## Design-decision → why (with rejected alternatives)
- KL **constraint** not penalty: theory's C too big → tiny steps; penalty coef hard
  to pick robustly; constraint δ is interpretable and gives consistent step sizes.
- **average** KL not max KL: max KL = one constraint per state = intractable;
  avg KL empirically ≈ max KL (max-KL ablation on cartpole).
- **Fisher/KL metric** not Euclidean: Euclidean ‖Δθ‖ is meaningless for
  distributions; KL metric is reparameterization-invariant, the natural geometry.
- **CG** not direct solve: can't form/invert A (10^4+ params); CG needs only FVPs.
- **FVP via grad-of-grad-dot-v + subsampling**: avoids dense Hessian; metric so
  subsample ok; FVP ≈ gradient cost.
- **analytic FIM** not empirical (cov of grads): integrates over action, lower
  storage, action-independent.
- **β = sqrt(2δ/x^T A x)** then **backtracking line search**: closed-form step
  saturates quadratic constraint, but approx is local → backtrack on the true
  objective+constraint to avoid catastrophic steps (the failure mode of fixed-step
  natural gradient).
- **A as advantage→Q swap, baseline/self-normalized IS**: A and Q differ by a
  state-constant → same gradient; self-normalized IS estimator implicitly subtracts
  a baseline (lower variance), no separate baseline needed.
- **CRN in vine**: variance of grad ∝ Var of Q-difference between actions; shared
  random seed correlates rollouts → Var_{+CRN}=σ1²+σ2²−2ρσ1σ2 < σ1²+σ2² whenever
  ρ>0; crucial as Δt→0 where SNR→0 otherwise.
- **damping `A→A+ηI`** (canonical impl): numerical stability of CG; small.
- **diagonal Gaussian policy (continuous), softmax categorical (discrete)**:
  state-independent log-std for continuous; factored categorical for discrete/Atari.

## Canonical implementation (grounding for final code)
OpenAI Spinning Up `trpo` (tf1) — GAEBuffer, mlp gaussian/categorical policy,
ratio*adv surrogate, flat_grad, hessian_vector_product = grad(grad(d_kl)·v),
cg(Ax,b), alpha=sqrt(2δ/x·Hx), backtracking line search (backtrack_coeff**j,
accept if kl≤δ and loss improves), separate value-fn Adam updates, damping_coeff,
cg_iters=10, delta=0.01. Code uses GAE for advantages and a learned value baseline
(modern; the original used MC returns / vine Q-estimates) — the final code mirrors
Spinning Up's structure faithfully. Scaffold for context.md = this harness hollowed
out: policy net stub, value net stub, the "compute step from g and KL-metric" slot
left as TODO, generic CG/line-search slots empty.
