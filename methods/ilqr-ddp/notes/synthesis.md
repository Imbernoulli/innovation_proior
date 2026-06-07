# Synthesis notes — DDP / iLQR

## Sources retrieved this run
- PRIMARY (Q-function form, exact eqns): Tassa, Erez, Todorov 2012 IROS "Synthesis and stabilization of complex behaviors through online trajectory optimization" — refs/tassa2012.pdf. Has Eqs (1)-(13): DP principle, Q-expansion (5a-5e), k/K (6), value update (7a-7c), regularization (9),(10a-d), improved value update (11a-c), line search ΔJ(α), accept test z (13), regularization schedule (μ_min=1e-6, Δ0=2).
- PRIMARY (iLQR origin, costate form): Li & Todorov 2004 ICINCO "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems" — refs/li_todorov2004.pdf. Derives iLQR via Hamiltonian/costate + matrix-inversion lemma, gives K=(B'S'B+R)^-1 B'S'A, Riccati-like S recursion, affine vk term. Historically the Gauss-Newton (only first derivatives of f) iterative LQR.
- BACKGROUND: Bellman DP / principle of optimality; discrete-time LQR backward Riccati (already in sibling methods/lqr and methods/mpc). DDP ancestor = Jacobson & Mayne 1970 / Mayne 1966 (named in-frame as the second-order DP idea; book unobtainable as full text this run — equations reconstructed from Tassa 2012 which states them explicitly and from the explainer).
- THIRD-PARTY explainer: studywolf blog "the iterative linear quadratic regulator method" (HTML, fetched) — confirms Q-expansion, k/K signs, V_x/V_xx update, iLQR-vs-DDP (drop f_xx,f_uu,f_ux → Gauss-Newton), LM regularization intuition.
- CANONICAL CODE: anassinator/ilqr controller.py — code/anassinator_controller.py. Backward pass computes Q_x,Q_u,Q_xx,Q_ux,Q_uu; k=-solve(Quu,Qu), K=-solve(Quu,Qux); value update in (11b),(11c) form; regularization added to V_xx inside Q_ux,Q_uu (Tassa's state-based reg, Eq 10); hessians flag toggles DDP tensor terms f_xx,f_ux,f_uu; forward pass with backtracking α (alphas=1.1**(-arange(10)**2)).
- NOTE / GAP: Jackson & Howell iLQR_Tutorial.pdf (rexlab.ri.cmu.edu) could not be downloaded — TLS/cert + curl-UA block; flagged. Covered by Tassa primary + studywolf explainer.

## Exact equations (the math that MUST be right)
Discrete dynamics x_{i+1}=f(x_i,u_i). Cost J = Σ ℓ(x_i,u_i) + ℓ_f(x_N).
Value V(x,i)=min_{U_i} J_i, with V(x,N)=ℓ_f(x_N). DP: V(x,i)=min_u[ℓ(x,u)+V(f(x,u),i+1)].
Q(δx,δu) = ℓ(x+δx,u+δu) − ℓ(x,u) + V(f(x+δx,u+δu),i+1) − V(f(x,u),i+1).
Let V' = V(·,i+1), primes = next step. Second-order coeffs:
  Q_x  = ℓ_x + f_x' V'_x
  Q_u  = ℓ_u + f_u' V'_x
  Q_xx = ℓ_xx + f_x' V'_xx f_x  (+ V'_x · f_xx)      ← tensor term DDP only
  Q_uu = ℓ_uu + f_u' V'_xx f_u  (+ V'_x · f_uu)      ← tensor term DDP only
  Q_ux = ℓ_ux + f_u' V'_xx f_x  (+ V'_x · f_ux)      ← tensor term DDP only
iLQR = DROP the three (V'_x · f_··) tensor terms → Gauss-Newton. Full DDP keeps them → Newton.
Minimize quadratic over δu (R/Q_uu ≻ 0):
  δu* = −Q_uu^{-1}(Q_u + Q_ux δx) = k + K δx,  k = −Q_uu^{-1} Q_u,  K = −Q_uu^{-1} Q_ux.
Value update (substitute δu*); two equivalent forms:
  compact: V_x = Q_x − Q_xu Q_uu^{-1} Q_u,  V_xx = Q_xx − Q_xu Q_uu^{-1} Q_ux,  ΔV = −½ Q_u Q_uu^{-1} Q_u
  k/K form (used in code, robust under reg):
    ΔV  = ½ k'Q_uu k + k'Q_u
    V_x = Q_x + K'Q_uu k + K'Q_u + Q_ux' k
    V_xx= Q_xx + K'Q_uu K + K'Q_ux + Q_ux' K   (symmetrize)
Regularization (Tassa Eq 10, state-based): add μI to V'_xx inside Q_uu, Q_ux only:
  Q̃_uu = ℓ_uu + f_u'(V'_xx+μI)f_u (+V'_x·f_uu);  Q̃_ux = ℓ_ux + f_u'(V'_xx+μI)f_x (+V'_x·f_ux).
  k=−Q̃_uu^{-1}Q_u, K=−Q̃_uu^{-1}Q̃_ux. Feedback K does NOT vanish as μ→∞ (pulls new traj toward old).
  (Classic control-based reg Eq 9: Q̃_uu=Q_uu+μI; simpler but K→0 as μ→∞.)
Forward pass (line search 0<α≤1):
  x̂_0 = x_0;  û_i = u_i + α k_i + K_i(x̂_i − x_i);  x̂_{i+1}=f(x̂_i,û_i).
  Expected reduction ΔJ(α) = α Σ k_i'Q_u(i) + (α²/2) Σ k_i'Q_uu(i)k_i.
  Accept if z=(J−Ĵ)/ΔJ(α) > c1 (0<c1<z); else shrink α and retry forward.
Regularization schedule: μ_min=1e-6, Δ0=2; back-pass non-PD Q̃_uu → increase μ (Δ←max(Δ0,Δ·Δ0), μ←max(μ_min,μΔ)) and restart back-pass; success → decrease (Δ←min(1/Δ0,Δ/Δ0), μ←μΔ or 0 if <μ_min).

## Discovery-order story (for reasoning.md)
1. Pain: LQR/MPC solve the LINEAR problem exactly via backward Riccati, but real plants f(x,u) are nonlinear; QP-MPC convexifies but for a genuinely nonlinear cost-landscape you want to exploit the special DP structure, not a generic NLP.
2. Have a nominal trajectory (x̄,ū). Want the best *correction*. Two routes: (a) Pontryagin costate two-point BVP around nominal (Li & Todorov route) — open-loop unless you assume δλ=Sδx+v; (b) Bellman cost-to-go but expanded LOCALLY around the nominal — the DDP route. The DP route gives feedback K for free.
3. Key move: don't approximate V globally (curse of dim). Quadratize the *Q-function* (the bracket inside Bellman's min) around the nominal pair, to 2nd order. Backward pass: assume V'(·) is quadratic (V'_x,V'_xx), form Q-expansion, minimize over δu → linear-in-δx correction k+Kδx, substitute back → new quadratic V → recurse. This is exactly the LQR Riccati pass, but with time-varying linearizations f_x,f_u and the cost Hessians — LQR falls out as the special case where f is already linear (f_xx=0, one iteration converges).
4. Wall: the dynamics curvature. Bellman's V'∘f means Q_xx etc. pick up a second-derivative-of-f tensor term V'_x·f_xx. Full DDP keeps it (true Newton on the trajectory). But f_xx is an n×n×n tensor — expensive and often indefinite, wrecking PD-ness of Q_uu. Drop it → Q_uu=ℓ_uu+f_u'V'_xx f_u is a Gauss-Newton Hessian (PSD by construction if ℓ_uu≻0), cheaper, and empirically nearly the full Newton step. That's iLQR.
5. Wall: Q_uu may not be PD far from optimum / quadratic model inaccurate → regularize (LM μ). Naive: Q_uu+μI; better (Tassa): add μI to V'_xx so the penalty is on state deviation, K survives μ→∞ and pulls trajectory toward the trusted nominal.
6. Wall: full step k may overshoot region where linearization valid → forward rollout with line-search α on the feedforward term only (feedback K stays full), accept by comparing actual vs expected ΔJ(α). Forward pass must RE-ROLLOUT the true nonlinear f (not the linear model) — this is what makes it a real trajectory and gives the closed-loop K its meaning.
7. Land on code = anassinator controller: forward_rollout (collect f_x,f_u, ℓ derivatives), backward_pass (_Q + k,K + V update with reg), _control (forward line search), fit loop (LM schedule).

## Design-decision → why
- Quadratize Q not V globally → keeps it finite-dim & local; feedback K drops out of the δu minimization automatically (closed-loop), unlike open-loop Pontryagin.
- Linearize dynamics each iteration (f_x,f_u time-varying) → turns nonlinear OC into a sequence of time-varying LQR subproblems; LQR = special case.
- Drop f_xx,f_uu,f_ux (iLQR) → Gauss-Newton: cheaper (no rank-3 tensor), keeps Q_uu PSD, ≈ full Newton step. Full DDP keeps them for quadratic convergence when curvature matters & is benign.
- Sign of k,K: from setting ∂Q/∂δu = Q_u+Q_uu δu+Q_ux δx=0 ⇒ δu=−Q_uu^{-1}(Q_u+Q_ux δx). Negative.
- State-based reg (μ on V'_xx) vs control-based (μ on Q_uu): state-based keeps feedback nonzero as μ→∞ → robust, ties new traj to nominal.
- Line search on α (feedforward only): full step can leave linearization's validity; α∈(0,1] backtracks; K kept at full strength so feedback still corrects rollout divergence.
- Forward pass re-rolls true f: model errors over the horizon would compound under open-loop; closed-loop û_i=u_i+αk_i+K_i(x̂_i−x_i) corrects on the fly.
- Accept test z=(J−Ĵ)/ΔJ(α)>c1: Armijo-like; ensures actual reduction tracks the model's predicted reduction.
</content>
</invoke>
