# LQR / algebraic Riccati equation — synthesis notes (pre-Phase-2)

## What pain point existed (research question, in-frame, ~1958-1960)
Servo / feedback design was dominated by frequency-domain methods (Bode/Nyquist/root-locus,
Hall, Newton-Gould-Kaiser). These are essentially SISO: you shape one loop's gain/phase by hand,
tune lead-lag compensators, read off phase/gain margins. Problems:
- No principled way to handle MIMO plants (many coupled inputs/outputs). Hand-tuning N loops with
  cross-coupling is ad hoc; no notion of *jointly optimal* gains.
- "Minimize integral-squared error" (Wiener, Hall, Newton-Gould-Kaiser) was the right *idea* but
  the formulations were limited to low-order systems and weren't a clean state-space theory.
- No constructive algorithm to compute the optimal feedback for a general n-th order linear plant.
- No guarantee the resulting controller stabilizes.
Goal: given a linear plant ẋ=Fx+Gu of arbitrary order, and a quadratic measure of "bad transient"
(state error + control effort), find the feedback control law u=k(x) that minimizes it, with a
constructive algorithm, valid for MIMO, and with a stability guarantee.

## Ancestors / load-bearing prior art (verified against the paper's own bibliography + history)
- **Classical frequency-domain / ISE design** (Wiener 1949, Hall 1943, Newton-Gould-Kaiser 1957):
  integral-squared-error minimization, but SISO, low-order, no state-space constructive algorithm.
  Kalman's preamble explicitly names Wiener[17], Hall[8], Newton-Gould-Kaiser[12] as the origin of
  "minimize integral of squared tracking error", and says the book formulation "remained
  unsatisfactory from a mathematical point of view" and "allowed application only to rather low
  order systems."
- **Calculus of variations / Carathéodory** (Carathéodory 1935, ref [11]): the Euler-Lagrange /
  Weierstrass machinery, second-variation condition L_uu ≥ 0 for a local minimum (eq 3.5). Kalman
  literally builds §4 on "well-known results ([11], Ch. 12)" — Carathéodory's "royal road":
  embed the problem in a field of extremals, get the HJ PDE. The Riccati ODE itself "had emerged
  earlier in the study of the second variations in the calculus of variations" (preamble). The
  variational route is *local* (open-loop extremal), the gap is a *feedback law* (global, closed-loop).
- **Bellman dynamic programming / principle of optimality / HJB** (Bellman, mid-1950s; ref [19]
  Bellman 1953 Stability Theory, [20] Bellman 1960 Matrix Analysis): the value function
  V°(x,t) = cost-to-go, principle of optimality → the HJB PDE V_t + min_u H = 0. Kalman's eq (4.13)
  V°_t + H(x, V°_x, t) = 0 *is* the HJB equation (he calls it Hamilton-Jacobi). Bellman's DP gives
  the feedback (closed-loop) viewpoint that the calculus of variations lacked.
- **Pontryagin maximum principle** (Pontryagin et al. 1956-1959; ref [12] Pontryagin 1959): the
  costate/adjoint ξ, Hamiltonian H = L + ξ'(Fx+Gu), necessary conditions; the conjugate variable
  ξ = V°_x (eq 4.11) is exactly the costate. PMP is the necessary-condition / open-loop two-point
  boundary value route; gives the same Hamiltonian §8 canonical equations (8.1-8.2). Gap vs Kalman:
  PMP gives open-loop u*(t); LQR wants the *closed-loop gain* k(x).
- **Wiener filtering / prediction** (Wiener 1949, ref [17]): the *dual* (estimation) problem; Kalman
  notes Problem (A)=estimation is the dual of Problem (B)=control via the duality theorem; this is the
  filtering paper's sister. Out of scope for the control derivation but the structural dual.
- **Lyapunov 2nd method** (Kalman-Bertram 1960, ref [10]; Hahn 1959, ref [18]): V° serves as a
  Lyapunov function → stability proof (§6.10). The parallel "between the calculus of variations and
  the second method of Lyapunov."

## State of the field (prevailing wisdom, ~1960)
Frequency-domain SISO design + ISE heuristics; DP and PMP just appearing (1956-59) but not yet
fused into a constructive linear-feedback algorithm. "Controllability" and "observability" did not
exist as named concepts — Kalman introduces them here and views them as his principal contribution.
The Riccati ODE was known in 2nd-variation theory but had never been used as the *algorithm* that
computes the state-feedback gain of an optimal controller for a general linear system.

## Notation reconciliation (CRITICAL for signs)
Kalman's paper:
- plant ẋ = Fx + Gu, output y = Hx (eq 2.1-2.2).
- cost (assumption A1, eq just before 6.1): L = ½(‖Hx‖²_Q + ‖u‖²_R), terminal ν = ½‖x‖²_A.
  Note the ½ and the *output*-weighting H'QH.
- conjugate variable ξ = V°_x (4.11); Hamiltonian H = L + ⟨ξ, Fx+Gu⟩ (4.12).
- value V°(x,t,t₁) = ½‖x‖²_P (6.2), P symmetric ≥ 0.
- Riccati ODE (6.3):  −dP/dt = F'P + PF − PGR⁻¹G'P + H'QH.
- optimal law (6.5): u° = R⁻¹ G' Π x   [Kalman's sign: see below].
- infinite-horizon: P̄(t)=lim_{t₁→∞} Π(t;0,t₁); steady law (6.8) u° = R⁻¹G'P̄ x.

Modern textbook convention (Tedrake/MIT underactuated, python-control, scipy) — I will DERIVE in this:
- ẋ = Ax + Bu, cost J = ∫(x'Qx + u'Ru)dt (NO ½, weight Q directly on state, i.e. H=I).
- value J*(x) = x'Sx (use S or P).
- HJB: 0 = min_u [ x'Qx + u'Ru + (∂J*/∂x)(Ax+Bu) ], with ∂J*/∂x = 2x'S.
- stationarity ∂/∂u: 2u'R + 2x'SB = 0 → u* = −R⁻¹B'Sx = −Kx,  K = R⁻¹B'S.
- substitute → ARE:  A'S + SA − SBR⁻¹B'S + Q = 0.
- finite horizon DRE: −Ṡ = A'S + SA − SBR⁻¹B'S + Q,  S(t_f)=Q_f, integrate backward.
- DARE (x[k+1]=Ax+Bu): S = Q + A'SA − A'SB(R+B'SB)⁻¹B'SA;  K=(R+B'SB)⁻¹B'SA, u=−Kx.

WHY the sign differs from Kalman's (6.5): Kalman's (6.5) reads u°=R⁻¹G'Πx (no minus) because his
Hamiltonian convention and the way ξ=V°_x is plugged into ψ (4.8) absorb the sign; but his
closed-loop matrix is F − GR⁻¹G'P (the Riccati term −PGR⁻¹G'P and §7.1 F̄=F−GR⁻¹G'P̄ confirm), and
his ½-factor cost makes V°_x = Px not 2Px. The physically meaningful, stabilizing law is
*negative* feedback u = −R⁻¹B'Sx — the minus is what makes A−BK Hurwitz. I'll derive the modern
form cleanly (J* = x'Sx so J*_x = 2Sx, no ½), land on u=−Kx, K=R⁻¹B'S, and note Kalman's H'QH /
½ bookkeeping as the same equation. Cross-check: the ARE sign A'S+SA−SBR⁻¹B'S+Q=0 matches scipy's
docstring XA+A^HX−XBR⁻¹B^HX+Q=0 and python-control care() exactly.

## The derivation chain (what reasoning.md must walk, in discovery order)
1. Pain: SISO hand-tuning, no MIMO-optimal, no algorithm, no stability guarantee. Want u=k(x).
2. Pick the objective: penalize transient state error AND control effort → quadratic
   J=∫(x'Qx+u'Ru)dt. Why quadratic: (a) ISE heritage; (b) Q,R symmetric PSD/PD encode the
   tradeoff knob; (c) quadratic + linear dynamics is the one case that closes in feedback form;
   (d) L_uu=2R>0 strict convexity in u (Kalman eq 3.5) guarantees a unique minimizer.
3. Two routes on the table — calculus of variations / PMP (open-loop, costate, two-point BVP) vs
   Bellman DP (closed-loop value function). Walk the variational route first: Euler-Lagrange,
   costate ξ, Hamiltonian, canonical equations (8.1-8.2) — get an open-loop u*(t), need to
   re-solve a TPBVP for every new x₀. Wall: that's not a *feedback law*; we want u=k(x) computable
   online from the current state.
4. Switch to DP / principle of optimality. Define cost-to-go V°(x,t). Principle of optimality →
   HJB: V_t + min_u [ x'Qx+u'Ru + V_x'(Ax+Bu) ] = 0.  (Kalman 4.13.)
5. The ANSATZ that closes it: guess V quadratic, V(x,t)=x'S(t)x (because cost is quadratic, dynamics
   linear — by symmetry the cost-to-go should be quadratic). Then V_x = 2Sx.
6. Inner min over u: it's an *unconstrained quadratic in u* (R≻0 convex) → set gradient 0:
   2Ru + 2B'Sx = 0 → u* = −R⁻¹B'Sx. THE AHA: the optimal control is LINEAR state feedback,
   u*=−Kx, K=R⁻¹B'S — falls out of the algebra, not assumed.
7. Substitute u* back into HJB; the x'(...)x must vanish for all x → the matrix equation:
   finite horizon: −Ṡ = A'S+SA−SBR⁻¹B'S+Q (DRE), S(t_f)=Q_f.
8. Infinite horizon (t₁→∞): if controllable, Π(t;0,t₁) converges to constant P̄, Ṡ→0 → the
   ALGEBRAIC Riccati equation A'S+SA−SBR⁻¹B'S+Q=0. Time-invariant constant gain K=R⁻¹B'S.
   (Kalman §6.6 existence of the limit under complete controllability.)
9. Why controllability matters: §5 — completely controllable iff Gramian W>0 iff rank[G,FG,...,Fⁿ⁻¹G]=n.
   It guarantees the infinite-horizon cost is finite (you can drive any x to 0 with finite energy),
   so the ARE solution exists/limit exists.
10. Why the closed loop is STABLE — non-obvious; Kalman explicitly: "often assumed (tacitly and
    incorrectly) that a system with optimal control law is necessarily stable." §6.10: under
    controllability + observability (+ Q,R bounded), V° is a Lyapunov function (V°>0, V̇°<0 along
    optimal motion) → A−BK Hurwitz. Must DERIVE this Lyapunov argument, not assert.
11. Solving the ARE numerically: §8 — the Hamiltonian/canonic equations (8.1-8.2), transition
    matrix Θ, P(t)=[Θ21+Θ22 P(t1)][Θ11+Θ12 P(t1)]⁻¹. Modern: stable invariant subspace of the
    2n×2n Hamiltonian matrix H=[[A, −BR⁻¹B'],[−Q, −A']] → Schur/eigen method (scipy forms the
    "extended hamiltonian matrix pencil"). This is the bridge to solve_continuous_are.
12. Land on code: scipy.linalg.solve_continuous_are(A,B,Q,R) → S; K=R⁻¹B'S; u=−Kx; apply to
    quadrotor (decoupled subsystems, Q,R as the tradeoff knob).

## Design decisions → why (table)
- Quadratic cost (not |x|, not |x|^4): convex in u (unique min), matches ISE heritage, gives
  closed-form linear feedback, Q/R are the interpretable tradeoff knobs. L_uu=2R>0 (eq 3.5).
- Q PSD, R PD (strictly): R≻0 needed so R⁻¹ exists and the u-min is well-posed/strictly convex
  (otherwise free control → ill-posed). Q≥0 enough (you don't need to penalize every state).
- Quadratic value ansatz V=x'Sx: the only ansatz that makes HJB algebraic for linear-quadratic;
  motivated by "quadratic cost + linear dynamics → quadratic cost-to-go" symmetry.
- Negative feedback sign: u=−Kx, the minus is what places A−BK in LHP. (Kalman's +sign is a
  convention artifact of his ξ=V_x / ½-cost bookkeeping; closed loop is A−BR⁻¹B'S either way.)
- Infinite vs finite horizon: finite → time-varying gain K(t) from backward DRE; infinite →
  constant gain (time-invariant), the steady ARE solution. Infinite-horizon needs controllability
  for the limit to exist & be stabilizing.
- Controllability requirement: ensures finite infinite-horizon cost and existence of stabilizing P.
- Observability (of Q^{1/2} i.e. H): ensures the closed loop is actually stable, not just that the
  cost is finite (a cheap-but-unobserved unstable mode could have finite cost yet blow up — that's
  why detectability/observability is the extra hypothesis in §6.10).
- Discrete DARE: same logic with sum instead of integral; the (R+B'SB)⁻¹ appears because the
  one-step min over u[k] now sees S through B'SB. K=(R+B'SB)⁻¹B'SA.
- ARE solved via Hamiltonian matrix stable invariant subspace (Schur), not by iterating the ODE —
  Kalman §8 transition-matrix formula is the ancestor; scipy's pencil method is the modern form.

## Quadrotor example (grounded, sundw2014/Quadrotor_LQR/3D_quadrotor.py)
Linearize about hover (φ=θ=0, thrust=mg). Decouples into 4 subsystems:
- X: state [x, ẋ, θ(pitch), θ̇], A has ẋ-row coupling to pitch via +g (a tilt of pitch accelerates
  x by g·θ), B=[0,0,0,1/Ix] (pitch torque). Ay symmetric with −g. Z: [z,ż], B=[0,1/m]. Yaw:[ψ,ψ̇],
  B=[0,1/Iz]. For each: Q=I with Q[0,0]=10 (weight position), R=[1], solve CARE, K=R⁻¹B'X, u=−Kx
  applied as K·(reference − state). The g-coupling is the load-bearing physics: you steer
  horizontal position by commanding a tilt, exactly the cascaded attitude-then-position structure.
- code uses K = inv(R)*(B.T*X), eig(A−B*K) for closed-loop poles. Faithful canonical pattern.

## Sources (retrieved & read this run)
- Kalman 1960 "Contributions to the Theory of Optimal Control", Bol. Soc. Mat. Mexicana 5:102-119
  (read in full, pages 4-20 of the reprint scan; §2-8 + appendix).
  URL: https://www.ee.iitb.ac.in/~belur/ee640/optimal-classic-paper.pdf
- MIT Underactuated Robotics Ch.8 LQR (Tedrake) — modern sign convention, DRE, DARE.
  URL: https://underactuated.mit.edu/lqr.html
- python-control statefbk.lqr/dlqr + mateqn.care/dare (gain & ARE form).
- scipy.linalg._solvers solve_continuous_are / solve_discrete_are (equation + pencil method).
- sundw2014/Quadrotor_LQR/3D_quadrotor.py (quadrotor decoupled LQR).
