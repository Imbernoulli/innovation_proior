# Synthesis — TOPP (Time-Optimal Path Parameterization)

## Pain point
Given a robot arm and a *fixed geometric path* q(s), s∈[0,1] (already collision-free,
from a planner), traverse it as fast as possible without exceeding actuator limits
(torque bounds, and/or joint velocity/acceleration bounds). The path geometry is locked;
only the *timing* s(t) is free. Want the minimum-time time-scaling.

## Key reduction (Bobrow/Shin&McKay 1985; Pham 2014/2018)
- Fix q(s). A time parameterization is an increasing scalar s:[0,T]→[0,s_end].
- q̇ = q'(s) ṡ, q̈ = q'(s) s̈ + q''(s) ṡ².  (' = d/ds)
- Substitute into rigid-body dynamics M(q)q̈ + q̇ᵀC(q)q̇ + g(q) = τ:
  τ = [M q'] s̈ + [M q'' + q'ᵀ C q'] ṡ² + g(q)
  = a(s) s̈ + b(s) ṡ² + c(s).   LINEAR in (s̈, ṡ²).
- Torque bounds τ_min ≤ τ ≤ τ_max → for each constraint row i:
  a_i(s) s̈ + b_i(s) ṡ² + c_i(s) ≤ 0  (generalized 2nd-order form, polytope C(s)).
- Phase plane: state (s, ṡ). Let x = ṡ². Then for fixed s, ṡ (i.e. fixed x), each row
  gives a linear bound on s̈:
    if a_i>0:  s̈ ≤ β_i = (−c_i − b_i x)/a_i   (upper / max-accel field β)
    if a_i<0:  s̈ ≥ α_i = (−c_i − b_i x)/a_i   (lower / min-accel field α)
    if a_i=0:  zero-inertia point (s̈ drops out → bound on x only)
  α(s,ṡ)=max_p α_p, β(s,ṡ)=min_q β_q. Admissible iff α ≤ s̈ ≤ β.
- Maximum Velocity Curve MVC(s): smallest ṡ≥0 where α(s,ṡ)=β(s,ṡ) (admissible interval
  for s̈ collapses to a point). Above it the constraints are infeasible. Velocity profile
  must stay below MVC.

## Optimality structure
- Minimize T = ∫dt = ∫ ds/ṡ  ⇒ maximize ṡ pointwise (1/ṡ decreasing). So push velocity
  as high as the constraints allow everywhere.
- Pontryagin / bang-bang: optimal s̈ is at a bound (α or β) almost everywhere; profile
  follows alternately β (max accel) and α (min/decel) staying below MVC.
- Switch points (α→β) occur on the MVC: three kinds — discontinuous, tangent, singular
  (dynamic singularity at a zero-inertia point where a_k=0; the α,β fields diverge there
  → main source of NI failure; needs special singular-acceleration treatment via the
  λ slope). This is the classic numerical-integration (NI) algorithm and its fragility.

## TOPP-RA (the modern convex/reachability formulation = the landing point)
- Discretize s into N segments, grid s_0..s_N. x_i = ṡ_i², u_i = s̈_i constant on segment.
  Trapezoidal: ṡ_{i+1}² = ṡ_i² + 2(s_{i+1}-s_i) s̈_i  ⇒  x_{i+1} = x_i + 2 Δ_i u_i (LINEAR).
- This is a discrete-time LINEAR system (state x, control u) with LINEAR control-state
  inequality constraints Ω_i = {(u,x): a_i u + b_i x + c_i ∈ C_i}. Ω_i is a polygon;
  admissible-state set X_i and admissible-control set U_i(x) are intervals.
- Borrow set-membership control (MPC): controllable set K_i(I_N) = states at stage i from
  which an admissible control sequence reaches the goal velocity set I_N at stage N.
  One-step set Q_i(I) = {x ∈ X_i : ∃ u∈U_i(x), x+2Δ_i u ∈ I}. Because everything is an
  interval/polygon, Q_i(I) is an interval with bounds computed by TWO small LPs (2 vars
  u,x; m+2 inequalities):
    x⁺ = max x  s.t.  a_i u + b_i x + c_i ∈ C_i  and  x+2Δ_i u ∈ I
    x⁻ = min x  (same constraints)
  Recursion: K_N = {ṡ_N²}; K_i = Q_i(K_{i+1}).  Backward pass, O(mN).
- Forward pass (greedy): x*_0 = ṡ_0²; for i: u*_i = max u s.t. (u,x*_i)∈Ω_i AND
  x*_i + 2Δ_i u ∈ K_{i+1}; x*_{i+1} = x*_i + 2Δ_i u*_i. Greedily take the highest control
  that keeps the next state controllable.
- Correctness: induction (backward: any admissible param has x_i∈K_i; forward: produced
  sequence is admissible). Optimality: if maximal transition function T^β_i is
  non-decreasing, greedy = optimal (Lemma); asymptotic optimality as Δ→0, with the
  zero-inertia-point sub-optimality gap →0 (Thm 1 no-ZIP, Thm 2 with-ZIP).
- Why RA beats NI: switch points are *implicit* (identified by the controllable sets), so
  no explicit switch-point search and no singular-acceleration special-casing → 100%
  success, easy to implement. Beats CO: the big convex program O(N) vars / O(mN)
  constraints solved as SLP is O(KmN³); RA solves 3N tiny 2-var LPs → O(mN).
- Reachable set L_i (dual): forward recursion, used for feasibility/AVP.

## a,b,c extraction trick (toppra JointTorqueConstraint)
inv_dyn(q,q̇,q̈)=τ. With p=q(s), ps=q'(s), pss=q''(s):
  c = inv_dyn(p, 0, 0)          = g(q)            (gravity)
  a = inv_dyn(p, 0, ps) − c     = M(q) q'         (since at q̇=0, C term=0, ID linear in q̈)
  b = inv_dyn(p, ps, pss) − c   = M q'' + q'ᵀC q'  (set q̇=q' ṡ with ṡ=1, q̈=q'')
Uses recursive Newton–Euler so the full M, C tensors are never formed — only their
products with q', q''. Torque polytope: F=[I;−I], g=[τ_max; −τ_min], F·w ≤ g.

## Design decisions → why
- x=ṡ² not ṡ: makes the dynamics LINEAR (x_{i+1}=x_i+2Δu) and the constraints linear in
  (u,x); ṡ would give nonlinear ṡ². Time integral ∫Δ/√x stays convex.
- Squared-velocity recursion x_{i+1}=x_i+2Δu: from ṡ dṡ = s̈ ds with constant s̈ over a
  segment (ṡ² as function of s is linear with slope 2s̈). Avoids time integration.
- Reachability instead of NI switch-point hunting: NI's divergent α/β fields at dynamic
  singularities are the dominant failure mode; RA never needs the field value there.
- LP (not QP) per step: with x=ṡ², objective for one-step bounds is linear (max/min x);
  2 variables, m+2 constraints; simplex/Seidel → microseconds.
- Backward-then-forward (controllable, not reachable, for TOPP): controllable sets let the
  greedy forward pass guarantee the *end* condition is reachable while maximizing speed.
- Greedy max-u forward: 1/ṡ decreasing ⇒ pointwise-max velocity is optimal when the
  transition map is monotone (Lemma); controllable-set membership keeps it feasible to the end.
- Collocation O(Δ) vs interpolation O(Δ²) discretization: collocation cheaper/fewer vars;
  interpolation more accurate. Either works.

## Canonical code (toppra)
- algorithm/reachabilitybased/reachability_algorithm.py: compute_controllable_sets (backward,
  _one_step = 2 LPs), compute_parameterization (forward greedy via _forward_step).
- algorithm/reachabilitybased/time_optimal_algorithm.py: TOPPRA._forward_step (max u i.e.
  min −u ... actually g_upper[0]=−2Δ, g_upper[1]=−1 → maximize x_{i+1}=x+2Δu).
- solverwrapper/qpoases_solverwrapper.py: assembles the stagewise LP: vars (u,x);
  rows A[0]=[−2Δ,−1] (x_next≥K_min), A[1]=[2Δ,1] (x_next≤K_max), plus F·a u + F·b x ≤ g−F·c.
- constraint/joint_torque.py: a,b,c via three inv_dyn calls.

## Sources (retrieved this run)
1. TOPP-RA: arXiv:1707.07239 LaTeX source (src/robust.tex) — primary algorithm.
2. Pham 2014 "A General, Fast, and Robust Implementation of TOPP", arXiv:1312.6533 source
   (refs/pham2014/topp-v2.tex) — the classic Bobrow-lineage NI derivation, phase plane,
   MVC, bang-bang, switch points, dynamic singularities. Background + secondary for
   Bobrow/Shin-McKay 1985 (pre-arXiv classics).
3. toppra library (github.com/hungpham2511/toppra, cloned) + docs hungpham2511.github.io/toppra
   — canonical implementation + third-party-style explainer (README/docs).
Note: Bobrow, Dubowsky & Gibson 1985 (IJRR 4:3) and Shin & McKay 1985 are pre-arXiv and
paywalled; their equations are recovered faithfully from Pham 2014 which reproduces them.
