# MPC synthesis notes

## Three sources (all read this run)
1. PRIMARY (foundational, reproduces the equations): Qin & Badgwell, "A survey of industrial
   model predictive control technology", Control Eng. Practice 11 (2003) 733-764 —
   reproduces IDCOM (Richalet 1978), DMC (Cutler & Ramaker 1980), QDMC (Cutler/Garcia-Morshedi
   1986) equations + the LQG-can't-handle-constraints argument + the general MPC algorithm.
   Stands in for Richalet 1978 / Cutler&Ramaker 1980 / Garcia-Prett-Morari 1989 (all paywalled
   Automatica/Elsevier — flagged). refs/qin_badgwell_survey.pdf
2. BACKGROUND: LQR/CARE (LQG, Kalman 1960; Riccati) + QP. The book Ch 8 (LQ optimal control,
   batch vs recursive) gives the condensed batch form and the DARE.
3. THIRD-PARTY EXPLAINER: Borrelli, Bemporad, Morari, "Predictive Control for Linear and
   Hybrid Systems" — Ch 8 (batch/condensed QP, DARE), Ch 11 (constrained 2-norm QP, sparse and
   condensed), Ch 12 (RHC idea, feasibility, stability Thm 12.2, terminal cost = LQR value
   function). refs/borrelli_bemporad_morari_book.pdf

## CANONICAL CODE: pyMPC (forgi86), OSQP-based linear constrained MPC.
code/pyMPC/mpc_no_slack.py is the clean core. Sparse formulation: decision vars
z = (x0..xN, u0..uN-1), dynamics as equality constraints, box constraints on x,u and rate Δu.
First input applied; warm-started OSQP; update measurement, re-solve.

## Pain point / research question
LQR (= LQG state feedback) is optimal, MIMO, constructive — but UNCONSTRAINED. Its law u=-Kx
is computed once offline; it has NO mechanism for hard limits:
- actuators saturate (valve fully open / closed, finite torque, finite voltage). LQR's -Kx
  will command u beyond umax; the real plant clips it, and the LQR's stability/optimality
  guarantees (built on u=-Kx being applied exactly) no longer hold — clipping can destabilize.
- states have limits (tank level, temperature, pressure, position, SOC). LQR ignores them.
- Economic operating point of a process unit sits AT the intersection of constraints (Prett &
  Gillette 1980): you make money by running as close to a constraint as possible WITHOUT
  crossing. LQR has no notion of "approach but don't violate." (context fact, sourced)
Qin-Badgwell list the reasons LQG "failed to have a strong impact" on process industries:
constraints; nonlinearities; model uncertainty; unique performance criteria; cultural. The
load-bearing one for us: CONSTRAINTS. (sourced)

## The derivation chain (reasoning.md order)
1. LQR is optimal but unconstrained; actuators saturate, states have limits; clipping -Kx
   voids the guarantee. (recall LQR result, do NOT re-derive it.)
2. Want to keep the same quadratic objective but ADD hard constraints u in U, x in X.
   With constraints the HJB no longer has a closed-form quadratic value function — the inner
   min over u is now a CONSTRAINED min, so u*(x) is piecewise-affine, not linear. No more
   one-shot Riccati gain. (book Cor 11.2: value fn is convex piecewise-quadratic.)
3. So instead: pose a FINITE-horizon problem over N steps from the CURRENT measured state x(t):
       min  sum_{k=0}^{N-1} (x_k'Q x_k + u_k'R u_k) + x_N' P x_N
       s.t. x_{k+1} = A x_k + B u_k,  x_0 = x(t),  u_k in U, x_k in X, x_N in X_f.
   Decision variable = the WHOLE input sequence U_0 = (u_0..u_{N-1}) (and states).
4. Solve it as a QP. Two constructions:
   (a) CONDENSED (book 8.5-8.9, 11.31): eliminate states by successive substitution.
       X = S^x x(0) + S^u U_0, with
       S^x = [I; A; A^2; ...; A^N],
       S^u = banded lower-triangular with blocks A^{i-1}B.
       J = X'Qbar X + U_0'Rbar U_0, Qbar=blockdiag(Q,...,Q,P), Rbar=blockdiag(R,...,R).
       Substitute: J = U_0' H U_0 + 2 x(0)' F U_0 + x(0)' Y x(0),
       H = S^u'Qbar S^u + Rbar  (>0 since Rbar>0),  F = S^x'Qbar S^u,  Y = S^x'Qbar S^x.
       Dense QP in U_0 only (dim mN). Constraints G_0 U_0 <= w_0 + E_0 x(0).
       Without constraints, gradient=0 gives U_0* = -H^{-1}F'x(0); first block = -Kx, the LQR
       gain falls back out -> sanity check.
   (b) SPARSE / banded (book 11.30, pyMPC): keep states as decision vars, dynamics as equality
       constraints. Bigger but sparse -> OSQP exploits sparsity. This is what pyMPC builds.
5. RECEDING HORIZON: apply ONLY u_0*, discard the rest. Re-measure x(t+1), re-solve. Why:
   - Feedback: re-solving from the measured state closes the loop. The open-loop sequence
     applied blindly is fragile to model error/disturbances; re-measuring each step injects
     feedback. (book 8.27 vs 8.28 discussion: feedback law more robust than open-loop seq.)
   - Constraints handled exactly: the QP enforces u in U, x in X over the whole horizon, so the
     applied u_0 never violates and future violations are ANTICIPATED and prevented.
   - This is exactly the Lee-Markus (1967) / Propoi (1963) moving-horizon idea, made practical.
6. But finite N breaks the LQR guarantees:
   - FEASIBILITY can be lost: feasible now does not imply feasible next step (book Ex 12.1,
     unstable double integrator: short horizon -> trajectory runs into infeasibility, x(2)=[1,2]
     stuck). Fix: terminal CONSTRAINT x_N in X_f with X_f control invariant -> persistent
     feasibility (Thm 12.1, Lemma 12.2).
   - STABILITY can be lost: minimizing a finite-horizon cost does not force decay (book Ex 12.2:
     N=2 -> all states diverge; bigger N or right terminal cost -> converge).
7. TERMINAL COST for stability — the LQR connection. Choose terminal cost p(x)=x'Px and
   terminal set X_f so that J_0*(x) becomes a Lyapunov function (book Thm 12.2). Need
   (A3): min_{v} [-p(x)+q(x,v)+p(Ax+Bv)] <= 0 on X_f, i.e. p is a control-Lyapunov fn.
   For the 2-norm case (book 12.24): pick X_f = invariant set of the LQR closed loop A+BF_inf
   and P = P_inf = the CARE solution; then (A3) holds WITH EQUALITY because P_inf solves
       A'(P - PB(B'PB+R)^{-1}B'P)A + Q - P = 0  (DARE).
   Interpretation (book 12.26): the terminal cost x'P_inf x = the infinite-horizon LQR cost of
   the tail, i.e. "run constrained MPC until you reach X_f, then the unconstrained LQR takes
   over and x'Px exactly accounts for the rest." So MPC = constrained head + LQR tail. The
   terminal cost is literally the LQR value function. This is the clean stability recipe.
   Telescoping proof (book 12.19-12.23): shift the optimal sequence by one, append the LQR
   move v at the end (feasible because X_f invariant), get J_0*(x(1)) <= J_0*(x(0)) - q(x_0,u_0),
   so J_0* strictly decreases -> Lyapunov -> asymptotic stability with domain of attraction X_0.

## Design decisions -> why
- Decision var = whole input sequence (not just u_0): need to predict the whole horizon to
  enforce future state constraints and to compute a meaningful cost-to-go; only u_0 is applied.
- Quadratic cost: convex QP (PD Hessian since R>0), solvable by standard QP codes (Garcia-
  Morshedi point: "one of the simplest optimization problems"). 1/inf-norm -> LP (variant).
- Move/rate penalty Q_Du on Δu = u_k - u_{k-1}: damps aggressiveness, robustness to model
  error, improves QP conditioning (Qin-Badgwell on DMC "move suppression"). Also lets you bound
  slew rate Δu in [Dumin,Dumax] (real actuators have rate limits). pyMPC includes it.
- Terminal constraint X_f control invariant -> persistent feasibility (else feasibility lost).
- Terminal cost P = CARE solution -> stability + recovers infinite-horizon behavior; without it
  short horizons can diverge.
- Soft state constraints (slack) in practice: a disturbance can make the QP infeasible; harden
  inputs (physical) but soften outputs with an l1/l2 penalty so the QP always returns something.
  (pyMPC full version mpc.py has slack; the no-slack core omits it.) Mention as variant.
- Sparse vs condensed: condensed -> small dense QP (good for short horizon / dense solvers);
  sparse/banded -> larger but sparse (good for OSQP / long horizon). pyMPC uses sparse.
- Warm start: consecutive QPs differ only in x(0) (and uminus1); warm-start OSQP from last
  solution -> fast online solve.

## Code framework scaffold (pre-method)
Known pre-method primitives: discrete LTI model x+ = A x + B u; a QP solver (osqp); the LQR
gain / CARE solver (scipy solve_discrete_are) as the terminal-cost ingredient. The empty slot:
a controller that, each step, builds and solves an optimization over a horizon and returns the
first input. Scaffold = build_qp(...) stub + a Controller with setup/step/update stubs + a
closed-loop sim loop.

## Empirical facts (context only; sourced; never fabricate)
- Actuators saturate; states have hard limits; economic optimum at constraint intersection
  (Prett & Gillette 1980 via Qin-Badgwell). -> context Background.
- LQG/LQR limitations (no constraints) -> Qin-Badgwell sec 2.1. -> context Baselines.
- Unstable double integrator: short horizon -> infeasible/diverge (book Ex 12.1, 12.2). This is
  a known FE pathology of finite-horizon RHC; reason about it, do NOT report new numbers.
- Industrial $ benefits ($150k/yr etc) — proposed-method-adjacent results; EXCLUDE.
