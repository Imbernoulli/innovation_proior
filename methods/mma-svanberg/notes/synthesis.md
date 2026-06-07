# MMA synthesis notes (grounded)

## Sources read this run
- Svanberg 2007, "MMA and GCMMA — two methods for nonlinear optimization" (KTH note, refs/svanberg_mmagcmma.pdf). FULL algorithm: subproblem (3.1)-(3.5), bounds (3.6)-(3.10), asymptote update (3.11)-(3.14), the standard-NLP reduction (a0=1, ai=0, di=1, ci large), KKT (5.7), relaxed/interior-point KKT (5.9), strictly-convex log-barrier equivalent (5.10), Newton system (5.11)-(5.14). This matches code/mma.py line-for-line.
- Fleury 1989, "CONLIN: an efficient dual optimizer based on convex approximation concepts" (Struct. Optim. 1:81-89, refs/conlin_fleury.pdf). The DIRECT ancestor. Convex linearization = mixed direct/reciprocal per sign of derivative; separable convex subproblem; DUAL method: per-variable closed form x_i(r) (eq 12-14), dual function l(r) (16), dual gradient = primal constraint values (17), dual Hessian (19), active-set discontinuities; max-min two-phase. Also: FE-cost framing, approximation-concepts lineage (Schmit & Miura 1976; Fleury & Schmit 1980), reciprocal variables exact for statically determinate, Starnes-Haftka 1979 conservativeness.
- code/mma.py: arjendeetman/GCMMA-MMA-Python (GPL-3.0), port of Svanberg's 2007 MATLAB. mmasub, subsolv (primal-dual interior point), kktcheck, gcmmasub, asymp, raaupdate, concheck. Two examples (toy, cantilever beam).
- WebFetch confirmations: p_ij/q_ij sign split and scaling, CONLIN = L_j→-inf/U_j→+inf special case (i.e. fixed reciprocal U=0... actually L_j=0, U_j=inf gives reciprocal). Wikipedia thin.

## The problem (research question)
minimize f0(x) s.t. fi(x)<=0, i=1..m, xmin<=x<=xmax. Functions are STRUCTURAL responses (compliance, stress, displacement, weight) evaluated by FINITE ELEMENT analysis. Each f-eval + gradient = one FE solve = EXPENSIVE. n large (one var per element/member), m moderate. Want: few FE calls; cheap explicit subproblem; robust (no fragile move-limit / line-search tuning); handle non-monotone responses.

## Ancestors and their gaps (load-bearing)
1. Optimality Criteria (OC): heuristic fixed-point update from KKT stationarity, e.g. x_e <- x_e * (B_e)^eta with B_e = -dC/dx_e / (lambda dV/dx_e); single monotone (volume) constraint, Lagrange multiplier by bisection. Cheap, scales, but: heuristic, no clean multi-constraint extension, monotone-constraint-bound, exponent/damping tuned by hand. (This is exactly the OC of SIMP topology opt — but I do NOT re-derive SIMP; OC is just the incumbent optimizer to beat.)
2. Sequential Linear Programming (SLP): first-order Taylor in DIRECT variables, solve LP each step. Linear approx is non-conservative and unbounded -> needs move limits, which are fragile and slow.
3. Reciprocal approximation: linearize in 1/x_j. Exact for stress/displacement in statically determinate structures (force/area). Convex only where df/dx<0; wrong curvature sign otherwise.
4. CONLIN (Fleury & Braibant 1986; Fleury 1989): pick direct var if df/dx_j>0, reciprocal if <0, per term -> always convex, separable, conservative. Solve by DUAL method (one var per constraint). GAP: the reciprocal "asymptote" is pinned at x_j=0 (origin). No control over curvature/step. For a given iterate the approximation is fixed -> if it overshoots you can only shrink with external move limits; CONLIN can be too curved (slow) or, when a derivative flips sign across iterations, oscillate. No knob to tune conservativeness.

## The MMA idea (what fills the gap)
Replace CONLIN's fixed singularity points {0, inf} with MOVING ones: lower asymptote L_j<x_j<U_j upper. Approx each fi:
  fi_tilde(x) = ri + sum_j [ p_ij/(U_j - x_j) + q_ij/(x_j - L_j) ]    (3.2)
Choose p_ij,q_ij to match value+gradient at x^k AND be convex:
  if dfi/dx_j > 0: put weight on the U-term (p_ij>0, q_ij=0): p_ij=(U_j-x_j)^2 * dfi/dx_j
  if dfi/dx_j < 0: put weight on the L-term (q_ij>0, p_ij=0): q_ij=-(x_j-L_j)^2 * dfi/dx_j
  ri = fi(x^k) - sum_j [p_ij/(U_j-x_j)+q_ij/(x_j-L_j)] at x^k.
(Code adds tiny convexity floor: 0.001*(p+q)+raa0/(xmax-xmin), eqs 3.3-3.5.)
Convex: each term's 2nd deriv = 2p/(U-x)^3 + 2q/(x-L)^3 >=0 for L<x<U, p,q>=0. Sum of convex -> convex; separable.
Curvature: 2nd deriv grows as asymptote approaches x_j -> moving U_j down / L_j up makes the approx MORE curved = MORE conservative = smaller step. Pull them away -> flatter -> closer to linear = bigger step. So asymptotes = a continuous conservativeness/step-size knob, replacing brittle move limits.
CONLIN special case: U_j=inf, L_j=0 reproduces direct/reciprocal mixing.

## Asymptote update (oscillation heuristic) (3.11-3.13)
k=1,2: L=x-0.5(xmax-xmin), U=x+0.5(xmax-xmin) (default asyinit=0.5).
k>=3: sign s = (x_j^k - x_j^{k-1})(x_j^{k-1} - x_j^{k-2}).
  s>0 (monotone progress same direction) -> gamma=1.2: PUSH asymptote away, relax, bigger steps (asyincr).
  s<0 (oscillating, overshot) -> gamma=0.7: PULL asymptote in toward x, tighten, damp (asydecr).
  s=0 -> gamma=1.
  L = x - gamma*(x^{k-1}-L^{k-1}); U = x + gamma*(U^{k-1}-x^{k-1}). Clamp to [0.01,10]*(xmax-xmin) (asymin/asymax).
Move/trust bounds alfa,beta (3.6-3.7): keep x at least 10% (albefa=0.1) away from asymptotes (so terms finite) and within move=0.5 fraction of range.

## The subproblem and its DUAL (the heart)
Subproblem (3.1): minimize fi0_tilde(x) s.t. fi_tilde(x)<=0, alfa<=x<=beta. Convex + SEPARABLE.
Original 1987 + CONLIN dual: form Lagrangian psi(x,lambda)=g0(x)+sum_i lambda_i gi(x) = sum_j [p_j(lambda)/(U_j-x_j)+q_j(lambda)/(x_j-L_j)], p_j(lambda)=p0j+sum_i lambda_i p_ij, q_j likewise (5.4-5.5). SEPARABLE in x given lambda -> minimize term by term:
  d/dx_j: p_j/(U_j-x_j)^2 - q_j/(x_j-L_j)^2 = 0  -> x_j(lambda) = (L_j*sqrt(p_j)+U_j*sqrt(q_j))/(sqrt(p_j)+sqrt(q_j)), clamped to [alfa,beta]. Closed form, O(n), no FE.
Dual function W(lambda)=min_x L = psi(x(lambda),lambda) - sum_i lambda_i b_i (+ y,z artificial-var terms). CONCAVE in lambda (min of affine-in-lambda family). 
Key (CONLIN 17): dW/dlambda_i = gi(x(lambda)) - bi = primal constraint value -> dual gradient is just constraint residual. Maximize W over lambda>=0 (m-dimensional, m small) by Newton / SQP in dual space; dual Hessian closed form (CONLIN 19), active-set for clamped x. Recover x*=x(lambda*).
Why dual wins: subproblem has n (large) primal vars but m (small) constraints; dual collapses to m variables. One subproblem solve uses ZERO FE evals.

## Artificial vars y,z (2007 generalization, in code)
General form (1.1): min f0+a0 z+sum(ci yi+0.5 di yi^2) s.t. fi - ai z - yi <= 0. Set a0=1,ai=0,di=1,ci large -> recovers plain NLP (1.2) but ALWAYS feasible (yi relax constraints at high cost). Also gives min-max, least-squares forms. Default a0=1,a=0,c=1000,d=1.

## Modern solve = interior-point on same KKT (2007 note sec 5, = subsolv)
Instead of explicit dual maximization, the 2007 code solves the subproblem KKT (5.7) by a primal-dual interior point: relax complementarity x*s=eps (5.9), equiv to log-barrier (5.10), Newton steps (5.11-5.14) eliminating xsi,eta,mu,zet,s -> reduced system in (dx,dy,dz,dlam); choose m<n vs m>=n elimination (mma.py lines 412/424); fraction-to-boundary step (1.01), backtracking on residual norm; shrink eps by 0.1. Same optimum, robust, handles the y,z vars uniformly.

## Design-decision -> why
- Two asymptotes (not one): need convex term for BOTH gradient signs; p-term convex+increasing, q-term convex+decreasing.
- Scale by (U-x)^2 / (x-L)^2: makes p/(U-x), q/(x-L) reproduce the gradient exactly at x^k (the (U-x)^-2 derivative cancels it).
- Sign split (only one of p,q nonzero per term): minimal, keeps p,q>=0 => convex; matches gradient with the single term on the correct side.
- gamma 0.7/1.2: damp on detected oscillation, accelerate on steady progress — generic line-search-free trust adaptation.
- albefa 0.1 / move 0.5: keep iterate off the asymptote (avoid 1/0) and bound the step.
- Dual solve: exploit m<<n; constraint values are free as dual gradient.
- ci "large": penalize relaxation so y=0 at optimum unless truly infeasible.
- Conservative-floor 0.001(p+q)+raa0/(xmax-xmin): strict convexity / well-defined x(lambda) even if a gradient is ~0.

## In-frame cautions
- Frontier anchor (PyMOTO SIMP) NEVER mentioned. SIMP not re-derived; OC is just the incumbent optimizer.
- No proposed-method results/convergence numbers. FE-cost + OC-limitation = pre-method context facts.
- Don't cite "the paper"/Svanberg-as-target. CONLIN/Fleury/OC/Schmit ancestor citations OK.
- Domain: Structural Optimization.
