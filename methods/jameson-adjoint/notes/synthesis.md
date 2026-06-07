# Synthesis — Adjoint-based aerodynamic shape optimization (Jameson continuous adjoint)

## Pain point
Aerodynamic shape design with CFD: pick N shape parameters α (Hicks–Henne bumps, B-spline control
points, free surface), compute a cost I (drag at fixed lift, or inverse-design pressure mismatch
½∫(p−p_d)²). To improve, need gradient dI/dα. Naive = finite differences: perturb each αᵢ, re-run the
(expensive, nonlinear, transonic) flow solve, divide. Cost ∝ (N+1) flow solves. Hicks–Henne 1978 did
exactly this with conjugate gradient + full-potential solver — works for a handful of variables, but
the cost is prohibitive as N grows (a real wing wants hundreds; the true optimum lives in an
infinite-dimensional space of shapes). Each transonic flow solve is the dominant cost; N+1 of them per
gradient kills it.

## The lineage / ancestors
- **Lions 1971** (Optimal Control of Systems Governed by PDEs): the abstract machinery — adjoint state /
  costate for PDE-constrained optimization, Lagrange multiplier as a function over the domain.
- **Bryson & Ho 1975** (Applied Optimal Control): gradient-via-adjoint for *trajectory* optimization —
  costate integrated backward in time gives the gradient w.r.t. all controls in one backward sweep.
  Jameson explicitly says his method "directly corresponds to the gradient technique for trajectory
  optimization pioneered by Bryson."
- **Pironneau 1973/1974** ("On optimum profiles in Stokes flow"; "On optimum design in fluid mechanics",
  JFM 64:97-110): first applied control-theory/adjoint optimal-shape-design to fluids (Stokes,
  low-Re drag), elliptic systems. Established that you can get the shape gradient from an adjoint
  state. Limitation: Stokes / elliptic / low speed; not transonic, not the nonlinear hyperbolic
  Euler equations with shocks.
- **Hicks & Henne 1978** (Wing Design by Numerical Optimization, J. Aircraft 15:407): CFD (full
  potential) + conjugate-gradient optimizer, gradients by finite differences. The baseline whose
  N+1-solves cost is the wall to break through.
- **Jameson's own jet/CFD solvers** (JST scheme, multigrid, FLO codes): the fast Euler/N-S solvers that
  make both the flow solve AND the (similar) adjoint solve affordable.

## The core idea (re-derive in reasoning.md)
Treat the flow PDE R(w, F)=0 as a *constraint*. I = I(w, F). Under shape change δF:
  δI = (∂I/∂w)δw + (∂I/∂F)δF.
The expensive unknown is δw (the flow sensitivity — needs a flow-linearization solve PER design var).
Constraint linearized: (∂R/∂w)δw + (∂R/∂F)δF = 0. Multiply δR=0 by an arbitrary multiplier ψ and
subtract:
  δI = [(∂I/∂w)ᵀ − ψᵀ(∂R/∂w)] δw + [(∂I/∂F)ᵀ − ψᵀ(∂R/∂F)] δF.
Choose ψ to kill the δw bracket → adjoint equation
  (∂R/∂w)ᵀ ψ = ∂I/∂w     (one linear solve, independent of N).
Then gradient
  G = ∂I/∂F − ψᵀ(∂R/∂F)   (cheap matrix-vector, evaluated for ALL design vars at once).
So full gradient = one flow solve (for w) + one adjoint solve (for ψ) + cheap assembly, regardless of N.
Adjoint solve ≈ same cost as flow solve (same Jacobian, transposed; similar hyperbolic structure, waves
reversed). Iteration cost ≈ 2 flow solves. This is THE result.

## Continuous vs discrete (both in scope)
- **Continuous adjoint** (Jameson 1988): linearize the PDE, integrate by parts → adjoint *PDE*
  Cᵢᵀ ∂ψ/∂ξᵢ = 0 (with field source if I has a field integral); the boundary terms that can't be
  cancelled fix the adjoint *boundary condition*; e.g. for inverse design I=½∫(p−p_d)² the wall BC is
  ψ·n = p − p_d (a transpiration condition on the momentum costate components). Then discretize ψ.
  Bonus: the gradient can be reduced to a pure *surface* integral (field/metric terms eliminated via
  the reduced-gradient identity), so it doesn't depend on how the interior mesh deforms.
- **Discrete adjoint**: discretize the flow first → R is the discrete residual vector, ∂R/∂w is the
  flow Jacobian; the adjoint is the linear system (∂R/∂w)ᵀ ψ = ∂I/∂w with the *transposed* Jacobian.
  Gives the *exact* gradient of the discrete cost. This is what the canonical teaching code does.

## Sobolev gradient (design choice — why)
Naive steepest descent δF = −λG: G involves derivatives of the shape (in the calculus-of-variations
toy, g = ∂F/∂y − d/dx ∂F/∂y' depends on y''), so each step δF=−λg *reduces smoothness by two classes*
→ the shape gets rougher each iteration → numerical instability / non-smooth airfoils. Fix: define the
gradient in a *Sobolev* inner product ⟨u,v⟩=∫(uv+ε u'v')dx instead of L². Then the implied smoothed
gradient Ḡ solves Ḡ − ε ∂²Ḡ/∂ξ² = G (a smoothing/elliptic solve), δF=−λḠ preserves smoothness class,
still descent (δI=−λ⟨Ḡ,Ḡ⟩<0). Acts as a preconditioner → allows much larger steps, far fewer design
cycles. (In the discrete teaching code this corresponds to applying a tridiagonal smoothing operator
A to the raw gradient; steepest descent step δF=−λAG.)

## Why fix lift when minimizing drag (design choice)
Induced (vortex) drag is a big fraction of total drag and drops if you just reduce lift / increase span
→ unconstrained drag-min cheats by shedding lift. So fix C_L: adjust angle of attack each flow solve to
hit target C_L, and include the α-sensitivity in the gradient. Also fix planform / minimum thickness so
span can't run away.

## Why continuous adjoint over pure FD (and over forward/tangent linearization)
- vs FD: FD cost ∝ N+1 solves, and is noisy (step-size dilemma). Adjoint cost ∝ 2 solves, N-independent.
- vs forward/tangent (direct differentiation, "direct method"): forward solves (∂R/∂w)(∂w/∂αᵢ)=−∂R/∂αᵢ
  ONCE PER design var i → cost ∝ N solves too. Adjoint flips the order of operations: one solve with the
  TRANSPOSE, then cheap dot products → N-independent. Forward wins only when #outputs ≫ #inputs; for
  shape design #inputs (design vars) ≫ #outputs (one scalar cost), so reverse/adjoint wins.
- Continuous adjoint additionally: removes ill-posed pointwise pressure sensitivities (a shock can make
  ∂p/∂F unbounded at a point, but the integrated drag is smooth); and the reduced-gradient form makes it
  mesh/discretization-agnostic, easy on unstructured/overset grids.

## Canonical code (dougshidong/quasiOneD, discrete adjoint, inverse design)
Quasi-1D Euler nozzle. State w=[ρ, ρu, e] per cell; area S(x) is the control, parameterized by design
vars (sine-bump / spline). Flow: time-march residual R(w,S)=0 to steady state (Roe/SW flux, explicit).
Cost: I = ½ Σ (p_i/p_t − p_d,i)² Δx_i (inverse design). Gradient via adjoint:
  dCostdW = (p−p_d)(∂p/∂w)Δx           [∂I/∂w]
  dRdW = flow Jacobian (analytic/AD)    [∂R/∂w]
  dAreadDes; dRdDes = dRdArea·dAreadDes [∂R/∂F]
  dCostdDes = dCostdArea·dAreadDes      [∂I/∂F]
  ψ = solve( dRdW.transpose(), dCostdW )         <-- adjoint eq (∂R/∂w)ᵀψ = ∂I/∂w
  dIdDes = dCostdDes − ψᵀ dRdDes                 <-- gradient assembly G
Optimizer: steepest descent or BFGS; Armijo line search new_cost ≤ cost + α c₁ g·p; loop until
‖g‖<tol. Matches Jameson's design cycle: flow solve → adjoint solve → gradient → (Sobolev) → update.

## SU2 canonical interface (production continuous adjoint)
shape_optimization.py -g CONTINUOUS_ADJOINT -o SLSQP -f inv_NACA0012_basic.cfg
OBJECTIVE_FUNCTION=DRAG, DV_KIND=HICKS_HENNE bumps, MATH_PROBLEM continuous adjoint; design cycle =
direct solve + adjoint solve + projected gradient.

## Empirical facts (→ context.md, NOT to be "measured" in reasoning):
- FD gradient costs N+1 flow solves; transonic flow solve is the dominant expense. (Hicks-Henne era)
- Adjoint solve ≈ cost of a flow solve (same transposed Jacobian / similar hyperbolic system).
- Iteration ≈ 2 flow solves; convergence usually in ~10-50 design cycles. (Jameson reports as method
  property — keep as the method's own claim, recall not fabricate; safe to state as design-time
  expectation.)
- Steepest descent on raw L² gradient roughens the shape (smoothness drops two classes/step) → the
  Sobolev-gradient motivation. (Calculus-of-variations fact, derivable.)

## URLs
- Jameson VKI 2003 lecture notes (clean full derivation): http://aero-comlab.stanford.edu/Papers/jameson.vki03.pdf
- Jameson, "Optimum Aero Design using N-S Eqns" 1998: http://aero-comlab.stanford.edu/Papers/jameson.tcfd.1998-10.pdf
- Jameson, Göttingen transonic wing control theory: http://aero-comlab.stanford.edu/Papers/jameson.gottingen.pdf
- Jameson, AIAA-95-1729 CFD+control theory: http://aero-comlab.stanford.edu/Papers/AIAA-1995-1729-807.pdf
- Nadarajah & Jameson continuous-vs-discrete adjoint AIAA-2000-0667: http://aero-comlab.stanford.edu/Papers/nadarajah.aiaa.00-0667.pdf
- Pironneau JFM 1974: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/on-optimum-design-in-fluid-mechanics/8A2CEB8E078BC0C6A790D67D97B3B174
- Hicks & Henne 1978 J. Aircraft: https://arc.aiaa.org/doi/10.2514/3.58379
- Lions 1971 Optimal Control of PDEs: https://link.springer.com/book/9783642650260
- quasiOneD code: https://github.com/dougshidong/quasiOneD
- SU2 continuous-adjoint NACA0012 tutorial: https://su2code.github.io/tutorials/Inviscid_2D_Unconstrained_NACA0012/
