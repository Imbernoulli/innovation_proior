# Context: explicit convex approximations for FE-driven structural optimization

## Research question

Given a structural design described by variables `x = (x_1, ..., x_n)` — member cross-sectional areas, plate thicknesses, or per-element material densities — find the design that minimizes a structural objective `f_0(x)` (weight, compliance) subject to behavioral constraints `f_i(x) <= 0`, `i = 1, ..., m` (stress, displacement, eigenvalue limits) and simple bounds `x_min <= x <= x_max`:

```
minimize    f_0(x)
subject to  f_i(x) <= 0,         i = 1, ..., m
            x_min <= x <= x_max.
```

The defining feature of this problem is that `f_0` and every `f_i`, together with their gradients, are *implicit* functions of `x`: to know `f_i(x)` and `∂f_i/∂x_j` at a point you must run a finite element (FE) analysis and a sensitivity (adjoint) analysis. One such evaluation is the dominant cost of the whole optimization. The number of variables `n` is large (one per element or member; thousands to millions in density-based designs), while the number of behavioral constraints `m` is typically modest. The responses are generally *non-monotone* and nonlinear in `x`.

The question is how to build cheap explicit surrogates at each evaluation point that can be optimized to completion without further FE calls, converging in as few analyses as possible, with `n` large and `m` small.

## Background

**The FE-cost regime.** Each design iteration begins with a complete analysis of structural behavior to evaluate the objective and constraint values plus their first derivatives with respect to the design variables, the analysis based on finite element discretization. For large-scale problems these repeated reanalyses dominate the runtime; a design iteration is concluded by feeding the analysis and sensitivity results to a minimization step that proposes the next point. That cost structure makes "few iterations, cheap explicit subproblem" the governing requirement.

**Approximation concepts.** The accepted way to cope with implicit, expensive responses (Schmit and Miura 1976; Schmit and Fleury 1980; Fleury and Schmit 1980) is to replace the primary problem with a sequence of explicit subproblems built by Taylor expansion of the objective and constraints in terms of intermediate linearization variables, solved cheaply, and updated as the design moves. The art is the *choice of intermediate variable*.

**Reciprocal variables.** Linearizing the constraints in the reciprocal variables `1/x_j` rather than `x_j` is a long-recognized, highly effective device. There is a physical reason: for a statically determinate structure, stresses and displacements are *exact linear* functions of the reciprocal sizing variables (force divided by area). Even for shape problems with no such exact relation, the change of variables markedly improves convergence (Braibant and Fleury 1985). A first-order expansion in `1/x_j` is `f(x) ≈ f(x^0) + Σ_j (∂f/∂x_j)·(x_j^0)^2·(1/x_j^0 - 1/x_j)`.

**Conservativeness and convexity.** Starnes and Haftka (1979) observed that a *conservative* approximation — one that bounds the true function from above near the expansion point — is what makes a subproblem safe to optimize aggressively, and that mixing direct and reciprocal variables according to the sign of the derivative yields the most conservative member of that family. A separable, convex surrogate is doubly attractive: separability lets the subproblem decompose variable by variable, and convexity guarantees a unique subproblem optimum.

**The dual formulation.** A separable convex subproblem with `n` primal variables and `m` constraints is most efficiently solved in the *dual* space. The constrained primal minimization is replaced by maximizing a (quasi-unconstrained) dual function depending only on the `m` Lagrange multipliers of the constraints, subject to simple non-negativity. The efficiency comes from the dual having dimension `m` (small) rather than `n` (large); in structural optimization this dual approach (Fleury 1979; Fleury 1982; Lasdon 1970) reconciled the older optimality-criteria techniques with mathematical programming. Separability means the inner primal minimization splits into `n` independent one-variable problems, each solvable in closed form, so the dual function and its derivatives are available explicitly.


## Baselines

**Optimality Criteria (OC).** The incumbent in density- and sizing-based structural optimization. From the stationarity (KKT) conditions of a single monotone resource constraint (typically a volume/material bound `Σ_e v_e x_e <= V`), one derives a heuristic multiplicative fixed-point update `x_e ← x_e · (B_e)^η`, with `B_e = (-∂f_0/∂x_e) / (λ · ∂g/∂x_e)` the ratio of objective sensitivity to constraint sensitivity, `η` a damping exponent, and the single multiplier `λ` found by bisection so the constraint binds. OC is cheap per step and scales to millions of variables.

**Sequential Linear Programming (SLP).** Take the first-order Taylor expansion of `f_0` and each `f_i` in the *direct* variables `x_j` and solve the resulting linear program for the step. The linearization carries no curvature, so the LP solution is bounded by artificial move limits `|x_j - x_j^k| <= δ_j` imposed on each variable.

**Reciprocal / Sequential Quadratic Programming.** Reciprocal linearization expands in `1/x_j`: where `∂f/∂x_j < 0` this term is convex; where `∂f/∂x_j > 0` it is concave. SQP builds a quadratic model with an approximate Hessian, requiring an `n×n` matrix.

**CONLIN (Convex Linearization; Fleury and Braibant 1986; Fleury 1989).** The direct predecessor and the strongest baseline. For each term of each function, choose the *direct* variable `x_j` if `∂f_i/∂x_j > 0` and the *reciprocal* variable `1/x_j` if `∂f_i/∂x_j < 0`:

```
f_i(x) ≈ f_i(x^0) + Σ_{∂f_i/∂x_j > 0} (∂f_i/∂x_j)(x_j - x_j^0)
                  + Σ_{∂f_i/∂x_j < 0} (∂f_i/∂x_j)(x_j^0)^2 (1/x_j^0 - 1/x_j).
```

Because each retained term is convex (linear, or a reciprocal term whose coefficient on `1/x_j` is `-(∂f_i/∂x_j)(x_j^0)^2 > 0`), the surrogate is convex and separable, and by the Starnes–Haftka argument it is the most conservative direct/reciprocal mix. CONLIN solves the resulting subproblem with a dual method: it forms the separable Lagrangian, minimizes it analytically variable by variable to get a closed-form `x_i(r)` (with the side bounds clamping some variables to `x_min`/`x_max`), and maximizes the resulting dual function `l(r)` over the multipliers `r >= 0` — the dual gradient being exactly the primal constraint values, the dual Hessian available in closed form, solved by a sequential quadratic method in the active-multiplier subspace. The reciprocal term's singularity is fixed at `x_j = 0`, so the curvature of each term — and hence the conservativeness of the approximation — is determined entirely by the current iterate.

## Evaluation settings

The natural proving ground is structural sizing and shape problems where responses come from FE analysis: cantilever and truss weight-minimization under stress and displacement limits, the classic multi-bar trusses, beam sizing, and density-based continuum problems with a material-volume constraint. The yardsticks are the number of FE evaluations (outer iterations) to converge, the quality of the final design, robustness across starting points without re-tuning move limits, and behavior as `m` (number of constraints) grows. A self-contained algebraic test that exercises the optimizer alone — minimize `x_1^2 + x_2^2 + x_3^2` subject to two quadratic ball constraints `(x - c_i)^2 <= 9` with `0 <= x_j <= 5` — and a small FE cantilever-beam sizing problem are convenient checks that need no large solver.

## Code framework

The harness that already exists: an outer loop that, at each design point, calls FE analysis for values and sensitivities, hands them to a subproblem solver to get the next point, and tests a stopping criterion. The contribution will be the subproblem builder and its solver — everything below is a stub.

```python
import numpy as np

def fe_analysis(xval):
    """Run the finite element + adjoint analysis at xval.
    Returns f0val, df0dx (n,1), fval (m,1), dfdx (m,n). The expensive call."""
    raise NotImplementedError

def build_subproblem(xval, xmin, xmax, xold1, xold2,
                     f0val, df0dx, fval, dfdx, state):
    """Build the explicit, cheap-to-optimize local surrogate of f0 and f_i
    around xval from gradient information, plus whatever per-variable
    state the surrogate carries between iterations.
    # TODO: this is the method.
    """
    pass  # returns subproblem coefficients + updated state

def solve_subproblem(subproblem):
    """Find the unique minimizer of the convex separable surrogate
    subject to the local bounds, WITHOUT any further FE call.
    # TODO: exploit separability + small constraint count (solve via the dual).
    """
    pass  # returns x_next (and multipliers)

def kkt_residual(x, mult, f0val, df0dx, fval, dfdx, xmin, xmax):
    """Norm of the KKT residual of the ORIGINAL problem, for stopping."""
    pass

def optimize(x0, xmin, xmax, maxit):
    xval = x0.copy(); xold1 = x0.copy(); xold2 = x0.copy()
    state = None
    f0val, df0dx, fval, dfdx = fe_analysis(xval)        # expensive
    for k in range(1, maxit + 1):
        sub, state = build_subproblem(xval, xmin, xmax, xold1, xold2,
                                      f0val, df0dx, fval, dfdx, state)
        x_next, mult = solve_subproblem(sub)                # cheap, no FE
        f0next, df0next, fnext, dfdxnext = fe_analysis(x_next)
        if kkt_residual(x_next, mult, f0next, df0next, fnext, dfdxnext,
                        xmin, xmax) < tol:
            xval = x_next
            break
        xold2, xold1, xval = xold1, xval, x_next
        f0val, df0dx, fval, dfdx = f0next, df0next, fnext, dfdxnext
    return xval
```

The two `pass` bodies — how the local surrogate is built and how the resulting convex separable subproblem is solved through its low-dimensional dual — are exactly the slots the method fills.
