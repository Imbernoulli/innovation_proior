# Adjoint-based aerodynamic shape optimization (continuous/discrete adjoint for CFD)

## Problem

Optimize an aerodynamic surface (wing, duct) by descending a CFD-evaluated cost I — drag at fixed
lift, or inverse-design pressure mismatch I = ½∫(p − p_d)² — over many shape design variables α. The
obstacle is the gradient dI/dα: finite differences cost N+1 flow solves per gradient (and are noisy),
and forward/tangent sensitivities cost one linear solve per design variable. Both scale with the
number of design variables N, which is prohibitive when N is large or the boundary is a free surface.

## Key idea

Treat the converged flow as a PDE constraint R(w, α) = 0 and form the Lagrangian I + ψᵀR. Because
δR = 0, the adjoint (costate) ψ is free; choose it to cancel the expensive flow-sensitivity term δw in
the cost variation. This decouples the gradient from N: a single transposed flow solve gives ψ, and
the gradient against *all* design variables follows from a cheap assembly. One design cycle costs
≈ two flow solves regardless of N.

## The method

Cost variation under a shape change, with the flow response δw and the direct shape term:

    δI = (∂I/∂w)ᵀ δw + (∂I/∂α)ᵀ δα,      (∂R/∂w) δw + (∂R/∂α) δα = 0.

Add ψᵀ δR = 0 and group:

    δI = [(∂I/∂w)ᵀ − ψᵀ(∂R/∂w)] δw + [(∂I/∂α)ᵀ − ψᵀ(∂R/∂α)] δα.

**Adjoint equation** (choose ψ to kill the δw bracket — one solve, transpose of the flow Jacobian,
independent of N):

    (∂R/∂w)ᵀ ψ = ∂I/∂w.

**Gradient assembly** (cheap, all design variables at once):

    G = ∂I/∂α − ψᵀ (∂R/∂α).

The adjoint matrix (∂R/∂w)ᵀ is the flow Jacobian transposed — the same hyperbolic system with waves
reversed — so the same fast (multigrid / multistage) solvers apply and the adjoint solve costs about
one flow solve.

**Continuous adjoint** (PDE-level, Euler): linearize ∂Fᵢ/∂ξᵢ = 0, multiply by a costate field ψ,
integrate by parts → adjoint hyperbolic system Cᵢᵀ ∂ψ/∂ξᵢ = 0 (with field source terms if the cost
has a field integral). The boundary terms fix the adjoint wall BC; for inverse design,

    ψⱼ nⱼ = p − p_d   on the wall

(a transpiration condition on the momentum-component costates). The flow-state variation cancels;
the remaining metric-variation terms are reduced to a pure surface integral over the boundary
displacement, making the final derivative well-posed through shocks (integrated, not pointwise,
sensitivities) and independent of arbitrary interior mesh motion.

**Discrete adjoint**: discretize first, so R is the residual vector and ∂R/∂w the flow Jacobian; the
adjoint is the linear system (∂R/∂w)ᵀ ψ = ∂I/∂w with the transposed Jacobian, giving the exact
gradient of the discrete cost and a direct cell-by-cell comparison target for finite differences.

**Sobolev gradient (smoothness + preconditioning).** Raw L² steepest descent on a function-valued
shape reduces its smoothness by two classes per step (the gradient g = ∂F/∂y − (d/dx)∂F/∂y′ contains a
derivative of the shape), driving instability. Descend instead in a weighted Sobolev inner product
⟨u,v⟩ = ∫(uv + ε u′v′)dx: the smoothed gradient ḡ solves ḡ − ε ḡ″ = G, preserving the smoothness class
and still descending (δI = −λ⟨ḡ,ḡ⟩ < 0). It acts as a preconditioner — larger steps, far fewer design
cycles. Discretely, build the tridiagonal matrix M with diagonal 1+2ε and off-diagonals −ε, solve
Mḡ = G, and use −ḡ as the descent direction; in the source notation the smoothing map is A = M⁻¹.

**Lift constraint.** Minimize drag at a fixed target C_L (trim angle of attack each flow solve, carry
its sensitivity), with planform and minimum thickness held, so the optimizer cannot reduce drag by
shedding lift or growing span.

**Design cycle.** solve flow → solve adjoint → assemble G → (Sobolev-smooth) → descent step with
line search → repeat until ‖G‖ < tol.

## Code (quasi-1D Euler nozzle, discrete adjoint, inverse design)

```cpp
// Adjoint gradient: G = ∂I/∂α − ψᵀ ∂R/∂α,  with  (∂R/∂w)ᵀ ψ = ∂I/∂w
VectorXd gradient_adjoint(
    const int cost_function,
    const std::vector<double>& x, const std::vector<double>& dx,
    const std::vector<double>& area,
    const Flow_options& flo_opts, const Flow_data<double>& flow_data,
    const Optimization_options<double>& opt_opts, const Design<double>& design)
{
    const int n_resi = flo_opts.n_elem * 3;
    const int n_face = flo_opts.n_elem + 1;

    // shape parameterization: ∂(area)/∂(design vars)
    MatrixXd dAreadDes = evaldAreadDes(x, dx, design);

    // ∂I/∂w and direct ∂I/∂α  (cost = ½ Σ (p/p_t − p_d)² dx ; dCostdArea = 0 for inverse design)
    VectorXd dCostdW    = evaldCostdW(opt_opts, flo_opts, flow_data.W, dx);
    VectorXd dCostdArea = evaldCostdArea(flo_opts.n_elem);
    VectorXd dCostdDes  = dCostdArea.transpose() * dAreadDes;

    // ∂R/∂α  =  ∂R/∂(area) · ∂(area)/∂α
    MatrixXd dRdArea = evaldRdArea(flo_opts, flow_data);
    MatrixXd dRdDes  = dRdArea * dAreadDes;

    // ∂R/∂w : flow Jacobian from the ADOL-C residual trace
    SparseMatrix<double> dRdW = eval_dRdW_dRdX_adolc(flo_opts, area, flow_data);

    // one extra solve: adjoint equation, transpose of the flow Jacobian (N-independent)
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.compute(dRdW.transpose());
    VectorXd psi = solver.solve(dCostdW);

    // gradient w.r.t. all design variables in one cheap assembly
    VectorXd dIdDes = dCostdDes.transpose() - psi.transpose() * dRdDes;
    return dIdDes;
}

// Sobolev / implicit smoothing of the gradient: M ḡ = g, the discrete form of ḡ − ε ḡ″ = g
VectorXd implicitSmoothing(VectorXd g, double epsilon) {
    int n = g.size();
    MatrixXd M = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++)             M(i, i)   = 1.0 + 2.0 * epsilon;
    for (int i = 0; i < n - 1; i++) { M(i+1, i) = -epsilon; M(i, i+1) = -epsilon; }
    return M.llt().solve(g);
}

// Design loop: flow solve → adjoint gradient → descent step with Armijo backtracking
void optimizer(/* constants, x, dx, flo_opts, opt_opts, initial_design */) {
    Design<double> current_design = initial_design;
    std::vector<double> area = evalS<double>(current_design, x, dx);
    VectorXd searchD(opt_opts.n_design_variables);
    int it_design = 0;

    Flow_data<double> flow;
    quasiOneD(x, area, flo_opts, &flow);
    double cost = evalFitness(dx, flo_opts, flow.W, opt_opts);

    VectorXd g = getGradient(opt_opts.gradient_type, opt_opts.cost_function,
                             x, dx, area, flo_opts, flow, opt_opts, current_design);

    while (g.norm() > opt_opts.opt_tol && it_design < opt_opts.opt_maxit) {
        it_design++;
        VectorXd pk = -50 * g;              // steepest descent branch in the local optimizer
        cost = linesearch_backtrack_unconstrained(   // Armijo: new_cost <= cost + alpha*c1*g.dot(pk)
            1.0, x, dx, pk, g, cost, flo_opts, opt_opts, &searchD, &flow, &current_design);
        area = evalS(current_design, x, dx);
        g    = getGradient(opt_opts.gradient_type, opt_opts.cost_function,
                           x, dx, area, flo_opts, flow, opt_opts, current_design);
    }
}
```

The local implementation also includes `test_grad`, which compares `getGradient(1)` against forward
direct differentiation, `getGradient(2)`, and central finite differences, `getGradient(-3)`. The forward
path computes `dCostdDes + dCostdW·dWdDes` with `dWdDes = solve(−dRdW, dRdDes)`, so the minus sign in
the adjoint assembly is checked against the same residual linearization.
