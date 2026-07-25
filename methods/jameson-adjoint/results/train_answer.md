The method I am presenting is Jameson's adjoint method for aerodynamic shape optimization. The goal is to let a computational fluid dynamics solver drive a wing or duct shape toward an optimum instead of requiring a human to inspect and nudge each candidate. I parameterize the surface by a vector of design variables, which could be weights on bump functions, spline control points, or in the limit a free boundary. A scalar cost is extracted from the flow field: drag at a fixed lift coefficient, or an inverse-design pressure mismatch such as one half the integral of the squared difference between the surface pressure and a target pressure distribution. To minimize that cost with a gradient-based optimizer I need the gradient of the cost with respect to every design variable. The obstacle is that the flow solve is expensive, and the number of design variables can be large, so any procedure whose cost grows with the number of variables will hit a wall.

The brute-force approach is finite differences. I perturb each design variable, re-run the flow solver, measure the change in cost, and divide by the perturbation. That costs one baseline flow solve plus one additional flow solve per design variable, so N plus one solves per gradient. For a realistic wing with hundreds of design variables this is prohibitive, and it is also noisy because the flow solver only converges to a finite residual tolerance. When the pressure difference is tiny and the perturbation is tiny, the division amplifies solver noise. A more refined brute-force approach is forward, or tangent, sensitivity. The converged flow satisfies a residual equation, and differentiating that equation gives a linear system for the sensitivity of the flow state to each design variable. That removes the step-size dilemma, but it still requires one linear solve per design variable, so the scaling with N remains.

The adjoint idea is to rearrange the same linear algebra so that the expensive part is done only once. I write the change in cost under a shape change as the direct dependence of the cost on the design variables plus the indirect dependence through the flow response. The flow response is tied to the shape change by the linearized residual equation. Substituting that constraint into the cost variation gives a matrix product whose order I am free to choose. Computing it left-to-right means inverting the flow Jacobian once per design variable, which is the forward approach. Computing it right-to-left means solving a single linear system whose matrix is the transpose of the flow Jacobian and whose right-hand side is the cost sensitivity with respect to the flow state. That single solution defines the adjoint, or costate, vector. Once the adjoint vector is known, the gradient with respect to every design variable is obtained from cheap dot products. The cost of the gradient becomes one nonlinear flow solve plus one adjoint linear solve, independent of the number of design variables.

The same construction follows from a Lagrangian viewpoint. I add the residual constraint multiplied by an arbitrary multiplier to the cost, which does not change the value because the constraint is zero. I then choose the multiplier to eliminate the expensive flow-state variation from the cost variation. The equation that eliminates it is the adjoint equation, and the multiplier is the costate. In a time-dependent optimal control problem the costate is integrated backward in time in a single sweep; in this steady boundary-shape problem the analog is a single transposed linear solve. The adjoint matrix is the transpose of the flow Jacobian, which has the same sparsity and spectral properties as the flow Jacobian, so the same multigrid and multistage machinery that converges the flow also converges the adjoint at roughly the same cost.

There are two main routes to implementation. The continuous adjoint derives an adjoint partial differential equation from the continuous Euler equations before discretization. Multiplying the linearized flow equations by a costate field and integrating by parts yields a hyperbolic adjoint system whose characteristics run opposite to the flow. The boundary terms fix the wall boundary condition; for inverse design the normal momentum component of the costate equals the pressure mismatch. Because the derivation works with the integrated functional, it avoids the ill-posed pointwise pressure sensitivities that blow up when a shock sweeps across a fixed point, and the final gradient reduces to a surface integral that does not depend on arbitrary interior mesh motion. The discrete adjoint discretizes the equations first and differentiates the discrete residual vector directly. It gives the exact gradient of the discrete cost and is easier to validate against finite differences. The core equation is the same: the transpose of the flow Jacobian times the costate equals the cost sensitivity with respect to the state, and the design gradient is the direct cost sensitivity minus the costate dotted with the residual sensitivity with respect to the design variables.

A practical design loop also needs to keep the shape smooth. If I descend using the raw gradient in an L2 inner product, the gradient contains derivatives of the shape, so each step makes the boundary two smoothness classes rougher. That causes numerical instability and unphysical oscillations. Instead I descend in a weighted Sobolev inner product, which is equivalent to applying an elliptic smoothing operator to the gradient before taking the step. The smoothed gradient solves a Helmholtz-like equation with a small smoothing parameter, preserving smoothness while still guaranteeing descent. I also constrain the problem properly: minimizing drag without holding lift fixed would cause the optimizer to shed lift and produce a degenerate shape, so the cost is drag at a fixed target lift coefficient with planform and thickness constraints held.

The design cycle is therefore: solve the flow, solve the adjoint, assemble the gradient, smooth it, take a line-search step, and repeat. Each cycle costs about two flow solves regardless of how many design variables are used. That is the breakthrough of the method: high-dimensional shape optimization becomes affordable because the gradient is obtained by a single reverse-mode solve rather than by N forward perturbations.

I land all of this on a quasi-1D Euler nozzle, where the state per cell is the usual conserved triple and the shape is the duct area, so I can check every number against a known answer: I pose an inverse-design cost, one half the sum over cells of the squared mismatch between the local pressure ratio and a target distribution, and construct the target from a nozzle area I already know, so the true optimum is recoverable and the gradient has a ground truth to be checked against. `gradient_adjoint` assembles exactly the pieces the derivation calls for: it chains the shape parameterization through `evaldAreadDes` to get the area sensitivity to the design variables, evaluates the direct cost sensitivity `dCostdW`/`dCostdArea`, forms the residual sensitivity to area and hence to the design variables, builds the flow Jacobian `dRdW` from an ADOL-C trace of the residual, and then does the one extra solve — a sparse LU factorization of the *transpose* of that Jacobian against the right-hand side `dCostdW` — to get the costate ψ. The gradient against every design variable then falls out of a single cheap matrix–vector assembly, `dCostdDes − ψᵀ dRdDes`. `implicitSmoothing` is the discretized Sobolev preconditioner: it builds the tridiagonal matrix for `1 − ε∂²` and solves it against the raw gradient, which is what keeps the shape smooth across design cycles instead of roughening it. `optimizer` is the design loop itself: solve the flow, get the adjoint gradient, take a steepest-descent step under Armijo backtracking line search, re-solve, and repeat until the gradient norm falls below tolerance.

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

The local implementation also includes `test_grad`, which compares `getGradient(1)` against forward direct differentiation, `getGradient(2)`, and central finite differences, `getGradient(-3)`. The forward path computes `dCostdDes + dCostdW·dWdDes` with `dWdDes = solve(−dRdW, dRdDes)`, so the minus sign in the adjoint assembly is checked against the same residual linearization.
