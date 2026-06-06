OK, let me think this through from scratch. I want the computer to *design* an aerodynamic
shape, not just analyze one I hand it. The setup is: I have a flow solver. I give it a wing, or for
now a duct, it returns the flow field, and from the flow I can compute some scalar I care about —
drag at a fixed lift, or, simplest of all, how far the surface pressure is from a target distribution
p_d. I want to push the shape until I is minimized. To push downhill I need the gradient of I with
respect to the knobs that define the shape. Let me parameterize the shape by α — weights on a few
bump functions, or spline control points, so F(x) = Σ αᵢ bᵢ(x). The cost is I = I(α). I need dI/dα.

The brute-force way is staring me in the face. Perturb α₁ by a little δ, re-run the whole flow solve,
see how much I changed, divide. Repeat for α₂, α₃, … all N of them. So

    ∂I/∂αᵢ ≈ (I(αᵢ + δαᵢ) − I(αᵢ)) / δαᵢ.

This works. Hicks and Henne did exactly this in '78 with a full-potential solver and a
conjugate-gradient optimizer, and it designed wings. But count the flow solves: one baseline plus one
per design variable, so N+1 of them per gradient. And the flow solve is the expensive thing — a
transonic Euler solve, marched to steady state, is the entire unit of cost here; everything else is
noise next to it. With a handful of variables that's fine. But the shape I actually want lives in an
infinite-dimensional space of surfaces — a real wing wants hundreds of parameters, and morally the
boundary is a free function. N+1 transonic solves per gradient, every design cycle, is a wall. And
there's a second, nastier problem with finite differences even at small N: the step size. Make δ too
big and I'm contaminating the slope with curvature; make it too small and the signal I'δ shrinks while
the floor doesn't — the solver only converges R to a fixed tolerance, so each I carries a fixed-size
jitter, and dividing the tiny difference (I(α+δ) − I(α)) by a tiny δ amplifies that fixed jitter by
1/δ. So FD is both expensive *and* noisy. I need something fundamentally different.

Let me try to be cleverer but stay "forward." The reason FD costs N solves is that I'm re-converging
the nonlinear flow N times. What if I only linearize? The converged flow satisfies the constraint
R(w, α) = 0 — the steady residual is zero. Differentiate that w.r.t. one design variable:

    (∂R/∂w)(∂w/∂αᵢ) + ∂R/∂αᵢ = 0   ⟹   (∂R/∂w)(∂w/∂αᵢ) = −∂R/∂αᵢ.

That's a *linear* system in the flow sensitivity ∂w/∂αᵢ — solve it once (no re-convergence of the
nonlinear flow), then dI/dαᵢ = ∂I/∂αᵢ + (∂I/∂w)·(∂w/∂αᵢ). Cleaner than FD, exact, no step-size
dilemma. But… I still have to solve that linear system once for *each* αᵢ, because the right-hand
side −∂R/∂αᵢ is different for every design variable. So it's still N solves. I've traded N nonlinear
solves for N linear solves — better, but the scaling didn't change. The N is still there. Hmm.

Stare at the structure of what I'm computing. Write the cost change under a shape change δα as

    δI = (∂I/∂w)ᵀ δw + (∂I/∂α)ᵀ δα.

The second term is cheap — it's the *direct* dependence of the cost on the shape, evaluable on the
spot. The first term is the whole problem: δw, the flow's response to the shape change, is exactly the
expensive object, the thing that needs a flow-linearization solve per design variable. The
constraint, linearized, ties δw to δα:

    (∂R/∂w) δw + (∂R/∂α) δα = 0.

So δw = −(∂R/∂w)⁻¹ (∂R/∂α) δα, and if I substitute that into δI I get

    δI = [ (∂I/∂α)ᵀ − (∂I/∂w)ᵀ (∂R/∂w)⁻¹ (∂R/∂α) ] δα.

There it is, fully assembled. And now look at the order of operations. That middle piece is a
row vector times a matrix inverse times a matrix. I've been computing it left-to-right as
(∂R/∂w)⁻¹(∂R/∂α) first — that's N right-hand sides, the forward sensitivities, N solves. But matrix
products associate however I like. What if I compute it *right-to-left*? Group it as

    [ (∂I/∂w)ᵀ (∂R/∂w)⁻¹ ] (∂R/∂α).

The thing in brackets is one row vector solved against the inverse — a *single* linear system,
completely independent of how many design variables there are. Then I just dot the result into
(∂R/∂α), once per design variable, but that's a cheap matrix–vector product, not a flow solve. The N
moved out of the expensive part and into the cheap part. That's the whole trick, and it's just
associativity.

Let me make the single solve concrete and respectable. Define the vector ψ by

    (∂I/∂w)ᵀ (∂R/∂w)⁻¹ = ψᵀ   ⟺   (∂R/∂w)ᵀ ψ = ∂I/∂w.

So ψ solves a linear system whose matrix is the **transpose** of the flow Jacobian and whose
right-hand side is ∂I/∂w. One solve, no α in it anywhere. Then

    δI = (∂I/∂α)ᵀ δα − ψᵀ (∂R/∂α) δα,   i.e.   G = ∂I/∂α − ψᵀ (∂R/∂α),

and the gradient w.r.t. *all* design variables falls out of cheap dot products of the single ψ
against the columns of ∂R/∂α. Cost of the whole gradient: one nonlinear flow solve (to get w, hence
the Jacobian and ∂I/∂w), plus one linear solve for ψ, plus cheap assembly — and that's it,
**independent of N**. The number of design variables can be a hundred or infinite and the gradient
costs the same.

I can also reach the exact same ψ-system the Lagrangian way, which is reassuring and shows where it
comes from. The constraint R = 0 means I'm free to add any multiple of it to the cost without
changing anything: δR = (∂R/∂w)δw + (∂R/∂α)δα = 0, so for an arbitrary multiplier ψ,

    δI = (∂I/∂w)ᵀ δw + (∂I/∂α)ᵀ δα − ψᵀ[(∂R/∂w)δw + (∂R/∂α)δα]
       = [(∂I/∂w)ᵀ − ψᵀ(∂R/∂w)] δw + [(∂I/∂α)ᵀ − ψᵀ(∂R/∂α)] δα.

ψ is mine to choose. The δw bracket is the only place the expensive flow sensitivity appears, so I
choose ψ to annihilate it: (∂R/∂w)ᵀψ = ∂I/∂w. Same adjoint equation. Same gradient. The multiplier ψ
*is* the costate — it's exactly the Lagrange-multiplier field from Lions' optimal-control-of-PDEs
machinery, and the "solve once, get the gradient w.r.t. all controls" structure is precisely Bryson's
trajectory-optimization trick: there the costate is integrated backward in time in one sweep and
delivers the gradient against every control at once; here the steady analog is one backward/transposed
solve delivering the gradient against every shape parameter at once. Pironneau already did this for
fluids — but for Stokes flow, an elliptic, low-speed, shock-free system. The open thing is to carry it
into the nonlinear, hyperbolic, transonic Euler equations with shocks.

Before I trust the cheap part I should sanity-check *why* the adjoint solve is itself affordable —
otherwise I've just hidden the cost. The adjoint matrix is (∂R/∂w)ᵀ, the transpose of the flow
Jacobian. The flow Jacobian is the thing my fast solvers already know how to handle; its transpose has
the same sparsity, the same spectral magnitude, the same hyperbolic character — it's the same wave
system run with the waves travelling in the *reverse* direction. So whatever multigrid / multistage
machinery converges the flow converges the adjoint at comparable cost. One design cycle ≈ two flow
solves: solve the flow, solve the adjoint, assemble. Two solves total, regardless of N. That's the
prize.

Now I have a fork. I derived all of this on the *discrete* residual vector R — the flow Jacobian
∂R/∂w is a finite matrix and the adjoint is a transposed linear solve. That gives me the exact
gradient of the *discretized* cost, which is what an optimizer actually descends, and it's
mechanical: form the Jacobian, transpose it, solve. That's the discrete adjoint, and it's what I'll
code. But there's another route. Suppose I do all of this on the *continuous* PDE first — linearize
the Euler PDE, introduce ψ as a costate *field*, integrate by parts, and read off an adjoint *PDE*
plus its boundary conditions; only then discretize ψ. Why would I bother with the harder continuous
route?

Two reasons surface, and they're exactly the two phenomena I flagged at the start. First, the
ill-posed pointwise sensitivity. If I naively ask for the sensitivity of pressure *at a fixed point*
to the shape, a shape change can sweep a shock across that point, and ∂p/∂F there is unbounded. But
drag is an *integral* of pressure over the surface, and the shock moves continuously with the shape,
so the integrated cost is smooth even where the pointwise sensitivity is not. The continuous adjoint,
because it works with the functional and its boundary integral, computes the sensitivity of the
*integrated* quantity directly and never has to form the ill-posed pointwise sensitivities. Second,
when I integrate by parts in the continuous derivation, the state-variation terms can be cancelled and
the remaining metric terms can be reduced to a boundary displacement. That makes the final gradient
independent of an arbitrary interior volume-mesh motion, which is a gift on overset and unstructured
grids. So the continuous route buys well-posedness and mesh-agnosticism; the discrete route buys an
exact discrete gradient and a mechanical implementation. I'll derive the continuous adjoint to
understand the structure, then build the discrete one because that's what validates cleanly against
finite differences.

Let me actually do the continuous Euler adjoint to see the boundary condition emerge, because that's
the part the abstract Lagrangian hides. Steady Euler: ∂fᵢ/∂xᵢ = 0 in D. Map to a fixed computational
domain ξ so the geometry change shows up in the metrics rather than in the domain. Write the
transformed fluxes Fᵢ = Sᵢⱼ fⱼ with S the metric cofactors (the projected face areas). Linearize:
δ(∂Fᵢ/∂ξᵢ) = 0, where δFᵢ splits into a part from δw through the flux Jacobian Cᵢ = Sᵢⱼ ∂fⱼ/∂w and a
part from the metric change δSᵢⱼ fⱼ (the shape moving the mesh). The cost is a boundary integral, say
the pressure mismatch on the wall. Multiply the linearized PDE by a costate field ψ, integrate over
the domain, and subtract that zero constraint contribution from δI:

    ∫ ψᵀ ∂(δFᵢ)/∂ξᵢ dD.

After integration by parts, the subtracted constraint contributes the volume term
∫ (∂ψᵀ/∂ξᵢ) δFᵢ dD. I want the δw piece of that term to vanish in the interior, which forces the
**adjoint field equation**

    Cᵢᵀ ∂ψ/∂ξᵢ = 0   in D

(with a source term added if the cost has a field-integral part — those terms are unrestricted, they
just get absorbed as sources). That's a hyperbolic system in ψ; its characteristics are the flow's run
backward, confirming the "reverse waves" picture. The integration by parts also throws off a
*boundary* term, ∫_B nᵢ ψᵀ δFᵢ dB, and this is where the boundary condition is born. On the wall the
flow satisfies no-penetration U₂ = 0, so δU₂ = 0 under a shape change, and the only surviving piece of
δFᵢ at the wall is the pressure variation δp pushing on the momentum components. The boundary integral
then carries a term in δp from the flow side and a term in (p − p_d)δp from the cost side. To cancel
the dependence on the unknown δw at the boundary, I must make the costate kill the δp coefficient, and
that pins down the **adjoint wall boundary condition**

    ψⱼ nⱼ = p − p_d   on the wall

— a transpiration condition on the momentum-component costates, with the cost's target-pressure
mismatch as the forcing. (It imposes nothing on ψ's tangential component.) Beautiful: the cost
functional doesn't enter the adjoint *interior* equation at all for inverse design; it enters only
through this wall BC. And once ψ solves its PDE with this BC, the flow-state variation is gone. What
remains is a metric-variation expression with the flow held fixed,

    δI = δI_II + ∫ ψᵀ δR_II dD,

and the reduced-gradient step then integrates the interior metric dependence along coordinate lines
emanating from the wall, so it becomes a pure boundary-displacement integral,

    δI = ∫_B G δF dB.

So G is a function defined over the boundary, and the final derivative does not depend on an arbitrary
choice of interior mesh motion. Good. The continuous picture is consistent with the abstract one: ψ
from a single transposed/backward solve, gradient from a cheap boundary assembly.

Now, descent. Naively I just step α ← α − λG. But there's a trap I should think about before I
iterate, and it's the calculus-of-variations smoothness issue. Take the toy I = ∫ F(y, y′) dx. Under
δy, δI = ∫ [∂F/∂y − (d/dx)(∂F/∂y′)] δy dx = (g, δy) in the L² inner product, so g = ∂F/∂y −
(d/dx)(∂F/∂y′). Look at g: it contains a *derivative* of the integrand's y′-dependence, so g is two
derivatives less smooth than y. If I set δy = −λg, then each descent step adds something two
smoothness classes rougher than the current shape. Iterate that and the shape gets rougher and
rougher — the surface develops oscillations, the flow solver struggles, the optimization goes
unstable. So raw L² steepest descent on a function-valued control is self-roughening. That's not a
solver bug; it's intrinsic to using the L² gradient as a *shape increment*.

The fix is to stop insisting the gradient live in L². The gradient is defined by δI = ⟨g, δy⟩ — but
the inner product is mine to choose, and the *direction of steepest descent depends on the metric*. If
I use a weighted Sobolev inner product

    ⟨u, v⟩ = ∫ (u v + ε u′ v′) dx

instead of L², then I define a *new* gradient ḡ by δI = ⟨ḡ, δy⟩. Equating ⟨ḡ, δy⟩ = (g, δy) for all
δy and integrating the ε-term by parts gives the smoothing equation

    ḡ − ε ḡ″ = g.

ḡ is g passed through an elliptic smoothing operator — it's *smoother* than g, not rougher, so the
step δy = −λḡ no longer degrades the smoothness class. And it's still genuinely downhill:
δI = ⟨ḡ, −λḡ⟩ = −λ⟨ḡ, ḡ⟩ < 0. So I keep descent and I keep smoothness. As a bonus this acts like a
preconditioner — the smoothing damps the high-frequency components of the raw gradient that were
forcing tiny steps, so I can take much larger λ and converge in far fewer design cycles. In the
discrete code this is: build the tridiagonal matrix M for (1 − ε ∂²), with diagonal 1+2ε and
off-diagonals −ε, solve Mḡ = G, then step δα = −λḡ. If I call the smoothing map A, then A = M⁻¹;
the local code builds M and applies A by a small linear solve, negligible next to a flow solve.

One more design decision before I trust the loop on a real wing: what cost? If I minimize drag with
nothing held fixed, the optimizer will discover that induced drag — a big chunk of total drag — drops
if it just sheds lift or stretches the span, and it'll happily hand me a barely-lifting wing with low
drag. Useless. So I minimize drag **at a fixed target C_L**: each flow solve trims the angle of attack
to hit the target lift, and I carry the angle-of-attack sensitivity into the gradient; I also fix the
planform and a minimum thickness so span and volume can't run away. The cost has to encode what I
actually want, and "low drag" alone doesn't.

Now let me land this on real code, on the quasi-1D nozzle where I can check every number. State w =
(ρ, ρu, e) per cell; the control is the duct area S(x), parameterized by a few design variables; the
flow solver marches R(w,S)=0 to steady state. Cost is inverse design, I = ½ Σᵢ (pᵢ/p_t − p_{d,i})²
Δxᵢ, so the known optimum is the area that produced p_d — which means I can check the gradient against
a known answer. The pieces I need to assemble the adjoint gradient G = ∂I/∂α − ψᵀ ∂R/∂α, with
(∂R/∂w)ᵀ ψ = ∂I/∂w:

```cpp
VectorXd gradient_adjoint(
    const int cost_function,
    const std::vector<double>& x, const std::vector<double>& dx,
    const std::vector<double>& area,
    const Flow_options& flo_opts, const Flow_data<double>& flow_data,
    const Optimization_options<double>& opt_opts, const Design<double>& design)
{
    const int n_elem = flo_opts.n_elem;
    const int n_resi = n_elem * 3;
    const int n_face = n_elem + 1;

    // ∂(area)/∂(design vars): chain through the shape parameterization
    MatrixXd dAreadDes = evaldAreadDes(x, dx, design);

    // ∂I/∂w  — cost is 1/2 sum (p/p_t - p_d)^2 dx, and p = (γ-1)(e - ½ρu²),
    //          so this is (p/p_t - p_d) * (∂p/∂w) * dx / p_t, cell by cell
    VectorXd dCostdW = evaldCostdW(opt_opts, flo_opts, flow_data.W, dx);

    // ∂I/∂α  — the *direct* shape dependence of the cost.
    //          For inverse design the cost depends on area only through w, so dCostdArea = 0,
    //          but keep the term for generality (drag costs do depend on area directly).
    VectorXd dCostdArea = evaldCostdArea(n_elem);
    VectorXd dCostdDes  = dCostdArea.transpose() * dAreadDes;

    // ∂R/∂α  — how the steady residual depends on the shape: chain ∂R/∂(area) · ∂(area)/∂α
    MatrixXd dRdArea = evaldRdArea(flo_opts, flow_data);
    MatrixXd dRdDes  = dRdArea * dAreadDes;

    // ∂R/∂w  — the flow Jacobian used by this implementation (ADOL-C trace of the residual)
    SparseMatrix<double> dRdW = eval_dRdW_dRdX_adolc(flo_opts, area, flow_data);

    // ----- the one extra solve: adjoint equation (∂R/∂w)ᵀ ψ = ∂I/∂w -----
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.compute(dRdW.transpose());            // *transpose* of the flow Jacobian — reverse waves
    VectorXd psi = solver.solve(dCostdW);        // independent of the number of design variables

    // gradient w.r.t. ALL design variables, one cheap assembly:  G = ∂I/∂α − ψᵀ ∂R/∂α
    VectorXd dIdDes = dCostdDes.transpose() - psi.transpose() * dRdDes;
    return dIdDes;
}
```

The sign is worth pinning down against the forward route, because it's the easy place to flip. Forward
differentiation builds the flow sensitivity from (∂R/∂w)(∂w/∂α) = −∂R/∂α, i.e. ∂w/∂α = −(∂R/∂w)⁻¹
∂R/∂α, and then dI/dα = ∂I/∂α + (∂I/∂w)ᵀ ∂w/∂α = ∂I/∂α − (∂I/∂w)ᵀ(∂R/∂w)⁻¹ ∂R/∂α. The adjoint just
regroups the last term as −ψᵀ ∂R/∂α with ψ = (∂R/∂w)⁻ᵀ ∂I/∂w — *same minus sign*. So in code the
forward path is `dCostdDes + dCostdW·dWdDes` where `dWdDes = solve(−dRdW, dRdDes)` (the minus folded
into the solve), and the adjoint path is `dCostdDes − psiᵀ·dRdDes`. They should agree within the
linearization and solver tolerances, and that equality, plus agreement with central FD, is exactly the
gradient test I'll run:

```cpp
adjoint_gradient = getGradient(1,  /* ... */);   //  ∂I/∂α − ψᵀ ∂R/∂α
direct_gradient  = getGradient(2,  /* ... */);   //  forward sensitivity, one solve per design var
cfinite_gradient = getGradient(-3, /* ... */);   //  central FD, 2N flow solves, the ground truth
// print (g_adj - g_dir)/g_adj and (g_adj - g_FD)/g_adj  — all should be ~0
```

And the design loop: solve flow, get the adjoint gradient, condition it (steepest descent here scales
the gradient; the Sobolev/implicit-smoothing solve is available to precondition it), take an
Armijo backtracking step, re-solve, repeat until ‖G‖ falls below tolerance.

```cpp
quasiOneD(x, area, flo_opts, &flow_data);
double cost = evalFitness(dx, flo_opts, flow_data.W, opt_opts);
VectorXd g  = getGradient(opt_opts.gradient_type, opt_opts.cost_function,
                          x, dx, area, flo_opts, flow_data, opt_opts, current_design);
while (g.norm() > opt_opts.opt_tol && it < opt_opts.opt_maxit) {
    VectorXd pk = -lambda * g;                  // steepest descent (or pk = -implicitSmoothing(g, eps))
    cost = linesearch_backtrack(/* Armijo: new_cost <= cost + alpha*c1*g.dot(pk) */);
    area = evalS(current_design, x, dx);
    g    = getGradient(opt_opts.gradient_type, opt_opts.cost_function,
                       x, dx, area, flo_opts, flow_data, opt_opts, current_design);
}
```

where the implicit-smoothing preconditioner is exactly the discretized (1 − ε∂²):

```cpp
VectorXd implicitSmoothing(VectorXd g, double epsilon) {
    int n = g.size();
    MatrixXd M = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++)            M(i,i)   = 1.0 + 2.0*epsilon;   // M ḡ = g
    for (int i = 0; i < n-1; i++) { M(i+1,i) = -epsilon; M(i,i+1) = -epsilon; }
    return M.llt().solve(g);               // applies M^{-1}; ḡ is smoother than g
}
```

So the causal chain, end to end: I want dI/dα to drive a shape downhill; finite differences cost N+1
flow solves and are noisy, and forward sensitivities still cost one solve per variable — the N never
leaves the expensive part. Treating the converged flow R(w,α)=0 as a constraint and writing the cost
variation as δI = (∂I/∂w)δw + (∂I/∂α)δα, the expensive object is δw; adding ψᵀδR (which is zero) and
choosing ψ to cancel the δw term turns the gradient into (∂R/∂w)ᵀψ = ∂I/∂w plus a cheap assembly
G = ∂I/∂α − ψᵀ∂R/∂α — one transposed solve, gradient against *all* design variables at once,
independent of N, ≈ two flow solves per design cycle. The transpose is the flow's waves run backward,
so the same fast solvers apply; on the continuous PDE the costate gives an adjoint hyperbolic system
whose wall BC ψⱼnⱼ = p − p_d carries the inverse-design cost, and the gradient reduces to a
mesh-independent surface integral that stays well-posed even through shocks. Stepping with the raw L²
gradient would roughen the shape two smoothness classes per cycle, so I descend in a Sobolev metric —
ḡ − εḡ″ = g — which keeps the shape smooth and preconditions the step; and I minimize drag at fixed
lift so the optimizer can't cheat by shedding it. The whole thing is two solves a cycle, no matter how
many design variables.
