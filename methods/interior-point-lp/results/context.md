# Context: Solving Linear Programs by Moving Through the Interior

## Research question

Given a linear program in standard form

```
minimize    cᵀx
subject to  Ax = b,  x ≥ 0,     A ∈ ℝ^{m×n},
```

find an algorithm that is **both** provably polynomial-time **and** fast in practice. By the early 1980s these two requirements pointed at completely different algorithms, and no single method delivered both. A satisfactory solution would have to (a) come with a worst-case bound polynomial in `n` and in `L`, the bit-length of the data, and (b) actually beat the simplex method on the large sparse problems that arise in operations research.

## Background

A linear program's feasible region is a polyhedron. The optimum, if it exists, is attained at a vertex. Two facts about the geometry frame everything:

- The number of vertices can be exponential in `n`. Any method that visits vertices risks visiting exponentially many.
- The interior of the polyhedron is a smooth convex set, on which the objective `cᵀx` is linear. The combinatorial vertex picture and the smooth convex picture are very different settings for an algorithm.

**The barrier idea (Frisch 1955; Fiacco–McCormick 1968).** To keep an iterate strictly inside `{x > 0}` one replaces the hard constraint by the **logarithmic barrier**

```
φ(x) = −Σⱼ ln xⱼ,
```

which is finite on the open orthant and `→ +∞` as any `xⱼ → 0⁺`. Minimizing the **barrier family**

```
F_t(x) = t·cᵀx + φ(x),     t > 0,
```

over `{x : Ax = b, x > 0}` gives, for each `t`, a unique strictly feasible minimizer `x*(t)`. As `t → ∞` the linear term dominates and `x*(t)` approaches the optimum. The set `{x*(t) : t > 0}` is a smooth curve through the interior — a path one could try to follow. Fiacco and McCormick's *Sequential Unconstrained Minimization Technique* (SUMT) packaged exactly this. These methods were among the most successful for nonlinear constrained optimization in the 1960s.

**Why the barrier fell out of favor.** Classical analysis of SUMT predicted that the subproblems become **increasingly ill-conditioned** as `t → ∞`: near the boundary the Hessian of `φ` blows up, so the unconstrained minimization slows down and accumulates round-off. By the late 1960s and 1970s the log-barrier had lost favor for precisely this reason. Crucially, **no polynomial-time guarantee was ever claimed** for it; the prevailing wisdom was that following the barrier to the end was numerically treacherous.

**Newton's method.** Newton's method solves `∇f(x) = 0` by the iteration `x⁺ = x − [∇²f(x)]⁻¹∇f(x)` and converges quadratically near a nondegenerate minimizer. The Newton step itself is affine-invariant: under a linear change of coordinates the iterates map over exactly. The classical statement of *where* quadratic convergence begins, however, is stated in terms of the condition number of `∇²f(x*)` and the Lipschitz constant of `∇²f`.

**Local rescaling (Dikin 1967).** A separate thread: at a strictly interior point `x̄`, inscribe the ellipsoid `{x : Σⱼ(xⱼ−x̄ⱼ)²/x̄ⱼ² ≤ 1}` (large where coordinates are large, small where they are small, so it stays inside the orthant) and take a steepest-descent step of the objective inside it. This affine scaling rescales the problem so that the current point sits at the "center" of a round body, where a gradient step is well-behaved.

**The complexity backdrop.** A polynomial bound is stated in terms of `L`, the number of bits needed to write the data; `L ≥ log(1 + |D_max|) + log(1 + max|cᵢ|)` where `D_max` is the largest subdeterminant of the constraint matrix. Two facts about LP at this scale: any two distinct vertex objective values differ by at least `2^{−kL}`, so once an interior point is within `2^{−L}` of optimal one can round to an exact optimal vertex; and representing the output can need `O(L)` bits per variable, so a factor of `L` must appear in any LP algorithm's complexity.

## Baselines

**Simplex method (Dantzig 1947).** Maintains a basic feasible solution at a vertex and pivots to an adjacent vertex that improves the objective, until no improving edge remains. Each pivot is cheap and exploits sparsity; in practice it solves enormous problems in roughly a linear number of pivots. *Gap:* its worst case is exponential. Klee and Minty (1972) constructed a deformed cube on which Dantzig's pivot rule visits all `2ⁿ` vertices. So simplex is not polynomial-time, and its behavior is tied to the combinatorial structure of the boundary.

**Ellipsoid method (Khachiyan 1979).** The first polynomial-time algorithm for LP. It brackets the feasible region in an ellipsoid and, at each step, uses a violated constraint (a separating hyperplane through the center) to produce a smaller ellipsoid containing the feasible set, shrinking the volume by a fixed factor. After `O(n²L)` steps the ellipsoid is too small to contain a feasible point unless one has been found; total `O(n⁶L²)` arithmetic on `O(L)`-bit numbers. *Gap:* it is hopeless in practice — the constants are enormous, it always runs near its worst case, the iterates do not exploit sparsity, and round-off errors **accumulate** so the precision needed grows with the iteration count. It proved LP ∈ P but did not threaten simplex on real problems.

**Sequential barrier minimization / SUMT (Fiacco–McCormick 1968), building on Frisch's 1955 log barrier.** Minimize `F_t(x) = t·cᵀx − Σ ln xⱼ` for an increasing sequence of `t`, warm-starting each minimization from the previous solution. *Gap:* analyzed as a generic unconstrained-minimization sequence with no exploitation of the barrier's special structure; the theory predicted slowdown and ill-conditioning as `t → ∞`, and offered no polynomial bound. Whether the path could be followed *efficiently and provably* was open.

**Affine scaling (Dikin 1967).** Rescale by `D = diag(x̄)`, take a projected steepest-descent step of `cᵀx` inside the inscribed ellipsoid, unscale, repeat. *Gap:* a clean interior iteration, but no polynomial-time guarantee; it can stall or behave badly without careful safeguards.

## Evaluation settings

The natural yardstick is the standard-form LP `min cᵀx s.t. Ax = b, x ≥ 0` (and its inequality form `min cᵀx s.t. Ax ≤ b`), with the constraint matrix `A ∈ ℝ^{m×n}` typically large and **sparse** as in operations-research and network applications. The metrics that matter: worst-case arithmetic-operation count as a function of `n` and `L`; iteration count to reduce the duality gap below a tolerance `ε`; and, for practice, wall-clock time and iteration count on benchmark LPs against the simplex method, where exploiting sparsity in whatever linear-algebra kernel the iteration relies on is decisive. Numerical robustness — whether round-off accumulates or is self-correcting across iterations — is part of the comparison.

## Code framework

The pre-existing pieces are dense/sparse linear algebra (matrix factorization and triangular solves) and the standard-form LP data `(A, b, c)`. Everything specific to an interior method — how to choose a strictly feasible start, what direction to move, how far, and when to stop — is left as empty slots.

```python
import numpy as np

def solve_lp(A, b, c, tol=1e-8, max_iter=100):
    """min cᵀx s.t. Ax = b, x ≥ 0.  Return x (and dual y, s)."""
    m, n = A.shape

    # Strictly feasible / strictly interior starting point.
    x, y, s = initial_point(A, b, c)          # TODO

    for k in range(max_iter):
        # Measure of optimality from the current interior point.
        gap = duality_measure(x, y, s)        # TODO
        if converged(A, b, c, x, y, s, tol):  # TODO
            break

        # The search direction: what does "toward the optimum, through
        # the interior" mean concretely?  This is the slot the method fills.
        dx, dy, ds = search_direction(A, b, c, x, y, s)   # TODO

        # How far to move while staying strictly interior.
        ax, as_ = step_length(x, s, dx, ds)   # TODO

        x = x + ax * dx
        y = y + as_ * dy
        s = s + as_ * ds

    return x, y, s


def initial_point(A, b, c):
    raise NotImplementedError   # TODO

def duality_measure(x, y, s):
    raise NotImplementedError   # TODO

def converged(A, b, c, x, y, s, tol):
    raise NotImplementedError   # TODO

def search_direction(A, b, c, x, y, s):
    raise NotImplementedError   # TODO  <-- the heart of the method

def step_length(x, s, dx, ds):
    raise NotImplementedError   # TODO
```
</content>
