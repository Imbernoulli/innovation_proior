## Research question

Given a system of `m ≥ 2` linear inequalities in `n ≥ 2` real variables with **integer** coefficients,

```
A_i x ≤ b_i,   i = 1, …, m,        A_i ∈ ℤ^n,  b_i ∈ ℤ,
```

decide whether the system has a solution `x ∈ ℝⁿ`, with a running time bounded by a polynomial in the *encoding length* of the data — the number of binary digits needed to write the system down,

```
L = ⌈ Σ_{i,j} log₂(|a_ij| + 1) + Σ_i log₂(|b_i| + 1) + log₂(nm) ⌉ + 1.
```

Why it matters: this feasibility test is the computational core of linear programming. Maximizing an integer linear form `cᵀx` subject to `A x ≤ b` reduces to it — by binary search on the objective value `d`, "is `max cᵀx ≥ d`?" becomes "is `{A x ≤ b, cᵀx ≥ d}` feasible?". A polynomial-time feasibility test therefore makes LP itself solvable in time polynomial in the encoding length. At the time, no algorithm was known to do this in the worst case, and it was a prominent open problem whether one could exist. A solution must run in `poly(L)`, must be exact (no probability of error), and must handle systems with many constraints gracefully.

## Background

**Linear programming and duality.** A system `A x ≤ b` is infeasible iff, by Farkas' lemma, there is `y ≥ 0` with `yᵀA = 0` and `yᵀb < 0`. So feasibility has a short certificate *and* infeasibility has a short certificate: the problem sits in `NP ∩ co-NP`. Membership in `NP ∩ co-NP` was long read as strong circumstantial evidence that a problem ought to be in `P` — yet LP feasibility was one of the few natural problems known to be in `NP ∩ co-NP` for which no polynomial algorithm had been found.

**The simplex method and its worst case.** Dantzig's simplex method (1947) walks vertex-to-vertex along the boundary of the polyhedron `{A x ≤ b}`, improving the objective at each pivot. It is fast and reliable in practice. But Klee and Minty (1972) constructed perturbed hypercubes on which a natural pivoting rule visits all `2ⁿ` vertices: simplex is *exponential* in the worst case. Whether *some* variant of simplex, or some entirely different method, runs in polynomial worst-case time was open.

**The complexity question.** Karp's 1972 work on NP-completeness sharpened the stakes: a great many combinatorial problems were shown NP-complete, and the question was raised explicitly whether *linear-inequality feasibility* is NP-complete. If it were, and if it were also in `P`, that would force `P = NP`. So a polynomial algorithm for feasibility would simultaneously settle that this problem is *not* NP-complete (unless `P = NP`).

**Two localization facts that make `poly(L)` conceivable.** Integer data endows the problem with a scale.
- *A solution, if any, is not far out.* If `A x ≤ b` is feasible, then it has a solution inside the Euclidean ball `S = { x : ‖x‖ ≤ 2^L }`. This follows from the same determinant bounds that give polynomial-size rational certificates for linear systems. So one need only search a ball of radius `2^L`.
- *Infeasibility leaves a margin.* Define the residual `σ(x) = max_i { A_i x − b_i }`; then `x` solves the system iff `σ(x) ≤ 0`. If the system is **infeasible**, then for *every* `x ∈ ℝⁿ`, `σ(x) ≥ 2^{−L}`. Choosing `ε < 2^{−L}` separates the two cases: the relaxed body `P_ε = { x : A_i x ≤ b_i + ε for all i }` is empty when the original system is infeasible, while a feasible integer system gives `P_ε` positive volume inside the bounded search region, with a lower bound `2^{−O(nL)}` from the coefficient-size and determinant bounds. This relaxed full-dimensional body is what a volume-shrinking localization argument can use for a finite exact decision.

**Volume of an ellipsoid.** An ellipsoid is the affine image of a ball: `E = { x + Q z : ‖z‖ ≤ 1 }` for a center `x` and a nonsingular matrix `Q`, equivalently `E(A,a) = { x : (x−a)ᵀ A⁻¹ (x−a) ≤ 1 }` with `A = Q Qᵀ` positive definite. Its volume is `vol(E) = |det Q| · V_n = √(det A) · V_n`, where `V_n` is the volume of the unit ball. Ratios of volumes are preserved by affine maps — so volume statements can always be checked after pulling the ellipsoid back to the unit ball.

## Baselines

**Cutting-plane localization with polyhedral sets (Newman 1965; Levin 1965).** To minimize a convex (or quasiconvex) function, maintain a "localization set" known to contain the minimizer; query a subgradient at a point inside it; the subgradient inequality cuts the set with a hyperplane, discarding a region that cannot contain the minimizer; recurse on what remains. In these methods the localization set is a **polyhedron**, refined by adding the new half-space each step. Core gap: the number of inequalities describing the localization set **grows without bound** as the algorithm runs, so the work *per iteration* increases, and there is no clean dimension-only bound on how fast the set shrinks. There is no encoding-length analysis and no polynomial bound.

**Subgradient method with space dilation (Shor 1970).** For nonsmooth convex minimization, Shor introduced a subgradient method that *dilates the space* in the direction of the (sub)gradient between steps, i.e. rescales coordinates by a variable metric. This is a fixed-size update — it carries a matrix, not a growing list of constraints — and in nonlinear-optimization language it is a rank-one update of that matrix, closely analogous to a variable-metric quasi-Newton method. Shor (1977) gave the first explicit statement of the resulting ellipsoid construction. Gap as it stood: it was framed and analyzed as a method for *general convex* optimization with real data, aiming at approximate minimizers, with convergence stated geometrically; it carried no analysis tying the iteration count to the *bit-length* of integer linear-inequality data, and so did not by itself yield a polynomial-time *decision* procedure for feasibility.

**Method of central sections / its modification (Yudin & Nemirovski 1976).** Studying the informational complexity of convex optimization, Yudin and Nemirovski observed that Shor's dilation algorithm answers a question of Levin (1965) on the complexity of convex minimization, and gave an outline of an ellipsoidal central-section scheme: localize a convex body, query the center, cut through the center, re-enclose. Gap: again the target is general convex optimization and information-theoretic complexity, not the exact, integer-data, polynomial-in-`L` feasibility decision for linear systems; the rational-arithmetic bookkeeping needed to keep the construction finite and provably correct under rounding is not the object of study.

**Simplex (Dantzig 1947), as the yardstick to beat.** Walks the boundary; excellent in practice; exponential worst case (Klee–Minty 1972); its per-iteration and total cost depend on the number of constraints `m`. A new method would be measured against it on worst-case complexity and on insensitivity to `m`.

## Evaluation settings

The natural yardstick is *worst-case complexity as a function of the encoding length* `L` (and of `n`, `m`): number of arithmetic operations, the precision (bit-length) each operation must be carried out to, and total memory, as functions of `n`, `m`, `L`. The benchmark family for *exponential* behaviour of competing methods is the Klee–Minty cubes (perturbed hypercubes on which boundary-walking pivots visit all `2ⁿ` vertices). The relevant correctness criterion is an **exact decision** — feasible or infeasible — for arbitrary integer systems `A x ≤ b`, including the reduction of an LP optimization `max cᵀx s.t. A x ≤ b` to a sequence of such decisions via binary search on the objective value. A further setting of interest is feasibility/optimization over a polyhedron given **implicitly** — not by an explicit constraint list but by a subroutine that, at any queried point, either confirms membership or returns a centered separating normal — with the cost measured by the number of such subroutine calls plus arithmetic.

## Code framework

Available ingredients: dense linear algebra over rationals/fixed-point (matrix-vector products, symmetric matrix updates, Cholesky-type factorizations, exact integer arithmetic for the data, controlled-precision arithmetic with `√`), the encoding-length quantity `L`, and a callable that can expose a centered separating normal at a queried point. The open slots are the fixed-size localization loop and the objective-value wrapper.

```python
import numpy as np

# --- input and arithmetic primitives ------------------------------------
def as_integer_system(Arows, brhs):
    """Validate the integer system Arows x <= brhs and return numeric arrays."""
    # TODO: check dimensions, finiteness, and integrality of every coefficient.
    pass

def encoding_length(A, b):
    """L: number of binary digits to write the integer system A x <= b."""
    # TODO: compute the bit-length from the integer data.
    pass

def residual(A, b, x):
    """sigma(x) = max_i (A_i . x - b_i); x solves the system iff sigma(x) <= 0."""
    # TODO: evaluate the largest row violation.
    pass

# --- separation callable ------------------------------------------------
def separation_for_system(Arows, brhs, tol=0.0):
    """Return a callable that gives a center-valid separating normal, or None."""
    def separate(x):
        # TODO: pick a row with A_i x - b_i > tol.
        # TODO: return A_i, which satisfies A_i (y - x) <= 0 for every y in {A y <= b + tol}.
        pass
    return separate

# --- localization loop to be designed -----------------------------------
def decide_feasibility(separate, n, R, max_iters, inflate=1.0):
    """Decide whether the target set is nonempty.
    TODO: choose a fixed-size localization set containing any solution;
    TODO: query `separate` at a distinguished point of that set;
    TODO: after a violated inequality, replace the localization set by a
          smaller fixed-size set that still contains every solution;
    TODO: stop after the integrality floor rules out a nonempty set;
    TODO: carry out bounded-precision arithmetic without losing containment."""
    pass

def decide_linear_inequalities(Arows, brhs, max_iters=None, tol=None):
    """Decide whether Arows x <= brhs has a solution."""
    # TODO: validate integer data, compute L, choose the starting radius and iteration horizon,
    #       choose epsilon below the integer infeasibility gap,
    #       then call decide_feasibility with separation_for_system(Arows, brhs, tol).
    pass

def maximize_linear(base_separate, c, n, L, tol=None):
    """max c.x over the (implicitly described) feasible set.
    TODO: binary-search the objective value d, calling decide_feasibility
          with an oracle that returns either a base-set cut or the objective
          cut -c for {c.x >= d}, to the bit precision needed to recover
          the rational optimum."""
    pass
```
