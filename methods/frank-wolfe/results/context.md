# Context: minimizing a smooth convex function over a constraint set

## Research question

The problem is to minimize a continuously differentiable convex function over a constraint set,

  minimize f(x)  subject to  x ∈ D,

where D ⊆ Rⁿ is a compact convex set and f is convex with a gradient ∇f that we can evaluate. We
only have first-order access to f: at any feasible point x we can read off f(x) and ∇f(x), nothing
more. This is the workhorse setting behind least-squares with a constraint, support-vector training,
matrix-completion under a trace-norm budget, density estimation over a mixture simplex, and the
convex relaxations of combinatorial "sparsity" problems, where D is the convex hull of a structured
set of atoms (sparse vectors, low-rank matrices, permutation matrices).

What a solution must achieve is concrete. (i) Every iterate has to stay **feasible** — the gradient
points wherever it likes, generally straight out of D, so the raw step x − η∇f(x) leaves the set and
must somehow be brought back. (ii) Each iteration must be **cheap** on the domains that matter in
practice. The natural way to enforce feasibility is Euclidean projection onto D, but on a trace-norm
ball that projection is a full singular-value decomposition, on the Birkhoff polytope it is a
quadratic program over n! vertices, on a structured atomic norm it can be as hard as the original
problem — so a method whose per-step cost is dominated by projection inherits exactly the cost we are
trying to avoid. (iii) On these atomic domains we would like the **iterate itself to be cheap to
represent** — sparse, or low-rank — not a dense point in a huge ambient space. And (iv) since f(x*)
is unknown, we would like a **computable certificate** of how far the current point is from optimal,
to know when to stop.

## Background

**Convexity and the supporting linearization.** For a differentiable convex f, the first-order
condition says the tangent plane at any point lies below the graph:

  f(y) ≥ f(x) + ⟨y − x, ∇f(x)⟩   for all x, y.

This is the single most-used fact about convex functions: the linear model L_x(y) = f(x) + ⟨y − x,
∇f(x)⟩ is a global underestimator of f. Among many consequences, minimizing f over D has its optimum
characterized by the variational inequality ⟨x* − x, ∇f(x*)⟩ ≥ 0 for all x ∈ D.

**Smoothness (Lipschitz gradient) and the descent lemma.** A function is β-smooth in a norm ‖·‖ if
‖∇f(x) − ∇f(y)‖_* ≤ β‖x − y‖, with ‖·‖_* the dual norm. Smoothness gives the upper companion to the
convex lower bound (Nesterov, *Introductory Lectures on Convex Optimization*, 2004, Lemma 1.2.3):

  f(y) ≤ f(x) + ⟨y − x, ∇f(x)⟩ + (β/2)‖y − x‖².

So a smooth convex f is sandwiched between its tangent plane and a quadratic. This quadratic
over-estimate is what lets a first-order method take a definite step and *guarantee* a decrease,
rather than only hoping for one; it is the engine behind every O(1/k)-type rate for smooth problems.

**Extreme points and linear optimization over D.** A linear functional ⟨·, c⟩ minimized over a
compact convex set has an extreme-point minimizer — for a polytope, a vertex. The
operation

  s = argmin_{s ∈ D} ⟨s, c⟩,

minimizing a *linear* functional over D, is exactly linear programming when D is a polytope given by
linear constraints; it is the support function −Ω*_D(−c) of the set. On the structured domains above
it has closed forms or fast combinatorial routines: over an ℓ_p-ball it is closed-form (a single
coordinate or a normalized vector, via Hölder duality); over the simplex it is "pick the smallest
coordinate"; over the trace-norm ball it is a single top singular-vector pair; over the Birkhoff
polytope it is the Hungarian assignment algorithm in O(n³); over a submodular polyhedron it is
Edmonds' greedy algorithm in O(n log n). When a face of minimizers exists, it can be
tie-broken to return an extreme point.

**The cost-of-projection observation.** The empirical/structural fact that motivates everything: on
the domains above, the projection Π_D(y) = argmin_{z ∈ D} ‖z − y‖² is itself a nontrivial convex
program. For an Euclidean ball or an ℓ₁-ball it has a fast combinatorial solution, but for a
trace-norm ball it needs the full SVD, for a general polytope it is a QP, and for atomic/structured
norms it can be as expensive as the problem we set out to solve. Whenever the bottleneck of a
constrained first-order method is the projection, the method is only as practical as that projection
is cheap.

**Caratheodory and sparse representations.** Any point of the convex hull of a set of vertices is a
convex combination of at most n + 1 of them, and a point that is a convex combination of only k ≪ n
vertices is correspondingly sparse (or low-rank) relative to a generic point of D. There is also a
hardness floor relating sparsity to accuracy: for f(x) = ‖x‖² on the simplex Δ_n, any x supported on
k coordinates has f(x) − f(x*) ≥ 1/k − 1/n, so no scheme building its iterate from k vertices can do
better than O(1/k) accuracy at sparsity k.

## Baselines

**Projected (sub)gradient descent.** The standard constrained first-order method (analysis as in
Nesterov 2004; Bubeck, *Convex Optimization: Algorithms and Complexity*, 2014). It iterates a
gradient step followed by a projection back onto D:

  y_{t+1} = x_t − η ∇f(x_t),   x_{t+1} = Π_D(y_{t+1}),   Π_D(y) = argmin_{z ∈ D} ‖z − y‖².

For Lipschitz f it gives the dimension-free rate f(x̄_t) − f(x*) ≤ RL/√t (Θ(1/ε²) oracle calls); for
β-smooth f, projected gradient with η = 1/β converges at O(1/t). Its analysis is intrinsically
Euclidean — it leans on the non-expansiveness of the Euclidean projection, ‖Π_D(y) − x‖ ≤ ‖y − x‖ for
x ∈ D — and its rate constants depend on the Euclidean geometry of D. **Gap it leaves open:** the
per-step bottleneck is the projection, which on trace-norm balls, the Birkhoff polytope, and
structured atomic norms is a full QP/SVD — as costly as the original problem. The iterates are also
generically dense, with no built-in sparsity or low-rank structure, and the method offers no
by-product optimality certificate.

**Mirror descent / proximal methods.** Replace the Euclidean projection by a Bregman projection
adapted to the geometry of D (entropic mirror map on the simplex, etc.). This improves the dependence
on the geometry/dimension for some domains, but it still requires solving a prox/projection
subproblem each step, and on spectral or combinatorial domains that subproblem remains expensive.
**Gap it leaves open:** same structural issue — a nontrivial projection-like solve per iteration, and
no automatic sparse/low-rank iterate.

**Second-order and interior-point methods.** Newton-type and interior-point methods converge in very
few iterations (log(1/ε) accuracy) but require solving linear systems / handling the Hessian, and
their per-iteration cost and memory scale poorly with dimension; their behavior is tied to the
conditioning (the "distortion") of the problem. **Gap it leaves open:** not viable when n is large or
when D is only accessible through a linear oracle rather than an explicit constraint description.

**The original quadratic-programming setting.** Linear programming over polyhedra was, by the
mid-1950s, solved efficiently by the simplex method, but minimizing a *quadratic* objective subject
to linear inequality constraints had no comparably practical algorithm. **Gap it
leaves open:** the mature, efficient LP machinery does not apply directly to a nonlinear (quadratic)
objective, and no practical algorithm for the constrained-quadratic case had emerged from it.

## Evaluation settings

Natural problem classes for this setting include:

- **Constrained quadratic programs:** minimize a convex quadratic subject to linear inequality
  constraints (the polyhedral setting), where the linearized subproblem is an LP.
- **Constrained least squares / LASSO-type problems:** min ‖Y − Dx‖² subject to ‖x‖₁ ≤ s, i.e.
  optimization over an ℓ₁-ball, including the structured-dictionary case where the dictionary is
  huge (size exponential in n) but admits a polynomial-time linear oracle (e.g. incidence vectors of
  spanning trees, optimized by a greedy algorithm).
- **Optimization over the unit simplex** Δ_n: density estimation, boosting, support-vector / minimum
  enclosing ball problems, where the linear oracle is a single argmin over coordinates.
- **Trace-norm / nuclear-norm constrained matrix problems** (matrix completion, low-rank recovery),
  where the linear oracle is a top singular-vector computation and the alternative — projection —
  needs a full SVD.
- **Combinatorial polytopes:** the Birkhoff polytope of doubly-stochastic matrices (permutation
  problems), rotation matrices, submodular polyhedra.

The yardsticks are: convergence rate in the primal error f(x_k) − f(x*) as a function of iteration
count k, the cost of one iteration on each domain (projection/SVD vs. linear oracle), and the
sparsity / rank of the produced iterate.

## Code framework

The scaffold is a generic constrained-optimization harness: a smooth convex objective exposing value
and gradient, a feasible set exposing whatever oracle it can support, a step-size rule, and an outer
loop. The open slot is how to use ∇f to produce the next feasible iterate.

```python
import numpy as np

class SmoothConvexObjective:
    """A differentiable convex f: provides value and gradient (first-order oracle only)."""
    def value(self, x):
        raise NotImplementedError
    def grad(self, x):
        raise NotImplementedError

class FeasibleSet:
    """The compact convex constraint set D. Exposes whatever oracle(s) D can support.

    Euclidean projection is the classical primitive but may be as hard as the whole problem
    on many D. Implementations fill in whatever operations the domain affords.
    """
    def project(self, y):
        # argmin_{z in D} ||z - y||^2  -- may be expensive (QP / SVD) or unavailable.
        raise NotImplementedError

def step_size(k):
    # TODO: choose a step-size schedule for the outer loop.
    raise NotImplementedError

def minimize(objective: SmoothConvexObjective, domain: FeasibleSet, x0, max_iter):
    """Outer loop: from x_k and grad f(x_k), produce the next FEASIBLE iterate x_{k+1}.

    The body is the open question. The classical fill-in is gradient-step-then-project.
    """
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = objective.grad(x)
        # TODO: use g to move to the next feasible point.
        pass
    return x
```
