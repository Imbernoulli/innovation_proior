# Context: minimizing convex functions that are not differentiable

## Research question

We want a general, provably convergent algorithm for the unconstrained problem

> minimize f(x), x ∈ R^n,

where f is **convex** but **not necessarily differentiable**. This is not a corner case. The convex
functions that show up in practice are routinely nonsmooth: the pointwise maximum of affine pieces
`f(x)=max_i (a_iᵀx+b_i)` (the objective of an LP in epigraph form, and of any minimax problem), the
ℓ1 norm `‖x‖₁` and other polyhedral penalties, the dual function of a constrained problem (a min of
affine functions, hence concave-nonsmooth — its negative is convex-nonsmooth), the maximum eigenvalue
`λ_max(A(x))` of a matrix that depends affinely on x, and any penalty built from `max`, `| · |`, or
`dist(·,C)`. At the points that matter — the kinks, which is often exactly where the minimizer sits —
there is no gradient.

A solution has to clear a high bar that the smooth methods cannot. It must (i) be defined at points
where f has no gradient, using only whatever first-order information *is* available there; (ii) come
with a convergence guarantee for the *whole* class of convex functions, not just nice ones — ideally
needing nothing beyond convexity and a Lipschitz bound; and (iii) be cheap enough per step to run on
very large problems where second-order methods are out of reach. The central difficulty is that the
two pillars the smooth theory rests on — "there is a descent direction" and "the function value
decreases each step" — both fail once f has a kink, and it is not obvious what replaces them.

## Background

**The smooth picture and why it breaks.** For differentiable convex f, gradient descent
`x_{k+1}=x_k−α∇f(x_k)` works because `−∇f(x)` is a *descent direction*: the directional derivative
`f'(x;−∇f) = −‖∇f‖² < 0`, so for small enough α (or with a line search) the value strictly decreases.
Convergence is proved by treating f(x_k) itself as the quantity that goes down. With Lipschitz
gradient the rate is O(1/ε) iterations to ε-accuracy. None of this survives at a kink: ∇f does not
exist there, and a gradient evaluated just to one side of a kink ignores the very corner the iterate
is approaching.

**The generalized first-order object: subgradients and supporting hyperplanes.** Convexity gives a
substitute for the gradient that exists even at kinks. A vector g is a **subgradient** of f at x if

> f(z) ≥ f(x) + gᵀ(z − x)  for all z,

i.e. the affine function on the right is a global underestimator of f — equivalently, `(g,−1)`
defines a supporting hyperplane to the epigraph `epi f` at `(x,f(x))`. The set of all such g is the
**subdifferential** ∂f(x), a closed convex set. Existence is a direct consequence of the supporting
hyperplane theorem applied to the convex set epi f at its boundary point `(x,f(x))`: a nonvertical
supporting hyperplane exists at every interior point of dom f, and its slope is a subgradient; if f
is continuous at x, ∂f(x) is also bounded. Where f is differentiable, `∂f(x)={∇f(x)}` — a single
element — so subgradients generalize the gradient. Where f has a kink, ∂f(x) is a whole set: for
`f=|x|`, `∂f(0)=[−1,1]`; for `f=max_i f_i`, `∂f(x)` is the convex hull of the gradients of the pieces
active at x; for `‖x‖₁`, `∂f(x)={g : ‖g‖_∞≤1, gᵀx=‖x‖₁}`.

Two facts about the subdifferential are load-bearing. First, optimality is characterized by
`0 ∈ ∂f(x*)` — a clean nonsmooth analogue of `∇f=0`. Second, the directional derivative of a convex
f is recovered from the subdifferential by

> f'(x;v) = sup_{g ∈ ∂f(x)} gᵀv.

This formula is the crux of the difficulty: because ∂f(x) is a *set*, picking one subgradient g and
stepping along −g gives `f'(x;−g) = sup_{h∈∂f(x)} hᵀ(−g) = −inf_{h∈∂f(x)} hᵀg`, and there can be an
`h∈∂f(x)` with `hᵀg<0`, making this supremum *positive*. In words: a subgradient need **not** be a
descent direction, and a step along its negative can *increase* f. The smooth proof strategy —
monotone decrease of the function value — is therefore unavailable in principle, not just in
practice.

**A second, monotone quantity.** What the subgradient inequality *does* give, at any minimizer x*, is
`f(x*) ≥ f(x) + gᵀ(x*−x)`, i.e. `gᵀ(x−x*) ≥ f(x)−f* ≥ 0`. So even when −g is not a descent direction
for f, it has a strictly positive component along the direction `(x*−x)` toward the optimum (whenever
x is suboptimal). That points at the **Euclidean distance to the optimal set**, `‖x−x*‖²`, as the
right thing to monitor: it is a candidate for the monotone-ish quantity that f itself cannot be.

**The intrinsic price of nonsmoothness.** The nonsmooth class is fundamentally harder than the
smooth one. For the black-box first-order oracle
model — at each queried point the algorithm receives f and one (adversarially chosen) subgradient,
and the next iterate must lie in the starting point plus the span of the subgradients seen so far —
there is a lower complexity bound. A "resisting oracle" built around the worst function
`f_{k+1}(x) = γ·max_{1≤i≤k+1} x^{(i)} + (μ/2)‖x‖²` reveals only one new informative coordinate per query,
so any method's iterates leave most coordinates untouched. This forces, for any first-order scheme,

> f(x_k) − f* ≥ M·R / (2(2+√(k+1)))

on functions that are M-Lipschitz on a ball of radius R about x*, with `R ≥ ‖x_0−x*‖`. That is an
Ω(1/√k) floor: ε-accuracy cannot be guaranteed in fewer than on the order of (MR/ε)² queries, no
matter how clever the method. Any method achieving O(1/√k) is therefore not just good — it is optimal
for this class.

**Step sizes are different here.** Because there is no usable line search (it would require f to
decrease along −g, which it may not), the step lengths must be chosen by an explicit rule rather
than by a decrease test. Several rules are standard: constant `α_k=α`; constant length
`α_k=γ/‖g_k‖`; the square-summable-but-not-summable family `Σα_k²<∞, Σα_k=∞` (e.g.
`α_k=a/(b+k)`); the nonsummable-diminishing family `α_k→0, Σα_k=∞` (e.g. `α_k=a/√k`); and
value-based rules that use a known or estimated optimal value f*. The pre-set schedules do not use
data computed during the run; the value-based rules use current oracle values but still avoid a
descent line search.

## Baselines

- **Gradient descent for smooth convex f** (the method this reacts to). Update
  `x_{k+1}=x_k−α_k∇f(x_k)`, with α_k fixed-and-small or set by backtracking line search. Works
  because `−∇f` is a descent direction and a line search enforces decrease; with Lipschitz gradient,
  O(1/ε) iterations to ε-accuracy, and the proof tracks f(x_k) downward. *Gap:* it requires f to be
  differentiable. At a kink ∇f does not exist; substituting an arbitrary subgradient destroys the
  descent property the convergence proof relies on, so the analysis does not transfer.

- **Newton / interior-point / second-order methods.** Fast (problem-scaling-independent, superlinear
  local rates) but they build a smooth local quadratic model and need ∇²f or a smooth barrier.
  *Gap:* inapplicable to nondifferentiable f, and the per-iteration linear-algebra cost and memory
  are prohibitive for very large problems even when f is smooth.

- **Cutting-plane / localization methods** (the conceptual cousins that *do* use subgradients).
  Each subgradient inequality `f(z) ≥ f(x_i)+g_iᵀ(z−x_i)` is a linear lower model of f; the
  optimality cut `g_iᵀ(x_i−x) ≥ 0` slices R^n into a half-space known to contain x*. Accumulating
  these cuts localizes x* in a shrinking polyhedron (center-of-gravity, ellipsoid). *Gap:* they store
  and manage a growing set of cuts and solve an auxiliary localization problem each step — heavy
  memory and per-iteration work — whereas the open question is whether something far simpler, keeping
  no history beyond the current point, can already converge.

- **Rule-based step sizes** as the alternative to line search. The constant-step, diminishing, and
  f*-based rules above are the candidate ingredients; the question a method must answer is which rule
  guarantees convergence for *every* Lipschitz convex f, and at what rate.

## Evaluation settings

The natural yardstick is the convergence of the *best objective value so far*,
`f_best^{(k)} − f*` (best, because the method is not a descent method, so the last iterate need not be
the best one found), plotted against iteration count k, on standard nonsmooth convex test problems
that existed independently of any new method:

- **Piecewise-linear minimization** `min_x max_{i=1..m}(a_iᵀx+b_i)` — the canonical nonsmooth test;
  exactly solvable by LP, so f* is available as ground truth. A representative instance: n=20
  variables, m=100 affine terms, data drawn from a unit normal, started at x_0=0.
- **ℓ1-regularized (lasso) and ℓ2-regularized (ridge) loss minimization** — e.g. logistic-regression
  loss plus a penalty `λ·P(β)` with `P=‖β‖₂²` (smooth, differentiable) or `P=‖β‖₁` (nonsmooth,
  needs subgradients); a representative size n=1000 examples, p=20 features.
- **Finding a point in an intersection of convex sets / solving convex feasibility**, posed as
  `min_x max_i dist(x,C_i)` with f*=0 attained iff a common point exists.

The relevant metrics are the suboptimality `f_best^{(k)}−f*` versus k (and versus the schedule
parameters), and the number of iterations to reach a target accuracy ε. The Lipschitz constant G
(an upper bound on subgradient norms) and a bound R on `‖x_0−x*‖` are the problem constants that any
rate would be expressed in.

## Code framework

The ingredients are an iterate, a first-order oracle that returns f and *one* subgradient at a
point, a step-size rule, and a running record of the best value.

```python
import numpy as np

def f_value(x):
    # objective value of the convex (possibly nonsmooth) f at x
    # TODO: problem-specific
    pass

def subgradient(x):
    # return ONE element g of the subdifferential of f at x.
    # at a kink the subdifferential is a set; any member is admissible
    # (the 'weak' calculus: e.g. for max_i f_i, pick a gradient of an active piece)
    # TODO: problem-specific
    pass

def step_size(k, fx, f_best, g):
    # choose a rule, not a backtracking line search:
    # constant / constant-length / square-summable-not-summable / diminishing / value-based
    # TODO: choose the schedule
    pass

def minimize(x0, num_iters):
    x = np.array(x0, dtype=float)
    f_best = np.inf
    x_best = x.copy()
    for k in range(1, num_iters + 1):
        fx = f_value(x)
        if fx < f_best:                 # function value can increase:
            f_best, x_best = fx, x.copy()  # keep the best point found so far
        g = subgradient(x)
        # TODO: fill in the move from x using g and step_size(k, fx, f_best, g).
        #       The quantity to control cannot be monotone decrease of f(x).
        pass
    return x_best, f_best
```
