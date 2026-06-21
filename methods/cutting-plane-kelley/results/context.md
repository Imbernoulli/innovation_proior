# Context: Minimizing a convex, possibly nonsmooth, function over a convex set

## Research question

We are handed a convex programming problem: minimize a continuous convex function over a
closed convex set,

$$ \min_{x}\ F(x) \qquad \text{s.t.}\qquad x \in H, $$

with $H \subseteq \mathbb{R}^n$ closed convex and $F$ continuous convex. The set may instead be
written through convex constraints, $x \in H$ replaced by $G(x) \le 0$ with $G$ convex; and
$F$ itself may be **nondifferentiable** — the canonical case being a finite maximum,
$F(x) = \max_i g_i(x)$, which arises directly in Chebyshev (minimax) curve fitting,
$\min_x \max_i \lvert a_i^\top x - b_i\rvert$.

Constrained-minimum problems of this kind are of perennial interest, and convex ones are the
one large class where there is real hope: with convexity, every local minimum is a global
minimum, so a method that gets "stuck" is still correct. The question is how to compute a
solution on the machines of the day under weak assumptions — continuity and convexity, not
smoothness, since the objectives we care about have kinks. Compactness of the feasible set is,
for practical purposes, no restriction when a finite minimum exists: if the set were only
closed, the inquiry can be confined to a compact convex subset containing the minimizer.

## Background

The load-bearing facts available at this point are these.

**Convexity and the first-order (support) inequality.** A differentiable convex function lies
above each of its tangents: $F(z) \ge F(t) + \nabla F(t)^\top (z - t)$ for all $z$. When $F$ is
not differentiable, the gradient is replaced by a **subgradient** $g \in \partial F(t)$, defined
exactly by the property that the affine function $F(t) + g^\top(z-t)$ underestimates $F$
everywhere: $F(z) \ge F(t) + g^\top (z - t)$ for all $z$. Geometrically this affine minorant is
a **supporting hyperplane** to the graph of $F$ at $t$. Such a support exists at every point of
a convex function (in the interior of its domain), and for a max-type function
$F = \max_i g_i$ a subgradient at $t$ is simply $\nabla g_h(t)$ for any index $h$ active at $t$
(or a convex combination of the active gradients) — so the first-order information is available
even where there is no gradient. On a compact working set the support slopes are
bounded, $\lVert g\rVert \le K < \infty$.

**The supporting-hyperplane / separation picture.** A closed convex set is the intersection of
all the halfspaces containing it; equivalently, a point not in a closed convex set can be
strictly separated from it by a hyperplane. So a convex feasible region $R = \{x : G(x)\le 0\}$
is, in principle, an intersection of (possibly infinitely many) linear inequalities — its
supporting halfspaces. This is the same content as the support inequality applied to the convex
constraint $G$.

**Outer approximation.** A standard idea for a hard set is to enclose it in a simpler set and
refine. If $R$ is convex, any finite collection of its supporting halfspaces forms a polyhedron
$P \supseteq R$; adding more supports shrinks $P$ toward $R$. The dual idea — replacing a curved
convex surface by its support planes during computation — is the rationale of Newton's method
read geometrically, and the basic idea of approximating a continuum of linear inequalities by a
finite subset goes back to Remez's exchange procedure for Chebyshev approximation and was used,
in one form or another, by a number of authors (Novodvorskii–Pinsker, Beale, Stiefel, Wolfe,
Stone).

**Linear programming is a mature technology.** By the late 1950s, LP is a reliable, large-scale
optimization technology. The simplex method (Dantzig, 1947) and especially the **revised
simplex / generalized simplex** method for minimizing a linear form under linear inequality
constraints (Dantzig, Orden, Wolfe, *Pacific J. Math.* 5 (1955) 183–195), together with the
**dual method** for the LP problem (Lemke, *Naval Res. Logist. Quart.* 1 (1954) 36–47), solve
LPs efficiently, and the revised simplex machinery makes it cheap to **re-solve an LP after
adding one new constraint or one new variable**, warm-started from the previous solution.

**The objectives we care about.** Among nonlinear convex programs, the minimax / Chebyshev
fitting problem $\min_x \max_i\lvert a_i^\top x - b_i\rvert$ has a kinked objective; its
feasible-set variants are stated through curved convex constraints $G(x)\le 0$. These are the
instances the present inquiry targets.

## Baselines

- **Simplex / revised simplex for linear programming** (Dantzig 1947; Dantzig, Orden, Wolfe
  1955). Minimizes a linear form $c^\top x$ over a polyhedron $\{x : Ax \ge b\}$ by walking
  vertices. Exact, fast, and reusable: re-optimizing after adding a constraint costs little.

- **Dual method for LP** (Lemke 1954) and **revised simplex** plumbing (Dantzig–Orden–Wolfe
  1955). These decide *how cheaply* an iterative scheme that repeatedly adds constraints can
  run: the dual LP gains one variable per added primal constraint, so the previous optimum is a
  warm start. A scheme that adds one supporting halfspace per step inherits this efficiency.

- **Gradient projection for nonlinear programming** (Rosen 1960). Core idea: from a feasible
  point, step along the negative gradient projected onto the active constraints' tangent
  subspace, then restore feasibility. It uses the objective's gradient and projects onto the
  feasible set at each step.

- **Gomory's integer-programming cutting planes** (Gomory, *An algorithm for integer solutions
  to linear programs*, Princeton IBM Math. Research Project TR No. 1, 1958; Bull. AMS 64 (1958)
  275–278). Core idea: solve the LP relaxation; if the optimal vertex is fractional, append a
  linear inequality (a "cut") that is satisfied by every integer-feasible point but violated by
  the current fractional optimum, removing it; re-solve; repeat until the optimum is integral.
  The cuts are tailored to *integrality* — the unwanted point is a fractional vertex, and the
  cut is derived from the simplex tableau.

- **Remez / Chebyshev exchange and the support-plane idea** (Remez; Cheney–Goldstein's "Newton's
  method for convex programming," *Numer. Math.* 1 (1959) 253–268). Core idea for the
  semi-infinite Chebyshev problem: a convex function is the upper envelope of its support planes,
  so an infinite system of linear inequalities (one per support point) can be approached by a
  finite, growing subset, and the convex hypersurface is replaced by its support planes during
  computation. It is stated for Chebyshev approximation / semi-infinite linear systems.

## Evaluation settings

The natural yardstick is small convex programs with an analytic solution so that the iterate
sequence and the model values can be checked without ambiguity. A canonical instance is
minimizing a linear form over an ellipse, e.g. $\min\, x_1 - x_2$ subject to
$3x_1^2 - 2x_1x_2 + x_2^2 - 1 \le 0$, whose feasible region is an ellipse with optimum at
$(0,1)$ and minimum value $-1$. The run records the lower-model value, the current constraint
violation $G(t_k)$, and the number of LP re-solves (cuts added). The broader class of interest is
Chebyshev (minimax) curve fitting, $\min_x \max_i\lvert a_i^\top x - b_i\rvert$, the nonsmooth
problem that motivates the whole construction. The relevant per-iteration primitive is a
linear-programming solver (revised simplex / dual simplex), and the quantities one measures are
the certified objective gap and the LP work needed to reach a target accuracy.

## Code framework

A reusable scaffold starts from an LP solver that minimizes a linear form over a polyhedron and
can be re-solved cheaply after adding a row, plus a first-order oracle that, at a query point,
returns a convex value and a subgradient (a supporting hyperplane). The one empty slot is the
procedure that uses these two primitives to solve the curved convex program.

```python
import numpy as np

def lp_minimize_linear(c, A_ub, b_ub):
    """Minimize c^T x s.t. A_ub x <= b_ub over a polyhedron.
    Pre-existing: revised/dual simplex. Returns (x_opt, opt_value)."""
    ...  # standard LP solver

def convex_oracle(x):
    """First-order oracle for the convex problem.
    Returns the function/constraint value at x and a supporting hyperplane
    (a subgradient). Pre-existing primitive."""
    ...  # returns (value, subgradient)

def solve_convex_program(c, initial_polyhedron, lp_minimize_linear, convex_oracle):
    # TODO: using only the LP solver and the first-order oracle above,
    #       solve  min c^T x  s.t.  x in R = {x : G(x) <= 0}.
    ...
```
