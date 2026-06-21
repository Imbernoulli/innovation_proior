## Research question

The core question is how to understand a convex minimization problem without staring only at
candidate minimizers. In the canonical Fenchel setting,

$$
\inf_x \; f(x)+g(Ax),
$$

with closed proper convex functions $f$ and $g$ and a linear map $A$, the direct problem asks for the
lowest value reached by the objective. Fenchel duality asks a different question: among all affine
functions that are guaranteed to lie below the objective everywhere, what is the highest lower bound
one can certify? The method should explain why a minimization problem can be converted into a
maximization problem over global lower bounds, and why the remaining error is measured by the
Fenchel-Young inequality.

## Background

The convex conjugate of a function $f$ is

$$
f^*(y)=\sup_x \{\langle y,x\rangle-f(x)\}.
$$

This is not just a transform formula. Rearranging the definition gives the Fenchel-Young inequality

$$
f(x)+f^*(y)\ge \langle y,x\rangle,
$$

or equivalently

$$
f(x)\ge \langle y,x\rangle-f^*(y).
$$

Thus each dual vector $y$ defines an affine global lower bound on $f$: slope $y$, intercept
$-f^*(y)$. The conjugate records the best intercept allowed for each slope. Fenchel duality builds
lower bounds for a composite objective by combining such conjugate-generated supports so that the
linear terms cancel. For $f(x)+g(Ax)$, introduce a vector $u$ in the range space and write

$$
f(x)\ge \langle -A^T u,x\rangle-f^*(-A^T u),
$$

$$
g(Ax)\ge \langle u,Ax\rangle-g^*(u).
$$

The two affine slopes cancel in $x$, leaving the constant lower bound

$$
f(x)+g(Ax)\ge -f^*(-A^T u)-g^*(u).
$$

Maximizing this constant over $u$ is the Fenchel dual.

## Baselines

- **Direct primal minimization.** Search over $x$ and try to lower $f(x)+g(Ax)$. This stays close to
  the original objective but can hide structure: nonsmooth terms, indicators, norms, and linear
  composition may make the geometry difficult to see.

- **Local stationarity / subgradient equations.** Look for $0\in \partial f(x)+A^T\partial g(Ax)$.
  This characterizes optimality under convexity, but it describes a point after the fact. It does
  not by itself explain how a certificate lower-bounds every primal point.

- **Lagrange duality.** Move constraints into a Lagrangian and optimize over multipliers. This gives
  lower bounds through constraints, but Fenchel duality reveals the deeper mechanism: a multiplier
  chooses supporting affine minorants of the functions themselves via conjugates.

- **Geometric supporting hyperplanes.** A convex epigraph has supporting hyperplanes. This is the
  geometric source of the idea, but Fenchel conjugacy makes the support calculation algebraic and
  composable across sums and linear maps.

## Evaluation settings

A useful explanation of Fenchel duality should work on examples where the lower-bound view is
visible. For a norm, the conjugate is the indicator of the dual unit ball, so the dual problem
becomes a feasibility-constrained search for the best certificate. For an indicator function
$\iota_C$, the conjugate is the support function $\sigma_C$, so constraints become support costs in
the dual. For regularized empirical risk, a loss plus a penalty becomes a dual problem balancing
loss conjugates and regularizer conjugates. Linear programming is the finite-dimensional limiting
case where all lower-bound certificates are familiar price vectors.

The success criterion is conceptual and mathematical: weak duality should follow immediately from
Fenchel-Young, strong duality should be stated under a standard regularity condition, and the
primal-dual gap should decompose into nonnegative Fenchel-Young slacks. When the gap is zero, the
chosen dual vector has found a lower supporting certificate that touches the primal optimum.

## Proof artifact

The final artifact should state the Fenchel dual pair

$$
\inf_x f(x)+g(Ax)
\quad\text{and}\quad
\sup_u -f^*(-A^T u)-g^*(u),
$$

derive weak duality by adding two Fenchel-Young inequalities, and interpret every feasible dual
$u$ as a global affine lower-bound certificate. It should then state a standard Fenchel-Rockafellar
strong-duality condition, such as continuity of one term at a point where both terms are finite, or
the corresponding relative-interior qualification in finite dimensions.

The proof should close with the optimality condition

$$
-A^T u \in \partial f(x), \qquad u\in \partial g(Ax),
$$

which says exactly that both conjugate inequalities are tight. This is the distinctive insight:
solving the primal is equivalent, at zero gap, to finding the strongest global lower bound whose
supporting hyperplanes touch the two pieces of the objective compatibly.
