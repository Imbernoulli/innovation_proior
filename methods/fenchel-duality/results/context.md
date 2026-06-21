## Research question

The core question is how to understand a convex minimization problem without staring only at
candidate minimizers. In the canonical setting,

$$
\inf_x \; f(x)+g(Ax),
$$

with closed proper convex functions $f$ and $g$ and a linear map $A$, the direct problem asks for the
lowest value reached by the objective. A related question is how to certify a lower bound on that
value: among affine functions that lie below the objective everywhere, how high a guaranteed floor
can one produce, and how is the remaining error between such a floor and the true minimum to be
measured?

## Background

The convex conjugate of a function $f$ is

$$
f^*(y)=\sup_x \{\langle y,x\rangle-f(x)\}.
$$

By construction of the supremum, for every $x$ and $y$,

$$
f(x)+f^*(y)\ge \langle y,x\rangle,
$$

or equivalently

$$
f(x)\ge \langle y,x\rangle-f^*(y).
$$

So a fixed vector $y$ together with the value $f^*(y)$ describes an affine function of $x$ with slope
$y$ and intercept $-f^*(y)$ that never exceeds $f$. The conjugate records, for each slope, the best
intercept allowed. For composite objectives such as $f(x)+g(Ax)$ the two pieces are coupled through
the linear map $A$, which acts on the argument of $g$.

## Baselines

- **Direct primal minimization.** Search over $x$ to lower $f(x)+g(Ax)$, working with the objective in
  its original form, including nonsmooth terms, indicators, norms, and the linear composition.

- **Local stationarity / subgradient equations.** Look for $x$ satisfying
  $0\in \partial f(x)+A^T\partial g(Ax)$, the first-order optimality condition under convexity.

- **Lagrange duality.** Move constraints into a Lagrangian and optimize over multipliers, obtaining
  lower bounds on the constrained primal value through the multipliers.

- **Geometric supporting hyperplanes.** A convex epigraph admits supporting hyperplanes at its
  boundary, the geometric origin of affine minorants of a convex function.

## Evaluation settings

The lower-bound view is examined on examples where conjugates are known in closed form. For a norm,
the conjugate is the indicator of the dual unit ball. For an indicator function $\iota_C$, the
conjugate is the support function $\sigma_C$. For regularized empirical risk, the objective is a loss
plus a penalty, each with its own conjugate. Linear programming is the finite-dimensional case where
the relevant certificates are price vectors.

The success criterion is conceptual and mathematical: a weak-duality statement (any certified floor
is at most the true minimum), a strong-duality statement under a standard regularity condition, and a
decomposition of the primal-dual gap into nonnegative slacks of the conjugate inequality.

## Proof artifact

The final artifact should treat the primal problem

$$
\inf_x f(x)+g(Ax)
$$

and an associated maximization over certificates built from the conjugates $f^*$ and $g^*$. It should
relate the two through the conjugate inequality, state a regularity condition of Fenchel-Rockafellar
type (such as continuity of one term at a point where both terms are finite, or the corresponding
relative-interior qualification in finite dimensions), and express the optimality conditions in terms
of the subdifferentials $\partial f$ and $\partial g$.
