## Problem Setting

Constrained optimization problems take the form

$$
\min_x f(x)\quad\text{s.t.}\quad h_i(x)=0,\ i=1,\ldots,m,\qquad g_j(x)\le 0,\ j=1,\ldots,p.
$$

A candidate optimum can lie on a boundary where the usual unconstrained condition $\nabla f(x^\star)=0$ does not hold. An optimality test must account for both the objective's local slope and the local geometry of the active constraints.

## Baseline Before the Method

A typical workflow is case-based: guess which boundary face contains the optimum, reduce the problem there, then separately check feasibility and compare candidates. Equality-constrained Lagrange multipliers give a cleaner rule, $\nabla f+\sum_i\nu_i\nabla h_i=0$, but inequalities require a way to handle which constraints are active at $x^\star$ without enumerating every active set by hand.

## KKT Certificate

Under suitable regularity assumptions called constraint qualifications, a local minimizer $x^\star$ has multipliers $\nu_i$ for equalities and $\lambda_j\ge 0$ for inequalities such that

$$
\nabla f(x^\star)+\sum_i\nu_i\nabla h_i(x^\star)+\sum_j\lambda_j\nabla g_j(x^\star)=0,
$$

with

$$
h_i(x^\star)=0,\qquad g_j(x^\star)\le 0,\qquad \lambda_j\ge 0,\qquad \lambda_j g_j(x^\star)=0.
$$

These four parts are the local certificate: stationarity, primal feasibility, dual feasibility, and complementary slackness. In convex optimization, with the usual convexity/regularity assumptions, the same certificate becomes sufficient for global optimality, not only necessary for local optimality.

## Geometric Mechanics View

The distinctive insight is that the optimum is a gradient-force balance. At an unconstrained optimum, the objective's gradient must vanish because every small direction is allowed. At a constrained boundary optimum, the objective gradient need not vanish; it only needs to be canceled by a nonnegative combination of the outward normals of active inequality constraints plus the normals of equality constraints.

Complementary slackness gives the mechanics its switch: an inactive inequality has slack $g_j(x^\star)<0$, so its multiplier must be zero and it exerts no local "constraint force." An active inequality may have $\lambda_j>0$, meaning the boundary is actually holding the optimizer in place. KKT therefore turns "the best point seems to be on this boundary" into "the objective's local push is exactly balanced by the active constraints' reaction forces."

## Why It Matters

KKT compresses constrained optimality into one reusable local language. It explains boundary optima geometrically, gives algorithms a target residual to drive to zero, and connects primal variables to economically meaningful shadow prices. The multipliers say how sensitive the optimum is to relaxing constraints, while the complementarity equations identify which constraints are binding.
