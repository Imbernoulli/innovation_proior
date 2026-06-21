## Research Question

The problem is to minimize a differentiable convex objective over a compact convex feasible set:

```text
minimize f(x) subject to x in D.
```

The feasible set may be a polytope, a simplex, an l1 or nuclear-norm ball, a convex hull of structured atoms, or another compact convex set that is easy to describe but not necessarily easy to project onto. The objective is accessed through a first-order oracle: at a feasible point we can evaluate the value and gradient, but we do not assume Hessians, exact global minimization of the objective, or a cheap dense representation of all feasible points.

The question is how to drive a first-order iteration toward the constrained optimum while keeping every iterate inside `D`. An unconstrained step such as `x - eta grad f(x)` is in general a point outside `D`.

Euclidean projection back to `D` is itself an optimization problem:

```text
Pi_D(y) = argmin_{z in D} ||z - y||^2.
```

On simple boxes or Euclidean balls this is a closed-form or cheap computation. On spectral, assignment, and general atomic domains it is itself a substantial optimization.

## Mathematical Tools

Convexity gives a global lower model at every point:

```text
f(y) >= f(x) + <y - x, grad f(x)> for all x,y.
```

This tangent inequality is the main certificate mechanism available from first-order information. If `x*` is optimal, substituting `y = x*` compares the current value to the unknown optimum using only a linear expression involving `grad f(x)`.

Smoothness gives the companion upper model. If the gradient is `L`-Lipschitz in a chosen norm, then

```text
f(y) <= f(x) + <y - x, grad f(x)> + (L/2)||y - x||^2.
```

This turns a local linear prediction into a guaranteed descent statement once the step length is controlled.

Compact convex sets also make linear optimization meaningful. A linear functional over a compact convex set attains its minimum, and over a polytope it may be chosen at a vertex. In many structured domains, minimizing a linear functional has a direct form: on the simplex it is a coordinate choice; on a nuclear-norm ball it is a top singular-vector computation; on the Birkhoff polytope it is an assignment problem; on many atomic hulls it is exactly the available combinatorial oracle.

Sparse representation is another geometric fact in the background. A point in a convex hull can be represented as a convex combination of atoms, and a point assembled from only a few atoms is sparse or low-rank relative to a generic feasible point.

## Baselines

Projected gradient descent first moves in the negative-gradient direction and then repairs feasibility:

```text
y_{t+1} = x_t - eta grad f(x_t)
x_{t+1} = Pi_D(y_{t+1})
```

For smooth convex objectives it has the familiar `O(1/t)` type rate under standard assumptions. Each iteration evaluates `Pi_D`, which on a given domain may be a quadratic program, a full singular-value decomposition, or another inner solve.

Mirror and proximal methods replace Euclidean projection by a Bregman or prox subproblem matched to the domain geometry. On domains with a cheap prox, such as the simplex with entropy, the per-iteration subproblem has a closed form. In general each iteration solves a projection-like subproblem.

Second-order and interior-point methods can converge in fewer iterations. They use linear-system solves, Hessian or barrier structure, and explicit constraint representations.

The older quadratic-programming setting is a historical reference point. Linear programming tools optimize a linear objective over linear constraints, while a convex quadratic objective over those same constraints is not itself a linear program.

## Evaluation Settings

Important test settings include convex quadratic programs over polyhedra, constrained least squares over an l1 ball, optimization over the probability simplex, nuclear-norm constrained matrix problems, and combinatorial polytopes such as assignment or flow polytopes.

The relevant measurements are:

- primal error `f(x_k) - f(x*)`;
- a computable certificate or duality gap;
- cost of one feasibility-preserving step;
- which domain primitive the iteration uses;
- sparsity, atom count, or rank of the iterate;
- robustness of the rate to affine reparameterizations and scaling of the feasible set.

## Code Framework

The implementation scaffold has a smooth convex objective and a feasible set. It exposes the available feasibility primitives without deciding which one the outer loop builds around.

```python
import numpy as np

class SmoothConvexObjective:
    def value(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

class FeasibleSet:
    def project(self, y):
        """Return argmin_{z in D} ||z - y||^2, if this is cheap."""
        raise NotImplementedError

    def linear_optimize(self, c):
        """Return an optimizer of a linear objective over D, if this is cheap."""
        raise NotImplementedError

def minimize(objective: SmoothConvexObjective, domain: FeasibleSet, x0, max_iter):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = objective.grad(x)
        # The design problem: use g and the available domain primitive
        # to produce a new feasible point.
        pass
    return x
```
