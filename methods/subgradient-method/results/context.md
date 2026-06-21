# Context: Nonsmooth Convex Minimization

## Research Question

We need to minimize a convex function `f : R^n -> R` when the objective is not necessarily differentiable. This is the natural form of many large-scale optimization problems: maxima of affine or smooth pieces, absolute-value and `l1` penalties, Lagrangian dual functions from decomposition, maximum-eigenvalue objectives, exact penalty functions, and feasibility objectives based on distances to convex sets. The minimizer can lie exactly at a kink, where an ordinary gradient does not exist.

The desired algorithmic primitive should use only first-order information available at the current point, should not require solving a growing auxiliary linear or quadratic program, and should come with a guarantee for a whole Lipschitz convex class rather than for a special smooth model.

## Available Geometry

Convexity still gives a local certificate at a nonsmooth point. If `g` satisfies

```text
f(z) >= f(x) + g^T(z-x)   for every z,
```

then the affine function on the right is a global lower support for `f`. Geometrically, `(g,-1)` is normal to a hyperplane supporting the epigraph at `(x,f(x))`. At interior points of the domain such supports exist by the supporting-hyperplane theorem. Where the function is differentiable, this set has one element, the ordinary gradient. At a kink, it is usually a whole closed convex set.

This geometry also gives calculus. For a pointwise maximum, one may choose a gradient of any active piece, and the full set is the convex hull of active-piece subgradients. For `|x|` at zero, all slopes between `-1` and `1` are valid. For `l1` norms and maximum eigenvalues, similarly cheap supporting slopes are available.

## Why Smooth Descent Is Fragile

Smooth gradient descent relies on a specific fact: the negative gradient is a descent direction, so a sufficiently small step or a line search can reduce the objective. That fact is no longer automatic when the first-order object is a set. The directional derivative of a convex function in direction `v` is the supremum of `h^T v` over all supporting slopes `h` at the point. If one selected slope is used to define a direction, another slope in the same set can dominate the directional derivative.

So the observable function value may rise even after a tiny move based on legitimate first-order information. A proof that watches `f(x_k)` decrease is therefore aimed at the wrong quantity. A line search that assumes decrease along the chosen direction is equally suspect.

## Existing Alternatives

Cutting-plane and localization procedures use the same supporting inequalities more conservatively. Each queried support gives a halfspace or affine lower model, and the method accumulates these cuts to localize the solution. This can provide stopping certificates, but it stores an expanding model and solves an auxiliary localization or master problem.

Second-order and interior-point methods can be fast when a smooth barrier or Hessian structure is available, but they do not directly handle a kinked objective as a black-box Lipschitz convex function. Smoothing can sometimes help, but then the problem being solved and its accuracy depend on the smoothing parameter.

## Evaluation Pressure

The worst-case target class is broad: a first-order oracle returns the value and one supporting slope at the queried point, and the next point can depend only on the span of information already observed. Nonsmooth black-box lower bounds show that one should not expect the smooth `O(1/epsilon)` iteration scale. The relevant guarantee is in terms of the best value seen so far, the initial distance `R` to a minimizer, and a Lipschitz or subgradient-norm bound `G`.

The design pressure is therefore precise: find the simplest current-point update whose analysis does not require objective descent, and choose step sizes so the progress term from the support inequality eventually dominates the accumulated overshoot.
