# Context: Nonsmooth Convex Minimization

## Research Question

We need to minimize a convex function `f : R^n -> R` when the objective is not necessarily differentiable. This is the natural form of many large-scale optimization problems: maxima of affine or smooth pieces, absolute-value and `l1` penalties, Lagrangian dual functions from decomposition, maximum-eigenvalue objectives, exact penalty functions, and feasibility objectives based on distances to convex sets. The minimizer can lie exactly at a kink, where an ordinary gradient does not exist.

The question is how to design an iterative first-order method for this setting, using only information available at the current queried point, and what convergence guarantees are achievable for the class of Lipschitz convex functions.

## Available Geometry

Convexity still gives a local certificate at a nonsmooth point. If `g` satisfies

```text
f(z) >= f(x) + g^T(z-x)   for every z,
```

then the affine function on the right is a global lower support for `f`. Geometrically, `(g,-1)` is normal to a hyperplane supporting the epigraph at `(x,f(x))`. At interior points of the domain such supports exist by the supporting-hyperplane theorem. Where the function is differentiable, this set has one element, the ordinary gradient. At a kink, it is usually a whole closed convex set.

This geometry also gives calculus. For a pointwise maximum, one may choose a gradient of any active piece, and the full set is the convex hull of active-piece subgradients. For `|x|` at zero, all slopes between `-1` and `1` are valid. For `l1` norms and maximum eigenvalues, similarly cheap supporting slopes are available.

## Existing Alternatives

Cutting-plane and localization procedures use the same supporting inequalities more conservatively. Each queried support gives a halfspace or affine lower model, and the method accumulates these cuts to localize the solution. This can provide stopping certificates, but it stores an expanding model and solves an auxiliary localization or master problem.

Second-order and interior-point methods can be fast when a smooth barrier or Hessian structure is available. Smoothing can sometimes help adapt smooth methods to kinked objectives, though the problem being solved and its accuracy then depend on the smoothing parameter.

## Evaluation Pressure

The worst-case target class is broad: a first-order oracle returns the value and one supporting slope at the queried point, and the next point can depend only on the span of information already observed. The relevant guarantee is in terms of the best value seen so far, the initial distance `R` to a minimizer, and a Lipschitz or subgradient-norm bound `G`.
