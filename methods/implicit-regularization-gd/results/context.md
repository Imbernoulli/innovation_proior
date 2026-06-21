## Problem Setting

A classifier is trained on a dataset that can be separated perfectly by a homogeneous linear predictor. After folding labels into the examples, the training data are vectors `x_1, ..., x_N` and separability means there is some `w_*` with `w_*^T x_n > 0` for every `n`.

The loss is not squared error. It is a classification loss such as logistic loss or exponential loss:

```text
L(w) = sum_n ell(w^T x_n),
```

where `ell` is smooth, positive, strictly decreasing, and approaches zero only as its argument goes to infinity. Gradient descent is run with a fixed step size small enough for the standard smooth-descent inequality:

```text
w(t + 1) = w(t) - eta grad L(w(t)).
```

The training loss can be driven arbitrarily close to zero without an explicit penalty on `||w||`. Separability admits many possible separating directions, and classical capacity reasoning addresses which functions generalize, not which one a particular optimizer approaches.

## Baseline Geometry

For an underdetermined least-squares problem, there is a familiar optimizer bias. If gradient descent starts at the origin, each update lies in the span of the data, so the iterate stays in the row space. When interpolation is possible, the row-space interpolant is the minimum Euclidean norm interpolant. That is a finite-minimizer story.

Classification on separable data is different. The logistic and exponential losses do not attain their infimum at any finite vector. Scaling any strict separator by a larger positive constant keeps decreasing the loss, so the norm of the weight vector grows without bound. The sign of prediction depends only on direction, so the meaningful quantity is

```text
w(t) / ||w(t)||.
```

The relevant regime is at infinity, after every training margin has become large.

## Prior Signals

The hard-margin support vector machine gives one classical low-complexity separator:

```text
argmin_w ||w||^2
subject to w^T x_n >= 1 for all n.
```

Its optimality conditions represent the separator as a nonnegative combination of the support vectors, the examples whose constraints are tight. This is a reference geometry for large margin, defined as the solution of an optimization problem rather than as an optimizer trajectory.

Several nearby facts describe related settings. Explicit `L_p` constrained or regularized paths for logistic-like and exponential losses tend toward `L_p` maximum-margin separators as the constraint is relaxed. Boosting and coordinate-descent analyses give `L_1` margin behavior for AdaBoost-type procedures. Matrix-factorization studies show that gradient-based optimization can prefer lower-complexity global minima without an explicit penalty. These describe optimizer-induced selection in adjacent problems.

## The Question

The object of study is the direction of the diverging trajectory of plain full-gradient descent on unregularized separable classification: whether `w(t) / ||w(t)||` has a limit, what that limiting separator is, and how it relates to the loss and to the optimization geometry.

The finite-norm least-squares analogy does not transfer directly, since the minimizer is at infinity. One route is to inspect the asymptotic gradient, because in the large-margin regime different examples contribute to the gradient at very different scales. As all margins go to infinity, the gradient may continue to average all training points or may concentrate on a special subset.

Rate is part of the question. Training loss may be nearly zero while the classifier direction is still moving, so the analysis distinguishes loss convergence from direction convergence and from margin convergence.

## Observables

A faithful experiment would not stop at zero training error or at a tiny training loss. It would track the norm, the normalized direction, the normalized margin, and loss on held-out points separately, because these quantities can move on different scales after the classifier already separates the training data.

The optimization rule must also stay explicit. Full-gradient, coordinate, stochastic, momentum, and adaptive updates all see the same separable loss landscape, but they do not share the same update geometry. Any claimed limiting separator has to come from the trajectory being analyzed, not from the loss function alone.
