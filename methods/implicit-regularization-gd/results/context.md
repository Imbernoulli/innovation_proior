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

The training loss can be driven arbitrarily close to zero without an explicit penalty on `||w||`. The difficulty is that separability gives many possible separating directions, and classical capacity reasoning does not say which one a particular optimizer will approach.

## Baseline Geometry

For an underdetermined least-squares problem, there is a familiar optimizer bias. If gradient descent starts at the origin, each update lies in the span of the data, so the iterate stays in the row space. When interpolation is possible, the row-space interpolant is the minimum Euclidean norm interpolant. That is a finite-minimizer story.

Classification on separable data is different. The logistic and exponential losses do not attain their infimum at any finite vector. Scaling any strict separator by a larger positive constant keeps decreasing the loss. Thus the norm of the weight vector cannot be the object that converges. The sign of prediction depends only on direction, so the meaningful quantity is

```text
w(t) / ||w(t)||.
```

Any analysis has to work at infinity, after every training margin has become large.

## Prior Signals

The hard-margin support vector machine gives one classical low-complexity separator:

```text
argmin_w ||w||^2
subject to w^T x_n >= 1 for all n.
```

Its optimality conditions represent the separator as a nonnegative combination of the support vectors, the examples whose constraints are tight. This is a reference geometry for large margin, not yet an optimizer trajectory.

Several nearby facts suggest where to look. Explicit `L_p` constrained or regularized paths for logistic-like and exponential losses tend toward `L_p` maximum-margin separators as the constraint is relaxed. Boosting and coordinate-descent analyses give `L_1` margin behavior for AdaBoost-type procedures. Matrix-factorization studies show that gradient-based optimization can prefer lower-complexity global minima without an explicit penalty. These facts point to optimizer-induced selection, but they do not identify the direction selected by plain full-gradient descent on unregularized separable classification.

## The Missing Mechanism

The open problem is to characterize the direction of the diverging trajectory itself. A complete answer must explain why the limit exists, which separator it is, what part of the loss is responsible, and why changing the optimization geometry could change the result.

The tempting finite-norm analogy is unavailable. A useful route must instead inspect the asymptotic gradient, because in the large-margin regime different examples contribute to the gradient at very different scales. The question becomes: as all margins go to infinity, does the gradient continue to average all training points, or does it focus on a special subset?

The answer also has to quantify rate. Training loss may be nearly zero while the classifier direction is still moving, so a useful analysis must separate loss convergence from direction convergence and from margin convergence.

## Observables

A faithful experiment would not stop at zero training error or at a tiny training loss. It would track the norm, the normalized direction, the normalized margin, and loss on held-out points separately, because these quantities can move on different scales after the classifier already separates the training data.

The optimization rule must also stay explicit. Full-gradient, coordinate, stochastic, momentum, and adaptive updates all see the same separable loss landscape, but they do not share the same update geometry. Any claimed limiting separator has to come from the trajectory being analyzed, not from the loss function alone.
