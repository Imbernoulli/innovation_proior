# Context

## Problem Family

The recurring optimization task is to recover a signal or image from indirect noisy measurements. A typical model is `b = A x + noise`, where `A` is a known linear operator and `x` is the unknown object. In deblurring, `A` is a blur operator composed with a transform such as a wavelet synthesis matrix, and the variable `x` stores coefficients rather than raw pixels.

Pure least squares is unstable when the inverse problem is ill-conditioned. A regularizer is needed, and the useful one for sparse transformed signals is an absolute-value penalty. The resulting objective has two visibly different pieces: a differentiable data-fit term that couples all coordinates through `A`, and a kinked sparsity term that acts coordinate by coordinate.

## Scale Constraint

The target instances are large enough that a method cannot rely on dense factorizations, generic cone-programming reductions, or repeated exact linear solves. Image-scale problems may have hundreds of thousands or millions of variables, and the operator is often applied implicitly through matrix-vector products.

This makes first-order computation attractive: each iteration should require only a small number of products with `A` and `A^T`, plus cheap vector operations. The drawback is that many iterations may be needed, so a usable method must come with a nonasymptotic function-value rate rather than only a convergence guarantee.

## Existing Smooth Tools

For a smooth convex function with Lipschitz gradient, a quadratic upper model is available at every point. This model is the basis for ordinary gradient descent: replace the smooth function locally by its tangent plus a curvature term, then minimize the model.

Smooth first-order methods can be much faster than plain descent when they use a carefully chosen auxiliary point. The best known smooth convex rates set a benchmark: a composite method should not lose the benefit of the differentiable data-fit term merely because another term in the objective has corners.

## Existing Nonsmooth Tools

A direct subgradient method can handle the whole objective as one nonsmooth convex function, but it treats the smooth data-fit term and the sparsity penalty as an undifferentiated lump. That discards curvature information from the data-fit term and leads to slow worst-case progress.

Earlier inverse-problem algorithms already show that sparsity penalties have a special computational structure. With suitable surrogate functionals, the coupled quadratic part can be separated from simple one-dimensional regularized subproblems. Operator-splitting work also gives a fixed-point language for combining an explicit step on one part of an objective with an implicit step on another.

## Required Result

The desired method must solve a convex composite objective while preserving the distinction between the two pieces. It should exploit the differentiable part without pretending the kinked penalty has a gradient, and it should turn the absolute-value penalty into cheap coordinate work rather than a dense linear solve.

The result should specify the update, the specialization to sparse least squares, the step-size or line-search condition, and a nonasymptotic objective-error guarantee for both the basic first-order scheme and its faster auxiliary-point variant.
