# The Kernel Trick and the Representer Theorem

The method has two linked claims.

First, a real positive definite kernel is a coordinate-free inner product. For any finite sample, the Gram matrix

```text
K_ij = k(x_i, x_j)
```

must be symmetric and satisfy

```text
sum_i sum_j c_i c_j k(x_i, x_j) >= 0.
```

When this holds for every finite sample, there is a Hilbert feature map `Phi` with

```text
k(x,z) = <Phi(x), Phi(z)>.
```

For a Mercer-style expansion `k(x,z) = sum_r lambda_r phi_r(x) phi_r(z)` with `lambda_r >= 0`, the feature coordinates are `sqrt(lambda_r) phi_r(x)`, so the inner product recovers the coefficients `lambda_r`, not `lambda_r^2`.

The canonical construction uses the RKHS feature map `Phi(x)=k(.,x)`, where

```text
<f, k(.,x)>_H = f(x)
<k(.,x), k(.,z)>_H = k(x,z)
```

So any lifted algorithm whose data dependence is through `<Phi(x_i), Phi(x_j)>` can replace those inner products by `k(x_i,x_j)`.

Second, many regularized solutions are finite because the optimizer lies in the training span. For data `(x_i,y_i)_{i=1}^m`, RKHS `H_k`, arbitrary empirical cost `c`, and strictly increasing norm penalty `g` on `[0, infinity)`, any minimizer of

```text
c((x_1,y_1,f(x_1)), ..., (x_m,y_m,f(x_m))) + g(||f||_H)
```

has the form

```text
f(.) = sum_{i=1}^m alpha_i k(.,x_i).
```

Proof sketch: decompose `f = f_parallel + f_perp`, with `f_parallel` in `span{k(.,x_i)}` and `f_perp` orthogonal to that span. The reproducing property makes every training value independent of `f_perp`, while `||f||^2 = ||f_parallel||^2 + ||f_perp||^2`. A strictly increasing norm penalty removes `f_perp`.

For squared-loss kernel ridge regression with `lambda > 0`, this gives

```text
f(x) = sum_i alpha_i k(x_i,x)
(K + lambda I) alpha = y
```

under the standard objective `sum_i (y_i - f(x_i))^2 + lambda alpha^T K alpha`. This solve remains the canonical choice when `K` is singular because `K + lambda I` is positive definite. In the unweighted scalar-regularization case, this matches the reference implementation pattern used by scikit-learn's `KernelRidge`: compute the training kernel matrix, store `dual_coef_ = solve(K + alpha I, y)` where `alpha` is the regularization strength, retain `X_fit_`, and predict with `k(X, X_fit_) @ dual_coef_`.

For spline and semiparametric variants with an unpenalized null space, the finite form is

```text
f(x) = sum_i alpha_i k(x_i,x) + sum_p beta_p psi_p(x).
```

The unpenalized basis must be handled separately; specific spline systems add rank conditions or side constraints such as orthogonality between the penalized coefficients and the null-space design.

The Aizerman-Braverman-Rozonoer potential-function method supplies the early coordinate-replacement pattern: update and predict with scalar products in a rectifying space, stored as potentials centered on correction examples. Kimeldorf-Wahba supplies the finite-span variational argument. Scholkopf-Herbrich-Smola gives the generalized RKHS statement for learning objectives.
