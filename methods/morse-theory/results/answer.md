# Morse Theory

Let `M` be a smooth `n`-manifold and let `f:M->R` be smooth. Write

`M^a = {x in M : f(x) <= a}`.

A point `p` is critical if `df_p=0`. It is nondegenerate if the Hessian at `p` is nonsingular. Its index `lambda` is the number of negative eigenvalues of the Hessian, counted with multiplicity.

## Theorem

Assume `f^{-1}([a,b])` is compact.

1. If `f` has no critical points in `f^{-1}([a,b])`, then `M^a` and `M^b` are diffeomorphic, and `M^a` is a deformation retract of `M^b`.
2. If `f^{-1}([c-epsilon,c+epsilon])` contains exactly one critical point `p`, with `f(p)=c`, and `p` is nondegenerate of index `lambda`, then `M^{c+epsilon}` is obtained from `M^{c-epsilon}` by attaching a `lambda`-handle

   `D^lambda x D^{n-lambda}`

   along `S^{lambda-1} x D^{n-lambda}`, up to the standard smoothing of corners. In particular, `M^{c+epsilon}` has the homotopy type of `M^{c-epsilon}` with one `lambda`-cell attached.

Consequently, if `M` is compact and `f` has distinct critical values, then crossing the critical values in increasing order gives a handle decomposition of `M`, with one `lambda`-handle for every critical point of index `lambda`. Up to homotopy, this gives a CW complex with one `lambda`-cell for every such critical point.

## Proof

First suppose the band `f^{-1}([a,b])` has no critical points. Choose a Riemannian metric. On the band, `grad f` is nonzero, so

`X = grad f / ||grad f||^2`

satisfies `df(X)=1`. Compactness of the band gives existence of the flow for the needed finite times. If `phi_t` is the flow of `X`, then

`f(phi_t(x)) = f(x)+t`

whenever the trajectory remains in the band. Thus flow for time `b-a` identifies the level `f^{-1}(a)` with `f^{-1}(b)` and identifies the band with a product over its lower level. With a cutoff that is zero below `a` and agrees with `X` inside the band, this produces a diffeomorphism from `M^a` to `M^b`. Reversing the flow on the band gives a deformation retraction of `M^b` onto `M^a`.

Now suppose the band around `c` has exactly one critical point `p`, and that `p` has index `lambda`. By the Morse lemma, there are local coordinates `(u,v) in R^lambda x R^{n-lambda}` centered at `p` such that

`f(u,v)=c-||u||^2+||v||^2`.

Choose the coordinate neighborhood small enough that it contains no other critical point. Outside this neighborhood, shrink `epsilon` if necessary so that the remaining part of `f^{-1}([c-epsilon,c+epsilon])` has no critical points. The regular-band argument then says the outside part contributes only a product; all new topology is contained in the quadratic model.

In the local model, crossing the level `c` changes the sublevel condition from

`-||u||^2+||v||^2 <= -epsilon`

to

`-||u||^2+||v||^2 <= epsilon`.

Choose radii `r,R>0` inside the coordinate neighborhood with `r^2<epsilon` and `R^2-r^2>epsilon`. The product

`H = { (u,v) : ||u|| <= R, ||v|| <= r }`

is a `lambda`-handle `D^lambda x D^{n-lambda}` contained in the upper local sublevel set, since `-||u||^2+||v||^2 <= r^2 < epsilon` on `H`. Its attaching face

`S^{lambda-1} x D^{n-lambda}`.

lies in the lower local sublevel set because on `||u||=R` and `||v||<=r`,

`-||u||^2+||v||^2 <= -R^2+r^2 < -epsilon`.

The remaining part of the local upper sublevel set is carried to the lower sublevel set or to this product handle by the gradient flow of the quadratic form, with the usual rounding of corners. Thus the passage through the critical level attaches a `lambda`-handle. Collapsing the `D^{n-lambda}` factor of that handle shows the homotopy version: crossing the critical value attaches one `lambda`-cell.

Combining the two parts proves the decomposition statement. Between critical values, normalized gradient flow identifies sublevel sets. At each nondegenerate critical value, the Morse lemma supplies the exact quadratic local model, and the number of negative Hessian directions supplies the handle dimension.
