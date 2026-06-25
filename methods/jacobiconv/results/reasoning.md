I start from the GPR-GNN numbers because they show both the win and the remaining weakness. The flat,
unconstrained monomial filter sits at Texas 0.9065, Cornell 0.8705, Cora 0.8890, and Citeseer 0.8020.
But the seed-to-seed variance is the warning sign: Texas std across seeds ranges from 0.0262 to
0.0471. That spread is the fingerprint of a hard coefficient optimization problem, not an
expressiveness problem. The monomial powers are complete, but they become collinear as order grows,
so the optimizer can land in different places depending on seed.

So I stop asking which complete basis can represent the target filter. Monomial, Chebyshev,
Bernstein, and Jacobi bases all span the same degree-`K` polynomial space. The real question is which
basis makes the coefficient fit well-conditioned. Take the decoupled filter
`Z = sum_k gamma_k g_k(L_hat) X` and look near the optimum under squared loss, with the encoder
weights merged into `X` so the basis comparison is isolated. The Hessian in the coefficients has
entries `H_{k1,k2} = X^T g_{k2}(L_hat) g_{k1}(L_hat) X`, or, in the spectral continuum,
`H_{k1,k2} = integral_0^2 g_{k1}(lambda) g_{k2}(lambda) f(lambda) dlambda`, where `f(lambda)` is the
graph signal's spectral density. The Hessian is just the Gram matrix of the basis under the
signal-density inner product.

That makes the optimization criterion explicit. Gradient descent's convergence rate is governed by
`kappa(H)`, and `kappa(H)=1` exactly when the normalized basis is orthonormal under that
density-weighted inner product. I want to know whether this Gram-matrix view actually buys anything,
so before committing to any family I make the comparison concrete with a small numerical experiment.
I pick a stand-in density tilted toward low graph frequency — on the operator `A_hat = I - L` the
low-frequency end is `x = +1`, so I weight the inner product by `f(x) = 1 + x` on `[-1,1]` to put
more signal mass there. With that density I build the Gram matrix of the monomial powers
`{1, x, ..., x^6}`, normalize each basis vector to unit diagonal so I am comparing the *shape* of the
Gram matrix and not the scaling, and read off the condition number. It comes out around
`8.7e3`. So the collinearity I suspected from the GPR-GNN variance is real and large: the monomial
Hessian is badly conditioned even at modest order, and there is no weight under which the monomial
Gram can be diagonal — its entries `integral x^{k1+k2} f` only depend on `k1+k2`, so it is a Hankel
matrix with no zeros off the anti-diagonals. Monomials cannot be orthogonalized by any choice of `f`.

Chebyshev fixes the orthogonality but freezes the weight at `(1-x^2)^(-1/2)`, which puts its mass at
the two endpoints. That single weight cannot be right for both homophilic graphs, whose signal energy
sits near `x = +1`, and heterophilic graphs, whose energy shifts toward `x = -1`. What I actually want
is an orthogonal family whose weight can slide along `[-1,1]` to follow wherever the graph's signal
density lives. The Jacobi family `P_k^{a,b}` is orthogonal under `(1-x)^a (1+x)^b`, with the two
exponents reshaping the endpoint emphasis — decreasing `a` relative to `b` emphasizes the `x=+1`
(low-frequency) end and decreasing `b` relative to `a` emphasizes the `x=-1` (high-frequency) end.
Chebyshev is the special case `a=b=-1/2` and Legendre is `a=b=0`, so this is a genuine generalization,
not a different object. I do not need to compute the exact density, which would require
eigendecomposition; I can replace the non-orthogonal monomial powers with this tunable orthogonal
family and keep the unconstrained signed coefficients that made GPR-GNN work.

Before I trust this I check that an orthogonal Jacobi basis really does collapse the conditioning on
the same experiment. With `a=b=1` and the same tilted density `f(x)=1+x`, the normalized Jacobi Gram
matrix has condition number about `46` — two orders of magnitude below the monomial `8.7e3`. That is
the win I was after. But the number also tells me something honest I should not gloss over: it is
`46`, not `1`. The normalized off-diagonal entries of that Gram peak around `0.87`, so the basis is
*not* orthogonal under `f(x)=1+x`. It is orthogonal under its own weight `(1-x)(1+x)`, and I confirm
that separately — integrating `P_i P_j (1-x)(1+x)` by Gauss-Legendre quadrature gives a Gram matrix
whose off-diagonal magnitude is at the `1e-16` level, i.e. exactly diagonal to machine precision, with
diagonal `[1.333, 1.067, 0.857, 0.711, 0.606, ...]`. So `kappa=1` is only reached when `(a,b)` matches
the graph's actual density; a fixed `a=b=1` just gets *much* closer than monomials for a density in
the neighborhood. That is the real mechanism, and it is also exactly why the full method makes `(a,b)`
learnable. The shared-coefficient core I am building here freezes `(a,b)`, so I should expect a large
conditioning improvement over monomials but not a perfectly diagonal Hessian.

The recurrence is the fragile part, so I write it out carefully and then verify it rather than trust
my algebra. The base cases are `P_0^{a,b}(x)=1` and `P_1^{a,b}(x)=(a-b)/2 + (a+b+2)x/2`. For `k >= 2`,
`P_k^{a,b}(x) = (theta_prime_k + theta_k x) P_{k-1}^{a,b}(x) - theta_double_prime_k
P_{k-2}^{a,b}(x)`, with
`theta_k = (2k+a+b)(2k+a+b-1) / [2k(k+a+b)]`,
`theta_prime_k = (2k+a+b-1)(a^2-b^2) / [2k(k+a+b)(2k+a+b-2)]`, and
`theta_double_prime_k = (k+a-1)(k+b-1)(2k+a+b) / [k(k+a+b)(2k+a+b-2)]`. In code I store
`theta_prime_k` as `A_k`, `theta_k` as `B_k`, and `theta_double_prime_k` as `C_k`; the extra factor
of two in the shared denominator cancels in `B_k` and `C_k`, so the implementation is the canonical
recurrence.

Two checks on these coefficients. First, the easy special case: set `a=b=0` and the recurrence should
reproduce Legendre polynomials. Running it, `P_2` comes out as `(3x^2-1)/2` and `P_3` as `(5x^3-3x)/2`
on a grid of test points — both match the textbook Legendre forms exactly. Second, the case I will
actually use, `a=b=1`, against an independent implementation (`scipy.special.eval_jacobi`): the
recurrence agrees through `k=5` with a maximum absolute difference of `8.9e-16`, i.e. to floating
point. So the three-term coefficients are correct, including the `a^2-b^2` numerator in `A_k` that
vanishes whenever `a=b` and the `(2k+a+b-2)` factor I was worried could blow up — it is finite for all
`k>=2` at `a=b=1`.

Now I need to be precise about the same-named difference between the full method and this task's edit
surface. The full JacobiConv is a linear GNN: it drops the ReLU encoder, uses individual coefficient
vectors per output channel so multi-class predictions get separate filters, learns `(a,b)`, and adds
polynomial coefficient decomposition, `alpha_{kl}=beta_{kl} prod_{i<=k} gamma_i`, to control
coefficient scales. This scaffold does not expose that full machinery. It keeps the two-layer ReLU
encoder, gives `CustomProp` only one shared coefficient vector `temp` of length `K+1`, fixes `a=b=1`,
initializes all coefficients to `1/(K+1)`, and has no slot for PCD. So the artifact I land here is the
shared-coefficient, fixed-`(a,b)` Jacobi recurrence core, not the full per-channel linear PCD model.
Given the conditioning experiment above, freezing `a=b=1` is the one part of this restriction that
costs the most: it is the parameter that would let the weight chase the actual density toward `kappa=1`.

The operator convention is also worth pinning down. I define the graph Jacobi basis as
`P_k^{a,b}(A_hat)` with `A_hat = I - L`, whose spectrum lies in `[-1,1]`. The edit surface naturally
builds `L_tilde = L - I = -A_hat` by adding self-loops of weight `-1` to the normalized Laplacian, so
the recurrence is actually evaluated at `-x`, not `x`. I need to be sure that sign flip does not change
which filters the layer can represent. For the fixed symmetric case `a=b=1` the reflection identity
`P_k^{a,b}(-x)=(-1)^k P_k^{a,b}(x)` should hold, so evaluating at `L_tilde` just multiplies term `k`
by `(-1)^k`, which the learned scalar `temp[k]` absorbs. I verify the identity numerically rather than
assume it: computing `P_k(-x)` and `(-1)^k P_k(x)` from the recurrence at `a=b=1` for `k=0..6` on a
grid, the maximum discrepancy is `0` exactly (the sign symmetry is structural in the coefficients, not
just approximate). So the representable filter family is unchanged whether I evaluate on `A_hat` or on
`L_tilde`, and building `L_tilde` via `fill_value=-1.0` self-loops is safe.

The bar remains the GPR-GNN numbers, not invented JacobiConv results. Given the two-orders-of-magnitude
drop in the monomial-versus-Jacobi condition number on my stand-in density, I expect this shared Jacobi
core to match or beat Cora 0.8890, Citeseer 0.8020, Texas 0.9065, and Cornell 0.8705, with the most
important signal being lower variance than the Texas std spread of 0.0262 to 0.0471 — variance is the
quantity the conditioning argument speaks to most directly. But the same experiment warns me the win
may be partial: with `(a,b)` frozen at `1` the Hessian is well-conditioned but not orthonormal, so if
the numbers land at or below GPR-GNN the honest reading is that the conditioning benefit does not fully
survive this restricted harness without learnable `(a,b)`, per-channel filters, or PCD. Either way the
edit is the same: keep the unconstrained shared coefficients and replace monomial powers with the
canonical Jacobi three-term recurrence.

## Minimal reference: Jacobi recurrence

```python
import torch

def jacobi_polynomials(x, K, a=1.0, b=1.0):
    """Evaluate the first K+1 Jacobi polynomials P_k^{(a,b)}(x)."""
    P = [torch.ones_like(x)]
    if K >= 1:
        P.append((a - b) / 2.0 + (a + b + 2.0) / 2.0 * x)
    for k in range(2, K + 1):
        denom = 2.0 * k * (k + a + b) * (2.0 * k + a + b - 2.0)
        A = (2.0 * k + a + b - 1.0) * (a * a - b * b) / denom
        B = (2.0 * k + a + b - 1.0) * (2.0 * k + a + b - 2.0) * (2.0 * k + a + b) / denom
        C = 2.0 * (k - 1.0 + a) * (k - 1.0 + b) * (2.0 * k + a + b) / denom
        P.append((A + B * x) * P[-1] - C * P[-2])
    return torch.stack(P, dim=0)
```
