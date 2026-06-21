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

That makes the optimization criterion explicit. Gradient descent is fastest when the condition number
`kappa(H)` is smallest, and `kappa(H)=1` exactly when the normalized basis is orthonormal under that
density-weighted inner product. The monomial basis is provably non-orthogonal under any valid weight,
so it has no way to make this Gram matrix diagonal. Chebyshev is orthogonal, but only under its fixed
weight `(1-x^2)^(-1/2)`. A fixed Chebyshev weight cannot be right for both homophilic graphs, whose
signal energy is low-frequency, and heterophilic graphs, whose signal energy shifts high-frequency.
The basis I want is an orthogonal family whose weight can move with the graph's signal density.

That requirement points to Jacobi polynomials. `P_k^{a,b}` is orthogonal under
`(1-x)^a(1+x)^b` on `[-1,1]`; Chebyshev is the special case `a=b=-1/2`, and Legendre is `a=b=0`.
The two exponents reshape the endpoint emphasis. On `A_hat = I - L`, low graph frequency maps to
`x=1` and high graph frequency maps to `x=-1`; decreasing `a` relative to `b` emphasizes the
`x=1` end, while decreasing `b` relative to `a` emphasizes the `x=-1` end. I do not need to compute
the exact density, which would require eigendecomposition, but I can replace the non-orthogonal
monomial powers with a tunable orthogonal family and keep the unconstrained signed coefficients that
made GPR-GNN work.

The recurrence is the fragile part, so I write it in the same symbols as the source. The base cases
are `P_0^{a,b}(x)=1` and `P_1^{a,b}(x)=(a-b)/2 + (a+b+2)x/2`. For `k >= 2`,
`P_k^{a,b}(x) = (theta_k x + theta_prime_k) P_{k-1}^{a,b}(x) - theta_double_prime_k
P_{k-2}^{a,b}(x)`, with
`theta_k = (2k+a+b)(2k+a+b-1) / [2k(k+a+b)]`,
`theta_prime_k = (2k+a+b-1)(a^2-b^2) / [2k(k+a+b)(2k+a+b-2)]`, and
`theta_double_prime_k = (k+a-1)(k+b-1)(2k+a+b) / [k(k+a+b)(2k+a+b-2)]`. In code I store
`theta_prime_k` as `A_k`, `theta_k` as `B_k`, and `theta_double_prime_k` as `C_k`; the extra factor
of two in the shared denominator cancels in `B_k` and `C_k`, so the implementation is the canonical
recurrence.

Now I need to be precise about the same-named difference between the full method and this task's edit
surface. The full JacobiConv is a linear GNN: it drops the ReLU encoder, uses individual coefficient
vectors per output channel so multi-class predictions get separate filters, and adds polynomial
coefficient decomposition, `alpha_{kl}=beta_{kl} prod_{i<=k} gamma_i`, to control coefficient scales.
This scaffold does not expose that full machinery. It keeps the two-layer ReLU encoder, gives
`CustomProp` only one shared coefficient vector `temp` of length `K+1`, fixes `a=b=1`, initializes all
coefficients to `1/(K+1)`, and has no slot for PCD. So the artifact I land here is the
shared-coefficient, fixed-`(a,b)` Jacobi recurrence core, not the full per-channel linear PCD model.

The operator convention is also safe. The paper defines the graph Jacobi basis as
`P_k^{a,b}(A_hat)` with `A_hat = I - L`, whose spectrum lies in `[-1,1]`. The edit surface naturally
builds `L_tilde = L - I = -A_hat` by adding self-loops of weight `-1` to the normalized Laplacian.
For the fixed symmetric case `a=b=1`, the reflection identity gives
`P_k^{a,b}(-x)=(-1)^k P_k^{a,b}(x)`. The learned scalar `temp[k]` absorbs that alternating sign, so
the representable filter family is unchanged even though the recurrence is evaluated on `L_tilde`.

The bar remains the GPR-GNN numbers, not invented JacobiConv results. I expect this shared Jacobi core
to match or beat Cora 0.8890, Citeseer 0.8020, Texas 0.9065, and Cornell 0.8705, with the most
important signal being lower variance than the Texas std spread of 0.0262 to 0.0471. If it falls
below GPR-GNN, the honest interpretation is that the conditioning benefit does not survive this
restricted harness without per-channel filters, tuned `a,b`, or PCD. But as a final edit surface, the
right move is still clear: keep the unconstrained shared coefficients and replace monomial powers with
the canonical Jacobi three-term recurrence.

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
