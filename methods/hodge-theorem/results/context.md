## Research question

De Rham cohomology records topology by taking closed differential forms modulo exact forms:

`H^k_dR(M) = ker(d: Omega^k -> Omega^{k+1}) / im(d: Omega^{k-1} -> Omega^k)`.

This is a quotient: a single topological class has many smooth representatives, since adding `d eta` does not change the class. The question is whether a Riemannian metric can pick out, from each de Rham class, a smooth form chosen intrinsically rather than by coordinates, triangulations, or arbitrary normalization.

The setting is a smooth compact Riemannian manifold without boundary. Compactness is what makes the elliptic operator behave Fredholm-like, and absence of boundary is what lets integration by parts identify the formal adjoint without extra boundary conditions.

## Background

The de Rham complex gives the starting language:

`0 -> Omega^0(M) --d--> Omega^1(M) --d--> ... --d--> Omega^n(M) -> 0`,

with `d^2=0`. Closed forms are the cycles, exact forms are the boundaries, and the quotient is topological through de Rham's theorem. A class is an affine space of representatives.

A Riemannian metric adds a pointwise inner product on `k`-forms, a volume form, and hence an `L^2` inner product

`<alpha,beta> = int_M <alpha,beta>_x dV`.

On an oriented manifold this can also be written using the Hodge star as `int_M alpha wedge *beta`. On a closed manifold, the exterior derivative has an `L^2` formal adjoint

`d^*: Omega^k(M) -> Omega^{k-1}(M)`,

defined by `<d eta, alpha> = <eta, d^* alpha>`. This gives the Laplace-de Rham operator on forms:

`Delta = d d^* + d^* d`.

The operator is self-adjoint and elliptic. On closed manifolds, the identity

`<Delta alpha, alpha> = ||d alpha||^2 + ||d^* alpha||^2`

shows that `Delta alpha = 0` is equivalent to being both closed and co-closed. Elliptic regularity is the analytic bridge: weak or Sobolev solutions of elliptic equations are smooth, so any such representatives are genuine smooth differential forms.

## Baselines

- **Arbitrary closed representatives.** A de Rham class can be represented by any closed form in that class. Representatives differ by exact forms.

- **Periods over cycles.** Integrating closed forms over cycles gives topological pairings and detects cohomology classes, identifying the class as a functional on cycles.

- **Degree-zero harmonic functions.** The ordinary Laplacian on functions supplies an analytic condition; on a compact connected closed manifold harmonic functions are constant, which captures `H^0`.

- **Vector-calculus Helmholtz decompositions.** In Euclidean vector calculus, fields can be split into gradient, curl-like, and divergence-free pieces under suitable domain or decay assumptions.

- **Manifolds with boundary.** There are Hodge decompositions with boundary conditions; the boundary changes the closed-manifold picture, and harmonic forms are characterized relative to the specified boundary conditions.

## Evaluation settings

The natural setting is a smooth compact oriented Riemannian `n`-manifold without boundary. The main objects are `Omega^k(M)`, the exterior derivative `d`, its formal adjoint `d^*`, and the Laplacian `Delta = d d^* + d^* d`.

Stress cases include exact forms; degree zero, where harmonic functions on a compact connected manifold are constants; tori, where nonzero parallel forms represent nontrivial classes; and boundary examples, where extra boundary conditions are required.

## Code framework

The proof scaffold is:

1. Put the `L^2` inner product on forms and define the adjoint `d^*`.
2. Define `Delta = d d^* + d^* d` and relate `ker Delta` to closed and co-closed forms.
3. Use compact elliptic self-adjoint theory to analyze the structure of `Omega^k(M)` under `Delta`.
4. Relate a closed form to the kernel of `Delta` and to the image of `d`.
5. Study which exact forms lie in `ker Delta`.
6. Interpret elements of `ker Delta` in terms of the `L^2` norm within a de Rham class.
