## Research question

De Rham cohomology records topology by taking closed differential forms modulo exact forms:

`H^k_dR(M) = ker(d: Omega^k -> Omega^{k+1}) / im(d: Omega^{k-1} -> Omega^k)`.

This is powerful but deliberately quotient-shaped. A single topological class has many smooth representatives, since adding `d eta` does not change the class. The question is whether a Riemannian metric can turn that quotient into a canonical analytic object: one smooth form per class, chosen intrinsically rather than by coordinates, triangulations, or arbitrary normalization.

The setting has to be precise. The clean theorem should live on a smooth compact Riemannian manifold without boundary. Compactness is what makes the elliptic operator behave Fredholm-like, and absence of boundary is what lets integration by parts identify the formal adjoint without extra boundary conditions.

## Background

The de Rham complex gives the starting language:

`0 -> Omega^0(M) --d--> Omega^1(M) --d--> ... --d--> Omega^n(M) -> 0`,

with `d^2=0`. Closed forms are the cycles, exact forms are the boundaries, and the quotient is topological through de Rham's theorem. The weakness of this description is not correctness; it is non-canonicity. A class is an affine space of representatives.

A Riemannian metric adds a pointwise inner product on `k`-forms, a volume form, and hence an `L^2` inner product

`<alpha,beta> = int_M <alpha,beta>_x dV`.

On an oriented manifold this can also be written using the Hodge star as `int_M alpha wedge *beta`. On a closed manifold, the exterior derivative has an `L^2` formal adjoint

`d^*: Omega^k(M) -> Omega^{k-1}(M)`,

defined by `<d eta, alpha> = <eta, d^* alpha>`. This gives the Laplace-de Rham operator on forms:

`Delta = d d^* + d^* d`.

The operator is self-adjoint and elliptic. On closed manifolds, the identity

`<Delta alpha, alpha> = ||d alpha||^2 + ||d^* alpha||^2`

shows that `Delta alpha = 0` is equivalent to being both closed and co-closed. Elliptic regularity is the analytic bridge: weak or Sobolev solutions of elliptic equations are smooth, so the final representatives are genuine smooth differential forms.

## Baselines

- **Arbitrary closed representatives.** A de Rham class can be represented by any closed form in that class. The gap is that representatives differ by exact forms, so the quotient remembers topology but does not select a preferred form.

- **Periods over cycles.** Integrating closed forms over cycles gives topological pairings and detects cohomology classes. The gap is that periods identify the class as a functional on cycles, not as a canonical differential form on the manifold.

- **Degree-zero harmonic functions.** The ordinary Laplacian on functions supplies an analytic condition, but on a compact connected closed manifold harmonic functions are constant. That captures `H^0`, not the higher-degree topology carried by forms.

- **Vector-calculus Helmholtz decompositions.** In Euclidean vector calculus, fields can be split into gradient, curl-like, and divergence-free pieces under suitable domain or decay assumptions. The gap is that this does not by itself give a coordinate-free theorem for all differential forms on an arbitrary compact manifold.

- **Manifolds with boundary.** There are Hodge decompositions with boundary conditions, but the boundary changes the clean closed-manifold picture: harmonic forms need not coincide simply with closed and co-closed forms unless boundary conditions are specified.

## Evaluation settings

The natural setting is a smooth compact oriented Riemannian `n`-manifold without boundary. The main objects are `Omega^k(M)`, the exterior derivative `d`, its formal adjoint `d^*`, the Laplacian `Delta = d d^* + d^* d`, and the harmonic space

`Harm^k(M) = ker(Delta: Omega^k(M) -> Omega^k(M))`.

Stress cases include exact forms, which should have zero harmonic representative; degree zero, where harmonic functions on a compact connected manifold are constants; tori, where nonzero parallel forms represent nontrivial classes; and boundary examples, where extra boundary conditions are required.

Success means proving that each real de Rham class has exactly one harmonic representative, that all harmonic representatives form a finite-dimensional vector space isomorphic to `H^k_dR(M)`, and that the representative is the `L^2` energy-minimizing form in its class.

## Code framework

The proof scaffold is:

1. Put the `L^2` inner product on forms and define the adjoint `d^*`.
2. Define `Delta = d d^* + d^* d` and identify `ker Delta` with closed and co-closed forms.
3. Use compact elliptic self-adjoint theory to obtain the orthogonal decomposition of `Omega^k(M)`.
4. Project a closed form onto the harmonic summand and show the remaining part is exact.
5. Prove uniqueness by showing no nonzero exact form is harmonic.
6. Interpret the harmonic representative as the unique `L^2` norm minimizer in its de Rham class.
