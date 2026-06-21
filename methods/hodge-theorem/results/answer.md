# Hodge theorem for closed Riemannian manifolds

## Theorem

Let `M` be a smooth compact oriented Riemannian manifold without boundary. For each degree `k`, let `d` be the exterior derivative on `Omega^k(M)`, let `d^*` be its `L^2` formal adjoint, and define the Hodge Laplacian

`Delta = d d^* + d^* d`.

Let

`Harm^k(M) = {alpha in Omega^k(M) : Delta alpha = 0}`.

Then:

1. `Omega^k(M)` has the orthogonal Hodge decomposition

   `Omega^k(M) = im d oplus im d^* oplus Harm^k(M)`.

2. A form is harmonic exactly when it is closed and co-closed:

   `Delta alpha = 0` iff `d alpha = 0` and `d^* alpha = 0`.

3. The natural map

   `Harm^k(M) -> H^k_dR(M), alpha |-> [alpha]`

   is an isomorphism.

Equivalently, every real de Rham cohomology class has a unique harmonic representative. This representative is the unique `L^2` norm-minimizing form in its class.

## Proof

Put the `L^2` inner product on forms by

`<alpha,beta> = int_M alpha wedge *beta`.

Because `M` has no boundary, integration by parts gives the formal adjoint `d^*` satisfying

`<d eta, alpha> = <eta, d^* alpha>`.

For every smooth `k`-form `alpha`,

`<Delta alpha, alpha> = <d d^* alpha, alpha> + <d^* d alpha, alpha>`

`= ||d^* alpha||^2 + ||d alpha||^2`.

Thus `Delta alpha=0` implies `d alpha=0` and `d^* alpha=0`; the converse is immediate from the definition of `Delta`.

The operator `Delta` is self-adjoint and elliptic. On a compact manifold, elliptic theory gives a finite-dimensional harmonic space, a Green operator `G`, and orthogonal projection `H` onto `Harm^k(M)` such that

`I = H + Delta G = H + G Delta`.

Elliptic regularity ensures that the resulting representatives are smooth. Expanding `Delta` gives, for every `omega in Omega^k(M)`,

`omega = H omega + d d^* G omega + d^* d G omega`.

So every form is a sum of a harmonic form, an exact form, and a co-exact form. The summands are orthogonal: harmonic forms are killed by both `d` and `d^*`, and

`<d a, d^* b> = <d^2 a, b> = 0`.

This proves the Hodge decomposition.

Now let `omega` be closed. Write

`omega = h + d a + d^* b`

by the decomposition. Since `d omega=0`, `d h=0`, and `d^2 a=0`, we get

`d d^* b = 0`.

Then

`||d^* b||^2 = <d^* b, d^* b> = <d d^* b, b> = 0`,

so `d^* b=0`. Hence

`omega = h + d a`,

and `omega` is cohomologous to the harmonic form `h`. Every de Rham class has a harmonic representative.

For uniqueness, suppose `h` is harmonic and exact, `h=d a`. Then

`||h||^2 = <h, d a> = <d^* h, a> = 0`,

so `h=0`. Therefore two harmonic forms in the same cohomology class are equal. The map `Harm^k(M) -> H^k_dR(M)` is an isomorphism.

Finally, if `h` is the harmonic representative of a class, every other representative is `h+d a`. Since `h` is orthogonal to exact forms,

`||h+d a||^2 = ||h||^2 + ||d a||^2`.

Thus `h` is the unique least-energy representative of its de Rham class. The theorem identifies topology with the finite-dimensional analytic space of harmonic forms.
