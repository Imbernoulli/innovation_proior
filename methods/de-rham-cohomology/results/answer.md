# De Rham cohomology

## Theorem

Let `M` be a smooth manifold. Let `Omega^k(M)` be the vector space of smooth `k`-forms and let

`d: Omega^k(M) -> Omega^{k+1}(M)`

be the exterior derivative. Define

`Z^k(M) = ker d`

and

`B^k(M) = im(d: Omega^{k-1}(M) -> Omega^k(M))`.

Then `B^k(M) subset Z^k(M)`, and the quotient

`H_dR^k(M) = Z^k(M) / B^k(M)`

measures the global obstruction to solving `d eta = omega` for a closed `k`-form `omega`.

For every smooth `k`-cycle `c`, integration defines a well-defined pairing

`I([omega])([c]) = int_c omega`.

The de Rham homomorphism

`I: H_dR^k(M) -> H^k(M; R)`

is an isomorphism. Thus closed forms modulo exact forms recover the real singular cohomology of `M`: they measure the holes of `M` through periods of differential forms.

If `M` is oriented of dimension `n` and admits the usual finite-good-cover hypotheses used for Poincare duality, integration also gives the complementary-degree pairing

`H_c^k(M) x H_dR^{n-k}(M) -> R`,

`([alpha],[beta]) -> int_M alpha wedge beta`,

and this identifies `H_dR^{n-k}(M)` with the dual of `H_c^k(M)`. If `M` is compact and oriented, this becomes the usual duality between `H_dR^k(M)` and `H_dR^{n-k}(M)`.

## Proof

For a form `alpha = sum_I alpha_I dx_I`, the exterior derivative is

`d alpha = sum_{I,j} (partial alpha_I / partial x_j) dx_j wedge dx_I`.

The graded product rule is

`d(alpha wedge beta) = d alpha wedge beta + (-1)^k alpha wedge d beta`

when `alpha` has degree `k`. Applying `d` twice gives zero. For functions, this is the cancellation of mixed partials:

`partial_i partial_j f dx_i wedge dx_j + partial_j partial_i f dx_j wedge dx_i = 0`.

The product rule extends the same cancellation to all forms. Hence `d^2 = 0`, so every exact form `d eta` is closed. This proves `B^k(M) subset Z^k(M)` and makes the quotient meaningful.

Stokes' theorem supplies the topological interpretation. If `c` is a smooth `k`-cycle and `omega = d eta` is exact, then

`int_c omega = int_c d eta = int_boundary c eta = 0`,

because `boundary c = 0`. Exact forms are therefore invisible to all closed cycles.

If `omega` is closed and `c'` is homologous to `c`, say `c' - c = boundary b`, then

`int_{c'} omega - int_c omega = int_boundary b omega = int_b d omega = 0`.

Thus a closed form has the same integral on homologous cycles. If `omega' = omega + d eta`, then

`int_c omega' - int_c omega = int_c d eta = int_boundary c eta = 0`.

So the integral depends only on the cohomology class `[omega]` and the homology class `[c]`. This proves that `I([omega])([c]) = int_c omega` is well-defined.

The quotient records holes because local closedness has no positive-degree content. On `R^n`, and on any chart diffeomorphic to `R^n`, the Poincare lemma says every closed positive-degree form is exact. One proof uses the homotopy operator on `R x X`: write a form as

`alpha(t,x) = dt wedge beta(t,x) + gamma(t,x)`,

define

`P alpha(t,x) = sum_J (int_0^t beta_J(s,x) ds) dx_J`,

and compute

`dP + Pd = 1 - pi^* s_0^*`,

where `pi: R x X -> X` is projection and `s_0: X -> R x X` is the zero section. This identity shows that cohomology is unchanged by multiplying by `R`; iterating reduces `R^n` to a point. Therefore positive-degree classes cannot be local phenomena.

The punctured plane shows the global obstruction. On `R^2 - {(0,0)}`, let

`omega = (x dy - y dx)/(x^2 + y^2)`.

This form is closed. On the unit circle `x = cos t`, `y = sin t`, its pullback is `dt`, so

`int_{S^1} omega = int_0^{2*pi} dt = 2*pi`.

If `omega` were exact, its integral over the closed circle would be zero by Stokes. Hence `omega` is closed but not exact. The nonzero period detects that the circle winds around a missing point.

It remains to identify the whole quotient with singular cohomology. First restrict singular chains to smooth singular chains; they compute the same homology. For a smooth simplex `sigma: Delta^k -> M`, define

`int_sigma omega = int_{Delta^k} sigma^* omega`,

and extend linearly to smooth chains. Stokes for chains gives

`int_c d eta = int_boundary c eta`.

The well-defined pairing above is therefore a natural homomorphism from de Rham cohomology to real singular cohomology.

On every convex open subset of `R^n`, the map is an isomorphism: the singular cohomology is that of a point, and the Poincare lemma gives the same result for forms. The statement is also compatible with disjoint unions. For a cover `M = U union V`, the de Rham complexes and the singular cochain complexes each have Mayer-Vietoris long exact sequences, and the integration homomorphism commutes with the maps in those sequences by naturality of pullback and Stokes' theorem. If the homomorphism is an isomorphism on `U`, `V`, and `U cap V`, the five lemma makes it an isomorphism on `U union V`. Applying this over good covers proves the de Rham theorem.

Consequently, `H_dR^k(M)` is not merely a space of differential equations modulo potentials. It is the real vector space of global periods of closed forms, with exact forms removed because they integrate to zero on cycles. Closed modulo exact forms measure holes.
