# Etale cohomology

## Core claim

Etale cohomology constructs, for algebraic varieties where ordinary topology is unavailable or inadequate, a cohomology theory that behaves like singular cohomology and can carry arithmetic operators. For a variety `X` over a finite field `F_q`, its central achievement is to produce finite-dimensional `Q_l`-vector spaces

`H^i_et(X_{\bar F_q}, Q_l)`, with `l != char(F_q)`,

on which Frobenius acts. This makes point-counting accessible through a Lefschetz trace formula:

`|X(F_{q^n})| = sum_i (-1)^i Tr(Frob_q^n | H^i_et(X_{\bar F_q}, Q_l))`.

The distinctive insight is that the replacement for singular cohomology is not found by forcing a finite-field variety to have a classical topology. It is found by changing what "local" means: the etale site treats etale maps as generalized open sets, and l-adic sheaves on that site record finite-cover and monodromy information that the Zariski topology misses.

## Construction

Let `X` be a scheme. Its etale site `X_et` has as objects etale morphisms `U -> X`; a covering is a jointly surjective family of etale maps. A sheaf on `X_et` assigns compatible data to these generalized opens. Etale cohomology is sheaf cohomology on this site:

`H^i_et(X, F) = R^i Gamma(X_et, F)`.

For arithmetic applications one usually starts with torsion sheaves `Z/l^m Z`, where `l` is invertible on `X`, then passes through the inverse system in `m` to obtain `Z_l`- and `Q_l`-cohomology. This produces cohomology groups with the expected formal features: pullback, cup product, compact support, proper base change, Kunneth formulas, and Poincare duality in the smooth proper case.

If `X` is defined over `F_q`, then `X_{\bar F_q}` has a Frobenius automorphism. Functoriality gives a linear action of Frobenius on each `H^i_et(X_{\bar F_q}, Q_l)`. The fixed points of `Frob_q^n` are exactly the `F_{q^n}`-points, so the Grothendieck-Lefschetz trace formula turns point counting into cohomological linear algebra.

## Why Grothendieck's strategy worked

Grothendieck's method was to build the right universe before attacking the numerical problem. The Weil conjectures were about point counts and zeta functions, but their predicted shape was cohomological: Betti numbers, duality, fixed-point traces, and eigenvalue bounds. Without a category in which finite-field varieties had cohomology resembling topology, the conjectures had no natural mechanism.

The etale site supplied that mechanism. It made algebraic local systems and finite covers visible. L-adic coefficients supplied stable linear spaces. The functorial formalism supplied Frobenius actions, duality, and trace maps. After this architecture was in place, the rationality and functional-equation parts of the Weil conjectures could be recast as consequences of a Lefschetz trace formula and Poincare duality. Deligne's proof of the Riemann-hypothesis part then became a theorem about weights of Frobenius eigenvalues.

The philosophical lesson is precise: Grothendieck did not merely solve a problem inside an existing language. He designed a language in which the problem changed type. Point counts over finite fields became traces of Frobenius on l-adic cohomology.

## Compact theorem statement

For a separated scheme `X` of finite type over `F_q` and a prime `l != char(F_q)`, l-adic etale cohomology assigns finite-dimensional `Q_l`-vector spaces `H^i_c(X_{\bar F_q}, Q_l)` with functorial Frobenius action. Under the usual hypotheses needed for the trace formula,

`|X(F_q)| = sum_i (-1)^i Tr(Frob_q | H^i_c(X_{\bar F_q}, Q_l))`.

For smooth proper `X`, compact support may be omitted, and the zeta function factors as

`Z(X,t) = product_i det(1 - t Frob_q | H^i_et(X_{\bar F_q}, Q_l))^{(-1)^{i+1}}`.

Thus etale cohomology is the cohomological bridge between algebraic geometry over finite fields and the topological pattern predicted by the Weil conjectures.
