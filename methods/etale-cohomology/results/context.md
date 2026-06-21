## Research question

How can algebraic geometry recover the role played by singular cohomology when the ground field is finite? A complex algebraic variety has an analytic topology, so its holes can be measured by singular cohomology. A variety over `F_q` has no comparable classical topology: its rational points form a finite set, and the Zariski topology is coarse.

The question is to construct cohomology groups for varieties over `F_q` that are finite-dimensional vector spaces, behave like ordinary cohomology under pullback, products, duality, and compact support, and carry the action of Frobenius — so that point counting over `F_{q^n}` can be expressed through a linear operator.

## Background

The Weil conjectures predict structure in the zeta function of a smooth projective variety over a finite field. Formally, the zeta function packages all point counts:

`Z(X,t) = exp(sum_{n >= 1} |X(F_{q^n})| t^n / n)`.

For a cohomology theory adapted to this setting, the fixed points of the `n`th power of Frobenius should be exactly the `F_{q^n}`-points. The intended formula is a Lefschetz trace formula:

`|X(F_{q^n})| = sum_i (-1)^i Tr(Frob_q^n | H^i(X_{\bar F_q}, Q_l))`.

This asks for cohomology groups over an algebraic closure, together with a natural Frobenius action coming from descent to `F_q`. The coefficients are `l`-adic, with `l` different from the characteristic `p`, because finite `l^m`-torsion sheaves behave like locally constant finite covers while avoiding inseparability problems.

## Baselines

- **Singular cohomology after embedding in `C`.** This works for complex varieties and gives the model for Betti numbers, Poincare duality, and Lefschetz formulas.

- **Zariski sheaf cohomology.** This is intrinsic to schemes and used for coherent sheaves, computing derived functors of global sections on the Zariski topology.

- **Direct point-counting.** Counting solutions over each `F_{q^n}` gives the data the zeta function stores.

- **Classical Lefschetz intuition.** Fixed points of a self-map are counted by traces on cohomology.

- **Ad hoc algebraic invariants.** Divisors, differentials, and Chow groups capture geometry intrinsic to schemes.

## Evaluation settings

The primary setting is a separated scheme of finite type over a finite field, especially smooth and proper varieties. The core tests are finite-dimensionality of `H^i(X_{\bar F_q}, Q_l)`, vanishing outside the expected range, functoriality, cup products, Poincare duality for smooth proper varieties, compactly supported cohomology for nonproper varieties, and compatibility with base change.

The decisive arithmetic test is the Grothendieck-Lefschetz trace formula for Frobenius. If the theory is correct, point counts become alternating traces, and the zeta function factors as determinants of `1 - t Frob_q` on cohomology. The weight estimates for Frobenius eigenvalues are the Riemann-hypothesis part of the Weil conjectures.

There is also a comparison test: over fields that can be embedded into `C`, cohomology with `Q_l` coefficients should match ordinary singular cohomology after the expected change of coefficients. This keeps the theory tied to the topological model.

## Code framework

No executable code is needed. The field-appropriate scaffold is a construction and theorem pipeline:

1. Define a Grothendieck-style site for `X`, with a notion of generalized open and of covering, and sheaves on that site.
2. Define cohomology as derived functors of global sections of sheaves on the site.
3. Build torsion cohomology with `Z/l^m Z`-coefficients and pass carefully to `l`-adic cohomology, producing `Q_l`-vector spaces.
4. Establish the formal operations: pullback, proper pushforward, compact support, cup product, base change, and duality.
5. For `X/F_q`, let Frobenius act on `X_{\bar F_q}` and hence on `H^i(X_{\bar F_q}, Q_l)`.
6. Prove the Grothendieck-Lefschetz trace formula, turning Weil-style point counting into a statement about traces, determinants, and eigenvalue weights.
