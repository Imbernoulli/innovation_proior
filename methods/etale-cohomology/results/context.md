## Research question

How can algebraic geometry recover the role played by singular cohomology when the ground field is finite? A complex algebraic variety has an analytic topology, so its holes can be measured by singular cohomology. A variety over `F_q` has no comparable classical topology: its rational points form a finite set, and the Zariski topology is too coarse to see the expected Betti-style invariants.

The problem is not only to define groups called cohomology. The desired theory must produce finite-dimensional vector spaces, behave like ordinary cohomology under pullback, products, duality, and compact support, and expose the action of Frobenius. Once Frobenius acts on cohomology, point counting over `F_{q^n}` can be converted into traces of a linear operator.

## Background

The Weil conjectures predict deep structure in the zeta function of a smooth projective variety over a finite field. Formally, the zeta function packages all point counts:

`Z(X,t) = exp(sum_{n >= 1} |X(F_{q^n})| t^n / n)`.

For a good cohomology theory, the fixed points of the `n`th power of Frobenius should be exactly the `F_{q^n}`-points. The intended formula is a Lefschetz trace formula:

`|X(F_{q^n})| = sum_i (-1)^i Tr(Frob_q^n | H^i(X_{\bar F_q}, Q_l))`.

This asks for cohomology groups over an algebraic closure, together with a natural Frobenius action coming from descent to `F_q`. The coefficients are `l`-adic, with `l` different from the characteristic `p`, because finite etale `l^m`-torsion sheaves behave like locally constant finite covers while avoiding inseparability problems.

Grothendieck's move was to replace the missing topology by a site. Instead of ordinary open subsets, the etale site uses etale maps `U -> X` as generalized opens. Sheaves on this site can see finite covers and Galois-theoretic variation invisible to the Zariski topology. Cohomology is then sheaf cohomology on this new universe.

## Baselines

- **Singular cohomology after embedding in `C`.** This works for complex varieties and gives the model for Betti numbers, Poincare duality, and Lefschetz formulas. Gap: a variety over a finite field has no embedding into `C` compatible with its arithmetic Frobenius.

- **Zariski sheaf cohomology.** This is intrinsic to schemes and useful for coherent sheaves. Gap: Zariski opens are too large and too few; locally constant finite covers and Galois monodromy are not visible enough.

- **Direct point-counting.** Counting solutions over each `F_{q^n}` gives the data the zeta function stores. Gap: it does not explain rationality, functional equations, Betti-number degrees, or cancellation by eigenvalues.

- **Classical Lefschetz intuition.** Fixed points should be traces on cohomology. Gap: without a cohomology theory carrying Frobenius, the slogan has no algebraic object on which to act.

- **Ad hoc algebraic invariants.** Divisors, differentials, and Chow groups capture important geometry. Gap: they do not by themselves provide the full graded cohomological package needed for the Weil conjectures.

## Evaluation settings

The primary setting is a separated scheme of finite type over a finite field, especially smooth and proper varieties. The core tests are finite-dimensionality of `H^i_et(X_{\bar F_q}, Q_l)`, vanishing outside the expected range, functoriality, cup products, Poincare duality for smooth proper varieties, compactly supported cohomology for nonproper varieties, and compatibility with base change.

The decisive arithmetic test is the Grothendieck-Lefschetz trace formula for Frobenius. If the theory is correct, point counts become alternating traces, and the zeta function factors as determinants of `1 - t Frob_q` on cohomology. Deligne's later proof of the Riemann hypothesis part adds the weight estimates for Frobenius eigenvalues.

There is also a comparison test: over fields that can be embedded into `C`, etale cohomology with `Q_l` coefficients should match ordinary singular cohomology after the expected change of coefficients. This keeps the new theory tied to the older topological model.

## Code framework

No executable code is needed. The field-appropriate scaffold is a construction and theorem pipeline:

1. Define the etale site `X_et`, whose objects are etale maps `U -> X` and whose coverings are jointly surjective etale families.
2. Define sheaves on `X_et`, then define etale cohomology as derived functors of global sections.
3. Build torsion cohomology with `Z/l^m Z`-coefficients and pass carefully to `l`-adic cohomology, producing `Q_l`-vector spaces.
4. Establish the formal operations: pullback, proper pushforward, compact support, cup product, base change, and duality.
5. For `X/F_q`, let Frobenius act on `X_{\bar F_q}` and hence on `H^i_et(X_{\bar F_q}, Q_l)`.
6. Prove the Grothendieck-Lefschetz trace formula, turning Weil-style point counting into a statement about traces, determinants, and eventually eigenvalue weights.
